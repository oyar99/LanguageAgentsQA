"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import math
import os
from typing import Any, Dict, List, Optional
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from evaluator.rogue_evaluator import rouge_score
from logger.logger import Logger
from models.action import Action
from models.agent import StatefulIntelligentAgent, NoteBook
from models.dataset import Dataset
from plugins.post_reflector import post_reflector
import utils.agent_worker as worker


class CognitiveAgent(StatefulIntelligentAgent):
    """
    ReactAgentCustom for reasoning over indexed documents using a custom instruction fine-tuned model
    with structured output schema following the ReAct prompting framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._prompt = None
        # Reverse index of questions to their full objects for post-reasoning evaluation
        self._questions_map = None
        self._args = args
        actions = {
            "search": Action(
                "Search for relevant documents for the given query using a semantic retriever.\
Argument d is the rate of the depth of the search. Must be a positive integer starting from 1.",
                self._search_documents
            )
        }
        prompt_examples = PROMPT_EXAMPLE_TOOLS_LOCOMO if args.dataset == 'locomo' else PROMPT_EXAMPLES_TOOLS

        # 4 agents that learn independently in parallel
        super().__init__(actions, prompt_examples, args, cores=2)
        self._base_prompt = self._prompt
        self._enable_reflection = False

        self._episodic_memory: List[Dict[str, Any]] = []

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval using ColBERT.
        """
        Logger().info("Indexing documents using ColbertV2")
        corpus = dataset.read_corpus()

        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        os.makedirs(colbert_dir, exist_ok=True)

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(
                nbits=2,
            )
            self._index = Indexer('colbert-ir/colbertv2.0', config=config)
            self._index.index(
                name=dataset.name or 'index',
                collection=[doc['content'] for doc in corpus],
                overwrite='reuse'  # type: ignore
            )

        self._index = dataset.name or 'index'
        self._corpus = corpus
        self._questions_map = {
            question['question']: question
            for question in ([q for questions in dataset.get_questions().values()
                              for q in questions])
        }
        Logger().info("Successfully indexed documents")

    def _search_documents(
            self,
            query: str,
            d: int = 1,
    ) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[List[str], List[str], Dict[str, int]]: Tuple containing list of observations (retrieved documents)
, list of sources if any, and metrics if any.
        """
        d = max(1, int(d))  # Ensure d is at least 1
        # determine how many documents to retrieve using a logistic curve since too many documents
        #  can lead to poor performance and increase token usage.
        k = int(round(30/(1 + 5 * math.exp(-0.62 * (d - 1)))))

        doc_ids, _ranking, scores = worker.searcher.search(
            query, k=k)

        documents = []
        for doc_id, _, score in zip(doc_ids, _ranking, scores):
            documents.append({
                'doc_id': self._corpus[doc_id]['doc_id'],
                'content': self._corpus[doc_id]['content'],
                'score': score,
                'original_id': doc_id
            })

        Logger().debug(
            f"Search results for query '{query}': Found {len(documents)} documents")

        return ([doc['content'] for doc in documents],
                doc_ids,
                {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0})

    def _init_searcher(self) -> None:
        """
        Initializes the searcher for the ReactAgentCustom.
        This is used to set up the searcher with the indexed documents.
        """
        if self._index is None or self._corpus is None:
            raise RuntimeError(
                "Index and corpus must be initialized before creating a searcher.")

        if worker.searcher is None:
            colbert_dir = os.path.join(os.path.normpath(
                os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

            Logger().debug("Initializing searcher")

            with worker.lock:
                with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
                    worker.searcher = Searcher(index=self._index, collection=[
                        doc['content'] for doc in self._corpus], verbose=1)

    def reason(self, question: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question using ReAct framework
        with a custom instruction fine-tuned model.
        """

        # Prepare any data that actions may need when they are executed by the ReAct engine
        # Ideally everything is ready once index is called. Unfortunately, some libraries like ColBERT use
        # objects that can't be pickled, and hence need to be instantiated in each process.
        self._init_searcher()

        # Pre-process the question before reasoning by extracting representative error samples
        # from episodic memory and updating the prompt with additional examples
        self._pre_reasoning(question)

        # Get the notebook from the parent reasoning process
        notebook = super().reason(question)

        # Extract information from the notebook for post-processing
        final_answer = notebook.get_notes()
        messages = notebook.get_messages()
        sources = [source['doc_id'] for source in notebook.get_sources()]

        # Perform post-processing analysis and update token usage
        post_reasoning_usage = self._post_reasoning(
            question, final_answer, messages, sources)

        # Update notebook with additional usage metrics from post-reasoning
        if post_reasoning_usage:
            current_usage = notebook.get_usage_metrics()
            updated_usage = {
                "completion_tokens": current_usage.get("completion_tokens", 0) + post_reasoning_usage.get("completion_tokens", 0),
                "prompt_tokens": current_usage.get("prompt_tokens", 0) + post_reasoning_usage.get("prompt_tokens", 0),
                "total_tokens": current_usage.get("total_tokens", 0) + post_reasoning_usage.get("total_tokens", 0)
            }
            notebook.update_usage_metrics(updated_usage)

        return notebook

    def _pre_reasoning(self, question: str) -> List[Dict[str, str]]:
        """
        Pre-process the question before reasoning by extracting representative error samples
        from episodic memory and updating the prompt with additional examples.
        """
        Logger().debug(f"Pre-reasoning for question: {question}")
        Logger().debug(f"Episodic memory size: {len(self._episodic_memory)}")

        if not self._episodic_memory:
            return

        # Group errors by category
        error_categories = {}
        for memory_entry in self._episodic_memory:
            entry_data = memory_entry.get("id", {})
            category = entry_data.get("category", "Unknown")

            if category not in error_categories:
                error_categories[category] = []
            error_categories[category].append(entry_data)

        Logger().debug(
            f"Episodic memory contains {len(self._episodic_memory)} entries across {len(error_categories)} categories")

        # Extract representative samples using heuristics
        representative_samples = self._extract_representative_samples(
            error_categories)

        Logger().debug(
            f"Extracted {len(representative_samples)} representative samples for prompt update")

        # Update prompt with episodic memory examples
        if representative_samples:
            episodic_examples = self._format_episodic_examples(
                representative_samples)

            # Add episodic memory section to the existing prompt
            episodic_section = f"\n\n## MY LEARNING EXPERIENCE\n\n{episodic_examples}"
            self._prompt = self._base_prompt + episodic_section

            Logger().debug(
                f"Updated prompt with {len(representative_samples)} episodic memory examples")

    def _extract_representative_samples(self,
                                        error_categories: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract at most 30 representative samples using distribution heuristics.
        Prioritizes the most recent entries for each category to ensure fresh learning.

        Args:
            error_categories: Dictionary mapping error categories to lists of error instances

        Returns:
            List of representative error samples
        """
        max_samples = 30
        total_errors = sum(len(errors) for errors in error_categories.values())

        if total_errors == 0:
            return []

        representative_samples = []

        # Calculate sample distribution based on category frequencies
        category_counts = {cat: len(errors)
                           for cat, errors in error_categories.items()}
        sorted_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True)

        # Heuristic 1: If one error type dominates (>50%), allocate more samples to it
        dominant_category = sorted_categories[0]
        if dominant_category[1] / total_errors > 0.5:
            # Allocate 60% samples to dominant category, remaining to others
            samples_for_dominant = min(
                int(max_samples * 0.6), len(error_categories[dominant_category[0]]))
            remaining_samples = max_samples - samples_for_dominant

            # Add most recent samples from dominant category
            representative_samples.extend(
                error_categories[dominant_category[0]][-samples_for_dominant:]
            )

            # Distribute remaining samples among other categories
            other_categories = sorted_categories[1:]
            if other_categories and remaining_samples > 0:
                samples_per_other = max(
                    1, remaining_samples // len(other_categories))
                for cat_name, _ in other_categories:
                    if remaining_samples <= 0:
                        break
                    samples_to_take = min(samples_per_other, len(
                        error_categories[cat_name]), remaining_samples)
                    # Take most recent samples from this category
                    representative_samples.extend(
                        error_categories[cat_name][-samples_to_take:])
                    remaining_samples -= samples_to_take
        else:
            # Heuristic 2: Balanced distribution - distribute samples evenly across categories
            samples_per_category = max(1, max_samples // len(error_categories))
            remaining_samples = max_samples

            for cat_name, errors in error_categories.items():
                if remaining_samples <= 0:
                    break
                samples_to_take = min(
                    samples_per_category, len(errors), remaining_samples)
                # Take most recent samples from this category
                representative_samples.extend(errors[-samples_to_take:])
                remaining_samples -= samples_to_take

        return representative_samples[:max_samples]

    def _format_episodic_examples(self, samples: List[Dict[str, Any]]) -> str:
        """
        Format episodic memory samples as monologue examples for the prompt.

        Args:
            samples: List of representative error samples

        Returns:
            Formatted string with episodic examples
        """
        formatted_examples = []

        for i, sample in enumerate(samples, 1):
            question = sample.get("question")
            ground_truth = sample.get("ground_truth")
            final_answer = sample.get("final_answer")
            correct_reasoning_chain = sample.get("correct_reasoning_chain")
            category = sample.get("category")
            # messages = sample.get("messages")

            example = f"""### Episodic Memory Example {i} - Error Type: {category}

Question: "{question}"
My Previous Answer: "{final_answer}"
Expected Answer: "{ground_truth}"

**Corrected Reasoning Chain:**

{correct_reasoning_chain}
"""

            formatted_examples.append(example)

        header = "Based on my past reasoning experiences, here are some errors I've made and lessons I've learned:\n\n"
        return header + "\n\n".join(formatted_examples)

    def _post_reasoning(self,
                        question: str,
                        final_answer: str,
                        messages: List[Dict[str, str]],
                        sources: List[str]) -> Optional[Dict[str, int]]:
        """
        Post-process the final answer after reasoning.

        Returns:
            Optional[Dict[str, int]]: Usage metrics from post-reasoning analysis, if any.
        """
        # Evaluate the final answer against the ground truth and if incorrect,
        # determine the reasoning step that led to the error.
        question_obj = self._questions_map.get(question)

        r1, _, _ = rouge_score(question_obj['answer'], final_answer)[0]

        if r1 < 0.5:  # Threshold for likely incorrect answer
            Logger().info(
                f"Final answer for question '{question}' is likely incorrect with \
ROUGE-1 score {r1:.2f}")

            actual_sources = [doc['doc_id']
                              for doc in question_obj.get('docs', [])]

            # Check if all actual sources are contained in the retrieved sources
            sources_set = set(sources)
            actual_sources_set = set(actual_sources)

            post_reflection_result = None
            if actual_sources_set.issubset(sources_set):
                # All actual sources were retrieved, use post reflector to analyze reasoning error
                post_reflection_result = post_reflector(
                    question,
                    question_obj['answer'][0],
                    final_answer,
                    messages,
                    [doc['content'] for doc in question_obj['docs']],
                    missing_evidence=False)
            else:
                # Missing evidence - not all actual sources were retrieved
                post_reflection_result = post_reflector(
                    question,
                    question_obj['answer'][0],
                    final_answer,
                    messages,
                    [doc['content'] for doc in question_obj['docs']],
                    missing_evidence=True)

            if post_reflection_result:
                correct_reasoning_chain, category, usage_metrics = post_reflection_result

                if correct_reasoning_chain and category:
                    # Write to episodic memory instances of incorrect reasoning for future learning
                    self._episodic_memory.append({
                        "id": {
                            "question": question,
                            "ground_truth": question_obj['answer'][0],
                            "final_answer": final_answer,
                            "messages": messages,
                            "correct_reasoning_chain": correct_reasoning_chain,
                            "category": category
                        }
                    })

                Logger().debug(
                    f"Updated episodic memory with new entry. Total entries: {len(self._episodic_memory)}")

                return usage_metrics

        return None


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

PROMPT_EXAMPLES_TOOLS = '''### Example 1

Question: "Were Scott Derrickson and Ed Wood of the same nationality?"

Iteration 1:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search('Scott Derrickson's nationality', 1)", "search('Ed Wood's nationality', 1)"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search('Scott Derrickson's nationality', 1)", "search('Ed Wood's nationality', 1)"],
    "observations": [["Scott Derrickson, known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange', has been recognized by the president of the United States."], ["Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic 'Plan 9 from Outer Space'."]]
}
```

Iteration 3:
```json
{
    "thought": "I found that Ed Wood was an American filmmaker, but I need to confirm Scott Derrickson nationality to determine if they are from the same country. \
I will search for more information on Scott Derrickson",
    "actions": ["search('Scott Derrickson's nationality', 2)"],
}
```

Iteration 4:
```json
{
    "thought": "I found that Ed Wood was an American filmmaker, but I need to confirm Scott Derrickson nationality to determine if they are from the same country. \
I will search for more information on Scott Derrickson",
    "actions": ["search('Scott Derrickson's nationality', 2)"],
    "observations": [["Scott Derrickson, known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange', has been recognized by the president of the United States.", "Scott Derrickson is an American film director, producer, and screenwriter"]]
}
```

Iteration 5:
```json
{
    "thought": "Both Scott Derrickson and Ed Wood are American based on the retrieved information, so they are of the same nationality.",
    "final_answer": "Yes"
}
```

### Example 2

Question: "In which county is Kimbrough Memorial Stadium located?"

Iteration 1:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium location', 1)"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium location', 1)"],
    "observations": [["Kimbrough Memorial Stadium has a seating capacity of 8,500."]]
}
```

Iteration 3:
```json
{
    "thought": "I did not find information about Kimbrough Memorial Stadium location, therefore I will search for more documents about it.",
    "actions": ["search('Kimbrough Memorial Stadium location', 2)"],
}
```

Iteration 4:
```json
{
    "thought": "I did not find information about Kimbrough Memorial Stadium location, therefore I will search for more documents about it.",
    "actions": ["search('Kimbrough Memorial Stadium location', 2)"],
    "observations": [["Kimbrough Memorial Stadium has a seating capacity of 8,500.", "Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football."]]
}
```

Iteration 5:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county', 1)"]
}
```

Iteration 6:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county', 1)"],
    "observations": [["Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census."]]
}
```

Iteration 7:
```json
{
    "thought": "Kimbrough Memorial Stadium is in Canyon, Texas, and Canyon is in Randall County.",
    "final_answer": "Randall County"
}
```
'''

PROMPT_EXAMPLE_TOOLS_LOCOMO = '''### Example 1

Question: "When did Jolene`s mother pass away?"

Iteration 1:
```json
{
    "thought": "I need to find information about Jolene's mother.",
    "actions": ["search('Jolenes's mother')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find information about Jolene's mother.",
    "actions": ["search('Jolenes's mother')"],
    "observations": [[""At around 4:06 pm on 23 January, 2023, during message 6, Jolene said: Sorry about your loss. My mother also passed away \
last year. This is my room in her house, I also have many memories there. Is there anything special about it you remember? - Attached image: a \
photo of a room with a bench and a window"]]
}
```

Iteration 3:
```json
{
    "thought": "Jolene mentioned on 23 January, 2023 that her mother passed away last year. She didn't mention a specific date. Therefore, we can
only infer she died in the year 2022.",
    "final_answer": "2022"
}
```

### Example 2

Question: "How many weeks passed between Maria adopting Coco and Shadow?"

Iteration 1:
```json
{
    "thought": "I need to find out when Maria adopted Coco and when she adopted Shadow.",
    "actions": ["search('Maria adopted Coco')", "search('Maria adopted Shadow')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find out when Maria adopted Coco and when she adopted Shadow.",
    "actions": ["search('Maria adopted Coco')", "search('Maria adopted Shadow')"],
    "observations": [["At around 12:10 am on 11 August, 2023, during message 1, Maria said: Hey John! Long time no talk! \
Guess what - I got a puppy two weeks ago! Her name's Coco and she's adorable. - Attached image: a photo of a white dog laying in the grass with a baseball"], ["At \
around 3:14 pm on 13 August, 2023, during message 4, Maria said: Her name is Shadow! She's full of energy and always puts a smile on my face. She's \
been a great addition to my life. They get along great."]]
}
```

Iteration 3:
```json
{
    "thought": "Maria mentioned she got Coco two weeks ago from 11 August, 2023. She did not specify when she adopted Shadow, \
so I need to find out if she shared more information about adopting a dog.",
    "actions": ["search('Maria adopted a dog')"]
}
```

Iteration 4:
```json
{
    "thought": "Maria mentioned she got Coco two weeks ago from 11 August, 2023. She did not specify when she adopted Shadow, \
so I need to find out if she shared more information about adopting a dog.",
    "actions": ["search('Maria adopted a dog')"],
    "observations": [["At around 12:10 am on 11 August, 2023, during message 1, Maria said: Hey John! Long time no talk! \
Guess what - I got a puppy two weeks ago! Her name's Coco and she's adorable. - Attached image: a photo of a white dog laying in the grass with a baseball", "At around \
3:14 pm on 13 August, 2023, during message 2, Maria said: Wow, John! You're doing great things. Kudos for helping kids learn. On another note, I just adopted \
this cute pup from a shelter last week. She brings so much joy! I feel blessed to be able to give her a home."]]
}
```

Iteration 5:
```json
{
    "thought": "Maria mentioned she adopted a dog from a shelter last week from 13 August, 2023. She got Coco two weeks ago from 11 August, 2023. \
This means there is more than one week between adopting Coco and Shadow.",
    "final_answer": "More than one week"
}
```
'''
