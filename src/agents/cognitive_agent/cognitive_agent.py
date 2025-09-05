"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluator.rogue_evaluator import rouge_score
from logger.logger import Logger
from models.action import Action
from models.agent import StatefulIntelligentAgent, NoteBook
from models.dataset import Dataset
from models.structural_search import StructuralSearchEngine
from plugins.post_reflector import post_reflector
import utils.agent_worker as worker
from utils.dataset_utils import get_complete_evidence
from utils.structure_response import parse_structured_response

# pylint: disable-next=too-many-instance-attributes


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
                "Search for relevant documents for the given query using a semantic retriever. \
You will obtain more relevant results by formulating queries scoped to specific entities or \
keywords related to the question.",
                self._search_documents
            )
        }
        prompt_examples = PROMPT_EXAMPLE_TOOLS_LOCOMO if args.dataset == 'locomo' else PROMPT_EXAMPLES_TOOLS

        # cores determines the number of agents to run in parallel
        super().__init__(actions, prompt_examples, args, cores=1)
        self._base_prompt = self._prompt

        self._episodic_memory: List[Dict[str, Any]] = []

        # Initialize structural search engine
        self._structural_search = StructuralSearchEngine()
        Logger().info("Initialized Structural Search Engine")

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
    ) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[List[str], List[str], Dict[str, int]]: Tuple containing list of observations (retrieved documents)
, list of sources if any, and metrics if any.
        """
        k = 5

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
        Initializes the searcher.
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

        # Pre-process the question before reasoning by extracting representative samples
        # from episodic memory and updating the prompt with additional examples
        pre_reasoning_usage = self._pre_reasoning(question)

        # Run the ReAct reasoning process
        notebook = super().reason(question)

        # Extract information from the notebook for post-processing
        final_answer = notebook.get_notes()
        messages = notebook.get_messages()

        # Perform post-processing analysis
        self._post_reasoning(
            question, final_answer, messages)

        # Update notebook with additional usage metrics from pre/post reasoning tools
        if pre_reasoning_usage:
            current_usage = notebook.get_usage_metrics()
            updated_usage = {
                "completion_tokens": (
                    current_usage.get("completion_tokens", 0) + pre_reasoning_usage.get("completion_tokens", 0)),
                "prompt_tokens": current_usage.get("prompt_tokens", 0) + pre_reasoning_usage.get("prompt_tokens", 0),
                "total_tokens": current_usage.get("total_tokens", 0) + pre_reasoning_usage.get("total_tokens", 0)
            }
            notebook.update_usage_metrics(updated_usage)

        return notebook

    # pylint: disable-next=too-many-locals, too-many-branches
    def _pre_reasoning(self, question: str) -> Dict[str, int] | None:
        """
        Pre-process the question before reasoning by finding structurally similar questions
        from episodic memory and updating the prompt with relevant examples.
        """
        Logger().debug(f"Pre-reasoning for question: {question}")
        Logger().debug(f"Episodic memory size: {len(self._episodic_memory)}")

        if not self._episodic_memory:
            Logger().debug("No episodic memory entries available")
            return None

        # Search for structurally similar questions
        similar_questions = self._structural_search.search(question, top_k=100)

        if not similar_questions:
            Logger().debug("No structurally similar questions found")
            return None

        # Filter examples based on similarity threshold (0.4) and select appropriate mix
        valid_examples = [
            (q_data, skeleton, score) for q_data, skeleton, score in similar_questions
            if score > 0.4
        ]

        if not valid_examples:
            Logger().debug("No examples meet similarity threshold of 0.4")
            return None

        # Separate into correct and incorrect examples
        correct_examples = [
            ex for ex in valid_examples if ex[0].get('is_correct')]
        incorrect_examples = [
            ex for ex in valid_examples if not ex[0].get('is_correct')]

        usage_metrics = None

        # For the incorrect examples, we implement lazy loading of reasoning chains
        # Only compute reasoning chains for the examples we'll actually use (up to 7)
        processed_incorrect_examples = []
        i = 0

        # pylint: disable-next=too-many-nested-blocks
        while len(processed_incorrect_examples) < 7 and i < len(incorrect_examples):
            q_data, skeleton, score = incorrect_examples[i]

            if 'correct_reasoning_chain' in q_data and q_data['correct_reasoning_chain']:
                Logger().debug(
                    f"Using existing reasoning chain for question: {q_data['question']}")
                processed_incorrect_examples.append((q_data, skeleton, score))
            else:
                question_obj = self._questions_map.get(q_data['question'])

                # get ground truth supporting documents
                evidence = get_complete_evidence(
                    question_obj, self._corpus, self._args.dataset)

                post_reflection_result = post_reflector(
                    self._args,
                    q_data['question'],
                    q_data['ground_truth'],
                    [doc['content'] for doc in evidence],
                    question_obj.get('decomposition', [])
                )

                if post_reflection_result:
                    reasoning_chain, usage_metrics = post_reflection_result

                    if reasoning_chain:
                        updated_q_data = {
                            **q_data, 'correct_reasoning_chain': reasoning_chain}
                        processed_incorrect_examples.append(
                            (updated_q_data, skeleton, score))

                        # Update the episodic memory entry as well
                        for mem_idx, mem_entry in enumerate(self._episodic_memory):
                            if mem_entry['question'] == q_data['question']:
                                self._episodic_memory[mem_idx]['correct_reasoning_chain'] = reasoning_chain
                                break

            i += 1

        selected_examples = []

        # Add the processed incorrect examples (up to 7)
        selected_examples.extend(processed_incorrect_examples)

        # Add 1 correct example if available
        if correct_examples:
            selected_examples.extend(correct_examples[:1])

        if selected_examples:
            Logger().debug(
                f"Selected {len(selected_examples)} structurally similar examples for prompt")

            # Reverse selected_examples so more similar structural questions appear at the bottom
            # so the model can pay more attention to it
            selected_examples.reverse()

            # Format and add examples to prompt
            episodic_examples = self._format_structural_examples(
                selected_examples)
            episodic_section = episodic_examples
            self._prompt = self._base_prompt + episodic_section

            Logger().debug("Updated prompt with structural examples")
        else:
            Logger().debug("No valid structural examples to include")

        return usage_metrics

    def _update_structural_index(self, memory_entry: Dict[str, Any]) -> None:
        """
        Update the structural search index with a new question entry.

        Args:
            question: The new question to add to the index.
        """
        if not self._episodic_memory:
            return

        question = memory_entry.get("question")

        # Convert question to structural skeleton
        skeleton = self._structural_search.question_to_skeleton(question)

        # Add to index
        self._structural_search.questions.append(
            memory_entry)
        self._structural_search.skeletons.append(skeleton)
        self._structural_search.skeleton_to_questions[skeleton].append(
            len(self._structural_search.questions) - 1)

        self._structural_search.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            token_pattern=r'\b\w+\b|<[^>]+>',
            lowercase=False
        )
        self._structural_search.skeleton_vectors = self._structural_search.vectorizer.fit_transform(
            self._structural_search.skeletons
        )

        Logger().debug(
            f"Rebuilt structural index with {len(self._structural_search.questions)} questions")

    def _format_structural_examples(self, examples: List[Tuple[Dict[str, Any], str, float]]) -> str:
        """
        Format structurally similar examples for the prompt.

        Args:
            examples: List of (question_data, skeleton, similarity_score) tuples

        Returns:
            Formatted string with structural examples
        """
        formatted_examples = []

        for i, (question_data, _, _) in enumerate(examples, 3):
            question = question_data.get("question")
            correct_reasoning_chain = question_data.get(
                "correct_reasoning_chain", "")

            example = f"""### Example {i}

Question: "{question}"

{correct_reasoning_chain}
"""

            formatted_examples.append(example)

        return "\n\n".join(formatted_examples)

    def _extract_reasoning_chain_from_messages(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract reasoning chain from messages for correct answers.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted reasoning chain string or None if extraction fails
        """
        system_messages = []
        user_question_found = False

        for message in messages:
            if message.get("role") == "user":
                user_question_found = True
                continue

            if user_question_found and message.get("role") == "system":
                content = message.get("content").strip()
                if content:
                    system_messages.append(content)

        if not system_messages:
            return None

        # Format messages as iterations, splitting intermediate iterations into two parts
        formatted_iterations = []
        iteration_num = 1

        for content in system_messages:
            structured_response = parse_structured_response(content)

            if structured_response is None:
                return None

            # Use the same fallback logic as the base agent
            thought = structured_response.get("thought", "")
            actions = structured_response.get("actions", [])
            final_answer = structured_response.get("final_answer", None)
            observations = structured_response.get("observations", [])

            # If this is a final answer, format it and we're done
            if final_answer is not None:
                iteration = (
                    f'**Iteration {iteration_num}:**'
                    f'\n```json\n{{\n    "thought": "{thought}",\n    "final_answer": "{final_answer}"\n}}\n```'
                )
                formatted_iterations.append(iteration)
                break

            if actions is None or len(actions) <= 0:
                Logger().warn(
                    f"Skipping iteration {iteration_num} - no actions found")
                return None

            # For intermediate responses, create two separate iterations:
            # 1. First iteration: thought + actions (the plan)
            iteration = (
                f'**Iteration {iteration_num}:**'
                f'\n```json\n{{\n    "thought": "{thought}",\n    "actions": {json.dumps(actions)}\n}}\n```'
            )
            formatted_iterations.append(iteration)
            iteration_num += 1

            # 2. Second iteration: thought + actions + observations (the execution result)
            iteration = (
                f'**Iteration {iteration_num}:**\n```json'
                f'\n{{\n    "thought": "{thought}",\n    "actions": {json.dumps(actions)},'
                f'\n    "observations": {json.dumps(observations)}\n}}\n```'
            )
            formatted_iterations.append(iteration)
            iteration_num += 1

        if not formatted_iterations:
            return None

        return "\n\n".join(formatted_iterations)

    def _post_reasoning(self,
                        question: str,
                        final_answer: str,
                        messages: List[Dict[str, str]]) -> None:
        """
        Post-process the final answer after reasoning and update episodic memory.

        Returns:
            Optional[Dict[str, int]]: Usage metrics from post-reasoning analysis, if any.
        """
        # Evaluate the final answer against the ground truth
        question_obj = self._questions_map.get(question)
        if not question_obj:
            Logger().warning(f"Question object not found for: {question}")
            return

        r1, _, _ = rouge_score(question_obj['answer'], final_answer)[0]
        is_correct = r1 >= 0.65  # Threshold for correctness

        Logger().info(
            f"Question '{question}' - ROUGE-1: {r1:.3f} - {'CORRECT' if is_correct else 'INCORRECT'}")

        correct_reasoning_chain = None

        if is_correct:
            correct_reasoning_chain = self._extract_reasoning_chain_from_messages(
                messages)

        # Delay computing correct_reasoning_chain until needed for incorrect examples
        episodic_entry = {
            "question": question,
            "ground_truth": question_obj['answer'][0],
            "final_answer": final_answer,
            "messages": messages,
            "correct_reasoning_chain": correct_reasoning_chain,
            "is_correct": is_correct,
            "rouge_score": r1
        }

        self._episodic_memory.append(episodic_entry)
        self._update_structural_index(episodic_entry)

        Logger().debug(
            f"Updated episodic memory with new {'correct' if is_correct else 'incorrect'} entry. "
            f"Total entries: {len(self._episodic_memory)}")


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
    "actions": ["search('Scott Derrickson's nationality')", "search('Ed Wood's nationality')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search('Scott Derrickson's nationality')", "search('Ed Wood's nationality')"],
    "observations": [["Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange'."], ["Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic 'Plan 9 from Outer Space'."]]
}
```

Iteration 3:
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
    "actions": ["search('Kimbrough Memorial Stadium location')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium location')"],
    "observations": [["Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football."]]
}
```

Iteration 3:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county')"]
}
```

Iteration 4:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search('Canyon Texas county')"],
    "observations": [["Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census."]]
}
```

Iteration 5:
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
