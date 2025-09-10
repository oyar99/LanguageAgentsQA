"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.action import Action
from models.agent import IntelligentAgent, NoteBook
from models.dataset import Dataset
from plugins.search_pruner import search_pruner
import utils.agent_worker as worker


class ReactAgentCustom(IntelligentAgent):
    """
    ReactAgentCustom for reasoning over indexed documents using a custom instruction fine-tuned model
    with structured output schema following the ReAct prompting framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._args = args
        actions = {
            "search": Action(
                "Search for relevant documents for the given query using a semantic retriever. \
You will obtain more relevant results by formulating queries scoped to specific entities or \
keywords related to the question.",
                self._search_documents
            )
        }
        self._enable_pruning = False
        prompt_examples = PROMPT_EXAMPLE_TOOLS_LOCOMO if args.dataset == 'locomo' else PROMPT_EXAMPLES_TOOLS
        super().__init__(actions, prompt_examples, args)
        self._enable_reflection = False

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
        Logger().info("Successfully indexed documents")

    def _search_documents(
            self,
            query: str,
            context: str = "",
    ) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[List[str], List[str], Dict[str, int]]: Tuple containing list of observations (retrieved documents)
, list of sources if any, and metrics if any.
        """
        doc_ids, _ranking, scores = worker.searcher.search(
            query, k=self._args.k or 5)

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

        if not self._enable_pruning:
            return ([doc['content'] for doc in documents],
                    doc_ids,
                    {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0})

        pruned_documents, usage_metrics = search_pruner(
            query, documents, context)

        Logger().debug(
            f"Search results for query '{query}': Found {len(pruned_documents)} documents after pruning")

        if len(pruned_documents) == 0:
            Logger().warn(f"No relevant documents found for query: {query}")
            pruned_documents = documents[:1]

        return ([doc['content'] for doc in pruned_documents],
                [doc['original_id'] for doc in pruned_documents],
                usage_metrics)

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

        return super().reason(question)


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
