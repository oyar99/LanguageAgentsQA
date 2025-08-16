"""ReactRetriever agent for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import math
import os
from typing import Dict, List
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.action import Action
from models.agent import IntelligentAgent, NoteBook
from models.dataset import Dataset
import utils.agent_worker as worker


class ReactRetriever(IntelligentAgent):
    """
    ReactAgentRetriever for reasoning over indexed documents using a custom instruction fine-tuned model
    with structured output schema following the ReAct prompting framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._args = args
        actions = {
            "search": Action(
                "Search for relevant documents for the given query using a semantic retriever. \
Argument d is the rate of the depth of the search. Must be a positive integer starting iteratively from 1.",
                self._search_documents
            )
        }
        super().__init__(actions, PROMPT_EXAMPLES_TOOLS, args)
        self._enable_interleave_reflection = False

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
