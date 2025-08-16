"""LexicalSemanticAgent for intelligent search using both lexical and semantic approaches.

This agent automatically decides between BM25 lexical search and ColBERT semantic search
based on the nature of the query, following ReAct framework with manual tool invocation.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List, Any

from rank_bm25 import BM25Okapi as BM25Ranker
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.action import Action
from models.agent import IntelligentAgent, NoteBook
from models.dataset import Dataset
from utils.tokenizer import PreprocessingMethod, tokenize
import utils.agent_worker as worker


class LexicalSemanticAgent(IntelligentAgent):
    """
    LexicalSemanticAgent that intelligently chooses between lexical and semantic search
    based on query characteristics using ReAct framework.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._bm25_index = None
        self._colbert_searcher = None
        self._args = args
        actions = {
            "search_lexical": Action(
                "Performs exact keyword matching using BM25 algorithm. \
Best for one-word queries or specific terms when precise term matching is needed",
                self._search_lexical
            ),
            "search_semantic": Action(
                "Performs meaning-based search using ColBERT. \
Best for conceptual queries, or natural language questions that require understanding of context.",
                self._search_semantic,
            ),
        }
        super().__init__(actions, PROMPT_EXAMPLES_TOOLS, args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for both lexical (BM25) and semantic (ColBERT) retrieval.
        """
        Logger().info("Indexing documents using LexicalSemanticAgent")
        corpus = dataset.read_corpus()

        # Index for BM25 lexical search
        Logger().info("Creating BM25 index for lexical search")
        self._index_bm25(corpus)

        # Index for ColBERT semantic search
        Logger().info("Creating ColBERT index for semantic search")
        self._index_colbert(dataset, corpus)

        self._corpus = corpus
        Logger().info("Successfully indexed documents for both lexical and semantic search")

    def _index_bm25(self, corpus: List[Dict[str, Any]]) -> None:
        """Create BM25 index for lexical search."""
        self._bm25_index = BM25Ranker(
            corpus,
            tokenizer=self._tokenize_doc,
            b=0.75,
            k1=0.5
        )

    def _tokenize_doc(self, doc: Dict[str, Any]) -> List[str]:
        """Tokenize a document for BM25 indexing."""
        return tokenize(
            doc['content'],
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )

    def _index_colbert(self, dataset: Dataset, corpus: List[Dict[str, Any]]) -> None:
        """Create ColBERT index for semantic search."""
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        os.makedirs(colbert_dir, exist_ok=True)

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(
                nbits=2,
            )
            indexer = Indexer('colbert-ir/colbertv2.0', config=config)
            indexer.index(
                name=dataset.name or 'index',
                collection=[doc['content'] for doc in corpus],
                overwrite='reuse'  # type: ignore
            )

        self._index = dataset.name or 'index'

    def _search_lexical(self, query: str) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Perform BM25 lexical search.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        if not self._bm25_index or not self._corpus:
            raise ValueError(
                "BM25 index not created. Please index the dataset first.")

        k = self._args.k or 5

        # Tokenize the query
        tokenized_query = tokenize(
            query,
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )

        # Get scores for the query
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top k documents with their scores
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        documents = []
        for idx, score in top_k:
            documents.append({
                'doc_id': self._corpus[idx]['doc_id'],
                'content': self._corpus[idx]['content'],
                'score': score,
                'original_id': idx
            })

        Logger().debug(
            f"Lexical search for {query} returned {len(documents)} results")

        return (
            [doc['content'] for doc in documents],
            [doc['original_id'] for doc in documents],
            {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        )

    def _search_semantic(self, query: str) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Perform ColBERT semantic search.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        k = self._args.k or 5

        doc_ids, _ranking, scores = worker.searcher.search(query, k=k)

        documents = []
        for doc_id, _, score in zip(doc_ids, _ranking, scores):
            documents.append({
                'doc_id': self._corpus[doc_id]['doc_id'],
                'content': self._corpus[doc_id]['content'],
                'score': score,
                'original_id': doc_id
            })

        Logger().debug(
            f"Semantic search for {query} returned {len(documents)} results")

        return (
            [doc['content'] for doc in documents],
            [doc['original_id'] for doc in documents],
            {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        )

    # pylint: disable=duplicate-code
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
        with intelligent lexical/semantic search selection.
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

PROMPT_EXAMPLES_TOOLS = '''Question: "Were Scott Derrickson and Ed Wood of the same nationality?"

Iteration 1:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search_lexical('Scott Derrickson's nationality')", "search_lexical('Ed Wood's nationality')")]
}
```

Iteration 2:
```json
{
    "thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",
    "actions": ["search_lexical('Scott Derrickson's nationality')", "search_lexical('Ed Wood's nationality')")],
    "observations": [["Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange'."], ["Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic 'Plan 9 from Outer Space'."]]
}

Iteration 3:
```json
{
    "thought": "Both Scott Derrickson and Ed Wood are American based on the retrieved information, so they are of the same nationality.",
    "final_answer": "Yes"
}
```

Consider another example:

Question: "In which county is Kimbrough Memorial Stadium located?"

Iteration 1:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search_lexical('Kimbrough Memorial Stadium')", "search_semantic('Kimbrough Memorial Stadium location')"]
}

Iteration 2:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search_lexical('Kimbrough Memorial Stadium')", "search_semantic('Kimbrough Memorial Stadium location')"],
    "observations": [["Kimbrough Memorial Stadium is owned by Canyon Independent School District, and is primarily used for American football"], \
["Kimbrough Memorial Stadium is a stadium in Canyon, Texas."]]
}
```

Iteration 3:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search_lexical('Canyon Texas county')"]
}
```

Iteration 4:
```json
{
    "thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",
    "actions": ["search_lexical('Canyon Texas county')"],
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
