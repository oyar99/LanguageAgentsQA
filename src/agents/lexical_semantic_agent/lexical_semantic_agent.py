"""LexicalSemanticAgent for intelligent search using both lexical and semantic approaches.

This agent automatically decides between BM25 lexical search and ColBERT semantic search
based on the nature of the query, following ReAct framework with manual tool invocation.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List, Any, Optional

from rank_bm25 import BM25Okapi as BM25Ranker
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.agent import IntelligentAgent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
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
            "search_lexical": self._search_lexical,
            "search_semantic": self._search_semantic,
        }
        super().__init__(actions, args)

        self.standalone = True
        self._prompt = LEXICAL_SEMANTIC_AGENT_PROMPT

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

    def _search_lexical(self, query: str, k: Optional[int] = None) -> tuple[List[str], List[str], Dict[str, int]]:
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

        if k is None:
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

    def _search_semantic(self, query: str, k: Optional[int] = None) -> tuple[List[str], List[str], Dict[str, int]]:
        """
        Perform ColBERT semantic search.

        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved documents with scores.
        """
        if k is None:
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

    def batch_reason(self, _: List[QuestionAnswer]) -> List[NoteBook]:  # type: ignore
        """
        Batch reasoning is not implemented for the LexicalSemanticAgent.

        Raises:
            NotImplementedError: Batch reasoning is not implemented.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the LexicalSemanticAgent.")


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

LEXICAL_SEMANTIC_AGENT_PROMPT = '''You are an intelligent search agent that can choose between two search methods
to find relevant documents: lexical search and semantic search.

## AVAILABLE TOOLS:

1. **search_lexical(query)**: Performs exact keyword matching using BM25 algorithm. Best for:
   - Names of people, places, organizations, products
   - Specific technical terms or proper nouns
   - Exact phrases or quotes
   - When you need precise term matching

2. **search_semantic(query)**: Performs meaning-based search using ColBERT. Best for:
   - Conceptual queries about topics or themes
   - Questions requiring understanding of context
   - When the answer might use different words than the query
   - Complex reasoning or analytical questions

## SEARCH STRATEGY:

Choose your search method based on the query characteristics:
- If the query contains **named entities** (people, places, organizations, products, etc.), use `search_lexical`
- For **conceptual or analytical** queries, use `search_semantic`
- You can use both methods if needed to gather comprehensive information

## RESPONSE FORMAT:

Respond with exactly one JSON object per response. Use either the intermediate format or the final format, never both.

**Intermediate responses** (while searching):
```json
{
  "thought": "Your reasoning about what information you need and which search method to use",
  "actions": ["search_lexical('specific name or term')", "search_semantic('conceptual query')"]
}
```

**Final response** (when ready to answer):
```json
{
  "thought": "Your final reasoning based on the retrieved information",
  "final_answer": "Concise, exact answer to the question"
}
```

## INSTRUCTIONS:

1. **Think step by step** about what information you need
2. **Choose the appropriate search method** based on query type
3. **Use retrieved information** to provide accurate, concise answers
4. **Be precise** - if the answer is a single word, date, or name, provide just that
5. **If information is insufficient**, set final_answer to "N/A"

## EXAMPLE:

Question: "What nationality is Scott Derrickson?"

Iteration 1:
```json
{
  "thought": "I need to find information about Scott Derrickson's nationality. Since this is asking about a specific person's nationality, I should use lexical search to find exact mentions of his name.",
  "actions": ["search_lexical('Scott Derrickson nationality')"]
}
```

Iteration 2 (after receiving search results):
```json
{
  "thought": "The search results show that Scott Derrickson is an American film director. Based on this information, I can provide the answer.",
  "final_answer": "American"
}
```

Your goal is to find the most relevant information using the appropriate search method \
and provide accurate, concise answers.'''
