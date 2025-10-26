"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.action import Action
from models.agent import NoteBook
from models.react_agent import IntelligentAgent
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
        self._prompt = ''
        super().__init__(actions, "", args)

        agent_args = args.agent_args or {}
        self._enable_pruning = agent_args.get('pruning', False)
        self._enable_interleave_reflection = agent_args.get('interleave_reflection', False)


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

        self._prompt += dataset.get_prompt('react_footer_2')
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
