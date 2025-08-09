"""ColbertV2 with LLM reranker RAG system for document retrieval and question answering."""
import os
import json
from colbert.infra import Run, RunConfig
from colbert import Searcher
from agents.colbertv2.colbertv2 import ColbertV2
from azure_open_ai.batch import queue_batch_job, wait_for_batch_job_and_save_result
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param


class ColbertV2Reranker(Agent):
    """
    ColbertV2 RAG system for document retrieval and question answering using late interaction.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        self._colbertv2 = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval.
        """
        self._colbertv2 = ColbertV2(self._args)
        self._colbertv2.index(dataset)

        self._index = dataset.name or 'index'
        # pylint: disable-next=protected-access
        self._corpus = self._colbertv2._corpus
        self._qa_prompt = dataset.get_prompt('qa_rel')

    def reason(self, _: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question.
        """
        Logger().error(
            "ColBERTV2 Reranker agent does not support single question reasoning. Use multiprocessing_reason instead."
        )
        raise NotImplementedError(
            "ColBERTV2 Reranker agent does not support single question reasoning. Use multiprocessing_reason instead."
        )

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the colbertv2 Reranker agent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the colbertv2 Reranker agent.")

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Multiprocessing reason over the indexed dataset to answer the questions.

        Args:
            questions (list[str]): List of questions to answer.
        """
        if not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")

        if not self._qa_prompt:
            raise ValueError(
                "QA prompt not created. Please index the dataset before retrieving documents.")

        # pylint: disable=duplicate-code
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            searcher = Searcher(index=self._index, collection=[
                                doc['content'] for doc in self._corpus])

        Logger().info("Searching for answers to questions")

        results = searcher.search_all(queries=dict(enumerate(questions)), k=10)

        grouped_results = {}

        Logger().info("Processing results")

        for q_id, doc_id, _, score in results.flat_ranking:
            if q_id not in grouped_results:
                grouped_results[q_id] = []
            grouped_results[q_id].append((doc_id, score))

        os.makedirs(os.path.join(colbert_dir, 'tmp'), exist_ok=True)
        relevance_results_path = os.path.join(
            colbert_dir, 'tmp/relevance_results.jsonl')

        # Create batch jobs for individual document relevance evaluation
        batch_jobs = []
        for q_id, docs in grouped_results.items():
            question = questions[q_id]
            for k, (doc_id, _) in enumerate(docs):
                doc_content = self._corpus[doc_id]['content']
                batch_jobs.append({
                    # Format: question_id_document_index
                    "custom_id": f"{q_id}_{k}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": self._args.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": RELEVANCE_AGENT_PROMPT.format(content=doc_content)
                            },
                            {
                                "role": "user",
                                "content": question
                            }
                        ],
                        "temperature": (default_job_args['temperature']
                                        if supports_temperature_param(self._args.model) else None),
                        "frequency_penalty": default_job_args['frequency_penalty'],
                        "presence_penalty": default_job_args['presence_penalty'],
                        "max_completion_tokens": 10,  # Only need a relevance score
                    }
                })

        batch = queue_batch_job(batch_jobs)

        # pylint: enable=duplicate-code
        Logger().info("Waiting for batch job to finish")

        wait_for_batch_job_and_save_result(batch, relevance_results_path)

        notebooks_dict = {}

        # Collect relevance scores for each document per question
        document_scores = {}  # {q_id: {doc_index: relevance_score}}

        with open(relevance_results_path, 'r', encoding='utf-8') as file:
            for line in file:
                result = json.loads(line)
                custom_id = result['custom_id']

                # Parse custom_id to get question_id and document_index
                try:
                    q_id, doc_idx = custom_id.split('_')
                    q_id = int(q_id)
                    doc_idx = int(doc_idx)
                except ValueError:
                    Logger().error(f"Invalid custom_id format: {custom_id}")
                    continue

                if q_id not in document_scores:
                    document_scores[q_id] = {}

                # Extract relevance score from response
                try:
                    score_text = result['response']['body']['choices'][0]['message']['content'].strip(
                    )
                    relevance_score = int(score_text)
                    # Ensure score is between 0 and 100
                    relevance_score = max(0, min(100, relevance_score))
                    document_scores[q_id][doc_idx] = relevance_score
                    Logger().debug(
                        f"Document {doc_idx} for question {q_id}: relevance score {relevance_score}")
                except (ValueError, KeyError) as e:
                    Logger().warning(
                        f"Failed to parse relevance score for {custom_id}: {e}")
                    # Assume relevant if parsing fails
                    document_scores[q_id][doc_idx] = 100

        # Filter documents based on relevance threshold and create notebooks
        threshold = 50.0

        for q_id, docs in grouped_results.items():
            relevant_docs = []

            if q_id in document_scores:
                for doc_idx, (doc_id, score) in enumerate(docs):
                    relevance_score = document_scores[q_id].get(
                        doc_idx, 100)  # Default to relevant

                    if relevance_score >= threshold:
                        relevant_docs.append((doc_id, score))
            else:
                # If no relevance scores available, use all documents
                relevant_docs = grouped_results[q_id]
                Logger().warning(
                    f"No relevance scores found for question {q_id}, using all documents")

            if len(relevant_docs) == 0:
                Logger().warning(
                    f"No relevant documents found for question {q_id} after applying relevance \
threshold. Using top document.")
                # If no documents are relevant, use the top document
                relevant_docs.append(docs[0])

            retrieved_results = [
                RetrievedResult(
                    doc_id=self._corpus[doc_id]['doc_id'],
                    content=self._corpus[doc_id]['content'],
                    score=score
                )
                for doc_id, score in relevant_docs
            ]

            notebook = NoteBook()
            notebook.update_sources(retrieved_results)

            notes = self._qa_prompt.format(
                context='\n'.join(
                    result['content'] for result in retrieved_results)
            )

            notebook.update_notes(notes)
            notebooks_dict[q_id] = notebook

        return [notebooks_dict[key] for key in sorted(notebooks_dict.keys())]


default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 10,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}


RELEVANCE_AGENT_PROMPT = '''You are a helpful assistant that evaluates the relevance of search \
results to a given query. Your task is to provide a relevance score \
between 0 and 100, where 0 means not relevant at all and 100 means highly relevant and can help answer the query.

The relevance score should be based on the content of the document and how well it matches the query.

Your response should be a single integer value representing the relevance score. Do not include any additional text or explanation.

Document: {content}
'''
