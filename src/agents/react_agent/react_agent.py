"""ReactAgent for reasoning over indexed documents using a retrieval-augmented generation approach.
"""
# pylint: disable=duplicate-code
import json
import os
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param


class ReactAgent(Agent):
    """
    ReactAgent for reasoning over indexed documents using a retrieval-augmented generation approach.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._searcher = None
        self._args = args
        super().__init__(args)

        self.standalone = True

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval.
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

        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            self._searcher = Searcher(index=self._index, collection=[
                doc['content'] for doc in self._corpus])

    # pylint: disable-next=too-many-locals
    def reason(self, question: str) -> NoteBook:  # type: ignore
        """
        Reason over the indexed dataset to answer the question.
        """
        messages = [
            {"role": "system", "content": REACT_AGENT_PROMPT},
            {"role": "user", "content": question}
        ]

        open_ai_requests = [
            {
                "custom_id": 1,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._args.model,
                    "messages": messages,
                    "stop": ["\n"],
                    "temperature": default_job_args['temperature'] 
                        if supports_temperature_param(self._args.model) else None,
                    "frequency_penalty": default_job_args['frequency_penalty'],
                    "presence_penalty": default_job_args['presence_penalty'],
                    "max_completion_tokens": 500,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "Returns a list of 5 most relevant documents \
for a given query orderd by relevance.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The search query.",
                                        },
                                    },
                                    "required": ["query"],
                                },
                            }
                        }
                    ],
                    "tool_choice": "auto",
                },
            }
        ]

        result = None
        finish_reason = None
        sources = set()

        while finish_reason != "stop":
            result = chat_completions([
                {
                    "custom_id": open_ai_request['custom_id'],
                    **open_ai_request['body'],
                    "messages": messages
                }
                for open_ai_request in open_ai_requests
            ])[0][0]

            Logger().debug(f"Chat completion response: {result}")

            messages.append(result.choices[0].message)

            tool_calls = result.choices[0].message.tool_calls

            finish_reason = result.choices[0].finish_reason

            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "search":
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        doc_ids, _ranking, scores = self._searcher.search(
                            query, k=self._args.k or 5)

                        documents = [RetrievedResult(
                            doc_id=self._corpus[doc_id]['doc_id'],
                            content=self._corpus[doc_id]['content'],
                            score=score
                        )
                            for doc_id, _, score in zip(doc_ids, _ranking, scores)]

                        sources.update(doc_id for doc_id in doc_ids)

                        Logger().debug(
                            f"Search results for query '{query}': {documents}")

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": json.dumps({
                                "documents": [doc['content'] for doc in documents]
                            })
                        })

        answer = result.choices[0].message.content.strip()

        Logger().debug(f"Final answer: {answer} for question: {question}")

        notebook = NoteBook()
        notebook.update_sources([
            RetrievedResult(
                doc_id=self._corpus[doc_id]['doc_id'], content=self._corpus[doc_id]['content'])
            for doc_id in sources
        ])
        notebook.update_notes(answer)

        return notebook

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the colbertv2 agent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the ReactAgent.")

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Reason over the indexed dataset to answer the question.
        """
        notebooks = []

        for question in questions:
            notebook = self.reason(question)
            notebooks.append(notebook)

        return notebooks


default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 100,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

REACT_AGENT_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a question, and you \
will need to search for relevant documents that support the answer to the question. You will then use these documents to provide an \
EXACT and CONCISE answer to the question. Your answer should only use words from the documents you found. UNDER no circumstances \
should you include any additional commentary, explanations, or reasoning in your final answer.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \
You should then use the retrieved documents to answer the original question, analyzing the retrieved information to provide a correct \
answer.

Your final answer MUST be formatted as a single line of text, containing ONLY the answer to the question. \
If the answer cannot be inferred with the information found in the passages, you MUST then respond with "N/A"
'''
