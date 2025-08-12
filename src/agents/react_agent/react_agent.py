"""ReactAgent for reasoning over indexed documents using a retrieval-augmented generation approach.
"""
# pylint: disable=duplicate-code
import json
from multiprocessing import Lock, Pool, cpu_count
import os
from queue import Queue
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger, MainProcessLogger
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

    def _search_documents(self, query: str) -> tuple[list[RetrievedResult], list[int]]:
        """
        Search for documents using the ColBERT retriever.

        Args:
            query (str): The search query.

        Returns:
            tuple[list[RetrievedResult], list[int]]: Tuple containing list of retrieved documents and list of doc_ids.
        """
        doc_ids, _ranking, scores = searcher.search(
            query, k=self._args.k or 5)

        documents = [RetrievedResult(
            doc_id=self._corpus[doc_id]['doc_id'],
            content=self._corpus[doc_id]['content'],
            score=score
        )
            for doc_id, _, score in zip(doc_ids, _ranking, scores)]

        Logger().debug(f"Search results for query '{query}': {documents}")

        return documents, doc_ids

    def _create_initial_request(self, question: str) -> list[dict]:
        """
        Create the initial OpenAI request configuration.

        Args:
            question (str): The question to answer.

        Returns:
            list[dict]: List containing the initial request configuration.
        """
        messages = [
            {"role": "system", "content": REACT_AGENT_PROMPT},
            {"role": "user", "content": question}
        ]

        return [
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
for a given query orderd by relevance, using a dense retriever.",
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

    def _process_tool_calls(self, tool_calls, messages: list, sources: set):
        """
        Process tool calls from the model response.

        Args:
            tool_calls: The tool calls from the model response.
            messages (list): The conversation messages list to update.
            sources (set): The set of source document IDs to update.
        """
        for tool_call in tool_calls:
            if tool_call.function.name == "search":
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query")

                documents, doc_ids = self._search_documents(query)

                sources.update(doc_ids)

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps({
                        "documents": [doc['content'] for doc in documents]
                    })
                })

    def _init_searcher(self) -> None:
        """
        Initializes the searcher for the ReactAgentCustom.
        This is used to set up the searcher with the indexed documents.
        """
        if self._index is None or self._corpus is None:
            raise RuntimeError(
                "Index and corpus must be initialized before creating a searcher.")

        # pylint: disable-next=global-variable-undefined
        global searcher

        if searcher is None:
            colbert_dir = os.path.join(os.path.normpath(
                os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

            Logger().debug("Initializing searcher")

            with lock:
                with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
                    searcher = Searcher(index=self._index, collection=[
                        doc['content'] for doc in self._corpus], verbose=1)

    # pylint: disable-next=too-many-locals
    def reason(self, question: str) -> NoteBook:  # type: ignore
        """
        Reason over the indexed dataset to answer the question.
        """
        Logger().debug(
            f"Starting reasoning for question: {question}, process ID: {os.getpid()}")
        self._init_searcher()

        open_ai_requests = self._create_initial_request(question)
        messages = open_ai_requests[0]["body"]["messages"]

        result = None
        finish_reason = None
        sources = set()
        iteration = 0
        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        while finish_reason != "stop":
            iteration += 1

            result = chat_completions([
                {
                    "custom_id": open_ai_request['custom_id'],
                    **open_ai_request['body'],
                    "messages": messages,
                    "tool_choice": "auto" if iteration < 5 else "none",
                }
                for open_ai_request in open_ai_requests
            ])[0][0]

            Logger().debug(f"Chat completion response: {result}")

            # update usage metrics
            usage_metrics["completion_tokens"] += result.usage.completion_tokens
            usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
            usage_metrics["total_tokens"] += result.usage.total_tokens

            messages.append(result.choices[0].message)

            tool_calls = result.choices[0].message.tool_calls
            finish_reason = result.choices[0].finish_reason

            if tool_calls:
                self._process_tool_calls(tool_calls, messages, sources)

        answer = result.choices[0].message.content.strip()

        Logger().debug(f"Final answer: {answer} for question: {question}")

        notebook = NoteBook()
        notebook.update_sources([
            RetrievedResult(
                doc_id=self._corpus[doc_id]['doc_id'], content=self._corpus[doc_id]['content'])
            for doc_id in sources
        ])
        notebook.update_notes(answer)
        notebook.update_usage_metrics(usage_metrics)

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
        Processes the questions in parallel using multiprocessing.
        This function is used to speed up the reasoning process by using multiple processes.
        It creates a pool of workers and maps the questions to the reason function.

        This function can be overridden by the agent to implement a custom multiprocessing strategy specially needed if 
        the agent will use another device (GPU) to process the questions.

        Args:
            question (list[str]): the given questions

        Returns:
            notebook (list[Notebook]): the detailed findings to help answer all questions (context)
        """
        l = Lock()

        results = []
        with Pool(min(40, cpu_count()), init_agent_worker, [MainProcessLogger().get_queue(), l]) as pool:
            results = pool.map(self.reason, questions)

        return results

def init_agent_worker(q: Queue, l):  # type: ignore
    """
    Initializes the ReactAgentCustom worker.
    """
    Logger(q)
    # pylint: disable-next=global-variable-undefined
    global searcher
    # pylint: disable-next=global-variable-undefined
    global lock
    lock = l
    searcher = None


default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 100,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

REACT_AGENT_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a question, and you \
will need to search for relevant documents that support the answer to the question. You will then use these documents to provide an \
EXACT answer, using only words found in the documents when possible. UNDER no circumstances should you include any additional commentary, \
explanations, reasoning, or notes in your response. Your response MUST be concise and to the point. If the answer can be a single word \
(e.g., Yes, No, a date, or an object), please provide just that word.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \
Each query needs to be optimized to maximize the probability of retrieving the most relevant documents using a semantic retriever. \
As such, you can rephrase the question to make it more specific or to focus on a particular aspect of the question. \
You should then use the retrieved documents to answer the original question.

For example, consider the following question:

"Were Scott Derrickson and Ed Wood of the same nationality?"

You can decompose this question into two sub-questions, and then search for relevant documents for each sub-question.

1. search("Scott Derrickson nationality")

Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like "The Exorcism of Emily Rose" and "Doctor Strange".

2. search("Ed Wood nationality")

Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic "Plan 9 from Outer Space".

Since both Scott Derrickson and Ed Wood are American, you can conclude that they are of the same nationality, and respond only with "Yes".

Consider another question:

"In which county is Kimbrough Memorial Stadium located?"

You can first search for the location of Kimbrough Memorial Stadium.

1. search("Kimbrough Memorial Stadium location")

Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football.

Then, you can search for the county of Canyon, Texas.

2. search("Canyon Texas county")

Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census.

Since Kimbrough Memorial Stadium is located in Canyon, Texas, and Canyon is in Randall County, you can conclude that Kimbrough \
Memorial Stadium is located in Randall County, and respond with "Randall County".

Your final answer MUST be formatted as a single line of text, containing ONLY the answer to the question following the aforementioned rules. \
If the answer cannot be inferred with the information found in the documents, you MUST then respond with "N/A".
'''

REACT_AGENT_PROMPT_SIMPLIFIED = '''You are a helpful Question Answering assistant. You will be presented with a \
question, and you will need to search for relevant documents that support the answer to the question. You will then use 
these documents to provide an EXACT and CONCISE answer to the question. Your answer should only use words from the documents \
you found. UNDER no circumstances should you include any additional commentary, explanations, or reasoning in your final answer.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \
You should then use the retrieved documents to answer the original question, analyzing the retrieved information to provide a correct \
answer.

Your final answer MUST be formatted as a single line of text, containing ONLY the answer to the question. \
If the answer cannot be inferred with the information found in the passages, you MUST then respond with "N/A"
'''
