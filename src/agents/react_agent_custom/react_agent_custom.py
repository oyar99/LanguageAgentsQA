"""ReactAgentCustom for reasoning using custom instruction fine-tuned model with structured output schema.
"""
# pylint: disable=duplicate-code
import os
from typing import Dict, List
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.agent import IntelligentAgent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
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
            "search": self._search_documents
        }
        self._enable_pruning = True
        super().__init__(actions, args)

        self.standalone = True
        self._prompt = REACT_AGENT_CUSTOM_PROMPT

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

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the ReactAgentCustom.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the ReactAgentCustom.")


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

REACT_AGENT_CUSTOM_PROMPT = '''You will be presented with a question, and you will need to search for relevant \
documents that support the answer to the question. You will then use these documents to provide an EXACT answer, \
using only words found in the documents when possible. UNDER no circumstances should you include any additional \
commentary, explanations, reasoning, or notes in your response. Your response MUST be concise and to the point. \
If the answer can be a single word (e.g., Yes, No, a date, or an object), please provide just that word.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \
Each query needs to be optimized to maximize the probability of retrieving the most relevant documents using a semantic retriever. \
As such, you can rephrase the question to make it more specific or to focus on a particular aspect of the question. \
You should then use the retrieved documents to answer the original question.

## EXAMPLES:

For example, consider the following question:

"Were Scott Derrickson and Ed Wood of the same nationality?"

You can decompose this question into two sub-questions, and then search for relevant documents for each sub-question.

{\
"thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",\
"actions": ["search('Scott Derrickson nationality')", "search('Ed Wood nationality')"]\
}

In the next iteration, you will be given the search results for both queries in the "observations" field:

{\
"thought": "I need to find the nationalities of both Scott Derrickson and Ed Wood to compare them.",\
"actions": ["search('Scott Derrickson nationality')", "search('Ed Wood nationality')"],\
"observations": [["Scott Derrickson is an American film director, producer, and screenwriter. He is known for his work in the horror genre, including \
films like 'The Exorcism of Emily Rose' and 'Doctor Strange'."], ["Ed Wood was an American filmmaker, actor, and writer, often regarded as one of the worst directors in film history. He is best known \
for his cult classic 'Plan 9 from Outer Space'."]]\
}

You can then use the information from the observations to answer the original question:

{\
"thought": "Both Scott Derrickson and Ed Wood are American based on the retrieved information, so they are of the same nationality.",\
"final_answer": "Yes"\
}

Consider another example question:

"In which county is Kimbrough Memorial Stadium located?"

You can first search for the location of Kimbrough Memorial Stadium:

{\
"thought": "I need to find where Kimbrough Memorial Stadium is located.",\
"actions": ["search('Kimbrough Memorial Stadium location')"]\
}

You will then receive the following observation:

{\
"thought": "I need to find where Kimbrough Memorial Stadium is located.",\
"actions": ["search('Kimbrough Memorial Stadium location')"],\
"observations": [["Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District, and is primarily \
used for American football."]]\
}

Then, you can search for the county of Canyon, Texas:

{\
"thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",\
"actions": ["search('Canyon Texas county')"]\
}

You will then receive the following observation:

{\
"thought": "The stadium is in Canyon, Texas, but I need to find which county Canyon is in.",\
"actions": ["search('Canyon Texas county')"],\
"observations": [["Canyon is a city in, and the county seat of, Randall County, Texas, United States. The population was 13,303 at the 2010 census."]]\
}

You will then be able to answer the original question:

{\
"thought": "Kimbrough Memorial Stadium is in Canyon, Texas, and Canyon is in Randall County.",\
"final_answer": "Randall County"\
}

## RESPONSE FORMAT RULES

Respond with exactly one JSON object per response and do not include any text before or after the JSON object. \
Use either the intermediate format or the final format, never both in the same response.

Your intermediate responses must be in valid JSON format with the following structure:

{\
"thought": "Your reasoning process",\
"actions": ["search('search query')"]\
}

During intermediate responses, actions must not be empty and must contain at least one action.

Your final answer must be formatted in valid JSON format with the following structure. Keep in mind that final_answer must contain \
ONLY the answer to the question. If the answer cannot be inferred with the information found in the documents, you must then set final_answer to "N/A":

{\
"thought": "Your final reasoning process",\
"final_answer": "your final answer"\
}

Ensure all string values in your JSON response have properly escaped quotes when necessary if using double quotes.

The schema must also adhere to the following rules:

- "thought" is a string that describes your reasoning process.
- "actions" is an array of strings that are valid python expressions that correspond to the actions you will take to answer the question. Supported actions:
    - "search('search query')": Search for relevant documents for the given query.

- "final_answer" is a string that contains your final answer to the question. This field should only be included when providing your final answer. \
If included, "actions" must not be included.
'''
