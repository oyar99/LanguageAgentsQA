"""An agent module."""

from abc import ABC, abstractmethod
from multiprocessing import Lock, Pool, cpu_count
import traceback
from typing import Dict, List, Optional
from logger.logger import Logger, MainProcessLogger, worker_init
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.agent_worker import init_agent_worker


class NoteBook:
    """
    A notebook class for storing notes and any other bookeeping stuff the agent needs.
    """

    def __init__(self):
        self._sources = []
        self._notes = None
        self._questions = None
        self._usage_metrics = {}
        self._messages = []
        self._context = None

    def update_notes(self, notes: str) -> None:
        """
        Updates the notebook with the given notes.

        Args:
            notes (str): the notes to be added to the notebook
        """
        self._notes = notes

    def get_notes(self) -> Optional[str]:
        """
        Gets the notes from the notebook.

        Returns:
            str: the notes in the notebook
        """
        return self._notes

    def update_sources(self, sources: list[RetrievedResult]) -> None:
        """
        Updates the notebook with the given sources.

        Args:
            sources (list[RetrievedResult]): the sources to be added to the notebook
        """
        self._sources = sources

    def get_sources(self) -> list[RetrievedResult]:
        """
        Gets the sources from the notebook.

        Returns:
            list[RetrievedResult]: the sources in the notebook
        """
        return self._sources

    def update_questions(self, questions: list[str]) -> None:
        """
        Updates the notebook with the given questions.

        Args:
            questions (list[str]): the questions to be added to the notebook
        """
        self._questions = questions

    def get_questions(self) -> Optional[list[str]]:
        """
        Gets the questions from the notebook.

        Returns:
            list[str]: the questions in the notebook
        """
        return self._questions

    def update_usage_metrics(self, metrics: dict) -> None:
        """
        Updates the notebook with the given usage metrics.

        Args:
            metrics (dict): the usage metrics to be added to the notebook
        """
        self._usage_metrics = metrics

    def get_usage_metrics(self) -> dict:
        """
        Gets the usage metrics from the notebook.

        Returns:
            dict: the usage metrics in the notebook
        """
        return self._usage_metrics

    def update_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Updates the notebook with the given messages.

        Args:
            messages (List[Dict[str, str]]): the messages to be added to the notebook
        """
        self._messages = messages

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Gets the messages from the notebook.

        Returns:
            List[Dict[str, str]]: the messages in the notebook
        """
        return self._messages
    
    def update_context(self, context: str) -> None:
        """
        Updates the notebook with the given context.

        Args:
            context (str): the context to be added to the notebook
        """
        self._context = context

    def get_context(self) -> Optional[str]:
        """
        Gets the context from the notebook.

        Returns:
            str: the context in the notebook
        """
        return self._context


class Agent(ABC):
    """
    An abstract class representing a language agent that interacts with the dataset by:

        - Indexing all content into its memory storage mechanism. Different agents
        may implement different memory tiers to store semantic or episodic momories.

    Given a question, the agent can

        - Create a rationale-driven plan based on pre-defined actions to aid in decision-making
        - Execute the plan and determine if it has gathered sufficient resources to answer the question
        - Iterate on the plan as needed

    Finally the agent will return a detailed notebook with its findings for this question so another language
    agent can answers all questions in bulk. Alternatively, the agent can also be a self-contained agent that
    answers the question directly.

    Args:
        ABC: an abstract base class
    """

    def __init__(self, args):
        self._args = args
        self._index = None
        self._corpus = None
        self.support_batch = False
        self.standalone = False

    @abstractmethod
    def index(self, dataset: Dataset) -> None:
        """
        Indexes the contents of the given dataset

        Args:
            dataset (Dataset): The given dataset already initialized
        """

    @abstractmethod
    def reason(self, question: str) -> NoteBook:
        """
        Given a question, reasons about it using its index (memory) and returns a 
        detailed notebook (str) with its findings to generate a correct response.
        The agent should not respond to the question directly. Instead, it should create the notes with all its findings
        so that the response can easily be explainable.

        Args:
            question (str): the given question

        Returns:
            notebook (Notebook): the detailed findings to help answer this question (context)
        """

    @abstractmethod
    def batch_reason(self, questions: list[QuestionAnswer]) -> list[NoteBook]:
        """
        Given a list of questions, reasons about them using its index (memory) and returns a
        detailed notebook (str) with its findings to generate a correct response.
        The agent should not respond to the questions directly. Instead, it should create the notes with all 
        its findings so that the response can easily be explainable.
        This is different from the multiprocessing_reason method since it will not use multiprocessing. Instead,
        it batches all the questions and returns a single notebook.

        Args:
            questions (list[QuestionAnswer]): the given questions
        Returns:
            notebooks (list[NoteBook]): the detailed findings to help answer all questions (context)
        """

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
        results = []
        with Pool(min(24, cpu_count()), worker_init, [MainProcessLogger().get_queue()]) as pool:
            results = pool.map(self.reason, questions)

        return results


class MultiprocessingSearchAgent(Agent, ABC):
    """
    An abstract class representing an agent that supports multiprocessing reasoning.
    It extends the Agent class and provides an advanced implementation for multiprocessing reasoning that exposes
    two resources to the worker processes: a lock and a searcher accesible via the utils.agent_worker module.
    """

    def __init__(self, args, cores=10):
        super().__init__(args)
        self._cores = cores

    def _safe_reason(self, question: str) -> NoteBook:
        try:
            return self.reason(question)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            Logger().error(
                f"Error in reasoning for question '{question}': {e}")
            Logger().debug(traceback.format_exc())
            # Return an empty notebook in case of error
            notebook = NoteBook()

            notebook.update_notes("N/A")

            return notebook

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
        with Pool(min(self._cores, cpu_count()), init_agent_worker, [MainProcessLogger().get_queue(), l]) as pool:
            results = pool.map(self._safe_reason, questions)

        return results

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for MultiprocessingSearchAgent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for MultiprocessingSearchAgent.")


class MultiprocessingStatefulSearchAgent(Agent, ABC):
    """
    An abstract class representing a stateful agent that supports multiprocessing reasoning.
    It extends the Agent class and provides an advanced implementation for multiprocessing reasoning that exposes
    two resources to the worker processes: a lock and a searcher accesible via the utils.agent_worker module.
    """

    def __init__(self, args, cores):
        super().__init__(args)
        self._cores = cores

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Processes the questions in parallel using multiprocessing
        by persisting the state across multiple reasoning sessions.
        This function is used to speed up the reasoning process by using multiple processes.

        Args:
            question (list[str]): the given questions

        Returns:
            notebook (list[Notebook]): the detailed findings to help answer all questions (context)
        """
        l = Lock()

        # Divide questions into groups based on number of cores
        question_groups = self._divide_questions_into_groups(questions)

        cores = 4 if not self._cores else self._cores

        # Create worker processes for each group
        results = []
        with Pool(
            min(cores, len(question_groups)),
            init_agent_worker,
            [MainProcessLogger().get_queue(), l]
        ) as pool:
            # Send each group to a worker process
            group_results = pool.map(
                self._process_question_batch, question_groups)

            # Flatten the results from all groups
            for group_result in group_results:
                results.extend(group_result)

        return results

    @abstractmethod
    def on_complete(self) -> None:
        """
        Called when all questions have been processed.
        """

    def _divide_questions_into_groups(self, questions: list[str]) -> list[list[str]]:
        """
        Divide questions into groups based on the number of cores.

        Args:
            questions: List of questions to divide

        Returns:
            List of question groups
        """
        if not questions:
            return []

        group_size = max(1, len(questions) // self._cores)
        groups = []

        for i in range(0, len(questions), group_size):
            group = questions[i:i + group_size]
            groups.append(group)

        return groups

    def _process_question_batch(self, question_batch: list[str]) -> list[NoteBook]:
        """
        Process a batch of questions synchronously since the agent is stateful.
        This method is called by worker processes.

        Args:
            question_batch: List of questions to process

        Returns:
            List of notebooks with results
        """
        results = []

        Logger().debug("Processing question batch")

        for question in question_batch:
            try:
                notebook = self.reason(question)
                results.append(notebook)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                Logger().error(f"Error processing question '{question}': {e}")
                Logger().error(f"Traceback: {traceback.format_exc()}")

                # Create error notebook
                error_notebook = NoteBook()
                error_notebook.update_notes("N/A")
                error_notebook.update_usage_metrics({
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0
                })
                results.append(error_notebook)

        self.on_complete()

        return results

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for MultiprocessingSearchAgent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for MultiprocessingSearchAgent.")


class SingleProcessAgent(Agent, ABC):
    """
    An abstract class representing an agent that runs in a single process.
    """

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Processes the questions sequentially in a single process.

        Args:
            question (list[str]): the given questions
        Returns:
            notebook (list[Notebook]): the detailed findings to help answer all questions (context)
        """
        results = []
        for question in questions:
            try:
                notebook = self.reason(question)
                results.append(notebook)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                Logger().error(f"Error processing question '{question}': {e}")
                Logger().error(f"Traceback: {traceback.format_exc()}")

                # Create error notebook
                error_notebook = NoteBook()
                error_notebook.update_notes("N/A")
                error_notebook.update_usage_metrics({
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0
                })
                results.append(error_notebook)

        return results

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for SingleProcessingSearchAgent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for SingleProcessingSearchAgent.")


class SelfContainedAgent(Agent, ABC):
    """
    An abstract class representing a self-contained agent that can answer questions directly.
    It extends the Agent class and sets the standalone attribute to True.
    """

    def __init__(self, args):
        Agent.__init__(self, args)
        self.standalone = True
