"""An agent module."""

from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Optional
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult


class NoteBook:
    """
    A notebook class for storing notes and any other bookeeping stuff the agent needs.
    """

    def __init__(self):
        self._sources = []
        self._notes = None
        self._questions = None
        self._usage_metrics = {}

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
        with Pool(min(4, cpu_count())) as pool:
            results = pool.map(self.reason, questions)

        return results
