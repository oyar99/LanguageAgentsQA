"""An agent module."""

from abc import ABC, abstractmethod
from inspect import signature
import json
from multiprocessing import Pool, cpu_count
import os
from typing import Any, Callable, Dict, Optional, Set, Tuple
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger, MainProcessLogger, worker_init
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param


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
        with Pool(min(24, cpu_count()), worker_init, [MainProcessLogger().get_queue()]) as pool:
            results = pool.map(self.reason, questions)

        return results


class IntelligentAgent(Agent, ABC):
    """
    An abstract class representing an intelligent agent that can reason over a dataset.
    It extends the Agent class and provides additional functionality for reasoning.
    """

    def __init__(self, actions: Dict[str, Callable[..., Any]], args):
        super().__init__(args)
        self._actions = actions
        self._max_iteratios = 8
        self._prompt = ""

    def _parse_structured_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the structured JSON response. It attempts to extract JSON content from the response string using 
        various common patterns such as markdown code blocks or JSON-like structures.

        Args:
            response_content (str): Raw response from the model.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response or None if parsing fails.
        """
        try:
            # Try to extract JSON from the response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                json_str = response_content[json_start:json_end].strip()
            elif '{' in response_content and '}' in response_content:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                json_str = response_content[json_start:json_end]
            else:
                # Fallback: try to parse the entire response
                json_str = response_content.strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            Logger().warn(f"Failed to parse structured response: {e}")
            Logger().debug(f"Raw response: {response_content}")
            return None

    def _parse_action(self, action: str) -> Tuple[Callable[..., Tuple[list[str], list[str], Dict[str, str]]], ...]:
        """
        Parse the action string to extract structured actions, and executes thems.

        Args:
            action (str): The action string to parse. Must be of the form action(args).
            args (list): The arguments to pass to the action function. 
            Only two primitive types are supported: str and int.
            Named parameters are not supported. There should no be optional parameters.

        Returns:
            R: The result of the executed action.
        """
        action_name = None
        action_input = None

        if isinstance(action, str) and '(' in action and ')' in action:
            # Extract function name and arguments from string like "search('query')" or "search(query, 5)"
            paren_start = action.find('(')
            paren_end = action.rfind(')')

            action_name = action[:paren_start].strip()
            args_str = action[paren_start + 1:paren_end].strip()

            # Parse comma-separated arguments while respecting quotes
            action_input = []
            if args_str:
                current_arg = ""
                in_quotes = False
                quote_char = None
                i = 0

                while i < len(args_str):
                    char = args_str[i]

                    if not in_quotes and char in ["'", '"']:
                        # Starting a quoted string
                        in_quotes = True
                        quote_char = char
                        current_arg += char
                    elif in_quotes and char == quote_char:
                        # Ending a quoted string
                        in_quotes = False
                        current_arg += char
                        quote_char = None
                    elif not in_quotes and char == ',':
                        # Found a separator outside quotes
                        action_input.append(current_arg.strip())
                        current_arg = ""
                    else:
                        current_arg += char

                    i += 1

                # Add the last argument
                if current_arg.strip():
                    action_input.append(current_arg.strip())

            # Process each argument
            for i, arg in enumerate(action_input):
                if arg.startswith(("'", '"')) and arg.endswith(("'", '"')):
                    # Remove quotes from arguments if present
                    arg = arg[1:-1]
                elif arg.isdigit():
                    # Convert numeric strings to integers
                    arg = int(arg)
                action_input[i] = arg

        if action_name:
            # Extract action from the actions dictionary
            action_func = self._actions.get(action_name.lower())

            Logger().debug(
                f"Parsed action: {action_name} with args: {action_input}")

            # Perform action
            return action_func, *action_input

        Logger().error(f"Invalid action format: {action}")
        raise ValueError(f"Invalid action format: {action}")

    # pylint: disable-next=too-many-locals
    def reason(self, question: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question.

        Args:
            question (str): The question to reason about.

        Returns:
            NoteBook: A notebook containing the reasoning results.
        """
        Logger().debug(
            f"Starting reasoning for question: {question}, process ID: {os.getpid()}")

        stm = []
        sources: Set[int] = set()

        iteration = 0
        final_answer = None

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        while iteration < self._max_iteratios and final_answer is None:
            Logger().debug(
                f"Iteration {iteration + 1} for question: {question}")
            iteration += 1

            messages = [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": question}
            ]

            messages.extend([{"role": "system", "content": memory}
                            for memory in stm])

            Logger().debug(f"Sending messages to model: {messages}")

            open_ai_request = {
                "custom_id": f"react_iteration_{iteration}",
                "model": self._args.model,
                "messages": messages,
                "temperature": default_job_args['temperature']
                if supports_temperature_param(self._args.model) else None,
                "frequency_penalty": default_job_args['frequency_penalty'],
                "presence_penalty": default_job_args['presence_penalty'],
                "max_completion_tokens": 1000,
            }

            result = chat_completions([open_ai_request])[0][0]

            # Update usage metrics
            usage_metrics["completion_tokens"] += result.usage.completion_tokens
            usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
            usage_metrics["total_tokens"] += result.usage.total_tokens

            response_content = result.choices[0].message.content.strip()

            Logger().debug(f"Model response: {response_content}")

            # Parse the structured response
            structured_response = self._parse_structured_response(
                response_content)

            if structured_response is None:
                Logger().error(
                    f"Failed to parse structured response for question: {question}")
                raise ValueError(
                    f"Failed to parse structured response for question: {question}")

            thought = structured_response.get("thought")
            actions = structured_response.get("actions", [])
            final_answer = structured_response.get("final_answer", None)

            turn = {'thought': thought, 'actions': actions, 'observations': []}

            for action in actions:
                # Parse action string to extract function name and arguments
                action_func, *action_args = self._parse_action(action)

                sig = signature(action_func)
                has_context = sig.parameters.get('context', 'None')

                # Execute the action function with or without context
                # Context explains why the action is being taken, and actions can use this information
                # to decide what to do next, and validate whether it makes sense to execute the action or not
                if has_context:
                    observations, action_sources, action_usage_metrics = action_func(
                        *action_args, context=thought)
                else:
                    observations, action_sources, action_usage_metrics = action_func(
                        *action_args)

                turn['observations'].append(observations)

                # Update usage metrics
                usage_metrics["completion_tokens"] += action_usage_metrics.get(
                    "completion_tokens", 0)
                usage_metrics["prompt_tokens"] += action_usage_metrics.get(
                    "prompt_tokens", 0)
                usage_metrics["total_tokens"] += action_usage_metrics.get(
                    "total_tokens", 0)

                # Track sources
                sources.update(action_sources)

            stm.append(json.dumps(turn))

        if final_answer is None:
            final_answer = "N/A"

        Logger().info(
            f"Final answer for question '{question}': {final_answer}")

        # Create notebook with results
        # TODO: Revisit whether this should have knowledge about the corpus
        notebook = NoteBook()
        notebook.update_sources([
            RetrievedResult(
                doc_id=self._corpus[doc_id]['doc_id'],
                content=self._corpus[doc_id]['content']
            )
            for doc_id in sources
        ])
        notebook.update_notes(final_answer)
        notebook.update_usage_metrics(usage_metrics)

        return notebook


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
