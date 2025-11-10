"""
A DAG-based agent that decomposes questions into directed acyclic graphs.
Each node represents a sub-question that must be answered to reach the final answer.
"""

# pylint: disable=duplicate-code, too-many-lines

from abc import ABC
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.action import Action
from models.agent import (
    MultiprocessingSearchAgent,
    MultiprocessingStatefulSearchAgent,
    NoteBook,
    SingleProcessAgent
)
from models.base_dag import BaseDAGAgent as BaseDag, BaseDAGNode, DEFAULT_DAG_JOB_ARGS
from models.dataset import Dataset
from models.react_agent import BaseIntelligentAgent
from models.retrieved_result import RetrievedResult
from models.question_answer import QuestionAnswer
from utils.model_utils import supports_temperature_param
from utils.structure_response import parse_structured_response


class ReactAgent(BaseIntelligentAgent, ABC):
    """
    A React agent that uses a single processing search agent as a helper to answer questions.
    """

    def __init__(self, actions: Dict[str, Action], extra_prompt: str, args):
        """
        Initialize the React agent.

        Args:
            actions (Dict[str, Action]): A dictionary of actions the agent can perform.
            extra_prompt (str): Additional instructions to include in the prompt for the agent such as \
few-shot examples.
            args: Agent arguments.
        """
        BaseIntelligentAgent.__init__(self, actions, extra_prompt, args)
        self._max_iterations = 6
        self.corpus = None

    def index(self, _) -> None:
        """
        Index the dataset (not implemented for ReactAgent helper).

        Raises:
            NotImplementedError: Indexing is not implemented for ReactAgent Helper.
        """
        raise NotImplementedError(
            "Indexing is not implemented for ReactAgent Helper.")

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for SingleProcessingSearchAgent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for ReactAgent Helper.")

# pylint: disable-next=too-few-public-methods, too-many-instance-attributes
class DAGNode(BaseDAGNode):
    """
    Represents a node in the DAG structure.
    Each node contains a sub-question and tracks its dependencies and completion status.
    """

    def __init__(self, node_id: str, sub_question: str, dependencies: List[str] = None):
        """
        Initialize a DAG node.

        Args:
            node_id (str): Unique identifier for this node
            sub_question (str): The sub-question this node represents
            dependencies (List[str]): List of node IDs that must be completed before this node
        """
        super().__init__(node_id, sub_question, dependencies)
        # Additional fields specific to original DAG agent
        self.context_history = []
        self.formulated_question = None

    def to_dict(self, include_sources: bool = False, include_context: bool = True) -> Dict:
        """
        Serialize the DAG node to a dictionary for the DAG execution agent.

        Args:
            include_sources (bool): Whether to include sources field (only needed for backtracking)
            include_context (bool): Whether to include context field

        Returns:
            Dict: Serialized node data
        """
        node_dict = {
            "node_id": self.node_id,
            "sub_question": self.sub_question,
            "dependencies": self.dependencies
        }

        # Add result if node is completed
        if self.is_completed and self.result:
            node_dict["result"] = self.result

        # Add context (reasoning thought - conclusion) for completed nodes
        # context is only not available for inferred results and failed nodes
        if include_context and self.context:
            node_dict["context"] = self.context

        # Add sources only when requested (for backtracking scenarios)
        if include_sources and self.unique_sources:
            node_dict["sources"] = [
                src["content"] for src in self.unique_sources
            ]

        return node_dict


class BaseDAGAgent(BaseDag, ABC):
    """
    A DAG-based agent that decomposes questions into a directed acyclic graph structure.
    Each node represents a sub-question that needs to be answered sequentially based on dependencies.

    The agent operates in two phases:
    1. Planning Phase: Decompose the main question into a DAG of sub-questions
    2. Execution Phase: Execute nodes in topological order, using results from dependencies
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args):
        """
        Initialize the DAG agent.

        Args:
            search_function: Function that takes a query and returns (documents, sources, metrics)
            args: Agent arguments
        """
        super().__init__(search_function, args)
        self._search_function = search_function
        self._max_iterations = 8

        self.solve_command_example = ""
        self.final_answer_command_example = ""
        self.alternative_answer_command_example = ""

    def _commands_support_tools(self, available_tools: List[str] = None) -> bool:
        """
        Check if any of the available commands support tool invocation.

        Args:
            available_tools (List[str]): List of available command names

        Returns:
            bool: True if any command supports tools, False otherwise
        """
        if not available_tools:
            return False

        # Commands that support tool invocation
        tool_supporting_commands = {"SOLVE"}

        return bool(set(available_tools) & tool_supporting_commands)

    # pylint: disable-next=too-many-locals
    def _query_dag_agent(
        self,
        user_query: str,
        custom_id: str,
        dag_state: str = None,
        available_tools: List[str] = None
    ) -> Tuple[Tuple[str, Optional[str], List[Any]], Dict[str, int]]:
        """
        Unified method to query the DAG execution agent with structured JSON response.

        Args:
            user_query (str): The tool call query to send to the DAG agent (e.g., "SOLVE(node_id='node_1')")
            custom_id (str): Custom ID for the request
            dag_state (str): Optional DAG state to inject into the system prompt
            available_tools (List[str]): Optional list of tools to make available \
(e.g., ["SOLVE", "ALTERNATIVE_ANSWER"])

        Returns:
            Tuple[Tuple[str, str, List[Any]], Dict[str, int]]: Answer (answer, tool, tool_args) and usage metrics
        """
        # Build the system prompt with tool filtering and DAG state if provided
        system_prompt = self._build_system_prompt_with_tools(available_tools)
        if dag_state:
            system_prompt = f"{system_prompt}\n## Current DAG State\n\n{dag_state}"

        Logger().debug(f"""DAG Agent system prompt:
{system_prompt}""")

        # Build dynamic schema based on available tools
        schema_properties = {
            "thought": {"type": "string"},
            "answer": {"type": "string"}
        }
        required_fields = ["thought", "answer"]

        # Add tool and query fields only for commands that support tools
        if self._commands_support_tools(available_tools):
            schema_properties["tool"] = {
                "type": "string",
                "description": "The tool to invoke, or null if providing an answer."
            }
            schema_properties["query"] = {
                "type": "string"
            }
            required_fields.extend(["tool", "query"])

        model = self._think_model if custom_id == 'dag_synthesis' else self._args.model

        request = {
            "custom_id": custom_id,
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": DEFAULT_DAG_JOB_ARGS['temperature']
            if supports_temperature_param(model) else None,
            "frequency_penalty": DEFAULT_DAG_JOB_ARGS['frequency_penalty'],
            "presence_penalty": DEFAULT_DAG_JOB_ARGS['presence_penalty'],
            "max_completion_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "dag_agent_response",
                    "schema": {
                        "type": "object",
                        "properties": schema_properties,
                        "required": required_fields,
                        "additionalProperties": False
                    }
                }
            }
        }

        result = chat_completions([request])[0][0]

        # Parse the structured response
        structured_response = parse_structured_response(
            result.choices[0].message.content.strip())

        answer = structured_response['answer']
        thought = structured_response['thought']
        tool: Optional[str] = structured_response.get('tool', None)
        tool_args = []

        if tool is not None and tool.upper() == "SEARCH":
            tool_args = [structured_response.get('query', '')]

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        Logger().debug(
            f"DAG agent query result - \
Thought: {structured_response['thought']}, Answer: {answer}, Tool: {tool}, Tool Args: {tool_args}")

        return (answer, thought, tool, tool_args), usage_metrics

    def _build_system_prompt_with_tools(self, available_tools: List[str] = None) -> str:
        """
        Build the system prompt with only the specified tools available.

        Args:
            available_tools (List[str]): List of tool names to include (e.g., ["SOLVE", "ALTERNATIVE_ANSWER"])

        Returns:
            str: The filtered system prompt
        """

        # Command definitions mapping
        command_definitions = {
            "SOLVE": """- SOLVE(node_id: str): Solve the sub-question for the given node ID using the results \
field and the sources dependencies. Only if the information is not enough to answer the question, you must \
invoke the SEARCH tool to find relevant documents using a query that contains all relevant keywords or entities \
from the sub-question dependencies **RESULT** and **CONTEXT** fields.""",

            "ALTERNATIVE_ANSWER": """- ALTERNATIVE_ANSWER(node_id: str, exclude: List[str]): Provide an \
alternative answer for the given node ID based on the sources provided. This answer must be different from \
all other answers in the exclude list. However, it should be FLEXIBLE in interpretation, and consider \
plausible assumptions or entities in the sources. Only when all alternatives are definitely exhausted \
should you return 'N/A'. The returned answer must be a string directly extracted or inferred from the \
sources. Remember to exhaust all possible alternatives before returning 'N/A' even if they assume typos, \
other interpretations, or entities that may not seem related at first glance.""",

            "FINAL_ANSWER": """- FINAL_ANSWER(question: str): Answer the given question based on \
the available information from each node to formulate a CONCISE and COMPLETE answer. Provide an EXACT answer \
using only words found in the results when possible. DO NOT REPEAT the question in your answer under any circumstances. \
If the answer can be a single word (e.g., Yes, No, a date, or an object), please provide just that word. If the \
information seems insufficient, please make plausible assumptions about the available information, assuming typos, flexible \
interpretations, or best possible answer with the available information. Only if **definitely** no answer exists, \
respond with 'N/A'."""
        }

        # Filter commands based on available_tools list
        filtered_commands = []
        for tool_name in available_tools:
            if tool_name in command_definitions:
                filtered_commands.append(command_definitions[tool_name])

        # Build the base prompt
        base_prompt = '''You are an intelligent DAG execution agent that responds to commands only. \
You help with complex commands related to a DAG (Directed Acyclic Graph) of sub-questions by thinking \
step-by-step about the command and the current state of the DAG. \

'''

        # Add tools section only if any command supports tools
        tools_section = ""
        if self._commands_support_tools(available_tools):
            tools_section = '''You have the following tools at your disposal to execute searches if required. \

## Tools

- SEARCH(query: str): Search for relevant documents for the given query using a semantic retriever. \
Queries must explicitly mention entities or keywords related to the question to obtain relevant results. \
Queries must not contain any node IDs.

Example:

{
    "tool": "SEARCH",
    "query": "What is the capital of France?"
}

When you do not invoke a tool, you must respond with the final answer for the provided command in the "answer" field, \
and not include any tool calls.

'''

        # Commands section
        commands_prompt = '''## Commands

You respond to the following types of commands. Do not provide any explanations or additional text. Only respond \
with the exact output as specified for each command.

'''

        commands_section = "\n\n".join(filtered_commands)

        # Build examples section based on available tools
        examples = {}
        examples["SOLVE"] = self.solve_command_example

        examples["ALTERNATIVE_ANSWER"] = self.alternative_answer_command_example

        examples["FINAL_ANSWER"] = self.final_answer_command_example

        # Filter examples based on available tools
        filtered_examples = []
        for tool_name in available_tools:
            if tool_name in examples:
                filtered_examples.append(examples[tool_name])

        examples_section = ""
        if filtered_examples:
            examples_section = "\n\n".join(filtered_examples)

        return f"{base_prompt}{tools_section}{commands_prompt}{commands_section}{examples_section}"

    def _extract_alternative_answer(self, node: DAGNode, nodes: Dict[str, DAGNode]) -> Tuple[str, str, Dict[str, int]]:
        """
        Try to extract an alternative answer from a node's sources using the unified DAG execution agent.

        Args:
            node (DAGNode): The node to extract alternative answer from

        Returns:
            Tuple[str, str, Dict[str, int]]: Alternative answer/thought and usage metrics
        """
        dag_state = self._serialize_dag_state(
            nodes, include_sources_for=[node.node_id], exclude_context=True)

        # Format the exclude list for the ALTERNATIVE_ANSWER tool
        exclude_list = str(node.alternative_results)

        # Use the new ALTERNATIVE_ANSWER tool format
        user_query = f'ALTERNATIVE_ANSWER(node_id="{node.node_id}", exclude={exclude_list})'

        Logger().debug(
            f"Extracting alternative answer with query: {user_query}")

        (alternative_answer, thought, _, _), usage_metrics = self._query_dag_agent(
            user_query, f"alternative_{node.node_id}", dag_state, available_tools=["ALTERNATIVE_ANSWER"])

        Logger().debug(
            f"Found alternative answer for node {node.node_id}: {alternative_answer}")
        return alternative_answer, thought, usage_metrics

    def _get_executable_nodes(self, nodes: Dict[str, DAGNode]) -> List[str]:
        """
        Get nodes that can be executed (all dependencies completed and not failed).

        Args:
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            List[str]: List of node IDs that can be executed
        """
        executable = []

        for node_id, node in nodes.items():
            if node.is_completed or node.is_failed:
                continue

            # Check if all dependencies are completed (and not failed)
            deps_completed = all(
                nodes[dep].is_completed and not nodes[dep].is_failed for dep in node.dependencies
            )

            if deps_completed:
                executable.append(node_id)

        return executable

    def _attempt_solve_subquestion(
        self,
        node: DAGNode,
        nodes: Dict[str, DAGNode]
    ) -> Tuple[Dict[str, int], Tuple[str, str]]:
        """
        Attempts to solve the question for the current node using the unified DAG execution agent.

        Args:
            node (DAGNode): The current node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            Tuple[Dict[str, int], Tuple[str, str]]: Usage metrics and the response (answer, formulated_question)
        """
        # If question does not have deps, then we just return it as is
        if not node.dependencies:
            return {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            }, ("", node.sub_question)

        # Serialize current DAG state
        dag_state = self._serialize_dag_state(nodes)

        # Use the SOLVE command
        user_query = f'SOLVE(node_id="{node.node_id}")'

        (answer, _, tool, tool_args), usage_metrics = self._query_dag_agent(
            user_query, f"dag_solve_subquestion_{node.node_id}", dag_state, available_tools=["SOLVE"])

        Logger().debug(
            f"Node {node.node_id} answer: {answer}, tool: {tool}, tool_args: {tool_args}")

        if tool == "SEARCH" and tool_args:
            return usage_metrics, (None, tool_args[0])

        return usage_metrics, (answer, node.sub_question)

    # pylint: disable-next=too-many-locals
    def _execute_node(
        self,
        node: DAGNode,
        nodes: Dict[str, DAGNode]
    ) -> Tuple[Dict[str, int], List[RetrievedResult]]:
        """
        Execute a single node by searching for information to answer its sub-question.

        Args:
            node (DAGNode): The node to execute
            nodes (Dict[str, DAGNode]): All nodes for context

        Returns:
            Tuple[Dict[str, int], List[RetrievedResult]]: Usage metrics and sources
        """
        formulate_metrics, (answer, question) = self._attempt_solve_subquestion(
            node, nodes)

        # If we have an answer directly, use it
        if answer is not None and answer != "":
            Logger().debug(
                f"Node {node.node_id} solved directly with answer: {answer}")
            node.result = answer
            node.is_completed = True
            node.alternative_results.append(answer)
            return formulate_metrics, []

        # If we do not have an answer for this sub-question, we invoke the search tool with the given query
        # Search tool is another ReAct agent call that uses the given search function

        # Store the formulated question for potential backtracking
        node.formulated_question = question

        notebook = self._subquestion_agent.reason(question)

        answer = notebook.get_notes()
        sources = notebook.get_sources()
        last_thought = notebook.get_context()

        total_usage_metrics = {
            "completion_tokens": (
                formulate_metrics["completion_tokens"] +
                notebook.get_usage_metrics().get("completion_tokens", 0)
            ),
            "prompt_tokens": (
                formulate_metrics["prompt_tokens"] +
                notebook.get_usage_metrics().get("prompt_tokens", 0)
            ),
            "total_tokens": (
                formulate_metrics["total_tokens"] +
                notebook.get_usage_metrics().get("total_tokens", 0)
            )
        }

        # In case a node is executed multiple times due to backtracking, we append new sources
        # so we can compute retrieval metrics
        node.sources.extend(sources)

        # Update unique sources by deduplicating based on doc_id
        seen_doc_ids = {src['doc_id'] for src in node.unique_sources}
        for source in sources:
            if source['doc_id'] not in seen_doc_ids:
                node.unique_sources.append(source)
                seen_doc_ids.add(source['doc_id'])

        if answer == 'N/A':
            Logger().debug(f"Node {node.node_id} failed to find an answer")
            node.is_failed = True
        else:
            Logger().debug(
                f"Node {node.node_id} succeeded with answer: {answer}")
            node.result = answer
            node.is_completed = True
            node.context = last_thought
            node.context_history.append(last_thought)
            node.alternative_results.append(answer)

        return total_usage_metrics, sources

    def _attempt_backtrack(self, failed_node_id: str, nodes: Dict[str, DAGNode]) -> Tuple[bool, Dict[str, int]]:
        """
        Attempt to backtrack when a node fails by finding alternative answers in dependencies.
        Tries all candidates sorted by distance until one succeeds.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            Tuple[bool, Dict[str, int]]: Success flag and usage metrics from backtracking
        """
        # Get all backtrack candidates sorted by distance (closest ancestors first)
        backtrack_candidates = self._find_backtrack_candidates(
            failed_node_id, nodes)

        if not backtrack_candidates:
            Logger().debug(
                f"No backtrack candidates found for failed node {failed_node_id}")
            return False, {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        total_usage = {"completion_tokens": 0,
                       "prompt_tokens": 0, "total_tokens": 0}

        # Try each candidate until we find one that works
        for candidate_id in backtrack_candidates:
            backtrack_node = nodes[candidate_id]

            Logger().debug(
                f"Attempting backtrack of node {candidate_id} for failed node {failed_node_id}")

            alternative_answer, thought, alt_usage = self._extract_alternative_answer(
                backtrack_node, {backtrack_node.node_id: backtrack_node})

            # Add usage metrics regardless of success
            for key in total_usage:
                total_usage[key] += alt_usage.get(key, 0)

            # Check if this is truly a new alternative (not N/A, not already tried)
            if (alternative_answer != "N/A" and
                    alternative_answer not in backtrack_node.alternative_results):

                # Update the node with alternative answer
                backtrack_node.result = alternative_answer
                # Add the new alternative to the list
                backtrack_node.alternative_results.append(alternative_answer)
                # Update the context to indicate that we backtracked
                backtrack_node.context = thought
                # Add the new thought to context history
                backtrack_node.context_history.append(thought)

                # Reset all dependent nodes (including the failed one)
                self._reset_dependent_nodes(candidate_id, nodes)

                if len(backtrack_node.alternative_results) > 5:
                    Logger().warn(
                        f"Node {candidate_id} has more than 5 alternative answers, \
which may indicate excessive backtracking.")
                    backtrack_node.alternatives_exhausted = True

                Logger().debug(
                    f"Successfully backtracked with alternative: {alternative_answer}")
                return True, total_usage

            # This candidate is exhausted, mark it and continue to next candidate
            backtrack_node.alternatives_exhausted = True
            # Revert to first answer as it is more likely correct
            if backtrack_node.alternative_results:
                backtrack_node.result = backtrack_node.alternative_results[0]
            # Revert to original context as well
            backtrack_node.context = backtrack_node.context_history[0]

            Logger().debug(
                f"Candidate {candidate_id} exhausted, trying next candidate")

        # All candidates exhausted
        Logger().debug(
            f"All backtrack candidates exhausted for failed node {failed_node_id}")
        return False, total_usage

    def _synthesize_final_answer(
        self,
        nodes: Dict[str, DAGNode],
        original_question: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Synthesize the final answer from all completed nodes using the unified DAG execution agent.

        Args:
            nodes (Dict[str, DAGNode]): All completed nodes
            original_question (str): The original question

        Returns:
            Tuple[str, Dict[str, int]]: Final answer and usage metrics
        """
        # Try to synthesize final answer even if no nodes completed

        # Create DAG state with sources only for failed nodes
        failed_node_ids = [node_id for node_id, node in nodes.items()
                           if node.is_failed and node.unique_sources]
        dag_state = self._serialize_dag_state(
            nodes, include_sources_for=failed_node_ids)

        # Use the new FINAL_ANSWER tool format
        user_query = f'FINAL_ANSWER(question="{original_question}")'

        Logger().debug(
            f"Synthesizing final answer with query: {user_query}")

        (final_answer, _, _, _), usage_metrics = self._query_dag_agent(
            user_query, "dag_synthesis", dag_state, available_tools=["FINAL_ANSWER"])

        return final_answer, usage_metrics

    def index(self, dataset: Dataset) -> None:
        """
        Prepares the agent with the given dataset.

        Args:
            dataset (Dataset): the dataset to process
        """
        self.solve_command_example = dataset.get_prompt(
            'solve_command_example')
        self.final_answer_command_example = dataset.get_prompt(
            'final_answer_command_example')
        self.alternative_answer_command_example = dataset.get_prompt(
            'alternative_answer_command_example')

        super().index(dataset)

    # pylint: disable-next=too-many-locals
    def reason(self, question: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question using DAG decomposition.

        Args:
            question (str): The question to reason about.

        Returns:
            NoteBook: A notebook containing the reasoning results.
        """
        Logger().debug(
            f"Starting DAG reasoning for question: {question}, process ID: {os.getpid()}")

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        messages = []
        sources: Dict[str, List[RetrievedResult]] = {}

        # Phase 1: Generate DAG plan
        response_content, planning_usage = self._create_dag_plan(question)

        # Update usage metrics
        for key in usage_metrics:
            usage_metrics[key] += planning_usage.get(key, 0)

        messages.append({"role": "assistant", "content": response_content})

        # Parse the DAG plan
        nodes = self._parse_dag_plan(response_content, DAGNode)

        # Phase 2: Execute DAG nodes with immediate backtracking
        iteration = 0
        while iteration < self._max_iterations:
            executable_nodes = self._get_executable_nodes(nodes)

            if not executable_nodes:
                # All nodes completed or no more executable nodes
                break

            Logger().debug(
                f"DAG iteration {iteration + 1}: Executing nodes {executable_nodes}")

            # Execute all available nodes
            for node_id in executable_nodes:
                node = nodes[node_id]
                node_usage, node_sources = self._execute_node(node, nodes)

                # Update usage metrics
                for key in usage_metrics:
                    usage_metrics[key] += node_usage.get(key, 0)

                # Track sources
                if node_sources:
                    folder_id = f"node_{node_id}"
                    sources[folder_id] = list(node_sources)

                # Check if node failed and attempt backtracking
                if node.is_failed:
                    Logger().debug(
                        f"Node {node_id} failed, attempting backtracking")
                    backtrack_success, backtrack_usage = self._attempt_backtrack(
                        node_id, nodes)

                    # Add backtracking usage metrics
                    for key in usage_metrics:
                        usage_metrics[key] += backtrack_usage.get(key, 0)

                    if backtrack_success:
                        break  # Break from node execution to restart with updated DAG state

                    # If a node fails, we still want to complete other independent nodes so
                    # the synthesis has as much information as possible. If nodes depend on
                    # the failed node, they won't be executed, and we will eventually exit the loop.

            iteration += 1

        # Phase 3: Synthesize final answer
        final_answer, synthesis_usage = self._synthesize_final_answer(
            nodes, question)

        # Update usage metrics
        for key in usage_metrics:
            usage_metrics[key] += synthesis_usage.get(key, 0)

        Logger().info(
            f"Final DAG answer for question '{question}': {final_answer}")

        # Create notebook with results
        notebook = NoteBook()

        # Flatten sources dictionary
        flattened_sources = []
        for folder_id, folder_sources in sources.items():
            flattened_sources.extend([
                RetrievedResult(
                    doc_id=res['doc_id'],
                    content=res['content'],
                    folder_id=f"{folder_id}-{res['folder_id']}"
                )
                for res in folder_sources
            ])

        notebook.update_sources(flattened_sources)
        notebook.update_notes(final_answer)
        notebook.update_usage_metrics(usage_metrics)
        notebook.update_messages(messages)

        return notebook


class SingleProcessDAGAgent(BaseDAGAgent, SingleProcessAgent, ABC):
    """
    A single-process DAG agent.
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args):
        SingleProcessAgent.__init__(self, args)
        BaseDAGAgent.__init__(self, search_function, args)


class DAGAgent(BaseDAGAgent, MultiprocessingSearchAgent, ABC):
    """
    A multiprocessing DAG agent.
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores=20):
        MultiprocessingSearchAgent.__init__(self, args, cores=cores)
        BaseDAGAgent.__init__(self, search_function, args)


class StatefulDAGAgent(BaseDAGAgent, MultiprocessingStatefulSearchAgent, ABC):
    """
    A stateful DAG agent that maintains state across multiple reasoning sessions.
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores):
        MultiprocessingStatefulSearchAgent.__init__(self, args, cores)
        BaseDAGAgent.__init__(self, search_function, args)
