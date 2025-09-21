"""
A DAG-based agent that decomposes questions into directed acyclic graphs.
Each node represents a sub-question that must be answered to reach the final answer.
"""

# pylint: disable=duplicate-code

from abc import ABC
import json
import os
from typing import Callable, Dict, List, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.action import Action
from models.agent import (
    MultiprocessingSearchAgent,
    MultiprocessingStatefulSearchAgent,
    NoteBook,
    SelfContainedAgent,
    SingleProcessAgent
)
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
class DAGNode:
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
        self.node_id = node_id
        self.sub_question = sub_question
        self.dependencies = dependencies or []
        self.is_completed = False
        self.is_failed = False
        self.result = None
        self.alternative_results = []  # Store alternative answers for backtracking
        # Track if we've exhausted all alternatives for this node
        self.alternatives_exhausted = False
        self.sources = []
        self.unique_sources = []
        self.formulated_question = None

    def to_dict(self, include_sources: bool = False) -> Dict:
        """
        Serialize the DAG node to a dictionary for the DAG execution agent.

        Args:
            include_sources (bool): Whether to include sources field (only needed for backtracking)

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

        # Add sources only when requested (for backtracking scenarios)
        if include_sources and self.unique_sources:
            node_dict["sources"] = [
                src["content"] for src in self.unique_sources
            ]

        return node_dict


class BaseDAGAgent(SelfContainedAgent, ABC):
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
        SelfContainedAgent.__init__(self, args)
        self._search_function = search_function
        self._max_iterations = 8

        actions = {
            "search": Action(
                "Search for relevant documents for the given query using a semantic retriever. \
The function accepts a single string argument which is the search query. \
You will obtain more relevant results by formulating queries scoped to specific entities or \
keywords related to the question.",
                search_function
            )
        }

        self._react_agent = ReactAgent(actions, PROMPT_EXAMPLES_TOOLS, args)

        Logger().debug(f"DAG Agent prompt: {DAG_AGENT_PROMPT}")

    def index(self, _):
        """
        Index the dataset for the DAG agent.
        """
        # TODO: Workaround to ensure corpus is set for the ReactAgent engine
        # Actual fix should be to re-design how react_agent returns results
        # pylint: disable-next=protected-access
        self._react_agent._corpus = self._corpus

    def _parse_dag_plan(self, response_content: str) -> Dict[str, DAGNode]:
        """
        Parse the DAG plan from the model response.

        Args:
            response_content (str): The response from the model containing the DAG plan

        Returns:
            Dict[str, DAGNode]: Dictionary mapping node IDs to DAGNode objects
        """
        structured_response = parse_structured_response(response_content)

        dag_data = structured_response['dag_plan']
        nodes = {}

        for node_data in dag_data:
            node_id = node_data.get('node_id')
            sub_question = node_data.get('sub_question')
            dependencies = node_data.get('dependencies', [])

            if not node_id or not sub_question:
                raise ValueError(f"Invalid node data: {node_data}")

            nodes[node_id] = DAGNode(node_id, sub_question, dependencies)

        # Validate DAG structure (no cycles, valid dependencies)
        if not self._validate_dag(nodes):
            raise ValueError("Invalid DAG structure detected")

        return nodes

    def _validate_dag(self, nodes: Dict[str, DAGNode]) -> bool:
        """
        Validate that the node structure forms a valid DAG (no cycles).

        Args:
            nodes (Dict[str, DAGNode]): The nodes to validate

        Returns:
            bool: True if valid DAG, False otherwise
        """
        # Check for self-references and invalid dependencies
        for node_id, node in nodes.items():
            if node_id in node.dependencies:
                Logger().error(f"Node {node_id} has self-dependency")
                return False

            for dep in node.dependencies:
                if dep not in nodes:
                    Logger().error(
                        f"Node {node_id} depends on non-existent node {dep}")
                    return False

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            rec_stack.add(node_id)

            for dep in nodes[node_id].dependencies:
                if has_cycle(dep):
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    Logger().error("Cycle detected in DAG")
                    return False

        return True

    def _serialize_dag_state(self, nodes: Dict[str, DAGNode], include_sources_for: List[str] = None) -> str:
        """
        Serialize the current DAG state for the execution agent.

        Args:
            nodes (Dict[str, DAGNode]): All nodes in the DAG
            include_sources_for (List[str]): List of node IDs to include sources for (backtracking scenarios)

        Returns:
            str: JSON representation of the DAG state
        """
        include_sources_for = include_sources_for or []

        dag_state = []
        for node_id, node in nodes.items():
            include_sources = node_id in include_sources_for
            dag_state.append(node.to_dict(include_sources=include_sources))

        return json.dumps(dag_state, indent=2)

    def _query_dag_agent(
        self,
        user_query: str,
        custom_id: str,
        dag_state: str = None,
        available_tools: List[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Unified method to query the DAG execution agent with structured JSON response.

        Args:
            user_query (str): The tool call query to send to the DAG agent (e.g., "REQUERY(node_id='node_1')")
            custom_id (str): Custom ID for the request
            dag_state (str): Optional DAG state to inject into the system prompt
            available_tools (List[str]): Optional list of tools to make available \
(e.g., ["REQUERY", "ALTERNATIVE_ANSWER"])

        Returns:
            Tuple[str, Dict[str, int]]: Answer and usage metrics
        """
        # Build the system prompt with tool filtering and DAG state if provided
        system_prompt = self._build_system_prompt_with_tools(available_tools)
        if dag_state:
            system_prompt = f"{system_prompt}\n\n## Current DAG State\n\n{dag_state}"

        request = {
            "custom_id": custom_id,
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": default_job_args['temperature']
            if supports_temperature_param(self._args.model) else None,
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "dag_agent_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "thought": {"type": "string"},
                            "answer": {"type": "string"}
                        },
                        "required": ["thought", "answer"],
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

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        Logger().debug(
            f"DAG agent query result - Thought: {structured_response['thought']}, Answer: {answer}")

        return answer, usage_metrics

    def _build_system_prompt_with_tools(self, available_tools: List[str] = None) -> str:
        """
        Build the system prompt with only the specified tools available.

        Args:
            available_tools (List[str]): List of tool names to include (e.g., ["REQUERY", "ALTERNATIVE_ANSWER"])

        Returns:
            str: The filtered system prompt
        """

        # Tool definitions mapping
        tool_definitions = {
            "REQUERY": """- REQUERY(node_id: str): Reformulate the sub-question for the given node ID \
based on results from its dependencies. The reformulated question should be clear and self-contained, \
not referencing node IDs. The returned question must be a string.""",

            "ALTERNATIVE_ANSWER": """- ALTERNATIVE_ANSWER(node_id: str, exclude: List[str]): Provide an \
alternative answer for the given node ID based on the sources provided. This answer must be different from \
all other answers in the exclude list. However, it should be flexible in interpretation, and consider \
plausible assumptions or entities in the sources. Only when all alternatives are definitely exhausted \
should you return 'N/A'. The returned answer must be a string directly extracted or inferred from the \
sources. Remember to exhaust all possible alternatives before returning 'N/A' even if they assume typos, \
other interpretations, or entities that may not seem related at first glance.""",

            "FINAL_ANSWER": """- FINAL_ANSWER(original_question: str): Answer the original question based on \
the state of the DAG. Incorporate the available information from each node to formulate a concise, accurate, \
and complete answer. Provide an EXACT answer using only words found in the node results and sources when \
possible. The response should be as brief as possible. Do not repeat the question in your answer. If the \
information seems insufficient, provide the best possible answer based on the available information. \
If definitely no answer can be found, respond with 'N/A'."""
        }

        # Filter tools based on available_tools list
        filtered_tools = []
        for tool_name in available_tools:
            if tool_name in tool_definitions:
                filtered_tools.append(tool_definitions[tool_name])

        # Build the prompt with only the specified tools
        base_prompt = '''You are an intelligent DAG execution agent. You help with complex \
reasoning questions related to a DAG (Directed Acyclic Graph) of sub-questions. Particularly, you assist \
with the following requests.

## Internal Tools

'''

        tools_section = "\n\n".join(filtered_tools)

        # Build examples section based on available tools
        examples = {}
        examples["REQUERY"] = '''### Example 1

DAG Plan:
[
    {
        "node_id": "node_1",
        "sub_question": "What stadium is owned by Canyon Independent School District?",
        "dependencies": [],
        "result": "Canyon Independent School District owns Kimbrough Memorial Stadium."
    },
    {
        "node_id": "node_2",
        "sub_question": "Where is the stadium identified in node 1 located?",
        "dependencies": ["node_1"]
    }
]

REQUERY(node_id="node_2")

Output:

{
    "thought": "Node_2 depends on node_1, which found that Canyon Independent School District owns \
Kimbrough Memorial Stadium. I need to replace the reference 'stadium identified in node 1' with the \
actual stadium name.",
    "answer": "Where is Kimbrough Memorial Stadium located?"
}'''

        examples["ALTERNATIVE_ANSWER"] = '''### Example 2

[
    {
        "node_id": "node_1",
        "sub_question": "What stadium is owned by Canyon Independent School District?",
        "dependencies": [],
        "result": "Canyon Independent School District owns Kimbrough Memorial Stadium.",
        "sources": [
            "Canyon Independent School District owns Kimbrough Memorial Stadium",
            "Liberty Stadium, owned by Kanyon ISD, has a capacity of 16,000"
        ]
    },
    {
        "node_id": "node_2",
        "sub_question": "Where is the stadium identified in node 1 located?",
        "dependencies": ["node_1"]
    }
]

ALTERNATIVE_ANSWER(node_id="node_1", exclude=["Lubbock, Texas"])

Output:

{
    "thought": "Analyzing the sources, I see that it mentions 'Kanyon ISD' which could be a typo \
or alternative name for 'Canyon ISD'. This suggests that 'Liberty Stadium' might also be owned by \
the same entity. Therefore, I can consider 'Liberty Stadium' as an alternative answer.",
    "answer": "Liberty Stadium"
}'''

        # Filter examples based on available tools
        filtered_examples = []
        for tool_name in available_tools:
            if tool_name in examples:
                filtered_examples.append(examples[tool_name])

        examples_section = ""
        if filtered_examples:
            examples_section = "\n\n## Examples\n\n" + \
                "\n\n".join(filtered_examples)

        return f"{base_prompt}{tools_section}{examples_section}"

    def _get_dependent_nodes(self, failed_node_id: str, nodes: Dict[str, DAGNode]) -> List[str]:
        """
        Get all nodes that depend on the given failed node.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            List[str]: List of node IDs that depend on the failed node
        """
        dependent_nodes = []
        for node_id, node in nodes.items():
            if failed_node_id in node.dependencies:
                dependent_nodes.append(node_id)
        return dependent_nodes

    def _reset_dependent_nodes(self, failed_node_id: str, nodes: Dict[str, DAGNode]) -> None:
        """
        Reset all nodes that depend on the failed node back to incomplete state.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG
        """
        dependent_nodes = self._get_dependent_nodes(failed_node_id, nodes)

        for dep_node_id in dependent_nodes:
            dep_node = nodes[dep_node_id]
            if dep_node.is_completed or dep_node.is_failed:
                Logger().debug(
                    f"Resetting node {dep_node_id} due to failed dependency {failed_node_id}")
                dep_node.is_completed = False
                dep_node.is_failed = False
                dep_node.result = None
                dep_node.alternatives_exhausted = False  # Reset alternatives exhausted flag
                # Recursively reset nodes that depend on this one
                self._reset_dependent_nodes(dep_node_id, nodes)

    def _find_backtrack_candidate(self, failed_node_id: str, nodes: Dict[str, DAGNode]) -> str:
        """
        Find a dependency node that can be backtracked to find alternative answers.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            str: Node ID to backtrack to, or None if no backtrack candidate found
        """
        failed_node = nodes[failed_node_id]

        # Look through dependencies to find one that might have alternative answers
        for dep_id in failed_node.dependencies:
            dep_node = nodes[dep_id]
            # If the dependency node completed successfully and hasn't exhausted alternatives, it might have more
            if dep_node.is_completed and dep_node.sources and not dep_node.alternatives_exhausted:
                Logger().debug(
                    f"Found backtrack candidate: {dep_id} for failed node {failed_node_id}")
                return dep_id

        # If no direct dependencies can be backtracked, check if any dependency has its own dependencies
        for dep_id in failed_node.dependencies:
            if nodes[dep_id].dependencies:
                candidate = self._find_backtrack_candidate(dep_id, nodes)
                if candidate:
                    return candidate

        return None

    def _extract_alternative_answer(self, node: DAGNode, nodes: Dict[str, DAGNode]) -> Tuple[str, Dict[str, int]]:
        """
        Try to extract an alternative answer from a node's sources using the unified DAG execution agent.

        Args:
            node (DAGNode): The node to extract alternative answer from

        Returns:
            Tuple[str, Dict[str, int]]: Alternative answer and usage metrics
        """
        dag_state = self._serialize_dag_state(
            nodes, include_sources_for=[node.node_id])

        # Format the exclude list for the ALTERNATIVE_ANSWER tool
        exclude_list = str(node.alternative_results)

        # Use the new ALTERNATIVE_ANSWER tool format
        user_query = f'ALTERNATIVE_ANSWER(node_id="{node.node_id}", exclude={exclude_list})'

        Logger().debug(
            f"Extracting alternative answer with query: {user_query}")

        alternative_answer, usage_metrics = self._query_dag_agent(
            user_query, f"alternative_{node.node_id}", dag_state, available_tools=["ALTERNATIVE_ANSWER"])

        Logger().debug(
            f"Found alternative answer for node {node.node_id}: {alternative_answer}")
        return alternative_answer, usage_metrics

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

    def _formulate_question(self, node: DAGNode, nodes: Dict[str, DAGNode]) -> Tuple[Dict[str, int], str]:
        """
        Formulate the question for the current node using the unified DAG execution agent.

        Args:
            node (DAGNode): The current node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            Tuple[Dict[str, int], str]: Usage metrics and the formulated question
        """
        # If question does not have deps, then we just return it as is
        if not node.dependencies:
            return {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            }, node.sub_question

        # Serialize current DAG state
        dag_state = self._serialize_dag_state(nodes)

        # Use the new REQUERY tool format
        user_query = f'REQUERY(node_id="{node.node_id}")'

        reformulated_query, usage_metrics = self._query_dag_agent(
            user_query, f"dag_question_formulation_{node.node_id}", dag_state, available_tools=["REQUERY"])

        Logger().debug(
            f"Node {node.node_id} reformulated query: {reformulated_query}")

        return usage_metrics, reformulated_query

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
        formulate_metrics, question = self._formulate_question(node, nodes)

        # Store the formulated question for potential backtracking
        node.formulated_question = question

        notebook = self._react_agent.reason(question)

        answer = notebook.get_notes()
        sources = notebook.get_sources()
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
            node.alternative_results.append(answer)

        return total_usage_metrics, sources

    def _attempt_backtrack(self, failed_node_id: str, nodes: Dict[str, DAGNode]) -> Tuple[bool, Dict[str, int]]:
        """
        Attempt to backtrack when a node fails by finding alternative answers in dependencies.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            Tuple[bool, Dict[str, int]]: Success flag and usage metrics from backtracking
        """
        backtrack_candidate = self._find_backtrack_candidate(
            failed_node_id, nodes)

        if not backtrack_candidate:
            Logger().debug(
                f"No backtrack candidate found for failed node {failed_node_id}")
            return False, {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        backtrack_node = nodes[backtrack_candidate]

        Logger().debug(
            f"Attempting backtrack of node {backtrack_candidate} for failed node {failed_node_id}")

        alternative_answer, alt_usage = self._extract_alternative_answer(
            backtrack_node, nodes)

        # Check if this is truly a new alternative (not N/A, not already tried)
        if (alternative_answer != "N/A" and
                alternative_answer not in backtrack_node.alternative_results):

            # Update the node with alternative answer
            backtrack_node.result = alternative_answer
            # Add the new alternative to the list
            backtrack_node.alternative_results.append(alternative_answer)

            # Reset all dependent nodes (including the failed one)
            self._reset_dependent_nodes(backtrack_candidate, nodes)

            Logger().debug(
                f"Successfully backtracked with alternative: {alternative_answer}")
            return True, alt_usage

        # No new alternative found - mark this node as having exhausted alternatives
        backtrack_node.alternatives_exhausted = True
        # If no alternatives left, revert to original result as it is more likely correct
        backtrack_node.result = backtrack_node.alternative_results[
            0] if backtrack_node.alternative_results else None
        Logger().debug(f"Backtracking failed for node {failed_node_id}")
        return False, alt_usage

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
        # Check if we have any completed nodes
        has_completed_nodes = any(
            node.is_completed and node.result for node in nodes.values())

        if not has_completed_nodes:
            return "N/A", {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Create DAG state with sources only for failed nodes
        failed_node_ids = [node_id for node_id, node in nodes.items()
                           if node.is_failed and node.unique_sources]
        dag_state = self._serialize_dag_state(
            nodes, include_sources_for=failed_node_ids)

        # Use the new FINAL_ANSWER tool format
        user_query = f'FINAL_ANSWER(original_question="{original_question}")'

        Logger().debug(
            f"Synthesizing final answer with query: {user_query}")

        final_answer, usage_metrics = self._query_dag_agent(
            user_query, "dag_synthesis", dag_state, available_tools=["FINAL_ANSWER"])

        return final_answer, usage_metrics

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
        open_ai_request = {
            "custom_id": "dag_planning",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": DAG_AGENT_PROMPT},
                {"role": "user", "content": f"Question: {question}"}
            ],
            "temperature": default_job_args['temperature']
            if supports_temperature_param(self._args.model) else None,
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 1000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "dag_planner",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "dag_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "node_id": {"type": "string"},
                                        "sub_question": {"type": "string"},
                                        "dependencies": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["node_id", "sub_question", "dependencies"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["reasoning", "dag_plan"],
                        "additionalProperties": False
                    }
                }
            }
        }

        result = chat_completions([open_ai_request])[0][0]

        # Update usage metrics
        usage_metrics["completion_tokens"] += result.usage.completion_tokens
        usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
        usage_metrics["total_tokens"] += result.usage.total_tokens

        response_content = result.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": response_content})

        Logger().debug(
            f"DAG planning response: {response_content} for question {question}")

        # Parse the DAG plan
        nodes = self._parse_dag_plan(response_content)

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

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores=16):
        MultiprocessingSearchAgent.__init__(self, args, cores)
        BaseDAGAgent.__init__(self, search_function, args)


class StatefulDAGAgent(BaseDAGAgent, MultiprocessingStatefulSearchAgent, ABC):
    """
    A stateful DAG agent that maintains state across multiple reasoning sessions.
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores):
        MultiprocessingStatefulSearchAgent.__init__(self, args, cores)
        BaseDAGAgent.__init__(self, search_function, args)


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

DAG_AGENT_PROMPT = '''You are an expert question decomposition agent. Your task is to analyze a complex \
question and break it down into a directed acyclic graph (DAG) of sub-questions. Each sub-question should \
build towards answering the main question, and should have clear dependencies on other sub-questions when needed. \
The sub-questions must be as specific and focused as possible, hence they should contain only ONE question. Do \
not combine or compound multiple questions in a single sub-question. Each sub-question must be identified \
by a unique node ID that follows the pattern "node_N" where N is a number (e.g., node_1, node_2, etc.).
Dependencies should reference the node IDs of other sub-questions that must be answered first.

If the question can be answered directly without decomposition, create a single node with no dependencies.

## EXAMPLES

Question: "Were Scott Derrickson and Ed Wood of the same nationality?"

Response:
{
    "reasoning": "To answer if Scott Derrickson and Ed Wood are of the same nationality, I need to find each person's nationality separately and then compare them.",
    "dag_plan": [
        {
            "node_id": "node_1",
            "sub_question": "What is Scott Derrickson's nationality?",
            "dependencies": []
        },
        {
            "node_id": "node_2",
            "sub_question": "What is Ed Wood's nationality?",
            "dependencies": []
        }
    ]
}

Question: "In which county is the stadium owned by Canyon Independent School District located?"

Response:
{
    "reasoning": "I need to first find what stadium is owned by Canyon Independent School District, then find where that stadium is located, \
and finally determine which county that location is in.",
    "dag_plan": [
        {
            "node_id": "node_1",
            "sub_question": "What stadium is owned by Canyon Independent School District?",
            "dependencies": []
        },
        {
            "node_id": "node_2",
            "sub_question": "Where is the stadium identified in node 1 located?",
            "dependencies": ["node_1"]
        },
        {
            "node_id": "node_3",
            "sub_question": "Which county is the location identified in node 2 in?",
            "dependencies": ["node_2"]
        }
    ]
}
'''

PROMPT_EXAMPLES_TOOLS = '''### Example

Question: Which United States Vice President was in office during the time Alfred Balk served as secretary of the Committee?

Iteration 1:
```json
{
    "thought": "I need to find information about United States Vice Presidents and information about Alfred Balk during the time \
he served as secretary.",
    "actions": ["search('United States Vice Presidents')", "search('Alfred Balk's biography')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find information about United States Vice Presidents and information about Alfred Balk during the time \
he served as secretary.",
    "actions": ["search('United States Vice Presidents')", "search('Alfred Balk's biography')"]
    "observations": [["Nelson Aldrich Rockefeller was an American businessman and politician who served as the 41st Vice President of the United States \
from 1974 to 1977"], ["Alfred Balk was an American reporter. He served as the secretary of Nelson Rockefeller's Committee on the Employment of Minority \
Groups in the News Media."]]
}
```

Iteration 3:
```json
{
    "thought": "I found that Nelson Rockefeller was Vice President from 1974 to 1977 and Alfred Balk served as secretary of the Committee on the Employment of Minority Groups \
in the News Media under Vice President Nelson Rockefeller. Therefore, the answer is Nelson Rockefeller.",
    "final_answer": "Nelson Rockefeller"
}
```

You must keep trying to find an answer until instructed otherwise. At which point, you must provide the final answer. \
If you cannot find an answer at that time, try to provide the best reasonable answer based on the retrieved documents even if incomplete. Otherwise respond \
with "N/A" as the final answer.
'''
