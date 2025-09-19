"""
A DAG-based agent that decomposes questions into directed acyclic graphs.
Each node represents a sub-question that must be answered to reach the final answer.
"""

# pylint: disable=duplicate-code

from abc import ABC
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

# pylint: disable-next=too-few-public-methods
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
        self.sources: List[RetrievedResult] = []


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

    def _get_executable_nodes(self, nodes: Dict[str, DAGNode]) -> List[str]:
        """
        Get nodes that can be executed (all dependencies completed).

        Args:
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            List[str]: List of node IDs that can be executed
        """
        executable = []

        for node_id, node in nodes.items():
            if node.is_completed:
                continue

            # Check if all dependencies are completed
            deps_completed = all(
                nodes[dep].is_completed for dep in node.dependencies
            )

            if deps_completed:
                executable.append(node_id)

        return executable

    def _parse_completed_deps(self, node: DAGNode, nodes: Dict[str, DAGNode]) -> str:
        """
        Parse and format the results of completed dependencies for context.

        Args:
            node (DAGNode): The current node
            nodes (Dict[str, DAGNode]): All nodes in the DAG
        Returns:
            str: Formatted context string from completed dependencies
        """
        dependency_context = ""
        if node.dependencies:
            dependency_results = []
            for dep_id in node.dependencies:
                dep_node = nodes[dep_id]
                if dep_node.is_completed and dep_node.result:
                    dependency_results.append(
                        f"- ({dep_node.node_id}) {dep_node.sub_question}: {dep_node.result}")

            if dependency_results:
                dependency_context = "Context:\n" + \
                    "\n".join(dependency_results) + "\n\n"
        return dependency_context

    def _formulate_question(self, node: DAGNode, nodes: Dict[str, DAGNode]) -> Tuple[Dict[str, int], str]:
        """
        Formulate the question for the current node, incorporating results from dependencies.

        Args:
            node (DAGNode): The current node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            str: The formulated question
        """
        # If question does not have deps, then we just return it as is
        if not node.dependencies:
            return {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            }, node.sub_question

        user_content = (
            f"{self._parse_completed_deps(node, nodes)}Question: {node.sub_question}"
        )

        formulate_request = {
            "custom_id": f"dag_question_formulation_{node.node_id}",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": FORMULATE_QUESTION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": default_job_args['temperature']
            if supports_temperature_param(self._args.model) else None,
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
        }

        result = chat_completions([formulate_request])[0][0]

        reformulated_query = result.choices[0].message.content.strip()

        Logger().debug(
            f"Node {node.node_id} reformulated query: {reformulated_query}")

        return {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }, reformulated_query

    # pylint: disable-next=too-many-locals

    def _execute_node(
        self,
        node: DAGNode,
        nodes: Dict[str, DAGNode]
    ) -> Tuple[Dict[str, int], List[RetrievedResult]]:
        """
        Execute a single node by searching for information to answer its sub-question.
        Uses a ReAct loop to reformulate queries if initial attempts don't find answers.

        Args:
            node (DAGNode): The node to execute
            nodes (Dict[str, DAGNode]): All nodes for context

        Returns:
            Tuple[Dict[str, int], List[str]]: Usage metrics and sources
        """
        formulate_metrics, question = self._formulate_question(node, nodes)

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

        if answer == 'N/A':
            Logger().warn(f"Node {node.node_id} could not find an answer.")
            node.is_failed = True
        else:
            node.result = answer
            node.is_completed = True

        node.sources = sources

        return total_usage_metrics, sources

    def _synthesize_final_answer(
        self,
        nodes: Dict[str, DAGNode],
        original_question: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Synthesize the final answer from all completed nodes.

        Args:
            nodes (Dict[str, DAGNode]): All completed nodes
            original_question (str): The original question

        Returns:
            Tuple[str, Dict[str, int]]: Final answer and usage metrics
        """
        # Collect all sub-question results
        sub_results = []
        for node in nodes.values():
            if node.is_completed and node.result:
                sub_results.append(f"- {node.sub_question}: {node.result}")

        user_content = f"Question: {original_question}\n\nSub-question Results:\n{chr(10).join(sub_results)}"

        open_ai_request = {
            "custom_id": "dag_synthesis",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": DAG_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": default_job_args['temperature']
            if supports_temperature_param(self._args.model) else None,
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
        }

        result = chat_completions([open_ai_request])[0][0]

        # Extract the final answer directly as a string
        final_answer = result.choices[0].message.content.strip()

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

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

        Logger().debug(f"DAG planning response: {response_content}")

        # Parse the DAG plan
        nodes = self._parse_dag_plan(response_content)

        # Phase 2: Execute DAG nodes
        iteration = 0
        stop = False
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

                # If a node fails, we stop further execution
                # We hope for the best once the answer is synthesized with all available info
                if node.is_failed:
                    Logger().warn(
                        f"Node {node.node_id} failed to produce an answer. Abandoning further execution."
                    )
                    stop = True
                    break

            if stop:
                break

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

FORMULATE_QUESTION_PROMPT = '''Given a list of sub-questions and their answers, formulate a new sub-question that \
replaces references to previous sub-questions with their answers. The purpose is to simplify the question by using known information.

The new sub-question should be clear and self-contained. It should not reference node IDs.

## Example

Context:
- (node_1) What stadium is owned by Canyon Independent School District?: Canyon Independent School District owns Kimbrough Memorial Stadium.

Question:
- Where is the stadium identified in node 1 located?

Response:

"Where is Kimbrough Memorial Stadium located?"
'''

DAG_SYNTHESIS_PROMPT = '''You are an intelligent QA agent. Based on the sub-question results provided by the \
user, provide an EXACT answer to the original question, using only words found in the sub-question results \
when possible. Under no circumstances should you include any additional commentary, explanations, reasoning, \
or notes in your final response. If the answer can be a single word (e.g., Yes, No, a date, or an object), \
please provide just that word. If the answer is not available based on the sub-question results, respond with "N/A".
'''

PROMPT_EXAMPLES_TOOLS = '''### Example

Question: "Where is Kimbrough Memorial Stadium?"

Iteration 1:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium')"]
}
```

Iteration 2:
```json
{
    "thought": "I need to find where Kimbrough Memorial Stadium is located.",
    "actions": ["search('Kimbrough Memorial Stadium')"]
    "observations": [["Kimbrough Memorial Stadium is located in Canyon, Texas"]]
}
```

Iteration 3:
```json
{
    "thought": "The retrieved information indicates that Kimbrough Memorial Stadium is located in Canyon, Texas.",
    "final_answer": "Canyon, Texas"
}
```

You must keep trying to find an answer until instructed otherwise. At which point, you must provide the final answer. \
If you cannot find an answer at that time, respond with "N/A" as the final answer.
'''
