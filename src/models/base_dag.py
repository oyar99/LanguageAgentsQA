"""
Base DAG Agent class that consolidates common DAG logic between different DAG implementations.
This abstract class provides common functionality for parsing, validating, and creating DAG plans.
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Callable, Dict, List, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from models.action import Action
from models.agent import SelfContainedAgent, NoteBook
from models.react_agent import BaseIntelligentAgent
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


class BaseDAGNode(ABC):
    """
    Base class for DAG nodes that defines the common interface.
    """

    def __init__(self, node_id: str, sub_question: str, dependencies: List[str] = None):
        """
        Initialize a base DAG node.

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
        self.alternatives_exhausted = False
        self.sources = []
        self.unique_sources = []
        # Use context instead of explanation for consistency with original DAG agent
        self.context = ""  # Store reasoning context from the agent

    @abstractmethod
    def to_dict(self, include_sources: bool = False, include_context: bool = True) -> Dict:
        """
        Serialize the DAG node to a dictionary.

        Args:
            include_sources (bool): Whether to include sources field
            include_context (bool): Whether to include context field

        Returns:
            Dict: Serialized node data
        """


class BaseDAGAgent(SelfContainedAgent, ABC):
    """
    Abstract base class for DAG-based agents that provides common DAG functionality.

    This class consolidates the common logic for:
    - Parsing DAG plans from model responses
    - Validating DAG structure (no cycles, valid dependencies)
    - Creating DAG plans using OpenAI models
    - Managing DAG job arguments and prompts
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args):
        """
        Initialize the base DAG agent.

        Args:
            args: Agent arguments
        """
        SelfContainedAgent.__init__(self, args)

        agent_args = args.agent_args or {}
        # Model to use for generating the DAG plan
        self._think_model = agent_args.get('think_model', args.model)

        actions = {
            "search": Action(
                "Search for relevant documents for the given query using a semantic retriever. \
The function accepts a single string argument which is the search query. \
You will obtain more relevant results by formulating queries scoped to specific entities or \
keywords related to the question.",
                search_function
            )
        }

        self._subquestion_agent = ReactAgent(actions, PROMPT_EXAMPLES_TOOLS, args)

    def index(self, _):
        """
        Index the dataset for the DAG agent.
        """
        # TODO: Workaround to ensure corpus is set for the ReactAgent engine
        # Actual fix should be to re-design how react_agent returns results
        # pylint: disable-next=protected-access
        self._subquestion_agent._corpus = self._corpus


    def _parse_dag_plan(self, response_content: str, node_class) -> Dict[str, Any]:
        """
        Parse the DAG plan from the model response.

        Args:
            response_content (str): The response from the model containing the DAG plan
            node_class: The class to use for creating nodes (DAGNode or DAGNodeV2)

        Returns:
            Dict[str, Any]: Dictionary mapping node IDs to node objects
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

            nodes[node_id] = node_class(node_id, sub_question, dependencies)

        # Validate DAG structure (no cycles, valid dependencies)
        if not self._validate_dag(nodes):
            raise ValueError("Invalid DAG structure detected")

        return nodes

    def _validate_dag(self, nodes: Dict[str, Any]) -> bool:
        """
        Validate that the node structure forms a valid DAG (no cycles).

        Args:
            nodes (Dict[str, Any]): The nodes to validate

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

    def _create_dag_plan(
        self,
        question: str,
        dag_prompt: str,
        custom_id: str = "dag_planning"
    ) -> Tuple[str, Dict[str, int]]:
        """
        Create a DAG plan using the specified model and prompt.

        Args:
            question (str): The question to decompose into a DAG
            dag_prompt (str): The prompt to use for DAG planning
            custom_id (str): Custom ID for the request

        Returns:
            Tuple[str, Dict[str, int]]: DAG plan response content and usage metrics
        """
        open_ai_request = {
            "custom_id": custom_id,
            "model": self._think_model,
            "messages": [
                {"role": "system", "content": dag_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            "temperature": DEFAULT_DAG_JOB_ARGS['temperature']
            if supports_temperature_param(self._think_model) else None,
            "frequency_penalty": DEFAULT_DAG_JOB_ARGS['frequency_penalty'],
            "presence_penalty": DEFAULT_DAG_JOB_ARGS['presence_penalty'],
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

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        response_content = result.choices[0].message.content.strip()

        Logger().debug(
            f"DAG planning response: {response_content} for question {question}")

        return response_content, usage_metrics

    def _find_backtrack_candidates(self, failed_node_id: str, nodes: Dict[str, BaseDAGNode]) -> List[str]:
        """
        Find all possible dependency nodes that can be backtracked to find alternative answers,
        sorted by distance (closest ancestors first).

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, DAGNode]): All nodes in the DAG

        Returns:
            List[str]: List of node IDs to backtrack to, sorted by distance
        """
        def get_ancestors(node_id: str, distance: int = 0) -> List[Tuple[str, int]]:
            """Get all ancestors of a node with their distances"""
            ancestors = []
            node = nodes[node_id]

            for dep_id in node.dependencies:
                dep_node = nodes[dep_id]
                # Only consider nodes that completed successfully and have sources and aren't exhausted
                if dep_node.is_completed and dep_node.sources and not dep_node.alternatives_exhausted:
                    ancestors.append((dep_id, distance + 1))

                # Recursively get ancestors of dependencies
                ancestors.extend(get_ancestors(dep_id, distance + 1))

            return ancestors

        # Get all ancestors with distances and sort by distance (ascending)
        all_ancestors = get_ancestors(failed_node_id)
        candidates = [ancestor_id for ancestor_id,
                      _ in sorted(all_ancestors, key=lambda x: x[1])]

        Logger().debug(
            f"Found backtrack candidates for {failed_node_id}: {candidates}")
        return candidates

    def _serialize_dag_state(
        self, 
        nodes: Dict[str, Any], 
        include_sources_for: List[str] = None,
        exclude_context: bool = False
    ) -> str:
        """
        Serialize the current DAG state for execution agents.

        Args:
            nodes (Dict[str, Any]): All nodes in the DAG
            include_sources_for (List[str]): List of node IDs to include sources for (backtracking scenarios)
            exclude_context (bool): Whether to exclude context/explanation from serialization

        Returns:
            str: JSON representation of the DAG state
        """
        include_sources_for = include_sources_for or []

        dag_state = []
        for node_id, node in nodes.items():
            include_sources = node_id in include_sources_for
            dag_state.append(node.to_dict(
                include_sources=include_sources, 
                include_context=not exclude_context
            ))

        return json.dumps(dag_state, indent=2)

    def _reset_dependent_nodes(self, failed_node_id: str, nodes: Dict[str, Any]) -> None:
        """
        Reset all nodes that depend on the failed node back to incomplete state.

        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, Any]): All nodes in the DAG
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
                dep_node.alternatives_exhausted = False
                dep_node.context = ""

                # Recursively reset nodes that depend on this one
                self._reset_dependent_nodes(dep_node_id, nodes)

    def _get_dependent_nodes(self, failed_node_id: str, nodes: Dict[str, Any]) -> List[str]:
        """
        Get all nodes that directly depend on the given failed node.
        
        Args:
            failed_node_id (str): The ID of the failed node
            nodes (Dict[str, Any]): All nodes in the DAG

        Returns:
            List[str]: List of node IDs that directly depend on the failed node
        """
        dependent_nodes = []
        for node_id, node in nodes.items():
            if failed_node_id in node.dependencies:
                dependent_nodes.append(node_id)
        return dependent_nodes


# Default job arguments for DAG planning
DEFAULT_DAG_JOB_ARGS = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

# Common DAG planning prompt used by both DAG agents
BASE_DAG_PROMPT = '''You are an expert question decomposition agent. Your task is to analyze a complex \
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
