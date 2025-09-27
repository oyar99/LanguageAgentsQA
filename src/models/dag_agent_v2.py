"""
A DAG-based agent v2 that decomposes questions into directed acyclic graphs and delegates execution to a React Agent.
This version creates a DAG plan and passes it to a React Agent with a answer tool for autonomous execution.
"""

# pylint: disable=duplicate-code, too-many-lines

from abc import ABC
import os
from typing import Callable, Dict, List, Tuple

from logger.logger import Logger
from models.action import Action
from models.agent import (
    MultiprocessingSearchAgent,
    MultiprocessingStatefulSearchAgent,
    NoteBook,
    SingleProcessAgent
)
from models.base_dag import BaseDAGAgent as BaseDag, BaseDAGNode, BASE_DAG_PROMPT
from models.react_agent import BaseIntelligentAgent
from models.retrieved_result import RetrievedResult
from models.question_answer import QuestionAnswer


class BaseDAGExecutionReactAgent(BaseIntelligentAgent, ABC):
    """
    A specialized React agent for executing DAG plans autonomously.
    This agent receives a DAG plan and uses the answer tool to execute it.
    """

    def __init__(self, actions: Dict[str, Action], examples: str, args):
        """
        Initialize the DAG execution React agent with a modified prompt.

        Args:
            actions (Dict[str, Action]): Actions available to the agent (should include answer)
            examples (str): Example interactions
            args: Agent arguments
        """
        BaseIntelligentAgent.__init__(
            self, actions, examples, args, DAG_EXECUTION_REACT_PROMPT)

    def index(self, _) -> None:
        """Not implemented for execution agent."""
        raise NotImplementedError(
            "Indexing is not implemented for DAG execution agent.")

    def batch_reason(self, _: List[QuestionAnswer]) -> List[NoteBook]:  # type: ignore
        """Not implemented for execution agent."""
        raise NotImplementedError(
            "Batch reasoning is not implemented for DAG execution agent.")


class SubQuestionReactAgent(BaseIntelligentAgent, ABC):
    """
    A specialized React agent for solving sub-questions with DAG context.
    This agent is designed to solve individual sub-questions as part of a larger DAG plan.
    """

    def __init__(self, actions: Dict[str, Action], extra_prompt: str, args):
        """
        Initialize the SubQuestionReactAgent.

        Args:
            actions (Dict[str, Action]): A dictionary of actions the agent can perform.
            extra_prompt (str): Additional instructions for the agent.
            args: Agent arguments.
        """
        BaseIntelligentAgent.__init__(self, actions, extra_prompt, args)
        self._max_iterations = 4
        self.corpus = None

    def index(self, _) -> None:
        """
        Index the dataset (not implemented for SubQuestionReactAgent helper).

        Raises:
            NotImplementedError: Indexing is not implemented for SubQuestionReactAgent Helper.
        """
        raise NotImplementedError(
            "Indexing is not implemented for SubQuestionReactAgent Helper.")

    def batch_reason(self, _: List[QuestionAnswer]) -> List[NoteBook]:  # type: ignore
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for SubQuestionReactAgent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for SubQuestionReactAgent Helper.")


# pylint: disable-next=too-few-public-methods, too-many-instance-attributes
class DAGNodeV2(BaseDAGNode):
    """
    Represents a node in the DAG structure for v2 implementation.
    Each node contains a sub-question and tracks its dependencies and completion status.
    """

    def to_dict(self, include_sources: bool = False, include_context: bool = True) -> Dict:
        """
        Serialize the DAG node to a dictionary for the React Agent.

        Args:
            include_sources (bool): Whether to include sources field
            include_context (bool): Whether to include context field

        Returns:
            Dict: Serialized node data
        """
        node_dict = {
            "node_id": self.node_id,
            "sub_question": self.sub_question,
            "dependencies": self.dependencies
        }

        # Add result and context if node is completed
        if self.is_completed and self.result:
            node_dict["result"] = self.result
            if include_context and self.context:
                node_dict["explanation"] = self.context

        # Add sources only when requested (for backtracking scenarios)
        if include_sources and self.unique_sources:
            node_dict["sources"] = [
                src["content"] for src in self.unique_sources
            ]

        return node_dict


class BaseDAGAgentV2(BaseDag, ABC):
    """
    A DAG-based agent v2 that decomposes questions into a directed acyclic graph structure
    and delegates execution to a React Agent with a answer tool.

    The agent operates in two phases:
    1. Planning Phase: Decompose the main question into a DAG of sub-questions
    2. Execution Phase: Pass the DAG plan to a React Agent that autonomously executes it
    """

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args):
        """
        Initialize the DAG agent v2.

        Args:
            search_function: Function that takes a query and returns (documents, sources, metrics)
            args: Agent arguments
        """
        super().__init__(search_function, args)
        self._max_iterations = 8

        actions = {
            "answer": Action(
                "Answers a sub-question. This tool takes a query string "
                "that should explicitly contain all entities and keywords needed to "
                "answer the query and must **NOT** contain any node IDs or references to other nodes. "
                "The query must be self-contained and cannot reference other nodes or dependencies. "
                "The second argument node_id identifies which node in the DAG plan this sub-question "
                "corresponds to. Returns the updated DAG state after solving the sub-question, or "
                "provides feedback if the query cannot be answered.",
                self._answer
            ),
            "update_node": Action(
                "Update a DAG node with an alternative answer and reset dependent nodes. This tool "
                "takes an answer and node strings. Use this when you need to provide an "
                "alternative answer for a node based on backtracking suggestions. Returns the updated "
                "DAG state after the node update and dependent node reset.",
                self._update_node
            )
        }

        # Create the main execution React Agent with modified prompt
        self._execution_agent = BaseDAGExecutionReactAgent(
            actions, DAG_EXECUTION_EXAMPLES, args)

        self.nodes = {}

        Logger().debug(f"DAG Agent prompt: {BASE_DAG_PROMPT}")

    def _update_node(self, alternative_answer: str, node_id: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        Update a DAG node with an alternative answer and reset dependent nodes.

        Args:
            node_id (str): The node ID to update
            answer (str): The new answer for the node

        Returns:
            Tuple[List[str], List[str], Dict[str, int]]: Updated DAG state, empty sources, usage metrics
        """
        alternative_answer = str(alternative_answer).strip()

        Logger().debug(
            f"Updating node {node_id} with alternative answer: {alternative_answer}")

        current_node = self.nodes.get(node_id)
        if not current_node:
            Logger().error(f"Node {node_id} not found in current DAG state")
            return ["Node not found"], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check if the alternative answer has already been provided
        if alternative_answer in current_node.alternative_results:
            Logger().debug(
                f"Answer '{alternative_answer}' already provided for node {node_id}")
            existing_alternatives = ", ".join(
                current_node.alternative_results) if current_node.alternative_results else "none"
            return [f"The answer '{alternative_answer}' has already been provided for node {node_id}. Previously tried alternatives: {existing_alternatives}. Please provide a different alternative answer."], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check for invalid or non-specific answers
        invalid_values = {"n/a", "unknown", "unclear",
                          "not found", "not available", "none", "null", ""}
        if alternative_answer.lower().strip() in invalid_values:
            Logger().debug(
                f"Invalid answer '{alternative_answer}' provided for node {node_id}")
            return [f"The answer '{alternative_answer}' is not a valid alternative answer. Please provide a meaningful, specific value extracted or inferred from the sources. Do not use generic values like 'unknown', 'N/A', 'unclear', or similar non-specific terms. If you cannot find a valid alternative, attempt to solve a different sub-question or try solving the original sub-question again with a modified query."], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Update the node with the new answer
        current_node.result = alternative_answer
        current_node.context = f"Alternative answer provided: {alternative_answer}"
        current_node.is_completed = True
        current_node.is_failed = False
        current_node.alternative_results.append(alternative_answer)

        if len(current_node.alternative_results) >= 4:
            current_node.alternatives_exhausted = True
            Logger().debug(
                f"Node {node_id} has exhausted alternative answers.")

        # Reset all dependent nodes
        self._reset_dependent_nodes(node_id, self.nodes)

        # Return updated DAG state
        updated_dag_state = self._serialize_dag_state(
            self.nodes, include_sources_for=self.nodes.keys())
        return [updated_dag_state], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

    def _answer(self, query: str, node_id: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        Solve a sub-question using the specialized sub-question React Agent.

        Args:
            query (str): The query to solve
            node_id (str): The node ID in the DAG

        Returns:
            Tuple[List[str], List[str], Dict[str, int]]: Updated DAG state or error message, sources, usage metrics
        """
        Logger().debug(f"Solving sub-question for node {node_id}: {query}")

        # Get the current node from our DAG state
        current_node = self.nodes.get(node_id)
        if not current_node:
            Logger().error(f"Node {node_id} not found in current DAG state")
            return ["Node not found"], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check if all dependencies are solved
        unsolved_dependencies = []
        for dep_id in current_node.dependencies:
            dep_node = self.nodes.get(dep_id)
            if not dep_node or not dep_node.is_completed:
                unsolved_dependencies.append(dep_id)

        if unsolved_dependencies:
            unsolved_str = ", ".join(unsolved_dependencies)
            return [f"Cannot solve node {node_id} because the following dependent nodes are not solved yet: {unsolved_str}. Please solve these dependencies first."], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Use the sub-question React Agent to solve the query
        notebook = self._subquestion_agent.reason(query)

        answer = notebook.get_notes()
        sources = notebook.get_sources()
        usage_metrics = notebook.get_usage_metrics()

        # Update the node state
        if answer == 'N/A':
            current_node.is_failed = True
            Logger().debug(f"Node {node_id} failed to find an answer")

            # Check for ancestor nodes for potential backtracking
            backtrack_candidates = self._find_backtrack_candidates(
                node_id, self.nodes)
            if backtrack_candidates:
                # Focus on the first (closest) candidate only
                candidate_id = backtrack_candidates[0]
                candidate_node = self.nodes.get(candidate_id)

                if candidate_node:
                    # Get sources for this candidate node
                    sources_text = ""
                    if candidate_node.unique_sources:
                        sources_list = [
                            f"- {src['content']}" for src in candidate_node.unique_sources]
                        sources_text = "\n".join(sources_list)

                    # Get previously tried alternatives
                    previous_alternatives = candidate_node.alternative_results if candidate_node.alternative_results else []
                    exclude_list = f"[{', '.join([repr(alt) for alt in previous_alternatives])}]"

                    suggestion_msg = f"I was unable to solve sub-question for node {node_id}. Consider providing an \
alternative answer for node {candidate_id} based on the sources provided below. This answer must be different from all \
previously tried answers in the exclude list. It should be FLEXIBLE in interpretation, and consider plausible assumptions \
or entities in the sources including typos. Only when all alternatives are definitely exhausted should you return 'N/A'.\n\nSources for \
node {candidate_id}:\n{sources_text}\n\nPreviously tried alternatives (exclude): {exclude_list}\n\nCall \
UPDATE_NODE('alternative_answer', '{candidate_id}') to update the DAG state with an alternative answer. \
Remember to exhaust all possible alternatives before giving up, even if they assume typos, other interpretations, or \
entities that may not seem related at first glance. If you cannot find any valid alternative answer after careful \
consideration, attempt to solve a different sub-question or try solving the original sub-question again \
with a modified query."
                    return [suggestion_msg], [], usage_metrics

            return [f"I was unable to solve sub-question for node {node_id}. Attempt to solve \
the sub-question again with a different query."], [], usage_metrics

        current_node.result = answer
        current_node.context = notebook.get_context()
        current_node.is_completed = True
        current_node.alternative_results.append(answer)
        Logger().debug(f"Node {node_id} succeeded with answer: {answer}")

        # Update sources
        current_node.sources.extend(sources)

        # Update unique sources by deduplicating based on doc_id
        seen_doc_ids = {src['doc_id']
                        for src in current_node.unique_sources}
        for source in sources:
            if source['doc_id'] not in seen_doc_ids:
                current_node.unique_sources.append(source)
                seen_doc_ids.add(source['doc_id'])

        # Return updated DAG state after successful solve
        updated_dag_state = self._serialize_dag_state(
            self.nodes, include_sources_for=self.nodes.keys())
        return [updated_dag_state], [], usage_metrics

    def reason(self, question: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question using DAG decomposition v2.

        Args:
            question (str): The question to reason about.

        Returns:
            NoteBook: A notebook containing the reasoning results.
        """
        Logger().debug(
            f"Starting DAG v2 reasoning for question: {question}, process ID: {os.getpid()}")

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        messages = []

        # Phase 1: Generate DAG plan
        response_content, planning_usage = self._create_dag_plan(
            question, BASE_DAG_PROMPT, "dag_planning_v2")

        # Update usage metrics
        for key in usage_metrics:
            usage_metrics[key] += planning_usage.get(key, 0)

        messages.append({"role": "assistant", "content": response_content})

        # Parse the DAG plan using base class method
        self.nodes = self._parse_dag_plan(response_content, DAGNodeV2)

        # Phase 2: Pass DAG plan to execution React Agent
        dag_state = self._serialize_dag_state(self.nodes)

        # Create execution prompt with DAG plan and original question
        execution_question = f"""DAG Plan:

{dag_state}

Question: {question}
"""

        # Execute the DAG plan using the execution React Agent
        execution_notebook = self._execution_agent.reason(execution_question)

        # Collect usage metrics from execution
        execution_usage = execution_notebook.get_usage_metrics()
        for key in usage_metrics:
            usage_metrics[key] += execution_usage.get(key, 0)

        final_answer = execution_notebook.get_notes()
        execution_messages = execution_notebook.get_messages()

        Logger().info(
            f"Final DAG v2 answer for question '{question}': {final_answer}")

        # Create notebook with results
        notebook = NoteBook()

        # Collect all sources from all nodes
        all_sources = []
        for node_id, node in self.nodes.items():
            if node.unique_sources:
                folder_sources = [
                    RetrievedResult(
                        doc_id=src['doc_id'],
                        content=src['content'],
                        folder_id=f"node_{node_id}"
                    )
                    for src in node.unique_sources
                ]
                all_sources.extend(folder_sources)

        notebook.update_sources(all_sources)
        notebook.update_notes(final_answer)
        notebook.update_usage_metrics(usage_metrics)

        # Combine planning and execution messages
        all_messages = messages + execution_messages
        notebook.update_messages(all_messages)
        notebook.update_context(execution_notebook.get_context())

        return notebook


class SingleProcessDAGAgentV2(BaseDAGAgentV2, SingleProcessAgent, ABC):
    """A single-process DAG agent v2."""

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args):
        SingleProcessAgent.__init__(self, args)
        BaseDAGAgentV2.__init__(self, search_function, args)


class DAGAgentV2(BaseDAGAgentV2, MultiprocessingSearchAgent, ABC):
    """A multiprocessing DAG agent v2."""

    def __init__(
        self,
        search_function: Callable[[str],
                                  Tuple[List[str], List[str], Dict[str, int]]],
        args,
        cores: int = 24
    ):
        MultiprocessingSearchAgent.__init__(self, args, cores=cores)
        BaseDAGAgentV2.__init__(self, search_function, args)


class StatefulDAGAgentV2(BaseDAGAgentV2, MultiprocessingStatefulSearchAgent, ABC):
    """A stateful DAG agent v2 that maintains state across multiple reasoning sessions."""

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores: int):
        MultiprocessingStatefulSearchAgent.__init__(self, args, cores)
        BaseDAGAgentV2.__init__(self, search_function, args)


DAG_EXECUTION_EXAMPLES = '''### Example

DAG Plan:
[
    {
        "node_id": "node_1",
        "sub_question": "What is the location of Kimbrough Memorial Stadium?",
        "dependencies": []
    },
    {
        "node_id": "node_2", 
        "sub_question": "In which county is the stadium from node 1 located?",
        "dependencies": ["node_1"]
    }
]

Question: In which county is Kimbrough Memorial Stadium located?

Iteration 1:
```json
{
    "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. I should start by solving node_1 first.",
    "actions": ["answer('Kimbrough Memorial Stadium location', 'node_1')"]
}
```

Iteration 2:
```json
{
    "thought": "I have a DAG plan with 2 nodes where node_2 depends on node_1. I should start by solving node_1 first.",
    "actions": ["answer('Kimbrough Memorial Stadium location', 'node_1')"],
    "observations": [["
    {
        [
            "node_id": "node_1",
            "sub_question": "What is the location of Kimbrough Memorial Stadium?",
            "dependencies": [],
            "result": "Canyon, Texas",
            "context": "Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District.",
            "sources": ["Doc1", "Doc2"]
        ],
        [
            "node_id": "node_2",
            "sub_question": "In which county is the stadium from node 1 located?",
            "dependencies": ["node_1"]
        ]
    }
"]]
}
```

Iteration 3:
```json
{
    "thought": "Node_1 is complete and shows the stadium is in Canyon, Texas. Now I need to solve node_2, but I must reformulate the query to replace 'stadium from node 1' with the actual location found.",
    "actions": ["answer('Canyon Texas' location', 'node_2')"]
}
```

Iteration 4:
```json
{
    "thought": "Node_1 is complete and shows the stadium is in Canyon, Texas. Now I need to solve node_2, but I must reformulate the query to replace 'stadium from node 1' with the actual location found.",
    "actions": ["answer('Canyon Texas' location', 'node_2')"],
    "observations": [["
    {
        [
            "node_id": "node_1",
            "sub_question": "What is the location of Kimbrough Memorial Stadium?",
            "dependencies": [],
            "result": "Canyon, Texas",
            "context": "Kimbrough Memorial Stadium is a stadium in Canyon, Texas. It is owned by Canyon Independent School District.",
            "sources": ["Doc1", "Doc2"]
        ],
        [
            "node_id": "node_2",
            "sub_question": "In which county is the stadium from node 1 located?",
            "dependencies": ["node_1"],
            "result": "Randall County",
            "context": "Canyon is a city in, and the county seat of, Randall County, Texas, United States.",
            "sources": ["Doc3", "Doc4"]
        ]
    }
"]]
}
```

Iteration 5:
```json
{
    "thought": "Both nodes are now complete. Node_1 found that Kimbrough Memorial Stadium is in Canyon, Texas, and node_2 found that Canyon is in Randall County. I can now answer the original question.",
    "final_answer": "Randall County"
}
```

**IMPORTANT**: Your responses MUST NEVER contain any observations under absolutely no circumstances. Observations are returned to you after each action
and are not part of your response. You must only respond with the JSON structure containing your thoughts and actions, or the final answer when ready.
'''

DAG_EXECUTION_REACT_PROMPT = '''You are an intelligent agent that solves complex questions by \
executing DAG (Directed Acyclic Graph) plans. You can use tools to solve sub-questions and update \
the state of the DAG dynamically. Once you believe the current DAG state is sufficient to answer the \
original question, you must provide an EXACT answer, using only words found in the the documents when \
possible. You should not include any additional commentary, explanations, or notes in your \
final response. If the answer can be a single word (e.g., Yes, No, a date, or an object), please provide \
just that word.

You may only execute one action per iteration. After each action, you will receive the updated DAG state \
or feedback on your action. Use this information to inform your next steps.

You can choose the following tools to execute the DAG plan.

'''
