"""
A DAG-based agent v2 that decomposes questions into directed acyclic graphs and delegates execution to a React Agent.
This version creates a DAG plan and passes it to a React Agent with an answer tool for autonomous execution.
"""

# pylint: disable=duplicate-code, too-many-lines

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
    SingleProcessAgent
)
from models.base_dag import BaseDAGAgent as BaseDag, BaseDAGNode, DEFAULT_DAG_JOB_ARGS
from models.dataset import Dataset
from models.react_agent import BaseIntelligentAgent
from models.retrieved_result import RetrievedResult
from models.question_answer import QuestionAnswer
from utils.model_utils import supports_temperature_param
from utils.structure_response import parse_structured_response


class BaseDAGExecutionReactAgent(BaseIntelligentAgent, ABC):
    """
    A specialized React agent for executing DAG plans autonomously.
    This agent receives a DAG plan and uses the answer tool to execute it.
    """

    def __init__(self, actions: Dict[str, Action], args):
        """
        Initialize the DAG execution React agent with a modified prompt.

        Args:
            actions (Dict[str, Action]): Actions available to the agent (should include answer)
            examples (str): Example interactions
            args: Agent arguments
        """
        BaseIntelligentAgent.__init__(
            self, actions, "", args, DAG_EXECUTION_REACT_PROMPT)

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

    The agent operates in three phases:
    1. Planning Phase: Decompose the main question into a DAG of sub-questions
    2. Execution Phase: Pass the DAG plan to a React Agent that autonomously executes it
    3. Synthesis Phase: Synthesize the final answer from the completed DAG state
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
                "that should **explicitly** contain all entities and keywords needed to "
                "answer the query. The query must **NOT** contain any node IDs such as node_1, node_2, under "
                " any circumstances. Instead, replace node references with actual values from completed nodes. "
                "The second argument node_id identifies which node in the DAG plan this sub-question "
                "corresponds to. This tool returns the updated node state after solving the sub-question, or "
                "provides feedback if the query cannot be answered.",
                self._answer
            ),
            "update_node": Action(
                "Update a DAG node with an answer. You **MUST** use this method when you can answer "
                "a sub-question directly without executing the answer tool. "
                "For example, if an answer can be inferred from the result of other nodes. "
                "You may also update an already processed node with an alternative answer to explore "
                "different paths. The first argument is a string with the value to update the node with, "
                "and the second argument is the node_id to update. Returns the updated node state. ",
                self._update_node
            )
        }

        # Create the main execution React Agent with modified prompt
        self._execution_agent = BaseDAGExecutionReactAgent(actions, args)

        self.nodes = {}
        # Track the node that failed for backtracking
        self._current_failed_node_id = None

        # Initialize prompt objects
        self.synthesis_prompt = ""

    def _get_backtrack_suggestion(self, failed_node_id: str) -> Tuple[str, bool]:
        """
        Get backtracking suggestion for a failed node, excluding already exhausted candidates.

        Args:
            failed_node_id (str): The node that failed to be solved
            exclude_node_ids (List[str]): Node IDs to exclude from candidates (already exhausted)

        Returns:
            Tuple[str, bool]: (suggestion_message, has_candidates)
        """
        # Check for ancestor nodes for potential backtracking
        backtrack_candidates = self._find_backtrack_candidates(
            failed_node_id, self.nodes)

        if backtrack_candidates:
            # Focus on the first (closest) available candidate
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

                suggestion_msg = f"I was unable to solve sub-question for node {failed_node_id}. Consider providing an \
alternative answer for node {candidate_id} based on the sources provided below. This answer must be different from all \
previously attempted answers. However, it should be FLEXIBLE in interpretation, and consider plausible assumptions \
or entities in the sources. Only when all alternatives are definitely exhausted should you return 'N/A'. \
Remember to exhaust all possible alternatives before returning 'N/A', even if they assume typos, other interpretations, or \
entities that may not seem related at first glance. \n\nSources for \
node {candidate_id}:\n{sources_text}\n\nPreviously attempted answers: {exclude_list}\n\nCall \
UPDATE_NODE('alternative_answer', '{candidate_id}') to update the node state with an alternative answer. \
If no alternative answer exists, you may call UPDATE_NODE('N/A', '{candidate_id}') to mark the node as exhausted"
                return suggestion_msg, True

        return f"I was unable to solve sub-question for node {failed_node_id}. Attempt to solve \
the sub-question again with a different query.", False

    def _update_node(self, answer: str, node_id: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """
        Update a DAG node with an alternative answer and reset dependent nodes.

        Args:
            node_id (str): The node ID to update
            answer (str): The new answer for the node
            context (str): Additional context for the update that explains why this action is being taken

        Returns:
            Tuple[List[str], List[str], Dict[str, int]]: Updated DAG state, empty sources, usage metrics
        """
        answer = str(answer).strip()

        Logger().debug(
            f"Updating node {node_id} with answer: {answer}")

        current_node = self.nodes.get(node_id)

        if not current_node:
            Logger().error(f"Node {node_id} not found in current DAG state")
            return ["Node not found"], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check if the alternative answer has already been provided
        if answer in current_node.alternative_results:
            Logger().debug(
                f"Answer '{answer}' already provided for node {node_id}")
            existing_alternatives = ", ".join(
                current_node.alternative_results) if current_node.alternative_results else "none"
            return [
                f"The answer '{answer}' has already been provided for node {node_id}. Previously \
tried alternatives: {existing_alternatives}. Please provide a different alternative answer. \
If no alternative answer exists, you may call UPDATE_NODE('N/A', '{node_id}') to mark the node as exhausted."
            ], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check if no more alternatives are possible
        if answer.lower().strip() == "n/a":
            Logger().debug(
                f"Invalid answer '{answer}' provided for node {node_id}, marking as exhausted")

            # Mark this node as exhausted and get another backtrack candidate
            current_node.alternatives_exhausted = True

            # Revert to first answer as it is more likely correct
            if current_node.alternative_results:
                current_node.result = current_node.alternative_results[0]
                current_node.context = ""

            # Use the stored failed node ID to get another backtrack suggestion
            if self._current_failed_node_id:
                # Get another backtrack suggestion, excluding the current exhausted node
                suggestion_msg, _ = self._get_backtrack_suggestion(
                    self._current_failed_node_id)
                return [suggestion_msg], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

            return [
                f"Node {node_id} marked as exhausted. No more alternatives available."
            ], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Update the node with the new answer
        current_node.result = answer
        current_node.context = ""
        current_node.is_completed = True
        current_node.is_failed = False
        current_node.alternative_results.append(answer)

        if len(current_node.alternative_results) >= 4:
            current_node.alternatives_exhausted = True
            Logger().debug(
                f"Node {node_id} has exhausted alternative answers.")

        # Reset all dependent nodes
        self._reset_dependent_nodes(node_id, self.nodes)

        # Find immediate dependent nodes that can now be solved
        immediate_dependents = []
        for potential_node_id, potential_node in self.nodes.items():
            if node_id in potential_node.dependencies:
                # Check if all other dependencies are completed
                other_deps_completed = all(
                    self.nodes[dep].is_completed and not self.nodes[dep].is_failed
                    for dep in potential_node.dependencies if dep != node_id
                )
                if other_deps_completed:
                    immediate_dependents.append(potential_node_id)

        if immediate_dependents:
            dependents_str = ", ".join(immediate_dependents)
            guidance_msg = f"Node {node_id} has been successfully updated with answer: '{answer}'. " \
                f"The following dependent nodes can now be solved: {dependents_str}. " \
                f"Continue by solving one of these dependent sub-questions."
        else:
            guidance_msg = f"Node {node_id} has been successfully updated with answer: '{answer}'. " \
                f"All dependent nodes are complete or no more nodes can be solved. DAG execution may be finished."

        return [guidance_msg], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

    # pylint: disable-next=too-many-locals
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

        # Reset the failed node tracker when starting a new answer attempt
        self._current_failed_node_id = None

        # Get the current node from our DAG state
        node = self.nodes.get(node_id)

        if not node:
            Logger().error(f"Node {node_id} not found in current DAG state")
            return ["Node not found"], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Check if all dependencies are solved
        unsolved_dependencies = []
        for dep_id in node.dependencies:
            dep_node = self.nodes.get(dep_id)
            if not dep_node or not dep_node.is_completed:
                unsolved_dependencies.append(dep_id)

        if unsolved_dependencies:
            unsolved_str = ", ".join(unsolved_dependencies)
            return [
                f"Cannot solve node {node_id} because the following dependent nodes are not \
solved yet: {unsolved_str}. Please solve these dependencies first."
            ], [], {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        # Use the sub-question React Agent to solve the query
        notebook = self._subquestion_agent.reason(query)

        answer = notebook.get_notes()
        sources = notebook.get_sources()
        last_thought = notebook.get_context()
        usage_metrics = notebook.get_usage_metrics()

        # Update sources
        node.sources.extend(sources)

        # Update unique sources by deduplicating based on doc_id
        seen_doc_ids = {src['doc_id'] for src in node.unique_sources}
        for source in sources:
            if source['doc_id'] not in seen_doc_ids:
                node.unique_sources.append(source)
                seen_doc_ids.add(source['doc_id'])

        # Update the node state
        if answer == 'N/A':
            Logger().debug(f"Node {node_id} failed to find an answer")
            node.is_failed = True
            self._current_failed_node_id = node_id  # Track this as the failed node

            # Get backtracking suggestion
            suggestion_msg, _ = self._get_backtrack_suggestion(
                node_id)
            return [suggestion_msg], [], usage_metrics

        Logger().debug(f"Node {node_id} succeeded with answer: {answer}")
        node.result = answer
        node.is_completed = True
        node.context = last_thought
        node.alternative_results.append(answer)

        updated_dag_state = self._serialize_dag_state(
            {node_id: node}, include_sources_for=[node_id])
        return [updated_dag_state], [], usage_metrics

    def _synthesize_final_answer(self, question: str) -> Tuple[str, Dict[str, int]]:
        """
        Synthesize the final answer from the completed DAG state using direct LLM call.

        Args:
            question (str): The original question to answer

        Returns:
            Tuple[str, Dict[str, int]]: Final answer and usage metrics
        """
        # Get failed node IDs to include sources
        failed_node_ids = [node_id for node_id, node in self.nodes.items()
                           if node.is_failed and node.unique_sources]

        # Serialize DAG state with sources for failed nodes
        dag_state = self._serialize_dag_state(
            self.nodes, include_sources_for=failed_node_ids)

        # Build the system prompt
        system_prompt = f"{self.synthesis_prompt}\n## DAG State\n\n{dag_state}"

        Logger().debug(f"""DAG Synthesis system prompt:
{system_prompt}""")

        # Create the request with structured JSON response
        request = {
            "custom_id": "dag_synthesis_v2",
            "model": self._think_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": DEFAULT_DAG_JOB_ARGS['temperature']
            if supports_temperature_param(self._think_model) else None,
            "frequency_penalty": DEFAULT_DAG_JOB_ARGS['frequency_penalty'],
            "presence_penalty": DEFAULT_DAG_JOB_ARGS['presence_penalty'],
            "max_completion_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "dag_synthesis_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["result", "reason"],
                        "additionalProperties": False
                    }
                }
            }
        }

        result = chat_completions([request])[0][0]

        # Parse the structured response
        structured_response = parse_structured_response(
            result.choices[0].message.content.strip())

        final_answer = structured_response['result']
        reason = structured_response['reason']

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        Logger().debug(
            f"Synthesized final answer: {final_answer}, reason: {reason}")
        return final_answer, usage_metrics

    def index(self, dataset: Dataset) -> None:
        """
        Prepares the agent with the given dataset.

        Args:
            dataset (Dataset): the dataset to process
        """
        self._execution_agent._prompt += dataset.get_prompt('dag_execution_footer')
        self.synthesis_prompt = DAG_SYNTHESIS_PROMPT + dataset.get_prompt('dag_synthesis_footer')

        super().index(dataset)

    # pylint: disable-next=too-many-locals
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
        response_content, planning_usage = self._create_dag_plan(question)

        # Update usage metrics
        for key in usage_metrics:
            usage_metrics[key] += planning_usage.get(key, 0)

        messages.append({"role": "assistant", "content": response_content})

        # Parse the DAG plan
        self.nodes = self._parse_dag_plan(response_content, DAGNodeV2)

        dag_state = self._serialize_dag_state(self.nodes)

        # Execute the DAG plan using the execution React Agent
        execution_notebook = self._execution_agent.reason(dag_state)

        # Collect usage metrics from execution
        execution_usage = execution_notebook.get_usage_metrics()
        for key in usage_metrics:
            usage_metrics[key] += execution_usage.get(key, 0)

        execution_messages = execution_notebook.get_messages()

        # Phase 3: Synthesize final answer
        final_answer, synthesis_usage = self._synthesize_final_answer(question)

        # Update usage metrics
        for key in usage_metrics:
            usage_metrics[key] += synthesis_usage.get(key, 0)

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
        cores: int = 32
    ):
        MultiprocessingSearchAgent.__init__(self, args, cores=cores)
        BaseDAGAgentV2.__init__(self, search_function, args)


class StatefulDAGAgentV2(BaseDAGAgentV2, MultiprocessingStatefulSearchAgent, ABC):
    """A stateful DAG agent v2 that maintains state across multiple reasoning sessions."""

    def __init__(self, search_function: Callable[[str], Tuple[List[str], List[str], Dict[str, int]]], args, cores: int):
        MultiprocessingStatefulSearchAgent.__init__(self, args, cores)
        BaseDAGAgentV2.__init__(self, search_function, args)


DAG_EXECUTION_REACT_PROMPT = '''You are an intelligent agent that executes DAG (Directed Acyclic Graph) plans. \
Your only responsibility is to systematically solve each sub-question in the DAG by using the available tools. \
You should work through the DAG nodes in dependency order, solving independent nodes first, then nodes that depend on them \
using the **RESULT** and **EXPLANATION** fields from completed nodes.

When all nodes that can be completed are finished, you should provide "EXECUTION_COMPLETE" as your final answer \
to indicate that DAG execution is finished.

After each action, you will receive the updated node state or feedback on your action. Use this information to inform your next steps.

You can choose the following tools to execute the DAG plan.
'''

DAG_SYNTHESIS_PROMPT = '''Answer the given question based on the available information from each node to formulate a \
CONCISE and COMPLETE answer. Provide an EXACT answer using only words found in the results when possible. DO NOT REPEAT the \
question in your answer under any circumstances. If the answer can be a single word (e.g., Yes, No, a date, or an object), \
please provide just that word. If the information seems insufficient, please make plausible assumptions about the available information, \
assuming typos, flexible interpretations, or best possible answer with the available information. Only if **definitely** no answer exists, \
respond with 'N/A'.

'''
