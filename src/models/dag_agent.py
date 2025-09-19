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
from models.agent import (
    MultiprocessingSearchAgent,
    MultiprocessingStatefulSearchAgent,
    NoteBook,
    SelfContainedAgent,
    SingleProcessAgent
)
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param
from utils.structure_response import parse_structured_response

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
        self.result = None
        self.search_results = []
        self.sources = []


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

        self._prompt = DAG_AGENT_PROMPT

        Logger().debug(f"DAG Agent prompt: {self._prompt}")

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

    def _reformulate_query(
            self,
            original_question: str,
            dependency_context: str,
            node_id: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Reformulate a question based on dependency results to make it more specific and searchable.

        Args:
            original_question (str): The original sub-question
            dependency_context (str): Context from completed dependencies
            node_id (str): Node ID for logging

        Returns:
            Tuple[str, Dict[str, int]]: Reformulated query and usage metrics
        """
        user_content = f"{dependency_context}Original question: {original_question}"

        reformulation_request = {
            "custom_id": f"reformulate_{node_id}",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": DAG_QUERY_REFORMULATION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": default_job_args['temperature']
            if supports_temperature_param(self._args.model) else None,
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 200,
        }

        result = chat_completions([reformulation_request])[0][0]
        reformulated_query = result.choices[0].message.content.strip()

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        Logger().debug(
            f"Reformulated query for node {node_id}: {reformulated_query}")

        return reformulated_query, usage_metrics

    def _expand_query(
        self,
        original_query: str,
        previous_attempts: List[str],
        node_id: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Expand/reformulate a query when previous attempts failed to find results.

        Args:
            original_query (str): The original search query
            previous_attempts (List[str]): List of previously tried queries
            node_id (str): Node ID for logging

        Returns:
            Tuple[str, Dict[str, int]]: Expanded query and usage metrics
        """
        attempts_text = "\n".join(
            [f"Attempt {i+1}: {query}" for i, query in enumerate(previous_attempts)])
        user_content = f"Original query: {original_query}\n\nPrevious unsuccessful attempts:\n{attempts_text}"

        expansion_request = {
            "custom_id": f"expand_{node_id}",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": DAG_QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,  # Slightly higher for more creative reformulations
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 200,
        }

        result = chat_completions([expansion_request])[0][0]
        expanded_query = result.choices[0].message.content.strip()

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        Logger().debug(f"Expanded query for node {node_id}: {expanded_query}")

        return expanded_query, usage_metrics

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

    # pylint: disable-next=too-many-locals
    def _execute_node(
        self,
        node: DAGNode,
        nodes: Dict[str, DAGNode]
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Execute a single node by searching for information to answer its sub-question.
        Uses a ReAct loop to reformulate queries if initial attempts don't find answers.

        Args:
            node (DAGNode): The node to execute
            nodes (Dict[str, DAGNode]): All nodes for context

        Returns:
            Tuple[Dict[str, int], List[str]]: Usage metrics and sources
        """
        max_attempts = 3
        total_usage_metrics = {"completion_tokens": 0,
                               "prompt_tokens": 0, "total_tokens": 0}
        working_memory = []  # Track attempts but not search results

        # Build context from completed dependencies
        dependency_context = ""
        if node.dependencies:
            dependency_results = []
            for dep_id in node.dependencies:
                dep_node = nodes[dep_id]
                if dep_node.is_completed and dep_node.result:
                    dependency_results.append(
                        f"- {dep_node.sub_question}: {dep_node.result}")

            if dependency_results:
                dependency_context = "Previous findings:\n" + \
                    "\n".join(dependency_results) + "\n\n"

        # Initial query - reformulate based on dependency results if available
        search_query = node.sub_question
        if dependency_context:
            search_query, reformulation_metrics = self._reformulate_query(
                node.sub_question, dependency_context, node.node_id
            )
            # Update total metrics
            for key in total_usage_metrics:
                total_usage_metrics[key] += reformulation_metrics.get(key, 0)

        # ReAct loop - try multiple query formulations
        for attempt in range(max_attempts):
            Logger().debug(
                f"Node {node.node_id} attempt {attempt + 1}: {search_query}")

            # Track this attempt in working memory (query only, not results)
            working_memory.append(f"Attempt {attempt + 1}: {search_query}")

            # Execute search
            documents, sources, search_metrics = self._search_function(
                search_query)

            # Update total metrics with search metrics
            for key in total_usage_metrics:
                total_usage_metrics[key] += search_metrics.get(key, 0)

            # Try to extract answer from search results
            search_results_text = "\n".join(
                [f'Document {i+1}: {doc}' for i, doc in enumerate(documents)])
            user_content = (
                f"{dependency_context}Question: {node.sub_question}\n\nSearch Results:\n{search_results_text}"
            )

            answer_request = {
                "custom_id": f"dag_node_{node.node_id}_attempt_{attempt + 1}",
                "model": self._args.model,
                "messages": [
                    {"role": "system", "content": DAG_NODE_ANSWER_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "temperature": default_job_args['temperature']
                if supports_temperature_param(self._args.model) else None,
                "frequency_penalty": default_job_args['frequency_penalty'],
                "presence_penalty": default_job_args['presence_penalty'],
                "max_completion_tokens": 500,
            }

            result = chat_completions([answer_request])[0][0]

            # Update total metrics
            total_usage_metrics["completion_tokens"] += result.usage.completion_tokens
            total_usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
            total_usage_metrics["total_tokens"] += result.usage.total_tokens

            # Extract the answer
            answer = result.choices[0].message.content.strip()

            # If we found an answer, we're done
            if answer != "N/A":
                node.result = answer
                node.is_completed = True
                node.search_results = documents  # Store final successful search results
                node.sources = sources

                Logger().debug(
                    f"Node {node.node_id} succeeded on attempt {attempt + 1}: {answer}")
                return total_usage_metrics, sources

            # If this was the last attempt, fail
            if attempt == max_attempts - 1:
                Logger().debug(
                    f"Node {node.node_id} failed after {max_attempts} attempts")
                raise ValueError(
                    f"Node {node.node_id} could not find answer for: {node.sub_question} after {max_attempts} attempts"
                )

            # Generate a new query for the next attempt
            search_query, expansion_metrics = self._expand_query(
                node.sub_question, working_memory, node.node_id
            )

            # Update total metrics
            for key in total_usage_metrics:
                total_usage_metrics[key] += expansion_metrics.get(key, 0)

        return total_usage_metrics, []

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
        sources: Dict[str, List[str]] = {}

        # Phase 1: Generate DAG plan
        open_ai_request = {
            "custom_id": "dag_planning",
            "model": self._args.model,
            "messages": [
                {"role": "system", "content": self._prompt},
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
            flattened_sources.extend([(folder_id, doc_id)
                                     for doc_id in folder_sources])

        notebook.update_sources([
            RetrievedResult(
                doc_id=self._corpus[doc_id]['doc_id'],
                content=self._corpus[doc_id]['content'],
                folder_id=folder_id
            )
            for folder_id, doc_id in flattened_sources
        ])
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
question and break it down into a directed acyclic graph (DAG) of sub-questions.

Each sub-question should:
1. Be answerable through document search
2. Build towards answering the main question
3. Have clear dependencies on other sub-questions when needed
4. Be as specific and focused as possible
5. Contain only ONE question - never combine multiple questions in a single sub-question

## CONSTRAINTS

- Node IDs must be unique and follow the pattern "node_N" where N is a number
- Dependencies must reference valid node IDs
- The DAG must be acyclic (no circular dependencies)
- Each sub-question should be clear and searchable
- Each sub-question must ask only ONE thing - avoid compound questions
- Focus on factual information that can be found in documents

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
    "reasoning": "I need to first find what stadium is owned by Canyon Independent School District, then find where that stadium is located, and finally determine which county that location is in.",
    "dag_plan": [
        {
            "node_id": "node_1",
            "sub_question": "What stadium is owned by Canyon Independent School District?",
            "dependencies": []
        },
        {
            "node_id": "node_2",
            "sub_question": "Where is this stadium located (city and state)?",
            "dependencies": ["node_1"]
        },
        {
            "node_id": "node_3",
            "sub_question": "Which county is this location in?",
            "dependencies": ["node_2"]
        }
    ]
}
'''

DAG_NODE_ANSWER_PROMPT = '''You are a helpful assistant that extracts precise answers from search results. \
Based on the search results provided by the user, provide a precise answer to the question. Use only information \
found in the search results.

If the search results don't contain enough information to answer the question, respond with "N/A".
'''

DAG_SYNTHESIS_PROMPT = '''You are an intelligent QA agent. Based on the sub-question results provided by the \
user, provide an EXACT answer to the original question, using only words found in the sub-question results \
when possible. Under no circumstances should you include any additional commentary, explanations, reasoning, \
or notes in your final response. If the answer can be a single word (e.g., Yes, No, a date, or an object), \
please provide just that word.
'''

DAG_QUERY_REFORMULATION_PROMPT = '''You are a helpful assistant that reformulates questions to be more specific \
and searchable by incorporating concrete details from previous findings. Your task is to replace abstract references \
with specific information while keeping the question focused and concise.

Guidelines:
- Replace references like "the person identified in node 1" with the actual name/entity found
- Remove redundant descriptive phrases that repeat information from previous findings
- Keep the reformulated question direct and searchable
- Do not add unnecessary context that was already established

Provide a reformulated question that incorporates specific details from the previous findings. Only return the reformulated question, nothing else.
'''

DAG_QUERY_EXPANSION_PROMPT = '''You are a helpful assistant that expands and reformulates search queries when \
previous attempts have failed to find results. Your task is to create alternative query formulations using different \
keywords, synonyms, or query structures that might yield better search results.

You can use synonyms or alternative terms, break down complex queries into simpler parts, add relevant keywords or \
context, try different phrasings or question structures, or focus on key entities or concepts.

Provide a new search query that takes a different approach from the previous attempts. Only return the new query, \
nothing else.
'''
