"""
An abstract class representing an intelligent agent that can reason over a dataset.
It extends the Agent class and provides additional functionality for reasoning.
"""

from abc import ABC
from inspect import signature
import json
import os
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from models.retrieved_result import RetrievedResult
from plugins.reflector import Reflector
from utils.model_utils import supports_temperature_param
from utils.structure_response import parse_structured_response


class BaseIntelligentAgent(SelfContainedAgent, ABC):
    """
    An abstract class representing an intelligent agent that can reason over a dataset.
    It extends the Agent class and provides additional functionality for reasoning.

    The agent supports interleaved reflection during reasoning, where each step can be 
    evaluated by a reflector that provides feedback to improve the reasoning process.
    The main agent decides how to integrate this feedback into its reasoning chain.

    Examples can be provided to help the agent understand how it can best use the available tools for reasoning.
    Please make sure the examples adhere to the ReACT framework and are formatted correctly for better results.
    """

    def __init__(self, actions: Dict[str, Action], examples: str, args, custom_prompt: Optional[str] = None):
        SelfContainedAgent.__init__(self, args)
        self._max_iterations = 8
        self._enable_interleave_reflection = False

        if len(actions) == 0:
            Logger().error("At least one action must be provided")
            raise ValueError("At least one action must be provided")

        self._actions = actions

        # Build tools prompt section based on the given function signatures ignoring the context parameter
        tools_prompt = "\n".join([
            (
                f"- **{name}"
                f"({', '.join([p for p in signature(act.function).parameters.keys() if p != 'context'])})**:"
                f" {act.description}"
            )
            for name, act in actions.items()
        ])

        # Pick the fist action to use as an example on how the model should format calls to actions
        tool_format_example = (
            f"\"actions\": [\""
            f"{list(actions.keys())[0]}"
            # pylint: disable-next=line-too-long
            f"({', '.join([p for p in signature(list(actions.values())[0].function).parameters.keys() if p != 'context'])})"
            f"\"]"
        )

        # Split the default prompt at "## AVAILABLE TOOLS" to separate role and tools sections
        default_prompt_parts = REACT_AGENT_PROMPT.split(
            "## AVAILABLE TOOLS", 1)
        role_section = default_prompt_parts[0].strip()
        tools_and_format_section = "## AVAILABLE TOOLS" + \
            default_prompt_parts[1] if len(default_prompt_parts) > 1 else ""

        # Use custom prompt as role replacement if provided, otherwise use default role section
        final_role_section = custom_prompt if custom_prompt is not None else role_section

        # Combine role section with tools section
        full_prompt = final_role_section + "\n\n" + tools_and_format_section
        template = Template(full_prompt)

        self._prompt = template.substitute(
            tools=tools_prompt, tool_format_example=tool_format_example)

        if len(examples) > 0:
            self._prompt += f"\n## EXAMPLES\n{examples}"

        Logger().debug(f"Agent prompt: {self._prompt}")

    # pylint: disable-next=too-many-branches
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
                        # Check if this is truly the end of the quoted string
                        # Look ahead to see if the next non-whitespace character is a comma or end of string
                        # Handles edge cases like "search('jon's house, or a car', 2)"
                        j = i + 1
                        while j < len(args_str) and args_str[j].isspace():
                            j += 1

                        if j >= len(args_str) or args_str[j] == ',':
                            # This is the closing quote
                            in_quotes = False
                            current_arg += char
                            quote_char = None
                        else:
                            # This is a quote inside the string, not the closing quote
                            current_arg += char
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

    def _reflect(
        self,
        reflector: Reflector,
        current_step: Dict[str, Any]
    ) -> bool:
        """
        Reflect on the current reasoning step using the reflector.

        Args:
            reflector (Reflector): The reflector instance to use for reflection.
            current_step (Dict[str, Any]): The current reasoning step to reflect on.
            iteration (int): The current iteration number for logging purposes.

        Returns:
            Tuple[Optional[str], Dict[str, int]]: (feedback_if_any, usage_metrics)
        """
        # Get reflection feedback from the reflector
        return reflector.reflect_on_step(current_step)

    def _turn_from_response_factory(self, structured_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a turn from the structured response.

        Args:
            structured_response (Dict[str, Any]): The structured response from the model.

        Returns:
            Dict[str, Any]: The extracted turn.
        """
        turn = {
            "thought": structured_response.get("thought", ""),
            "actions": structured_response.get("actions", []),
            "observations": [],
            "final_answer": structured_response.get("final_answer", None)
        }
        return turn

    def _filtered_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a filtered turn for the given job.

        Args:
            job (dict): The job dictionary to remove empty fields from.

        Returns:
            dict: A dictionary representing the filtered turn.
        """
        return {k: v for k, v in turn.items() if v not in (None, "", [], {})}

    # pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
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
        sources: Dict[str, List[int]] = {}
        prev_iteration_feedback = False

        iteration = 0
        final_answer = None
        final_thought = None

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": question}
        ]

        reflector = Reflector(question=question)

        while final_answer is None:
            Logger().debug(
                f"Iteration {iteration + 1} for question: {question}")

            # If this is the last iteration, we force the model to provide a final answer
            if iteration == self._max_iterations - 1:
                messages.append(
                    {"role": "system", "content": REACT_AGENT_LAST_ITERATION_PROMPT})

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

            # Replace \' with ' to handle escaped single quotes
            response_content = response_content.replace("\\'", "'")

            # Parse the structured response
            structured_response = parse_structured_response(
                response_content)

            if structured_response is None:
                # If parsing failed, give up
                break

            turn = self._turn_from_response_factory(structured_response)
            thought = turn["thought"]
            actions = turn["actions"]
            final_answer = turn["final_answer"]
            final_thought = thought if final_answer else None

            # If we reached max iterations and no final answer, force it to N/A
            # Do not execute any more actions to save cost
            if iteration == self._max_iterations and final_answer is None:
                Logger().warn(
                    f"Max iterations reached for question: {question} without a final answer.")
                final_answer = "N/A"
                final_thought = thought
                break

            turn = {'thought': thought, 'actions': actions,
                    'observations': [], 'final_answer': final_answer}

            if self._enable_interleave_reflection and not prev_iteration_feedback and not final_answer:
                # Create step with latest observations and current thought
                reflection_step = {
                    'thought': thought,
                }

                # Get reflection feedback on the current turn
                reflect_feedback = self._reflect(
                    reflector,
                    reflection_step,
                )

                # If we have reflection feedback, self-reflect and potentially adjust the approach
                if reflect_feedback:
                    # Save the current turn to STM first
                    filtered_turn = self._filtered_turn(turn)
                    messages.append(
                        {"role": "system", "content": json.dumps(filtered_turn)})
                    stm.append(json.dumps(filtered_turn))

                    # Add reflection feedback to the conversation
                    messages.append(
                        {"role": "system", "content": REACT_AGENT_REFLECTION_PROMPT})

                    # Let the agent process the feedback and potentially update its approach
                    # The agent can choose to continue with actions or provide a final answer
                    prev_iteration_feedback = True
                    continue

            # Mark that the current iteration did not have feedback
            prev_iteration_feedback = False

            for action_index, action in enumerate(actions, 1):
                # Parse action string to extract function name and arguments
                action_func, *action_args = self._parse_action(action)

                sig = signature(action_func)
                has_context = sig.parameters.get('context', None)

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

                # Track sources with folder_id to maintain retrieval order
                if action_sources:
                    folder_id = f"iter_{iteration}_action_{action_index}"
                    sources[folder_id] = list(action_sources)

            filtered_turn = self._filtered_turn(turn)
            messages.append(
                {"role": "system", "content": json.dumps(filtered_turn)})
            stm.append(json.dumps(filtered_turn))
            iteration += 1

        # Update usage metrics with reflection metrics
        usage_metrics["completion_tokens"] += reflector.usage_metrics.get(
            "completion_tokens", 0)
        usage_metrics["prompt_tokens"] += reflector.usage_metrics.get(
            "prompt_tokens", 0)
        usage_metrics["total_tokens"] += reflector.usage_metrics.get(
            "total_tokens", 0)

        if final_answer is None:
            final_answer = "N/A"

        Logger().info(
            f"Final answer for question '{question}': {final_answer}")

        # Create notebook with results
        # TODO: Revisit whether this should have knowledge about the corpus
        notebook = NoteBook()

        # Flatten sources dictionary maintaining insertion order (folder_ids appear in order)
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
        notebook.update_context(final_thought)

        return notebook


class SingleProcessIntelligentAgent(BaseIntelligentAgent, SingleProcessAgent, ABC):
    """
    A class representing a single-process intelligent agent that uses a single-process.
    """

    def __init__(self, actions: Dict[str, Action], examples: str, args, custom_prompt: Optional[str] = None):
        SingleProcessAgent.__init__(self, args)
        BaseIntelligentAgent.__init__(
            self, actions, examples, args, custom_prompt)


class IntelligentAgent(BaseIntelligentAgent, MultiprocessingSearchAgent, ABC):
    """
    A class representing an intelligent agent that combines multiprocessing capabilities with intelligent reasoning.
    """

    def __init__(self, actions: Dict[str, Action], examples: str, args, cores=4, custom_prompt: Optional[str] = None):
        MultiprocessingSearchAgent.__init__(self, args, cores)
        BaseIntelligentAgent.__init__(
            self, actions, examples, args, custom_prompt)


class StatefulIntelligentAgent(BaseIntelligentAgent, MultiprocessingStatefulSearchAgent, ABC):
    """
    A class representing a stateful intelligent agent that maintains state across multiple reasoning sessions.
    """

    def __init__(self, actions: Dict[str, Action], examples: str, args, cores, custom_prompt: Optional[str] = None):
        MultiprocessingStatefulSearchAgent.__init__(self, args, cores)
        BaseIntelligentAgent.__init__(
            self, actions, examples, args, custom_prompt)

# pylint: disable=duplicate-code


# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

REACT_AGENT_PROMPT = '''You are an intelligent QA agent that will be presented with a question, and you will \
need to search for relevant documents that support the answer to the question. You will then use these documents to provide an EXACT answer, \
using only words found in the documents when possible. Under no circumstances should you include any additional \
commentary, explanations, reasoning, or notes in your final response. \
If the answer can be a single word (e.g., Yes, No, a date, or an object), please provide just that word.

You should decompose the question into multiple sub-questions if necessary, and search for relevant documents for each sub-question. \

You can choose one or more tool calls to gather information. Use them wisely based on the intent of the query.
You can choose the following tools to find relevant documents.

## AVAILABLE TOOLS

$tools

## RESPONSE FORMAT

Respond with exactly one JSON object per response and do not include any text before or after the JSON object. \
Use either the intermediate format or the final format, never both in the same response.

Your intermediate responses must be in valid JSON format with the following structure:

```json
{
    "thought": "Your reasoning process that clearly explains what you are trying to find out",
    $tool_format_example
}
```

During intermediate responses, actions must not be empty and must contain at least one action.

Your final answer must be formatted in valid JSON format with the following structure:

```json
{
    "thought": "Your final reasoning process that clearly explains how you arrived at the final answer and why the answer is both correct and complete",
    "final_answer": "your final answer 'formatted' as a string"
}
```

Ensure all string values in your JSON response have properly escaped quotes when necessary if using double quotes.
'''

REACT_AGENT_LAST_ITERATION_PROMPT = '''You must now try to provide a final answer with \
the information gathered so far even if the answer may be incomplete or not fully correct to the best of your knowledge. \

Take a moment to revisit the question, reflect on the information you have gathered, and provide the best possible answer.

Unless strictly necessary, you will answer with 'N/A' if you definitely cannot provide an answer.
'''

REACT_AGENT_REFLECTION_PROMPT = '''Since you are not making significant progress towards the final answer, \
you are likely to be omitting important information or making mistakes in your reasoning. \
Analyze the observations you have made so far, and identify any plausible assumptions you can make from them to move on towards the next step. \
Plausible assumptions are information that may not be explicitly stated in the observations, but can be reasonably inferred from them. \
Do not continue to search for the same information again. Instead, use these assumptions to guide your next steps. Assumption should be based \
only on the retrieved information. \
'''
