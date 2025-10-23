"""reflector.py
This module provides a Reflector class that detects when an agent is stuck in its reasoning
process and provides guidance for moving forward.
"""
from typing import Dict, List
from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from utils.structure_response import parse_structured_response

# pylint: disable-next=too-few-public-methods
class Reflector:
    """
    A class that tracks agent thoughts and detects when the agent is stuck in its reasoning,
    providing guidance for next steps when no progress is being made.
    """

    def __init__(self, question: str, args):
        """
        Initialize the Reflector with a question and set up tracking for thoughts.

        Args:
            question (str): The question being answered.
        """
        self.question: str = question
        self.thoughts: List[str] = []  # Track all thoughts to detect patterns
        self.usage_metrics: Dict[str, int] = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }
        self.stuck_threshold: int = 2  # Number of similar thoughts before considering stuck
        self.args = args

    def reflect_on_step(self, reasoning_step: Dict[str, str]) -> bool:
        """
        Reflect on a single reasoning step, detect if the agent is stuck, and provide guidance.

        Args:
            reasoning_step (Dict[str, str]): A single step containing thought, actions, etc.

        Returns:
            bool: True if the agent is detected as stuck, False otherwise
        """
        try:
            return self._unsafe_reflect_on_step(reasoning_step)
        # pylint: disable-next=broad-except
        except Exception as e:
            Logger().error(f"Error during reflection: {e}")
            return False

    def _unsafe_reflect_on_step(self, reasoning_step: Dict[str, str]) -> bool:
        """
        Internal method to reflect on a reasoning step and detect stuck patterns.

        Args:
            reasoning_step (Dict[str, str]): A single step containing thought, actions, etc.

        Returns:
            Tuple[Optional[str], Dict[str, int]]: (feedback_if_stuck, usage_metrics)
        """
        filtered_reasoning_step = {
            k: v for k, v in reasoning_step.items() if v not in (None, "", [], {})}

        thought = filtered_reasoning_step.get('thought', '')
        final_answer = filtered_reasoning_step.get('final_answer')

        # Add thought to tracking list
        if thought:
            self.thoughts.append(thought)

        # If this is a final answer, don't provide stuck detection feedback
        if final_answer:
            return False

        # Check if agent is stuck
        if self._is_agent_stuck():
            Logger().debug(
                f"Agent detected as stuck for question: {self.question}")
            return True

        return False

    def _is_agent_stuck(self) -> bool:
        """
        Determine if the agent is stuck by analyzing recent thought patterns.

        Returns:
            bool: True if the agent appears to be stuck, False otherwise
        """
        if len(self.thoughts) < self.stuck_threshold:
            return False

        # Get the last few thoughts
        recent_thoughts = self.thoughts[-self.stuck_threshold:]

        # Check for repetitive patterns or lack of progress
        messages = [
            {
                "role": "system",
                "content": STUCK_DETECTION_PROMPT,
            },
            {
                "role": "user",
                "content": f'''Question: {self.question}

Recent thoughts:
{chr(10).join([f"{i+1}. {thought}" for i, thought in enumerate(recent_thoughts)])}'''
            }
        ]

        open_ai_request = {
            "custom_id": "stuck_detection",
            "model": self.args.model,
            "messages": messages,
            "temperature": default_job_args['temperature'],
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "stuck_detector",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_stuck": {"type": "boolean"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["is_stuck", "reasoning"],
                        "additionalProperties": False
                    }
                }
            }
        }

        try:
            result = chat_completions([open_ai_request])[0][0]
            content = result.choices[0].message.content.strip()

            # Update usage metrics
            self.usage_metrics["completion_tokens"] += result.usage.completion_tokens
            self.usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
            self.usage_metrics["total_tokens"] += result.usage.total_tokens

            structured_response = parse_structured_response(content)

            Logger().debug(
                f"Stuck detection for '{self.question}': {structured_response}")

            return structured_response.get('is_stuck', False)

        # pylint: disable-next=broad-except
        except Exception as e:
            Logger().error(f"Error in stuck detection: {e}")
            return False


# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

STUCK_DETECTION_PROMPT = '''
You are analyzing an agent's reasoning process to determine if it is stuck in a repetitive pattern without making progress.

An agent is considered "stuck" if:
1. It keeps searching for the same information repeatedly
2. It's not making meaningful progress toward answering the question
3. The thoughts show circular reasoning or repetitive searching patterns
4. It's unable to move forward despite having relevant information

An agent is NOT stuck if:
1. It's methodically breaking down a complex question into sub-questions
2. It's building upon previous information to get closer to an answer
3. Each search is exploring a different aspect of the question

Analyze the recent thoughts and determine if the agent is stuck in a non-productive loop.
'''
