"""reflector.py
This module provides a Reflector class that evaluates the correctness of answers based on a 
given context and thought process.
"""
# pylint: disable=duplicate-code
import json
from string import Template
from typing import Dict, List, Optional, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from utils.structure_response import parse_structured_response


class Reflector:
    """
    A class that tracks messages sent to chat_completions and evaluates reasoning correctness.
    """

    def __init__(self, question: str):
        """
        Initialize the Reflector with a question and set up initial system message.

        Args:
            question (str): The question being answered.
        """
        self.question = question
        self.usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

        # Initialize messages with system instructions
        template = Template(REFLECT_AGENT_PROMPT_V3)
        self.messages = [
            {
                "role": "system",
                "content": template.substitute(question=question)
            }
        ]

    def reflect_on_chain(self, reasoning_chain: List[Dict[str, str]]) -> Optional[str]:
        """
        Reflect on a chain of reasoning steps, stopping at the first incorrect step.

        Args:
            reasoning_chain (List[Dict[str, str]]): List of reasoning steps to evaluate.

        Returns:
            Optional[str]: Feedback for the first incorrect step, or None if all steps are correct.
        """
        for step in reasoning_chain:
            feedback, is_correct = self.reflect_on_step(step)
            if not is_correct:
                return feedback
        return None

    def reflect_on_step(self, reasoning_step: Dict[str, str]) -> Tuple[Optional[str], bool]:
        """
        Reflect on a single reasoning step and determine if it's correct.

        Args:
            reasoning_step (Dict[str, str]): A single step containing thought, actions, observations, etc.

        Returns:
            Tuple[Optional[str], bool]: (feedback_if_incorrect, is_correct)
        """
        filtered_reasoning_step = {k: v for k, v in reasoning_step.items() if v not in (None, "", [], {})}

        # Add user message with the reasoning step
        self.messages.append({
            "role": "user",
            "content": json.dumps(filtered_reasoning_step)
        })

        # Create request
        open_ai_request = {
            "custom_id": "reflector_step",
            "model": 'gpt-4o-mini-2',
            "messages": self.messages,
            "temperature": default_job_args['temperature'],
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
        }

        Logger().debug(
            f"Reflector request for question {self.question}: {open_ai_request}")

        # Send request and track usage
        result = chat_completions([open_ai_request])[0][0]

        self.usage_metrics["completion_tokens"] += result.usage.completion_tokens
        self.usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
        self.usage_metrics["total_tokens"] += result.usage.total_tokens

        content = result.choices[0].message.content.strip()

        # Add assistant response to messages
        self.messages.append({
            "role": "assistant",
            "content": content
        })

        Logger().debug(
            f"Reflector response for question '{self.question}': {content}")

        # Parse the structured response
        structured_response = parse_structured_response(content)
        correctness = str(structured_response.get('correctness', ''))

        if correctness.lower() == 'incorrect':
            Logger().debug(
                f"Reflector determined the reasoning is incorrect for question '{self.question}'")
            return structured_response.get('thought', None), False

        return None, True

    def update_observations(self, observations: List[List[str]]) -> None:
        """
        Update the observations in the reflector.

        Args:
            observations (List[List[str]]): List of observations to update.
        """
        # Add the observations as a system message
        self.messages.append({
            "role": "user",
            "content": json.dumps({
                "Observations": observations
            })
        })

    def get_usage_metrics(self) -> Dict[str, int]:
        """
        Get the accumulated usage metrics.

        Returns:
            Dict[str, int]: Usage metrics with completion_tokens, prompt_tokens, total_tokens
        """
        return self.usage_metrics.copy()


# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

REFLECT_AGENT_PROMPT_V3 = '''You are a reasoning process evaluator.
Your task is to judge whether a reasoning step is logically correct, and will likely lead to a correct answer for the given question.
Never use outside knowledge. Never assume facts unless they are explicitly stated or unambiguously implied by the context.

You will first be given a question, and then you will be provided with the reasoning steps that can help answer the question.

The reasoning steps have the following structure.

```json
{
    "thought": "<Brief explanation of the reasoning step>",
    "actions": ["<List of actions to be taken supported by the reasoning>"],
    "final_answer": "<Final answer if enough information is available>"
}
```

During each step, you will analyze the reasoning step following these steps in order carefully:

1. Only if a reasoning step contains a claim or fact, verify it is supported either explicitely or implicitely by an observation. \
To satisfy this, you must cite the observation that supports the claim in your response, ensuring entities from the \
observations match with those in the claim.

2. If an action is provided, ensure they are relevant to the question and logically follow from the reasoning step. \

3. If a final answer is provided, ensure it is supported by the reasoning step and observations similarly to the first step. \


### OUTPUT FORMAT

During each reasoning step, you must respond in valid JSON format:

```json
{
  "thought": "<Brief explanation, citing exact quotes from observations for each factual claim. State if any claim is unsupported or ambiguous.>",
  "cite": ["<List of observations that support the claims made in the thought>"],
  "correctness": "correct" | "incorrect"
}
```

### EXAMPLES

**Example 1**

Question: "Who is the child of Caroline LeRoy's spouse?"

Iteration 1:
```json
{
    "thought": "I need to find out who Caroline LeRoy's spouse is and then identify their child.",
    "actions": ["search('Caroline LeRoy spouse')"]
}
```

You must respond with:

```json
{
    "thought": "Finding out who Caroline LeRoy's spouse is a logical action to take next.",
    "cite": [],
    "correctness": "correct"
}
```

Iteration 2:
```json
{
    "observations": [
        ["Caroline LeRoy Webster (September 28, 1797 in New York City – February 26, 1882) was the second wife of 19th Century statesman Daniel Webster. \
Her father was Herman LeRoy, who was once head of the commercial house of Leroy, Bayard, McKiven & Co."]]
}
```

```json
{
    "thought": "I found that Caroline LeRoy was the second wife of Daniel Webster. I need to find out if they had any children together.",
    "actions": ["search('Daniel Webster children')"]
}
```

You must respond with:

```json
{
    "thought": "Daniel Webster is indeed the husband of Caroline LeRoy as stated in the observations: "Caroline LeRoy Webster \
(September 28, 1797 in New York City – February 26, 1882) was the second wife of 19th Century statesman Daniel Webster". \
However, assuming they had any children together is not relevant to the question, as the question asks for the child of \
Daniel Webster, not whether she had children herself.",
    "cite": ["Caroline LeRoy Webster (September 28, 1797 in New York City – February 26, 1882) was the second wife of 19th Century statesman Daniel Webster."],
    "correctness": "incorrect"
}
```

Here is the question you will judge reasoning steps to arrive at an answer for:

$question
'''