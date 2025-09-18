"""reflector.py
This module provides a Reflector class that evaluates the correctness of answers based on a 
given context and thought process.
"""
# pylint: disable=duplicate-code
from typing import Dict, Optional, Tuple

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
        self.observations = set()
        self.usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0
        }

    def reflect_on_step(self, reasoning_step: Dict[str, str]) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Reflect on a single reasoning step and determine if it's correct.
        """
        try:
            return self.unsafe_reflect_on_step(reasoning_step)
        # pylint: disable-next=broad-except
        except Exception as e:
            Logger().error(f"Error during reflection: {e}")
            return None, {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        
    def reflect_on_answer(self, answer: str) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Reflect on the final answer and determine if it's correct.

        Args:
            answer (str): The final answer to evaluate.

        Returns:
            Optional[str]: feedback
        """
        open_ai_request = {
            "custom_id": "fact_checking_final_answer",
            "model": 'gpt-4.1-mini',
            "messages": [
                {
                    "role": "system",
                    "content": REFLECT_ANSWER_PROMPT,
                },
                {
                    "role": "user",
                    "content": f'Question: "{self.question}"\nFinal Answer: "{answer}"'
                }
            ],
            "temperature": default_job_args['temperature'],
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "reflector",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                            "category": {"type": "string", "enum": ["CORRECT", "INCORRECT"]},
                        },
                        "required": ["analysis", "category"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        result = chat_completions([open_ai_request])[0][0]
        content = result.choices[0].message.content.strip()

        Logger().debug(
            f"""Reflector final answer response for question '{self.question}':
{content}""")
        
        result_structured = parse_structured_response(content)

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens,
        }

        if result_structured['category'] == 'CORRECT':
            return None, usage_metrics
        
        feedback = result_structured['analysis']
        return feedback, usage_metrics
        
    def reflect_on_qa_understanding(self, thought: str) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Reflect on whether the thought correctly understands the question.

        Args:
            thought (str): The thought process to evaluate.

        Returns:
            Optional[str]: feedback
        """
        open_ai_request = {
            "custom_id": "fact_checking_qa_understanding",
            "model": 'gpt-4.1-mini',
            "messages": [
                {
                    "role": "system",
                    "content": REFLECT_QA_UNDERSTANDING_PROMPT,
                },
                {
                    "role": "user",
                    "content": f'Input: "{self.question}"\nThought: "{thought}"'
                }
            ],
            "temperature": default_job_args['temperature'],
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "reflector",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                            "category": {"type": "string", "enum": ["CORRECT", "INCORRECT"]},
                        },
                        "required": ["analysis", "category"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        result = chat_completions([open_ai_request])[0][0]
        content = result.choices[0].message.content.strip()

        Logger().debug(
            f"""Reflector QA understanding response for question '{self.question}':
{content}""")

        result_structured = parse_structured_response(content)

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens,
        }

        if result_structured['category'] == 'CORRECT':
            return None, usage_metrics
        
        feedback = result_structured['analysis']
        return feedback, usage_metrics

    def unsafe_reflect_on_step(self, reasoning_step: Dict[str, str]) -> Tuple[Optional[str], Dict[str, int]]:
        """
        Reflect on a single reasoning step and determine if it's correct.

        Args:
            reasoning_step (Dict[str, str]): A single step containing thought, actions, observations, etc.

        Returns:
            Optional[str]: feedback
        """
        filtered_reasoning_step = {k: v for k, v in reasoning_step.items() if v not in (None, "", [], {})}

        thought = filtered_reasoning_step.get('thought')
        observations = filtered_reasoning_step.get('observations', [])
        final_answer = filtered_reasoning_step.get('final_answer')

        if not observations:
            return self.reflect_on_qa_understanding(thought)
        
        if final_answer:
            return self.reflect_on_answer(final_answer)

        # Flatten observations list of lists into a single text
        flattened_observations = []
        for obs_list in observations:
            flattened_observations.extend(obs_list)

        self.observations.update(flattened_observations)

        observations_text = '\n'.join(self.observations)

        messages = [
            {
                "role": "system",
                "content": REFLECT_PROMPT,
            },
            {
                "role": "user",
                "content": f'''## Observations
{observations_text}
## Thought
{thought}
'''
            }
        ]

        # Create request
        open_ai_request = {
            "custom_id": "reflector_step",
            "model": 'gpt-4.1-mini',
            "messages": messages,
            "temperature": default_job_args['temperature'],
            "frequency_penalty": default_job_args['frequency_penalty'],
            "presence_penalty": default_job_args['presence_penalty'],
            "max_completion_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "reflector",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                        },
                        "required": ["analysis"],
                        "additionalProperties": False
                    }
                }
            }
        }

        Logger().debug(
            f"Reflector request for question {self.question}: {messages}")

        # Send request and track usage
        result = chat_completions([open_ai_request])[0][0]

        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens,
        }

        content = result.choices[0].message.content.strip()

        Logger().debug(
            f"""Reflector response for question '{self.question}':
{content}""")

        feedback = parse_structured_response(content)

        final_feedback = (
            f"Please consider this feedback when thinking about next steps: {feedback['analysis']}\n" 
            if feedback['analysis'] != "N/A" else None
        )

        return final_feedback, usage_metrics


# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

REFLECT_PROMPT = '''
You are given a thought and a set of documents. Determine if the documents already contain any information relevant to what the thought is seeking.

Relevant information may be:

- Explicitly stated, OR
- Indirectly available through logical inference or assumptions.

For example: if a thought claims that no information is known about someone's birthplace, but the documents mention where they went to school, \
it is a plausible assumption that they were born in that place if no other information mentions otherwise.

Always provide a brief analysis explaining your reasoning, and possible assumptions that could be made to answer the thought with the given documents.
Only if definitely no relevant information is available, respond with "N/A".

use only the provided documents. Do not rely on external or prior knowledge.

### Example

## Observations:

Riverside Plaza:Riverside Plaza is a modernist and brutalist apartment complex designed by Ralph Rapson that opened in Minneapolis, Minnesota in 1973

## Thought:

I need to find information that clearly states where Ralph Rapson died to determine the relevant city for the treaty inquiry.

Output:

The thought suggests that it is not known where Ralph Rapson died. However, the documents mention that he designed Riverside Plaza in Minneapolis, Minnesota. \
While this does not explicitly state where he died, it is a plausible inference that he died in Minneapolis because no other document says otherwise. \
Therefore, the thought is INCORRECT because there is information in the documents that could reasonably be used to address it.
'''

REFLECT_ANSWER_PROMPT = '''
You are given a question, and a final answer.
Your task is to evaluate whether the final answer addresses the question.

If the answer is N/A, that should be considered CORRECT.

You should label the answer as either CORRECT or INCORRECT.

Do not use any external or prior knowledge, only whether the answer could correspond to the question based on its content.
'''

REFLECT_QA_UNDERSTANDING_PROMPT = '''
You are given a question and a thought that attempts to break down the question into smaller steps. \
Your task is to evaluate whether the thought correctly understands the question. \
If the thought misunderstands the question, or makes incorrect assumptions about the question, \
identify the specific parts of the thought that are incorrect and explain why. \

Input: "Whose navigator father explored the east coast of the region where Ignacio Esparza was later born?"
Thought: "I need to find out who Ignacio Esparza is and what region he was born in. Then, I will look for information about \
his navigator father\'s explorations of the east coast of that region."

Output: The thought incorrectly assumes that the question is asking about Ignacio Esparza's father. \
Instead, the question is asking about the father of a navigator who explored the east coast of the region where Ignacio Esparza was born. \
'''