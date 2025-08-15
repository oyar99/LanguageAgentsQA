"""reflector.py
This module provides a Reflector class that evaluates the correctness of answers based on a 
given context and thought process.
"""
# pylint: disable=duplicate-code
from typing import Dict, Tuple

from azure_open_ai.chat_completions import chat_completions


def reflector(
        question: str,
        candidate_answer: str,
        thought_process: str,
) -> Tuple[str, Dict[str, int]]:
    """
    Reflects on the correctness of an answer based on the provided context and thought process.

    Args:
        question (str): The question being answered.
        candidate_answer (str): The answer to be evaluated.
        thought_process (str): The thought process leading to the answer.

    Returns:
        str: A string indicating whether the answer is correct or not, along with the reasoning.
    """
    feedback = ""
    usage_metrics = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    open_ai_request = {
        "custom_id": "relevance_doc",
        "model": 'gpt-4o-mini-2', # TODO: Use a reasoning model like Phi or similar for better results.
        "messages": [
            {
                "role": "system",
                "content": REFLECT_AGENT_PROMPT
            },
            {
                "role": "user",
                "content": f"Question: {question}\n"
                           f"Candidate Answer: {candidate_answer}\n"
                           f"Thought Process: {thought_process}"
            }
        ],
        "temperature": default_job_args['temperature'],
        "frequency_penalty": default_job_args['frequency_penalty'],
        "presence_penalty": default_job_args['presence_penalty'],
        "max_completion_tokens": 500,
    }

    result = chat_completions([open_ai_request])[0][0]

    feedback = result.choices[0].message.content.strip()
    usage_metrics["completion_tokens"] += result.usage.completion_tokens
    usage_metrics["prompt_tokens"] += result.usage.prompt_tokens
    usage_metrics["total_tokens"] += result.usage.total_tokens

    return feedback, usage_metrics


# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

REFLECT_AGENT_PROMPT = '''You are a helpful assistant that must judge whether a candidate answer is correct for \
a given question, based solely on the provided thought process and observations.
You must rely entirely on the context given and never use or suggest using outside or prior knowledge.

A candidate answer is correct if it is supported by the reasoning and observations.

If the answer is not stated explicitly, you may still mark it correct if it follows naturally from the provided observations using synonyms, \
clear equivalences, or logical inferences, including reasonable assumptions that a person, place, or role mentioned \
in the reasoning implies the answer.

If the observations contain facts that clearly allow answering the question through inference or directly, an answer of "N/A" or similar is incorrect.
If definitely no information is available to answer the question in the observations, then "N/A" is a totally acceptable answer and you must mark it as correct.

You must not critique style, length, or formatting of the candidate answer, and you must consider that the expected answer must be concise, and should
not include any additional commentary, explanations, or notes.

Your feedback must be concise, one sentence starting with either "The answer is correct because ..." or "The answer is not correct because ...".

## Examples

**Example 1 - Answer missing despite evidence**
Question: "Who is the child of Caroline LeRoy's spouse?"
Candidate Answer: "None"
Thought Process:

```json
[
    {
        "thought": "I need to find out who Caroline LeRoy's spouse is and then identify their child.",
        "actions": ["search('Caroline LeRoy spouse')"],
            "observations": [
                ["Caroline LeRoy:Caroline LeRoy Webster (September 28, 1797 in New York \
City \\u2013 February 26, 1882) was the second wife of 19th Century statesman Daniel Webster. \
Her father was Herman LeRoy, who was once head of the commercial house of Leroy, Bayard, McKiven & Co."]
        ]
    },
    {
        "thought": "I found that Caroline LeRoy was the second wife of Daniel Webster. I need to find out if they had any children together.",
            "actions": ["search('Daniel Webster children')"],
        "observations": [
            ["Fletcher Webster:Daniel Fletcher Webster, commonly known as \
Fletcher Webster (July 25, 1813 in Portsmouth, New Hampshire \\u2013 August \
30, 1862) was the son of renowned politician Daniel Webster and Grace Fletcher Webster."]
        ]
    },
    {
        "thought": "I have not found any information indicating that Caroline LeRoy and Daniel Webster had children together. Therefore, I conclude that there is no child of Caroline LeRoy's spouse.",
        "final_answer": "None"
    }
]
```

Output:

The answer is not correct because Daniel Webster, Caroline LeRoy's spouse, had a child named Fletcher Webster. \
The question asks for the child of Caroline LeRoy's spouse, not whether she had children herself, and the observations clearly indicate that Daniel Webster had a child. \
Therefore, the candidate answer "None" is incorrect.

**Example 2 - Answer not supported by facts**
Question: "What award did the author of The Red Tree receive?"
Candidate Answer: "Astrid Lindgren Memorial Award"
Thought Process:

```json
[
    {
        "thought": "I need to find out who the the author of The Red Tree is.",
        "actions": ["search('The Red Tree author')"],
        "observations": [
            ["The Red Tree (Shaun Tan):The Red Tree (2001), written and illustrated by Shaun Tan, is a picture book \
that presents a fragmented journey through a dark world"]
        ]
    },
    {
        "thought": "I found that the author of The Red Tree is Shaun Tan, now I need to find what awards they have received.",
        "actions": ["search('Shan Tan awards')"],
        "observations": [
            ["Passion Australia produced \\"The Lost Thing\\", directed by Andrew Ruhemann and Shaun Tan, which won an Academy \
Award for Best Animated Short Film in 2011."]
    }
    },
    {
        "thought": "Shaun Tan has received multiple awards, including the prestigious Astrid Lindgren Memorial Award in 2011.",
        "final_answer": "Astrid Lindgren Memorial Award"
    }
]
```

Output:

The answer is not correct because there's no mention of Shaun Tan receiving the Astrid Lindgren Memorial Award in the provided observations, \
only that they won an Academy Award for Best Animated Short Film. Therefore, the candidate answer "Astrid Lindgren Memorial Award" is not supported by the facts.
"""
'''
