"""post_reflector.py
This module provides functionality to analyze why an answer is incorrect by comparing
it to expected answers and evidence.
"""
from typing import Dict, List, Optional, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from utils.structure_response import parse_structured_response


def post_reflector(
        question: str,
        expected_answer: str,
        final_answer: str,
        messages: List[Dict[str, str]],
        expected_evidences: List[str],
        missing_evidence: bool = True
) -> Optional[Tuple[str, str]]:
    """
    Analyze why an answer is incorrect by comparing it to the expected answer and evidence.

    Args:
        question (str): The question being answered.
        expected_answer (str): The correct/expected answer.
        final_answer (str): The answer provided by the agent.
        messages (List[Dict[str, str]]): List of exchanges between QA agent and user.
        expected_evidences (List[str]): List of expected supporting evidence.

    Returns:
        str: A short explanation of why the answer is incorrect or doesn't match expected_answer.
    """
    # Format the conversation history excluding the system prompt
    conversation_history = "\n".join([
        f"{msg.get('role')}: {msg.get('content')}"
        for msg in messages[1:]
    ])

    # Format expected evidences
    formatted_evidences = "\n".join([
        f"- {evidence}" for evidence in expected_evidences
    ])

    # Build the prompt dynamically based on missing_evidence parameter
    prompt = POST_REFLECTOR_PROMPT_BASE
    if missing_evidence:
        prompt += POST_REFLECTOR_MISSING_EVIDENCE_SECTION
    prompt += POST_REFLECTOR_PROMPT_FOOTER

    # Build categories list dynamically
    categories = [
        "Misinterpretation",
        "Hallucination",
        "Reasoning Error"
    ]
    if missing_evidence:
        categories.append("Missing Evidence")

    open_ai_request = {
        "custom_id": "post_reflection_analysis",
        "model": 'gpt-4o-mini-2',
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content":
                f"Question: \"{question}\"\n"
                f"Expected Answer: \"{expected_answer}\"\n"
                f"Supporting Evidence:\n{formatted_evidences}\n\n"
                f"Agent's Answer: \"{final_answer}\"\n"
                f"Agent's Reasoning:\n{conversation_history}"
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
                "name": "post_reflection_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": categories},
                        "explanation": {"type": "string"},
                    },
                    "required": ["category", "explanation"],
                    "additionalProperties": False
                }
            }
        }
    }

    Logger().debug(
        f"Post-reflector request for question '{question}': {open_ai_request}")

    try:
        result = chat_completions([open_ai_request])[0][0]

        Logger().debug(
            f"Post-reflection response: {result.choices[0].message.content.strip()}")

        structured_response = parse_structured_response(
            result.choices[0].message.content.strip())

        return (structured_response.get('explanation', None), structured_response.get('category', 'Other'))

    # pylint: disable=broad-except
    except Exception as e:
        Logger().error(
            f"Encountered error while analyzing incorrect answer: {e}")
        return None


# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

POST_REFLECTOR_PROMPT_BASE = '''You are an expert analysis assistant that evaluates why a QA \
agent's answer is incorrect or doesn't match the expected answer. Your task is to identify the \
specific reasoning error or misinterpretation that led to the incorrect response.

Analyze the agent's reasoning process through the conversation history and compare it against the expected answer \
and supporting evidence. Focus on identifying:

1. **Misinterpretation**: Did the agent misunderstand the question or misinterpret part of the evidence?
    For example, confusing two similar-sounding entities or mixing up details from different sources.
2. **Hallucination**: Did the agent use not explicitly available information to produce its answer?
    For example, inventing facts or details that were not present in the provided evidence.
3. **Reasoning Error**: Did the agent make logical mistakes in connecting evidence to conclusions?
    For example, drawing incorrect inferences or mathematical errors.
'''

POST_REFLECTOR_MISSING_EVIDENCE_SECTION = '''4. **Missing Evidence**: \
Did the agent fail to retrieve relevant evidence needed to answer the question?
Only consider this error when all previous categories do not apply.
'''

POST_REFLECTOR_PROMPT_FOOTER = '''

Provide a clear concrete explanation of the error that led to the wrong answer.

### Example

Question: "Who was married to the star of No Escape?"
Expected Answer: "John Profumo"
Supporting Evidence:
- No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, Leslie Perrins and Robert Cochran. \
The screenplay concerns a man who attempts to hide his friend for a month.
- Profumo was born in London, the son of former British government minister John Profumo and his wife, actress Valerie Hobson. The Profumo family is of \
Italian origin. David Profumo would have succeeded his father as 6th Baron Profumo in the nobility of the Kingdom of Sardinia, if the nobility of Italy \
had not been legally abolished after WW2. Like his father, he does not use this title.

Agent's Answer: "Goldie Hawn"
Agent's Reasoning:

user: Who was married to the star of No Escape?
system: {\
"thought": "I need to find the correct film titled 'No Escape' to identify its star and then look for their marital status.", \
"actions": ["search('No Escape film cast')"],\
"observations": [["No Escape (1936 film):No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, \
Leslie Perrins and Robert Cochran. The screenplay concerns a man who attempts to hide his friend for a month.", "Escape from L.A.:Escape from \
L.A. (also known as John Carpenter's Escape from L.A. or Escape from Los Angeles) is a 1996 American postapocalyptic action film co-written, \
co-scored, and directed by John Carpenter, co-written and produced by Debra Hill and Kurt Russell, with Russell also starring as Snake Plissken."]]\
}
system: {\
"thought": "Kurt Russell is the star of the 1996 film 'No Escape'. I need to find out who he is married to.", \
"actions": ["search('Kurt Russell spouse')"],
"observations": [["The Fox and the Hound:Mickey Rooney as Tod Kurt Russell as Copper Pearl Bailey as Big Mama Jack Albertson as Amos Slade Sandy \
Duncan as Vixey Jeanette Nolan as Widow Tweed Pat Buttram as Chief John Fiedler as The Porcupine John McIntire as The Badger Dick Bakalyan as \
Dinky Paul Winchell as Boomer Keith Coogan as Young Tod Corey Feldman as Young Copper"]]\
}
system: {\
"thought": "Kurt Russell is married to Goldie Hawn, so I will provide that as the final answer.", \
"final_answer": "Goldie Hawn"\
}

Your response should be:

{\
"explanation": "The agent confused the film 'Escape from L.A' with the film 'No Escape (1936 film)'. \
As a result, the agent was not able to retrieve other relevant evidences to answer the question.",\
"category": "Misinterpretation"\
}
'''
