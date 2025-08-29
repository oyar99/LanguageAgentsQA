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
) -> Optional[Tuple[str, str, Dict[str, int]]]:
    """
    Analyze why an answer is incorrect by comparing it to the expected answer and evidence.

    Args:
        question (str): The question being answered.
        expected_answer (str): The correct/expected answer.
        final_answer (str): The answer provided by the agent.
        messages (List[Dict[str, str]]): List of exchanges between QA agent and user.
        expected_evidences (List[str]): List of expected supporting evidence.
        missing_evidence (bool): Whether missing evidence should be included in analysis.

    Returns:
        Optional[Tuple[str, str, Dict[str, int]]]: A tuple containing explanation, category, and usage metrics.
        Returns None if analysis fails.
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

        # Extract usage metrics
        usage_metrics = {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        }

        structured_response = parse_structured_response(
            result.choices[0].message.content.strip())

        return (structured_response.get('explanation', None), 
                structured_response.get('category', 'Other'),
                usage_metrics)

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
agent's answer is incorrect. Your task is to identify the specific error.

Focus on connecting the evidence by identifying:

1. **Misinterpretation**: Quote the specific text the agent used incorrectly and explain how it should have been interpreted.
2. **Hallucination**: Quote any facts the agent invented that weren't in the retrieved text.
3. **Reasoning Error**: Show the logical error by comparing what the agent concluded vs. what the evidence actually supports.
'''

POST_REFLECTOR_MISSING_EVIDENCE_SECTION = '''4. **Missing Evidence**: Quote the key text that was missing from \
retrieval but available in the supporting evidence.
'''

POST_REFLECTOR_PROMPT_FOOTER = '''
Your explanation should be written as a self-reflective monologue, quoting the specific text pieces, connecting them to the error, and providing actionable suggestions for improvement.

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
"observations": [["No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, \
Leslie Perrins and Robert Cochran. The screenplay concerns a man who attempts to hide his friend for a month.", "Escape from L.A.: Escape from \
L.A. (also known as John Carpenter's Escape from L.A. or Escape from Los Angeles) is a 1996 American postapocalyptic action film co-written, \
co-scored, and directed by John Carpenter, co-written and produced by Debra Hill and Kurt Russell, with Russell also starring as Snake Plissken."]]\
}
system: {\
"thought": "Kurt Russell is the star of the 1996 film 'No Escape'. I need to find out who he is married to.", \
"actions": ["search('Kurt Russell spouse')"],
"observations": [["The Fox and the Hound: Mickey Rooney as Tod Kurt Russell as Copper Pearl Bailey as Big Mama Jack Albertson as Amos Slade Sandy \
Duncan as Vixey Jeanette Nolan as Widow Tweed Pat Buttram as Chief John Fiedler as The Porcupine John McIntire as The Badger Dick Bakalyan as \
Dinky Paul Winchell as Boomer Keith Coogan as Young Tod Corey Feldman as Young Copper"]]\
}
system: {\
"thought": "Kurt Russell is married to Goldie Hawn, so I will provide that as the final answer.", \
"final_answer": "Goldie Hawn"\
}

Your response should be:

{\
"explanation": "Looking at my reasoning process, I can see where I went wrong. When I searched for 'No Escape film cast', I retrieved information about \
both 'No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, Leslie Perrins and Robert Cochran' \
and 'Escape from L.A.: Escape from L.A. (also known as John Carpenter's Escape from L.A. or Escape from Los Angeles) is a 1996 American postapocalyptic \
action film co-written, co-scored, and directed by John Carpenter, co-written and produced by Debra Hill and Kurt Russell, with Russell also starring as \
Snake Plissken.' The problem was that I focused on Kurt Russell from 'Escape from L.A.' instead of recognizing that the question was asking about 'No \
Escape (1936 film)' which clearly starred Valerie Hobson. I should have paid closer attention to the exact film title and realized that 'Escape from L.A.' \
is a completely different movie from 'No Escape.' This misidentification led me down the wrong path entirely, causing me to search for Kurt Russell's marriage \
instead of Valerie Hobson's. In the future, I need to be more careful about matching entity names exactly and not confusing similar titles.",\
"category": "Misinterpretation"\
}
'''
