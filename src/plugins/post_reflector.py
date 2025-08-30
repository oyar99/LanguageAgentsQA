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

    # Use template to inject the correct reasoning chain
    prompt += POST_REFLECTOR_PROMPT_FOOTER.format(
        correct_reasoning_chain=CORRECT_REASONING_CHAIN)

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
        "max_completion_tokens": 1000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "post_reflection_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": categories},
                        "correct_reasoning_chain": {"type": "string"},
                    },
                    "required": ["category", "correct_reasoning_chain"],
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

        return (structured_response.get('correct_reasoning_chain', None),
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

POST_REFLECTOR_PROMPT_BASE = '''You are an expert analysis assistant that creates correct reasoning chains. \
You will be provided with a question, the expected answer, the supporting evidence, and the answer provided \
by a different QA agent along with its reasoning. \
Your task is to demonstrate the proper thought process and search actions that would have led to the correct answer.

Analyze the agent's incorrect reasoning and generate a corrected chain that strictly follows this format:

**Iteration N:**
```json
{
    "thought": "Clear reasoning about what information is needed",
    "actions": ["search('specific query', depth)"]
}
```

**Iteration N+1:**
```json
{
    "thought": "Clear reasoning about what information is needed",
    "actions": ["search('specific query', depth)"],
    "observations": [["Retrieved document content that contains the needed information"]]
}
```

**Final Iteration:**
```json
{
    "thought": "Clear reasoning showing how the evidence leads to the correct conclusion",
    "final_answer": "The correct answer"
}
```

An iteration is either a thought/action pair, a thought/action/observation triplet, or the final thought/final_answer pair.
When provided, actions and observations must not be empty lists.
Observations must be strings from the actual provided supporting evidence.

Identify the error category:
1. **Misinterpretation**: Agent misunderstood retrieved text
2. **Hallucination**: Agent invented facts not in retrieved text  
3. **Reasoning Error**: Agent made logical errors in connecting evidence
'''

POST_REFLECTOR_MISSING_EVIDENCE_SECTION = '''4. **Missing Evidence**: Agent failed to retrieve key supporting evidence
'''

POST_REFLECTOR_PROMPT_FOOTER = '''
Generate a correct reasoning chain that would lead to the right answer, using the supporting evidence provided.

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
system: {{\
"thought": "I need to find the correct film titled 'No Escape' to identify its star and then look for their marital status.", \
"actions": ["search('No Escape film cast', 1)"],\
"observations": [["No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, \
Leslie Perrins and Robert Cochran. The screenplay concerns a man who attempts to hide his friend for a month.", "Escape from L.A.: Escape from \
L.A. (also known as John Carpenter's Escape from L.A. or Escape from Los Angeles) is a 1996 American postapocalyptic action film co-written, \
co-scored, and directed by John Carpenter, co-written and produced by Debra Hill and Kurt Russell, with Russell also starring as Snake Plissken."]]\
}}
system: {{\
"thought": "Kurt Russell is the star of the 1996 film 'No Escape'. I need to find out who he is married to.", \
"actions": ["search('Kurt Russell spouse', 1)"],
"observations": [["The Fox and the Hound: Mickey Rooney as Tod Kurt Russell as Copper Pearl Bailey as Big Mama Jack Albertson as Amos Slade Sandy \
Duncan as Vixey Jeanette Nolan as Widow Tweed Pat Buttram as Chief John Fiedler as The Porcupine John McIntire as The Badger Dick Bakalyan as \
Dinky Paul Winchell as Boomer Keith Coogan as Young Tod Corey Feldman as Young Copper"]]\
}}
system: {{\
"thought": "Kurt Russell is married to Goldie Hawn, so I will provide that as the final answer.", \
"final_answer": "Goldie Hawn"\
}}

Your response should be:

{{
"correct_reasoning_chain": "{correct_reasoning_chain}",
"category": "Misinterpretation"
}}
'''

CORRECT_REASONING_CHAIN = """**Iteration 1:**
```json
{
    "thought": "I need to find information about the film 'No Escape' and identify its star to determine who they were married to.",
    "actions": ["search('No Escape film cast', 1)"]
}
```

**Iteration 2:**
```json
{
    "thought": "I need to find information about the film 'No Escape' and identify its star to determine who they were married to.",
    "actions": ["search('No Escape film cast', 1)"],
    "observations": [["No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, \
Leslie Perrins and Robert Cochran. The screenplay concerns a man who attempts to hide his friend for a month.", "Escape from L.A.: Escape from \
L.A. (also known as John Carpenter's Escape from L.A. or Escape from Los Angeles) is a 1996 American postapocalyptic action film co-written, co-scored, and \
directed by John Carpenter, co-written and produced by Debra Hill and Kurt Russell, with Russell also starring as Snake Plissken."]]
}
```

**Iteration 3:**
```json
{
    "thought": "I can see two different films in the results. There's 'No Escape (1936 film)' starring Valerie Hobson, Leslie Perrins and Robert Cochran, and \
'Escape from L.A.' starring Kurt Russell. The question asks about 'No Escape', so I should focus on the 1936 film. Valerie Hobson appears to be the main star. \
Now I need to find who was married to Valerie Hobson.",
    "actions": ["search('Valerie Hobson husband', 1)"]
}
```

**Iteration 4:**
```json
{
    "thought": "I can see two different films in the results. There's 'No Escape (1936 film)' starring Valerie Hobson, Leslie Perrins and Robert Cochran, and 'Escape from L.A.' starring Kurt Russell. The question asks about 'No Escape', so I should focus on the 1936 film. Valerie Hobson appears to be the main star. Now I need to find who was married to Valerie Hobson.",
    "actions": ["search('Valerie Hobson husband', 1)"],
    "observations": [["Profumo was born in London, the son of former British government minister John Profumo and his wife, actress Valerie Hobson. The Profumo family is of Italian origin. David Profumo would have succeeded his father as 6th Baron Profumo in the nobility of the Kingdom of Sardinia, if the nobility of Italy had not been legally abolished after WW2. Like his father, he does not use this title."]]
}
```

**Iteration 5:**
```json
{
    "thought": "The evidence shows that Valerie Hobson was married to John Profumo. Since Valerie Hobson was the star of 'No Escape' (1936 film), the answer to who was married to the star of No Escape is John Profumo.",
    "final_answer": "John Profumo"
}
```"""
