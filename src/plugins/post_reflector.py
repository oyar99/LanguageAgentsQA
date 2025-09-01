"""post_reflector.py
This module provides functionality to analyze why an answer is incorrect by comparing
it to expected answers and evidence.
"""
from typing import Dict, List, Optional, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger


def post_reflector(
        question: str,
        expected_answer: str,
        expected_evidences: List[str],
        decomposition: List[dict] = None
) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Analyze why an answer is incorrect by comparing it to the expected answer and evidence.

    Args:
        question (str): The question being answered.
        expected_answer (str): The correct/expected answer.
        expected_evidences (List[str]): List of expected supporting evidence.
        decomposition (List[dict], optional): Question decomposition steps for grounding.

    Returns:
        Optional[Tuple[str, Dict[str, int]]]: A tuple containing explanation, and usage metrics.
        Returns None if analysis fails.
    """

    decomposition = decomposition or []

    # Format expected evidences
    formatted_evidences = "\n".join([
        f"- {evidence}" for evidence in expected_evidences
    ])

    # Format decomposition if available
    decomposition_text = ""
    if decomposition:
        decomposition_steps = []
        for i, step in enumerate(decomposition, 1):
            if 'question' in step and 'answer' in step:
                # MuSiQue style decomposition
                decomposition_steps.append(
                    f"{i}. {step['question']} → {step['answer']}")
            elif 'title' in step:
                # Hotpot style supporting facts
                decomposition_steps.append(
                    f"{i}. Supporting fact from: {step['title']}")

        if decomposition_steps:
            decomposition_text = "Reasoning Steps:\n" + \
                "\n".join(decomposition_steps) + "\n"

    prompt = POST_REFLECTOR_PROMPT_BASE

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
                f"Supporting Evidence:\n{formatted_evidences}\n"
                f"{decomposition_text}\n"
            }
        ],
        "temperature": default_job_args['temperature'],
        "frequency_penalty": default_job_args['frequency_penalty'],
        "presence_penalty": default_job_args['presence_penalty'],
        "max_completion_tokens": 1000
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

        # Extract the reasoning chain directly from the response
        correct_reasoning_chain = result.choices[0].message.content.strip()

        return (correct_reasoning_chain, usage_metrics)
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
You will be provided with a question, the expected answer, supporting evidence, and reasoning steps. \
Your task is to demonstrate the proper thought process and search actions that lead to the correct answer.

Use reasoning steps as a guide to structure your search queries and reasoning flow. \
Each reasoning step should typically correspond to one or more search actions in your reasoning chain.

Generate a reasoning chain that strictly follows this format:

**Iteration N:**
```json
{
    "thought": "Clear reasoning about what information is needed",
    "actions": ["search('specific query')"]
}
```

**Iteration N+1:**
```json
{
    "thought": "Clear reasoning about what information is needed",
    "actions": ["search('specific query')"],
    "observations": [["Retrieved document content that contains the needed information"]]
}
```

**Iteration N+2:**
... (repeat as necessary, alternating between thought/action and thought/action/observation) ...

**Final Iteration:**
```json
{
    "thought": "Clear reasoning showing how the evidence leads to the correct conclusion",
    "final_answer": "The correct answer"
}
```

Each iteration must strictly adhere to the following rules:

- Thoughts can ONLY refer to information from previous observations from prior iterations, never directly from the provided supporting evidence
- Observations must contain EXACT text snippets from the supporting evidence provided
- Actions and observations must never be empty when present
- Each intermediate iteration follows the exact same thought+actions, followed by another iteration with observations added

An iteration is either a thought/action pair, a thought/action/observation triplet, or the final thought/final_answer pair.
When provided, actions and observations must not be empty lists.
Observations must be strings from the actual provided supporting evidence.

### Example

Question: "Who was married to the star of No Escape?"
Expected Answer: "John Profumo"
Supporting Evidence:
- No Escape (1936 film): No Escape is a 1936 British thriller film directed by Norman Lee and starring Valerie Hobson, Leslie Perrins and Robert Cochran. \
The screenplay concerns a man who attempts to hide his friend for a month.
- Profumo was born in London, the son of former British government minister John Profumo and his wife, actress Valerie Hobson. The Profumo family is of \
Italian origin. David Profumo would have succeeded his father as 6th Baron Profumo in the nobility of the Kingdom of Sardinia, if the nobility of Italy \
had not been legally abolished after WW2. Like his father, he does not use this title.
Reasoning Steps: 

No Escape >> cast member → Valerie Hobson
#1 >> spouse → John Profumo

Your response:

**Iteration 1:**
```json
{
    "thought": "I need to find information about the film 'No Escape' and identify its star to determine who they were married to.",
    "actions": ["search('No Escape film')"]
}
```

**Iteration 2:**
```json
{
    "thought": "I need to find information about the film 'No Escape' and identify its star to determine who they were married to.",
    "actions": ["search('No Escape film')"],
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
    "actions": ["search('Valerie Hobson husband')"]
}
```

**Iteration 4:**
```json
{
    "thought": "I can see two different films in the results. There's 'No Escape (1936 film)' starring Valerie Hobson, Leslie Perrins and Robert Cochran, and 'Escape from L.A.' starring Kurt Russell. The question asks about 'No Escape', so I should focus on the 1936 film. Valerie Hobson appears to be the main star. Now I need to find who was married to Valerie Hobson.",
    "actions": ["search('Valerie Hobson husband')"],
    "observations": [["Profumo was born in London, the son of former British government minister John Profumo and his wife, actress Valerie Hobson. The Profumo family is of Italian origin. David Profumo would have succeeded his father as 6th Baron Profumo in the nobility of the Kingdom of Sardinia, if the nobility of Italy had not been legally abolished after WW2. Like his father, he does not use this title."]]
}
```

**Iteration 5:**
```json
{
    "thought": "The evidence shows that Valerie Hobson was married to John Profumo. Since Valerie Hobson was the star of 'No Escape' (1936 film), the answer to who was married to the star of No Escape is John Profumo.",
    "final_answer": "John Profumo"
}
```
'''
