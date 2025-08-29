"""search_pruner.py
This module provides functionality to prune search results based on a relevance threshold.
"""
from typing import Any, Dict, List, Tuple

from azure_open_ai.chat_completions import chat_completions
from logger.logger import Logger
from utils.structure_response import parse_structured_response


def search_pruner(
        query: str,
        documents: List[Dict[str, Any]],
        thought: str,
        threshold=30.0
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Prune search results based on a relevance threshold if the result is relevant to the query and thought.

    Args:
        query (str): The search query.
        documents (list): List of document tuples (doc_id, content).
        thought (str): The thought process or context for relevance evaluation.
        threshold (float): Relevance threshold for pruning.

    Returns:
        list: Pruned list of document IDs that meet the relevance threshold.
    """
    pruned_results = []
    usage_metrics = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    for doc in documents:
        relevance_score, usage_metrics_loc = _calculate_relevance(
            query, doc['content'], thought)

        usage_metrics["completion_tokens"] += usage_metrics_loc["completion_tokens"]
        usage_metrics["prompt_tokens"] += usage_metrics_loc["prompt_tokens"]
        usage_metrics["total_tokens"] += usage_metrics_loc["total_tokens"]

        if relevance_score >= threshold:
            pruned_results.append(doc)

    return (pruned_results, usage_metrics)


def _calculate_relevance(query: str, content: str, thought: str) -> Tuple[float, Dict[str, int]]:
    """
    Calculate a relevance score for the given query and content using an LLM agent.

    Args:
        query (str): The search query.
        content (str): The content of the document.
        thought (str): The thought process or context for relevance evaluation.
    """

    open_ai_request = {
        "custom_id": "relevance_doc",
        "model": 'gpt-4o-mini-2',
        "messages": [
            {
                "role": "system",
                "content": RELEVANCE_AGENT_PROMPT.format(content=content, thought=thought)
            },
            {
                "role": "user",
                "content": f"Document: \"{content}\"\nThought: \"{thought}\"\n\nQuery: \"{query}\"\n\n"
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
                "name": "relevance_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "explanation": {"type": "string"},
                    },
                    "required": ["relevance_score", "explanation"],
                    "additionalProperties": False
                }
            }
        }
    }

    try:
        result = chat_completions([open_ai_request])[0][0]

        Logger().debug(f"Relevance response: {result.choices[0].message.content.strip()}")

        structured_response = parse_structured_response(result.choices[0].message.content.strip())

        score = int(structured_response.get('relevance_score', 100))
    # pylint: disable=broad-except
    except Exception as e:
        Logger().error(f"Encountered error while processing document score: {e}")
        # Assuming invalid scores should be treated as relevant
        return (100.0, {
            "completion_tokens": result.usage.completion_tokens,
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.total_tokens
        })

    return (max(0.0, min(100.0, score)), {
        "completion_tokens": result.usage.completion_tokens,
        "prompt_tokens": result.usage.prompt_tokens,
        "total_tokens": result.usage.total_tokens
    })

# pylint: disable=duplicate-code
# Default job arguments
default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 1000,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
# pylint: enable=duplicate-code

RELEVANCE_AGENT_PROMPT = '''You are a helpful assistant that must evaluate the relevance of a given passage \
to a given query, and thought process. Your task is to provide a relevance score \
between 0 and 100, where 0 means not relevant at all and 100 means highly relevant.
If you are unsure about whether the document could be relevant, respond with a score of 50.


### Example

Document: "The Girl in the Taxi (1937 film): The Girl in the Taxi is a 1937 British musical comedy fil directed by \
Andrew Berthomieu and starring Frances Day, Henri Garat and Lawrence Grossmith.
Thought: "I need to find the director of the film The Girl in White to determine which director was born first."
Query: "The Girl in White director"

Your response should be:

relevance_score: 0
explanation: "The document discusses a different film, The Girl in the Taxi, and does not provide any information about \
the film The Girl in White or its director.
'''
