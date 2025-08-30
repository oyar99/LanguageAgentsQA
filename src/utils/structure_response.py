"""Utility to parse structured JSON responses from model outputs.
"""

import json
from typing import Any, Dict, Optional

from logger.logger import Logger


def parse_structured_response(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse the structured JSON response. It attempts to extract JSON content from the response string using 
    various common patterns such as markdown code blocks or JSON-like structures.

    Args:
        response_content (str): Raw response from the model.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON response or None if parsing fails.
    """
    try:
        # Try to extract JSON from the response
        # Prioritize extracting JSON from code blocks if present in case there are nested JSON objects
        if '{' in response_content and '}' in response_content:
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            json_str = response_content[json_start:json_end]
        elif '```json' in response_content:
            json_start = response_content.find('```json') + 7
            json_end = response_content.find('```', json_start)
            json_str = response_content[json_start:json_end].strip()
        else:
            # Fallback: try to parse the entire response
            json_str = response_content.strip()

        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        Logger().warn(f"Failed to parse structured response: {e}")
        Logger().debug(f"Raw response: {response_content}")
        return None
