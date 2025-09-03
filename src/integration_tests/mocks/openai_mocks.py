"""Mock implementations for OpenAI client components."""
import json
from unittest.mock import MagicMock
from typing import Any, Dict, List, Union
from utils.singleton import Singleton


class MockOpenAIResponse:
    """Mock OpenAI chat completion response."""

    def __init__(self, content: Union[str, Dict[str, Any]],
                 role: str = "assistant",
                 finish_reason: str = "stop",
                 completion_tokens: int = 100,
                 prompt_tokens: int = 200):
        """
        Initialize mock response.

        Args:
            content: Response content (string or dict that will be JSON-serialized)
            role: Message role (default: "assistant")
            finish_reason: Completion finish reason (default: "stop")
            completion_tokens: Number of completion tokens (default: 100)
            prompt_tokens: Number of prompt tokens (default: 200)
        """
        self.choices = [MagicMock()]

        # If content is a dict, serialize it to JSON
        if isinstance(content, dict):
            content = json.dumps(content)

        self.choices[0].message.content = content
        self.choices[0].message.role = role
        self.choices[0].finish_reason = finish_reason

        self.usage = MagicMock()
        self.usage.completion_tokens = completion_tokens
        self.usage.prompt_tokens = prompt_tokens
        self.usage.total_tokens = completion_tokens + prompt_tokens


class MockOpenAIClient(metaclass=Singleton):
    """
    Mock OpenAI client that can be configured with multiple responses.
    Implemented as a singleton using the existing Singleton metaclass.

    Usage:
    @patch('azure_open_ai.openai_client.OpenAIClient.get_client', new=MockOpenAIClient)

    Then in your test:
    MockOpenAIClient.configure_responses([
        {"thought": "First response", "final_answer": "Answer 1"},
        {"thought": "Second response", "final_answer": "Answer 2"}
    ])
    """

    _responses: List[MockOpenAIResponse] = []
    _call_count: int = 0

    @classmethod
    def configure_responses(cls, responses: List[Union[str, Dict[str, Any], MockOpenAIResponse]]) -> None:
        """
        Configure the responses that will be returned by chat.completions.create().

        Args:
            responses: List of responses. Can be strings, dicts, or MockOpenAIResponse objects.
        """
        cls._responses = []
        for response in responses:
            if isinstance(response, MockOpenAIResponse):
                cls._responses.append(response)
            else:
                cls._responses.append(MockOpenAIResponse(response))
        cls._call_count = 0

    @classmethod
    def configure_single_response(cls, content: Union[str, Dict[str, Any]]) -> None:
        """
        Configure a single response (convenience method).

        Args:
            content: Response content (string or dict)
        """
        cls.configure_responses([content])

    @classmethod
    def get_call_count(cls) -> int:
        """
        Get the number of times the OpenAI client has been called.

        Returns:
            int: The call count
        """
        return cls._call_count

    def __init__(self):
        """Initialize mock client."""
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create_completion

    # pylint: disable-next=unused-argument
    def _create_completion(self, *args, **kwargs) -> MockOpenAIResponse:
        """Mock the chat completion creation (accepts all parameters like the real client)."""
        if not self._responses:
            raise ValueError(
                "No responses configured. Call MockOpenAIClient.configure_responses() first.")

        # Return the current response (cycling through if more calls than responses)
        response = self._responses[self._call_count % len(self._responses)]
        self.__class__._call_count += 1
        return response

    @classmethod
    def reset(cls) -> None:
        """Reset the mock client state."""
        cls._responses = []
        cls._call_count = 0
