"""
Base class for integration testing
"""
import os
import unittest
from unittest.mock import patch

from integration_tests.mocks.file_mocks import cleanup_test
from integration_tests.mocks.openai_mocks import MockOpenAIClient

class IntegrationTest(unittest.TestCase):
    """
    Base class to setup tests that run in a safe controlled environment
    """

    def tearDown(self) -> None:
        """Clean up test environment."""
        # Reset OpenAI mock for clean state between tests
        MockOpenAIClient.reset()
        # Clean up test fixtures and temp directory
        cleanup_test()

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment."""
        # Mock environment variables to use OpenAI instead of AzureOpenAI
        cls.env_patcher = patch.dict(os.environ, {
            'LLM_ENDPOINT': 'https://api.openai.com/v1',  # Use OpenAI endpoint
            'REMOTE_LLM': '0',  # Don't force remote LLM
            'ENABLE_CONSOLE_LOGGING': '1',  # Enable console logging for tests
            'DISABLE_FILE_LOGGING': '1',
            # Clear Azure OpenAI env vars to ensure we use OpenAI
            'AZURE_OPENAI_ENDPOINT': '',
            'AZURE_OPENAI_API_KEY': '',
            'AZURE_OPENAI_API_VERSION': ''
        })
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test environment after all tests"""
        cls.env_patcher.stop()
