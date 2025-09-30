"""A module to create a singleton instance of the Azure OpenAI client."""
import os
from openai import AzureOpenAI, Timeout
from openai import OpenAI

from logger.logger import Logger
from utils.singleton import Singleton


class OpenAIClient(metaclass=Singleton):
    """
    A class that provides a singleton instance of the Azure OpenAI client.
    """

    def __init__(self):
        self._client = None
        self.initialize_client()

    def initialize_client(self):
        """
        Initializes the Azure OpenAI client if it is not already initialized.

        Raises:
            ValueError: if any of the required environment variables are not set
        """
        if self._client is None:
            llm_endpoint = os.getenv("LLM_ENDPOINT", None)
            force_remote_llm = os.getenv("REMOTE_LLM", None)

            if (llm_endpoint is not None and len(llm_endpoint) > 0) and (force_remote_llm != "1"):
                Logger().info(
                    f"Using custom LLM endpoint: {llm_endpoint}, force_remote: {force_remote_llm}"
                )
                self._client = OpenAI(
                    base_url=llm_endpoint,
                    api_key='PLACEHOLDER',
                    timeout=Timeout(180.0),
                )
            else:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or ""
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                api_version = os.getenv(
                    "AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

                self._client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    max_retries=18
                )

    def get_client(self):
        """
        Returns the Azure OpenAI client.

        Returns:
            client (AzureOpenAI): A singleton instance of the Azure OpenAI client.
        """
        return self._client
