"""Azure OpenAI Chat Completions Module"""
from openai.types.chat.chat_completion import ChatCompletion
from azure_open_ai.openai_client import OpenAIClient
from logger.logger import Logger


def chat_completions(
    jobs: list[dict],
) -> list[tuple[ChatCompletion, str]]:
    """
    Function to handle chat completions using Azure OpenAI.

    Args:
        jobs (list[dict]): List of jobs to process.

    Returns:
        list[dict]: List of processed jobs with chat completions.
    """
    openai_client = OpenAIClient().get_client()

    if not openai_client:
        Logger().error("OpenAI client is not initialized.")
        raise RuntimeError("OpenAI client is not initialized.")

    results = []

    for job in jobs:
        completion = openai_client.chat.completions.create(
            model=job["model"],
            messages=job["messages"],
            temperature=job["temperature"],
            frequency_penalty=job["frequency_penalty"],
            presence_penalty=job["presence_penalty"],
            max_tokens=job["max_completion_tokens"],
            stop=job["stop"],
            tools=job.get("tools", None),
            tool_choice=job.get("tool_choice", None),
        )

        Logger().debug(
            f"Chat completion for job {job['custom_id']} with model {job['model']} completed"
        )

        results.append((completion, job['custom_id']))

    return results
