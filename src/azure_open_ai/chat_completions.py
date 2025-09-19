"""Azure OpenAI Chat Completions Module"""
from openai import NOT_GIVEN, BadRequestError
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from azure_open_ai.openai_client import OpenAIClient
from logger.logger import Logger


def _create_empty_chat_completion(job: dict) -> ChatCompletion:
    """
    Create an empty ChatCompletion object when content filtering occurs.

    Args:
        job (dict): The original job configuration.

    Returns:
        ChatCompletion: A ChatCompletion object with empty content.
    """
    # Create a mock ChatCompletion object with empty content
    return ChatCompletion(
        id="filtered-content",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="",
                    role="assistant"
                )
            )
        ],
        created=0,
        model=job["model"],
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0
        )
    )


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
        try:
            completion = openai_client.chat.completions.create(
                model=job["model"],
                messages=job["messages"],
                temperature=job["temperature"],
                frequency_penalty=job["frequency_penalty"],
                presence_penalty=job["presence_penalty"],
                max_tokens=job["max_completion_tokens"],
                stop=job.get("stop", None),
                tools=job.get("tools", NOT_GIVEN),
                tool_choice=job.get("tool_choice", NOT_GIVEN),
                response_format=job.get("response_format", NOT_GIVEN),
            )

            Logger().debug(
                f"Chat completion for job {job['custom_id']} with model {job['model']} completed"
            )

        except BadRequestError as e:
            # Check if this is a ResponsibleAIPolicyViolation
            error_code = None

            Logger().error(
                f"BadRequestError for job {job['custom_id']}: {e}"
            )

            # Try to extract error details from the exception
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                error_data = e.response.json()
                error_code = error_data.get('error', {}).get(
                    'innererror', {}).get('code')

            # Also check the string representation for ResponsibleAIPolicyViolation
            if error_code == 'ResponsibleAIPolicyViolation':

                Logger().warn(
                    f"Content filtered for job {job['custom_id']}: {error_code}"
                )

                completion = _create_empty_chat_completion(job)
            else:
                Logger().error(
                    "received BadRequestError that is not content filtering"
                )
                completion = _create_empty_chat_completion(job)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            Logger().error(
                f"Unexpected error for job {job['custom_id']}: {e}"
            )
            completion = _create_empty_chat_completion(job)

        results.append((completion, job['custom_id']))

    return results
