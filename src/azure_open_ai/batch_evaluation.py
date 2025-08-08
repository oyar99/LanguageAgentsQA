"""Module to queue batch jobs for evaluation using Azure OpenAI."""
from typing import Optional
from openai.types import Batch

from azure_open_ai.batch import queue_batch_job
from logger.logger import Logger
from utils.token_utils import estimate_cost, estimate_num_tokens

EVALUATION_PROMPT = '''You are a helpful judge evaluating the quality of an answer. \
You will answer 'Yes' or 'No' to indicate whether the provided answer matches the expected answer.

The question is: {question}. \
The expected answer is: {answer}. \
Please answer with 'Yes' or 'No' only. \
'''


def queue_evaluation_batch_job(
    model: str,
    question_answers: list[tuple[str, str, str]],
    job_args: Optional[dict] = None
) -> Optional[Batch]:
    """
    Queues a batch job for evaluation using Azure OpenAI.
    The evaluation is done by comparing the expected answer with the provided answer.

    Args:
        model (str): model name to be used for evaluation
        question_answers (list[tuple[str, str, str]]): list of tuples containing question, \
expected and actual answer pairs
        job_args (dict, optional): a dictionary of job arguments. Defaults to None. Possible arguments are:
            temperature (float, optional): the sampling temperature to use. Defaults to 0.0.
            max_tokens (int, optional): the maximum number of tokens that can be generated in the completion.
            Defaults to 1000.
            frequency_penalty (float, optional): penalize new tokens based on their existing frequency in the 
            text so far. Defaults to 0.0.
            presence_penalty (float, optional): penalize new tokens based on whether they appear in the text so far. 
            Defaults to 0.0.

    Raises:
        ValueError: if any of the input parameters are invalid
        RuntimeError: if the file upload fails

    Returns:
        Optional[Batch]: the batch job object if the job is queued successfully, None otherwise
    """
    if job_args is None:
        job_args = {
            'temperature': 0.0,
            'max_tokens': 50,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }

    if not isinstance(model, str) or len(model) <= 0:
        raise ValueError(
            "model must be a non-empty string.")

    cost = 0.0

    for question, expected, actual in question_answers:
        prompt = EVALUATION_PROMPT.format(
            question=question,
            answer=expected,
        ) + actual
        token_count = estimate_num_tokens(prompt, model)
        cost += estimate_cost(token_count, model)
        if token_count > 1000:
            Logger().warn(
                "Prompt for evaluation exceeds 1,000 tokens. Truncation is recommended.")

    Logger().info(f"Estimated cost: {cost}")

    jobs = [
        {
            "custom_id": idx + 1,
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": EVALUATION_PROMPT.format(
                         question=question,
                         answer=expected,
                     )},
                    {"role": "user", "content": actual}
                ],
                "temperature": job_args['temperature'],
                "frequency_penalty": job_args['frequency_penalty'],
                "presence_penalty": job_args['presence_penalty'],
                "max_tokens": job_args['max_tokens']
            },
        }
        for idx, (question, expected, actual) in enumerate(question_answers)
    ]

    return queue_batch_job(jobs)
