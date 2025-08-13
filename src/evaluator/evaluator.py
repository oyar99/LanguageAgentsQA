"""Evaluator module."""

import os
import json
from typing import Optional, Any
from openai.types import Batch
from azure_open_ai.batch import wait_for_batch_job_and_save_result
from evaluator.bleu_evaluator import eval_bleu_score
from evaluator.exact_match_evaluator import eval_exact_match
from evaluator.f1_evaluator import eval_f1_score
from evaluator.bert_evaluator import eval_bert_score
from evaluator.judge_evaluator import eval_judge_score, eval_judge_score_with_file
from evaluator.metrics_evaluator import eval_metrics
from evaluator.retrieval_evaluator import eval_retrieval_recall
from evaluator.rogue_evaluator import eval_rogue_score
from logger.logger import Logger
from models.dataset import Dataset
from models.document import Document


def evaluator(args, dataset: Dataset) -> None:
    """
    Orchestrates the evaluation of the model's performance on the dataset.
    It evaluates the model's performance based on the provided arguments and dataset.

    Args:
        args (Namespace): the arguments passed to the script
        dataset (Dataset): the dataset to be processed

    Raises:
        ValueError: if the evaluation file is not provided
    """

    _ = dataset.read()

    Logger().info("Running evaluation")

    if args.evaluation is None and not (args.judge_eval and args.judge_eval_path):
        Logger().error("Evaluation file not provided. Please provide the evaluation file path using the -ev flag.")
        raise ValueError("Evaluation file not provided")

    if args.judge_eval and args.judge_eval_path:
        eval_judge_score_with_file(args.judge_eval_path)
        return

    with open(args.evaluation, "r", encoding="utf-8") as evaluation_file:
        evaluation = [json.loads(line) for line in evaluation_file]

        if args.retrieval:
            doc_pairs = [pair for pair in (extract_doc_pair(dataset,
                                                            eval_item) for eval_item in evaluation) if pair is not None]
            evaluate_retrieval(doc_pairs)
        elif args.metric:
            metrics = [m for m in (extract_metrics(eval_item)
                                   for eval_item in evaluation) if m is not None]
            eval_metrics(metrics)
        elif args.judge_eval and not args.judge_eval_path:
            batch: Optional[Batch] = None
            doc_pairs = []
            if args.eval_batch:
                doc_pairs = [(q, ans[0], act)
                             for pairs in
                             (extract_qa_pairs_with_question(dataset, eval_item)
                              for eval_item in evaluation)
                             for q, ans, act in pairs]
            else:
                doc_pairs = [(q, ans[0], act)
                             for q, ans, act in
                             [pair for pair in
                             (extract_qa_pair_with_question(dataset, eval_item)
                              for eval_item in evaluation)
                             if pair is not None]]
            batch = eval_judge_score(args.model, doc_pairs)

            if batch is not None:
                Logger().info(
                    f"Batch job {batch.id} submitted. Waiting for completion ...")
                wait_for_batch_job_and_save_result(
                    batch, get_eval_output_path())
        elif args.eval_batch:
            qa_pairs = [pair for pairs in (extract_qa_pairs(dataset,
                                                            eval_item) for eval_item in evaluation) for pair in pairs]
            evaluate(qa_pairs, args)
        else:
            qa_pairs = [pair for pair in (extract_qa_pair(dataset,
                                                          eval_item) for eval_item in evaluation) if pair is not None]
            evaluate(qa_pairs, args)


def evaluate_retrieval(doc_pairs: list[tuple[list[Document] | None, list[Document]]]) -> None:
    """
    Evaluates retrieval performance based on the provided document pairs.
    Evaluates the recall score across various Ks.

    Args:
        doc_pairs (list[tuple[list[Document], list[Document]]]): the ground truth documents and the model's documents
    """
    if len(doc_pairs) == 0:
        Logger().error("No doc pairs found. Please check the evaluation file.")
        raise ValueError("No doc pairs found")

    Logger().info("Evaluating retrieval score")

    recall_at_k = eval_retrieval_recall(doc_pairs)

    for k, recall in recall_at_k.items():
        Logger().info(f"Recall at {k}: {recall}")

# pylint: disable=too-many-locals
def evaluate(qa_pairs: list[tuple[list[str], str]], args) -> None:
    """
    Evaluates question answering performance based on the provided question-answer pairs.
    Evaluates the exact match score, F1 score, precision, recall, and BERT score (if applicable).

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

        args (Namespace): the arguments passed to the script
    """
    if len(qa_pairs) == 0:
        Logger().error("No question-answer pairs found. Please check the evaluation file.")
        raise ValueError("No question-answer pairs found")

    Logger().info("Evaluating exact match score")

    em = eval_exact_match(qa_pairs)
    f1, precision, recall = eval_f1_score(qa_pairs)
    rouge_scores = eval_rogue_score(qa_pairs)
    bleu_score = eval_bleu_score(qa_pairs)
    bert_score = None

    if args.bert_eval:
        bert_score = eval_bert_score(qa_pairs)
        Logger().info(f"BERT score: {bert_score}")

    rogue, rogue_precision, rogue_recall = rouge_scores[0]
    rogue_2, rogue_precision_2, rogue_recall_2 = rouge_scores[1]

    Logger().info(f"Exact match score: {em}")
    Logger().info(f"F1 score: {f1}")
    Logger().info(f"Precision: {precision}")
    Logger().info(f"Recall: {recall}")
    Logger().info(f"ROUGE score: {rogue}")
    Logger().info(f"ROUGE precision: {rogue_precision}")
    Logger().info(f"ROUGE recall: {rogue_recall}")
    Logger().info(f"ROUGE-2 score: {rogue_2}")
    Logger().info(f"ROUGE-2 precision: {rogue_precision_2}")
    Logger().info(f"ROUGE-2 recall: {rogue_recall_2}")
    Logger().info(f"BLEU score: {bleu_score}")


def extract_metrics(eval_item: dict[str, Any]) -> Optional[dict[str, int]]:
    """
    Extracts the metrics from the evaluation item.

    Args:
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        Optional[dict[str, any]]: the metrics extracted from the evaluation item
    """
    Logger().debug(
        f"Extracting metrics for evaluation item: {eval_item['custom_id']}")
    try:
        completion_tokens = eval_item['response']['body']['usage']['completion_tokens']
        prompt_tokens = eval_item['response']['body']['usage']['prompt_tokens']
        total_tokens = eval_item['response']['body']['usage']['total_tokens']
    except KeyError:
        Logger().error(
            f"KeyError: 'usage metrics' not found in the evaluation item: {eval_item['custom_id']}")
        return None

    return {
        'completion_tokens': int(completion_tokens),
        'prompt_tokens': int(prompt_tokens),
        'total_tokens': int(total_tokens)
    }


def extract_doc_pair(
    dataset: Dataset,
    eval_item: dict[str, Any]
) -> Optional[tuple[list[Document] | None, list[Document]]]:
    """
    Extracts the document pairs for evaluation from the evaluation item.

    Args:
        dataset (Dataset): the dataset to be processed
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        Optional[tuple[list[Document], list[Document]]]: the ground truth documents and the model's documents
    """
    Logger().debug(
        f"Extracting docs pair for evaluation item: {eval_item['custom_id']}")
    question = dataset.get_question(eval_item['custom_id'])

    if question is None:
        Logger().warn(
            f"Sample id {eval_item['custom_id']} not found in the dataset. Skipping evaluation ...")
        return None

    Logger().debug(f"Question found: {question['question']}")

    expected_docs = dataset.get_supporting_docs(eval_item['custom_id'])
    actual_docs = [Document(
        doc_id=result['doc_id'],
        content=result['content']
    ) for result in eval_item['result']]

    doc_pairs = (
        expected_docs,
        actual_docs,
    )

    Logger().debug(
        f"Retrieval extracted: (question, truth, predicted): {question['question']}{doc_pairs}")

    return doc_pairs


def extract_qa_pairs(dataset: Dataset, eval_item: dict[str, Any]) -> list[tuple[list[str], str]]:
    """
    Extracts the question-answer pairs for evaluation from the evaluation item.

    Used when multiple questions are present in the evaluation item.

    Args:
        dataset (Dataset): the dataset to be processed
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        list[tuple[list[str], str]]: the ground truth answers and the model's answers
    """
    pairs = extract_qa_pairs_with_question(dataset, eval_item)

    return [(answers, actual) for _, answers, actual in pairs]


def extract_qa_pairs_with_question(dataset: Dataset, eval_item: dict[str, Any]) -> list[tuple[str, list[str], str]]:
    """
    Extracts the question-answer pairs for evaluation from the evaluation item.

    Used when multiple questions are present in the evaluation item.

    Args:
        dataset (Dataset): the dataset to be processed
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        list[tuple[list[str], str]]: the question, the ground truth answers and the model's answers
    """
    Logger().debug(
        f"Extracting QA pairs for evaluation item: {eval_item['custom_id']}")
    qa_pairs = []

    try:
        json_obj = json.loads(
            eval_item['response']['body']['choices'][0]['message']['content'])
    except KeyError:
        Logger().error(
            f"KeyError: 'response' or 'body' not found in the evaluation item: {eval_item['custom_id']}")
        return qa_pairs

    results = json_obj.get('result', [])

    Logger().info(
        f"Found {len(results)} results in the evaluation item: {eval_item['custom_id']}")

    question_ids = set()

    for qa in json_obj.get('result', []):
        question_id = qa.get('question_id')

        if question_id in question_ids:
            Logger().warn(
                f"Duplicate question ID {question_id} found in the evaluation item: \
{eval_item['custom_id']}. Skipping ...")
            continue

        question_ids.add(question_id)
        answer = qa.get('answer')
        if question_id and answer:
            question = dataset.get_question(question_id)
            if question is None:
                Logger().warn(
                    f"Sample id {question_id} not found in the dataset. Skipping evaluation ...")
            else:
                qa_pairs.append(
                    (question['question'], question['answer'], answer))
        else:
            Logger().error(
                f"Question ID or answer not found in the evaluation item: {eval_item['custom_id']}")
            raise ValueError(
                f"Question ID or answer not found in the evaluation item: {eval_item['custom_id']}")

    return qa_pairs


def extract_qa_pair(dataset: Dataset, eval_item: dict[str, Any]) -> Optional[tuple[list[str], str]]:
    """
    Extracts the question-answer pair for evaluation from the evaluation item.
    Used when a single question is present in the evaluation item.

    Args:
        dataset (Dataset): the dataset to be processed
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        Optional[tuple[list[str], str]]: the ground truth answer and the model's answer
    """
    pair = extract_qa_pair_with_question(dataset, eval_item)

    if pair is None:
        return None

    question, answer, actual = pair

    qa_pair = (answer, actual)

    Logger().debug(
        f"QA extracted: (question, truth, predicted): {question}{qa_pair}")
    return qa_pair


def extract_qa_pair_with_question(dataset: Dataset, eval_item: dict[str, Any]) -> Optional[tuple[str, list[str], str]]:
    """
    Extracts the question-answer pair for evaluation from the evaluation item.
    Used when a single question is present in the evaluation item.
    The function retrieves the question from the dataset and the answer from the evaluation item.

    Args:
        dataset (Dataset): the dataset to be processed
        eval_item (dict[str, any]): the evaluation item to be processed

    Returns:
        Optional[tuple[str, list[str], str]]: the question, ground truth answer and the model's answer
    """
    Logger().debug(
        f"Extracting QA pair with question for evaluation item: {eval_item['custom_id']}")
    question = dataset.get_question(eval_item['custom_id'])

    if question is None:
        Logger().warn(
            f"Sample id {eval_item['custom_id']} not found in the dataset. Skipping evaluation ...")
        return None

    Logger().debug(f"Question found: {question['question']}")

    content = (str(eval_item['response']['body']['choices'][0]['message']['content']).strip()
               if eval_item['response']['body']['choices'][0]['message'].get('content') else None)

    if content is None:
        Logger().warn(
            "Content not found in the response. Setting to Empty string ...")
        content = ""

    qa_pair = (question['question'], question['answer'], str(
        content))

    return qa_pair


def get_eval_output_path() -> str:
    """
    Get the output path for the L1 evaluation job results.

    Returns:
        str: the output path
    """
    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'eval_jobs')
    return os.path.join(
        output_dir, f'eval_results_{Logger().get_run_id()}.jsonl')
