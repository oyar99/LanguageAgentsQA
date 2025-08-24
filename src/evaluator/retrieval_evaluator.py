
"""Retrieval Evaluator Module."""

from typing import Tuple
from logger.logger import Logger
from models.document import Document


K_LIST = [1, 2, 5, 10, 20, 100]


def eval_retrieval_recall(doc_pairs: list[tuple[list[Document], list[Document]]]) -> dict[int, float]:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents.

    Args:
        doc_pairs (list[tuple[list[Document], list[Document]]]): \
A list of pairs with the ground documents and the retrieved documents.

    Returns:
        recall_at_k (dict[int, float]): the recall score across various Ks
    """
    recall_at_k = [recall_score(gt, a) for (gt, a) in doc_pairs]

    avg_recall_at_k = {
        k: sum(d[k] for d in recall_at_k) / len(recall_at_k)
        for k in K_LIST
    }

    return avg_recall_at_k


def recall_score(expected_docs: list[Document], actual_docs: list[Document]) -> dict[int, float]:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents.

    Args:
        expected_docs (list[Document]): the ground truth documents
        actual_docs (list[Document]): the model's retrieved documents

    Returns:
        recall_at_k (dict[int, float]): the recall score across various Ks
    """
    assert expected_docs, "Expected documents list is empty."
    assert actual_docs, "Actual documents list is empty."

    recall_at_k = {}
    for k in K_LIST:
        if len(actual_docs) < k:
            Logger().warn(f'Length of actual docs is less than {k}, retrieval at K may not be accurate')
        top_k_docs = actual_docs[:k]

        recall_at_k[k] = recall(expected_docs, top_k_docs)

    return recall_at_k

def eval_retrieval_unranked(doc_pairs: list[tuple[list[Document], list[Document]]]) -> Tuple[float, float, float]:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents,
    treating the retrieved documents as an unranked set.

    Args:
        doc_pairs (list[tuple[list[Document], list[Document]]]): \
A list of pairs with the ground documents and the retrieved documents.
    Returns:
        recall (Tuple[float, float, float]): the overall recall score, precision, and F1 score
    """
    recall_scores = [recall(expected, actual) for expected, actual in doc_pairs]
    overall_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    precision_scores = [precision(expected, actual) for expected, actual in doc_pairs]
    overall_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    f1_scores = [f1_score(expected, actual) for expected, actual in doc_pairs]
    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return (overall_recall, overall_precision, overall_f1)

def recall(expected_docs: list[Document], actual_docs: list[Document]) -> float:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents.

    Args:
        expected_docs (list[Document]): the ground truth documents
        actual_docs (list[Document]): the model's retrieved documents

    Returns:
        recall (float): the recall score
    """
    assert expected_docs, "Expected documents list is empty."

    correct = sum(1 for doc in actual_docs if doc['doc_id'] in [
                  expected_doc['doc_id'] for expected_doc in expected_docs])
    return correct / len(expected_docs)

def precision(expected_docs: list[Document], actual_docs: list[Document]) -> float:
    """
    Evaluates the precision between the ground truth documents and the model's retrieved documents.

    Args:
        expected_docs (list[Document]): the ground truth documents
        actual_docs (list[Document]): the model's retrieved documents

    Returns:
        precision (float): the precision score
    """
    if len(actual_docs) == 0:
        return 0.0

    correct = sum(1 for doc in actual_docs if doc['doc_id'] in [
                  expected_doc['doc_id'] for expected_doc in expected_docs])
    return correct / len(actual_docs)

def f1_score(expected_docs: list[Document], actual_docs: list[Document]) -> float:
    """
    Evaluates the F1 score between the ground truth documents and the model's retrieved documents.

    Args:
        expected_docs (list[Document]): the ground truth documents
        actual_docs (list[Document]): the model's retrieved documents

    Returns:
        f1_score (float): the F1 score
    """
    prec = precision(expected_docs, actual_docs)
    rec = recall(expected_docs, actual_docs)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)
