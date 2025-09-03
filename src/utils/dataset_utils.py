"""Utilities for processing datasets."""

import re
from typing import Optional
from models.document import Document
from models.question_answer import QuestionAnswer


def get_complete_evidence(question: QuestionAnswer, corpus: list[Document], dataset_name: str = None) -> list[Document]:
    """
    Get complete evidence for a question, including previous messages for Locomo dataset.

    Args:
        question: A QuestionAnswer object
        corpus: List of Document objects from the corpus
        dataset_name (str, optional): Name of the dataset to determine behavior

    Returns:
        list[Document]: List of Document objects with complete evidence
    """
    if dataset_name != 'locomo':
        # For non-Locomo datasets, just return the original documents
        return question['docs']

    # For Locomo dataset, include previous messages
    complete_evidence = []

    # Extract sample_id from question_id
    question_id = question['question_id']
    match = re.match(r'^(conv-.\d+)-(.*)$', question_id)
    if not match:
        return question['docs']

    sample_id = match.group(1)

    # For each document in the question's evidence
    for doc in question['docs']:
        # Find and add the previous message if it exists
        prev_message_doc = _find_previous_message_in_corpus(
            corpus, doc['doc_id'], sample_id)
        if prev_message_doc:
            complete_evidence.append(prev_message_doc)

        # Add the original evidence document after the previous message to keep chronological order
        complete_evidence.append(doc)

    return complete_evidence


def _find_previous_message_in_corpus(corpus: list[Document], doc_id: str, sample_id: str) -> Optional[Document]:
    """
    Find the previous message for a given document ID in the corpus.

    Args:
        corpus: List of Document objects from the corpus
        doc_id (str): The document ID (format: "D1:12:conv-26")
        sample_id (str): The sample ID (format: "conv-26")

    Returns:
        Optional[Document]: The previous message Document object, or None if not found
    """
    # Extract session and message index from doc_id
    doc_parts = doc_id.split(':')
    if len(doc_parts) < 2:
        return None

    session_num = int(doc_parts[0][1:])  # Extract number from "D1" -> 1
    message_idx = int(doc_parts[1])  # Get 1-based message index

    # Look for the previous message (message_idx - 1) in the same session
    prev_message_id = f"D{session_num}:{message_idx - 1}:{sample_id}"

    # Search for the previous message in the corpus
    for document in corpus:
        if document['doc_id'] == prev_message_id and document['folder_id'] == sample_id:
            return document

    return None
