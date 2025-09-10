"""Dataset utilities for reading and processing datasets."""
import json
from typing import Callable, Any


def generate_corpus(
    input_path: str,
    output_path: str,
    context_extractor: Callable[[Any], dict[str, str]],
    context_key='context',
    limit=None,
) -> None:
    """
    Generate a corpus from the input dataset by extracting documents from the context.
    The documents are saved to a new file in JSON format.

    Args:
        input_path (str): the path to the input dataset file
        output_path (str): the path to the output file where the documents will be saved
        context_extractor (Callable[[Any], dict[str, str]]): a function that extracts documents from the context
        context_key (str, optional): the property to extract the context from the dataset. Defaults to 'context'.
        limit (int, optional): maximum number of entries to process. Defaults to None (process all).
    """
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as outfile:
        dataset = json.load(f)

        dataset = dataset if limit is None else dataset[:limit]

        documents = [
            context_extractor(context)
            for item in dataset
            for context in item[context_key]
        ]

        total_docs = len(documents)
        print(f'Extracted {total_docs} documents from {input_path}.')

        # Dedup documents based on identical content
        seen_texts = set()
        unique_documents = []
        for doc in documents:
            text = doc.get('text')
            if text not in seen_texts:
                seen_texts.add(text)
                unique_documents.append(doc)
        documents = unique_documents

        print(f'Deduped {total_docs - len(documents)} documents.')
        print(f'Generated {len(documents)} documents.')

        # Save the documents to a new file
        json.dump(documents, outfile, indent=4)
