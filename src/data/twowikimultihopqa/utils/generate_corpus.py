"""Utility script to generate a corpus from the 2WikimultihopQA dataset."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# pylint: disable-next=wrong-import-position
from data.utils.dataset_utils import generate_corpus

if __name__ == "__main__":
    # Read the 2wikimultihopQA dataset
    generate_corpus(
        input_path='dev.json',
        output_path='corpus.json',
        context_extractor=lambda context: {'title': context[0], 'text': ' '.join(context[1])})

    # Read the first 1000 entries from the 2wikimultihopQA train split
    generate_corpus(
        input_path='train.json',
        output_path='corpus_2.json',
        context_extractor=lambda context: {
            'title': context[0], 'text': ' '.join(context[1])},
        limit=1000
    )
