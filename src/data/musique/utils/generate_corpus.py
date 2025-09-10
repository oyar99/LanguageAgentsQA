"""Utility script to generate a corpus from the MuSiQue dataset."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# pylint: disable-next=wrong-import-position
from data.utils.dataset_utils import generate_corpus

if __name__ == "__main__":
    # Read the MusiQue dataset
    generate_corpus(
        input_path='musique_dev.json',
        output_path='musique_corpus.json',
        context_extractor=lambda context: {
            'title': context['title'], 'text': context['paragraph_text']},
        context_key='paragraphs'
    )

    # Read the MusiQue test dataset
    generate_corpus(
        input_path='musique_dev_2.json',
        output_path='musique_corpus_2.json',
        context_extractor=lambda context: {
            'title': context['title'], 'text': context['paragraph_text']},
        context_key='paragraphs',
        limit=1000
    )
