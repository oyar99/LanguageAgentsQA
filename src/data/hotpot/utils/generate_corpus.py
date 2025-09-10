"""Utility script to generate a corpus from the HotpotQA dataset."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# pylint: disable-next=wrong-import-position
from data.utils.dataset_utils import generate_corpus

if __name__ == "__main__":
    # Read the hotpot dataset
    generate_corpus(
        input_path='hotpot_dev_distractor_v1.json',
        output_path='hotpot_corpus.json',
        context_extractor=lambda context: { 'title': context[0], 'text': ''.join(context[1]) }
    )

    # Read the first 1000 entries from the hotpot dataset
    generate_corpus(
        input_path='hotpot_train_v1.1.json',
        output_path='hotpot_corpus_2.json',
        context_extractor=lambda context: { 'title': context[0], 'text': ''.join(context[1]) },
        limit=1000
    )
