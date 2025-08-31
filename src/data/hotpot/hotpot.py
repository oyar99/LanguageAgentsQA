"""Hotpot dataset module."""

import json
import os
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.hash_utils import get_content_hash
from utils.question_utils import filter_questions


class Hotpot(Dataset):
    """Hotpot dataset class."""

    def __init__(self, args):
        super().__init__(args, name="hotpot")
        Logger().info("Initialized an instance of the Hotpot dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the Hotpot dataset.

        Returns:
            dataset (list[DatasetSample]): the dataset samples
        """
        Logger().info("Reading the Hotpot dataset")
        conversation_id = self._args.conversation

        # pylint: disable=duplicate-code
        file_path = os.path.join(
            "data", "hotpot", "hotpot_dev_distractor_v1.json")
        with open(file_path, encoding="utf-8") as hotpot_dataset:
            dataset = [
                DatasetSample(
                    sample_id=sample['_id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(doc_id=get_content_hash(''.join(doc[1])), content=''.join(doc[1]))
                                  for doc in sample['context']
                                  if any(doc[0] == fact[0] for fact in sample['supporting_facts'])],
                            question_id=sample['_id'],
                            question=sample['question'],
                            answer=[str(sample['answer'])],
                            category=QuestionCategory.MULTI_HOP
                            if sample['type'] == 'bridge' else (
                                QuestionCategory.COMPARISON if sample['type'] == 'comparison' else QuestionCategory.NONE
                            ),
                            decomposition=[{
                                'title': fact[0],
                            } for fact in sample.get('supporting_facts', [])]
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(hotpot_dataset)
                if conversation_id is None or sample['_id'] == conversation_id
            ]
            dataset = super().process_dataset(dataset)
            Logger().info(
                f"Hotpot dataset read successfully. Total samples: {len(dataset)}")

            return dataset
        # pylint: enable=duplicate-code

    def read_corpus(self) -> list[Document]:
        """
        Reads the Hotpot dataset and returns the corpus.

        Returns:
            corpus (list[Document]): the corpus
        """
        Logger().info("Reading the Hotpot dataset corpus")
        file_path = os.path.join("data", "hotpot", "hotpot_corpus.json")
        with open(file_path, encoding="utf-8") as hotpot_corpus:
            corpus = json.load(hotpot_corpus)
            corpus = [
                Document(doc_id=get_content_hash(
                    doc['text']), content=f'{doc["title"]}:{doc["text"]}')
                for doc in corpus
            ]
            super()._log_dataset_stats(corpus)

            return corpus
