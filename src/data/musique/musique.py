"""Musique dataset class."""

import json
import os
import random
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.hash_utils import get_content_hash
from utils.question_utils import filter_questions


class MuSiQue(Dataset):
    """MuSiQue dataset class"""

    def __init__(self, args):
        super().__init__(args, name=args.dataset)
        Logger().info("Initialized an instance of the MuSiQue dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the MuSiQue dataset.

        Returns:
            dataset (list[DatasetSample]): the dataset samples
        """
        Logger().info("Reading the MuSiQue dataset")
        conversation_id = self._args.conversation

        file_name = "musique_dev.json" if self._args.dataset == "musique" else "musique_dev_2.json"

        with open(os.path.join("data", "musique", file_name), encoding="utf-8") as musique_dataset:
            data = json.load(musique_dataset)

            if self._args.shuffle:
                random.shuffle(data)
                Logger().info("Questions shuffled randomly")

            dataset = [
                DatasetSample(
                    sample_id=sample['id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(
                                doc_id=get_content_hash(doc['paragraph_text']),
                                content=f'{doc["title"]}:{doc["paragraph_text"]}')
                                for doc in sample['paragraphs'] if doc['is_supporting']],
                            question_id=sample['id'],
                            question=sample['question'],
                            answer=[str(sample['answer'])] +
                            sample['answer_aliases'],
                            category=QuestionCategory.MULTI_HOP,
                            decomposition=[{
                                'question': step['question'],
                                'answer': step['answer']
                            } for step in sample.get('question_decomposition', [])]
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in data
                # if sample['id'].startswith('4hop')
                if conversation_id is None or sample['id'] == conversation_id
            ]
            dataset = super().process_dataset(dataset)
            Logger().info(
                f"MuSiQue dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    def read_corpus(self) -> list[Document]:
        """
        Reads the MuSiQue dataset and returns the corpus.

        Returns:
            corpus (list[str]): the corpus
        """
        file_name = "musique_corpus.json" if self._args.dataset == "musique" else "musique_corpus_2.json"
        file_path = os.path.join("data", "musique", file_name)
        with open(file_path, encoding="utf-8") as musique_corpus:
            corpus = json.load(musique_corpus)
            # pylint: disable=duplicate-code
            corpus = [
                Document(doc_id=get_content_hash(
                    doc['text']), content=f'{doc["title"]}:{doc["text"]}', title=doc["title"])
                for doc in corpus
            ]
            super()._log_dataset_stats(corpus)
            # pylint: disable=enable-code

            return corpus
