"""A module to create a question-answer class."""
from enum import IntEnum

from models.document import Document


class QuestionCategory(IntEnum):
    """
    Enum class to represent the category of a question.
    The categories are represented as integer values.

    Args:
        IntEnum: inherits from IntEnum
    """
    MULTI_HOP = 1
    TEMPORAL = 2
    OPEN_DOMAIN = 3
    SINGLE_HOP = 4
    ADVERSARIAL = 5
    COMPARISON = 6
    NONE = 7


class QuestionAnswer(dict):
    """
    QuestionAnswer class to store the question and answer.
    It inherits from dict and initializes the dictionary with the given parameters.

    Args:
        dict (Any): dictionary to store the question and answer
        question_id (str): the id of the question
        question (str): the question text
        answer (list[str]): the list of possible answers
        category (QuestionCategory): the category of the question
        docs (list[Document]): list of documents that support the answer to the question
        decomposition (list[dict], optional): question decomposition steps for multi-hop reasoning
    """

    # pylint: disable-next=too-many-positional-arguments,too-many-arguments
    def __init__(
        self,
        question_id: str,
        question: str,
        answer: list[str],
        category: QuestionCategory,
        docs: list[Document],
        decomposition: list[dict] = None
    ):
        decomposition = decomposition or []
        dict.__init__(self, question_id=question_id,
                      question=question, answer=answer, category=category, docs=docs, decomposition=decomposition)

    def __repr__(self):
        return f"""Question(question_id={self.get('question_id')}, question={self.get('question')}),\
    answer={self.get('answer')}, category={self.get('category')}), docs={self.get('docs')}, \
    decomposition={self.get('decomposition')}"""
