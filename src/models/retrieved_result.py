"""RetrievedResult class."""


from typing import Optional


class RetrievedResult(dict):
    """
    RetrievedResult class to store the retrieved results.
    It inherits from dict and initializes the dictionary with the given parameters.

    Args:
        dict (Any): dictionary to store the retrieved results
        doc_id (int): the id of the document
        content (str): the content of the document
        score (float): the relevance score of the document
        folder_id (str): the folder id of the document - useful to group documents by folder
    """

    def __init__(
        self,
        doc_id: int,
        content: str,
        score: Optional[float] = None,
        folder_id: Optional[str] = None
    ) -> None:
        dict.__init__(self, doc_id=doc_id,
                      content=content, score=score, folder_id=folder_id)

    def __repr__(self):
        return f"""RetrievedResult(doc_id={self.get('doc_id')}, content={self.get('content')}),\
    score={self.get('score')})"""
