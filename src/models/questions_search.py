"""
Structural Question Search System

This system matches questions by syntactic/structural similarity rather than content.
It converts questions into POS-tagged structural skeletons and finds similar patterns.
"""

from typing import Any, List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from logger.logger import Logger

class QuestionsSearchEngine:
    """Search engine for finding semantically similar questions."""

    def __init__(self):
        """Initialize the search engine.
        """

        # Storage for indexed questions
        self._questions = []
        self._question_embeddings = None

        # Initialize embedding model
        self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def build_index(self, questions: List[Dict[str, Any]]) -> None:
        """
        Build the index from a list of questions.

        Args:
            questions: List of question data to index
        """
        Logger().debug(f"Building index with {len(questions)} questions.")

        self._questions = questions

        # Compute embeddings for all questions
        question_texts = [q['question'] for q in questions]
        self._question_embeddings = self._embedding_model.encode(
            question_texts, convert_to_numpy=True)

        Logger().debug(f"Index built successfully with length {len(self._questions)}.")

    def rebuild_index(self, question: Dict[str, Any]) -> None:
        """
        Rebuild the index with a new question.

        Args:
            question: New question data to add to the index
        """
        Logger().debug(f"Rebuilding index with new question: {question}")

        # Add new question
        self._questions.append(question)

        # Recompute embeddings for all questions
        question_texts = [q['question'] for q in self._questions]
        self._question_embeddings = self._embedding_model.encode(
            question_texts, convert_to_numpy=True)

        Logger().debug(f"Index rebuilt successfully with lenght {len(self._questions)}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """
        Search for semantically similar questions using sentence embeddings on actual question text.

        Args:
            query: User's query question
            top_k: Number of results to return

        Returns:
            List[Tuple[Dict, str, float]]: List of (question_data, skeleton, similarity_score)
        """
        if not self._questions:
            Logger().warn("Index not built. No questions available for search.")
            return []

        Logger().debug(f"Searching semantically for query: {query}")

        # Encode the query
        query_embedding = self._embedding_model.encode([query])

        # Compute cosine similarities
        semantic_similarities = cosine_similarity(
            query_embedding, self._question_embeddings).flatten()

        # Get top similar questions
        top_k = min(top_k, len(self._questions))
        top_indices = np.argpartition(semantic_similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(
            semantic_similarities[top_indices])[::-1]]

        # Format results - return actual question data
        results = []
        for idx in top_indices:
            # Only include non-zero similarities
            if semantic_similarities[idx] > 0:
                Logger().debug(f"Found semantically similar question: \
 {self._questions[idx]} with similarity {semantic_similarities[idx]:.4f}")
                results.append(
                    (self._questions[idx], semantic_similarities[idx])
                )

        Logger().debug(
            f"Found {len(results)} semantically similar questions using embeddings")
        return results
