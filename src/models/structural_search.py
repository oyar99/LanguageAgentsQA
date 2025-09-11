"""
Structural Question Search System

This system matches questions by syntactic/structural similarity rather than content.
It converts questions into POS-tagged structural skeletons and finds similar patterns.
"""

import itertools
from collections import defaultdict
import difflib
from typing import List, Dict, Tuple
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from logger.logger import Logger


class StructuralSearchEngine:
    """Search engine for finding structurally similar questions."""

    def __init__(self, use_semantic_search=False):
        """Initialize the search engine.
        
        Args:
            use_semantic_search (bool): If True, use semantic search instead of structural search
        """
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Search mode configuration
        self.use_semantic_search = use_semantic_search

        # Storage for indexed questions
        self.questions = []  # Original questions
        self.skeletons = []  # Structural skeletons
        self.skeleton_to_questions = defaultdict(
            list)  # Map skeletons to question indices

        # TF-IDF vectorizer for skeleton similarity
        self.vectorizer: TfidfVectorizer = None
        self.skeleton_vectors = None
        
        # Embedding model for semantic similarity
        self.embedding_model: SentenceTransformer = None
        self.question_embeddings = None
        
        # Initialize embedding model if using semantic search
        if self.use_semantic_search:
            Logger().debug("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            Logger().debug("Sentence transformer model loaded.")

        Logger().debug("Structural Search Engine initialized.")

    # pylint: disable-next=too-many-branches,too-many-return-statements
    def normalize_token(self, token):
        """
        Convert a spaCy token to its structural representation.

        Args:
            token: spaCy token object

        Returns:
            str: Normalized structural representation
        """
        # WH-words: preserve distinct question words
        if token.tag_ in ["WP", "WRB", "WDT"]:
            return token.text.lower()

        # Auxiliary verbs (is, was, do, can, etc.)
        if token.pos_ == "AUX":
            return "<AUX>"

        # Proper nouns -> entities (ignore specific names)
        if token.pos_ == "PROPN":
            return "<ENTITY>"

        # Common nouns
        if token.pos_ == "NOUN":
            return "<NOUN>"

        # Numbers and quantifiers
        if token.pos_ == "NUM":
            return "<NUMBER>"

        # Verbs
        if token.pos_ == "VERB":
            # Distinguish past participles (passive voice)
            if token.tag_ == "VBN":
                return "<VERB_PP>"
            return "<VERB>"

        # Adjectives
        if token.pos_ == "ADJ":
            if token.tag_ == "JJR":  # Comparative
                return "<COMP_ADJ>"
            if token.tag_ == "JJS":  # Superlative
                return "<SUP_ADJ>"
            return "<ADJ>"

        # Adverbs
        if token.pos_ == "ADV":
            if token.tag_ == "RBR":  # Comparative
                return "<COMP_ADV>"
            if token.tag_ == "RBS":  # Superlative
                return "<SUP_ADV>"
            return "<ADV>"

        # Prepositions - keep some key ones distinct for better structure
        if token.pos_ == "ADP":
            if token.text.lower() in ["of", "in", "on", "at", "to", "for", "with", "by", "after", "from"]:
                return f"<{token.text.upper()}>"
            return "<PREP>"

        # Conjunctions
        if token.pos_ in ["CCONJ", "SCONJ"]:
            return "<CONJ>"

        # Pronouns
        if token.pos_ == "PRON":
            return "<PRON>"

        # Determiners
        if token.pos_ == "DET":
            return "<DET>"

        # Possessive markers
        if token.tag_ == "POS":
            return "'s"

        # Keep punctuation and other words as-is (lowercased)
        return token.text.lower()

    def question_to_skeleton(self, question: str) -> str:
        """
        Convert a question to its structural skeleton.

        Args:
            question: Original question text

        Returns:
            str: Structural skeleton
        """
        doc = self.nlp(question)
        tokens = [self.normalize_token(token) for token in doc]

        # Collapse consecutive duplicates (e.g., <NOUN> <NOUN> -> <NOUN>)
        collapsed = [key for key, _ in itertools.groupby(tokens)]

        return " ".join(collapsed)

    def compute_edit_distance(self, skeleton1: str, skeleton2: str) -> float:
        """
        Compute normalized edit distance between two skeletons.

        Args:
            skeleton1: First skeleton
            skeleton2: Second skeleton

        Returns:
            float: Normalized edit distance (0 = identical, 1 = completely different)
        """
        tokens1 = skeleton1.split()
        tokens2 = skeleton2.split()

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        similarity = matcher.ratio()

        return 1.0 - similarity  # Convert similarity to distance

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """
        Search for semantically similar questions using sentence embeddings on actual question text.

        Args:
            query: User's query question
            top_k: Number of results to return

        Returns:
            List[Tuple[Dict, str, float]]: List of (question_data, skeleton, similarity_score)
        """
        if not self.questions:
            Logger().warn("Index not built. No questions available for search.")
            return []

        Logger().debug(f"Searching semantically for query: {query}")

        # Encode the query
        query_embedding = self.embedding_model.encode([query])

        # Compute cosine similarities
        semantic_similarities = cosine_similarity(
            query_embedding, self.question_embeddings).flatten()

        # Get top similar questions
        top_k = min(top_k, len(self.questions))
        top_indices = np.argpartition(semantic_similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(semantic_similarities[top_indices])[::-1]]

        # Format results - return actual question data with corresponding skeletons
        results = []
        for idx in top_indices:
            if semantic_similarities[idx] > 0:  # Only include non-zero similarities
                Logger().debug(f"Found semantically similar question: \
 {self.questions[idx]} with similarity {semantic_similarities[idx]:.4f}")
                results.append(
                    (self.questions[idx], self.skeletons[idx], semantic_similarities[idx])
                )

        Logger().debug(f"Found {len(results)} semantically similar questions using embeddings")
        return results

    # pylint: disable-next=too-many-locals
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """
        Search for similar questions. Delegates to semantic or structural search based on configuration.

        Args:
            query: User's query question
            top_k: Number of results to return

        Returns:
            List[Tuple[Dict, str, float]]: List of (question_data, skeleton, similarity_score)
        """
        if self.use_semantic_search:
            return self.search_semantic(query, top_k)
        else:
            return self.search_structural(query, top_k)

    # pylint: disable-next=too-many-locals
    def search_structural(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """
        Search for structurally similar questions.

        Args:
            query: User's query question
            top_k: Number of results to return

        Returns:
            List[Tuple[Dict, str, float]]: List of (question_data, skeleton, similarity_score)
        """
        if not self.questions:
            Logger().warning("Index not built. No questions available for search.")
            return []

        # Convert query to skeleton
        query_skeleton = self.question_to_skeleton(query)
        Logger().debug(f"Query skeleton: {query_skeleton}")

        # Method 1: Exact skeleton matches (highest priority)
        exact_matches = []
        if query_skeleton in self.skeleton_to_questions:
            for idx in self.skeleton_to_questions[query_skeleton]:
                exact_matches.append(
                    (self.questions[idx], query_skeleton, 1.0))

        # Method 2: TF-IDF cosine similarity
        """
        query_vector = self.vectorizer.transform([query_skeleton])
        cosine_similarities = cosine_similarity(
            query_vector, self.skeleton_vectors).flatten()

        top_k = min(top_k, len(self.questions))

        # Get top similar skeletons (excluding exact matches)
        top_indices = np.argpartition(cosine_similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(cosine_similarities[top_indices])[::-1]]

        similar_indices = [
            (idx, cosine_similarities[idx])
            for idx in top_indices
            if self.skeletons[idx] != query_skeleton
        ]
        

        # Method 3: Edit distance for very similar structures
        edit_distance_results = []
        # Check by cosine similarity
        for idx, cosine_score in similar_indices:
            edit_dist = self.compute_edit_distance(
                query_skeleton, self.skeletons[idx])
            # Combine cosine similarity and edit distance
            combined_score = (cosine_score + (1.0 - edit_dist)) / 2.0
            edit_distance_results.append((idx, combined_score))

        # Sort by combined score and take top results
        edit_distance_results.sort(key=lambda x: x[1], reverse=True)
        """

        """
        edit_distance_results = []

        for idx, skeleton in enumerate(self.skeletons):
            if skeleton == query_skeleton:
                continue  # Skip exact matches
            edit_dist = self.compute_edit_distance(query_skeleton, skeleton)
            similarity_score = 1.0 - edit_dist
            edit_distance_results.append((idx, similarity_score))

        # Sort by similarity score
        edit_distance_results.sort(key=lambda x: x[1], reverse=True)
        """

        query_vector = self.vectorizer.transform([query_skeleton])
        cosine_similarities = cosine_similarity(
            query_vector, self.skeleton_vectors).flatten()

        top_k = min(top_k, len(self.questions))

        # Get top similar skeletons (excluding exact matches)
        top_indices = np.argpartition(cosine_similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(cosine_similarities[top_indices])[::-1]]

        similar_indices = [
            (idx, cosine_similarities[idx])
            for idx in top_indices
            if self.skeletons[idx] != query_skeleton
        ]

        edit_distance_results = similar_indices

        # Combine results
        results = exact_matches[:top_k]  # First add exact matches

        remaining_slots = top_k - len(results)
        if remaining_slots > 0:
            for idx, score in edit_distance_results[:remaining_slots]:
                results.append(
                    (self.questions[idx], self.skeletons[idx], score))

        return results[:top_k]
