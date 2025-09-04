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
from logger.logger import Logger


class StructuralSearchEngine:
    """Search engine for finding structurally similar questions."""

    def __init__(self):
        """Initialize the search engine."""
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Storage for indexed questions
        self.questions = []  # Original questions
        self.skeletons = []  # Structural skeletons
        self.skeleton_to_questions = defaultdict(
            list)  # Map skeletons to question indices

        # TF-IDF vectorizer for skeleton similarity
        self.vectorizer = None
        self.skeleton_vectors = None

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

    # pylint: disable-next=too-many-locals
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
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
        query_vector = self.vectorizer.transform([query_skeleton])
        cosine_similarities = cosine_similarity(
            query_vector, self.skeleton_vectors).flatten()

        # Get top similar skeletons (excluding exact matches)
        similar_indices = []
        for idx in np.argsort(cosine_similarities)[::-1]:
            if self.skeletons[idx] != query_skeleton:  # Exclude exact matches
                similar_indices.append((idx, cosine_similarities[idx]))

        # Method 3: Edit distance for very similar structures
        edit_distance_results = []
        # Check top_k by cosine similarity
        for idx, cosine_score in similar_indices[:top_k]:
            edit_dist = self.compute_edit_distance(
                query_skeleton, self.skeletons[idx])
            # Combine cosine similarity and edit distance
            combined_score = (cosine_score + (1.0 - edit_dist)) / 2.0
            edit_distance_results.append((idx, combined_score))

        # Sort by combined score and take top results
        edit_distance_results.sort(key=lambda x: x[1], reverse=True)

        # Combine results
        results = exact_matches[:top_k]  # First add exact matches

        remaining_slots = top_k - len(results)
        if remaining_slots > 0:
            for idx, score in edit_distance_results[:remaining_slots]:
                results.append(
                    (self.questions[idx], self.skeletons[idx], score))

        return results[:top_k]
