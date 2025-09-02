"""
Unit tests for rouge_evaluator module.

Tests the ROUGE score evaluation functionality including edge cases,
error handling, and standard evaluation scenarios.
"""
import unittest
from typing import List, Tuple

from evaluator.rogue_evaluator import rouge_score, eval_rogue_score


class TestRougeScore(unittest.TestCase):
    """Test cases for the rouge_score function."""

    def test_rouge_score_perfect_match(self) -> None:
        """Test ROUGE score with perfect match between expected and actual."""
        expected = ["The quick brown fox"]
        actual = "The quick brown fox"

        scores = rouge_score(expected, actual)

        # Should have ROUGE-1 and ROUGE-2 scores
        self.assertEqual(len(scores), 2)

        # Perfect match should have F1, precision, and recall of 1.0
        rouge1_f1, rouge1_p, rouge1_r = scores[0]
        self.assertAlmostEqual(rouge1_f1, 1.0, places=2)
        self.assertAlmostEqual(rouge1_p, 1.0, places=2)
        self.assertAlmostEqual(rouge1_r, 1.0, places=2)

    def test_rouge_score_partial_match(self) -> None:
        """Test ROUGE score with partial match."""
        expected = ["The quick brown fox jumps over the lazy dog"]
        actual = "The quick brown fox"

        scores = rouge_score(expected, actual)

        # Should have ROUGE-1 and ROUGE-2 scores
        self.assertEqual(len(scores), 2)

        # Partial match should have scores between 0 and 1
        rouge1_f1, rouge1_p, rouge1_r = scores[0]
        self.assertGreater(rouge1_f1, 0.0)
        self.assertLess(rouge1_f1, 1.0)
        self.assertGreater(rouge1_p, 0.0)
        self.assertLessEqual(rouge1_p, 1.0)
        self.assertGreater(rouge1_r, 0.0)
        self.assertLessEqual(rouge1_r, 1.0)

    def test_rouge_score_no_match(self) -> None:
        """Test ROUGE score with no overlap between expected and actual."""
        expected = ["The quick brown fox"]
        actual = "Completely different words here"

        scores = rouge_score(expected, actual)

        # Should have ROUGE-1 and ROUGE-2 scores
        self.assertEqual(len(scores), 2)

        # No match should have scores of 0.0
        rouge1_f1, rouge1_p, rouge1_r = scores[0]
        self.assertEqual(rouge1_f1, 0.0)
        self.assertEqual(rouge1_p, 0.0)
        self.assertEqual(rouge1_r, 0.0)

    def test_rouge_score_multiple_expected_answers(self) -> None:
        """Test ROUGE score with multiple possible expected answers."""
        expected = ["The quick brown fox",
                    "A fast brown fox", "Quick brown animal"]
        actual = "The quick brown fox"

        scores = rouge_score(expected, actual)

        # Should pick the best match (first one in this case)
        rouge1_f1, _, _ = scores[0]
        self.assertAlmostEqual(rouge1_f1, 1.0, places=2)

    def test_rouge_score_empty_actual(self) -> None:
        """Test ROUGE score with empty actual answer."""
        expected = ["The quick brown fox"]
        actual = ""

        scores = rouge_score(expected, actual)

        # Empty actual should result in zero scores
        rouge1_f1, rouge1_p, rouge1_r = scores[0]
        self.assertEqual(rouge1_f1, 0.0)
        self.assertEqual(rouge1_p, 0.0)
        self.assertEqual(rouge1_r, 0.0)

    def test_rouge_score_empty_expected(self) -> None:
        """Test ROUGE score with empty expected answers."""
        expected = [""]
        actual = "Some answer"

        scores = rouge_score(expected, actual)

        # Empty expected should result in zero scores
        rouge1_f1, rouge1_p, rouge1_r = scores[0]
        self.assertEqual(rouge1_f1, 0.0)
        self.assertEqual(rouge1_p, 0.0)
        self.assertEqual(rouge1_r, 0.0)

    def test_rouge_score_whitespace_handling(self) -> None:
        """Test ROUGE score handles whitespace properly."""
        expected = ["  The   quick  brown   fox  "]
        actual = "The quick brown fox"

        scores = rouge_score(expected, actual)

        # Should handle whitespace and still get perfect match
        rouge1_f1, _, _ = scores[0]
        self.assertAlmostEqual(rouge1_f1, 1.0, places=2)

    def test_rouge_score_case_sensitivity(self) -> None:
        """Test ROUGE score case handling."""
        expected = ["THE QUICK BROWN FOX"]
        actual = "the quick brown fox"

        scores = rouge_score(expected, actual)

        # Should handle case differences (tokenizer should normalize)
        rouge1_f1, _, _ = scores[0]
        # Should be high similarity
        self.assertAlmostEqual(rouge1_f1, 1.0, places=2)

    def test_rouge_score_punctuation_handling(self) -> None:
        """Test ROUGE score punctuation handling."""
        expected = ["The quick, brown fox!"]
        actual = "The quick brown fox"

        scores = rouge_score(expected, actual)

        # Should handle punctuation differences
        rouge1_f1, _, _ = scores[0]
        self.assertAlmostEqual(rouge1_f1, 1.0, places=2)


class TestEvalRougeScore(unittest.TestCase):
    """Test cases for the eval_rogue_score function."""

    def test_eval_rogue_score_single_pair(self) -> None:
        """Test eval_rogue_score with a single QA pair."""
        qa_pairs = [(["The quick brown fox"], "The quick brown fox")]

        result = eval_rogue_score(qa_pairs)

        # Should return ROUGE-1 and ROUGE-2 averages
        self.assertEqual(len(result), 2)

        # Each result should be a tuple of (f1, precision, recall)
        rouge1_avg = result[0]
        rouge2_avg = result[1]

        self.assertEqual(len(rouge1_avg), 3)
        self.assertEqual(len(rouge2_avg), 3)

        # Perfect match should have high scores
        self.assertAlmostEqual(rouge1_avg[0], 1.0, places=2)  # F1
        self.assertAlmostEqual(rouge1_avg[1], 1.0, places=2)  # Precision
        self.assertAlmostEqual(rouge1_avg[2], 1.0, places=2)  # Recall

    def test_eval_rogue_score_multiple_pairs(self) -> None:
        """Test eval_rogue_score with multiple QA pairs."""
        qa_pairs = [
            (["The quick brown fox"], "The quick brown fox"),
            (["Hello world"], "Hello world"),
            (["Python programming"], "Python coding")
        ]

        result = eval_rogue_score(qa_pairs)

        # Should return ROUGE-1 and ROUGE-2 averages
        self.assertEqual(len(result), 2)

        rouge1_avg = result[0]

        # Should have reasonable scores (not perfect due to third pair)
        self.assertGreater(rouge1_avg[0], 0.5)
        self.assertLess(rouge1_avg[0], 1.0)

    def test_eval_rogue_score_empty_list(self) -> None:
        """Test eval_rogue_score with empty QA pairs list."""
        qa_pairs: List[Tuple[List[str], str]] = []

        with self.assertRaises(ZeroDivisionError):
            eval_rogue_score(qa_pairs)

    def test_eval_rogue_score_mixed_quality(self) -> None:
        """Test eval_rogue_score with mixed quality answers."""
        qa_pairs = [
            (["Perfect match"], "Perfect match"),
            (["No match at all"], "Completely different"),
            (["Partial overlap"], "Partial similarity")
        ]

        result = eval_rogue_score(qa_pairs)

        rouge1_avg = result[0]

        # Average should be between 0 and 1
        self.assertGreater(rouge1_avg[0], 0.0)
        self.assertLess(rouge1_avg[0], 1.0)

    def test_eval_rogue_score_all_zeros(self) -> None:
        """Test eval_rogue_score when all pairs have no overlap."""
        qa_pairs = [
            (["First answer"], "Unrelated response"),
            (["Second answer"], "Different response"),
            (["Third answer"], "Another response")
        ]

        result = eval_rogue_score(qa_pairs)

        rouge1_avg = result[0]

        # Should have very low or zero scores
        self.assertLessEqual(rouge1_avg[0], 0.1)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
