"""
Unit tests for evaluation metrics.
Run with: python -m pytest tests/
"""

# CRITICAL: Import force_cpu FIRST to enforce CPU-only mode during tests
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import force_cpu  # noqa: F401 - imported for side effects

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluators.era1_word_overlap import (
    compute_rouge_scores,
    compute_bleu_score,
    compute_meteor_score,
    compute_levenshtein_score
)
from src.evaluators.era2_embeddings import (
    compute_bertscore,
    compute_moverscore
)
from src.evaluators.era3_logic_checkers import compute_nli_score


# Test data
SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."
SUMMARY_IDENTICAL = "The quick brown fox jumps over the lazy dog."
SUMMARY_SIMILAR = "A fast brown fox leaps over a lazy dog."
SUMMARY_DIFFERENT = "Cats and dogs are popular pets."


class TestEra1Metrics:
    """Tests for Era 1 word overlap metrics."""

    def test_rouge_identical(self):
        """Test ROUGE with identical texts."""
        scores = compute_rouge_scores(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'rouge1' in scores
        assert scores['rouge1'] == 1.0
        assert scores['rouge2'] == 1.0
        assert scores['rougeL'] == 1.0

    def test_rouge_similar(self):
        """Test ROUGE with similar texts."""
        scores = compute_rouge_scores(SOURCE_TEXT, SUMMARY_SIMILAR)
        assert 'rouge1' in scores
        assert 0 < scores['rouge1'] < 1

    def test_bleu_identical(self):
        """Test BLEU with identical texts."""
        scores = compute_bleu_score(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'bleu' in scores
        assert scores['bleu'] > 0.9

    def test_meteor_identical(self):
        """Test METEOR with identical texts."""
        scores = compute_meteor_score(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'meteor' in scores
        assert scores['meteor'] > 0.9

    def test_levenshtein_identical(self):
        """Test Levenshtein with identical texts."""
        scores = compute_levenshtein_score(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'levenshtein' in scores
        assert scores['levenshtein'] == 1.0

    def test_levenshtein_different(self):
        """Test Levenshtein with different texts."""
        scores = compute_levenshtein_score(SOURCE_TEXT, SUMMARY_DIFFERENT)
        assert 'levenshtein' in scores
        assert 0 <= scores['levenshtein'] < 1


class TestEra2Metrics:
    """Tests for Era 2 embedding-based metrics."""

    def test_bertscore_identical(self):
        """Test BERTScore with identical texts."""
        scores = compute_bertscore(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'f1' in scores
        if 'error' not in scores:
            assert scores['f1'] > 0.9

    def test_bertscore_similar(self):
        """Test BERTScore with similar texts."""
        scores = compute_bertscore(SOURCE_TEXT, SUMMARY_SIMILAR)
        if 'error' not in scores:
            assert 'f1' in scores
            assert 0 < scores['f1'] < 1

    def test_moverscore_format(self):
        """Test MoverScore returns expected format."""
        scores = compute_moverscore(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'moverscore' in scores
        # Note: MoverScore may not be installed, so we check for error or score
        if 'error' not in scores:
            assert isinstance(scores['moverscore'], float)


class TestEra3Metrics:
    """Tests for Era 3 logic checker metrics (NLI-based)."""

    def test_nli_format(self):
        """Test NLI score returns expected format."""
        scores = compute_nli_score(SOURCE_TEXT, SUMMARY_IDENTICAL)
        assert 'nli_score' in scores
        if 'error' not in scores:
            assert 0 <= scores['nli_score'] <= 1

    def test_nli_similar(self):
        """Test NLI with semantically similar texts."""
        scores = compute_nli_score(SOURCE_TEXT, SUMMARY_SIMILAR)
        if 'error' not in scores:
            assert 'nli_score' in scores
            assert isinstance(scores['nli_score'], float)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_texts(self):
        """Test handling of empty texts."""
        scores = compute_rouge_scores("", "")
        assert 'rouge1' in scores

    def test_very_long_text(self):
        """Test handling of very long texts."""
        long_text = " ".join(["word"] * 10000)
        scores = compute_rouge_scores(long_text, long_text[:100])
        assert 'rouge1' in scores

    def test_special_characters(self):
        """Test handling of special characters."""
        special_text = "Test @#$% with special & characters!"
        scores = compute_rouge_scores(special_text, special_text)
        assert 'rouge1' in scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
