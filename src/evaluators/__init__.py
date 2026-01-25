"""Evaluation metrics for text summarization quality assessment."""

from .era1_word_overlap import (
    compute_rouge_scores,
    compute_bleu_score,
    compute_meteor_score,
    compute_levenshtein_score,
    compute_perplexity
)
from .era2_embeddings import (
    compute_bertscore,
    compute_moverscore
)
from .era3_logic_checkers import (
    compute_nli_score
)

__all__ = [
    # Era 1: Word Overlap & Fluency
    'compute_rouge_scores',
    'compute_bleu_score',
    'compute_meteor_score',
    'compute_levenshtein_score',
    'compute_perplexity',
    # Era 2: Embeddings
    'compute_bertscore',
    'compute_moverscore',
    # Era 3: Logic Checkers
    'compute_nli_score'
]
