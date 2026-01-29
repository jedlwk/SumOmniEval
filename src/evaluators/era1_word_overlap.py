"""
Era 1: Word Overlap Metrics (Bean Counters).

These metrics measure text similarity through exact word matching,
with some variants supporting synonyms and stemming.
"""

from typing import Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global cache for GPT-2 perplexity model (~600MB)
_perplexity_model = None
_perplexity_tokenizer = None


def compute_rouge_scores(source: str, summary: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    overlap of n-grams between the summary and source.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with keys: 'rouge1', 'rouge2', 'rougeL' (F1 scores).
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        scores = scorer.score(source, summary)

        return {
            'rouge1': round(scores['rouge1'].fmeasure, 4),
            'rouge2': round(scores['rouge2'].fmeasure, 4),
            'rougeL': round(scores['rougeL'].fmeasure, 4)
        }

    except Exception as e:
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'error': str(e)
        }


def compute_bleu_score(source: str, summary: str) -> Dict[str, float]:
    """
    Compute BLEU score.

    BLEU (Bilingual Evaluation Understudy) measures precision of n-grams,
    originally designed for machine translation evaluation.

    Args:
        source: The original source text (treated as reference).
        summary: The generated summary (treated as hypothesis).

    Returns:
        Dictionary with key 'bleu' containing the score.
    """
    try:
        from sacrebleu import sentence_bleu

        # BLEU expects references as a list
        score = sentence_bleu(summary, [source])

        return {
            'bleu': round(score.score / 100, 4)  # Normalize to 0-1
        }

    except Exception as e:
        return {
            'bleu': 0.0,
            'error': str(e)
        }


def compute_meteor_score(source: str, summary: str) -> Dict[str, float]:
    """
    Compute METEOR score.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    extends BLEU by considering synonyms, stemming, and word order.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with key 'meteor' containing the score.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        import nltk
        import ssl

        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Ensure required NLTK data is available
        required_data = [
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]

        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                except:
                    pass  # Continue even if download fails

        # Tokenize texts
        source_tokens = word_tokenize(source.lower())
        summary_tokens = word_tokenize(summary.lower())

        # Compute METEOR
        score = meteor_score([source_tokens], summary_tokens)

        return {
            'meteor': round(score, 4)
        }

    except Exception as e:
        error_msg = str(e)
        if 'punkt_tab' in error_msg:
            error_msg = "NLTK punkt_tab missing. Run: python3 -c \"import nltk; nltk.download('punkt_tab')\""
        return {
            'meteor': 0.0,
            'error': error_msg
        }


def compute_levenshtein_score(source: str, summary: str) -> Dict[str, float]:
    """
    Compute normalized Levenshtein distance.

    Levenshtein distance measures the minimum number of single-character
    edits needed to transform one string into another. This is normalized
    to a 0-1 similarity score.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with key 'levenshtein' containing normalized similarity.
    """
    try:
        import Levenshtein

        # Compute distance
        distance = Levenshtein.distance(source, summary)

        # Normalize to similarity score (0-1)
        max_length = max(len(source), len(summary))
        if max_length == 0:
            similarity = 1.0
        else:
            similarity = 1 - (distance / max_length)

        return {
            'levenshtein': round(similarity, 4)
        }

    except Exception as e:
        return {
            'levenshtein': 0.0,
            'error': str(e)
        }


def compute_perplexity(source: str, summary: str) -> Dict[str, float]:
    """
    Compute Perplexity score for the summary.

    Perplexity measures how "surprised" a language model is by the text.
    Lower perplexity indicates more fluent, natural text.
    Note: This measures fluency, not factual accuracy.

    Args:
        source: The original source text (for context).
        summary: The generated summary to evaluate.

    Returns:
        Dictionary with key 'perplexity' containing the score.
    """
    global _perplexity_model, _perplexity_tokenizer

    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch
        import math

        # Load GPT-2 model (cached globally to avoid reload overhead)
        if _perplexity_model is None:
            model_id = "gpt2"
            _perplexity_model = GPT2LMHeadModel.from_pretrained(model_id)
            _perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            _perplexity_model.eval()

        model = _perplexity_model
        tokenizer = _perplexity_tokenizer

        # Encode the summary
        encodings = tokenizer(summary, return_tensors="pt", truncation=True, max_length=512)

        # Calculate perplexity
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        # Normalize perplexity to 0-1 scale (lower is better)
        # Use inverse and normalize: good perplexity < 50, bad > 200
        normalized = 1 / (1 + math.log(perplexity))

        return {
            'perplexity': round(perplexity, 4),
            'normalized_score': round(normalized, 4)
        }

    except Exception as e:
        return {
            'perplexity': 0.0,
            'normalized_score': 0.0,
            'error': str(e)
        }


def compute_chrf_score(reference: str, summary: str) -> Dict[str, float]:
    """
    Compute chrF++ score (character n-gram F-score).

    chrF++ uses character-level matching, making it more robust to:
    - Morphological variations (word endings)
    - Typos and minor spelling differences
    - Languages with rich morphology

    Args:
        reference: The reference text.
        summary: The generated summary.

    Returns:
        Dictionary with chrF++ score.
    """
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF(word_order=2)  # chrF++ includes word bigrams
        score = chrf.sentence_score(summary, [reference])

        return {
            'chrf': round(score.score / 100, 4),  # Normalize to 0-1
            'raw_score': round(score.score, 2)
        }

    except ImportError:
        return {
            'chrf': 0.0,
            'error': 'sacrebleu not installed. Run: pip install sacrebleu'
        }
    except Exception as e:
        return {
            'chrf': 0.0,
            'error': str(e)
        }


def compute_all_era1_metrics(source: str, summary: str) -> Dict[str, Dict[str, float]]:
    """
    Compute all Era 1 metrics at once (Lexical Conformance).

    Args:
        source: The reference text (for reference-based comparison).
        summary: The generated summary.

    Returns:
        Dictionary with keys for each metric, containing their scores.
    """
    return {
        'ROUGE': compute_rouge_scores(source, summary),
        'BLEU': compute_bleu_score(source, summary),
        'METEOR': compute_meteor_score(source, summary),
        'chrF++': compute_chrf_score(source, summary),
        'Levenshtein': compute_levenshtein_score(source, summary),
        'Perplexity': compute_perplexity(source, summary)
    }
