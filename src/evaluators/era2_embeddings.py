"""
Era 2: Embedding-Based Metrics (Vibe Checkers).

These metrics measure semantic similarity using contextual embeddings,
capturing paraphrasing and meaning beyond exact word matches.
"""

import os
from typing import Dict
import warnings

# CRITICAL: Force CPU before ANY imports that might trigger PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['USE_CUDA'] = '0'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global flag to track if MoverScore import has been attempted and failed
_MOVERSCORE_UNAVAILABLE = False
_MOVERSCORE_ERROR_MSG = None


def compute_bertscore(
    source: str,
    summary: str,
    model_type: str = "distilbert-base-uncased"
) -> Dict[str, float]:
    """
    Compute BERTScore using contextual embeddings.

    BERTScore calculates cosine similarity between BERT embeddings
    of the source and summary tokens, providing semantic similarity.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_type: The BERT model to use for embeddings.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1'.
    """
    try:
        from bert_score import score

        # Compute BERTScore
        P, R, F1 = score(
            [summary],
            [source],
            model_type=model_type,
            verbose=False,
            device='cpu'  # Use 'cuda' if GPU available
        )

        return {
            'precision': round(P.item(), 4),
            'recall': round(R.item(), 4),
            'f1': round(F1.item(), 4)
        }

    except Exception as e:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'error': str(e)
        }


def compute_moverscore(
    source: str,
    summary: str,
    model_name: str = "distilbert-base-uncased"
) -> Dict[str, float]:
    """
    Compute MoverScore using Earth Mover's Distance.

    MoverScore uses optimal transport (Earth Mover's Distance) to compute
    the semantic alignment between source and summary embeddings.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_name: The model to use for embeddings.

    Returns:
        Dictionary with key 'moverscore' containing the score.
    """
    global _MOVERSCORE_UNAVAILABLE, _MOVERSCORE_ERROR_MSG

    # If MoverScore import previously failed, return cached error immediately
    if _MOVERSCORE_UNAVAILABLE:
        return {
            'moverscore': 0.0,
            'error': _MOVERSCORE_ERROR_MSG or "MoverScore unavailable (previous import failed)"
        }

    # Use the safe wrapper to import MoverScore
    try:
        from collections import defaultdict

        # Import from wrapper instead of moverscore_v2 directly
        try:
            from .moverscore_wrapper import get_idf_dict, word_mover_score
        except (ImportError, RuntimeError, AssertionError) as import_error:
            # CUDA error during import
            error_str = str(import_error)
            _MOVERSCORE_UNAVAILABLE = True
            _MOVERSCORE_ERROR_MSG = (
                f"MoverScore unavailable: {error_str}\n"
                "Try running setup.sh again to reinstall dependencies."
            )
            return {
                'moverscore': 0.0,
                'error': _MOVERSCORE_ERROR_MSG
            }

        # Prepare texts (MoverScore expects list of strings, not list of word lists)
        references = [source]
        hypotheses = [summary]

        # Create IDF dictionary (uniform weights for simplicity)
        idf_dict_ref = defaultdict(lambda: 1.0)
        idf_dict_hyp = defaultdict(lambda: 1.0)

        # Compute MoverScore with explicit CPU device
        scores = word_mover_score(
            references,
            hypotheses,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=1,
            device='cpu'  # Force CPU usage
        )

        return {
            'moverscore': round(scores[0], 4)
        }

    except ImportError as e:
        return {
            'moverscore': 0.0,
            'error': (
                "MoverScore not installed. "
                "Run: pip3 install git+https://github.com/AIPHES/emnlp19-moverscore.git"
            )
        }
    except (AssertionError, RuntimeError) as e:
        # Handle CUDA-related errors during execution
        error_str = str(e)
        if "CUDA" in error_str or "cuda" in error_str.lower():
            return {
                'moverscore': 0.0,
                'error': (
                    "MoverScore CUDA error. The package has a known issue with CPU-only PyTorch. "
                    "Reinstall PyTorch: pip3 uninstall torch && pip3 install torch --index-url https://download.pytorch.org/whl/cpu"
                )
            }
        return {
            'moverscore': 0.0,
            'error': f"MoverScore error: {error_str}"
        }
    except Exception as e:
        error_msg = str(e)
        if "No module named" in error_msg or "cannot import" in error_msg:
            error_msg = "MoverScore not properly installed."
        elif "CUDA" in error_msg or "cuda" in error_msg.lower():
            error_msg = "MoverScore CUDA initialization error (known issue with CPU-only mode)."

        return {
            'moverscore': 0.0,
            'error': error_msg
        }


def compute_all_era2_metrics(source: str, summary: str) -> Dict[str, Dict[str, float]]:
    """
    Compute all Era 2 metrics at once.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with keys for each metric, containing their scores.
    """
    return {
        'BERTScore': compute_bertscore(source, summary),
        'MoverScore': compute_moverscore(source, summary)
    }
