"""
UniEval: Unified Multi-Dimensional Evaluator.

UniEval is a unified evaluator that converts evaluation tasks into Boolean QA problems.
It provides multi-dimensional evaluation: coherence, consistency, fluency.

This serves as the backup for BLEURT (which has TensorFlow conflicts).

Paper: "Towards a Unified Multi-Dimensional Evaluator for Text Generation" (EMNLP 2022)
Model: MingZhong/unieval-sum (~1.5GB)
"""

import os
from typing import Dict, List, Optional
import warnings

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore')

# Global cache for UniEval evaluator (lazy loaded)
_unieval_evaluator = None


def compute_unieval(
    source: str,
    summary: str,
    dimensions: Optional[List[str]] = None
) -> Dict:
    """
    Compute UniEval scores for summarization evaluation.

    UniEval evaluates multiple dimensions:
    - coherence: Is the text logically connected?
    - consistency: Does the summary align with the source? (factuality)
    - fluency: Is the text natural and well-written?

    Args:
        source: The original source text.
        summary: The generated summary to evaluate.
        dimensions: List of dimensions to evaluate. Default: all three.

    Returns:
        Dictionary with scores for each dimension and interpretations.
    """
    global _unieval_evaluator

    if dimensions is None:
        dimensions = ["coherence", "consistency", "fluency"]

    try:
        # Try to import UniEval
        from UniEval.utils import convert_to_json
        from UniEval.metric.evaluator import get_evaluator

        # Load evaluator if not cached
        if _unieval_evaluator is None:
            _unieval_evaluator = get_evaluator("summarization", device="cpu")

        # Format data for UniEval
        data = convert_to_json(
            output_list=[summary],
            src_list=[source]
        )

        # Evaluate
        eval_scores = _unieval_evaluator.evaluate(
            data,
            dims=dimensions,
            overall=False,
            print_result=False
        )

        # Extract scores from first result
        result = eval_scores[0] if eval_scores else {}

        return {
            'coherence': round(result.get('coherence', 0), 4),
            'consistency': round(result.get('consistency', 0), 4),
            'fluency': round(result.get('fluency', 0), 4),
            'interpretations': {
                'coherence': _interpret_unieval_score(result.get('coherence')),
                'consistency': _interpret_unieval_score(result.get('consistency')),
                'fluency': _interpret_unieval_score(result.get('fluency'))
            }
        }

    except ImportError:
        # UniEval not installed - try alternative approach using transformers directly
        return _compute_unieval_fallback(source, summary, dimensions)

    except Exception as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': str(e)
        }


def _compute_unieval_fallback(
    source: str,
    summary: str,
    dimensions: List[str]
) -> Dict:
    """
    Fallback implementation using transformers directly.

    Uses the UniEval model from HuggingFace with direct inference.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "MingZhong/unieval-sum"

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

        results = {}

        # UniEval uses Boolean QA format for each dimension
        dimension_questions = {
            'coherence': f"question: Is this a coherent summary? </s> summary: {summary}",
            'consistency': f"question: Is this summary consistent with the document? </s> document: {source} </s> summary: {summary}",
            'fluency': f"question: Is this summary fluent? </s> summary: {summary}"
        }

        for dim in dimensions:
            if dim not in dimension_questions:
                continue

            input_text = dimension_questions[dim]

            # Truncate if too long
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=1
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

            # UniEval outputs "yes" or "no" - convert to score
            if "yes" in response:
                score = 1.0
            elif "no" in response:
                score = 0.0
            else:
                # Try to extract probability-based score
                score = 0.5

            results[dim] = round(score, 4)

        return {
            'coherence': results.get('coherence', None),
            'consistency': results.get('consistency', None),
            'fluency': results.get('fluency', None),
            'interpretations': {
                'coherence': _interpret_unieval_score(results.get('coherence')),
                'consistency': _interpret_unieval_score(results.get('consistency')),
                'fluency': _interpret_unieval_score(results.get('fluency'))
            },
            'note': 'Using fallback implementation (UniEval library not installed)'
        }

    except ImportError as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': f'transformers not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'coherence': None,
            'consistency': None,
            'fluency': None,
            'error': str(e)
        }


def _interpret_unieval_score(score: Optional[float]) -> str:
    """Interpret UniEval dimension scores (0-1 scale)."""
    if score is None:
        return "N/A"
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Fair"
    elif score >= 0.30:
        return "Poor"
    else:
        return "Very Poor"


def compute_all_unieval_metrics(source: str, summary: str) -> Dict[str, Dict]:
    """
    Compute all UniEval metrics.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with UniEval results.
    """
    return {
        'UniEval': compute_unieval(source, summary)
    }
