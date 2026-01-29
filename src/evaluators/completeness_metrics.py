"""
Part 1 - Completeness Metrics (Substance Check).

These metrics ensure key information and main points are captured.
Comparison: Generated Summary ↔ Source Text

Metrics:
- Semantic Coverage: Sentence-level embedding similarity
- BERTScore Recall (vs Source): What fraction of source is in summary
- BARTScore: Probability-based coverage P(Summary|Source)
"""

import os
from typing import Dict, List
import warnings

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore')


def compute_semantic_coverage(
    source: str,
    summary: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.7
) -> Dict:
    """
    Compute Semantic Coverage using sentence embeddings.

    Measures how well summary sentences semantically cover key source sentences.
    Uses sentence-transformers to compute pairwise similarity.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_name: Sentence transformer model (default: all-MiniLM-L6-v2, ~80MB).
        threshold: Similarity threshold to consider a sentence "covered" (default: 0.7).

    Returns:
        Dictionary with coverage score and details.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Split into sentences
        def split_sentences(text):
            """Simple sentence splitting."""
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]

        source_sentences = split_sentences(source)
        summary_sentences = split_sentences(summary)

        if not source_sentences:
            return {
                'score': 1.0,
                'source_sentences': 0,
                'covered_sentences': 0,
                'interpretation': 'No sentences in source'
            }

        if not summary_sentences:
            return {
                'score': 0.0,
                'source_sentences': len(source_sentences),
                'covered_sentences': 0,
                'interpretation': 'No sentences in summary'
            }

        # Load model
        model = SentenceTransformer(model_name)

        # Encode sentences
        source_embeddings = model.encode(source_sentences)
        summary_embeddings = model.encode(summary_sentences)

        # Compute coverage: for each source sentence, find max similarity to any summary sentence
        covered_count = 0
        coverage_details = []

        for i, src_emb in enumerate(source_embeddings):
            # Cosine similarity with all summary sentences
            similarities = np.dot(summary_embeddings, src_emb) / (
                np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(src_emb)
            )
            max_sim = float(np.max(similarities))
            is_covered = max_sim >= threshold

            if is_covered:
                covered_count += 1

            coverage_details.append({
                'sentence': source_sentences[i][:50] + '...' if len(source_sentences[i]) > 50 else source_sentences[i],
                'max_similarity': round(max_sim, 3),
                'covered': is_covered
            })

        coverage_score = covered_count / len(source_sentences)

        return {
            'score': round(coverage_score, 4),
            'source_sentences': len(source_sentences),
            'covered_sentences': covered_count,
            'threshold': threshold,
            'interpretation': _interpret_semantic_coverage(coverage_score)
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'sentence-transformers not installed. Run: pip install sentence-transformers. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_semantic_coverage(score: float) -> str:
    """Interpret Semantic Coverage score."""
    if score >= 0.9:
        return "Excellent Coverage"
    elif score >= 0.7:
        return "Good Coverage"
    elif score >= 0.5:
        return "Partial Coverage"
    elif score >= 0.3:
        return "Low Coverage"
    else:
        return "Poor Coverage"


def compute_bertscore_recall_source(
    source: str,
    summary: str
) -> Dict:
    """
    Compute BERTScore Recall comparing Summary against Source.

    This repurposes BERTScore for completeness checking:
    - High recall = summary captures most of the source content
    - Low recall = summary misses key information

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with recall score and interpretation.
    """
    try:
        from bert_score import score as bert_score

        # BERTScore expects references and candidates
        # For completeness: how much of source is captured in summary
        # We use summary as candidate, source as reference
        P, R, F1 = bert_score(
            [summary],  # candidates
            [source],   # references
            lang='en',
            rescale_with_baseline=True,
            device='cpu'
        )

        recall = float(R[0])

        return {
            'recall': round(recall, 4),
            'precision': round(float(P[0]), 4),
            'f1': round(float(F1[0]), 4),
            'interpretation': _interpret_bertscore_recall(recall)
        }

    except ImportError:
        return {
            'recall': None,
            'error': 'bert-score not installed. Run: pip install bert-score'
        }
    except Exception as e:
        return {
            'recall': None,
            'error': str(e)
        }


def _interpret_bertscore_recall(score: float) -> str:
    """Interpret BERTScore Recall for completeness."""
    if score >= 0.5:
        return "High Coverage"
    elif score >= 0.3:
        return "Moderate Coverage"
    elif score >= 0.1:
        return "Low Coverage"
    else:
        return "Very Low Coverage"


def compute_bartscore(
    source: str,
    summary: str,
    model_name: str = "facebook/bart-large-cnn"
) -> Dict:
    """
    Compute BARTScore for coverage assessment.

    BARTScore measures the log probability of generating the summary
    given the source. Higher score = better information coverage.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_name: BART model to use (default: bart-large-cnn, ~1.6GB).

    Returns:
        Dictionary with BARTScore and interpretation.
    """
    try:
        import torch
        from transformers import BartTokenizer, BartForConditionalGeneration

        # Load model and tokenizer
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        model.eval()

        # Tokenize
        inputs = tokenizer(
            source,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        outputs = tokenizer(
            summary,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        # Compute log probability of summary given source
        with torch.no_grad():
            # Get model output
            result = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=outputs['input_ids']
            )
            # Negative log likelihood loss
            nll = result.loss.item()

        # Convert to score (higher is better)
        # BARTScore is typically the negative of the loss
        bartscore = -nll

        return {
            'score': round(bartscore, 4),
            'interpretation': _interpret_bartscore(bartscore)
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'transformers not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_bartscore(score: float) -> str:
    """Interpret BARTScore."""
    # BARTScore is negative log likelihood, so higher (less negative) is better
    if score >= -1.0:
        return "Excellent"
    elif score >= -2.0:
        return "Good"
    elif score >= -3.0:
        return "Moderate"
    else:
        return "Poor"


def compute_all_completeness_metrics(
    source: str,
    summary: str,
    use_semantic_coverage: bool = True,
    use_bertscore_recall: bool = True,
    use_bartscore: bool = False  # Disabled by default (large model)
) -> Dict[str, Dict]:
    """
    Compute all Part 1 Completeness metrics.

    Args:
        source: The original source text.
        summary: The generated summary.
        use_semantic_coverage: Whether to compute Semantic Coverage.
        use_bertscore_recall: Whether to compute BERTScore Recall.
        use_bartscore: Whether to compute BARTScore (requires ~1.6GB model).

    Returns:
        Dictionary with all completeness metric results.
    """
    results = {}

    if use_semantic_coverage:
        results['SemanticCoverage'] = compute_semantic_coverage(source, summary)

    if use_bertscore_recall:
        results['BERTScoreRecall'] = compute_bertscore_recall_source(source, summary)

    if use_bartscore:
        results['BARTScore'] = compute_bartscore(source, summary)

    return results
