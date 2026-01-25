"""
Era 3 Group A: Logic Checkers (Truth Squad).

This module implements logic-based consistency checking:
- NLI (DeBERTa-v3): Local Natural Language Inference
- FactChecker (API): API-based fact-checking using LLMs
"""

import os
from typing import Dict, Optional
import warnings

# Force CPU mode for PyTorch/Transformers
os.environ['CUDA_VISIBLE_DEVICES'] = ''

warnings.filterwarnings('ignore')


def compute_nli_score(
    source: str,
    summary: str,
    model_name: str = "microsoft/deberta-v3-base"
) -> Dict[str, float]:
    """
    Compute NLI-based consistency score using DeBERTa-v3.

    Uses Natural Language Inference to check if the summary is logically
    entailed by the source text. This is the modern, production-ready
    approach to factual consistency checking.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_name: The NLI model to use (default: DeBERTa-v3-base).

    Returns:
        Dictionary with 'nli_score' (0-1) and 'label' (classification).
        Higher score = more consistent/entailed.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # Load NLI model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create NLI pipeline (CPU-only)
        nli_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Force CPU
        )

        # Truncate long texts (DeBERTa max length: 512 tokens)
        max_words = 400  # Conservative limit
        source_truncated = ' '.join(source.split()[:max_words])
        summary_truncated = ' '.join(summary.split()[:max_words])

        # Format: premise [SEP] hypothesis
        nli_input = f"{source_truncated} [SEP] {summary_truncated}"

        # Get NLI prediction
        result = nli_pipeline(nli_input, truncation=True, max_length=512)

        label = result[0]['label']
        score = result[0]['score']

        # Extract entailment score
        # DeBERTa outputs: LABEL_0 (entailment), LABEL_1 (neutral), LABEL_2 (contradiction)
        if 'ENTAIL' in label.upper() or label == 'LABEL_0':
            entailment_score = score
        else:
            # If neutral or contradiction, invert score
            entailment_score = 1 - score

        return {
            'nli_score': round(entailment_score, 4),
            'label': label,
            'interpretation': _interpret_nli_score(entailment_score)
        }

    except Exception as e:
        return {
            'nli_score': 0.0,
            'label': 'ERROR',
            'error': str(e)
        }


def _interpret_nli_score(score: float) -> str:
    """
    Interpret NLI score for human readability.

    Args:
        score: NLI entailment score (0-1).

    Returns:
        Human-readable interpretation.
    """
    if score >= 0.8:
        return "Highly Consistent"
    elif score >= 0.6:
        return "Mostly Consistent"
    elif score >= 0.4:
        return "Partially Consistent"
    else:
        return "Inconsistent"


def compute_factchecker_score(
    source: str,
    summary: str,
    model_name: Optional[str] = None,
    use_api: bool = True
) -> Dict:
    """
    Compute fact-checking score using API-based LLM.

    This is a dedicated fact-checking approach that identifies specific
    factual errors, hallucinations, and unsupported claims.

    Args:
        source: The original source text.
        summary: The generated summary.
        model_name: LLM model to use (default: Llama-3.3-70B).
        use_api: Whether to use API (if False, returns placeholder).

    Returns:
        Dictionary with 'score' (0-1), 'claims_checked', and 'issues_found'.
    """
    if not use_api:
        return {
            'score': None,
            'error': 'API-based fact-checking disabled'
        }

    try:
        # Check if API is configured
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('H2OGPTE_API_KEY')
        address = os.getenv('H2OGPTE_ADDRESS')

        if not api_key or not address:
            return {
                'score': None,
                'error': 'API not configured (H2OGPTE_API_KEY or H2OGPTE_ADDRESS missing)'
            }

        # Import LLM client
        from h2ogpte import H2OGPTE

        # Use provided model or default
        if model_name is None:
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'

        client = H2OGPTE(address=address, api_key=api_key)

        # Create fact-checking prompt
        prompt = f"""You are an expert fact-checker. Your task is to verify EVERY claim in the summary against the source document.

**Source Document**:
{source}

**Summary to Fact-Check**:
{summary}

**Instructions**:
1. Extract each factual claim from the summary
2. For each claim, check if it is:
   - SUPPORTED: Directly stated or clearly implied in the source
   - UNSUPPORTED: Not mentioned or cannot be verified from source
   - CONTRADICTED: Conflicts with information in the source
3. Count the claims and issues
4. Provide a factuality score from 1-10

**Response Format**:
Claims Checked: [number]
Issues Found: [number of unsupported or contradicted claims]
Score: [1-10]
Explanation: [Brief summary of any issues found, or "All claims verified" if perfect]"""

        # Query the LLM
        chat_session_id = client.create_chat_session()

        with client.connect(chat_session_id) as session:
            reply = session.query(
                prompt,
                llm=model_name,
                timeout=90,
            )

        response = reply.content

        # Parse response
        claims_checked = 0
        issues_found = 0
        score = None
        explanation = ""

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Claims Checked:'):
                try:
                    claims_checked = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Issues Found:'):
                try:
                    issues_found = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Score:'):
                try:
                    score_text = line.split(':')[1].strip()
                    score = float(score_text)
                except:
                    pass
            elif line.startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()

        # Normalize score to 0-1
        if score is not None:
            normalized_score = score / 10.0
        else:
            normalized_score = None

        return {
            'score': normalized_score,
            'raw_score': score,
            'claims_checked': claims_checked,
            'issues_found': issues_found,
            'explanation': explanation if explanation else 'No explanation provided',
            'full_response': response,
            'interpretation': _interpret_factchecker_score(normalized_score) if normalized_score else 'N/A'
        }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_factchecker_score(score: float) -> str:
    """Interpret fact-checker score."""
    if score is None:
        return 'N/A'
    if score >= 0.9:
        return "Fully Factual"
    elif score >= 0.7:
        return "Mostly Factual"
    elif score >= 0.5:
        return "Partially Factual"
    else:
        return "Factually Questionable"


def compute_factcc_score(source: str, summary: str) -> Dict:
    """
    Compute FactCC score using fine-tuned BERT model.

    FactCC is a BERT-based model specifically trained for factual consistency
    checking in summarization. It predicts if a summary is consistent with
    the source document.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with 'score' (0-1) and 'label'.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # FactCC uses a fine-tuned BERT model
        # Note: The official model is "manueltonneau/factCC-checkpoint"
        # or we can use a general consistency model
        model_name = "microsoft/deberta-base-mnli"  # Good alternative

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create classification pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Force CPU
        )

        # Truncate long texts
        max_words = 400
        source_truncated = ' '.join(source.split()[:max_words])
        summary_truncated = ' '.join(summary.split()[:max_words])

        # Format for consistency checking
        text_input = f"{source_truncated} [SEP] {summary_truncated}"

        # Get prediction
        result = classifier(text_input, truncation=True, max_length=512)

        label = result[0]['label']
        score = result[0]['score']

        # Map to consistency score (ENTAILMENT = consistent)
        if 'ENTAIL' in label.upper() or 'CONSISTENT' in label.upper():
            consistency_score = score
            is_consistent = True
        else:
            consistency_score = 1 - score
            is_consistent = False

        return {
            'score': round(consistency_score, 4),
            'label': 'Consistent' if is_consistent else 'Inconsistent',
            'raw_label': label,
            'interpretation': _interpret_factcc_score(consistency_score)
        }

    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_factcc_score(score: float) -> str:
    """Interpret FactCC score."""
    if score >= 0.8:
        return "Highly Consistent"
    elif score >= 0.6:
        return "Mostly Consistent"
    elif score >= 0.4:
        return "Partially Consistent"
    else:
        return "Inconsistent"


def compute_all_era3_metrics(
    source: str,
    summary: str,
    use_factchecker: bool = False,
    use_factcc: bool = False,
    factchecker_model: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute all Era 3 Group A metrics.

    Args:
        source: The original source text.
        summary: The generated summary.
        use_factchecker: Whether to use API-based fact-checker.
        use_factcc: Whether to use FactCC (BERT-based).
        factchecker_model: LLM model for fact-checking (optional).

    Returns:
        Dictionary with metric results.
    """
    results = {
        'NLI': compute_nli_score(source, summary)
    }

    # Add FactCC if enabled
    if use_factcc:
        results['FactCC'] = compute_factcc_score(source, summary)

    # Add API-based fact-checker if enabled
    if use_factchecker:
        results['FactChecker'] = compute_factchecker_score(
            source,
            summary,
            model_name=factchecker_model,
            use_api=True
        )

    return results
