"""
Era 3 Group A: Logic Checkers (Truth Squad).

This module implements logic-based consistency checking:
- NLI (DeBERTa-v3): Local Natural Language Inference
- AlignScore: Unified alignment-based factual consistency metric
- FactChecker (API): API-based fact-checking using LLMs
"""

import os
from typing import Dict, Optional
import warnings

# Force CPU mode for PyTorch/Transformers
os.environ['CUDA_VISIBLE_DEVICES'] = ''

warnings.filterwarnings('ignore')

# Global model caches to avoid reloading on each call
_nli_pipeline = None
_factcc_pipeline = None


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
    global _nli_pipeline

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # Load NLI model (cached globally)
        if _nli_pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # Force CPU
            )

        nli_pipeline = _nli_pipeline

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

    Note: Uses DeBERTa-base-MNLI as the original FactCC checkpoint has
    compatibility issues. This provides similar functionality.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with 'score' (0-1) and 'label'.
    """
    global _factcc_pipeline

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        # Load FactCC model (cached globally)
        # Uses deberta-base-mnli as alternative to original FactCC checkpoint
        if _factcc_pipeline is None:
            model_name = "microsoft/deberta-base-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _factcc_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # Force CPU
            )

        classifier = _factcc_pipeline

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


# Global AlignScore model cache to avoid reloading
_alignscore_model = None
_alignscore_tokenizer = None


def compute_alignscore(
    source: str,
    summary: str,
    model_name: str = "liuyanyi/AlignScore-large-hf"
) -> Dict:
    """
    Compute AlignScore for factual consistency evaluation.

    AlignScore is a unified alignment-based factual consistency metric that
    achieves state-of-the-art performance across multiple benchmarks. It uses
    a RoBERTa model fine-tuned on diverse alignment tasks.

    Paper: "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function"

    Uses the HuggingFace model from: https://huggingface.co/liuyanyi/AlignScore-large-hf

    Args:
        source: The original source text (context/premise).
        summary: The generated summary (claim).
        model_name: HuggingFace model name (default: liuyanyi/AlignScore-large-hf).

    Returns:
        Dictionary with 'score' (0-1), higher = more factually consistent.
    """
    global _alignscore_model, _alignscore_tokenizer

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, RobertaTokenizer

        # Load model and tokenizer if not cached
        if _alignscore_model is None:
            _alignscore_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            _alignscore_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            _alignscore_model.eval()

        # Tokenize inputs
        inputs = _alignscore_tokenizer(
            source,
            summary,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Get score
        with torch.no_grad():
            outputs = _alignscore_model(**inputs)
            score = outputs.reg_label_logits.item()

        return {
            'score': round(float(score), 4),
            'interpretation': _interpret_alignscore(float(score))
        }

    except ImportError as e:
        return {
            'score': None,
            'error': f'Required packages not installed. Error: {str(e)}'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_alignscore(score: float) -> str:
    """Interpret AlignScore for human readability."""
    if score >= 0.9:
        return "Fully Consistent"
    elif score >= 0.7:
        return "Highly Consistent"
    elif score >= 0.5:
        return "Mostly Consistent"
    elif score >= 0.3:
        return "Partially Consistent"
    else:
        return "Inconsistent"


def compute_coverage_score(source: str, summary: str) -> Dict:
    """
    Compute Coverage Score using Named Entity Recognition (NER).

    Measures how many named entities (People, Places, Organizations, Dates, etc.)
    from the source document appear in the summary. High coverage indicates
    the summary captures the key factual elements.

    Args:
        source: The original source text.
        summary: The generated summary.

    Returns:
        Dictionary with coverage metrics.
    """
    try:
        import spacy

        # Load spaCy English model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed
            return {
                'score': None,
                'error': 'spaCy model not installed. Run: python -m spacy download en_core_web_sm'
            }

        # Extract entities from source
        source_doc = nlp(source)
        source_entities = set()
        for ent in source_doc.ents:
            source_entities.add(ent.text.lower())

        if not source_entities:
            return {
                'score': 1.0,  # If no entities in source, summary trivially covers all
                'source_entities': 0,
                'covered_entities': 0,
                'interpretation': 'No named entities in source'
            }

        # Extract entities from summary
        summary_doc = nlp(summary)
        summary_entities = set()
        for ent in summary_doc.ents:
            summary_entities.add(ent.text.lower())

        # Calculate coverage
        covered = source_entities.intersection(summary_entities)
        coverage_score = len(covered) / len(source_entities)

        return {
            'score': round(coverage_score, 4),
            'source_entities': len(source_entities),
            'covered_entities': len(covered),
            'missing_entities': list(source_entities - summary_entities)[:5],  # Show up to 5 missing
            'interpretation': _interpret_coverage_score(coverage_score)
        }

    except ImportError:
        return {
            'score': None,
            'error': 'spaCy not installed. Run: pip install spacy'
        }
    except Exception as e:
        return {
            'score': None,
            'error': str(e)
        }


def _interpret_coverage_score(score: float) -> str:
    """Interpret Coverage Score for human readability."""
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


def compute_all_era3_metrics(
    source: str,
    summary: str,
    use_factchecker: bool = False,
    use_factcc: bool = False,
    use_alignscore: bool = False,
    use_coverage: bool = False,
    use_unieval: bool = False,
    factchecker_model: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute all Era 3 Group A metrics (Faithfulness checks).

    Args:
        source: The original source text.
        summary: The generated summary.
        use_factchecker: Whether to use API-based fact-checker.
        use_factcc: Whether to use FactCC (BERT-based).
        use_alignscore: Whether to use AlignScore (unified alignment metric).
        use_coverage: Whether to use Coverage Score (NER overlap).
        use_unieval: Whether to use UniEval (multi-dimensional evaluator).
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

    # Add AlignScore if enabled
    if use_alignscore:
        results['AlignScore'] = compute_alignscore(source, summary)

    # Add Coverage Score if enabled (for completeness check)
    if use_coverage:
        results['Coverage'] = compute_coverage_score(source, summary)

    # Add UniEval if enabled (BLEURT backup - multi-dimensional evaluation)
    if use_unieval:
        from src.evaluators.era3_unieval import compute_unieval
        results['UniEval'] = compute_unieval(source, summary)

    # Add API-based fact-checker if enabled
    if use_factchecker:
        results['FactChecker'] = compute_factchecker_score(
            source,
            summary,
            model_name=factchecker_model,
            use_api=True
        )

    return results
