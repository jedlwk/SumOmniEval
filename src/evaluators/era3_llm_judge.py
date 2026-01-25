#!/usr/bin/env python3
"""
Era 3 Group B: LLM-as-a-Judge Evaluators (API-based)
Uses H2OGPTE API to evaluate summaries using powerful LLMs.

Implements G-Eval style evaluation with structured prompts.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMJudgeEvaluator:
    """
    LLM-as-a-Judge evaluator using H2OGPTE API.

    Evaluates summaries across multiple dimensions:
    - Faithfulness: Is the summary factually correct?
    - Coherence: Does the summary flow logically?
    - Relevance: Does it capture the main points?
    - Fluency: Is it well-written and grammatical?
    """

    def __init__(self, model_name: str = 'meta-llama/Llama-3.3-70B-Instruct'):
        """
        Initialize the LLM Judge evaluator.

        Args:
            model_name: The H2OGPTE model to use for evaluation
        """
        self.model_name = model_name
        self.api_key = os.getenv('H2OGPTE_API_KEY')
        self.address = os.getenv('H2OGPTE_ADDRESS')

        if not self.api_key or not self.address:
            raise ValueError(
                "H2OGPTE_API_KEY and H2OGPTE_ADDRESS must be set in .env file"
            )

        # Lazy import to avoid loading if not needed
        self.client = None

    def _get_client(self):
        """Lazy load the H2OGPTE client."""
        if self.client is None:
            from h2ogpte import H2OGPTE
            self.client = H2OGPTE(
                address=self.address,
                api_key=self.api_key,
            )
        return self.client

    def _create_faithfulness_prompt(self, source: str, summary: str) -> str:
        """
        Create a prompt to evaluate faithfulness/factual consistency.

        Based on G-Eval methodology.
        """
        return f"""You are an expert evaluator for text summarization. Your task is to evaluate the FAITHFULNESS of a summary.

**Faithfulness Definition**: A summary is faithful if all claims and facts in it are directly supported by the source document. Faithful summaries contain no hallucinations, contradictions, or unsupported claims.

**Evaluation Criteria**:
- Score 1-2: Multiple unsupported claims or contradictions
- Score 3-4: Some claims not fully supported by source
- Score 5-6: Mostly faithful with minor issues
- Score 7-8: Highly faithful, all major claims supported
- Score 9-10: Perfect faithfulness, every claim verifiable

**Source Document**:
{source}

**Summary to Evaluate**:
{summary}

**Instructions**:
1. Check each claim in the summary against the source
2. Identify any unsupported claims or contradictions
3. Rate faithfulness from 1-10
4. Provide a brief explanation (1-2 sentences)

**Response Format**:
Score: [1-10]
Explanation: [Your reasoning]"""

    def _create_coherence_prompt(self, summary: str) -> str:
        """Create a prompt to evaluate coherence."""
        return f"""You are an expert evaluator for text summarization. Your task is to evaluate the COHERENCE of a summary.

**Coherence Definition**: A coherent summary flows logically, with clear connections between sentences and ideas. It is well-structured and easy to follow.

**Evaluation Criteria**:
- Score 1-2: Disjointed, confusing structure
- Score 3-4: Some logical flow issues
- Score 5-6: Adequate flow, minor gaps
- Score 7-8: Good logical structure
- Score 9-10: Excellent flow, perfectly structured

**Summary to Evaluate**:
{summary}

**Instructions**:
1. Assess the logical flow between sentences
2. Check for clear topic progression
3. Rate coherence from 1-10
4. Provide a brief explanation (1-2 sentences)

**Response Format**:
Score: [1-10]
Explanation: [Your reasoning]"""

    def _create_relevance_prompt(self, source: str, summary: str) -> str:
        """Create a prompt to evaluate relevance."""
        return f"""You are an expert evaluator for text summarization. Your task is to evaluate the RELEVANCE of a summary.

**Relevance Definition**: A relevant summary captures the most important information from the source document without including trivial or off-topic details.

**Evaluation Criteria**:
- Score 1-2: Misses main points, includes irrelevant info
- Score 3-4: Captures some key points, has extraneous content
- Score 5-6: Covers main points adequately
- Score 7-8: Captures all important information
- Score 9-10: Perfect selection of key information

**Source Document**:
{source}

**Summary to Evaluate**:
{summary}

**Instructions**:
1. Identify the main points in the source
2. Check if the summary captures them
3. Rate relevance from 1-10
4. Provide a brief explanation (1-2 sentences)

**Response Format**:
Score: [1-10]
Explanation: [Your reasoning]"""

    def _create_fluency_prompt(self, summary: str) -> str:
        """Create a prompt to evaluate fluency."""
        return f"""You are an expert evaluator for text summarization. Your task is to evaluate the FLUENCY of a summary.

**Fluency Definition**: A fluent summary is grammatically correct, uses natural language, and is easy to read without awkward phrasing.

**Evaluation Criteria**:
- Score 1-2: Multiple grammar errors, unnatural phrasing
- Score 3-4: Several fluency issues
- Score 5-6: Generally fluent with minor issues
- Score 7-8: Highly fluent, natural language
- Score 9-10: Perfect fluency, publication-quality

**Summary to Evaluate**:
{summary}

**Instructions**:
1. Check for grammatical correctness
2. Assess naturalness of language
3. Rate fluency from 1-10
4. Provide a brief explanation (1-2 sentences)

**Response Format**:
Score: [1-10]
Explanation: [Your reasoning]"""

    def _parse_llm_response(self, response: str) -> tuple[Optional[float], Optional[str]]:
        """
        Parse the LLM response to extract score and explanation.

        Returns:
            (score, explanation) tuple
        """
        try:
            lines = response.strip().split('\n')
            score = None
            explanation = None

            for line in lines:
                line = line.strip()
                if line.startswith('Score:'):
                    # Extract number from "Score: 8" or "Score: [8]"
                    score_text = line.replace('Score:', '').strip()
                    score_text = score_text.replace('[', '').replace(']', '')
                    score = float(score_text)
                elif line.startswith('Explanation:'):
                    explanation = line.replace('Explanation:', '').strip()

            return score, explanation
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None, None

    def _query_llm(self, prompt: str, timeout: int = 60) -> str:
        """
        Query the LLM with a prompt.

        Args:
            prompt: The evaluation prompt
            timeout: Query timeout in seconds

        Returns:
            The LLM's response text
        """
        client = self._get_client()
        chat_session_id = client.create_chat_session()

        with client.connect(chat_session_id) as session:
            reply = session.query(
                prompt,
                llm=self.model_name,
                timeout=timeout,
            )
            return reply.content

    def evaluate_faithfulness(
        self,
        source: str,
        summary: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness of summary using LLM-as-a-Judge.

        Args:
            source: Source document text
            summary: Summary text to evaluate
            timeout: Query timeout in seconds

        Returns:
            Dictionary with 'score' (0-1) and 'explanation'
        """
        try:
            prompt = self._create_faithfulness_prompt(source, summary)
            response = self._query_llm(prompt, timeout)
            score, explanation = self._parse_llm_response(response)

            if score is not None:
                # Normalize to 0-1 scale
                normalized_score = score / 10.0
                return {
                    'score': normalized_score,
                    'raw_score': score,
                    'explanation': explanation or 'No explanation provided',
                    'full_response': response,
                }
            else:
                return {
                    'score': None,
                    'error': 'Failed to parse score from LLM response',
                    'full_response': response,
                }
        except Exception as e:
            return {
                'score': None,
                'error': str(e),
            }

    def evaluate_coherence(
        self,
        summary: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate coherence of summary using LLM-as-a-Judge.

        Args:
            summary: Summary text to evaluate
            timeout: Query timeout in seconds

        Returns:
            Dictionary with 'score' (0-1) and 'explanation'
        """
        try:
            prompt = self._create_coherence_prompt(summary)
            response = self._query_llm(prompt, timeout)
            score, explanation = self._parse_llm_response(response)

            if score is not None:
                normalized_score = score / 10.0
                return {
                    'score': normalized_score,
                    'raw_score': score,
                    'explanation': explanation or 'No explanation provided',
                    'full_response': response,
                }
            else:
                return {
                    'score': None,
                    'error': 'Failed to parse score from LLM response',
                    'full_response': response,
                }
        except Exception as e:
            return {
                'score': None,
                'error': str(e),
            }

    def evaluate_relevance(
        self,
        source: str,
        summary: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate relevance of summary using LLM-as-a-Judge.

        Args:
            source: Source document text
            summary: Summary text to evaluate
            timeout: Query timeout in seconds

        Returns:
            Dictionary with 'score' (0-1) and 'explanation'
        """
        try:
            prompt = self._create_relevance_prompt(source, summary)
            response = self._query_llm(prompt, timeout)
            score, explanation = self._parse_llm_response(response)

            if score is not None:
                normalized_score = score / 10.0
                return {
                    'score': normalized_score,
                    'raw_score': score,
                    'explanation': explanation or 'No explanation provided',
                    'full_response': response,
                }
            else:
                return {
                    'score': None,
                    'error': 'Failed to parse score from LLM response',
                    'full_response': response,
                }
        except Exception as e:
            return {
                'score': None,
                'error': str(e),
            }

    def evaluate_fluency(
        self,
        summary: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate fluency of summary using LLM-as-a-Judge.

        Args:
            summary: Summary text to evaluate
            timeout: Query timeout in seconds

        Returns:
            Dictionary with 'score' (0-1) and 'explanation'
        """
        try:
            prompt = self._create_fluency_prompt(summary)
            response = self._query_llm(prompt, timeout)
            score, explanation = self._parse_llm_response(response)

            if score is not None:
                normalized_score = score / 10.0
                return {
                    'score': normalized_score,
                    'raw_score': score,
                    'explanation': explanation or 'No explanation provided',
                    'full_response': response,
                }
            else:
                return {
                    'score': None,
                    'error': 'Failed to parse score from LLM response',
                    'full_response': response,
                }
        except Exception as e:
            return {
                'score': None,
                'error': str(e),
            }

    def evaluate_dag(
        self,
        source: str,
        summary: str,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate using DAG (Directed Acyclic Graph) approach.

        DAG breaks evaluation into a decision tree of smaller questions
        rather than one big score. This is from the DeepEval framework.

        Args:
            source: Source document text
            summary: Summary text to evaluate
            timeout: Query timeout in seconds

        Returns:
            Dictionary with decision tree results
        """
        try:
            prompt = f"""You are an expert evaluator using a structured decision tree approach (DAG - Directed Acyclic Graph).

**Evaluation Task**: Assess this summary using a step-by-step decision tree.

**Source Document**:
{source}

**Summary to Evaluate**:
{summary}

**Decision Tree Evaluation**:

Step 1: FACTUAL ACCURACY
Question: Are all facts in the summary correct?
- If YES → Score 2 → Go to Step 2
- If MOSTLY YES → Score 1 → Go to Step 2
- If NO → Score 0 → STOP (Final: 0-2)

Step 2: COMPLETENESS
Question: Does the summary cover the main points?
- If YES → Score 2 → Go to Step 3
- If MOSTLY YES → Score 1 → Go to Step 3
- If NO → Score 0 → Add to Step 1 score

Step 3: CLARITY
Question: Is the summary clear and well-written?
- If YES → Score 2 → Calculate final
- If MOSTLY YES → Score 1 → Calculate final
- If NO → Score 0 → Calculate final

**Response Format**:
Step 1 Score: [0/1/2]
Step 2 Score: [0/1/2]
Step 3 Score: [0/1/2]
Total Score: [0-6]
Explanation: [Brief explanation of the decision path taken]"""

            response = self._query_llm(prompt, timeout)

            # Parse response
            step1 = None
            step2 = None
            step3 = None
            total = None
            explanation = ""

            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Step 1 Score:'):
                    try:
                        step1 = int(line.split(':')[1].strip().split('/')[0])
                    except:
                        pass
                elif line.startswith('Step 2 Score:'):
                    try:
                        step2 = int(line.split(':')[1].strip().split('/')[0])
                    except:
                        pass
                elif line.startswith('Step 3 Score:'):
                    try:
                        step3 = int(line.split(':')[1].strip().split('/')[0])
                    except:
                        pass
                elif line.startswith('Total Score:'):
                    try:
                        score_text = line.split(':')[1].strip().split('-')[0].strip()
                        total = int(score_text)
                    except:
                        pass
                elif line.startswith('Explanation:'):
                    explanation = line.split(':', 1)[1].strip()

            if total is not None:
                normalized_score = total / 6.0  # Normalize to 0-1
            else:
                normalized_score = None

            return {
                'score': normalized_score,
                'raw_score': total,
                'step1_factual': step1,
                'step2_completeness': step2,
                'step3_clarity': step3,
                'explanation': explanation if explanation else 'No explanation provided',
                'full_response': response,
            }

        except Exception as e:
            return {
                'score': None,
                'error': str(e)
            }

    def evaluate_all(
        self,
        source: str,
        summary: str,
        timeout: int = 60,
        include_dag: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all LLM-as-a-Judge evaluations.

        Args:
            source: Source document text
            summary: Summary text to evaluate
            timeout: Query timeout in seconds per metric
            include_dag: Whether to include DAG evaluation

        Returns:
            Dictionary with results for each dimension
        """
        results = {
            'faithfulness': self.evaluate_faithfulness(source, summary, timeout),
            'coherence': self.evaluate_coherence(summary, timeout),
            'relevance': self.evaluate_relevance(source, summary, timeout),
            'fluency': self.evaluate_fluency(summary, timeout),
        }

        # Add DAG if requested
        if include_dag:
            results['dag'] = self.evaluate_dag(source, summary, timeout)

        return results
