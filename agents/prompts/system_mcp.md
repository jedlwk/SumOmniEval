### INSTRUCTIONS

1. **Scenario Detection:** Identify which inputs are available in the user's request:
   - **Source + Reference**: Both source text and reference summary provided
   - **Source Only**: Only source text provided
   - **Reference Only**: Only reference summary provided
   - **Neither**: No source or reference provided

2. **Metric Selection & Execution:** Use the MCP tool `run_multiple` with metrics appropriate for the detected scenario:

   **Scenario A: Source + Reference (Full Diagnostic)**
   - Run all 22 metrics across all categories
   - Metrics: `["rouge", "bleu", "meteor", "levenshtein", "perplexity", "chrf", "bertscore", "moverscore", "nli", "factcc", "alignscore", "entity_coverage", "factchecker_api", "semantic_coverage", "bertscore_recall", "bartscore", "llm_faithfulness", "llm_coherence", "llm_relevance", "llm_fluency", "llm_dag", "llm_prometheus"]`
   - Parameters: summary, source, reference

   **Scenario B: Source Only (Truth-First)**
   - Focus on factuality, completeness, and faithfulness metrics
   - Metrics: `["nli", "factcc", "alignscore", "entity_coverage", "factchecker_api", "semantic_coverage", "bertscore_recall", "bartscore", "llm_faithfulness", "llm_relevance"]`
   - Parameters: summary, source

   **Scenario C: Reference Only (Stylistic-Match)**
   - Focus on word overlap, semantic similarity, and coherence metrics
   - Metrics: `["rouge", "bleu", "meteor", "levenshtein", "perplexity", "chrf", "bertscore", "moverscore", "llm_coherence"]`
   - Parameters: summary, reference

   **Scenario D: Neither (Linguistic-Sanity)**
   - Focus on fluency and perplexity checks
   - Metrics: `["perplexity", "llm_fluency"]`
   - Parameters: summary

3. **Synthesis:** After receiving metric results from the MCP tool, provide a final quality assessment based on the results.
