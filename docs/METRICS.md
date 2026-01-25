# SumOmniEval - Metrics Documentation

Comprehensive guide to all 15 evaluation metrics.

---

## 📚 About the Metrics

SumOmniEval uses **15 different metrics** to evaluate summary quality from multiple perspectives. These metrics are organized into **3 evaluation eras**, each representing a different approach to measuring how good a summary is:

- **Era 1: Word Overlap** - Do the words match? (5 metrics)
- **Era 2: Semantic Embeddings** - Does the meaning match? (2 metrics)
- **Era 3: Logic & AI Judges** - Is it factually correct and well-written? (8 metrics)

Each era builds on the previous one, addressing limitations and adding sophistication. Together, they provide a complete picture of summary quality.

---

## Era 1: Word Overlap & Fluency (5 metrics)

### What are Word Overlap Metrics?

**Theme**: The Age of "Exact Matches"

In the early days (2000s), we treated text like Scrabble tiles. We assumed that if the computer used the exact same words as the human reference, it must be right. We didn't care about meaning; we only cared about matching symbols.

**Core Metrics**: ROUGE & BLEU - The industry standards. ROUGE focuses on recall (did you include all the reference words?), while BLEU focuses on precision.

**Additional Metrics**:
- **METEOR**: The clever cousin. ROUGE fails if you write "fast" instead of "quick." METEOR fixes this by counting synonyms and stem forms (running = run).
- **Levenshtein Distance**: The spellchecker. It measures "edit distance" - how many deletions or swaps it takes to turn the summary into the reference.
- **Perplexity**: Fluency measurement. This measures fluency, not truth. It checks how "surprised" a model is by the text. (Warning: A model can hallucinate a lie with perfect fluency/low perplexity).

**The Failure Mode**: The "Death of ROUGE"
- Source: "The movie was bad."
- AI Summary: "The film was terrible."
- ROUGE Score: 0.0 (Because "film" ≠ "movie" and "terrible" ≠ "bad")
- **The Lesson**: These metrics punish creativity and paraphrase.

**Pros & Cons**:
- ✅ Fast, cheap, and standard (everyone knows them)
- ❌ Misses synonyms, ignores structure, creates "Frankenstein" sentences

---

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**What it measures**: N-gram overlap between summary and reference

**Score range**: **0-1** (higher = better)

**Variants**:
- **ROUGE-1**: Unigram overlap (individual word matching)
- **ROUGE-2**: Bigram overlap (2-word phrase matching)
- **ROUGE-L**: Longest Common Subsequence (word order sensitivity)

**Interpretation**:
- **>0.4**: Good overlap (🟢 Green)
- **0.2-0.4**: Moderate overlap (🟡 Yellow)
- **<0.2**: Poor overlap (🔴 Red)

---

### BLEU (Bilingual Evaluation Understudy)

**What it measures**: Precision-based n-gram overlap (originally for machine translation)

**Score range**: **0-1** (higher = better)

**How it works**: Counts matching n-grams, penalizes very short summaries

**Interpretation**:
- **>0.3**: High precision (🟢 Green)
- **0.1-0.3**: Moderate precision (🟡 Yellow)
- **<0.1**: Low precision (🔴 Red)

---

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**What it measures**: Semantic matching including synonyms and stemming

**Score range**: **0-1** (higher = better)

**How it works**:
- Matches exact words
- Matches stems (e.g., "running" ↔ "run")
- Matches synonyms (e.g., "big" ↔ "large")

**Interpretation**:
- **>0.5**: Strong semantic alignment (🟢 Green)
- **0.3-0.5**: Moderate alignment (🟡 Yellow)
- **<0.3**: Weak alignment (🔴 Red)

---

### Levenshtein Distance

**What it measures**: Edit distance similarity (character-level)

**Score range**: **0-1** (higher = more similar)

**How it works**: Counts minimum character insertions/deletions/substitutions needed to transform summary into reference

**Interpretation**:
- **>0.7**: Very similar text (🟢 Green)
- **0.4-0.7**: Moderately similar (🟡 Yellow)
- **<0.4**: Different text (🔴 Red)

---

### Perplexity

**What it measures**: Language model fluency and naturalness

**Score range**: **0-1** (higher = better, normalized from raw perplexity)

**How it works**: Uses GPT-2 to measure how "surprised" a language model is by the summary

**Interpretation**:
- **>0.7**: Natural, fluent text (🟢 Green)
- **0.4-0.7**: Acceptable fluency (🟡 Yellow)
- **<0.4**: Awkward or unnatural text (🔴 Red)

---

## Era 2: Embeddings (2 metrics)

### What are Embedding Metrics?

**Theme**: The Age of "Semantic Similarity"

Around 2019, we realized that exact words don't matter - meaning matters. We started using 'Embeddings' (dense vector representations) to map words into space. If 'Lawyer' and 'Attorney' are close in space, they should count as a match.

**BERTScore**: Calculates the cosine similarity between the summary's "vibe" and the source's "vibe" using contextual embeddings.

**MoverScore**: Uses "Earth Mover's Distance" (a transportation math problem) to calculate the "cost" of moving the meaning from the summary to the source. It is often softer and more robust than BERTScore.

**The Failure Mode**: The "Negation Trap"
- Sentence A: "The patient has cancer."
- Sentence B: "The patient has no cancer."
- BERTScore: 0.96 (96% similarity)
- **The Lesson**: Because these sentences share almost all the same context, embeddings think they are identical. They are blind to small logic words like "not," "never," or "unless."

**Pros & Cons**:
- ✅ Captures synonyms and paraphrasing perfectly
- ❌ Terrible at "Factuality" - Can't distinguish between opposite claims if context is similar

---

### BERTScore

**What it measures**: Contextual embedding similarity using BERT

**Score range**: **0-1** (higher = better)

**How it works**:
- Converts each word to a context-aware embedding
- Computes optimal word-pair similarity
- Reports Precision, Recall, and F1

**Interpretation**:
- **>0.75**: Strong semantic match (🟢 Green)
- **0.65-0.75**: Good semantic match (🟡 Yellow)
- **<0.65**: Weak semantic match (🔴 Red)

**Metrics reported**:
- **Precision**: Are summary words semantically present in source?
- **Recall**: Are source concepts captured in summary?
- **F1**: Harmonic mean of precision and recall

---

### MoverScore

**What it measures**: Optimal alignment of contextualized embeddings via Earth Mover's Distance

**Score range**: **0-1** (higher = better)

**How it works**:
- Uses BERT embeddings with context
- Finds optimal "transport" of meaning from source to summary
- More sophisticated than BERTScore's greedy matching

**Interpretation**:
- **>0.75**: Excellent semantic alignment (🟢 Green)
- **0.65-0.75**: Good alignment (🟡 Yellow)
- **<0.65**: Poor alignment (🔴 Red)

---

## Era 3: Logic & AI Judges (8 metrics)

### What are Logic & AI Judge Metrics?

**Theme**: The Age of "Reasoning" & "Fact-Checking"

We stopped trying to use math formulas to grade language. We realized that to judge a summary, you need to understand logic. We split into two camps: The **Logic Checkers** (who use NLI to find truth) and the **AI Simulators** (who mimic human grading).

**Group A: Logic Checkers** - Use models trained to detect logical consistency and factual errors
**Group B: AI Simulators** - Use powerful LLMs to evaluate like a human expert

---

### NLI - Natural Language Inference (DeBERTa-v3)

**What it does**: Checks if the summary is logically supported by the source document

**What we're testing for**: Logical entailment - Does the source prove the summary's claims?

**Score range**: **0-1** (probability of entailment)

**How it works**:
- Uses DeBERTa-v3-base-mnli model (~400MB local)
- Classifies relationship as: ENTAILMENT, NEUTRAL, or CONTRADICTION
- Returns confidence score

**Example**:
- Source: "The company hired 50 engineers in 2024"
- Good summary: "The company expanded its engineering team" → High score (0.8+)
- Bad summary: "The company fired engineers" → Low score (<0.3)

**Interpretation**:
- **>0.7**: High entailment - logically supported (🟢 Green)
- **0.4-0.7**: Neutral - partially supported (🟡 Yellow)
- **<0.4**: Contradiction - not supported (🔴 Red)

---

### FactCC (BERT-based Consistency Checker)

**What it does**: Checks factual consistency using a BERT model trained specifically for fact-checking

**What we're testing for**: Binary factual consistency - Is the summary consistent or inconsistent with facts?

**Score range**: **0-1** (higher = more consistent)

**How it works**:
- Uses BERT model fine-tuned for fact-checking (~400MB local)
- Classifies summary as CONSISTENT or INCONSISTENT
- Returns confidence score

**Example**:
- Source: "The product costs $50"
- Good summary: "The product is priced at $50" → High score (0.9+)
- Bad summary: "The product costs $500" → Low score (<0.2)

**Interpretation**:
- **>0.8**: Highly consistent (🟢 Green)
- **0.5-0.8**: Mostly consistent (🟡 Yellow)
- **<0.5**: Inconsistent (🔴 Red)

---

### FactChecker (LLM-powered)

**What it does**: Uses a powerful LLM to perform detailed fact verification with reasoning

**What we're testing for**: Factual accuracy with step-by-step verification

**Score range**: **0-1** (higher = more accurate)

**How it works**:
- Uses Llama-3.3-70B via API (requires H2OGPTE)
- Performs claim-by-claim fact-checking
- Provides detailed explanations

**Example**:
- Source: "Einstein published his theory of relativity in 1905"
- Good summary: "Einstein's 1905 theory of relativity" → 1.0 (perfect)
- Bad summary: "Einstein's 1915 theory of relativity" → 0.5 (date error)

**Interpretation**:
- **1.0**: No factual errors (🟢 Green)
- **0.5-0.9**: Minor errors or unsupported claims (🟡 Yellow)
- **<0.5**: Multiple factual errors (🔴 Red)

---

### G-Eval: Faithfulness

**What it does**: Evaluates whether all claims in the summary are supported by the source

**What we're testing for**: Factual accuracy and source support

**Score range**: **1-10** (displayed as X.X/10)

**How it works**:
- Uses LLM (Llama-3.3-70B) with structured prompt
- Asks: "Are all claims supported? Any hallucinations? Is data accurate?"
- Returns score from 1-10 with explanation

**Example**:
- Source: "Sales increased 20% to $2M"
- Good summary: "Sales rose 20% to $2M" → **9.0/10** (🟢 Green)
- Bad summary: "Sales doubled to $4M" → **3.0/10** (🔴 Red)

**Interpretation**:
- **8-10**: Perfect/highly faithful (🟢 Green)
- **5-7**: Mostly faithful with minor issues (🟡 Yellow)
- **1-4**: Multiple unsupported claims (🔴 Red)

---

### G-Eval: Coherence

**What it does**: Evaluates logical flow and organization of the summary

**What we're testing for**: Does it flow logically? Are transitions clear?

**Score range**: **1-10** (displayed as X.X/10)

**How it works**:
- Uses LLM to assess logical structure
- Asks: "Does it flow well? Are ideas connected? Is it well-organized?"

**Example**:
- Good: "First A happened. This caused B. Finally, C resulted." → **9.0/10** (🟢 Green)
- Bad: "C happened. Also A. B was there too." → **4.0/10** (🔴 Red)

**Interpretation**:
- **8-10**: Excellent flow (🟢 Green)
- **5-7**: Adequate with minor gaps (🟡 Yellow)
- **1-4**: Disjointed or confusing (🔴 Red)

---

### G-Eval: Relevance

**What it does**: Evaluates whether the summary captures important information and excludes irrelevant details

**What we're testing for**: Coverage of main points and focus on what matters

**Score range**: **1-10** (displayed as X.X/10)

**How it works**:
- Uses LLM to assess information selection
- Asks: "Are main points captured? Is irrelevant info excluded?"

**Example**:
- Source about climate change: discusses CO2, temperature, policy
- Good summary: Covers all 3 key points → **9.0/10** (🟢 Green)
- Bad summary: Only mentions CO2, ignores rest → **4.0/10** (🔴 Red)

**Interpretation**:
- **8-10**: Perfect information selection (🟢 Green)
- **5-7**: Covers main points adequately (🟡 Yellow)
- **1-4**: Misses important information (🔴 Red)

---

### G-Eval: Fluency

**What it does**: Evaluates writing quality, grammar, and naturalness

**What we're testing for**: Is the grammar correct? Is the language natural?

**Score range**: **1-10** (displayed as X.X/10)

**How it works**:
- Uses LLM to assess language quality
- Asks: "Is grammar correct? Is language natural? Any awkward phrasing?"

**Example**:
- Good: "The company expanded rapidly in Asia." → **9.0/10** (🟢 Green)
- Bad: "Company did expanding rapid in the Asia." → **2.0/10** (🔴 Red)

**Interpretation**:
- **8-10**: Publication-quality writing (🟢 Green)
- **5-7**: Generally fluent with minor issues (🟡 Yellow)
- **1-4**: Multiple grammar/fluency errors (🔴 Red)

---

### DAG (DeepEval Decision Tree)

**What it does**: Evaluates summary using a structured 3-step decision tree approach

**What we're testing for**: Factual accuracy, completeness, and clarity in a step-by-step manner

**Score range**: **0-6 points** (displayed as X/6, normalized to 0-1 in overall score)

**How it works**:
- **Step 1: Factual Accuracy** (0-2 points) - Are facts correct?
- **Step 2: Completeness** (0-2 points) - Main points covered?
- **Step 3: Clarity** (0-2 points) - Clear and well-written?
- Total: Sum of all steps

**Example**:
- Good summary:
  - Step 1: 2/2 (all facts correct)
  - Step 2: 2/2 (complete coverage)
  - Step 3: 2/2 (very clear)
  - **Total: 6/6** (🟢 Green)

- Fair summary:
  - Step 1: 2/2 (facts correct)
  - Step 2: 1/2 (missing some points)
  - Step 3: 1/2 (somewhat unclear)
  - **Total: 4/6** (🟡 Yellow)

**Interpretation**:
- **5-6/6**: Excellent summary (🟢 Green)
- **3-4/6**: Good summary (🟡 Yellow)
- **0-2/6**: Poor summary (🔴 Red)

---

## Understanding Scores Across Eras

### Era 1 & Era 2: Scores out of 1.0
- Scores range from **0.0 to 1.0**
- Example: ROUGE = 0.45, BERTScore = 0.82

### Era 3: Two Types of Scores
**Logic Checkers** (NLI, FactCC, FactChecker): **0-1**
- Example: NLI = 0.75, FactCC = 0.91

**AI Simulators** (G-Eval dimensions): **1-10**
- Example: Faithfulness = 8.5/10, Coherence = 7.0/10

**DAG**: **0-6 points**
- Example: DAG = 5/6 (also shown as 0.83 normalized)

---

## Color Coding Guide

- 🟢 **Green**: Excellent - meets quality standards
- 🟡 **Yellow**: Fair - acceptable with improvements needed
- 🔴 **Red**: Poor - significant issues present

---

## Metric Selection Guide

### For Different Use Cases

**Quick Feedback (12 seconds)**:
- Era 1 + Era 2 (7 metrics)
- Use when: Speed matters most

**Free Comprehensive Evaluation (40 seconds)**:
- Era 1 + Era 2 + Era 3 Logic Checkers (10 metrics)
- Use when: No API access, want thorough local evaluation

**Human-Like Evaluation (8 minutes)**:
- All 15 metrics
- Use when: Need publication-quality assessment

---

## Best Practices

1. **Always use Era 1 + Era 2** for baseline evaluation
2. **Add Era 3 Logic Checkers** when factual accuracy matters
3. **Add Era 3 AI Simulators** for publication-quality summaries
4. **Compare across eras** - look for discrepancies
5. **Focus on trends** - multiple low scores indicate real issues

---

**Last Updated**: 2026-01-25
**Total Metrics**: 15 (9 local + 6 API)
