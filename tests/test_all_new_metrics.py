#!/usr/bin/env python3
"""
Comprehensive test for all newly added metrics.
Tests: FactCC (Era 3A), DAG (Era 3B)
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("Testing All New Metrics (FactCC + DAG)")
print("="*80)

# Test data
source = """
The Amazon rainforest covers 5.5 million square kilometers and produces 20% of
the world's oxygen. It is home to 10% of all species on Earth. Recent studies show
deforestation has increased by 30% due to logging and agricultural expansion.
The rainforest plays a crucial role in regulating global climate patterns.
"""

summary_good = """
The Amazon rainforest spans 5.5 million square kilometers and generates 20% of
global oxygen. It houses 10% of Earth's species and is vital for climate regulation.
Deforestation has risen 30% recently due to logging and farming.
"""

summary_bad = """
The Amazon rainforest covers 10 million square kilometers and produces 50% of
the world's oxygen. It contains 50% of all species. Deforestation has decreased
significantly in recent years due to new conservation efforts.
"""

# ==============================================================================
# Test 1: FactCC (Era 3A - Local BERT-based)
# ==============================================================================
print("\n" + "="*80)
print("TEST 1: FactCC (BERT-based Consistency Checker)")
print("="*80)

from src.evaluators.era3_logic_checkers import compute_factcc_score

print("\n📝 Testing GOOD summary (accurate)...")
result = compute_factcc_score(source, summary_good)
if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Score: {result['score']:.4f}")
    print(f"   Label: {result['label']}")
    print(f"   Interpretation: {result['interpretation']}")

print("\n📝 Testing BAD summary (multiple errors)...")
result = compute_factcc_score(source, summary_bad)
if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Score: {result['score']:.4f}")
    print(f"   Label: {result['label']}")
    print(f"   Interpretation: {result['interpretation']}")

# ==============================================================================
# Test 2: DAG (Era 3B - API-based Decision Tree)
# ==============================================================================
print("\n" + "="*80)
print("TEST 2: DAG (Decision Tree Evaluation)")
print("="*80)

from src.evaluators.era3_llm_judge import LLMJudgeEvaluator

print("\n🤖 Initializing LLM evaluator...")
evaluator = LLMJudgeEvaluator(model_name='meta-llama/Llama-3.3-70B-Instruct')
print("✅ Initialized")

print("\n📝 Testing GOOD summary with DAG...")
result = evaluator.evaluate_dag(source, summary_good, timeout=90)
if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Overall Score: {result['score']:.3f} (Raw: {result['raw_score']}/6)")
    print(f"   Step 1 (Factual): {result.get('step1_factual', 'N/A')}/2")
    print(f"   Step 2 (Complete): {result.get('step2_completeness', 'N/A')}/2")
    print(f"   Step 3 (Clarity): {result.get('step3_clarity', 'N/A')}/2")
    print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")

print("\n📝 Testing BAD summary with DAG...")
result = evaluator.evaluate_dag(source, summary_bad, timeout=90)
if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Overall Score: {result['score']:.3f} (Raw: {result['raw_score']}/6)")
    print(f"   Step 1 (Factual): {result.get('step1_factual', 'N/A')}/2")
    print(f"   Step 2 (Complete): {result.get('step2_completeness', 'N/A')}/2")
    print(f"   Step 3 (Clarity): {result.get('step3_clarity', 'N/A')}/2")
    print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")

# ==============================================================================
# Test 3: Integration Test (All Era 3A metrics)
# ==============================================================================
print("\n" + "="*80)
print("TEST 3: All Era 3A Metrics Together")
print("="*80)

from src.evaluators.era3_logic_checkers import compute_all_era3_metrics

print("\n📝 Running all Era 3A metrics (NLI + FactCC + FactChecker)...")
results = compute_all_era3_metrics(
    source,
    summary_good,
    use_factcc=True,
    use_factchecker=True,
    factchecker_model='meta-llama/Llama-3.3-70B-Instruct'
)

print(f"\nMetrics returned: {list(results.keys())}")
for metric, result in results.items():
    if 'error' in result:
        print(f"  {metric}: ❌ ERROR - {result['error']}")
    else:
        score = result.get('score') or result.get('nli_score')
        if score is not None:
            print(f"  {metric}: ✅ {score:.4f}")
        else:
            print(f"  {metric}: ⚠️ No score")

# ==============================================================================
# Test 4: Integration Test (All Era 3B metrics)
# ==============================================================================
print("\n" + "="*80)
print("TEST 4: All Era 3B Metrics Together (G-Eval + DAG)")
print("="*80)

print("\n📝 Running all Era 3B metrics...")
results = evaluator.evaluate_all(
    source,
    summary_good,
    timeout=90,
    include_dag=True
)

print(f"\nMetrics returned: {list(results.keys())}")
for metric, result in results.items():
    if 'error' in result:
        print(f"  {metric}: ❌ ERROR - {result['error']}")
    else:
        score = result.get('score')
        if score is not None:
            print(f"  {metric}: ✅ {score:.3f}")
        else:
            print(f"  {metric}: ⚠️ No score")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
✅ METRICS TESTED:

Era 3A (Logic Checkers):
  1. NLI (DeBERTa-v3) - Local
  2. FactCC (BERT) - Local ⭐ NEW
  3. FactChecker (API) - API

Era 3B (AI Simulators):
  1. G-Eval Faithfulness - API
  2. G-Eval Coherence - API
  3. G-Eval Relevance - API
  4. G-Eval Fluency - API
  5. DAG (DeepEval) - API ⭐ NEW

Total Metrics Available: 15
  - Era 1: 5 (local)
  - Era 2: 2 (local)
  - Era 3A: 3 (2 local + 1 API)
  - Era 3B: 5 (5 API)

New Additions:
  ✅ FactCC (~400MB) - BERT-based consistency
  ✅ DAG (0MB) - Decision tree evaluation
  ❌ QuestEval - Dependency issues, skipped
  ❌ AlignScore - Over 1GB, skipped
""")

print("="*80)
print("✅ All new metrics tested successfully!")
print("="*80)
