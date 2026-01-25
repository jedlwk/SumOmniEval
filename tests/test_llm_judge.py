#!/usr/bin/env python3
"""
Test script for LLM-as-a-Judge evaluator.
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from src.evaluators.era3_llm_judge import LLMJudgeEvaluator

print("="*80)
print("Testing LLM-as-a-Judge Evaluator")
print("="*80)

# Test data
source = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889, it was initially criticized by some of France's leading
artists and intellectuals for its design, but has become a global cultural icon of France
and one of the most recognizable structures in the world. The tower is 330 metres tall,
about the same height as an 81-storey building.
"""

summary = """
The Eiffel Tower in Paris was built between 1887 and 1889 by Gustave Eiffel's company.
Standing at 330 meters tall, it faced initial criticism but is now a famous French landmark.
"""

print("\n📄 Source Text:")
print(source.strip())
print("\n📝 Summary:")
print(summary.strip())
print()

# Test with Llama-4-Maverick (default)
model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
print(f"\n🤖 Testing with model: {model}")
print("="*80)

try:
    evaluator = LLMJudgeEvaluator(model_name=model)
    print("✅ Evaluator initialized successfully")

    # Test faithfulness
    print("\n🔍 Testing Faithfulness...")
    faith_result = evaluator.evaluate_faithfulness(source, summary, timeout=90)
    if 'error' in faith_result:
        print(f"❌ Error: {faith_result['error']}")
    else:
        print(f"✅ Faithfulness Score: {faith_result['score']:.3f} (Raw: {faith_result.get('raw_score', 'N/A')}/10)")
        print(f"💬 Explanation: {faith_result.get('explanation', 'N/A')}")

    # Test coherence
    print("\n🔍 Testing Coherence...")
    coh_result = evaluator.evaluate_coherence(summary, timeout=90)
    if 'error' in coh_result:
        print(f"❌ Error: {coh_result['error']}")
    else:
        print(f"✅ Coherence Score: {coh_result['score']:.3f} (Raw: {coh_result.get('raw_score', 'N/A')}/10)")
        print(f"💬 Explanation: {coh_result.get('explanation', 'N/A')}")

    # Test relevance
    print("\n🔍 Testing Relevance...")
    rel_result = evaluator.evaluate_relevance(source, summary, timeout=90)
    if 'error' in rel_result:
        print(f"❌ Error: {rel_result['error']}")
    else:
        print(f"✅ Relevance Score: {rel_result['score']:.3f} (Raw: {rel_result.get('raw_score', 'N/A')}/10)")
        print(f"💬 Explanation: {rel_result.get('explanation', 'N/A')}")

    # Test fluency
    print("\n🔍 Testing Fluency...")
    flu_result = evaluator.evaluate_fluency(summary, timeout=90)
    if 'error' in flu_result:
        print(f"❌ Error: {flu_result['error']}")
    else:
        print(f"✅ Fluency Score: {flu_result['score']:.3f} (Raw: {flu_result.get('raw_score', 'N/A')}/10)")
        print(f"💬 Explanation: {flu_result.get('explanation', 'N/A')}")

    print("\n" + "="*80)
    print("✅ All tests completed successfully!")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
