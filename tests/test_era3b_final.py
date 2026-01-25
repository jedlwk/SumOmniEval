#!/usr/bin/env python3
"""
Final test of Era 3 Group B with corrected setup.
Tests default Llama-3.3-70B model with G-Eval.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.evaluators.era3_llm_judge import LLMJudgeEvaluator

print("="*80)
print("Era 3 Group B: AI Simulators (LLM-as-a-Judge) - Final Test")
print("="*80)

# Test data
source = """
The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait,
the one-mile-wide channel between San Francisco Bay and the Pacific Ocean.
The structure links the city of San Francisco to Marin County. It took seven years
to build and was completed in 1937. The bridge is 1.7 miles long and was once the
longest suspension bridge in the world.
"""

summary = """
The Golden Gate Bridge, completed in 1937 after seven years of construction,
connects San Francisco to Marin County across the Golden Gate strait. The 1.7-mile
bridge was formerly the world's longest suspension bridge.
"""

print("\n📄 Source Text:")
print(source.strip())
print("\n📝 Summary:")
print(summary.strip())
print()

# Test with default model (Llama-3.3-70B)
print("\n🤖 Testing with DEFAULT model (Llama-3.3-70B-Instruct)")
print("="*80)

try:
    # Initialize with default (should be Llama-3.3-70B now)
    evaluator = LLMJudgeEvaluator()
    print(f"✅ Evaluator initialized")
    print(f"   Model: {evaluator.model_name}")

    # Test all dimensions
    print("\n🔍 Running G-Eval (all 4 dimensions)...")
    results = evaluator.evaluate_all(source, summary, timeout=90)

    print("\n📊 Results:")
    print("-" * 80)

    for dimension, result in results.items():
        print(f"\n**{dimension.upper()}**")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   Score: {result['score']:.3f} (Raw: {result.get('raw_score', 'N/A')}/10)")
            print(f"   💬 {result.get('explanation', 'N/A')}")

    print("\n" + "="*80)
    print("✅ Era 3 Group B test completed successfully!")
    print("="*80)

    # Verify all scores are present
    all_ok = all('score' in results[d] and results[d]['score'] is not None
                 for d in ['faithfulness', 'coherence', 'relevance', 'fluency'])

    if all_ok:
        print("\n✅ All 4 G-Eval metrics working correctly")
        print(f"   - Faithfulness: {results['faithfulness']['score']:.3f}")
        print(f"   - Coherence: {results['coherence']['score']:.3f}")
        print(f"   - Relevance: {results['relevance']['score']:.3f}")
        print(f"   - Fluency: {results['fluency']['score']:.3f}")
    else:
        print("\n⚠️ Some metrics failed")

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
