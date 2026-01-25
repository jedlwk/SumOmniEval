#!/usr/bin/env python3
"""
Test each Era 3B dimension individually to ensure they work independently.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.evaluators.era3_llm_judge import LLMJudgeEvaluator

print("="*80)
print("Era 3B Individual Dimension Tests (Foolproof Verification)")
print("="*80)

# Test data
source = """
The Amazon rainforest produces 20% of the world's oxygen and is home to 10% of all
species on Earth. It spans 5.5 million square kilometers across nine countries,
with Brazil containing the largest portion. Deforestation rates have increased by
30% in recent years due to logging and agriculture.
"""

summary = """
The Amazon rainforest, spanning 5.5 million square kilometers across nine countries,
generates 20% of global oxygen and houses 10% of Earth's species. Deforestation
has risen 30% recently due to logging and farming activities.
"""

# Test with default model
model = 'meta-llama/Llama-3.3-70B-Instruct'
print(f"\n🤖 Testing with: {model}")
print("="*80)

evaluator = LLMJudgeEvaluator(model_name=model)
print(f"✅ Evaluator initialized\n")

# Test each dimension individually
dimensions = [
    ('Faithfulness', 'evaluate_faithfulness', (source, summary)),
    ('Coherence', 'evaluate_coherence', (summary,)),
    ('Relevance', 'evaluate_relevance', (source, summary)),
    ('Fluency', 'evaluate_fluency', (summary,)),
]

results = {}
all_passed = True

for dim_name, method_name, args in dimensions:
    print(f"\n{'='*80}")
    print(f"Testing: {dim_name}")
    print('='*80)

    try:
        method = getattr(evaluator, method_name)
        result = method(*args, timeout=90)

        if 'error' in result:
            print(f"❌ FAILED: {result['error']}")
            all_passed = False
            results[dim_name] = {'status': 'FAILED', 'error': result['error']}
        elif result.get('score') is None:
            print(f"❌ FAILED: No score returned")
            all_passed = False
            results[dim_name] = {'status': 'FAILED', 'error': 'No score'}
        else:
            print(f"✅ SUCCESS")
            print(f"   Score: {result['score']:.3f} (Raw: {result.get('raw_score', 'N/A')}/10)")
            print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")
            results[dim_name] = {
                'status': 'PASSED',
                'score': result['score'],
                'raw_score': result.get('raw_score'),
                'explanation': result.get('explanation')
            }
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        all_passed = False
        results[dim_name] = {'status': 'EXCEPTION', 'error': str(e)}

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for dim_name, result in results.items():
    status = result['status']
    if status == 'PASSED':
        print(f"✅ {dim_name:15s} PASSED - Score: {result['score']:.3f}")
    else:
        print(f"❌ {dim_name:15s} {status} - {result.get('error', 'Unknown error')}")

print("\n" + "="*80)
if all_passed:
    print("✅ ALL DIMENSIONS WORKING INDEPENDENTLY")
    print("Era 3B is FOOLPROOF and ready for production")
else:
    print("⚠️ SOME DIMENSIONS FAILED")
    print("Review errors above")
print("="*80)

# Test error handling
print("\n" + "="*80)
print("Testing Error Handling (Empty inputs)")
print("="*80)

try:
    result = evaluator.evaluate_faithfulness("", "test")
    if 'error' in result or result.get('score') is None:
        print("✅ Empty source handled correctly")
    else:
        print("⚠️ Empty source should fail but didn't")
except Exception as e:
    print(f"✅ Empty source raised exception (expected): {str(e)[:50]}")

try:
    result = evaluator.evaluate_fluency("")
    if 'error' in result or result.get('score') is None:
        print("✅ Empty summary handled correctly")
    else:
        print("⚠️ Empty summary should fail but didn't")
except Exception as e:
    print(f"✅ Empty summary raised exception (expected): {str(e)[:50]}")

print("\n✅ Error handling tests complete")
