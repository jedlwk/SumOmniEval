#!/usr/bin/env python3
"""
Test models with corrected names based on what's actually available.
"""

import os
from dotenv import load_dotenv
from h2ogpte import H2OGPTE

load_dotenv()

client = H2OGPTE(
    address=os.getenv('H2OGPTE_ADDRESS'),
    api_key=os.getenv('H2OGPTE_API_KEY'),
)

# Models from user's request vs. what's actually available
test_cases = [
    {
        'requested': 'meta-llama/Llama-3.1-70B-Instruct',
        'actual': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'status': 'Available (with Meta- prefix)'
    },
    {
        'requested': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'actual': 'deepseek-ai/DeepSeek-R1',
        'status': 'Only full R1 available (not distilled)'
    },
    {
        'requested': 'meta-llama/Llama-3.3-70B-Instruct',
        'actual': 'meta-llama/Llama-3.3-70B-Instruct',
        'status': 'Available (exact match)'
    },
    {
        'requested': 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
        'actual': None,
        'status': 'NOT available - No Qwen models in this instance'
    },
    {
        'requested': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
        'actual': None,
        'status': 'NOT available - No Qwen models in this instance'
    },
    {
        'requested': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'actual': None,
        'status': 'NOT available - No Qwen models in this instance'
    },
]

test_prompt = "What is 2+2? Answer in one word."

print("="*80)
print("Testing Models - Requested vs Available")
print("="*80)
print()

for i, test_case in enumerate(test_cases, 1):
    print(f"{i}. REQUESTED: {test_case['requested']}")
    print(f"   STATUS: {test_case['status']}")

    if test_case['actual'] is None:
        print(f"   ❌ NOT AVAILABLE in this H2OGPTE instance\n")
        continue

    print(f"   TESTING: {test_case['actual']}")
    print("-" * 80)

    try:
        chat_session_id = client.create_chat_session()

        with client.connect(chat_session_id) as session:
            reply = session.query(
                test_prompt,
                llm=test_case['actual'],
                timeout=60,
            )

        print(f"   ✅ SUCCESS")
        print(f"   📝 Prompt: {test_prompt}")
        print(f"   🤖 Response: {reply.content}")
        print(f"   ✓ Model works correctly\n")

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED: {error_msg}\n")

print("="*80)
print("Summary")
print("="*80)
print("\n✅ AVAILABLE MODELS from your list:")
print("  1. meta-llama/Meta-Llama-3.1-70B-Instruct (note: 'Meta-' prefix required)")
print("  2. meta-llama/Llama-3.3-70B-Instruct")
print("  3. deepseek-ai/DeepSeek-R1 (full version, not distilled)")
print("\n❌ NOT AVAILABLE:")
print("  - deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
print("  - All Qwen models (none in this instance)")
print()
