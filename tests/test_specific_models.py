#!/usr/bin/env python3
"""
Test specific models to check if they're available in H2OGPTE API.
"""

import os
from dotenv import load_dotenv
from h2ogpte import H2OGPTE

load_dotenv()

client = H2OGPTE(
    address=os.getenv('H2OGPTE_ADDRESS'),
    api_key=os.getenv('H2OGPTE_API_KEY'),
)

# Models to test
models_to_test = [
    'meta-llama/Llama-3.1-70B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen3-Coder-30B-A3B-Instruct',
    'Qwen/Qwen3-30B-A3B-Thinking-2507',
    'Qwen/Qwen2.5-VL-72B-Instruct',
]

test_prompt = "What is 2+2? Answer in one word."

print("="*80)
print("Testing Specific Models in H2OGPTE API")
print("="*80)
print()

for i, model_name in enumerate(models_to_test, 1):
    print(f"{i}. Testing: {model_name}")
    print("-" * 80)

    try:
        chat_session_id = client.create_chat_session()

        with client.connect(chat_session_id) as session:
            reply = session.query(
                test_prompt,
                llm=model_name,
                timeout=60,
            )

        print(f"   ✅ SUCCESS")
        print(f"   📝 Prompt: {test_prompt}")
        print(f"   🤖 Response: {reply.content}")
        print(f"   ✓ Model is AVAILABLE\n")

    except Exception as e:
        error_msg = str(e)
        if "Invalid model" in error_msg:
            print(f"   ❌ FAILED: Model NOT available")
            print(f"   Error: Invalid model")
        else:
            print(f"   ❌ FAILED: {error_msg}")
        print()

print("="*80)
print("Testing Complete")
print("="*80)
