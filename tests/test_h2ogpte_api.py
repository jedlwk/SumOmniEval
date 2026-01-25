#!/usr/bin/env python3
"""
Test H2OGPTE API connection and capabilities.
Tests both native h2ogpte client and OpenAI-compatible API.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv('H2OGPTE_API_KEY')
ADDRESS = os.getenv('H2OGPTE_ADDRESS')

if not API_KEY or not ADDRESS:
    print("❌ Error: H2OGPTE_API_KEY and H2OGPTE_ADDRESS must be set in .env file")
    exit(1)

print("="*80)
print("H2OGPTE API Testing")
print("="*80)
print(f"\n📍 Address: {ADDRESS}")
print(f"🔑 API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
print()


# ============================================================================
# Test 1: Native h2ogpte Client
# ============================================================================
print("="*80)
print("TEST 1: Native h2ogpte Client")
print("="*80)

try:
    from h2ogpte import H2OGPTE

    print("\n✅ h2ogpte package installed")

    # Create client
    print(f"\n🔌 Connecting to {ADDRESS}...")
    client = H2OGPTE(
        address=ADDRESS,
        api_key=API_KEY,
    )
    print("✅ Client created successfully")

    # Test simple query without collection
    print("\n📝 Testing simple query (no collection)...")
    chat_session_id = client.create_chat_session()
    print(f"✅ Chat session created: {chat_session_id}")

    with client.connect(chat_session_id) as session:
        reply = session.query(
            'What is 2+2? Answer in one sentence.',
            timeout=60,
        )
        print(f"\n💬 Query: 'What is 2+2? Answer in one sentence.'")
        print(f"🤖 Response: {reply.content}")

    print("\n✅ Native h2ogpte client test PASSED!")

except ImportError:
    print("\n⚠️  h2ogpte package not installed")
    print("   Install with: pip install h2ogpte==1.6.54")
except Exception as e:
    print(f"\n❌ Error with native client: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Test 2: OpenAI-Compatible API
# ============================================================================
print("\n" + "="*80)
print("TEST 2: OpenAI-Compatible API")
print("="*80)

try:
    from openai import OpenAI

    print("\n✅ openai package installed")

    # Create OpenAI client pointing to H2OGPTE
    print(f"\n🔌 Connecting to {ADDRESS}/openai_api/v1...")
    openai_client = OpenAI(
        api_key=API_KEY,
        base_url=f"{ADDRESS}/openai_api/v1"
    )
    print("✅ OpenAI client created successfully")

    # List available models
    print("\n📋 Listing available models...")
    models = openai_client.models.list()
    model_list = []
    for m in models:
        model_list.append(m.id)

    print(f"\n✅ Found {len(model_list)} models:")
    for i, model in enumerate(model_list, 1):
        print(f"   {i}. {model}")

    # Test chat completion with first available model or gpt-4o
    test_model = "gpt-4o" if "gpt-4o" in model_list else model_list[0]
    print(f"\n📝 Testing chat completion with model='{test_model}'...")

    response = openai_client.chat.completions.create(
        model=test_model,
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one word.",
            },
        ],
    )

    print(f"\n💬 Query: 'What is the capital of France? Answer in one word.'")

    if response.choices and len(response.choices) > 0:
        answer = response.choices[0].message.content
        print(f"🤖 Response: {answer}")
    else:
        print(f"⚠️  No choices in response")

    if hasattr(response, 'model'):
        print(f"📊 Model used: {response.model}")
    if hasattr(response, 'usage') and response.usage:
        print(f"📊 Tokens used: {response.usage.total_tokens}")

    print("\n✅ OpenAI-compatible API test PASSED!")

except ImportError:
    print("\n⚠️  openai package not installed")
    print("   Install with: pip install openai")
except Exception as e:
    print(f"\n❌ Error with OpenAI API: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n✅ Available API Methods:")
print("   1. Native h2ogpte client - Full featured")
print("   2. OpenAI-compatible API - Easy migration from OpenAI")

print("\n📚 Next Steps:")
print("   - Use native client for RAG (collections, documents)")
print("   - Use OpenAI API for simple LLM completions")
print("   - Check available models list above")

print("\n" + "="*80)
