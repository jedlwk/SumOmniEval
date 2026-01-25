#!/usr/bin/env python3
"""
Check which workshop plan metrics can be added.
"""

import subprocess
import sys

print("="*80)
print("Checking Available Metrics for Implementation")
print("="*80)

metrics_to_check = [
    {
        "name": "FactCC",
        "packages": ["transformers"],
        "github": "https://github.com/salesforce/factCC",
        "model": "bert-base-uncased fine-tuned",
        "size": "~400MB",
        "method": "Manual model loading",
        "status": "CAN IMPLEMENT"
    },
    {
        "name": "QuestEval",
        "packages": ["questeval"],
        "github": "https://github.com/ThomasScialom/QuestEval",
        "model": "T5-base",
        "size": "~850MB",
        "method": "pip install questeval",
        "status": "CHECK IF AVAILABLE"
    },
    {
        "name": "AlignScore",
        "packages": ["alignscore"],
        "github": "https://github.com/yuh-zha/AlignScore",
        "model": "RoBERTa-large",
        "size": "~1.4GB",
        "method": "pip install alignscore",
        "status": "❌ OVER BUDGET"
    },
    {
        "name": "DeepEval (DAG)",
        "packages": ["deepeval"],
        "github": "https://github.com/confident-ai/deepeval",
        "model": "API-based or local",
        "size": "0MB (if API)",
        "method": "pip install deepeval",
        "status": "CHECK IF AVAILABLE"
    }
]

print("\nChecking package availability...\n")

for metric in metrics_to_check:
    print(f"{'='*80}")
    print(f"Metric: {metric['name']}")
    print(f"{'='*80}")
    print(f"Size: {metric['size']}")
    print(f"Status: {metric['status']}")

    if metric['status'] == "❌ OVER BUDGET":
        print("❌ SKIPPING - Over 1GB budget\n")
        continue

    print(f"\nChecking packages: {', '.join(metric['packages'])}")

    for package in metric['packages']:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ {package} is installed")
            else:
                print(f"  ⚠️ {package} not installed")
                print(f"     Install: pip install {package}")
        except Exception as e:
            print(f"  ❌ Error checking {package}: {e}")
    print()

print("="*80)
print("IMPLEMENTATION PLAN")
print("="*80)

plan = """
✅ WILL IMPLEMENT:

1. FactCC (~400MB)
   - Use Hugging Face transformers
   - Load fine-tuned BERT model from Salesforce
   - Method: Manual implementation

2. QuestEval (~850MB)
   - Try: pip install questeval
   - If not available: Skip (complex setup)
   - Method: Package if available, else manual

3. DeepEval/DAG (0MB if API-based)
   - Try: pip install deepeval
   - Can use with our API or standalone
   - Method: Use with H2OGPTE API

❌ WILL NOT IMPLEMENT:

1. AlignScore (~1.4GB)
   - Over 1GB budget

2. MENLI
   - Similar to existing NLI
   - Would be redundant

Implementation order:
1. FactCC (Era 3A) - Straightforward
2. DeepEval/DAG (Era 3B) - API-based
3. QuestEval (Era 3A) - If package works
"""

print(plan)
