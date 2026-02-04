#!/usr/bin/env python3
"""
Download the first 10 rows of CNN/DailyMail dataset from HuggingFace.
Uses streaming to avoid downloading the entire dataset.
"""

import json
from pathlib import Path
from datasets import load_dataset

def download_cnn_dm_sample(num_rows=10, output_file="cnn_dm_sample.json"):
    """
    Download a sample of CNN/DailyMail dataset using streaming.

    Args:
        num_rows: Number of rows to download (default: 10)
        output_file: Output filename (default: cnn_dm_sample.json)
    """
    print(f"Loading CNN/DailyMail dataset with streaming...")

    # Load dataset with streaming enabled to save space
    # Using the train split and version 3.0.0
    dataset = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="train",
        streaming=True
    )

    print(f"Downloading first {num_rows} rows...")

    # Collect the first N rows
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_rows:
            break

        # Extract article and highlights (summary)
        sample = {
            "id": item.get("id", f"sample_{i}"),
            "source": item["article"],
            "summary": item["highlights"]
        }
        samples.append(sample)
        print(f"  Downloaded row {i+1}/{num_rows}")

    # Save to JSON file in data/raw/ directory
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "raw" / output_file

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Successfully saved {len(samples)} rows to {output_path}")
    print(f"\nSample structure:")
    print(f"  - id: Document identifier")
    print(f"  - source: Full article text")
    print(f"  - summary: Reference highlights/summary")

if __name__ == "__main__":
    # Download first 10 rows
    download_cnn_dm_sample(num_rows=10, output_file="cnn_dm_sample.json")
