#!/usr/bin/env python3
"""
Create template files from CNN/DM samples.
Extracts 3 samples and exports to CSV, JSON, and XLSX formats.
"""

import json
import pandas as pd
from pathlib import Path

def create_templates(
    input_file="cnn_dm_sample_with_gen_sum.json",
    num_samples=3,
    include_reference=False,
    output_prefix="cnn_dm_template"
):
    """
    Create template files from CNN/DM samples in CSV, JSON, and XLSX formats.

    Args:
        input_file: Input JSON file with CNN/DM samples
        num_samples: Number of samples to extract (default: 3)
        include_reference: Whether to include reference_summary column (default: False)
        output_prefix: Prefix for output files (default: cnn_dm_template)
    """
    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir.parent / "processed" / input_file
    examples_dir = script_dir.parent / "examples"

    # Ensure examples directory exists
    examples_dir.mkdir(exist_ok=True)

    print(f"Loading samples from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract first N samples
    samples = data[:num_samples]
    print(f"Extracted {len(samples)} samples")

    # Prepare data for export
    export_data = []
    for sample in samples:
        item = {
            "source": sample["source"],
            "summary": sample["summary"]
        }
        if include_reference and "reference_summary" in sample:
            item["reference_summary"] = sample["reference_summary"]
        export_data.append(item)

    # Create DataFrame
    df = pd.DataFrame(export_data)

    # Output file paths
    csv_path = examples_dir / f"{output_prefix}.csv"
    json_path = examples_dir / f"{output_prefix}.json"
    xlsx_path = examples_dir / f"{output_prefix}.xlsx"

    # 1. Export to CSV
    print(f"\nExporting to CSV: {csv_path}")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  ✓ Created {csv_path.name}")

    # 2. Export to JSON
    print(f"\nExporting to JSON: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Created {json_path.name}")

    # 3. Export to XLSX
    print(f"\nExporting to XLSX: {xlsx_path}")
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print(f"  ✓ Created {xlsx_path.name}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Samples extracted: {len(samples)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Output files:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {xlsx_path}")
    print(f"\nPreview of first sample:")
    print(f"  Source length: {len(export_data[0]['source'])} characters")
    print(f"  Summary length: {len(export_data[0]['summary'])} characters")
    if include_reference:
        print(f"  Reference summary length: {len(export_data[0].get('reference_summary', ''))} characters")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "cnn_dm_sample_with_gen_sum.json"
    NUM_SAMPLES = 3
    INCLUDE_REFERENCE = True  # Set to False to exclude reference_summary column
    OUTPUT_PREFIX = "cnn_dm_template"

    print("="*60)
    print("CNN/DM Template Generator")
    print("="*60)
    print()

    create_templates(
        input_file=INPUT_FILE,
        num_samples=NUM_SAMPLES,
        include_reference=INCLUDE_REFERENCE,
        output_prefix=OUTPUT_PREFIX
    )

    print("\n" + "="*60)
    print("Done! Template files created in data/examples/")
    print("="*60)
