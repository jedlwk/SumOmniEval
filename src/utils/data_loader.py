"""Data loading utilities for sample data."""

import pandas as pd
import os
from typing import Dict, Optional


def load_sample_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample data from CSV file.

    Args:
        data_path: Path to the CSV file. If None, uses default location.

    Returns:
        DataFrame with 'report' and 'summary' columns.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If the CSV doesn't have required columns.
    """
    if data_path is None:
        # Default path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'data', 'sample_data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sample data file not found at: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Validate required columns
    required_columns = ['report', 'summary']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    return df


def get_sample_by_index(index: int, data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get a specific sample by index.

    Args:
        index: Zero-based index of the sample to retrieve.
        data_path: Path to the CSV file. If None, uses default location.

    Returns:
        Dictionary with 'source' and 'summary' keys.

    Raises:
        IndexError: If index is out of bounds.
    """
    df = load_sample_data(data_path)

    if index < 0 or index >= len(df):
        raise IndexError(
            f"Index {index} out of bounds. "
            f"Valid range: 0 to {len(df) - 1}"
        )

    row = df.iloc[index]

    return {
        'source': row['report'],
        'summary': row['summary']
    }


def get_sample_titles(data_path: Optional[str] = None, max_length: int = 100) -> list:
    """
    Get abbreviated titles for each sample (first N chars of source).

    Args:
        data_path: Path to the CSV file. If None, uses default location.
        max_length: Maximum length of each title.

    Returns:
        List of abbreviated titles.
    """
    df = load_sample_data(data_path)

    titles = []
    for idx, row in df.iterrows():
        source = row['report']
        # Extract first sentence or first max_length characters
        first_part = source[:max_length].strip()
        if len(source) > max_length:
            first_part += "..."
        titles.append(f"Sample {idx + 1}: {first_part}")

    return titles
