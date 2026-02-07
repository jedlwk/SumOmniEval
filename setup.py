"""
H2O SumBench - One-shot setup script.

Installs all Python dependencies, downloads the spaCy language model,
and fetches required NLTK data. Works on macOS, Linux, and Windows.

Usage:
    python setup.py
"""

import subprocess
import sys


def run(description, cmd):
    """Run a command, printing status and raising on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode}).")
        sys.exit(result.returncode)
    print(f"\n  Done.\n")


def main():
    py = sys.executable  # current Python interpreter

    print("\nH2O SumBench Setup")
    print(f"Python: {py}\n")

    # 1. pip install
    run(
        "Installing Python dependencies (requirements.txt)",
        [py, "-m", "pip", "install", "-r", "requirements.txt"],
    )

    # 2. spaCy model
    run(
        "Downloading spaCy model (en_core_web_sm)",
        [py, "-m", "spacy", "download", "en_core_web_sm"],
    )

    # 3. NLTK data
    run(
        "Downloading NLTK data (punkt_tab)",
        [py, "-c", "import nltk; nltk.download('punkt_tab')"],
    )

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. (Optional) Copy .env.example to .env and add your H2OGPTE API key")
    print("  2. Launch the app:")
    print("       streamlit run ui/app.py")
    print()


if __name__ == "__main__":
    main()
