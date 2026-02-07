# Getting Started with H2O SumBench

**For a colleague seeing this project for the first time**

This guide will get you up and running in 10 minutes.

---

## What Is This?

H2O SumBench is a **text summarization evaluation tool** with **24 metrics** organized into 5 dimensions:

- **Faithfulness** (4 metrics) - Does the summary stick to the source?
- **Completeness** (4 metrics) - How much key information was captured?
- **Semantic Alignment** (3 metrics) - Does the meaning match a reference summary?
- **Surface Overlap** (7 metrics) - How many words/phrases match the reference?
- **Linguistic Quality** (5 metrics) - Is it readable, logical and well structured?

**Use case**: Evaluate how good a summary is compared to the original text.

---

## Prerequisites

**You need**:
- Python 3.8+ ([check](https://www.python.org/downloads/))
- ~6GB disk space
- 8GB+ RAM

**Optional** (for API metrics):
- H2OGPTE API key and instance URL

---

## Step 1: Install (5 minutes)

```bash
# Navigate to project directory
cd H2O SumBench

# One-shot install (dependencies + spaCy model + NLTK data)
python setup.py
```

<details>
<summary>Manual install (if you prefer)</summary>

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
```
</details>

**What happens**:
- Installs Python packages (~2 min)
- Downloads spaCy language model
- Downloads NLTK tokenizer data
- First run will download ML models (~3-5 min, one-time only)

**Troubleshooting**:
- If `pip install` fails: `pip install --upgrade pip` first
- If out of memory: close other apps

---

## Step 2: Configure (Optional - 1 minute)

**Skip this if you only want the 14 local metrics.**

For API metrics (G-Eval, DAG, Prometheus), create `.env` file:

```bash
# In project root directory
cat > .env << 'EOF'
H2OGPTE_API_KEY=your_api_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
EOF
```

Replace `your_api_key_here` and `your-instance.h2ogpte.com` with actual credentials.

**Test API connection**:
```bash
python3 tests/test_h2ogpte_api.py
```

---

## Step 3: Launch (30 seconds)

```bash
streamlit run ui/app.py
```

**Browser opens automatically** at `http://localhost:8501`

If it doesn't, manually visit: `http://localhost:8501`

---

## Step 4: Evaluate Your First Summary (2 minutes)

### In the Web UI:

1. **Enter text** in left column:
   - **Source Document**: Paste the original long text
   - **Summary**: Paste the summary to evaluate

2. **Click "Evaluate Summary"**
   - All available metrics run automatically
   - Faithfulness + Completeness + Semantic + Lexical always run (local, free)
   - Linguistic Quality local metrics always run
   - API metrics run if API key is configured

3. **Wait**:
   - Local only: ~30 seconds
   - With API: ~2 minutes

4. **View results**:
   - Scores from 0-1 (higher = better)
   - Green (>0.7) = Good
   - Yellow (0.4-0.7) = Fair
   - Red (<0.4) = Poor

---

## Understanding the Metrics

### Quick Reference

**Faithfulness**:
- NLI: Is summary logically supported by source?
- FactCC: BERT-based consistency check
- AlignScore: Unified factual alignment score
- G-Eval Faithfulness: LLM-judged factual accuracy (API)

**Completeness**:
- Entity Coverage: Are named entities preserved?
- Semantic Coverage: % of source sentences covered
- BERTScore Recall: What % of source meaning captured?
- G-Eval Relevance: LLM-judged information coverage (API)

**Semantic Alignment**:
- BERTScore: Contextual meaning similarity
- MoverScore: Advanced semantic alignment
- BARTScore: Generation likelihood score

**Surface Overlap**:
- ROUGE-1/2/L: Word and phrase overlap
- BLEU: N-gram precision
- METEOR: Semantic matching with synonyms
- chrF++: Character-level F-score
- Levenshtein: Edit distance similarity

**Linguistic Quality**:
- Perplexity: GPT-2 language model fluency
- G-Eval Fluency / Coherence: LLM-judged quality (API)
- DAG: Decision tree evaluation (API)
- Prometheus: Open-source LLM judge (API)

**For details**: See [METRICS.md](METRICS.md)

---

## Common Workflows

### Fast Local Evaluation (No API)
```
Dimensions: All 5 (local metrics only)
Time: ~30 seconds
Metrics: 14
Cost: Free
```

### Full Evaluation (With API)
```
Dimensions: All 5 (local + API metrics)
Time: ~2 minutes
Metrics: 24
Cost: API calls
```

---

## Testing

Verify everything works:

```bash
# Test all metrics (comprehensive)
python3 -m pytest tests/test_all_metrics.py -v

# Test API connection
python3 tests/test_h2ogpte_api.py
```

**Expected**: All tests show pass

---

## Project Structure

```
H2O SumBench/
│
├── setup.py                        # One-shot install script
├── ui/
│   └── app.py                      # Main application - START HERE
├── requirements.txt                # Python dependencies
├── .env                            # API config (create this if needed)
├── README.md                       # Project overview
│
├── src/                            # Source code
│   ├── evaluators/
│   │   ├── tool_logic.py           # Unified tool interface
│   │   ├── era1_word_overlap.py    # Surface Overlap metrics
│   │   ├── era2_embeddings.py      # Semantic Alignment metrics
│   │   ├── era3_logic_checkers.py  # Faithfulness metrics (local)
│   │   ├── era3_llm_judge.py       # Linguistic Quality + API metrics
│   │   └── completeness_metrics.py # Completeness metrics
│   └── utils/
│       ├── force_cpu.py            # Force CPU-only PyTorch mode
│       └── data_loader.py          # Data loading utilities
│
├── tests/                          # All test scripts
│   ├── test_all_metrics.py         # Main test suite
│   └── README.md                   # Testing guide
│
└── docs/                           # Documentation
    ├── GETTING_STARTED.md          # This file
    ├── METRICS.md                  # Detailed metric explanations
    ├── SETUP.md                    # Installation & troubleshooting
    ├── FILE_FORMATS.md             # Dataset upload guide
    └── CHANGELOG.md                # Version history
```

---

## Key Files to Know

### For Using the Tool
- **ui/app.py** - The main application (run this)
- **.env** - API configuration (create if using API metrics)

### For Understanding Implementation
- **src/evaluators/era1_word_overlap.py** - Surface overlap metrics
- **src/evaluators/era2_embeddings.py** - Semantic alignment metrics
- **src/evaluators/era3_logic_checkers.py** - Faithfulness metrics
- **src/evaluators/era3_llm_judge.py** - API-based evaluation metrics
- **src/evaluators/completeness_metrics.py** - Completeness metrics

### For Documentation
- **README.md** - Project overview and quick reference
- **docs/METRICS.md** - Detailed metric explanations
- **docs/SETUP.md** - Installation and troubleshooting
- **docs/CHANGELOG.md** - What's changed in each version

### For Testing
- **tests/test_all_metrics.py** - Comprehensive test suite
- **tests/README.md** - Testing documentation

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt --force-reinstall
```

### "Model download failed"
- Check internet connection
- Models download on first use (~6GB)
- Models cache to: `~/.cache/huggingface/`

### "Out of memory"
- Close other applications
- Run fewer metrics at once

### "API connection failed"
- Check `.env` file exists and has correct credentials
- Test: `python3 tests/test_h2ogpte_api.py`

### App won't start
```bash
# Check if Streamlit is installed
streamlit --version

# Try different port
streamlit run ui/app.py --server.port 8502
```

**More help**: See [SETUP.md](SETUP.md)

---

## Next Steps

### Learn More
1. Read [METRICS.md](METRICS.md) for detailed metric explanations
2. Read [SETUP.md](SETUP.md) for advanced configuration
3. Read [CHANGELOG.md](CHANGELOG.md) for version history

### Run Tests
```bash
# Verify everything works
python3 -m pytest tests/test_all_metrics.py -v
```

### Customize
- Edit metric selection in `ui/app.py`
- Adjust timeouts in `src/evaluators/era3_llm_judge.py`
- Add custom metrics in `src/evaluators/`

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Install | `python setup.py` |
| Run app | `streamlit run ui/app.py` |
| Test all | `python3 -m pytest tests/test_all_metrics.py -v` |
| Test API | `python3 tests/test_h2ogpte_api.py` |
| View docs | Open `docs/METRICS.md` in browser |

| Dimension | Count | Type | Time | Best For |
|-----------|-------|------|------|----------|
| Faithfulness | 4 | Mixed | ~15s | Fact-checking |
| Completeness | 4 | Mixed | ~10s | Information coverage |
| Semantic Alignment | 3 | Local | ~10s | Reference matching |
| Surface Overlap | 7 | Local | ~2s | Word/phrase overlap |
| Linguistic Quality | 5 | Mixed | ~5s | Writing quality |

---

## Support

- **Documentation**: See `docs/` folder
- **Issues**: Check SETUP.md troubleshooting section
- **Tests**: Run `tests/test_all_metrics.py`

---

**Version**: 3.0
**Last Updated**: 2026-02-07
**Status**: Production Ready

**Ready to go?** Run: `streamlit run ui/app.py`
