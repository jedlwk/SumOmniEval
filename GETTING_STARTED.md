# Getting Started with SumOmniEval

**For a colleague seeing this project for the first time**

This guide will get you up and running in 10 minutes.

---

## What Is This?

SumOmniEval is a **text summarization evaluation tool** with **15 different metrics** organized into 3 "eras":

- **Era 1**: Word-level matching (5 metrics) - Fast, basic
- **Era 2**: Semantic embeddings (2 metrics) - Meaning-aware
- **Era 3A**: Fact-checking (3 metrics) - Accuracy-focused
- **Era 3B**: AI evaluation (5 metrics) - Human-like quality assessment

**Use case**: Evaluate how good a summary is compared to the original text.

---

## Prerequisites

**You need**:
- Python 3.8+ ([check](https://www.python.org/downloads/))
- ~3GB disk space
- 8GB+ RAM

**Optional** (for API metrics):
- H2OGPTE API key and instance URL

---

## Step 1: Install (5 minutes)

```bash
# Navigate to project directory
cd SumOmniEval

# Install dependencies
pip install -r requirements.txt
```

**What happens**:
- Installs Python packages (~2 min)
- First run will download ML models (~3-5 min, one-time only)

**Troubleshooting**:
- If `pip install` fails: `pip install --upgrade pip` first
- If out of memory: close other apps

---

## Step 2: Configure (Optional - 1 minute)

**Skip this if you only want local metrics (Era 1, 2, 3A).**

For API metrics (Era 3B), create `.env` file:

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
streamlit run app.py
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
   - Era 1 and Era 2 always run (local, free)
   - Era 3A runs if models are available
   - Era 3B runs if API key is configured

3. **Wait**:
   - Local only: ~40 seconds
   - With API: ~8 minutes

4. **View results**:
   - Scores from 0-1 (higher = better)
   - 🟢 Green (>0.7) = Good
   - 🟡 Yellow (0.4-0.7) = Fair
   - 🔴 Red (<0.4) = Poor

---

## Understanding the Metrics

### Quick Reference

**Era 1 (Word Overlap)**:
- ROUGE: Do summary words appear in source?
- BLEU: Precision of word matching
- METEOR: Semantic matching with synonyms

**Era 2 (Embeddings)**:
- BERTScore: Contextual meaning similarity
- MoverScore: Advanced semantic alignment

**Era 3A (Logic Checkers)**:
- NLI: Is summary logically supported by source?
- FactCC: BERT-based consistency check
- FactChecker: LLM-powered fact verification (requires API)

**Era 3B (AI Simulators)** (requires API):
- Faithfulness: Are facts accurate?
- Coherence: Does it flow logically?
- Relevance: Are main points captured?
- Fluency: Is writing quality good?
- DAG: Step-by-step decision tree evaluation

**For details**: See [docs/METRICS.md](docs/METRICS.md)

---

## Common Workflows

### Fast Evaluation (No API)
```
✓ Era 1 + 2 + 3A (NLI only)
Time: ~30 seconds
Metrics: 8
Cost: Free
```

### Comprehensive Local (No API)
```
✓ Era 1 + 2 + 3A (all)
Time: ~40 seconds
Metrics: 9
Cost: Free
```

### Full Evaluation (With API)
```
✓ All metrics enabled
Time: ~8 minutes
Metrics: 15
Cost: 6 API calls
```

---

## Testing

Verify everything works:

```bash
# Test all metrics (comprehensive)
python3 tests/test_all_new_metrics.py

# Test local metrics only (no API needed)
python3 tests/test_evaluators.py

# Test API connection
python3 tests/test_h2ogpte_api.py
```

**Expected**: All tests show ✅

---

## Project Structure

```
SumOmniEval/
│
├── app.py                      # Main application - START HERE
├── requirements.txt            # Python dependencies
├── .env                        # API config (create this if needed)
├── README.md                   # Project overview
├── GETTING_STARTED.md          # This file
│
├── src/                        # Source code
│   ├── evaluators/
│   │   ├── era1_word_overlap.py  # Era 1 metrics (ROUGE, BLEU, etc.)
│   │   ├── era2_embeddings.py # Era 2 metrics (BERTScore, MoverScore)
│   │   ├── era3_logic_checkers.py  # Era 3A (NLI, FactCC, FactChecker)
│   │   └── era3_llm_judge.py  # Era 3B (G-Eval, DAG)
│   └── utils/
│       └── data_loader.py     # Data loading utilities
│
├── tests/                      # All test scripts
│   ├── README.md              # Testing guide
│   └── test_all_metrics.py    # Main test suite
│
└── docs/                       # Documentation
    ├── METRICS.md             # Detailed metric explanations
    ├── SETUP.md               # Installation & troubleshooting
    ├── CHANGELOG.md           # Version history
    └── workshop_lession_plan.txt  # Original requirements
```

---

## Key Files to Know

### For Using the Tool
- **app.py** - The main application (run this)
- **.env** - API configuration (create if using API metrics)

### For Understanding Implementation
- **src/evaluators/era1_word_overlap.py** - Basic word overlap metrics
- **src/evaluators/era2_embeddings.py** - Embedding-based metrics
- **src/evaluators/era3_logic_checkers.py** - Fact-checking metrics
- **src/evaluators/era3_llm_judge.py** - AI evaluation metrics

### For Documentation
- **README.md** - Project overview and quick reference
- **docs/METRICS.md** - Detailed metric explanations
- **docs/SETUP.md** - Installation and troubleshooting
- **docs/CHANGELOG.md** - What's changed in each version

### For Testing
- **tests/test_all_new_metrics.py** - Comprehensive test suite
- **tests/README.md** - Testing documentation

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt --force-reinstall
```

### "Model download failed"
- Check internet connection
- Models download on first use (~3GB)
- Models cache to: `~/.cache/huggingface/`

### "Out of memory"
- Close other applications
- Run fewer metrics at once
- Disable Era 2 or 3A to save RAM

### "API connection failed"
- Check `.env` file exists and has correct credentials
- Test: `python3 tests/test_h2ogpte_api.py`

### App won't start
```bash
# Check if Streamlit is installed
streamlit --version

# Try different port
streamlit run app.py --server.port 8502
```

**More help**: See [docs/SETUP.md](docs/SETUP.md)

---

## Next Steps

### Learn More
1. Read [docs/METRICS.md](docs/METRICS.md) for detailed metric explanations
2. Read [docs/SETUP.md](docs/SETUP.md) for advanced configuration
3. Read [docs/CHANGELOG.md](docs/CHANGELOG.md) for version history

### Run Tests
```bash
# Verify everything works
python3 tests/test_all_new_metrics.py
```

### Customize
- Edit metric selection in `app.py`
- Adjust timeouts in `src/evaluators/era3_llm_judge.py`
- Add custom metrics in `src/evaluators/`

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` |
| Test all | `python3 tests/test_all_new_metrics.py` |
| Test API | `python3 tests/test_h2ogpte_api.py` |
| View docs | Open `docs/METRICS.md` in browser |

| Metric Era | Count | Type | Time | Best For |
|------------|-------|------|------|----------|
| Era 1 | 5 | Local | ~2s | Word overlap |
| Era 2 | 2 | Local | ~10s | Semantic similarity |
| Era 3A | 3 | Mixed | ~40s | Fact-checking |
| Era 3B | 5 | API | ~7min | Human-like evaluation |

---

## Support

- **Documentation**: See `docs/` folder
- **Issues**: Check SETUP.md troubleshooting section
- **Tests**: Run `tests/test_all_new_metrics.py`

---

**Version**: 1.0
**Last Updated**: 2026-01-25
**Status**: Production Ready ✅

**Ready to go?** Run: `streamlit run app.py`
