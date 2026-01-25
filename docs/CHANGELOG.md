# SumOmniEval - Changelog

All notable changes to this project are documented here.

---

## [1.0.0] - 2026-01-25

### Major Release: 15 Metrics Complete

Complete implementation of workshop plan metrics with comprehensive testing.

### Added

#### New Metrics (Era 3A)
- **FactCC (BERT)** - Local BERT-based consistency checker (~400MB)
  - Fine-tuned model for factual consistency
  - Binary classification: Consistent vs Inconsistent
  - Confidence scoring (0-1 scale)
  - Fast inference (~15 seconds)
  - UI checkbox: "Era 3A: Logic Checkers → + FactCC (BERT)"

#### New Metrics (Era 3B)
- **DAG (DeepEval)** - API-based decision tree evaluation
  - Structured 3-step evaluation process
  - Step 1: Factual Accuracy (0-2 points)
  - Step 2: Completeness (0-2 points)
  - Step 3: Clarity (0-2 points)
  - Total scoring: 0-6 points (normalized to 0-1)
  - UI checkbox: "Era 3B: AI Simulators → + DAG (DeepEval)"

#### Documentation
- **README.md** - Comprehensive project overview
- **docs/METRICS.md** - Detailed metric explanations
- **docs/SETUP.md** - Installation and configuration guide
- **docs/CHANGELOG.md** - This file
- **tests/README.md** - Testing documentation

#### Tests
- **test_all_new_metrics.py** - Comprehensive test suite
  - Tests FactCC with good/bad summaries
  - Tests DAG decision tree evaluation
  - Tests Era 3A integration (all 3 metrics)
  - Tests Era 3B integration (all 5 metrics)
- Moved all test files to `tests/` directory

### Changed

#### Model Configuration
- **Default model**: Changed to `meta-llama/Llama-3.3-70B-Instruct`
- **Model dropdown**: Limited to 3 tested models only
  - meta-llama/Llama-3.3-70B-Instruct
  - meta-llama/Meta-Llama-3.1-70B-Instruct
  - deepseek-ai/DeepSeek-R1
- Removed untested models from dropdown

#### Era Naming
- **Corrected**: "Era 4" → "Era 3B: AI Simulators"
- Aligned with workshop plan structure
- Updated all documentation and UI labels
- Renamed: `ERA4_LLM_JUDGE.md` → `ERA3B_LLM_JUDGE.md`

#### UI Improvements
- Added 2-column layout for Era 3A metrics
- Added step-by-step display for DAG evaluation
- Updated metric count: "Up to 15 metrics"
- Improved help text for new checkboxes

#### Project Structure
- Organized tests into `tests/` directory
- Organized documentation into `docs/` directory
- Cleaned up redundant markdown files
- Created modular documentation structure

### Fixed

- Model name verification (added "Meta-" prefix for Llama-3.1)
- Era 3B foolproofing - each dimension works individually
- Improved error handling for API metrics
- Fixed timeout issues with long evaluations

### Removed

- **Redundant files**:
  - FINAL_IMPLEMENTATION_SUMMARY.md (consolidated into README)
  - H2OGPTE_API_SUMMARY.md (consolidated into SETUP)
  - PROJECT_STRUCTURE.md (integrated into README)
  - QUICK_START.md (integrated into README)
  - Multiple duplicate test files

### Metrics Not Implemented

Documented reasons for skipped workshop plan metrics:

- **AlignScore** (~1.4GB) - Over 1GB budget constraint
- **QuestEval** - Cython compilation errors with spacy dependency
- **MENLI** - Redundant with existing NLI implementation
- **Prometheus** - Complex local model setup, would require significant infrastructure

### Workshop Plan Coverage

**Total**: 9/12 workshop metrics (75%)

**Implemented**:
- ✅ Era 1: All 5 word overlap metrics (ROUGE, BLEU, METEOR, Levenshtein, Perplexity)
- ✅ Era 2: Embedding metrics (BERTScore + MoverScore)
- ✅ Era 3A: NLI (DeBERTa-v3) + FactCC (BERT)
- ✅ Era 3A: FactChecker (LLM-powered API)
- ✅ Era 3B: G-Eval (4 dimensions: Faithfulness, Coherence, Relevance, Fluency)
- ✅ Era 3B: DAG (Decision tree evaluation)

**Skipped** (with valid technical reasons):
- ❌ AlignScore (budget)
- ❌ QuestEval (dependencies)
- ❌ MENLI (redundant)
- ❌ Prometheus (complexity)

---

## [0.9.0] - 2026-01-24

### Era 3B Implementation

#### Added
- **G-Eval** implementation with 4 dimensions:
  - Faithfulness: Factual accuracy evaluation
  - Coherence: Logical flow assessment
  - Relevance: Information coverage check
  - Fluency: Writing quality analysis
- **Era 3B UI** with individual metric toggles
- **LLM Judge** class for API-based evaluation
- **Configurable model selection** in UI

#### Changed
- Updated Era 3 to support both local (NLI) and API (LLM) metrics
- Enhanced UI with Era 3B section
- Added timeout configuration for API calls

---

## [0.8.0] - 2026-01-23

### Era 3A Implementation

#### Added
- **NLI (DeBERTa-v3)** - Natural Language Inference
- **FactChecker (API)** - LLM-based fact-checking
- Era 3A UI section with checkboxes
- H2OGPTE API integration

#### Changed
- Refactored Era 3 evaluator structure
- Added API key configuration via .env
- Updated requirements.txt with h2ogpte

---

## [0.7.0] - 2026-01-20

### Era 2 Implementation

#### Added
- **BERTScore** - Contextual embedding similarity
- **MoverScore** - Earth Mover's Distance alignment
- Model caching for faster subsequent runs

#### Changed
- Improved error handling for model downloads
- Added progress indicators for model loading

---

## [0.6.0] - 2026-01-18

### Era 1 Complete

#### Added
- **ROUGE** (1, 2, L variants)
- **BLEU**
- **METEOR**
- **Levenshtein Distance**
- **Perplexity**
- Basic Streamlit UI

#### Changed
- Organized evaluators by era
- Created modular metric structure

---

## [0.5.0] - 2026-01-15

### Initial Implementation

#### Added
- Project structure
- Basic ROUGE implementation
- requirements.txt
- Initial README

---

## Metric Evolution Summary

| Version | Era 1 | Era 2 | Era 3A | Era 3B | Total |
|---------|-------|-------|--------|--------|-------|
| 0.5.0   | 1     | 0     | 0      | 0      | 1     |
| 0.6.0   | 5     | 0     | 0      | 0      | 5     |
| 0.7.0   | 5     | 2     | 0      | 0      | 7     |
| 0.8.0   | 5     | 2     | 2      | 0      | 9     |
| 0.9.0   | 5     | 2     | 2      | 4      | 13    |
| **1.0.0** | **5** | **2** | **3** | **5** | **15** |

---

## Performance Evolution

| Version | Local Metrics | API Metrics | Evaluation Time |
|---------|---------------|-------------|-----------------|
| 0.6.0   | 5             | 0           | ~5s             |
| 0.7.0   | 7             | 0           | ~15s            |
| 0.8.0   | 8             | 1           | ~45s            |
| 0.9.0   | 8             | 5           | ~7min           |
| **1.0.0** | **9**     | **6**       | **~8min**       |

---

## Breaking Changes

### None in 1.0.0
All changes are backward compatible. Existing workflows continue to work.

### Future Considerations
- May add batch evaluation API
- May add result export functionality
- May add custom metric plugins

---

## Known Issues

### Current Limitations
1. **Era 3B speed**: API metrics take ~1-2 minutes each
   - Workaround: Use local metrics only for fast evaluation
2. **Model download size**: Initial download is ~3GB
   - Workaround: Pre-download models during setup
3. **Memory usage**: Full evaluation uses ~4GB RAM
   - Workaround: Run fewer metrics concurrently

### Planned Improvements
- [ ] Add result caching to avoid re-evaluation
- [ ] Add batch processing for multiple summaries
- [ ] Add progress bars for long evaluations
- [ ] Add result export (CSV, JSON)
- [ ] Add visualization charts

---

## Upgrade Guide

### From 0.9.0 to 1.0.0

**No breaking changes!** Simply:

1. Pull latest code
2. Update dependencies:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
3. Run tests to verify:
   ```bash
   python3 tests/test_all_new_metrics.py
   ```

**New features available**:
- ✅ FactCC checkbox in Era 3A
- ✅ DAG checkbox in Era 3B
- ✅ 2 additional metrics (13 → 15)

---

## Contributors

- Initial implementation and all metrics
- Workshop plan alignment
- Documentation
- Testing suite

---

## License

See LICENSE file

---

**Current Version**: 1.0.0
**Release Date**: 2026-01-25
**Status**: Production Ready ✅
