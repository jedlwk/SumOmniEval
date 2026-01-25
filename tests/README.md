# SumOmniEval - Testing Guide

This directory contains all test scripts for verifying metric functionality.

## Quick Test

Run the comprehensive test suite to verify all metrics:

```bash
python3 tests/test_all_new_metrics.py
```

**Expected output**: All metrics should pass with ✅ indicators.

---

## Test Files

### Comprehensive Tests

**`test_all_new_metrics.py`** - Main test suite
- Tests all newly added metrics (FactCC, DAG)
- Tests Era 3A integration (NLI + FactCC + FactChecker)
- Tests Era 3B integration (G-Eval + DAG)
- **Run this first** to verify everything works

```bash
python3 tests/test_all_new_metrics.py
```

### Era-Specific Tests

**`test_era3a_factchecker.py`** - Era 3A Logic Checkers
- Tests NLI (DeBERTa-v3)
- Tests FactCC (BERT)
- Tests FactChecker (API)

```bash
python3 tests/test_era3a_factchecker.py
```

**`test_era3b_individual.py`** - Era 3B AI Simulators
- Tests each G-Eval dimension individually
- Tests DAG decision tree evaluation
- Verifies all API metrics work independently

```bash
python3 tests/test_era3b_individual.py
```

**`test_era3b_final.py`** - Era 3B Integration
- Tests all Era 3B metrics together
- Verifies combined evaluation

```bash
python3 tests/test_era3b_final.py
```

### API Tests

**`test_h2ogpte_api.py`** - API Connectivity
- Tests H2OGPTE API connection
- Verifies API key and address configuration
- Simple query to confirm access

```bash
python3 tests/test_h2ogpte_api.py
```

**`test_corrected_models.py`** - Model Verification
- Tests which models are available
- Verifies model names are correct
- Useful for debugging model access issues

```bash
python3 tests/test_corrected_models.py
```

### Legacy Tests

**`test_evaluators.py`** - Original test suite
- Tests Era 1 and Era 2 metrics
- Basic functionality verification

```bash
python3 tests/test_evaluators.py
```

**`test_llm_judge.py`** - LLM Judge tests
- Early Era 3B testing
- Superseded by test_era3b_individual.py

---

## Research Scripts

**`check_available_metrics.py`** - Metric Research
- Investigates which workshop plan metrics can be implemented
- Checks model sizes and dependencies
- Documents why certain metrics were skipped

```bash
python3 tests/check_available_metrics.py
```

---

## Expected Test Results

### test_all_new_metrics.py

```
================================================================================
Testing All New Metrics (FactCC + DAG)
================================================================================

TEST 1: FactCC (BERT-based Consistency Checker)
✅ Good summary: 0.9941 (Highly Consistent)
✅ Bad summary: 0.0055 (Inconsistent)

TEST 2: DAG (Decision Tree Evaluation)
✅ Good summary: 6/6 (1.000)
   Step 1 (Factual): 2/2
   Step 2 (Complete): 2/2
   Step 3 (Clarity): 2/2

✅ Bad summary: 2/6 (0.333)
   Step 1 (Factual): 0/2
   Step 2 (Complete): 0/2
   Step 3 (Clarity): 2/2

TEST 3: All Era 3A Metrics Together
  NLI: ✅ 0.4418
  FactCC: ✅ 0.9941
  FactChecker: ✅ 1.0000

TEST 4: All Era 3B Metrics Together
  faithfulness: ✅ 0.900
  coherence: ✅ 0.700
  relevance: ✅ 0.900
  fluency: ✅ 0.800
  dag: ✅ 1.000

✅ All new metrics tested successfully!
```

---

## Troubleshooting

### API Tests Failing

**Error**: "H2OGPTE_API_KEY not found"
**Solution**: Create `.env` file in project root:
```bash
H2OGPTE_API_KEY=your_key_here
H2OGPTE_ADDRESS=https://your-instance.h2ogpte.com
```

**Error**: "Invalid model"
**Solution**: Run `test_corrected_models.py` to check available models

### Local Tests Failing

**Error**: "Model not found" or "Download error"
**Solution**: Ensure internet connection for first-time model downloads

**Error**: "Out of memory"
**Solution**: Close other applications, or run fewer metrics at once

### Import Errors

**Error**: "ModuleNotFoundError"
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

---

## Test Data

All tests use consistent test data:

**Good Summary**: Accurate facts matching source
- Should score high (>0.7) on all metrics

**Bad Summary**: Multiple factual errors
- Should score low (<0.5) on factual consistency metrics
- May still score okay on fluency/coherence

---

## Adding New Tests

When adding new metrics:

1. Create test in this directory
2. Follow naming convention: `test_<feature>.py`
3. Include both positive and negative test cases
4. Update this README with test description

---

## Performance

Test execution times (approximate):

| Test File | Time | API Calls |
|-----------|------|-----------|
| test_evaluators.py | ~30s | 0 |
| test_era3a_factchecker.py | ~60s | 1 |
| test_era3b_individual.py | ~5min | 5 |
| test_all_new_metrics.py | ~6min | 6 |

**Total test suite**: ~10 minutes (if running all tests sequentially)

---

**Last Updated**: 2026-01-25
**Total Test Files**: 9
**Coverage**: All 15 metrics
