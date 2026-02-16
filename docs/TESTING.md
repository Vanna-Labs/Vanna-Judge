# Testing

## Goals

- Catch logic regressions before release.
- Validate parser/runner behavior across input shapes.
- Validate end-to-end preflight quality with live judge calls.

## Test Layers

## 1) Unit + Integration (Fast, No API)

Run:

```bash
pytest -q
```

Coverage includes:

- Judge behavior
  - file: `tests/test_judge.py`
  - abstention fast-path, retry logic, sync-wrapper constraints, batch order
- Runner behavior
  - file: `tests/test_runner.py`
  - fallback keys, error propagation from answer functions
- Schema math/serialization
  - file: `tests/test_schemas.py`
- High-volume regression matrices
  - file: `tests/test_large_regression.py`
  - large deterministic sets across all verdict classes and ID edge cases
- CLI behavior
  - file: `tests/test_cli.py`
  - accepted input formats, row limit handling, output file generation
- Utility functions
  - file: `tests/test_utils_extended.py`
  - data loading, cache persistence, result round-trip, similarity helpers
- Preflight asset and threshold checks (mocked runtime)
  - file: `tests/test_preflight_assets.py`

## 2) Live Preflight (Uses `.env`)

Run:

```bash
python3 scripts/run_preflight_live_eval.py \
  --input eval_datasets/preflight_large.json \
  --system-name MyRAG \
  --concurrency 8
```

What it checks:

- Judge error rate (`errors / total_questions`)
- Verdict match rate against `expected_verdict` labels in dataset

Default thresholds:

- `--max-error-rate 0.05`
- `--min-verdict-match 0.75`

Exit codes:

- `0`: pass
- `1`: thresholds failed
- `2`: setup/runtime issue (missing API key or dependencies)

## Dataset

- File: `eval_datasets/preflight_large.json`
- Current composition:
  - `correct`: 30
  - `partially`: 20
  - `abstained`: 15
  - `incorrect`: 15

## Local Runbook Before Push

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `python -m pip install -e ".[dev]"`
3. `pytest -q`
4. `python3 scripts/run_preflight_live_eval.py --system-name MyRAG --concurrency 8`
5. `git status` and verify `.env`/`.venv` remain ignored

