# Architecture

## Purpose

`vanna-judge` evaluates RAG answers using an LLM-as-judge pattern with structured verdicts and summary metrics.

## Core Components

- `src/vanna_judge/judge.py`
  - `LLMJudge` wraps the LLM call.
  - Enforces structured output (`JudgeOutput`) with fields:
    - `verdict`
    - `matched_facts`
    - `missing_facts`
    - `contradictions`
    - `reasoning`
  - Handles:
    - abstention fast-path
    - retry/backoff for transient failures
    - timeout handling
    - verdict parsing into `JudgeVerdict`
- `src/vanna_judge/runner.py`
  - `EvaluationRunner` orchestrates batch evaluation.
  - Supports:
    - `evaluate_live`: calls your answer function first, then judges output.
    - `evaluate_precomputed`: judges already-generated answers.
  - Uses concurrency limits via `asyncio.Semaphore`.
- `src/vanna_judge/schemas.py`
  - `JudgeVerdict`: `correct`, `partially`, `abstained`, `incorrect`, `error`.
  - `EvalResult`: per-question result payload.
  - `EvalSummary`: aggregate metrics and percentages.
- `src/vanna_judge/utils.py`
  - Dataset loaders: `load_chunks`, `load_qa_dataset`
  - Result persistence: `save_eval_results`, `load_eval_results`
  - Embedding helpers and similarity utilities
- `src/vanna_judge/cli.py`
  - CLI entrypoint for precomputed evaluation from JSON files.

## Execution Flow

## 1) Precomputed Evaluation

1. Load rows from JSON (`question`, `expected_answer`/`answer`, `system_answer`/fallbacks).
2. For each row, call `LLMJudge.judge(...)`.
3. Store `EvalResult` with verdict and reasoning.
4. Aggregate into `EvalSummary`.
5. Save output JSON with metadata and summary.

## 2) Live Evaluation

1. Load QA set (`question`, `answer`).
2. Call your `answer_fn(question)` (sync or async).
3. Judge answer with `LLMJudge`.
4. If `answer_fn` fails, emit `JudgeVerdict.ERROR` for that row.
5. Aggregate and save results.

## Error Model

- `JudgeVerdict.ERROR` is reserved for runtime/judge failures.
- `ERROR` is counted separately from answer-quality verdicts.
- Accuracy is reported both:
  - including errors
  - excluding errors

## Prompting and Structured Output

- Judge prompt templates live in `src/vanna_judge/judge.py`:
  - `JUDGE_SYSTEM_PROMPT`
  - `JUDGE_USER_PROMPT`
- Output is validated through the `JudgeOutput` schema to keep verdict format consistent.

## Deployment/Preflight Assets

- `eval_datasets/preflight_large.json`: curated benchmark dataset.
- `scripts/run_preflight_live_eval.py`: live preflight runner with thresholds.

