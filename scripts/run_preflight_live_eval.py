#!/usr/bin/env python3
"""Run a live preflight evaluation against a larger curated test set."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

LLMJudge: Any | None = None
EvaluationRunner: Any | None = None


def _configure_warning_filters() -> None:
    """Silence known noisy serializer warnings from dependency internals."""
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings:.*",
        category=UserWarning,
        module=r"pydantic\.main",
    )


def _load_runtime_classes() -> tuple[type[Any], type[Any]]:
    global LLMJudge, EvaluationRunner
    if LLMJudge is not None and EvaluationRunner is not None:
        return LLMJudge, EvaluationRunner

    try:
        from vanna_judge import EvaluationRunner as _evaluation_runner
        from vanna_judge import LLMJudge as _llm_judge
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "vanna_judge":
            raise RuntimeError(
                "Missing dependency "
                f"'{exc.name}'. Install dependencies with: "
                "python3 -m pip install -e '.[dev]'"
            ) from exc

        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "src"))
        try:
            from vanna_judge import EvaluationRunner as _evaluation_runner
            from vanna_judge import LLMJudge as _llm_judge
        except ModuleNotFoundError as inner_exc:
            missing = inner_exc.name or "required dependency"
            raise RuntimeError(
                "Missing dependency "
                f"'{missing}'. Install dependencies with: "
                "python3 -m pip install -e '.[dev]'"
            ) from inner_exc

    EvaluationRunner = _evaluation_runner
    LLMJudge = _llm_judge
    return _llm_judge, _evaluation_runner


def _load_env_file(filepath: str) -> None:
    """Populate os.environ from a simple .env file if present."""
    env_path = Path(filepath)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _load_items(filepath: str) -> list[dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of evaluation items.")
    return [dict(item) for item in data]


async def _run(args: argparse.Namespace) -> int:
    _configure_warning_filters()
    _load_env_file(args.env_file)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Add it to environment or .env, then rerun.",
        )
        return 2

    try:
        llm_judge_cls, evaluation_runner_cls = _load_runtime_classes()
    except RuntimeError as exc:
        print(str(exc))
        return 2

    items = _load_items(args.input)
    if args.limit is not None:
        items = items[: args.limit]

    judge = llm_judge_cls(
        model=args.model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
    )
    runner = evaluation_runner_cls(
        judge=judge,
        system_name=args.system_name,
        max_concurrency=args.concurrency,
    )
    results, summary = await runner.evaluate_precomputed(items)

    expected_verdicts = [
        str(item.get("expected_verdict", "")).strip().lower() for item in items
    ]
    labeled_total = sum(1 for verdict in expected_verdicts if verdict)
    matches = sum(
        1
        for result, expected in zip(results, expected_verdicts)
        if expected and result.verdict.value == expected
    )
    verdict_match_rate = (matches / labeled_total) if labeled_total else 0.0
    error_rate = (
        (summary.errors / summary.total_questions) if summary.total_questions else 0.0
    )

    expected_counts = Counter(v for v in expected_verdicts if v)
    actual_counts = Counter(result.verdict.value for result in results)

    print("\nPreflight Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.input}")
    print(f"System: {args.system_name}")
    print(f"Model: {args.model}")
    print(f"Rows evaluated: {summary.total_questions}")
    print(f"Judge error rate: {error_rate:.1%}")
    if labeled_total:
        print(
            f"Verdict match rate: "
            f"{verdict_match_rate:.1%} ({matches}/{labeled_total})"
        )
    print("")
    print(f"Expected counts: {dict(expected_counts)}")
    print(f"Actual counts:   {dict(actual_counts)}")
    print("=" * 60)

    meets_error_target = error_rate <= args.max_error_rate
    meets_match_target = (
        True if labeled_total == 0 else verdict_match_rate >= args.min_verdict_match
    )
    passed = meets_error_target and meets_match_target

    print(
        "Thresholds: "
        f"max_error_rate<={args.max_error_rate:.1%}, "
        f"min_verdict_match>={args.min_verdict_match:.1%}"
    )
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a live preflight benchmark before deployment.",
    )
    parser.add_argument(
        "--input",
        default="eval_datasets/preflight_large.json",
        help="Path to JSON dataset with precomputed system answers.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env file to load before running.",
    )
    parser.add_argument(
        "--system-name",
        default="PreflightSystem",
        help="Name for summary reporting.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VANNA_JUDGE_MODEL", "gpt-5.1"),
        help="Judge model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Timeout per judge call in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry attempts for transient errors.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent judge calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to evaluate.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Fail if judge error rate exceeds this fraction.",
    )
    parser.add_argument(
        "--min-verdict-match",
        type=float,
        default=0.75,
        help="Fail if verdict match rate is below this fraction.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
