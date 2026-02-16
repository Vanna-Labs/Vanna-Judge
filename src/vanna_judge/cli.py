"""Command line interface for vanna_judge."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from vanna_judge.judge import LLMJudge
from vanna_judge.runner import EvaluationRunner
from vanna_judge.utils import print_eval_summary, save_eval_results


def _load_input_items(filepath: str) -> list[dict[str, Any]]:
    """Load precomputed evaluation items from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [dict(item) for item in data]

    if isinstance(data, dict):
        for key in ("items", "rows", "qa_pairs", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]

    raise ValueError(
        "Expected a JSON list, or a JSON object with one of keys: "
        "'items', 'rows', 'qa_pairs', 'results'."
    )


async def _run(args: argparse.Namespace) -> int:
    items = _load_input_items(args.input)
    if args.limit is not None:
        items = items[: args.limit]

    judge = LLMJudge(
        model=args.model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
    )
    runner = EvaluationRunner(
        judge=judge,
        system_name=args.system_name,
        max_concurrency=args.concurrency,
    )

    results, summary = await runner.evaluate_precomputed(items)
    print_eval_summary(summary)

    config = {
        "input_file": args.input,
        "model": args.model,
        "temperature": args.temperature,
        "timeout_s": args.timeout_s,
        "max_retries": args.max_retries,
        "retry_backoff_s": args.retry_backoff_s,
        "concurrency": args.concurrency,
    }
    output_path = save_eval_results(
        results=results,
        summary=summary,
        config=config,
        output_dir=args.output_dir,
    )
    print(f"Results saved to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="vanna-judge",
        description="Evaluate precomputed RAG answers with an LLM judge.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to JSON file containing question/expected_answer/system_answer items. "
            "Accepted fallbacks: answer for expected_answer, prediction/model_answer "
            "for system_answer."
        ),
    )
    parser.add_argument(
        "--system-name",
        default="RAGSystem",
        help="Name used in evaluation summaries.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="OpenAI model to use for judging.",
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
        help="Retry attempts for transient model errors.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds (exponential).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max number of concurrent evaluations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for saved result JSON files.",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
