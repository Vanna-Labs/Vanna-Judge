from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from vanna_judge.schemas import EvalResult, EvalSummary, JudgeVerdict

_SCRIPT_PATH = Path("scripts/run_preflight_live_eval.py")
_SPEC = importlib.util.spec_from_file_location(
    "run_preflight_live_eval",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
preflight = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preflight)


def test_preflight_dataset_is_large_and_balanced() -> None:
    dataset = Path("eval_datasets/preflight_large.json")
    with open(dataset, "r", encoding="utf-8") as f:
        rows = json.load(f)

    assert len(rows) >= 80
    expected = {row["expected_verdict"] for row in rows}
    assert expected == {"correct", "partially", "abstained", "incorrect"}


@pytest.mark.asyncio
async def test_preflight_run_exits_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = argparse.Namespace(
        input="eval_datasets/preflight_large.json",
        env_file="missing.env",
        system_name="PreflightSystem",
        model="test-model",
        temperature=0.0,
        timeout_s=1.0,
        max_retries=0,
        retry_backoff_s=0.0,
        concurrency=4,
        limit=5,
        max_error_rate=0.05,
        min_verdict_match=0.75,
    )

    exit_code = await preflight._run(args)
    assert exit_code == 2


@pytest.mark.asyncio
async def test_preflight_run_threshold_logic_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyJudge:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    class DummyRunner:
        def __init__(
            self,
            judge: object,
            system_name: str,
            max_concurrency: int,
        ) -> None:
            _ = judge, system_name, max_concurrency

        async def evaluate_precomputed(
            self,
            items: list[dict[str, str]],
        ) -> tuple[list[EvalResult], EvalSummary]:
            results: list[EvalResult] = []
            for item in items:
                verdict = JudgeVerdict(item["expected_verdict"])
                results.append(
                    EvalResult(
                        question_id=int(item["id"]),
                        question=item["question"],
                        expected_answer=item["expected_answer"],
                        system_answer=item["system_answer"],
                        verdict=verdict,
                        judge_reasoning="mock",
                        timing_ms=1,
                    )
                )
            return results, EvalSummary.from_results("PreflightSystem", results)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(preflight, "LLMJudge", DummyJudge)
    monkeypatch.setattr(preflight, "EvaluationRunner", DummyRunner)

    args = argparse.Namespace(
        input="eval_datasets/preflight_large.json",
        env_file=".env",
        system_name="PreflightSystem",
        model="test-model",
        temperature=0.0,
        timeout_s=1.0,
        max_retries=0,
        retry_backoff_s=0.0,
        concurrency=4,
        limit=20,
        max_error_rate=0.05,
        min_verdict_match=0.75,
    )

    exit_code = await preflight._run(args)
    assert exit_code == 0
