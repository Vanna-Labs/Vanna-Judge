from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import vanna_judge.cli as cli
from vanna_judge.schemas import JudgeVerdict


class DummyJudge:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

    async def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        _ = question, expected_answer
        lowered = system_answer.lower()
        if "wrong" in lowered:
            return JudgeVerdict.INCORRECT, "wrong"
        if "don't know" in lowered:
            return JudgeVerdict.ABSTAINED, "abstained"
        return JudgeVerdict.CORRECT, "correct"


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.mark.parametrize("key", [None, "items", "rows", "qa_pairs", "results"])
def test_load_input_items_accepts_supported_formats(
    tmp_path: Path,
    key: str | None,
) -> None:
    rows = [
        {
            "id": 1,
            "question": "Q1",
            "expected_answer": "A1",
            "system_answer": "A1",
        }
    ]
    payload: object = rows if key is None else {key: rows}
    input_file = tmp_path / f"input_{key or 'list'}.json"
    _write_json(input_file, payload)

    loaded = cli._load_input_items(str(input_file))
    assert loaded == rows


def test_load_input_items_rejects_invalid_shape(tmp_path: Path) -> None:
    input_file = tmp_path / "invalid.json"
    _write_json(input_file, {"unexpected": "value"})

    with pytest.raises(ValueError, match="Expected a JSON list"):
        cli._load_input_items(str(input_file))


@pytest.mark.asyncio
async def test_cli_run_writes_results_with_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "LLMJudge", DummyJudge)

    rows = [
        {"id": 10, "question": "Q1", "answer": "A1", "prediction": "A1"},
        {"id": 11, "question": "Q2", "expected_answer": "A2", "model_answer": "wrong"},
        {
            "id": 12,
            "question": "Q3",
            "expected_answer": "A3",
            "system_answer": "I don't know",
        },
        {"id": 13, "question": "Q4", "answer": "A4", "prediction": "A4"},
    ]
    input_file = tmp_path / "rows.json"
    _write_json(input_file, rows)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        input=str(input_file),
        system_name="CLI Test System",
        model="test-model",
        temperature=0.0,
        timeout_s=5.0,
        max_retries=1,
        retry_backoff_s=0.0,
        concurrency=3,
        limit=3,
        output_dir=str(output_dir),
    )

    exit_code = await cli._run(args)
    assert exit_code == 0

    output_files = sorted(output_dir.glob("eval_cli_test_system_*.json"))
    assert len(output_files) == 1

    with open(output_files[0], "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["metadata"]["system_name"] == "CLI Test System"
    assert payload["metadata"]["config"]["model"] == "test-model"
    assert payload["summary"]["total_questions"] == 3
    assert payload["summary"]["correct"] == 1
    assert payload["summary"]["abstained"] == 1
    assert payload["summary"]["incorrect"] == 1
    assert len(payload["results"]) == 3
