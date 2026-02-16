from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vanna_judge.schemas import EvalResult, EvalSummary, JudgeVerdict
from vanna_judge.utils import (
    EmbeddingCache,
    batch_cosine_similarity,
    compute_cosine_similarity,
    format_chunk_for_display,
    load_chunks,
    load_eval_results,
    load_qa_dataset,
    save_eval_results,
)


def test_load_chunks_normalizes_fields_and_preserves_breadcrumbs(
    tmp_path: Path,
) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "body": "body text",
                    "header_path": "A > B",
                    "metadata": {"source": "unit"},
                    "breadcrumbs": ["A", "B"],
                }
            )
            + "\n"
        )
        f.write("\n")
        f.write(json.dumps({"body": "fallback fields"}) + "\n")

    chunks = load_chunks(str(chunks_file))

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "c1"
    assert chunks[0]["breadcrumbs"] == ["A", "B"]
    assert chunks[1]["chunk_id"] == "chunk_3"
    assert chunks[1]["doc_id"] == "unknown"
    assert chunks[1]["metadata"] == {}


def test_load_chunks_invalid_json_reports_line_number(tmp_path: Path) -> None:
    chunks_file = tmp_path / "bad_chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"ok": True}) + "\n")
        f.write("{invalid json}\n")

    with pytest.raises(json.JSONDecodeError, match="line 2"):
        load_chunks(str(chunks_file))


def test_load_qa_dataset_supports_list_and_wrapped_formats(tmp_path: Path) -> None:
    list_file = tmp_path / "qa_list.json"
    wrapped_file = tmp_path / "qa_wrapped.json"

    rows = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
    with open(list_file, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    with open(wrapped_file, "w", encoding="utf-8") as f:
        json.dump({"qa_pairs": rows}, f)

    list_rows = load_qa_dataset(str(list_file))
    wrapped_rows = load_qa_dataset(str(wrapped_file))

    assert len(list_rows) == 2
    assert len(wrapped_rows) == 2
    assert list_rows[0]["id"] == 1
    assert wrapped_rows[1]["id"] == 2


def test_load_qa_dataset_rejects_invalid_structure(tmp_path: Path) -> None:
    bad_file = tmp_path / "qa_bad.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump({"not_qa_pairs": []}, f)

    with pytest.raises(KeyError, match="qa_pairs"):
        load_qa_dataset(str(bad_file))


def test_save_and_load_eval_results_round_trip(tmp_path: Path) -> None:
    results = [
        EvalResult(
            question_id=1,
            question="Q1",
            expected_answer="A1",
            system_answer="A1",
            verdict=JudgeVerdict.CORRECT,
            judge_reasoning="correct",
            timing_ms=12,
        ),
        EvalResult(
            question_id=2,
            question="Q2",
            expected_answer="A2",
            system_answer="wrong",
            verdict=JudgeVerdict.ERROR,
            judge_reasoning="timeout",
            timing_ms=25,
            judge_error="TimeoutError",
        ),
    ]
    summary = EvalSummary.from_results("RoundTrip", results)
    config = {"model": "test-model", "concurrency": 4}

    out_path = save_eval_results(results, summary, config, output_dir=str(tmp_path))
    loaded_results, loaded_summary, loaded_config = load_eval_results(out_path)

    assert loaded_config == config
    assert len(loaded_results) == 2
    assert loaded_results[1].judge_error == "TimeoutError"
    assert loaded_summary.system_name == "RoundTrip"
    assert loaded_summary.errors == 1
    assert loaded_summary.correct == 1


def test_embedding_cache_persists_and_clear_removes_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "embeddings.pkl"
    calls = {"count": 0}

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        calls["count"] += 1
        return [[float(i), float(i + 1)] for i, _ in enumerate(texts)]

    cache = EmbeddingCache(str(cache_file))
    first = cache.get_or_compute(["alpha", "beta"], embedding_fn)
    second = cache.get_or_compute(["alpha", "beta", "gamma"], embedding_fn)

    assert calls["count"] == 2
    assert first.shape == (2, 2)
    assert second.shape == (3, 2)
    assert len(cache) == 3

    reloaded = EmbeddingCache(str(cache_file))
    before = calls["count"]
    third = reloaded.get_or_compute(["alpha", "gamma"], embedding_fn)
    assert calls["count"] == before
    assert third.shape == (2, 2)

    reloaded.clear()
    assert len(reloaded) == 0
    assert not cache_file.exists()


def test_similarity_and_format_helpers() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    zero = np.array([0.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert compute_cosine_similarity(a, b) == pytest.approx(0.0)
    assert compute_cosine_similarity(a, zero) == 0.0
    assert batch_cosine_similarity(a, np.array([])).size == 0
    sims = batch_cosine_similarity(a, docs)
    assert sims[0] > sims[1]

    chunk = {
        "chunk_id": "chunk-1",
        "header_path": "Docs > Intro",
        "body": "x" * 240,
        "metadata": {"topic": "testing"},
    }
    output = format_chunk_for_display(
        chunk,
        max_body_length=40,
        include_metadata=True,
    )
    assert "[chunk-1]" in output
    assert "Docs > Intro" in output
    assert "..." in output
    assert "topic" in output
