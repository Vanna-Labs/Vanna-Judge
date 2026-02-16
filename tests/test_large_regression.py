from __future__ import annotations

from collections import Counter

import pytest

from vanna_judge.runner import EvaluationRunner
from vanna_judge.schemas import JudgeVerdict


class MarkerJudge:
    """Deterministic judge used for high-volume regression tests."""

    async def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        _ = question, expected_answer
        lowered = system_answer.lower()
        markers = {
            "[[correct]]": JudgeVerdict.CORRECT,
            "[[partially]]": JudgeVerdict.PARTIALLY_CORRECT,
            "[[abstained]]": JudgeVerdict.ABSTAINED,
            "[[incorrect]]": JudgeVerdict.INCORRECT,
            "[[error]]": JudgeVerdict.ERROR,
        }
        for marker, verdict in markers.items():
            if marker in lowered:
                return verdict, f"marker:{verdict.value}"
        return JudgeVerdict.ERROR, "marker:error"


def _build_precomputed_items() -> tuple[list[dict[str, str]], Counter, list[int]]:
    pattern = (
        [JudgeVerdict.CORRECT] * 10
        + [JudgeVerdict.PARTIALLY_CORRECT] * 6
        + [JudgeVerdict.ABSTAINED] * 4
        + [JudgeVerdict.INCORRECT] * 3
        + [JudgeVerdict.ERROR]
    )

    items: list[dict[str, str]] = []
    expected_counts: Counter = Counter()
    expected_ids: list[int] = []

    for index, verdict in enumerate(pattern * 10):
        row: dict[str, str] = {
            "question": f"Matrix question {index}",
            "expected_verdict": verdict.value,
        }

        expected_answer = f"Expected fact set {index}"
        if index % 2 == 0:
            row["expected_answer"] = expected_answer
        else:
            row["answer"] = expected_answer

        system_answer = f"Candidate answer {index} [[{verdict.value}]]"
        if index % 3 == 0:
            row["system_answer"] = system_answer
        elif index % 3 == 1:
            row["prediction"] = system_answer
        else:
            row["model_answer"] = system_answer

        if index % 11 == 0:
            row["id"] = f"bad-id-{index}"
            expected_ids.append(index + 1)
        elif index % 11 == 1:
            row["id"] = str(10000 + index)
            expected_ids.append(10000 + index)
        else:
            row["id"] = str(20000 + index)
            expected_ids.append(20000 + index)

        items.append(row)
        expected_counts[verdict] += 1

    return items, expected_counts, expected_ids


def _build_live_cases() -> tuple[list[dict[str, str]], dict[str, object], Counter]:
    qa_pairs: list[dict[str, str]] = []
    answer_map: dict[str, object] = {}
    expected_counts: Counter = Counter()

    for index in range(180):
        question = f"Live question {index}"
        qa_row: dict[str, str] = {
            "question": question,
            "answer": f"Ground truth {index}",
        }

        if index % 13 == 0:
            qa_row["id"] = f"non-int-{index}"
        elif index % 13 == 1:
            qa_row["id"] = str(30000 + index)
        else:
            qa_row["id"] = str(40000 + index)

        qa_pairs.append(qa_row)

        if index % 17 == 0:
            answer_map[question] = RuntimeError("upstream rag failure")
            expected_counts[JudgeVerdict.ERROR] += 1
            continue

        if index % 5 == 0:
            verdict = JudgeVerdict.PARTIALLY_CORRECT
        elif index % 5 == 1:
            verdict = JudgeVerdict.ABSTAINED
        elif index % 5 == 2:
            verdict = JudgeVerdict.INCORRECT
        else:
            verdict = JudgeVerdict.CORRECT

        answer_map[question] = f"Live response {index} [[{verdict.value}]]"
        expected_counts[verdict] += 1

    return qa_pairs, answer_map, expected_counts


@pytest.mark.asyncio
async def test_large_precomputed_regression_matrix() -> None:
    items, expected_counts, expected_ids = _build_precomputed_items()
    runner = EvaluationRunner(
        judge=MarkerJudge(),
        system_name="MatrixSystem",
        max_concurrency=32,
    )

    results, summary = await runner.evaluate_precomputed(items)

    assert len(results) == 240
    assert [result.question_id for result in results] == expected_ids
    assert Counter(result.verdict for result in results) == expected_counts

    assert summary.total_questions == 240
    assert summary.correct == expected_counts[JudgeVerdict.CORRECT]
    assert summary.partially_correct == expected_counts[JudgeVerdict.PARTIALLY_CORRECT]
    assert summary.abstained == expected_counts[JudgeVerdict.ABSTAINED]
    assert summary.incorrect == expected_counts[JudgeVerdict.INCORRECT]
    assert summary.errors == expected_counts[JudgeVerdict.ERROR]

    expected_accuracy = (
        (
            expected_counts[JudgeVerdict.CORRECT]
            + expected_counts[JudgeVerdict.PARTIALLY_CORRECT]
        )
        / 240
    ) * 100
    assert summary.accuracy_pct == pytest.approx(expected_accuracy)


@pytest.mark.asyncio
async def test_large_live_regression_matrix_sync_answer_fn() -> None:
    qa_pairs, answer_map, expected_counts = _build_live_cases()
    runner = EvaluationRunner(
        judge=MarkerJudge(),
        system_name="LiveMatrix",
        max_concurrency=24,
    )

    def answer_fn(question: str) -> str:
        response = answer_map[question]
        if isinstance(response, Exception):
            raise response
        return str(response)

    results, summary = await runner.evaluate_live(qa_pairs, answer_fn)

    assert len(results) == 180
    assert summary.total_questions == 180
    assert summary.correct == expected_counts[JudgeVerdict.CORRECT]
    assert summary.partially_correct == expected_counts[JudgeVerdict.PARTIALLY_CORRECT]
    assert summary.abstained == expected_counts[JudgeVerdict.ABSTAINED]
    assert summary.incorrect == expected_counts[JudgeVerdict.INCORRECT]
    assert summary.errors == expected_counts[JudgeVerdict.ERROR]

    failing_questions = {
        f"Live question {index}" for index in range(180) if index % 17 == 0
    }
    error_results = [
        result for result in results if result.verdict == JudgeVerdict.ERROR
    ]
    assert {result.question for result in error_results} == failing_questions
    assert all(result.system_answer == "" for result in error_results)
    assert all("RuntimeError" in (result.judge_error or "") for result in error_results)
