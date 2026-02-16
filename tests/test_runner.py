import pytest

from vanna_judge.runner import EvaluationRunner
from vanna_judge.schemas import JudgeVerdict


class StubJudge:
    async def judge(
        self, question: str, expected_answer: str, system_answer: str
    ) -> tuple[JudgeVerdict, str]:
        if "wrong" in system_answer:
            return JudgeVerdict.INCORRECT, "contradiction"
        return JudgeVerdict.CORRECT, "ok"


@pytest.mark.asyncio
async def test_evaluate_precomputed_uses_fallback_keys() -> None:
    runner = EvaluationRunner(judge=StubJudge(), system_name="Test", max_concurrency=2)
    items = [
        {"id": 10, "question": "Q1", "answer": "A1", "prediction": "A1"},
        {"id": 11, "question": "Q2", "expected_answer": "A2", "system_answer": "wrong"},
    ]

    results, summary = await runner.evaluate_precomputed(items)
    assert len(results) == 2
    assert results[0].question_id == 10
    assert results[0].verdict == JudgeVerdict.CORRECT
    assert results[1].verdict == JudgeVerdict.INCORRECT
    assert summary.total_questions == 2
    assert summary.incorrect == 1


@pytest.mark.asyncio
async def test_evaluate_live_marks_answer_fn_errors() -> None:
    runner = EvaluationRunner(judge=StubJudge(), system_name="Test", max_concurrency=1)
    qa_pairs = [
        {"id": 1, "question": "ok", "answer": "A"},
        {"id": 2, "question": "boom", "answer": "B"},
    ]

    async def answer_fn(question: str) -> str:
        if question == "boom":
            raise RuntimeError("rag failure")
        return "A"

    results, summary = await runner.evaluate_live(qa_pairs, answer_fn)
    assert results[0].verdict == JudgeVerdict.CORRECT
    assert results[1].verdict == JudgeVerdict.ERROR
    assert results[1].judge_error is not None
    assert summary.errors == 1
