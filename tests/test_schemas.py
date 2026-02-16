from vanna_judge.schemas import EvalResult, EvalSummary, JudgeVerdict


def test_eval_summary_counts_errors_separately() -> None:
    results = [
        EvalResult(
            question_id=1,
            question="q1",
            expected_answer="a1",
            system_answer="s1",
            verdict=JudgeVerdict.CORRECT,
            judge_reasoning="ok",
            timing_ms=100,
        ),
        EvalResult(
            question_id=2,
            question="q2",
            expected_answer="a2",
            system_answer="s2",
            verdict=JudgeVerdict.PARTIALLY_CORRECT,
            judge_reasoning="partial",
            timing_ms=200,
        ),
        EvalResult(
            question_id=3,
            question="q3",
            expected_answer="a3",
            system_answer="s3",
            verdict=JudgeVerdict.ERROR,
            judge_reasoning="timeout",
            timing_ms=300,
            judge_error="TimeoutError",
        ),
    ]

    summary = EvalSummary.from_results("TestSystem", results)
    assert summary.total_questions == 3
    assert summary.correct == 1
    assert summary.partially_correct == 1
    assert summary.errors == 1
    assert summary.accuracy_pct == (2 / 3) * 100
    assert summary.accuracy_pct_excluding_errors == 100.0


def test_eval_result_from_dict_backward_compatible() -> None:
    raw = {
        "question_id": 1,
        "question": "q",
        "expected_answer": "e",
        "system_answer": "s",
        "verdict": "incorrect",
        "judge_reasoning": "bad",
        "timing_ms": 11,
    }
    result = EvalResult.from_dict(raw)
    assert result.verdict == JudgeVerdict.INCORRECT
    assert result.judge_error is None
