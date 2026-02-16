import asyncio
from types import SimpleNamespace

import pytest

from vanna_judge.judge import LLMJudge
from vanna_judge.schemas import JudgeVerdict


class FakeChain:
    def __init__(self, result: object | None = None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    async def ainvoke(self, payload: dict[str, str]) -> object:
        if self._exc is not None:
            raise self._exc
        return self._result


def make_judge(chain: object, max_retries: int = 0) -> LLMJudge:
    judge = LLMJudge.__new__(LLMJudge)
    judge.model = "test-model"
    judge.timeout_s = 2.0
    judge.max_retries = max_retries
    judge.retry_backoff_s = 0.0
    judge.chain = chain
    return judge


@pytest.mark.asyncio
async def test_judge_fast_path_abstained() -> None:
    judge = make_judge(chain=FakeChain(result=None))

    verdict, reasoning = await judge.judge("q", "e", "I don't have that information.")
    assert verdict == JudgeVerdict.ABSTAINED
    assert "abstained" in reasoning.lower()


@pytest.mark.asyncio
async def test_judge_unknown_verdict_returns_error() -> None:
    response = SimpleNamespace(
        verdict="unexpected",
        matched_facts=[],
        missing_facts=[],
        contradictions=[],
        reasoning="bad format",
    )
    judge = make_judge(chain=FakeChain(result=response))

    verdict, reasoning = await judge.judge("q", "e", "system answer")
    assert verdict == JudgeVerdict.ERROR
    assert "unsupported verdict" in reasoning.lower()


@pytest.mark.asyncio
async def test_judge_retries_transient_error() -> None:
    class FlakyChain:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, payload: dict[str, str]) -> object:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("rate limit")
            return SimpleNamespace(
                verdict="correct",
                matched_facts=["fact"],
                missing_facts=[],
                contradictions=[],
                reasoning="all good",
            )

    chain = FlakyChain()
    judge = make_judge(chain=chain, max_retries=1)
    verdict, reasoning = await judge.judge("q", "e", "system answer")
    assert chain.calls == 2
    assert verdict == JudgeVerdict.CORRECT
    assert "matched" in reasoning.lower()


@pytest.mark.asyncio
async def test_judge_sync_raises_inside_active_loop() -> None:
    judge = make_judge(chain=FakeChain())
    with pytest.raises(RuntimeError, match="active event loop"):
        judge.judge_sync("q", "e", "s")


@pytest.mark.asyncio
async def test_batch_judge_preserves_input_order() -> None:
    judge = make_judge(chain=FakeChain())

    async def fake_judge(question: str, expected: str, system: str):
        delay = 0.03 if question == "slow" else 0.0
        await asyncio.sleep(delay)
        return JudgeVerdict.CORRECT, question

    judge.judge = fake_judge  # type: ignore[method-assign]

    outputs = await judge.batch_judge(
        [("slow", "e1", "s1"), ("fast", "e2", "s2")], max_concurrency=2
    )
    assert outputs[0][1] == "slow"
    assert outputs[1][1] == "fast"
