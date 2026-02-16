"""Evaluation orchestration utilities for running judge workflows at scale."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from vanna_judge.judge import LLMJudge
from vanna_judge.schemas import EvalResult, EvalSummary, JudgeVerdict


AnswerFn = Callable[[str], str | Awaitable[str]]


class EvaluationRunner:
    """Run evaluations over datasets using an LLMJudge.

    Supports:
    - live evaluation: call a RAG function to produce system answers
    - precomputed evaluation: judge already-generated system answers
    """

    def __init__(
        self,
        judge: LLMJudge,
        system_name: str = "RAGSystem",
        max_concurrency: int = 5,
    ):
        self.judge = judge
        self.system_name = system_name
        self.max_concurrency = max_concurrency

    def _normalize_eval_item(
        self, index: int, item: dict[str, Any]
    ) -> tuple[int, str, str, str]:
        """Normalize common input formats into a standard tuple."""
        raw_id = item.get("id", index + 1)
        try:
            question_id = int(raw_id)
        except (TypeError, ValueError):
            question_id = index + 1
        question = str(item.get("question", "")).strip()
        expected_answer = str(
            item.get("expected_answer", item.get("answer", ""))
        ).strip()
        system_answer = str(
            item.get(
                "system_answer",
                item.get("prediction", item.get("model_answer", "")),
            )
        ).strip()
        return question_id, question, expected_answer, system_answer

    async def _resolve_answer(self, answer_fn: AnswerFn, question: str) -> str:
        """Call an answer function that may be sync or async."""
        result = answer_fn(question)
        if inspect.isawaitable(result):
            return str(await result)
        return str(result)

    async def evaluate_live(
        self,
        qa_pairs: list[dict[str, Any]],
        answer_fn: AnswerFn,
    ) -> tuple[list[EvalResult], EvalSummary]:
        """Evaluate a live RAG function against a QA dataset.

        Expected QA item keys:
        - question
        - answer (or expected_answer)
        - optional id
        """
        if not qa_pairs:
            return [], EvalSummary.from_results(self.system_name, [])

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(index: int, qa: dict[str, Any]) -> EvalResult:
            raw_id = qa.get("id", index + 1)
            try:
                question_id = int(raw_id)
            except (TypeError, ValueError):
                question_id = index + 1
            question = str(qa.get("question", "")).strip()
            expected_answer = str(
                qa.get("expected_answer", qa.get("answer", ""))
            ).strip()

            start = time.perf_counter()
            try:
                async with semaphore:
                    system_answer = await self._resolve_answer(answer_fn, question)
                    verdict, reasoning = await self.judge.judge(
                        question=question,
                        expected_answer=expected_answer,
                        system_answer=system_answer,
                    )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                return EvalResult(
                    question_id=question_id,
                    question=question,
                    expected_answer=expected_answer,
                    system_answer="",
                    verdict=JudgeVerdict.ERROR,
                    judge_reasoning="System under test failed before a verdict was produced.",
                    timing_ms=elapsed_ms,
                    judge_error=f"{type(exc).__name__}: {str(exc)}",
                )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return EvalResult(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                system_answer=system_answer,
                verdict=verdict,
                judge_reasoning=reasoning,
                timing_ms=elapsed_ms,
                judge_error=reasoning if verdict == JudgeVerdict.ERROR else None,
            )

        tasks = [run_one(i, qa) for i, qa in enumerate(qa_pairs)]
        results = list(await asyncio.gather(*tasks))
        summary = EvalSummary.from_results(self.system_name, results)
        return results, summary

    async def evaluate_precomputed(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[EvalResult], EvalSummary]:
        """Evaluate precomputed answers.

        Expected item keys:
        - question
        - expected_answer or answer
        - system_answer (fallback: prediction or model_answer)
        - optional id
        """
        if not items:
            return [], EvalSummary.from_results(self.system_name, [])

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(index: int, item: dict[str, Any]) -> EvalResult:
            question_id, question, expected_answer, system_answer = (
                self._normalize_eval_item(index, item)
            )
            start = time.perf_counter()
            async with semaphore:
                verdict, reasoning = await self.judge.judge(
                    question=question,
                    expected_answer=expected_answer,
                    system_answer=system_answer,
                )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return EvalResult(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                system_answer=system_answer,
                verdict=verdict,
                judge_reasoning=reasoning,
                timing_ms=elapsed_ms,
                judge_error=reasoning if verdict == JudgeVerdict.ERROR else None,
            )

        tasks = [run_one(i, item) for i, item in enumerate(items)]
        results = list(await asyncio.gather(*tasks))
        summary = EvalSummary.from_results(self.system_name, results)
        return results, summary

    def evaluate_precomputed_sync(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[EvalResult], EvalSummary]:
        """Sync wrapper for evaluate_precomputed()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_precomputed(items))

        raise RuntimeError(
            "evaluate_precomputed_sync() cannot run inside an active event loop. "
            "Use: await runner.evaluate_precomputed(items)"
        )
