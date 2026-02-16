"""
MODULE: LLM Judge
DESCRIPTION: LLM-based judge for evaluating RAG system answers against expected answers.
             Distinguishes between 4 verdict categories: correct, partially correct,
             abstained (said "don't know" when answer exists), and incorrect.
             Runtime/model failures are returned as ERROR.
"""

import asyncio
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from vanna_judge.schemas import JudgeVerdict


class JudgeOutput(BaseModel):
    """Structured output from the LLM judge."""

    verdict: Literal["correct", "partially", "abstained", "incorrect"] = Field(
        description="Final verdict."
    )
    matched_facts: list[str] = Field(
        default_factory=list,
        description="Expected facts that are present and correct in the system answer.",
    )
    missing_facts: list[str] = Field(
        default_factory=list,
        description="Expected facts that are absent from the system answer.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Any direct factual contradictions versus the expected answer.",
    )
    reasoning: str = Field(
        description="Brief rationale (1-3 sentences)."
    )


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for RAG system outputs.

Evaluate only by comparing:
1) expected answer (ground truth)
2) system answer (candidate)

Use exactly one verdict:
- "correct": all key expected facts are present, no contradictions.
- "partially": some expected facts are present, no contradictions.
- "abstained": candidate does not answer (e.g., "I don't know", "not in context", empty).
- "incorrect": any key factual contradiction exists.

Rules:
- Missing facts without contradiction => "partially", not "incorrect".
- Extra details are neutral unless they contradict expected facts.
- Prefer semantic matching over exact wording.
- If contradiction exists, verdict must be "incorrect".

Return JSON matching the schema with:
- verdict
- matched_facts
- missing_facts
- contradictions
- reasoning (1-3 sentences, concise)
"""

JUDGE_USER_PROMPT = """## Question
{question}

## Expected Answer (Ground Truth)
{expected_answer}

## System Answer (To Evaluate)
{system_answer}

Evaluate the system answer and provide your verdict."""


class LLMJudge:
    """LLM-based judge for evaluating RAG system answers.

    Uses structured output to ensure consistent verdict format.

    Usage:
        judge = LLMJudge(model="gpt-5.1")
        verdict, reasoning = await judge.judge(question, expected, system_answer)
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ):
        """Initialize the LLM judge.

        Args:
            model: OpenAI model to use for judging. Defaults to gpt-5.1.
            temperature: LLM temperature. Defaults to 0.0 for deterministic output.
            timeout_s: Timeout per model call in seconds.
            max_retries: Retry attempts for transient failures.
            retry_backoff_s: Initial backoff for retries (exponential).
        """
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(JudgeOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", JUDGE_USER_PROMPT),
        ])

        self.chain = self.prompt | self.structured_llm

    def _parse_verdict(self, verdict_str: str) -> JudgeVerdict | None:
        """Parse a verdict string into a JudgeVerdict enum.

        Args:
            verdict_str: The verdict string from the LLM output.

        Returns:
            The corresponding JudgeVerdict enum value.
        """
        verdict_str = verdict_str.lower().strip()

        # Map possible outputs to enum values
        verdict_map = {
            "correct": JudgeVerdict.CORRECT,
            "partially": JudgeVerdict.PARTIALLY_CORRECT,
            "partially_correct": JudgeVerdict.PARTIALLY_CORRECT,
            "partial": JudgeVerdict.PARTIALLY_CORRECT,
            "abstained": JudgeVerdict.ABSTAINED,
            "abstain": JudgeVerdict.ABSTAINED,
            "incorrect": JudgeVerdict.INCORRECT,
            "wrong": JudgeVerdict.INCORRECT,
        }

        return verdict_map.get(verdict_str)

    def _is_empty_or_abstained(self, answer: str) -> bool:
        """Check if an answer is empty or clearly an abstention.

        Args:
            answer: The system answer to check.

        Returns:
            True if the answer is empty or a clear abstention.
        """
        if not answer or not answer.strip():
            return True

        answer_lower = answer.lower().strip()

        # Check for common abstention patterns
        abstention_phrases = [
            "i don't have",
            "i do not have",
            "i cannot find",
            "i can't find",
            "unable to find",
            "no information available",
            "no relevant information",
            "the context doesn't contain",
            "the context does not contain",
            "not mentioned in",
            "no data available",
            "i'm unable to",
            "i am unable to",
        ]

        # Short answers that are just abstentions
        if len(answer_lower) < 100:
            for phrase in abstention_phrases:
                if phrase in answer_lower:
                    return True

        return False

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine whether an error is likely transient and safe to retry."""
        if isinstance(error, asyncio.TimeoutError):
            return True

        message = str(error).lower()
        retry_signals = [
            "rate limit",
            "429",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "connection",
            "overloaded",
            "try again",
        ]
        return any(signal in message for signal in retry_signals)

    async def _invoke_with_retry(self, payload: dict[str, str]) -> JudgeOutput:
        """Call the model with timeout and retry behavior."""
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.chain.ainvoke(payload), timeout=self.timeout_s
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable_error(exc):
                    raise
                backoff = self.retry_backoff_s * (2**attempt)
                await asyncio.sleep(backoff)

        # Should never reach here due to return/raise above.
        raise RuntimeError("Retry loop exited unexpectedly.")

    def _build_reasoning(self, result: JudgeOutput) -> str:
        """Create a concise, audit-friendly reasoning string."""
        details: list[str] = []
        if result.matched_facts:
            details.append(f"Matched: {', '.join(result.matched_facts)}")
        if result.missing_facts:
            details.append(f"Missing: {', '.join(result.missing_facts)}")
        if result.contradictions:
            details.append(f"Contradictions: {', '.join(result.contradictions)}")

        if result.reasoning.strip():
            details.append(f"Rationale: {result.reasoning.strip()}")

        return " | ".join(details) if details else "No detailed reasoning returned."

    async def judge(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        """Judge a system answer against the expected answer.

        Args:
            question: The question that was asked.
            expected_answer: The expected (ground truth) answer.
            system_answer: The system's answer to evaluate.

        Returns:
            Tuple of (verdict, reasoning).
        """
        # Fast-path obvious abstentions without a model call.
        if self._is_empty_or_abstained(system_answer):
            return (
                JudgeVerdict.ABSTAINED,
                "System answer appears empty or abstained from answering.",
            )

        try:
            result = await self._invoke_with_retry(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "system_answer": system_answer,
                }
            )

            verdict = self._parse_verdict(result.verdict)
            if verdict is None:
                return (
                    JudgeVerdict.ERROR,
                    f"Judge returned unsupported verdict '{result.verdict}'.",
                )

            return (verdict, self._build_reasoning(result))

        except Exception as e:
            # Keep model/runtime failures separate from quality verdicts.
            return (
                JudgeVerdict.ERROR,
                f"Error during judgment: {type(e).__name__}: {str(e)}",
            )

    async def batch_judge(
        self,
        judgments: list[tuple[str, str, str]],
        max_concurrency: int = 10,
    ) -> list[tuple[JudgeVerdict, str]]:
        """Judge multiple answers concurrently.

        Args:
            judgments: List of (question, expected_answer, system_answer) tuples.
            max_concurrency: Maximum number of concurrent LLM calls. Defaults to 10.

        Returns:
            List of (verdict, reasoning) tuples in the same order as input.
        """
        if not judgments:
            return []

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def judge_with_semaphore(
            question: str, expected: str, system: str
        ) -> tuple[JudgeVerdict, str]:
            async with semaphore:
                return await self.judge(question, expected, system)

        # Create tasks for all judgments
        tasks = [
            judge_with_semaphore(q, e, s) for q, e, s in judgments
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        return list(results)

    def judge_sync(
        self,
        question: str,
        expected_answer: str,
        system_answer: str,
    ) -> tuple[JudgeVerdict, str]:
        """Synchronous wrapper for judge().

        Args:
            question: The question that was asked.
            expected_answer: The expected (ground truth) answer.
            system_answer: The system's answer to evaluate.

        Returns:
            Tuple of (verdict, reasoning).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.judge(question, expected_answer, system_answer))

        raise RuntimeError(
            "judge_sync() cannot run inside an active event loop. "
            "Use: await judge.judge(question, expected_answer, system_answer)"
        )
