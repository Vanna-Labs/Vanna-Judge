#!/usr/bin/env python3
"""
Example: Evaluating a custom RAG system with vanna_judge.

This template shows how to:
1. Load a Q&A dataset
2. Query your RAG system
3. Judge answers with LLMJudge + EvaluationRunner
4. Aggregate and save results

Usage:
    pip install vanna-judge
    python eval_my_rag.py --qa-file my_questions.json --limit 5
"""

import argparse
import asyncio

from vanna_judge import (
    EvalResult,
    EvalSummary,
    EvaluationRunner,
    LLMJudge,
    load_qa_dataset,
    print_eval_summary,
    save_eval_results,
)


# =============================================================================
# REPLACE THIS WITH YOUR RAG SYSTEM
# =============================================================================
class MyRAGSystem:
    """Placeholder RAG system - replace with your actual implementation.

    Your RAG system should have an async query method that takes a question
    and returns an answer string.
    """

    def __init__(self):
        # Initialize your RAG system here
        # e.g., load embeddings, connect to vector DB, etc.
        pass

    async def query(self, question: str) -> str:
        """Query your RAG system.

        Args:
            question: The question to answer.

        Returns:
            The answer string from your RAG system.
        """
        # Replace this with your actual RAG logic
        return "This is a placeholder answer. Replace MyRAGSystem with your implementation."


# =============================================================================
# EVALUATION LOGIC (usually no changes needed)
# =============================================================================
async def run_evaluation(
    qa_file: str,
    limit: int | None = None,
    concurrency: int = 5,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run evaluation on all questions."""
    # Load Q&A dataset
    qa_pairs = load_qa_dataset(qa_file)
    if limit:
        qa_pairs = qa_pairs[:limit]

    print(f"Evaluating {len(qa_pairs)} questions...")

    # Initialize systems
    rag = MyRAGSystem()
    judge = LLMJudge()
    runner = EvaluationRunner(
        judge=judge,
        system_name="MyRAG",
        max_concurrency=concurrency,
    )

    async def answer_fn(question: str) -> str:
        return await rag.query(question)

    results, summary = await runner.evaluate_live(qa_pairs, answer_fn)

    for result in results:
        status = "✓" if result.verdict.value == "correct" else result.verdict.value
        print(f"  Q{result.question_id}: {status}")

    return results, summary


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG system using vanna_judge",
    )
    parser.add_argument("--qa-file", type=str, required=True, help="Path to Q&A dataset JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent evaluations (default: 5)")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save results")
    args = parser.parse_args()

    # Run evaluation
    results, summary = await run_evaluation(
        qa_file=args.qa_file,
        limit=args.limit,
        concurrency=args.concurrency,
    )

    # Print summary
    print_eval_summary(summary)

    # Save results
    config = {"qa_file": args.qa_file, "limit": args.limit, "concurrency": args.concurrency}
    filepath = save_eval_results(results, summary, config, args.output_dir)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())
