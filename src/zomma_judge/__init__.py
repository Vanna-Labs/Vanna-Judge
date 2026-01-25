"""
Zomma Judge - LLM-as-Judge evaluation framework for RAG systems.

Usage:
    from zomma_judge import LLMJudge, JudgeVerdict, EvalResult, EvalSummary

    judge = LLMJudge()
    verdict, reasoning = await judge.judge(
        question="What is X?",
        expected_answer="X is Y",
        system_answer="X is Y with additional context",
    )
"""

from zomma_judge.schemas import EvalResult, EvalSummary, JudgeVerdict
from zomma_judge.judge import LLMJudge
from zomma_judge.utils import (
    EMBEDDING_DIMENSION,
    EmbeddingCache,
    batch_cosine_similarity,
    compute_cosine_similarity,
    format_chunk_for_display,
    get_openai_embedding_fn,
    load_chunks,
    load_eval_results,
    load_qa_dataset,
    print_eval_summary,
    save_eval_results,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Schemas
    "EvalResult",
    "EvalSummary",
    "JudgeVerdict",
    # Judge
    "LLMJudge",
    # Data loading
    "load_chunks",
    "load_qa_dataset",
    # Results I/O
    "save_eval_results",
    "load_eval_results",
    # Embeddings
    "EmbeddingCache",
    "get_openai_embedding_fn",
    "EMBEDDING_DIMENSION",
    # Similarity
    "compute_cosine_similarity",
    "batch_cosine_similarity",
    # Display
    "format_chunk_for_display",
    "print_eval_summary",
]
