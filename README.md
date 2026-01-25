# Zomma Judge

LLM-as-Judge evaluation framework for RAG systems.

## Installation

```bash
pip install zomma-judge
```

Or install from source:

```bash
git clone https://github.com/Zomma-Labs/zomma-judge.git
cd zomma-judge
pip install -e .
```

## Quick Start

```python
import asyncio
from zomma_judge import LLMJudge, JudgeVerdict

async def main():
    judge = LLMJudge()

    verdict, reasoning = await judge.judge(
        question="What was inflation in Boston?",
        expected_answer="Inflation was 3.2%",
        system_answer="Boston reported inflation at 3.2% for the quarter.",
    )

    print(f"Verdict: {verdict}")  # JudgeVerdict.CORRECT
    print(f"Reasoning: {reasoning}")

asyncio.run(main())
```

## Verdict Categories

The judge classifies answers into 4 categories:

| Verdict | Description |
|---------|-------------|
| `CORRECT` | All key facts present, no errors |
| `PARTIALLY_CORRECT` | Some facts present, nothing wrong |
| `ABSTAINED` | System said "don't know" when answer exists |
| `INCORRECT` | Contradicts expected answer |

## Batch Evaluation

```python
from zomma_judge import LLMJudge, EvalResult, EvalSummary

judge = LLMJudge()

# Batch judge multiple answers
results = await judge.batch_judge([
    (question1, expected1, system1),
    (question2, expected2, system2),
], max_concurrency=10)
```

## Utilities

### Loading Data

```python
from zomma_judge import load_chunks, load_qa_dataset

# Load JSONL chunks (for RAG corpus)
chunks = load_chunks("corpus.jsonl")

# Load Q&A dataset
qa_pairs = load_qa_dataset("questions.json")
```

### Embedding Cache

```python
from zomma_judge import EmbeddingCache, get_openai_embedding_fn

cache = EmbeddingCache("embeddings.pkl")
embed_fn = get_openai_embedding_fn()

embeddings = cache.get_or_compute(texts, embed_fn)
```

### Saving Results

```python
from zomma_judge import save_eval_results, load_eval_results

# Save
filepath = save_eval_results(results, summary, config, "eval_output/")

# Load
results, summary, config = load_eval_results(filepath)
```

## Example

See `examples/eval_my_rag.py` for a complete template showing how to evaluate your own RAG system.

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

The default judge model is `gpt-5.1`. Override it:

```python
judge = LLMJudge(model="gpt-4o", temperature=0.0)
```

## License

MIT License - see LICENSE file.
