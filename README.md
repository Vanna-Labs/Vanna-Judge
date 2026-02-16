# Vanna Judge

LLM-as-Judge evaluation framework for RAG systems.

## Documentation

- Architecture: `docs/ARCHITECTURE.md`
- Testing runbook: `docs/TESTING.md`

## Installation

```bash
pip install vanna-judge
```

Install directly from GitHub:

```bash
pip install "git+https://github.com/Vanna-Labs/Vanna-Judge.git"
```

Or install from source:

```bash
git clone https://github.com/Vanna-Labs/vanna-judge.git
cd vanna-judge
pip install -e .
```

## Smoke Test (Published Package)

This verifies that the public GitHub package installs and can execute a real
judge call.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Vanna-Labs/Vanna-Judge.git"
```

Set your API key (or put it in `.env` and load it in your shell):

```bash
export OPENAI_API_KEY=your-key-here
```

Run a minimal live call:

```bash
python - <<'PY'
import asyncio
from vanna_judge import LLMJudge

async def main():
    judge = LLMJudge(model="gpt-5.1", temperature=0.0, timeout_s=20, max_retries=0)
    verdict, reasoning = await judge.judge(
        question="What is 2 + 2?",
        expected_answer="4",
        system_answer="The answer is 4.",
    )
    print("verdict:", verdict.value)
    print("reasoning:", reasoning)

asyncio.run(main())
PY
```

Expected: `verdict: correct`

## Quick Start

```python
import asyncio
from vanna_judge import LLMJudge, JudgeVerdict

async def main():
    judge = LLMJudge(
        model="gpt-5.1",
        temperature=0.0,
        timeout_s=30,
        max_retries=2,
    )

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

The judge classifies answers into 5 categories:

| Verdict | Description |
|---------|-------------|
| `CORRECT` | All key facts present, no errors |
| `PARTIALLY_CORRECT` | Some facts present, nothing wrong |
| `ABSTAINED` | System said "don't know" when answer exists |
| `INCORRECT` | Contradicts expected answer |
| `ERROR` | Judge failed (timeout/API/parse issue) |

Judge failures are tracked separately from model quality verdicts to avoid corrupting accuracy.

## Evaluation Runner (Recommended)

```python
from vanna_judge import LLMJudge, EvaluationRunner, load_qa_dataset

qa_pairs = load_qa_dataset("questions.json")

judge = LLMJudge()
runner = EvaluationRunner(judge=judge, system_name="MyRAG", max_concurrency=8)

async def answer_fn(question: str) -> str:
    return await my_rag.query(question)

results, summary = await runner.evaluate_live(qa_pairs, answer_fn)
```

## CLI for Precomputed Answers

Evaluate pre-generated system answers from JSON:

```bash
vanna-judge \
  --input eval_inputs.json \
  --system-name MyRAG \
  --model gpt-5.1 \
  --concurrency 8
```

Accepted row format keys:
- `question`
- `expected_answer` (fallback: `answer`)
- `system_answer` (fallbacks: `prediction`, `model_answer`)

## Utilities

### Loading Data

```python
from vanna_judge import load_chunks, load_qa_dataset

# Load JSONL chunks (for RAG corpus)
chunks = load_chunks("corpus.jsonl")

# Load Q&A dataset
qa_pairs = load_qa_dataset("questions.json")
```

### Embedding Cache

```python
from vanna_judge import EmbeddingCache, get_openai_embedding_fn

cache = EmbeddingCache("embeddings.pkl")
embed_fn = get_openai_embedding_fn()

embeddings = cache.get_or_compute(texts, embed_fn)
```

### Saving Results

```python
from vanna_judge import save_eval_results, load_eval_results

# Save
filepath = save_eval_results(results, summary, config, "eval_output/")

# Load
results, summary, config = load_eval_results(filepath)
```

## Example

See `examples/eval_my_rag.py` for a complete template showing how to evaluate your own RAG system.

## Preflight Before Going Live

Run the full local regression suite (fast, no API calls):

```bash
pytest -q
```

Run a larger live benchmark that uses your `.env` OpenAI key:

```bash
python3 scripts/run_preflight_live_eval.py \
  --input eval_datasets/preflight_large.json \
  --system-name MyRAG \
  --concurrency 8
```

The preflight script exits non-zero if:
- judge error rate is too high (`--max-error-rate`, default `0.05`)
- verdict match rate against `expected_verdict` labels is too low (`--min-verdict-match`, default `0.75`)

## Prompt Design

The default judge prompt is compact and enforces structured output:
- `verdict` (`correct|partially|abstained|incorrect`)
- `matched_facts`
- `missing_facts`
- `contradictions`
- `reasoning`

You can import the defaults:

```python
from vanna_judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
```

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
