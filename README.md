# Jev vs LLM — Loan Eligibility Comparison

A small, honest side-by-side demo of two ways to make the **same structured decision**
from the **same messy natural-language input**:

- **LLM engine** — OpenAI chat model asked to read an applicant's free-text summary and
  return a verdict (`approve` / `review` / `deny`) with a reason. Flexible, but the output
  is prose you must parse, it's uncalibrated, and it drifts on borderline cases.
- **Jev engine** — [Jev](https://www.langchain.com/blog/building-a-harness-with-jev) (TypeSafe AI's
  "System One" classifier) answers a few **typed questions** (`noul` / `score`) about the same
  text and returns **calibrated probabilities**. A tiny, auditable Python policy turns those
  into the final decision. Reportedly ~200x faster and ~400x cheaper than an LLM on classification.

Both engines are traced in **LangSmith** so you can inspect every call.

## The point

| | OpenAI LLM | Jev System One |
|---|---|---|
| Output | prose to parse | typed values + probabilities |
| Directly usable by software | needs coercion | yes |
| Calibrated confidence | no | yes (per question) |
| Determinism on borderline cases | flips | stable |
| Latency / cost | high | ~200x / ~400x better |

The comparison is deliberately a **fair fight**: the LLM is given the *exact same policy*
in its prompt. The story isn't "the LLM doesn't know the rules" — it's "even told the rules,
the free-text engine is slower, uncalibrated, and less stable than typed structured decisions."

## Layout

```
applicants.py   # hand-labeled test cases (clear approve/deny + borderline)
policy.py       # the eligibility rules (shared ground truth + Jev decision policy)
llm_engine.py   # OpenAI verdict, LangSmith-traced
jev_engine.py   # Jev typed questions + policy, LangSmith-traced
compare.py      # runs both, repeats borderline cases, scores accuracy/determinism/latency
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
python compare.py
```
