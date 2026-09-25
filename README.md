# Jev vs LLM: the same decisions, side by side

**Question:** for a structured decision made from messy text, how does TypeSafe's
[Jev](https://docs.typesafe.ai) ("System One": typed questions in, calibrated
probabilities out) compare with an LLM on accuracy, costly mistakes, stability,
cost and latency?

This repo is set up to *test* that, not to assume the answer. Every engine gets
the same items, the same policy and the same labels, and every call is saved.

## Engines

| Engine | Reads | Answers | Who decides |
|---|---|---|---|
| `jev` | the task's `state` | Jev's typed questions (`Choice` / `Score` / `Noul`) | task policy (Python) |
| `llm-questions:<model>` | the **same** `state` | the **same** questions, as probability distributions | the **same** task policy |
| `llm-direct:<model>` | the whole item + the policy in prose | the decision itself (JSON enum) | the LLM |

`jev` vs `llm-questions` is the fair model comparison: only who answers the
questions changes. `llm-direct` is the usual "just ask the LLM" design, so
`llm-questions` vs `llm-direct` shows what separating perception (model) from
judgment (code) is worth on its own.

Default LLMs: `anthropic/claude-opus-5` (strongest) and `anthropic/claude-haiku-4.5`
(fast and cheap), both through OpenRouter. Pass any OpenRouter ids with `--models`.

## Tasks

| Task | Items | Decisions | Source |
|---|---|---|---|
| `claims` | 50 hand-labelled insurance claims (FNOL) | 7 routes: fast_track, below_deductible, standard/senior adjuster, coverage_review, siu_review, human_triage | copied from the `jev_poc` project |
| `loan` | 8 hand-labelled applicants, 3 borderline | approve / review / deny | original loan demo |

Each task (`src/jev_compare/tasks/<name>/`) defines its question set, the state
Jev sees, the deterministic policy, the same policy in prose for `llm-direct`,
and which mistakes are **costly** (claims: fast-tracking a claim that needed a
person, missing fraud, sending a serious claim to a junior path; loan: approving
someone who should not be approved, not denying a deny).

Two fixes were made to the claims policy relative to `jev_poc`:
injury/severity/liability are now checked **before** the deductible (a small
property estimate can't hide an injury), and SIU needs at least one narrative
red flag (date facts alone can't send a claim to investigation).

## Metrics

| Metric | Meaning |
|---|---|
| accuracy | decision == label, over every call |
| costly/pass | costly mistakes per full pass over the items |
| flip rate | share of items whose decision changed across repeats (`--repeats` > 1) |
| invalid / errors | outputs that couldn't become a decision / failed calls |
| $/decision | billed cost reported by OpenRouter |
| p50 / p95 ms | call latency |
| type acc | claims only: claim type correct (engines that answer the questions) |
| agree w/ jev | share of items whose majority decision matches Jev's |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # package (editable) + pytest
cp .env.example .env                     # add OPENROUTER_API_KEY: it covers Jev and the LLMs

pytest                                    # offline: policies, answer conversion, Jev via the simulator
jev-compare --limit 3 --repeats 1 --no-save            # quick live smoke run (~$0.10)
jev-compare                                            # both tasks, all engines, 3 repeats (~$4.50)
jev-compare --task claims --engines jev llm-questions --models anthropic/claude-haiku-4.5
```

Jev runs through OpenRouter as `~typesafe/jev-latest` with the `typesafe-sdk`
package (a `TYPESAFE_API_KEY` for api.typesafe.ai wins if set). With no key the
Jev engine falls back to an offline simulator, which is for tests only.

Every run writes `results/<timestamp>.json` (every call: decision, reason, cost,
latency, raw output or Jev answers) and `results/<timestamp>.md` (tables plus each
engine's misses and flips).

## Results

Live run on 2026-09-25 (`results/20260925T001621Z.md`): Jev `typesafe/jev-1.13` via
OpenRouter, 3 repeats per item, 870 calls, no errors or invalid outputs, **$4.50 total**
(Jev's share: $0.006).

### claims (50 items, 7 routes)

| engine | accuracy | costly/pass | flip rate | $/decision | p50 ms | p95 ms | type acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| jev | 97% | 0.33 | 2% | $0.00004 | 199 | 731 | 100% |
| llm-questions: Opus 5 | 95% | 0 | 2% | $0.01644 | 3,284 | 7,488 | 100% |
| llm-direct: Opus 5 | **100%** | 0 | 0% | $0.00779 | 4,446 | 6,952 | - |
| llm-questions: Haiku 4.5 | 94% | 2 | 0% | $0.00258 | 2,034 | 2,535 | 94% |
| llm-direct: Haiku 4.5 | 82% | 0 | 0% | $0.00102 | 1,445 | 1,874 | - |

### loan (8 items)

Every engine scored 100% with no flips. The task is too easy to separate them; only
cost ($0.00002 Jev vs $0.0006 to $0.006 LLM) and latency (293 ms vs 1.3 to 3.3 s) differ.

### What this run shows (and doesn't)

- **Cost and latency: Jev wins by orders of magnitude.** On claims, a Jev decision
  cost ~400x less than Opus 5 answering the same questions (~200x less than Opus
  deciding directly, ~25x less than Haiku) and was ~10 to 20x faster at p50.
- **Accuracy: Jev did not win outright.** Opus 5 deciding directly was perfect on
  claims (100%, 0 flips); Jev was 97%. With only 50 synthetic items, that gap is 1 to
  2 claims and not statistically meaningful either way.
- **Same questions, different model:** Jev (97%) beat both LLMs answering its
  questions (Opus 95%, Haiku 94%). Haiku missed two fraud cases (L-040, L-043) on
  every repeat; Opus was too cautious on three minor claims (sent to an adjuster
  instead of fast-track), a safe error.
- **Code-decides vs LLM-decides depends on the model:** for Haiku, answering the
  questions and letting the policy decide (94%) beat deciding directly (82%). For
  Opus it was the reverse (95% vs 100%).
- **Jev is not bit-for-bit deterministic.** Identical calls wobbled by about ±0.01
  in probability. On L-043 that moved the fraud score from 0.402 to 0.399, across
  the 0.4 threshold tuned on these same claims, so 1 of 3 runs missed fraud (the
  0.33 costly/pass). The fix is a policy one: don't put a threshold where a
  labelled case sits right on it, and use an "uncertain → human" band around cut-offs.
- **The honest headline** for this data: Jev reaches LLM-level accuracy on these
  classification questions at a tiny fraction of the cost and latency. It is not
  shown to be *more* accurate than a frontier LLM. Proving that needs harder,
  real, held-out data.

## Caveats

- **Small, synthetic data.** 50 + 8 hand-written items. The claims labels were
  written alongside the claims question set and policy, so the questions fit the
  data well. Treat this as a method and a first signal, not a benchmark; the next
  step is real labelled history with a held-out split.
- **Thresholds were tuned on the same claims** (in `jev_poc`) using Jev's answers,
  which favours Jev on `claims`. The LLM-questions engine is scored through
  thresholds it never had a say in.
- **The LLM "confidence"** for `llm-questions` uses Jev's shape formula
  `(n·max−1)/(n−1)` on the LLM's stated probabilities. LLM-stated probabilities
  are not calibrated the way Jev's are claimed to be.
- **`temperature=0` via OpenRouter** doesn't guarantee determinism; some flips
  may come from provider routing rather than the model.
- The claims `other` second-level call (which only picks a human desk, not the
  route) is not part of the comparison.

## Layout

```
src/jev_compare/
  compare.py          # harness: runs engines x items x repeats, scores, saves results/
  config.py           # .env loading, EngineResult
  engines/jev.py      # Jev via typesafe-sdk (OpenRouter / TypeSafe / offline)
  engines/llm.py      # llm-questions and llm-direct via OpenRouter chat + JSON schema
  simulator.py        # offline Jev stand-in (from jev_poc), used by tests
  tasks/claims/       # questions, typed response, routing policy (from jev_poc, fixed)
  tasks/loan/         # questions, policy, applicants
data/claims/          # labelled_claims.json; jev_cache.json = real Jev answers used by a regression test
tests/                # offline tests, no keys needed
results/              # saved runs
```
