"""Run both engines over the applicants and score them side by side.

Metrics:
  - accuracy    : decision == ground-truth label, across all runs
  - determinism : for repeated runs of the same applicant, do we always get the
                  same decision? (flip rate on borderline cases is the headline)
  - latency     : mean ms per call
  - agreement   : how often the two engines land on the same decision

Borderline applicants are run REPEATS times to expose LLM instability; clear cases
run once. Every engine call is traced in LangSmith.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass

from jev_compare.applicants import APPLICANTS, Applicant
from jev_compare.config import EngineResult
from jev_compare.engines import jev, llm

REPEATS = 5  # runs per borderline applicant
ENGINES = {"LLM": llm.evaluate, "Jev": jev.evaluate}


@dataclass
class Run:
    applicant: Applicant
    engine: str
    result: EngineResult
    correct: bool


def _run_engine(name: str, fn, app: Applicant, runs: int) -> list[Run]:
    out = []
    for _ in range(runs):
        try:
            res = fn(app.text)
        except Exception as exc:  # noqa: BLE001 - surface any engine/setup error per-call
            res = EngineResult(decision="error", reason=f"{type(exc).__name__}: {exc}",
                               latency_ms=0.0)
        out.append(Run(app, name, res, res.decision == app.label))
    return out


def run_all(repeats: int = REPEATS, engines: list[str] | None = None) -> list[Run]:
    engines = engines or list(ENGINES)
    runs: list[Run] = []
    for app in APPLICANTS:
        n = repeats if app.borderline else 1
        for name in engines:
            runs += _run_engine(name, ENGINES[name], app, n)
    return runs


def _by_engine(runs: list[Run]) -> dict[str, list[Run]]:
    d: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        d[r.engine].append(r)
    return d


def _flip_rate(runs: list[Run]) -> float:
    """Fraction of applicants (with >1 run) whose decision was not unanimous."""
    per_app: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for r in runs:
        per_app[r.applicant.id].add(r.result.decision)
        counts[r.applicant.id] += 1
    repeated = [aid for aid, c in counts.items() if c > 1]
    if not repeated:
        return 0.0
    flipped = sum(1 for aid in repeated if len(per_app[aid]) > 1)
    return flipped / len(repeated)


def summarize(runs: list[Run]) -> None:
    print("\n" + "=" * 78)
    print("PER-APPLICANT DECISIONS")
    print("=" * 78)
    header = f"{'ID':<4}{'truth':<9}{'border':<8}{'LLM':<26}{'Jev':<26}"
    print(header)
    print("-" * 78)
    for app in APPLICANTS:
        llm = [r for r in runs if r.applicant.id == app.id and r.engine == "LLM"]
        jev = [r for r in runs if r.applicant.id == app.id and r.engine == "Jev"]
        print(f"{app.id:<4}{app.label:<9}{'yes' if app.borderline else '-':<8}"
              f"{_cell(llm):<26}{_cell(jev):<26}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    by = _by_engine(runs)
    for name in ENGINES:
        rs = by[name]
        if not rs:
            continue
        acc = sum(r.correct for r in rs) / len(rs) if rs else 0.0
        lats = [r.result.latency_ms for r in rs if r.result.latency_ms]
        lat = statistics.mean(lats) if lats else 0.0
        flips = _flip_rate(rs)
        print(f"{name:<5} accuracy={acc:6.1%}   flip_rate={flips:6.1%}   "
              f"mean_latency={lat:8.1f} ms   n={len(rs)}")

    if all(by[name] for name in ENGINES):
        print(f"\nagreement between engines = {_agreement(runs):.1%}")
    errors = [r for r in runs if r.result.decision == "error"]
    if errors:
        print(f"\n{len(errors)} call(s) errored; first: [{errors[0].engine}] "
              f"{errors[0].result.reason}")
    print("=" * 78)


def _cell(runs: list[Run]) -> str:
    """Compact 'decision(s) xN' cell, marking disagreement with the truth."""
    if not runs:
        return "-"
    counts: dict[str, int] = defaultdict(int)
    for r in runs:
        counts[r.result.decision] += 1
    parts = [f"{d}x{c}" if c > 1 else d for d, c in counts.items()]
    mark = "" if all(r.correct for r in runs) else "  <-off"
    return ",".join(parts) + mark


def _agreement(runs: list[Run]) -> float:
    """Per applicant, compare each engine's majority decision."""
    def majority(rs: list[Run]) -> str:
        c: dict[str, int] = defaultdict(int)
        for r in rs:
            c[r.result.decision] += 1
        return max(c, key=c.get) if c else "-"

    agree = total = 0
    for app in APPLICANTS:
        llm = majority([r for r in runs if r.applicant.id == app.id and r.engine == "LLM"])
        jev = majority([r for r in runs if r.applicant.id == app.id and r.engine == "Jev"])
        if "error" in (llm, jev):
            continue  # an errored call says nothing about agreement
        total += 1
        agree += int(llm == jev)
    return agree / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM vs Jev on loan eligibility.")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="runs per borderline applicant (default 5)")
    parser.add_argument("--engines", nargs="+", choices=list(ENGINES), default=list(ENGINES),
                        help="which engines to run (default: both)")
    args = parser.parse_args()

    print(f"Running {' vs '.join(args.engines)} over {len(APPLICANTS)} applicants "
          f"(borderline cases x{args.repeats})...")
    runs = run_all(args.repeats, args.engines)
    summarize(runs)


if __name__ == "__main__":
    main()
