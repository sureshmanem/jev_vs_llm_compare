"""Run Jev and LLM engines over the same labelled items and score them side by side.

    python -m jev_compare                               # both tasks, all engines
    python -m jev_compare --task claims --repeats 3
    python -m jev_compare --engines jev llm-questions --models anthropic/claude-haiku-4.5
    python -m jev_compare --limit 5 --repeats 1         # quick smoke run

Engines per task:
  jev             Jev answers the task's typed questions; task policy decides
  llm-questions   an LLM answers the SAME questions; the SAME policy decides
  llm-direct      an LLM reads the item + the policy in prose and decides

Metrics per engine:
  accuracy        decision == label, over every run
  costly errors   mistakes that cost money or harm (task-defined), per pass
  flip rate       share of items whose decision changed across repeats
  invalid/error   outputs that could not become a decision, failed calls
  cost, latency   USD per decision (billed by OpenRouter where reported), p50/p95 ms
  agree w/ jev    share of items whose majority decision matches Jev's

Every call is saved to results/<timestamp>.json and a summary to results/<timestamp>.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from jev_compare.config import ROOT, EngineResult
from jev_compare.engines import jev as jev_mod
from jev_compare.engines.jev import JevEngine
from jev_compare.engines.llm import LLMEngine, make_http_client
from jev_compare.tasks import TASK_NAMES, Task, load_task

DEFAULT_MODELS = ["anthropic/claude-opus-5", "anthropic/claude-haiku-4.5"]
ENGINE_KINDS = ("jev", "llm-questions", "llm-direct")
LIMITS = {"jev": 16, "llm": 8}  # concurrent calls per engine kind
RESULTS = ROOT / "results"


@dataclass
class Run:
    task: str
    engine: str
    item_id: str
    repeat: int
    label: str
    result: EngineResult

    @property
    def correct(self) -> bool:
        return self.result.decision == self.label


async def _run_task(task: Task, kinds: list[str], models: list[str], repeats: int, limit: int | None) -> list[Run]:
    items = task.items[:limit] if limit else task.items
    gates = {k: asyncio.Semaphore(n) for k, n in LIMITS.items()}
    async with jev_mod.make_client() as jev_client:
        http = make_http_client() if any(k.startswith("llm") for k in kinds) else None
        try:
            engines: list[Any] = []
            if "jev" in kinds:
                engines.append(JevEngine(task, jev_client))
            for model in models:
                for mode in ("questions", "direct"):
                    if f"llm-{mode}" in kinds:
                        engines.append(LLMEngine(task, http, model, mode))

            async def one(engine, item, k) -> Run:
                async with gates[engine.kind]:
                    try:
                        res = await engine.evaluate(item)
                    except Exception as exc:  # noqa: BLE001 - record per-call failures, keep going
                        res = EngineResult("error", f"{type(exc).__name__}: {exc}"[:300], 0.0)
                return Run(task.name, engine.name, item["id"], k, item["label"], res)

            jobs = [one(e, item, k) for e in engines for item in items for k in range(repeats)]
            print(f"[{task.name}] {len(items)} items x {repeats} repeats x {len(engines)} engines = {len(jobs)} calls")
            return list(await asyncio.gather(*jobs))
        finally:
            if http is not None:
                await http.aclose()


# ── scoring ────────────────────────────────────────────────────────────────

def _majority(runs: list[Run]) -> str:
    return Counter(r.result.decision for r in runs).most_common(1)[0][0]


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def summarize(task: Task, runs: list[Run], repeats: int) -> list[dict[str, Any]]:
    by_engine: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_engine[r.engine].append(r)
    jev_major = {}
    if "jev" in by_engine:
        per_item = defaultdict(list)
        for r in by_engine["jev"]:
            per_item[r.item_id].append(r)
        jev_major = {i: _majority(rs) for i, rs in per_item.items()}

    rows = []
    for name, rs in by_engine.items():
        per_item: dict[str, list[Run]] = defaultdict(list)
        for r in rs:
            per_item[r.item_id].append(r)
        ok = [r for r in rs if r.result.decision not in ("error", "invalid")]
        costly = Counter(k for r in rs if (k := task.costly(r.label, r.result.decision)))
        lat = [r.result.latency_ms for r in ok]
        type_hits = [r.result.detail["claim_type_ok"] for r in ok if "claim_type_ok" in r.result.detail]
        agree = [_majority(v) == jev_major[i] for i, v in per_item.items() if i in jev_major]
        rows.append({
            "engine": name,
            "calls": len(rs),
            "accuracy": sum(r.correct for r in rs) / len(rs),
            "costly_per_pass": sum(costly.values()) / repeats,
            "costly_kinds": dict(costly),
            "flip_rate": (sum(len({r.result.decision for r in v}) > 1 for v in per_item.values()) / len(per_item))
            if repeats > 1 else None,
            "invalid": sum(r.result.decision == "invalid" for r in rs),
            "errors": sum(r.result.decision == "error" for r in rs),
            "cost_per_decision": sum(r.result.cost for r in rs) / len(rs),
            "p50_ms": statistics.median(lat) if lat else 0.0,
            "p95_ms": _pct(lat, 0.95),
            "claim_type_accuracy": sum(type_hits) / len(type_hits) if type_hits else None,
            "agree_with_jev": sum(agree) / len(agree) if agree and name != "jev" else None,
        })
    return rows


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "-"
    return {"pct": f"{v:.0%}", "ms": f"{v:,.0f}", "usd": f"${v:.5f}", "num": f"{v:.2g}"}[kind]


COLUMNS = [
    ("engine", "engine", None), ("accuracy", "accuracy", "pct"), ("costly/pass", "costly_per_pass", "num"),
    ("flip rate", "flip_rate", "pct"), ("invalid", "invalid", "num"), ("errors", "errors", "num"),
    ("$/decision", "cost_per_decision", "usd"), ("p50 ms", "p50_ms", "ms"), ("p95 ms", "p95_ms", "ms"),
    ("type acc", "claim_type_accuracy", "pct"), ("agree w/ jev", "agree_with_jev", "pct"),
]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    head = "| " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" if c[2] is None else "---:" for c in COLUMNS) + "|"
    body = [
        "| " + " | ".join(str(r[key]) if kind is None else _fmt(r[key], kind) for _, key, kind in COLUMNS) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def misses(task: Task, runs: list[Run]) -> list[str]:
    """Per engine, the items it got wrong (majority over repeats) or flipped on."""
    out = []
    grouped: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        grouped[(r.engine, r.item_id)].append(r)
    for (engine, item_id), rs in sorted(grouped.items()):
        decisions = Counter(r.result.decision for r in rs)
        if len(decisions) > 1 or _majority(rs) != rs[0].label:
            got = ", ".join(f"{d}x{c}" if c > 1 else d for d, c in decisions.items())
            tag = task.costly(rs[0].label, _majority(rs))
            out.append(f"- `{engine}` {item_id}: expected {rs[0].label}, got {got}" + (f" **[{tag}]**" if tag else ""))
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*TASK_NAMES, "all"], default="all")
    parser.add_argument("--engines", nargs="+", choices=ENGINE_KINDS, default=list(ENGINE_KINDS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="OpenRouter model ids for the LLM engines")
    parser.add_argument("--repeats", type=int, default=3, help="runs per item per engine (flip rate needs >1)")
    parser.add_argument("--limit", type=int, help="only the first N items of each task")
    parser.add_argument("--no-save", action="store_true", help="don't write results/")
    args = parser.parse_args()

    print(f"Jev backend: {jev_mod.backend()} ({jev_mod.model_id()})   LLMs: {', '.join(args.models)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = [f"# Jev vs LLM run {stamp}", "",
              f"Jev: `{jev_mod.model_id()}` via {jev_mod.backend()}. LLMs: {', '.join(f'`{m}`' for m in args.models)}. "
              f"Repeats: {args.repeats}.", ""]
    all_runs: list[Run] = []
    for name in (TASK_NAMES if args.task == "all" else [args.task]):
        task = load_task(name)
        runs = asyncio.run(_run_task(task, args.engines, args.models, args.repeats, args.limit))
        all_runs += runs
        rows = summarize(task, runs, args.repeats)
        table = markdown_table(rows)
        wrong = misses(task, runs)
        first_error = next((r for r in runs if r.result.decision == "error"), None)
        print(f"\n## {name}\n\n{table}\n")
        if first_error:
            print(f"first error [{first_error.engine}]: {first_error.result.reason}\n")
        report += [f"## {name}", "", table, "", "Misses and flips:", "", *(wrong or ["- none"]), ""]

    if not args.no_save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{stamp}.json").write_text(json.dumps(
            {"args": vars(args), "jev_model": jev_mod.model_id(), "runs": [
                {**{k: v for k, v in asdict(r).items() if k != "result"}, **asdict(r.result)} for r in all_runs
            ]}, indent=1, default=str))
        (RESULTS / f"{stamp}.md").write_text("\n".join(report))
        print(f"saved results/{stamp}.json and results/{stamp}.md")


if __name__ == "__main__":
    main()
