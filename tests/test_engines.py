"""Offline engine tests: no network, no keys."""

import asyncio
import json

import pytest

from jev_compare.config import ROOT
from jev_compare.engines import jev as jev_mod
from jev_compare.engines.llm import decode, loads, questions_payload, questions_schema, to_jev_answers
from jev_compare.tasks import TASK_NAMES, load_task
from jev_compare.tasks.claims.response import response_from_json


def test_cached_live_jev_answers_still_route_49_of_50():
    # Real Jev answers (jev-1.13, cached from jev_poc) through the fixed claims policy.
    task = load_task("claims")
    cache = json.loads((ROOT / "data" / "claims" / "jev_cache.json").read_text())
    hits = sum(task.decide(c, response_from_json(cache[c["id"]]))[0] == c["label"] for c in task.items)
    assert hits == 49


@pytest.mark.parametrize("name", TASK_NAMES)
def test_llm_answers_convert_to_jev_format_and_decide(name):
    task = load_task(name)
    questions = questions_payload(task)
    schema = questions_schema(questions)
    assert set(schema["required"]) == set(questions)
    # A flat "LLM" answer: uniform distributions, p=0.5 nouls.
    out = {}
    for key, q in questions.items():
        if q["type"] == "choice":
            out[key] = {o: 1 for o in q["criteria"]}
        elif q["type"] == "score":
            out[key] = {str(i): 1 for i in range(len(q["criteria"]))}
        else:
            out[key] = 0.5
    response = decode(task.response_model, {"model": "test", "answers": to_jev_answers(questions, out)})
    decision, _ = task.decide(task.items[0], response)
    assert decision in task.decisions


def test_confidence_matches_jev_shape():
    answers = to_jev_answers(
        {"q": {"type": "choice", "criteria": {"a": "", "b": ""}}}, {"q": {"a": 0.9, "b": 0.1}}
    )
    assert answers["q"]["choice"] == "a"
    assert answers["q"]["confidence"] == pytest.approx(0.8)


def test_loads_tolerates_fences():
    assert loads('```json\n{"decision": "deny"}\n```') == {"decision": "deny"}


@pytest.mark.parametrize("name", TASK_NAMES)
def test_jev_engine_runs_offline_through_the_real_sdk(name, monkeypatch):
    for var in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "TYPESAFE_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    task = load_task(name)

    async def run():
        async with jev_mod.make_client() as client:
            return await jev_mod.JevEngine(task, client).evaluate(task.items[0])

    result = asyncio.run(run())
    assert jev_mod.backend() == "offline"
    assert result.decision in task.decisions
    assert result.cost >= 0 and result.latency_ms > 0
