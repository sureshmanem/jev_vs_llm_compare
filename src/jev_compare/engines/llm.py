"""LLM engines, called through OpenRouter's OpenAI-compatible chat endpoint.

Two ways to put an LLM on the same task as Jev:

* ``questions``: the LLM answers Jev's exact question set (same state, same
  questions) as probability distributions. The answers are converted to Jev's
  wire format and decided by the *same* task policy. Only the model that
  answers the questions changes, so this isolates Jev vs LLM.
* ``direct``: the LLM reads the whole item and the policy written in prose,
  and returns the decision itself. This is the usual "just ask the LLM" design.

Both request a strict JSON schema, so output is parseable; values that still
don't fit (an unknown decision, missing keys) are counted as "invalid".
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx2
from langsmith import traceable
from typesafe_sdk import SystemOneResponse

from jev_compare.config import EngineResult, require_env
from jev_compare.tasks import Task

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODES = ("questions", "direct")

QUESTIONS_SYSTEM = """You are a classifier. You receive a `state` and a set of typed questions \
about it. Answer every question from the state alone.

- choice: give a probability for every option; they must sum to 1.
- score: the options are ordered levels "0".."n-1"; give a probability for every level; they must sum to 1.
- noul: give the probability (0 to 1) that the statement is true.

Use the full range: be near 1 or 0 when the text is clear, and spread probability when it is \
genuinely ambiguous. Return only the JSON object required by the schema."""


def make_http_client() -> httpx2.AsyncClient:
    key = require_env("OPENROUTER_API_KEY")
    return httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {key}", "X-Title": "jev-vs-llm-compare"},
        timeout=httpx2.Timeout(180.0),
    )


# ── questions mode: same questions as Jev, answers in Jev's wire format ──────

def _confidence(probs: list[float]) -> float:
    # Jev's shape-based confidence: 1 when all mass is on one option, 0 when spread evenly.
    n = len(probs)
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))


def _normalise(values: dict[str, Any]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in values.items()}
    total = sum(clipped.values()) or 1.0
    return {k: v / total for k, v in clipped.items()}


def questions_payload(task: Task) -> dict[str, Any]:
    return {k: q.model_dump(mode="json", exclude_none=True) for k, q in task.questions().items()}


def questions_schema(questions: dict[str, Any]) -> dict[str, Any]:
    def probs(keys: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {k: {"type": "number"} for k in keys},
            "required": keys,
            "additionalProperties": False,
        }

    props: dict[str, Any] = {}
    for key, q in questions.items():
        if q["type"] == "choice":
            props[key] = probs(list(q["criteria"]))
        elif q["type"] == "score":
            props[key] = probs([str(i) for i in range(len(q["criteria"]))])
        else:
            props[key] = {"type": "number"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def to_jev_answers(questions: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Convert the LLM's distributions into Jev's /v1/systemone answer format."""
    answers: dict[str, Any] = {}
    for key, q in questions.items():
        if q["type"] == "choice":
            p = _normalise(out[key])
            answers[key] = {
                "type": "choice",
                "choice": max(p, key=p.get),
                "probabilities": p,
                "confidence": _confidence(list(p.values())),
            }
        elif q["type"] == "score":
            p = _normalise(out[key])
            answers[key] = {
                "type": "score",
                "score": sum(int(level) * v for level, v in p.items()),
                "legend": {str(i): text for i, text in enumerate(q["criteria"])},
                "probabilities": p,
                "confidence": _confidence(list(p.values())),
            }
        else:
            answers[key] = {"type": "noul", "noul": min(1.0, max(0.0, float(out[key])))}
    return answers


def decode(model: type[SystemOneResponse], payload: dict[str, Any]) -> SystemOneResponse:
    """Build a typed response through the SDK's own decoder (as jev_poc's
    response_from_json does), so validation matches a live Jev call."""
    payload = {"usage": {"input_tokens": 0, "output_tokens": 0}, **payload}
    request = httpx2.Request("POST", "https://llm.invalid/v1/systemone")
    return model._decode(httpx2.Response(200, json=payload, request=request))


def decision_schema(decisions: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": list(decisions)}, "reason": {"type": "string"}},
        "required": ["decision", "reason"],
        "additionalProperties": False,
    }


def loads(raw: str) -> Any:
    """Parse model JSON, tolerating ```json fences despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return json.loads(raw)


class LLMEngine:
    kind = "llm"

    def __init__(self, task: Task, http: httpx2.AsyncClient, model: str, mode: str):
        assert mode in MODES
        self.task, self.http, self.model, self.mode = task, http, model, mode
        self.name = f"llm-{mode}:{model.split('/')[-1]}"

    async def _chat(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
            "usage": {"include": True},
        }
        start = time.perf_counter()
        resp = await self.http.post(CHAT_URL, json=body)
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        return data, data["choices"][0]["message"].get("content") or "", latency_ms

    @traceable(run_type="llm", name="llm_engine")
    async def evaluate(self, item: dict[str, Any]) -> EngineResult:
        if self.mode == "questions":
            questions = questions_payload(self.task)
            user = json.dumps({"state": self.task.state(item), "questions": questions}, indent=1)
            schema = questions_schema(questions)
            system = QUESTIONS_SYSTEM
        else:
            user = json.dumps(self.task.direct_record(item), indent=1)
            schema = decision_schema(self.task.decisions)
            system = self.task.direct_policy

        data, raw, latency_ms = await self._chat(system, user, schema)
        usage = data.get("usage") or {}
        base = {
            "model": data.get("model", self.model),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "raw": raw,
        }
        cost = float(usage.get("cost") or 0.0)
        try:
            out = loads(raw)
            if self.mode == "questions":
                answers = to_jev_answers(questions, out)
                response = decode(self.task.response_model, {"model": self.model, "answers": answers})
                decision, detail = self.task.decide(item, response)
                reason = "; ".join(detail.get("reasons", [])) or str(detail.get("factors", ""))
                return EngineResult(decision, reason, latency_ms, cost, {**base, **detail, "answers": answers})
            decision = out["decision"]
            if decision not in self.task.decisions:
                raise ValueError(decision)
            return EngineResult(decision, str(out.get("reason", "")), latency_ms, cost, base)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return EngineResult("invalid", f"{type(exc).__name__}: {exc}", latency_ms, cost, base)
