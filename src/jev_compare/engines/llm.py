"""LLM engine: Claude Opus 5 (via OpenRouter) reads the applicant text and returns a verdict.

Given the SAME policy the Jev engine uses (fair fight), the model must output a
strict JSON object. We still have to parse/validate the prose-y result, which is
part of the point: the output isn't natively structured.

OpenRouter exposes an OpenAI-compatible API, so we use the `openai` SDK with a
different base_url. Swap models with OPENROUTER_MODEL (e.g. openai/gpt-4o-mini).

Traced in LangSmith via the @traceable decorator.
"""
from __future__ import annotations

import json
import os
import re
import time

from langsmith import traceable
from openai import OpenAI

from jev_compare.config import EngineResult, require_env
from jev_compare.policy import POLICY_TEXT, APPROVE, REVIEW, DENY

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-opus-5")
_VALID = {APPROVE, REVIEW, DENY}

_SYSTEM = f"""You are a loan eligibility officer. Apply the following policy exactly.

{POLICY_TEXT}

Respond with ONLY a JSON object, no markdown, of the form:
{{"decision": "approve|review|deny", "reason": "<one short sentence>"}}
"""

_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI(
            api_key=require_env("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            default_headers={"X-Title": "jev-vs-llm-compare"},
        )
    return _client_singleton


@traceable(run_type="llm", name="llm_engine")
def evaluate(text: str) -> EngineResult:
    """Run the LLM verdict on one applicant's free text."""
    client = _client()
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,  # even at 0, borderline cases tend to drift across runs
        max_tokens=300,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    raw = resp.choices[0].message.content or "{}"

    # The "parse the prose" tax: coerce and validate.
    decision, reason = parse(raw)
    return EngineResult(
        decision=decision,
        reason=reason,
        latency_ms=latency_ms,
        detail={"model": resp.model or _MODEL, "raw": raw},
    )


def parse(raw: str) -> tuple[str, str]:
    # Models sometimes wrap JSON in ```json fences despite instructions.
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        obj = json.loads(cleaned)
        decision = str(obj.get("decision", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return REVIEW, f"unparseable LLM output: {raw[:120]!r}"
    if decision not in _VALID:
        return REVIEW, f"invalid decision {decision!r} from LLM; defaulted to review"
    return decision, reason or "(no reason given)"
