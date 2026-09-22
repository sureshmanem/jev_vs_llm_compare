"""LLM engine: OpenAI reads the applicant text and returns a verdict.

Given the SAME policy the Jev engine uses (fair fight), the model must output a
strict JSON object. We still have to parse/validate the prose-y result, which is
part of the point: the output isn't natively structured.

Traced in LangSmith via the @traceable decorator.
"""
from __future__ import annotations

import json
import os
import time

from langsmith import traceable
from openai import OpenAI

from common import EngineResult, require_env
from policy import POLICY_TEXT, APPROVE, REVIEW, DENY

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_VALID = {APPROVE, REVIEW, DENY}

_SYSTEM = f"""You are a loan eligibility officer. Apply the following policy exactly.

{POLICY_TEXT}

Respond with ONLY a JSON object, no markdown, of the form:
{{"decision": "approve|review|deny", "reason": "<one short sentence>"}}
"""


def _client() -> OpenAI:
    return OpenAI(api_key=require_env("OPENAI_API_KEY"))


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
    )
    latency_ms = (time.perf_counter() - start) * 1000
    raw = resp.choices[0].message.content or "{}"

    # The "parse the prose" tax: coerce and validate.
    decision, reason = _parse(raw)
    return EngineResult(
        decision=decision,
        reason=reason,
        latency_ms=latency_ms,
        detail={"model": _MODEL, "raw": raw},
    )


def _parse(raw: str) -> tuple[str, str]:
    try:
        obj = json.loads(raw)
        decision = str(obj.get("decision", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return "review", f"unparseable LLM output: {raw[:120]!r}"
    if decision not in _VALID:
        return "review", f"invalid decision {decision!r} from LLM; defaulted to review"
    return decision, reason or "(no reason given)"
