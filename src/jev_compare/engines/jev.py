"""Jev engine: TypeSafe's System One answers the task's typed questions in one
pass, then the task's deterministic policy makes the decision.

Backend selection is copied from jev_poc (claims_triage/pipeline.py): a direct
TYPESAFE_API_KEY wins, then OPENROUTER_API_KEY (OpenRouter serves the TypeSafe
API shape at ~typesafe/jev-latest), otherwise an offline simulator.
"""
from __future__ import annotations

import time
from typing import Any

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, SystemOneResponse

from jev_compare import simulator
from jev_compare.config import EngineResult, env
from jev_compare.tasks import Task

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_MODEL = "~typesafe/jev-latest"
PRICE_PER_MTOK = 0.042  # USD per million input tokens; output is free


def backend() -> str:
    if env("TYPESAFE_API_KEY"):
        return "typesafe"
    if env("OPENROUTER_API_KEY"):
        return "openrouter"
    return "offline"


def model_id() -> str:
    """TYPESAFE_DEFAULT_MODEL pins a version; otherwise the backend's latest alias."""
    default = OPENROUTER_MODEL if backend() == "openrouter" else "jev-latest"
    return env("TYPESAFE_DEFAULT_MODEL") or default


def make_client() -> AsyncTypeSafeClient:
    match backend():
        case "typesafe":
            return AsyncTypeSafeClient(model=model_id())
        case "openrouter":
            return AsyncTypeSafeClient(
                api_key=env("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL, model=model_id()
            )
        case _:
            return AsyncTypeSafeClient(
                api_key="sk-offline-simulator", model=model_id(), transport=simulator.transport()
            )


def call_cost(response: SystemOneResponse) -> float:
    """Billed cost when the backend reports it (OpenRouter), else estimated."""
    billed = (response.raw_http_response.json().get("usage") or {}).get("cost")
    if billed is not None:
        return float(billed)
    return (response.usage.input_tokens or 0) / 1e6 * PRICE_PER_MTOK


class JevEngine:
    kind = "jev"

    def __init__(self, task: Task, client: AsyncTypeSafeClient):
        self.task = task
        self.client = client
        self.name = "jev"
        self.model = model_id()

    @traceable(run_type="chain", name="jev_engine")
    async def evaluate(self, item: dict[str, Any]) -> EngineResult:
        start = time.perf_counter()
        response = await self.client.system_one(
            self.task.state(item), self.task.questions(), response_model=self.task.response_model
        )
        latency_ms = (time.perf_counter() - start) * 1000
        decision, detail = self.task.decide(item, response)
        return EngineResult(
            decision=decision,
            reason="; ".join(detail.get("reasons", [])) or str(detail.get("factors", "")),
            latency_ms=latency_ms,
            cost=call_cost(response),
            detail={**detail, "model": response.model, "answers": response.raw_http_response.json()["answers"]},
        )
