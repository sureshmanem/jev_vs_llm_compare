"""Shared types + env loading for both engines."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # pull .env into os.environ (LangSmith reads these too)


@dataclass
class EngineResult:
    decision: str                 # "approve" | "review" | "deny"
    reason: str                   # human-readable justification
    latency_ms: float             # wall-clock for the engine call
    detail: dict = field(default_factory=dict)  # engine-specific extras (probs, raw)


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val
