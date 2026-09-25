"""Paths, .env loading and the result type shared by every engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")  # pull .env into os.environ (LangSmith reads these too)


@dataclass
class EngineResult:
    decision: str                 # a task decision, or "invalid" / "error"
    reason: str                   # human-readable justification
    latency_ms: float             # wall-clock for the engine call
    cost: float = 0.0             # USD, billed where the backend reports it
    detail: dict = field(default_factory=dict)  # engine-specific extras (probs, raw)


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def require_env(name: str) -> str:
    val = env(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val
