"""Decision tasks the engines are compared on.

A task bundles everything an engine needs to make one decision per item:
Jev's typed question set, the state Jev (and the questions LLM) sees, the
deterministic policy over the answers, and the same policy in prose for the
direct LLM engine. Each item carries its ground-truth decision in "label".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from typesafe_sdk import Question, SystemOneResponse


@dataclass(frozen=True)
class Task:
    name: str
    items: list[dict[str, Any]]                      # each has "id" and "label"
    decisions: tuple[str, ...]
    questions: Callable[[], dict[str, Question]]
    state: Callable[[dict[str, Any]], Any]           # what Jev / the questions LLM reads
    response_model: type[SystemOneResponse]
    decide: Callable[[dict[str, Any], SystemOneResponse], tuple[str, dict[str, Any]]]
    direct_policy: str                               # the policy in prose, for the direct LLM
    direct_record: Callable[[dict[str, Any]], dict[str, Any]]  # what the direct LLM reads
    costly: Callable[[str, str], str | None]         # (label, predicted) -> costly error kind


def load_task(name: str) -> Task:
    if name == "claims":
        from .claims import TASK
    elif name == "loan":
        from .loan import TASK
    else:
        raise ValueError(f"unknown task {name!r}")
    return TASK


TASK_NAMES = ("claims", "loan")
