"""Jev engine: TypeSafe "System One" answers typed questions, Python applies policy.

We ask Jev a handful of calibrated yes/no (noul) questions about the SAME applicant
text, then feed the answers through policy.decide() — the identical rules the LLM is
told to follow. The difference: here the decision is deterministic Python over typed,
calibrated probabilities, not parsed prose.

Docs: https://www.langchain.com/blog/building-a-harness-with-jev
Uses the documented `Noul` primitive. TypeSafe also offers a `Score` type
(low/medium/high) which would map credit_risk more directly; once its exact
constructor/accessor is confirmed against the langchain-typesafe docs, credit_risk
can be swapped to a single Score question. For now we derive it from two nouls so the
demo runs against the documented API.

Traced in LangSmith via @traceable.
"""
from __future__ import annotations

import time

from langsmith import traceable

from jev_compare.config import EngineResult, require_env
from jev_compare.policy import Factors, decide

# Probability at/above which a noul is treated as "true".
THRESHOLD = 0.5


def _classifier():
    # Imported lazily so the module loads even before the dep is installed.
    try:
        from langchain_typesafe import TypeSafeClassifier  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "langchain-typesafe is not installed. Jev is served by TypeSafe AI "
            "(not OpenRouter); install their SDK to run this engine."
        ) from exc
    require_env("TYPESAFE_API_KEY")
    return TypeSafeClassifier()  # reads TYPESAFE_API_KEY from env


def _questions():
    from langchain_typesafe import Noul  # noqa: WPS433
    return {
        "income_sufficient": Noul(
            instructions="The applicant's stated annual income is at least 50,000."
        ),
        "employment_stable": Noul(
            instructions="The applicant has held steady employment for 2 or more years."
        ),
        "credit_high": Noul(
            instructions=(
                "The applicant is a HIGH credit risk: recent defaults, collections, "
                "missed payments, or heavily maxed-out credit."
            )
        ),
        "credit_low": Noul(
            instructions=(
                "The applicant is a LOW credit risk: clean history, no defaults, "
                "debts well managed."
            )
        ),
    }


def _credit_risk(p_high: float, p_low: float) -> str:
    if p_high >= THRESHOLD:
        return "high"
    if p_low >= THRESHOLD:
        return "low"
    return "medium"


@traceable(run_type="chain", name="jev_engine")
def evaluate(text: str) -> EngineResult:
    """Run Jev's typed questions on one applicant, then apply the policy."""
    classifier = _classifier()
    start = time.perf_counter()
    response = classifier.invoke({"state": text, "questions": _questions()})
    latency_ms = (time.perf_counter() - start) * 1000

    p = {name: response.nouls[name].noul for name in
         ("income_sufficient", "employment_stable", "credit_high", "credit_low")}

    factors = Factors(
        income_sufficient=p["income_sufficient"] >= THRESHOLD,
        employment_stable=p["employment_stable"] >= THRESHOLD,
        credit_risk=_credit_risk(p["credit_high"], p["credit_low"]),
    )
    decision = decide(factors)
    reason = (
        f"income_sufficient={factors.income_sufficient} (p={p['income_sufficient']:.2f}), "
        f"employment_stable={factors.employment_stable} (p={p['employment_stable']:.2f}), "
        f"credit_risk={factors.credit_risk} "
        f"(p_high={p['credit_high']:.2f}, p_low={p['credit_low']:.2f})"
    )
    return EngineResult(
        decision=decision,
        reason=reason,
        latency_ms=latency_ms,
        detail={"probabilities": p, "factors": factors.__dict__},
    )
