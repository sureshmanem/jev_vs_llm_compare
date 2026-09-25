"""Insurance claim (FNOL) triage: 50 hand-labelled claims, 7 routes.

`questions.py`, `response.py` and `policy.py` are copied from the jev_poc
project (claims_triage/), with two policy fixes: injury/severity/liability are
checked before the deductible, and SIU needs a narrative red flag, not date
facts alone.
"""

from __future__ import annotations

import json
from typing import Any

from jev_compare.config import ROOT

from .policy import Route, decide, jev_state
from .questions import build_questions
from .response import ClaimTriageResponse

DATA = ROOT / "data" / "claims" / "labelled_claims.json"

DIRECT_POLICY = """Route the insurance claim (first notice of loss) to exactly one queue. \
Check these rules in order; the first one that applies decides the route.

1. human_triage: you cannot tell what kind of loss is being reported.
2. siu_review: the narrative itself contains at least one fraud red flag (the story contradicts \
itself or changes; pressure for immediate or cash payment or to skip the inspection; refusal or \
inability to give basic details such as where or when; hints that damage existed before this \
incident) AND overall fraud suspicion is material. A policy that started less than 30 days \
before the incident, or a report more than 30 days after it, adds suspicion but never justifies \
SIU on its own.
3. human_triage: the loss fits none of these types: auto collision; auto comprehensive (theft, \
break-in, vandalism, glass); property water damage; property fire/smoke; third-party liability.
4. coverage_review: the loss is probably not the kind of loss covered by any of the policy's coverages.
5. senior_adjuster: any person was physically injured, OR the damage is major or catastrophic \
(likely total loss of a vehicle or room, home uninhabitable), OR it is a third-party liability claim.
6. below_deductible: the claimed amount is less than or equal to the deductible.
7. fast_track (automatic payment, no adjuster): ONLY if all hold: the claim type is clear; damage \
is minor or at most moderate (repairable); no sign of injury; clearly covered; no fraud red flag \
and the policy did not start less than 30 days before the incident; claimed amount at most $2,500.
8. standard_adjuster: everything else.

Return JSON with the route and a one-sentence reason."""


def _items() -> list[dict[str, Any]]:
    return [{**c, "label": c["labels"]["route"]} for c in json.loads(DATA.read_text())]


def _decide(claim: dict[str, Any], r: ClaimTriageResponse) -> tuple[str, dict[str, Any]]:
    d = decide(claim, r)
    return d.route.value, {
        "claim_type": d.claim_type,
        "claim_type_ok": d.claim_type == claim["labels"]["claim_type"],
        "fraud_score": d.fraud_score,
        "reasons": d.reasons,
    }


def _record(claim: dict[str, Any]) -> dict[str, Any]:
    return {k: claim[k] for k in ("narrative", "claimed_amount", "incident_date", "reported_date", "policy")}


def _costly(label: str, pred: str) -> str | None:
    """Mistakes that cost money or harm (from jev_poc's evaluate.py)."""
    if label == pred:
        return None
    if pred == Route.FAST_TRACK.value:
        return "paid_wrongly"
    if label == Route.SIU_REVIEW.value:
        return "missed_fraud"
    if label == Route.SENIOR_ADJUSTER.value and pred in {
        Route.FAST_TRACK.value, Route.STANDARD_ADJUSTER.value, Route.BELOW_DEDUCTIBLE.value,
    }:
        return "missed_senior"
    return None


def _task():
    from jev_compare.tasks import Task

    return Task(
        name="claims",
        items=_items(),
        decisions=tuple(r.value for r in Route),
        questions=build_questions,
        state=jev_state,
        response_model=ClaimTriageResponse,
        decide=_decide,
        direct_policy=DIRECT_POLICY,
        direct_record=_record,
        costly=_costly,
    )


TASK = _task()
