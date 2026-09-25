"""Deterministic business policy over Jev's typed answers.

Jev tells us *what* it sees and *how sure* it is. This module decides what to
do about it. Everything here is ordinary, testable code: thresholds scale with
the cost of a wrong decision (see docs.typesafe.ai/confidence), and the facts
Jev is weak at (dates, money, arithmetic) are handled here, not by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from typesafe_sdk import ChoiceAnswer

from .questions import FRAUD_SIGNALS
from .response import ClaimTriageResponse


@dataclass(frozen=True)
class Thresholds:
    """Routing cut-offs. `evaluate.py` sweeps these against labelled claims."""

    min_type_confidence: float = 0.4          # below this Jev doesn't know what the claim is
    fast_track_type_confidence: float = 0.7   # auto-paying money needs a much clearer read
    fast_track_severity_confidence: float = 0.6
    fast_track_max_severity: float = 1.0      # score is 0..3; tuned 0.8 -> 1.0 on labelled set
    fast_track_max_amount: int = 2_500
    fast_track_max_fraud: float = 0.2
    fast_track_max_injury: float = 0.2
    fast_track_min_coverage: float = 0.8
    siu_fraud: float = 0.4                    # tuned 0.5 -> 0.4: 0.5 missed 2/7 fraud cases
    fraud_flag: float = 0.5                   # a narrative red flag counts as present
    min_coverage: float = 0.3
    senior_severity: float = 2.0              # "major" or worse
    injury: float = 0.5
    priority_distress: float = 1.5            # between "anxious" and "highly distressed"


DEFAULT_THRESHOLDS = Thresholds()

NEW_POLICY_DAYS = 30
LATE_REPORT_DAYS = 30
NEW_POLICY_WEIGHT = 0.3
LATE_REPORT_WEIGHT = 0.15


# Desk for each second-level "other" sub-type (see questions.OTHER_SUBTYPES).
OTHER_QUEUES = {
    "personal_property": "personal-property desk",
    "identity_theft": "identity-theft desk",
    "workmanship_dispute": "coverage team (workmanship is usually excluded)",
    "tenant_damage": "landlord/rental desk",
    "not_enough_information": "call claimant for details",
}
GENERAL_QUEUE = "general triage"


class Route(str, Enum):
    FAST_TRACK = "fast_track"                 # auto-approve, pay without an adjuster
    BELOW_DEDUCTIBLE = "below_deductible"     # tell the claimant, no payout expected
    STANDARD_ADJUSTER = "standard_adjuster"
    SENIOR_ADJUSTER = "senior_adjuster"       # injuries, liability, big losses
    COVERAGE_REVIEW = "coverage_review"       # loss may not be covered by the policy
    SIU_REVIEW = "siu_review"                 # special investigations unit
    HUMAN_TRIAGE = "human_triage"             # model unsure; a person classifies it


@dataclass
class Decision:
    claim_id: str
    route: Route
    claim_type: str
    type_confidence: float
    severity: float
    fraud_score: float
    priority: bool
    reasons: list[str] = field(default_factory=list)
    queue: str | None = None  # which human desk, for HUMAN_TRIAGE claims


def jev_state(claim: dict[str, Any]) -> dict[str, Any]:
    """What we send to Jev: only the text it needs to read.

    Amounts and dates are left out on purpose; the model cannot compare them
    reliably and they would only add noise.
    """
    return {
        "narrative": claim["narrative"],
        "policy": {"coverages": claim["policy"]["coverages"]},
    }


def rule_signals(claim: dict[str, Any]) -> dict[str, float]:
    """Fraud signals that are facts, computed exactly in code."""
    policy = claim["policy"]
    incident = date.fromisoformat(claim["incident_date"])
    reported = date.fromisoformat(claim["reported_date"])
    started = date.fromisoformat(policy["start_date"])
    signals: dict[str, float] = {}
    if (incident - started).days < NEW_POLICY_DAYS:
        signals["new_policy"] = NEW_POLICY_WEIGHT
    if (reported - incident).days > LATE_REPORT_DAYS:
        signals["late_report"] = LATE_REPORT_WEIGHT
    return signals


def fraud_score(r: ClaimTriageResponse, rules: dict[str, float]) -> float:
    """Noisy-OR over weighted signals: any strong signal alone can raise the
    score, and several weak ones compound, but it never exceeds 1."""
    keep = 1.0
    for key, (_text, weight) in FRAUD_SIGNALS.items():
        keep *= 1 - weight * getattr(r, key).noul
    for weight in rules.values():
        keep *= 1 - weight
    return round(1 - keep, 3)


def decide(
    claim: dict[str, Any],
    r: ClaimTriageResponse,
    t: Thresholds = DEFAULT_THRESHOLDS,
    other: ChoiceAnswer | None = None,
) -> Decision:
    """`other` is the second-level sub-type answer, present only when Jev's
    first pass said the claim fits no standard type."""
    ctype = r.claim_type
    severity = r.severity
    injury = r.injury_reported.noul
    coverage = r.coverage_match.noul
    rules = rule_signals(claim)
    fraud = fraud_score(r, rules)
    amount = claim["claimed_amount"]
    deductible = claim["policy"]["deductible"]

    def done(route: Route, *reasons: str, queue: str | None = None) -> Decision:
        return Decision(
            claim_id=claim["id"],
            route=route,
            claim_type=ctype.choice,
            type_confidence=round(ctype.confidence, 3),
            severity=round(severity.score, 2),
            fraud_score=fraud,
            priority=r.claimant_distress.score >= t.priority_distress,
            reasons=list(reasons),
            queue=queue,
        )

    # Checked in order of how costly a wrong decision would be.
    # A confident second-level answer picks the human desk; otherwise a
    # generalist takes it.
    sub_ok = other is not None and other.confidence >= t.min_type_confidence
    sub_reasons = [f"sub-type {other.choice} ({other.confidence:.2f})"] if sub_ok else []
    desk = OTHER_QUEUES.get(other.choice, GENERAL_QUEUE) if sub_ok else GENERAL_QUEUE

    if ctype.confidence < t.min_type_confidence:
        top = sorted(ctype.probabilities.items(), key=lambda kv: -kv[1])[:2]
        return done(
            Route.HUMAN_TRIAGE,
            f"claim type unclear (confidence {ctype.confidence:.2f}): "
            + " vs ".join(f"{k} {v:.0%}" for k, v in top),
            *sub_reasons,
            queue=desk,
        )

    # SIU needs at least one red flag in the narrative itself. Date facts
    # (new policy, late report) raise the score but can't send a claim to
    # investigation on their own; they still block fast-track below.
    flags = [k.removeprefix("fraud_") for k in FRAUD_SIGNALS if getattr(r, k).noul >= t.fraud_flag]
    if fraud >= t.siu_fraud and flags:
        return done(Route.SIU_REVIEW, f"fraud score {fraud:.2f}", *flags, *rules)

    # Confidence says how sure Jev is of its label, not how clear the claim is:
    # a confident "other" means "I'm sure this fits none of the types".
    if ctype.choice == "other":
        reason = f"claim fits no known type (other, confidence {ctype.confidence:.2f})"
        return done(Route.HUMAN_TRIAGE, reason, *sub_reasons, queue=desk)

    if coverage < t.min_coverage:
        return done(
            Route.COVERAGE_REVIEW,
            f"loss may not match policy coverages (p={coverage:.2f})",
        )

    senior = []
    if injury >= t.injury:
        senior.append(f"injury reported (p={injury:.2f})")
    if severity.score >= t.senior_severity:
        senior.append(f"severity {severity.score:.2f} ({severity.legend[round(severity.score)].split(':')[0]})")
    if ctype.choice == "liability":
        senior.append("third-party liability")
    if senior:
        return done(Route.SENIOR_ADJUSTER, *senior)

    # After the senior checks: at first notice the amount is the claimant's
    # estimate of property damage, so a small figure must not hide an injury.
    if amount <= deductible:
        return done(Route.BELOW_DEDUCTIBLE, f"claimed ${amount:,} <= deductible ${deductible:,}")

    blockers = []
    if ctype.confidence < t.fast_track_type_confidence:
        blockers.append(f"type confidence {ctype.confidence:.2f} < {t.fast_track_type_confidence}")
    if severity.score > t.fast_track_max_severity:
        blockers.append(f"severity {severity.score:.2f} > {t.fast_track_max_severity}")
    if severity.confidence < t.fast_track_severity_confidence:
        blockers.append(f"severity confidence {severity.confidence:.2f}")
    if injury >= t.fast_track_max_injury:
        blockers.append(f"possible injury (p={injury:.2f})")
    if coverage < t.fast_track_min_coverage:
        blockers.append(f"coverage not certain (p={coverage:.2f})")
    if fraud >= t.fast_track_max_fraud:
        blockers.append(f"fraud score {fraud:.2f}")
    if amount > t.fast_track_max_amount:
        blockers.append(f"amount ${amount:,} > ${t.fast_track_max_amount:,}")

    if not blockers:
        return done(Route.FAST_TRACK, "minor, covered, low risk, clear read")
    return done(Route.STANDARD_ADJUSTER, *blockers)

