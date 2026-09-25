"""Loan eligibility: 8 hand-labelled applicants (3 borderline), 3 decisions."""

from __future__ import annotations

from typing import Any

from typesafe_sdk import NoulAnswer, ScoreAnswer, SystemOneResponse

from .applicants import APPLICANTS
from .policy import APPROVE, DENY, POLICY_TEXT, REVIEW, Factors, decide
from .questions import CREDIT_LEVELS, build_questions

THRESHOLD = 0.5  # a Noul at/above this counts as true


class LoanResponse(SystemOneResponse):
    income_sufficient: NoulAnswer
    employment_stable: NoulAnswer
    credit_risk: ScoreAnswer


def _items() -> list[dict[str, Any]]:
    return [{"id": a.id, "text": a.text, "label": a.label, "borderline": a.borderline} for a in APPLICANTS]


def _decide(item: dict[str, Any], r: LoanResponse) -> tuple[str, dict[str, Any]]:
    level = min(len(CREDIT_LEVELS) - 1, max(0, round(r.credit_risk.score)))
    factors = Factors(
        income_sufficient=r.income_sufficient.noul >= THRESHOLD,
        employment_stable=r.employment_stable.noul >= THRESHOLD,
        credit_risk=("low", "medium", "high")[level],
    )
    return decide(factors), {
        "factors": factors.__dict__,
        "p_income": round(r.income_sufficient.noul, 3),
        "p_employment": round(r.employment_stable.noul, 3),
        "credit_score": round(r.credit_risk.score, 3),
    }


def _costly(label: str, pred: str) -> str | None:
    if label != APPROVE and pred == APPROVE:
        return "approved_wrongly"
    if label == DENY and pred != DENY:
        return "missed_deny"
    return None


def _task():
    from jev_compare.tasks import Task

    return Task(
        name="loan",
        items=_items(),
        decisions=(APPROVE, REVIEW, DENY),
        questions=build_questions,
        state=lambda item: item["text"],
        response_model=LoanResponse,
        decide=_decide,
        direct_policy=POLICY_TEXT + "\nReturn JSON with the decision (approve, review or deny) and a one-sentence reason.",
        direct_record=lambda item: {"applicant_summary": item["text"]},
        costly=_costly,
    )


TASK = _task()
