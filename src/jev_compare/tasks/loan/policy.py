"""Shared eligibility policy.

This is the single source of truth for what "approve / review / deny" means.
- The LLM engine is *told* this policy in its prompt (fair fight).
- The Jev engine *applies* this policy deterministically over Jev's typed answers.

Keeping it here means the rules are auditable and both engines are judged against
the same standard.
"""
from __future__ import annotations

from dataclasses import dataclass

Decision = str  # "approve" | "review" | "deny"

APPROVE = "approve"
REVIEW = "review"
DENY = "deny"

# Human-readable policy, injected verbatim into the LLM prompt.
POLICY_TEXT = """\
Loan eligibility policy. Evaluate three factors from the applicant summary:

  1. income_sufficient  - Is stated annual income at least 50,000?
  2. employment_stable  - Has the applicant held steady employment for 2+ years?
  3. credit_risk        - Overall credit risk: low, medium, or high
                          (based on stated credit history, debts, defaults).

Decision rules, applied in order:
  - If credit_risk is "high"                         -> deny
  - If income_sufficient AND employment_stable
        AND credit_risk is "low"                     -> approve
  - Otherwise                                        -> review
"""


@dataclass
class Factors:
    income_sufficient: bool
    employment_stable: bool
    credit_risk: str  # "low" | "medium" | "high"


def decide(f: Factors) -> Decision:
    """The policy as executable code. Both the ground-truth labels and the Jev
    engine route through this exact function."""
    if f.credit_risk == "high":
        return DENY
    if f.income_sufficient and f.employment_stable and f.credit_risk == "low":
        return APPROVE
    return REVIEW
