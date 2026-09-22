"""Hand-labeled test applicants.

Each case has messy, realistic free text plus the known-correct factors and the
ground-truth decision (derived from policy.decide, so labels can't drift from the rules).

Mix: 3 clear-approve, 2 clear-deny, 3 deliberately borderline. The borderline ones
are where the LLM is expected to flip across repeated runs.
"""
from __future__ import annotations

from dataclasses import dataclass

from policy import Factors, decide, Decision


@dataclass
class Applicant:
    id: str
    text: str
    factors: Factors      # ground-truth reading of the text
    borderline: bool = False

    @property
    def label(self) -> Decision:
        return decide(self.factors)


APPLICANTS: list[Applicant] = [
    # ---- clear approve ----
    Applicant(
        id="A1",
        text=(
            "Hi there — I'm 41, been a senior nurse at the same hospital for 12 years now. "
            "I pull in about 88k a year. No missed payments ever, one credit card I pay off "
            "every month, no other debts. Looking to borrow for a kitchen remodel."
        ),
        factors=Factors(income_sufficient=True, employment_stable=True, credit_risk="low"),
    ),
    Applicant(
        id="A2",
        text=(
            "Software engineer, 6 years at my current company, salary 130,000. "
            "Excellent credit, no defaults, small car loan almost paid off."
        ),
        factors=Factors(income_sufficient=True, employment_stable=True, credit_risk="low"),
    ),
    Applicant(
        id="A3",
        text=(
            "I run a small bakery I've owned for 8 years. Takes home roughly 72k after a "
            "good year. Never defaulted on anything, credit history is clean."
        ),
        factors=Factors(income_sufficient=True, employment_stable=True, credit_risk="low"),
    ),
    # ---- clear deny (high credit risk) ----
    Applicant(
        id="D1",
        text=(
            "Look, I'll be honest, I've had two loan defaults in the last year and a "
            "credit card that went to collections. I make 95k though as a consultant."
        ),
        factors=Factors(income_sufficient=True, employment_stable=True, credit_risk="high"),
    ),
    Applicant(
        id="D2",
        text=(
            "Between jobs right now, income basically zero for 4 months. Maxed out three "
            "credit cards and missed the last few payments. Need cash urgently."
        ),
        factors=Factors(income_sufficient=False, employment_stable=False, credit_risk="high"),
    ),
    # ---- borderline (should be 'review'; LLM likely to flip) ----
    Applicant(
        id="B1",
        text=(
            "27, started my current job 14 months ago, earning 61k. Credit's okay I think — "
            "one late payment last year on a phone bill but otherwise fine. Some student debt."
        ),
        # income ok, employment < 2yr -> not stable, credit medium -> review
        factors=Factors(income_sufficient=True, employment_stable=False, credit_risk="medium"),
        borderline=True,
    ),
    Applicant(
        id="B2",
        text=(
            "Been teaching for 15 years, very stable. But part-time now so income is around "
            "44k. Credit is spotless, never missed anything in my life."
        ),
        # income < 50k -> insufficient, employment stable, credit low -> review
        factors=Factors(income_sufficient=False, employment_stable=True, credit_risk="low"),
        borderline=True,
    ),
    Applicant(
        id="B3",
        text=(
            "Salary 78k, 5 years at the firm. Had a rough patch two years ago — one default "
            "that's since been settled, and I'm carrying a fair bit of credit card debt still."
        ),
        # income ok, employment stable, but lingering debt/past default -> medium -> review
        factors=Factors(income_sufficient=True, employment_stable=True, credit_risk="medium"),
        borderline=True,
    ),
]
