"""Jev's typed questions for the loan task (the three factors in policy.py)."""

from typesafe_sdk import Noul, Question, Score

CREDIT_LEVELS: list[str] = [
    "Low: clean credit history, no defaults, debts well managed",
    "Medium: minor blemishes such as a late payment, a settled past default, or notable debt",
    "High: recent defaults, collections, missed payments, or maxed-out credit",
]


def build_questions() -> dict[str, Question]:
    return {
        "income_sufficient": Noul(instructions="The applicant's stated annual income is at least 50,000."),
        "employment_stable": Noul(
            instructions="The applicant has held steady employment for 2 or more years."
        ),
        "credit_risk": Score(instructions="How risky is the applicant's credit?", criteria=CREDIT_LEVELS),
    }
