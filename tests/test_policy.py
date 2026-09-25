from jev_compare.applicants import APPLICANTS
from jev_compare.policy import APPROVE, DENY, REVIEW, Factors, decide


def test_high_credit_risk_always_denies():
    assert decide(Factors(True, True, "high")) == DENY


def test_all_good_approves():
    assert decide(Factors(True, True, "low")) == APPROVE


def test_anything_else_reviews():
    assert decide(Factors(False, True, "low")) == REVIEW
    assert decide(Factors(True, False, "low")) == REVIEW
    assert decide(Factors(True, True, "medium")) == REVIEW


def test_borderline_applicants_are_review():
    assert all(a.label == REVIEW for a in APPLICANTS if a.borderline)
