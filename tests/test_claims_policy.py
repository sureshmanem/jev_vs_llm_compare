"""Claims routing policy tests with hand-written Jev answers (copied from jev_poc, plus regressions)."""

import copy

import pytest

from jev_compare.tasks.claims.policy import Route, decide, fraud_score, rule_signals
from jev_compare.tasks.claims.questions import FRAUD_SIGNALS
from jev_compare.tasks.claims.response import ClaimTriageResponse, response_from_json

BASE_CLAIM = {
    "id": "T-1",
    "narrative": "irrelevant: the policy only reads Jev's answers",
    "claimed_amount": 900,
    "incident_date": "2026-09-10",
    "reported_date": "2026-09-11",
    "policy": {"start_date": "2023-01-01", "deductible": 250, "coverages": ["collision"]},
}


def choice(probs: dict[str, float], confidence: float) -> dict:
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": confidence}


def score(value: float, confidence: float, levels: int = 4) -> dict:
    probs = {str(i): 1.0 if i == round(value) else 0.0 for i in range(levels)}
    legend = {str(i): f"Level {i}: description" for i in range(levels)}
    return {"type": "score", "score": value, "legend": legend, "probabilities": probs, "confidence": confidence}


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def response(**overrides) -> ClaimTriageResponse:
    answers = {
        "claim_type": choice({"auto_collision": 0.95, "other": 0.05}, 0.9),
        "severity": score(0.2, 0.85),
        "claimant_distress": score(0.3, 0.8, levels=3),
        "injury_reported": noul(0.02),
        "coverage_match": noul(0.95),
        **{key: noul(0.03) for key in FRAUD_SIGNALS},
    }
    answers.update(overrides)
    payload = {"model": "test", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}
    return response_from_json(payload)


def claim(**overrides) -> dict:
    c = copy.deepcopy(BASE_CLAIM)
    for key, value in overrides.items():
        if key in c["policy"]:
            c["policy"][key] = value
        else:
            c[key] = value
    return c


def test_clean_minor_claim_is_fast_tracked():
    assert decide(claim(), response()).route is Route.FAST_TRACK


def test_low_type_confidence_goes_to_a_human_before_anything_else():
    r = response(claim_type=choice({"auto_collision": 0.4, "other": 0.35, "liability": 0.25}, 0.1),
                 fraud_payment_pressure=noul(0.99))
    d = decide(claim(), r)
    assert d.route is Route.HUMAN_TRIAGE
    assert "auto_collision 40%" in d.reasons[0]


def test_fraud_signals_route_to_siu():
    r = response(fraud_payment_pressure=noul(0.95), fraud_vague_details=noul(0.9), fraud_inconsistent_story=noul(0.9))
    d = decide(claim(), r)
    assert d.route is Route.SIU_REVIEW
    assert "payment_pressure" in d.reasons


def test_new_policy_is_a_code_signal_not_a_model_question():
    assert rule_signals(claim(start_date="2026-09-01")) == {"new_policy": 0.3}
    assert rule_signals(claim()) == {}


def test_fraud_score_is_bounded_noisy_or():
    all_yes = response(**{key: noul(1.0) for key in FRAUD_SIGNALS})
    s = fraud_score(all_yes, {"new_policy": 0.3, "late_report": 0.15})
    assert 0.8 < s < 1.0
    assert fraud_score(response(**{key: noul(0.0) for key in FRAUD_SIGNALS}), {}) == 0.0


def test_uncovered_loss_goes_to_coverage_review():
    assert decide(claim(), response(coverage_match=noul(0.1))).route is Route.COVERAGE_REVIEW


def test_amount_below_deductible():
    assert decide(claim(claimed_amount=200), response()).route is Route.BELOW_DEDUCTIBLE


@pytest.mark.parametrize("override", [
    {"injury_reported": noul(0.8)},
    {"severity": score(2.4, 0.7)},
    {"claim_type": choice({"liability": 0.9, "other": 0.1}, 0.85)},
])
def test_high_stakes_claims_go_to_senior_adjuster(override):
    assert decide(claim(), response(**override)).route is Route.SENIOR_ADJUSTER


def test_fast_track_blocked_when_model_is_only_moderately_sure():
    r = response(claim_type=choice({"auto_collision": 0.7, "other": 0.3}, 0.55))
    d = decide(claim(), r)
    assert d.route is Route.STANDARD_ADJUSTER
    assert d.reasons == ["type confidence 0.55 < 0.7"]


def test_fast_track_blocked_by_amount():
    d = decide(claim(claimed_amount=5_000), response())
    assert d.route is Route.STANDARD_ADJUSTER
    assert "amount $5,000 > $2,500" in d.reasons


def test_distressed_claimant_is_prioritised():
    assert decide(claim(), response(claimant_distress=score(1.8, 0.7, levels=3))).priority


def test_confident_other_goes_to_a_human():
    d = decide(claim(), response(claim_type=choice({"other": 0.99, "auto_collision": 0.01}, 0.99)))
    assert d.route is Route.HUMAN_TRIAGE
    assert "fits no known type" in d.reasons[0]


def test_fraud_still_beats_other():
    r = response(claim_type=choice({"other": 0.99, "auto_collision": 0.01}, 0.99),
                 fraud_payment_pressure=noul(0.95), fraud_vague_details=noul(0.95), fraud_inconsistent_story=noul(0.95))
    assert decide(claim(), r).route is Route.SIU_REVIEW


OTHER = {"claim_type": choice({"other": 0.99, "auto_collision": 0.01}, 0.99)}


def subtype(name: str, confidence: float):
    from typesafe_sdk import ChoiceAnswer
    return ChoiceAnswer(choice=name, probabilities={name: 1.0}, confidence=confidence)


def test_other_subtype_picks_the_desk():
    d = decide(claim(), response(**OTHER), other=subtype("identity_theft", 0.95))
    assert d.route is Route.HUMAN_TRIAGE
    assert d.queue == "identity-theft desk"
    assert "sub-type identity_theft (0.95)" in d.reasons


def test_unsure_subtype_falls_back_to_general_triage():
    d = decide(claim(), response(**OTHER), other=subtype("tenant_damage", 0.2))
    assert d.queue == "general triage"


def test_no_subtype_call_means_general_triage():
    assert decide(claim(), response(**OTHER)).queue == "general triage"


def test_non_human_routes_have_no_queue():
    assert decide(claim(), response()).queue is None


def test_injury_is_not_hidden_by_a_small_claimed_amount():
    # At first notice the amount is a property estimate; an injury still needs a senior adjuster.
    d = decide(claim(claimed_amount=200), response(injury_reported=noul(0.9)))
    assert d.route is Route.SENIOR_ADJUSTER


def test_date_flags_alone_do_not_send_a_claim_to_siu():
    # New policy (0.3) + late report (0.15) crosses siu_fraud with no narrative red flag.
    c = claim(start_date="2026-09-01", incident_date="2026-09-10", reported_date="2026-11-01")
    assert set(rule_signals(c)) == {"new_policy", "late_report"}
    d = decide(c, response())
    assert d.fraud_score >= 0.4
    assert d.route is Route.STANDARD_ADJUSTER
