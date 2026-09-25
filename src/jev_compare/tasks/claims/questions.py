"""The Jev question set for first-notice-of-loss (FNOL) claim triage.

This file is the "prompt" of a System One app. Instead of asking an LLM to write
prose that we then parse, we ask Jev a set of *typed* questions and get back typed
answers with probabilities, all in one parallel pass:

    Choice -> pick one of N labelled options      (claim_type)
    Score  -> position on an ordered rubric       (severity, claimant_distress)
    Noul   -> probability that a statement is true (injury, coverage, fraud signals)

Things Jev is documented to be bad at (arithmetic, ordering dates, comparing
amounts) are deliberately NOT asked here. They are computed in `policy.py`.
"""

from typesafe_sdk import Choice, Noul, Question, Score

CLAIM_TYPES: dict[str, str] = {
    "auto_collision": "A vehicle collided with another vehicle, object, or person",
    "auto_comprehensive": "A vehicle was stolen, broken into, vandalized, or had its glass damaged by something other than a collision",
    "property_water": "Home or building damage caused by water, leaks, burst pipes, or flooding",
    "property_fire": "Home or building damage caused by fire, smoke, or electrical burning",
    "liability": "Someone else was hurt or had property damaged and holds the policyholder responsible",
    "other": "Anything that does not clearly fit the categories above",
}

SEVERITY_LEVELS: list[str] = [
    "Minor: cosmetic damage, everything still usable",
    "Moderate: repairable damage, temporarily unusable",
    "Major: extensive damage, likely total loss of the vehicle or room",
    "Catastrophic: home uninhabitable or multiple vehicles/structures destroyed",
]

DISTRESS_LEVELS: list[str] = [
    "Calm and matter-of-fact",
    "Worried or anxious",
    "Highly distressed or panicking",
]

# Fraud red flags, fanned out as independent Nouls. Each one is a narrow,
# literal yes/no question, which is what System One models are best at. The
# weights are ours (business policy), not the model's.
FRAUD_SIGNALS: dict[str, tuple[str, float]] = {
    "fraud_inconsistent_story": (
        "The claimant's account of events contradicts itself or changes partway through",
        0.35,
    ),
    "fraud_payment_pressure": (
        "The claimant pushes for immediate payment, cash, or skipping the inspection",
        0.25,
    ),
    "fraud_vague_details": (
        "The claimant cannot or will not give basic details such as where or when it happened",
        0.2,
    ),
    "fraud_preexisting_damage": (
        "The narrative hints that some of the damage existed before this incident",
        0.2,
    ),
}


def build_questions() -> dict[str, Question]:
    """All questions for one claim. Jev answers every one of them in a single call."""
    questions: dict[str, Question] = {
        "claim_type": Choice(
            instructions="What kind of loss is the claimant reporting in `narrative`?",
            criteria=CLAIM_TYPES,
        ),
        "severity": Score(
            instructions="How severe is the physical damage described in `narrative`?",
            criteria=SEVERITY_LEVELS,
        ),
        "claimant_distress": Score(
            instructions="How emotionally distressed does the claimant sound in `narrative`?",
            criteria=DISTRESS_LEVELS,
        ),
        "injury_reported": Noul(
            instructions="Does `narrative` say that any person was physically injured?",
            criteria={
                "true": "Someone was hurt, went to hospital, or needed medical care",
                "false": "No one was hurt, or injuries are not mentioned",
            },
        ),
        # Structured instructions: the question refers to a field of `state` by
        # name in backticks, so Jev compares the narrative against the policy.
        "coverage_match": Noul(
            instructions={
                "question": "Is the loss in `narrative` the kind of loss described by at least one entry in `policy.coverages`?",
            },
        ),
    }
    for key, (text, _weight) in FRAUD_SIGNALS.items():
        questions[key] = Noul(instructions=text)
    return questions


# Second level of a hierarchical classification: asked only when the first pass
# says "other", so the common case stays a single call.
OTHER_SUBTYPES: dict[str, str] = {
    "personal_property": "Personal belongings such as a phone, jewelry or luggage were lost, damaged or stolen away from a vehicle or insured building",
    "identity_theft": "Someone used the claimant's identity, accounts or credit fraudulently",
    "workmanship_dispute": "A complaint about poor work by a contractor, repairer or builder",
    "tenant_damage": "Damage or mess left by a tenant, or a landlord-tenant dispute",
    "not_enough_information": "The narrative does not say what was lost or damaged, or how it happened",
}


def build_other_questions() -> dict[str, Question]:
    return {
        "other_subtype": Choice(
            instructions="The claim in `narrative` fits none of the standard auto or property loss types. What kind of request is it?",
            criteria=OTHER_SUBTYPES,
        ),
    }
