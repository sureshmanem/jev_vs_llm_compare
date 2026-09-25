"""A typed response model: one attribute per question, checked by pydantic.

Passing this as `response_model=` to `system_one` means a missing or wrongly
typed answer fails loudly at the boundary instead of deep inside routing code.
"""

import httpx2
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse


class ClaimTriageResponse(SystemOneResponse):
    claim_type: ChoiceAnswer
    severity: ScoreAnswer
    claimant_distress: ScoreAnswer
    injury_reported: NoulAnswer
    coverage_match: NoulAnswer
    fraud_inconsistent_story: NoulAnswer
    fraud_payment_pressure: NoulAnswer
    fraud_vague_details: NoulAnswer
    fraud_preexisting_damage: NoulAnswer


def response_from_json(payload: dict) -> ClaimTriageResponse:
    """Rebuild a typed response from a saved API payload (tests, cached evals).

    Goes through the SDK's own decoder so validation matches a live call.
    """
    request = httpx2.Request("POST", "https://cache.invalid/v1/systemone")
    return ClaimTriageResponse._decode(httpx2.Response(200, json=payload, request=request))
