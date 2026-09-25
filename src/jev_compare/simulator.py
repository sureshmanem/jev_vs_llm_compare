"""Offline stand-in for the Jev API (copied from jev_poc), used when no Jev key is set.

It is plugged into the real SDK as an httpx transport, so the request the SDK
builds and the response parsing (including `response_model` validation) are
exactly the same as in live mode. Only the "model" differs: here it is a small
keyword heuristic that returns the same wire format as POST /v1/systemone
(answers with `probabilities`, `confidence`, `legend`, `noul`, and `usage`).

Its judgements are crude by design. Use it to learn the API shape and exercise
the routing code, never to judge Jev's accuracy.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from typing import Any

import httpx2

SIMULATED_MODEL = "jev-simulated"

# Keywords per question id -> per option/level. The real API never sees the
# question id; the simulator uses it only to look up these cue lists.
CHOICE_CUES: dict[str, dict[str, list[str]]] = {
    "claim_type": {
        "auto_collision": ["rear-ended", "collided", "crash", "hit a", "hit the", "fender", "bumper", "swerved", "intersection", "backed into", "t-boned", "scraped", "pillar", "red light", "other driver"],
        "auto_comprehensive": ["stolen", "broke into my car", "smashed", "break-in", "car was gone", "catalytic", "theft", "windshield", "vandal"],
        "property_water": ["pipe", "leak", "flood", "water", "burst", "ceiling stain", "soaked", "sump"],
        "property_fire": ["fire", "smoke", "burned", "flames", "scorched", "electrical"],
        "liability": ["slipped", "tripped", "neighbor's", "sued", "lawyer", "their fence", "fell on my", "dog bit"],
        "other": [],
    },
    "other_subtype": {
        "personal_property": ["phone", "jewelry", "luggage", "wallet", "fell in", "lost my"],
        "identity_theft": ["identity", "credit card", "in my name", "bank account"],
        "workmanship_dispute": ["contractor", "renovation", "bad job", "repair shop", "builder"],
        "tenant_damage": ["tenant", "renter", "moved out", "landlord"],
        "not_enough_information": ["something happened", "not sure", "call me", "file a claim", "not really sure"],
    },
}
SCORE_CUES: dict[str, list[list[str]]] = {
    "severity": [
        ["scratch", "scuff", "small dent", "minor", "chip", "cosmetic", "small"],
        ["dent", "replace", "repair", "bumper", "window", "ceiling", "drywall", "carpet", "not drivable"],
        ["totaled", "total loss", "destroyed", "gutted", "entire kitchen", "whole floor", "airbags"],
        ["uninhabitable", "burned down", "whole house", "collapsed", "several cars"],
    ],
    "claimant_distress": [
        ["no rush", "whenever", "just reporting", "fyi"],
        ["worried", "stressed", "not sure what to do", "anxious", "please"],
        ["panicking", "desperate", "!!!", "nowhere to live", "kids", "can't sleep", "terrified"],
    ],
}
NOUL_CUES: dict[str, list[str]] = {
    "injury_reported": ["hospital", "injured", "whiplash", "hurt", " er ", "ambulance", "broke his", "broke her", "stitches", "neck pain"],
    "coverage_match": [],  # handled specially below
    "fraud_inconsistent_story": ["actually", "or maybe", "wait,", "i mean", "not sure if it was"],
    "fraud_payment_pressure": ["cash", "today", "right away", "skip the inspection", "no need to send", "immediately", "wire"],
    "fraud_vague_details": ["don't remember", "somewhere", "can't recall", "not sure where", "sometime"],
    "fraud_preexisting_damage": ["already", "before this", "old damage", "had a crack", "previous"],
}
COVERAGE_FOR_TYPE = {
    "auto_collision": "collision",
    "auto_comprehensive": "comprehensive",
    "property_water": "water",
    "property_fire": "fire",
    "liability": "liability",
}


def _text(value: Any) -> str:
    return (value if isinstance(value, str) else json.dumps(value)).lower()


def _hits(text: str, cues: list[str]) -> int:
    return sum(1 for cue in cues if cue in text)


def _words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", _text(value)))


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _confidence(probs: list[float]) -> float:
    # Same shape-based measure the docs illustrate: 1 when all mass is on one
    # option, 0 when it is spread evenly.
    n = len(probs)
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))


def _choice(qid: str, q: dict[str, Any], text: str) -> dict[str, Any]:
    options = list(q["criteria"])
    cues = CHOICE_CUES.get(qid)
    if cues:
        logits = [1.8 * _hits(text, cues.get(o, [])) + (0.6 if o == "other" else 0.0) for o in options]
    else:  # generic fallback: word overlap between option text and state
        logits = [0.6 * len(_words([o, q["criteria"][o]]) & _words(text)) for o in options]
    probs = _softmax(logits)
    best = options[probs.index(max(probs))]
    return {
        "type": "choice",
        "choice": best,
        "probabilities": {o: round(p, 4) for o, p in zip(options, probs)},
        "confidence": round(_confidence(probs), 4),
    }


def _score(qid: str, q: dict[str, Any], text: str) -> dict[str, Any]:
    levels = q["criteria"]
    cues = SCORE_CUES.get(qid) or [[] for _ in levels]
    logits = [1.6 * _hits(text, cues[i] if i < len(cues) else []) for i in range(len(levels))]
    logits[0] += 0.8  # with no evidence, lean towards the lowest level
    probs = _softmax(logits)
    return {
        "type": "score",
        "score": round(sum(i * p for i, p in enumerate(probs)), 4),
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
        "confidence": round(_confidence(probs), 4),
    }


def _noul(qid: str, q: dict[str, Any], state: Any, text: str) -> dict[str, Any]:
    if qid == "coverage_match" and isinstance(state, dict):
        ctype = _choice("claim_type", {"criteria": CHOICE_CUES["claim_type"]}, text)["choice"]
        needed = COVERAGE_FOR_TYPE.get(ctype)
        covered = _text(state.get("policy", {}).get("coverages", []))
        p = 0.93 if needed and needed in covered else 0.12 if needed else 0.45
    else:
        cues = NOUL_CUES.get(qid) or list(_words(q["instructions"]))
        p = 1 / (1 + math.exp(-(-2.6 + 3.2 * _hits(text, cues))))
    return {"type": "noul", "noul": round(p, 4)}


def answer(body: dict[str, Any]) -> dict[str, Any]:
    state = body["state"]
    text = _text(state.get("narrative", state) if isinstance(state, dict) else state)
    answers = {}
    for qid, q in body["questions"].items():
        kind = q["type"]
        if kind == "choice":
            answers[qid] = _choice(qid, q, text)
        elif kind == "score":
            answers[qid] = _score(qid, q, text)
        else:
            answers[qid] = _noul(qid, q, state, text)
    return {
        "model": SIMULATED_MODEL,
        "answers": answers,
        "usage": {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 0},
    }


async def _handler(request: httpx2.Request) -> httpx2.Response:
    if not request.url.path.endswith("/v1/systemone"):
        return httpx2.Response(404, json={"detail": "simulator only implements /v1/systemone"})
    body = json.loads(request.content)
    await asyncio.sleep(random.uniform(0.07, 0.25))  # Jev's documented 70-500ms range
    return httpx2.Response(200, json=answer(body))


def transport() -> httpx2.AsyncBaseTransport:
    return httpx2.MockTransport(_handler)
