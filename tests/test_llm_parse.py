from jev_compare.engines.llm import parse


def test_plain_json():
    assert parse('{"decision": "Approve", "reason": "ok"}') == ("approve", "ok")


def test_fenced_json():
    assert parse('```json\n{"decision": "deny", "reason": "defaults"}\n```')[0] == "deny"


def test_garbage_defaults_to_review():
    decision, reason = parse("I think approve")
    assert decision == "review" and "unparseable" in reason


def test_invalid_decision_defaults_to_review():
    assert parse('{"decision": "maybe"}')[0] == "review"
