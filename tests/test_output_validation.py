from backend.intent import IntentKind
from backend.output_validation import ResponseLike, ValidationVerdict, validate_response_quality


def test_validation_pass_for_normal_inventory_response():
    response = ResponseLike(
        reply="Here are matching vehicles in inventory.",
        intent=IntentKind.INVENTORY_SEARCH,
    )
    report = validate_response_quality(response)
    assert report.verdict == ValidationVerdict.PASS


def test_validation_reject_for_empty_reply():
    response = ResponseLike(
        reply="",
        intent=IntentKind.GENERAL_CHAT,
    )
    report = validate_response_quality(response)
    assert report.verdict == ValidationVerdict.REJECT


def test_validation_calibrate_for_policy_without_context():
    response = ResponseLike(
        reply="Refund policy is available.",
        intent=IntentKind.POLICY_QUESTION,
        policy_context_used=False,
    )
    report = validate_response_quality(response)
    assert report.verdict == ValidationVerdict.PASS
