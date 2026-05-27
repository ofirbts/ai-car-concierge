from backend.config import reset_settings_cache
from backend.intent import IntentKind
from backend.output_validation import ValidationVerdict, ResponseLike, validate_response_quality


def test_validation_profile_strict_calibrates_earlier(monkeypatch):
    monkeypatch.setenv("VALIDATION_PROFILE", "strict")
    reset_settings_cache()
    report = validate_response_quality(
        ResponseLike(
            reply="short",
            intent=IntentKind.POLICY_QUESTION,
            policy_context_used=False,
        )
    )
    assert report.profile == "strict"
    assert report.verdict == ValidationVerdict.CALIBRATE
    monkeypatch.delenv("VALIDATION_PROFILE", raising=False)
    reset_settings_cache()


def test_validation_profile_relaxed_allows_more_risk(monkeypatch):
    monkeypatch.setenv("VALIDATION_PROFILE", "relaxed")
    reset_settings_cache()
    report = validate_response_quality(
        ResponseLike(
            reply="short",
            intent=IntentKind.POLICY_QUESTION,
            policy_context_used=False,
        )
    )
    assert report.profile == "relaxed"
    assert report.verdict == ValidationVerdict.PASS

    monkeypatch.delenv("VALIDATION_PROFILE", raising=False)
    reset_settings_cache()
