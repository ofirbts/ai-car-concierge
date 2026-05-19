from backend.intent import ExtractedIntent, IntentKind
from backend.intent_validate import normalize_extracted_intent


def test_unknown_make_cleared(isolated_db):
    extracted = ExtractedIntent(
        intent=IntentKind.INVENTORY_SEARCH,
        make="Ferrari",
        model="488",
    )
    normalized = normalize_extracted_intent(extracted)
    assert normalized.make is None
    assert normalized.model is None


def test_unknown_model_cleared_for_known_make(isolated_db):
    extracted = ExtractedIntent(
        intent=IntentKind.INVENTORY_SEARCH,
        make="Tesla",
        model="Cybertruck Ultra",
    )
    normalized = normalize_extracted_intent(extracted)
    assert normalized.make == "Tesla"
    assert normalized.model is None


def test_valid_make_model_kept(isolated_db):
    extracted = ExtractedIntent(
        intent=IntentKind.INVENTORY_SEARCH,
        make="Tesla",
        model="Model 3",
    )
    normalized = normalize_extracted_intent(extracted)
    assert normalized.make == "Tesla"
    assert normalized.model == "Model 3"
