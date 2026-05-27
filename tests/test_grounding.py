from unittest.mock import patch

from backend.conversation_state import ConversationState
from backend.conversational_nlg import generate_recommendations
from backend.database import Vehicle, VehicleSearchFilters, get_vehicle_by_id, search_vehicles
from backend.grounding import reply_prices_grounded
from backend.intent import IntentKind
from backend.output_validation import ResponseLike, ValidationVerdict, validate_response_quality


def _vehicle(vid: int) -> Vehicle:
    vehicle = get_vehicle_by_id(vid)
    assert vehicle is not None
    return vehicle


def test_reply_prices_grounded_rejects_hallucinated_amount(isolated_db):
    vehicle = _vehicle(55)
    assert reply_prices_grounded("Great pick at $99,999.", [vehicle]) is False
    assert reply_prices_grounded(f"Listed at ${vehicle.price:,.0f}.", [vehicle]) is True


def test_validation_rejects_ungrounded_prices(isolated_db):
    vehicle = _vehicle(55)
    report = validate_response_quality(
        ResponseLike(
            reply="I'd start with this one at $99,999.",
            intent=IntentKind.INVENTORY_SEARCH,
            vehicles=[vehicle],
        )
    )
    assert report.verdict == ValidationVerdict.REJECT


def test_generate_recommendations_falls_back_on_bad_llm_price(isolated_db):
    state = ConversationState(session_id="nlg-ground")
    state.turn_count = 3
    state.budget = 70000
    state.passengers = 4
    vehicles = search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=3))
    with patch("backend.conversational_nlg.generate_text", return_value="Try this gem at $99,999 today."):
        text = generate_recommendations(state, vehicles)
    assert "$99,999" not in text
    assert str(vehicles[0].id) in text or vehicles[0].make in text
