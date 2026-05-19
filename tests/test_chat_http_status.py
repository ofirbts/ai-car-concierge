from backend.chat_http import chat_http_status
from backend.database import Vehicle
from backend.intent import IntentKind
from backend.orchestrator import ChatResponse


def _vehicle(year: int = 2024) -> Vehicle:
    return Vehicle(
        id=1,
        make="Tesla",
        model="Model 3",
        year=year,
        color="Black",
        price=50000,
        fuel_type="Electric",
        stock_count=1,
        pending_delisting=year < 2022,
    )


def test_chat_http_status_ok():
    r = ChatResponse(reply="ok", intent=IntentKind.INVENTORY_SEARCH)
    assert chat_http_status(r) == 200


def test_chat_http_status_legacy_blocked_is_200():
    r = ChatResponse(
        reply="legacy",
        intent=IntentKind.LEGACY_YEAR_CONFLICT,
        blocked=True,
        block_reason="policy",
    )
    assert chat_http_status(r) == 200


def test_chat_http_status_reserve_blocked_is_409():
    r = ChatResponse(
        reply="blocked",
        intent=IntentKind.RESERVE_INTENT,
        blocked=True,
        vehicles=[_vehicle(2020)],
    )
    assert chat_http_status(r) == 409


def test_chat_http_status_purchase_blocked_is_409():
    r = ChatResponse(
        reply="blocked",
        intent=IntentKind.PURCHASE_INTENT,
        blocked=True,
    )
    assert chat_http_status(r) == 409
