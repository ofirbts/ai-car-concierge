from backend.intent import IntentKind
from backend.orchestrator import ChatResponse


def chat_http_status(response: ChatResponse) -> int:
    if not response.blocked:
        return 200
    if response.intent == IntentKind.LEGACY_YEAR_CONFLICT:
        return 200
    if response.intent in (IntentKind.RESERVE_INTENT, IntentKind.PURCHASE_INTENT):
        return 409
    return 409
