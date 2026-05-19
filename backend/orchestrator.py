from __future__ import annotations

import logging

from pydantic import BaseModel, EmailStr, Field

from backend.audit import audit
from backend.automations import send_purchase_email, send_purchase_inquiry_email
from backend.database import (
    POLICY_BLOCK_MESSAGE,
    SALES_MIN_YEAR,
    PolicyViolationError,
    Vehicle,
    VehicleSearchFilters,
    OutOfStockError,
    assert_sellable,
    get_vehicle_by_id,
    reserve_vehicle,
    search_vehicles,
    try_claim_purchase_notification,
)
from backend.intent import (
    ExtractedIntent,
    IntentKind,
    classify_intent,
)
from backend.intent_validate import normalize_extracted_intent
from backend.rag_service import PolicyRAGService, get_policy_rag_service

logger = logging.getLogger(__name__)

__all__ = [
    "IntentKind",
    "ExtractedIntent",
    "classify_intent",
    "ChatRequest",
    "ChatResponse",
    "handle_chat",
    "log_chat_outcome",
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_email: EmailStr | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    reply: str
    intent: IntentKind
    vehicles: list[Vehicle] = Field(default_factory=list)
    policy_context_used: bool = False
    rag_mode: str | None = None
    email_sent: bool = False
    email_error: str | None = None
    reserved_vehicle: Vehicle | None = None
    blocked: bool = False
    block_reason: str | None = None


def _format_vehicle_list(vehicles: list[Vehicle]) -> str:
    if not vehicles:
        return "No vehicles matched your search."
    lines = []
    for v in vehicles[:10]:
        status = "Pending De-listing (not for sale)" if v.pending_delisting else f"stock: {v.stock_count}"
        lines.append(
            f"- #{v.id}: {v.year} {v.make} {v.model} ({v.color}) ${v.price:,.0f} — {status}"
        )
    return "\n".join(lines)


def _policy_context(message: str, rag: PolicyRAGService) -> tuple[str, bool, str]:
    result = rag.search(message, top_k=3)
    context = PolicyRAGService.format_context(result)
    return context, bool(context), result.retrieval_mode


def log_chat_outcome(response: ChatResponse) -> None:
    logger.info(
        "chat_outcome intent=%s blocked=%s rag_mode=%s email_sent=%s",
        response.intent.value,
        response.blocked,
        response.rag_mode,
        response.email_sent,
    )


def _purchase_email_allowed(
    request: ChatRequest,
    email: str,
    vehicle_id: int | None,
) -> bool:
    key = request.idempotency_key
    if not key:
        return True
    if try_claim_purchase_notification(key, email, vehicle_id):
        return True
    audit(
        "purchase_email",
        "replay",
        vehicle_id=vehicle_id,
        customer_email=email,
    )
    return False


def _purchase_replay_response(intent: IntentKind) -> ChatResponse:
    return ChatResponse(
        reply=(
            "We already received your purchase interest. "
            "Our sales team will contact you."
        ),
        intent=intent,
        email_sent=False,
    )


def _search_inventory(extracted: ExtractedIntent) -> list[Vehicle]:
    filters = VehicleSearchFilters(
        make=extracted.make,
        model=extracted.model,
        year=extracted.year,
        year_min=extracted.year_min if extracted.year is None else None,
        price_max=extracted.price_max,
        limit=15,
    )
    return search_vehicles(filters)


def _sellable_alternatives(extracted: ExtractedIntent, limit: int = 5) -> list[Vehicle]:
    filters = VehicleSearchFilters(
        make=extracted.make,
        model=extracted.model,
        year_min=SALES_MIN_YEAR,
        price_max=extracted.price_max,
        in_stock_only=True,
        limit=limit,
    )
    return [v for v in search_vehicles(filters) if not v.pending_delisting]


def _handle_inventory(
    message: str,
    extracted: ExtractedIntent,
    rag: PolicyRAGService,
    intent: IntentKind,
) -> ChatResponse:
    vehicles = _search_inventory(extracted)
    delisted = [v for v in vehicles if v.pending_delisting]
    inventory_block = _format_vehicle_list(vehicles)
    fallback = inventory_block
    if delisted:
        fallback += (
            f"\n\nNote: {len(delisted)} result(s) are Pending De-listing (pre-2022) "
            "and cannot be sold or reserved per our 2022+ Sales Policy."
        )
    context = f"Inventory results:\n{inventory_block}"
    reply = fallback
    return ChatResponse(reply=reply, intent=intent, vehicles=vehicles, rag_mode="sqlite")


def _handle_legacy_year_conflict(
    message: str,
    extracted: ExtractedIntent,
    rag: PolicyRAGService,
) -> ChatResponse:
    legacy = _search_inventory(extracted)
    alternatives = _sellable_alternatives(extracted)
    policy_ctx, _, rag_mode = _policy_context("2022+ sales policy pending de-listing", rag)

    legacy_block = _format_vehicle_list(legacy) if legacy else "No matching pre-2022 vehicles in stock records."
    alt_block = (
        _format_vehicle_list(alternatives)
        if alternatives
        else "No in-stock 2022+ alternatives matched — try broadening make/model."
    )

    fallback = (
        "Yes — we may still show pre-2022 vehicles in inventory records for transparency, "
        f"but they are classified as Pending De-listing and **cannot be sold or reserved**. "
        f"{POLICY_BLOCK_MESSAGE}\n\n"
        f"Matching pre-2022 inventory:\n{legacy_block}\n\n"
        f"Eligible 2022+ alternatives you can buy or reserve:\n{alt_block}"
    )
    context = (
        f"{policy_ctx}\n\nPre-2022 inventory:\n{legacy_block}\n\n"
        f"Sellable alternatives:\n{alt_block}"
    )
    reply = fallback
    return ChatResponse(
        reply=reply,
        intent=IntentKind.LEGACY_YEAR_CONFLICT,
        vehicles=legacy + alternatives,
        policy_context_used=bool(policy_ctx),
        rag_mode=rag_mode,
        blocked=True,
        block_reason=POLICY_BLOCK_MESSAGE,
    )


def _handle_hybrid(
    message: str,
    extracted: ExtractedIntent,
    rag: PolicyRAGService,
) -> ChatResponse:
    from backend.intent import is_legacy_year_focus

    if is_legacy_year_focus(extracted, message):
        return _handle_legacy_year_conflict(message, extracted, rag)

    vehicles = _search_inventory(extracted)
    policy_ctx, policy_used, rag_mode = _policy_context(message, rag)
    inventory_block = _format_vehicle_list(vehicles)
    fallback = (
        f"**Inventory (SQLite):**\n{inventory_block}\n\n"
        f"**Policies (RAG / {rag_mode}):**\n{policy_ctx or 'No policy section matched.'}"
    )
    context = (
        f"Structured inventory query results:\n{inventory_block}\n\n"
        f"Relevant policy excerpts:\n{policy_ctx or 'none'}"
    )
    reply = fallback
    return ChatResponse(
        reply=reply,
        intent=IntentKind.HYBRID_RAG,
        vehicles=vehicles,
        policy_context_used=policy_used,
        rag_mode=f"sqlite+{rag_mode}",
    )


def _handle_policy(message: str, rag: PolicyRAGService) -> ChatResponse:
    policy_ctx, used, rag_mode = _policy_context(message, rag)
    if not policy_ctx:
        fallback = (
            "I could not find a specific policy section for that. "
            "Ask about refunds, test drives, maintenance, shipping, or our 2022+ sales policy."
        )
        return ChatResponse(reply=fallback, intent=IntentKind.POLICY_QUESTION, rag_mode=rag_mode)
    fallback = f"From our policy documents:\n\n{policy_ctx}"
    reply = fallback
    return ChatResponse(
        reply=reply,
        intent=IntentKind.POLICY_QUESTION,
        policy_context_used=used,
        rag_mode=rag_mode,
    )


def handle_chat(request: ChatRequest, rag: PolicyRAGService | None = None) -> ChatResponse:
    service = rag if rag is not None else get_policy_rag_service()
    extracted = normalize_extracted_intent(
        classify_intent(request.message, request.user_email)
    )
    message = request.message

    if extracted.intent == IntentKind.HYBRID_RAG:
        return _handle_hybrid(message, extracted, service)

    if extracted.intent == IntentKind.LEGACY_YEAR_CONFLICT:
        return _handle_legacy_year_conflict(message, extracted, service)

    if extracted.intent == IntentKind.POLICY_QUESTION:
        return _handle_policy(message, service)

    if extracted.intent == IntentKind.INVENTORY_SEARCH:
        return _handle_inventory(message, extracted, service, IntentKind.INVENTORY_SEARCH)

    if extracted.intent == IntentKind.RESERVE_INTENT:
        if extracted.vehicle_id is None:
            return ChatResponse(
                reply="Please specify the vehicle id to reserve (e.g. reserve vehicle #16).",
                intent=extracted.intent,
            )
        vehicle = get_vehicle_by_id(extracted.vehicle_id)
        if vehicle is None:
            return ChatResponse(
                reply=f"Vehicle #{extracted.vehicle_id} was not found.",
                intent=extracted.intent,
            )
        if vehicle.pending_delisting:
            audit(
                "reserve",
                "blocked",
                vehicle_id=vehicle.id,
                detail=POLICY_BLOCK_MESSAGE,
            )
            return ChatResponse(
                reply=(
                    f"Vehicle #{vehicle.id} ({vehicle.year} {vehicle.make} {vehicle.model}) "
                    f"is in stock records but {POLICY_BLOCK_MESSAGE}"
                ),
                intent=extracted.intent,
                vehicles=[vehicle],
                blocked=True,
                block_reason=POLICY_BLOCK_MESSAGE,
            )
        try:
            reserved = reserve_vehicle(
                extracted.vehicle_id,
                idempotency_key=request.idempotency_key,
            )
        except OutOfStockError:
            audit("reserve", "blocked", vehicle_id=extracted.vehicle_id, detail="out_of_stock")
            return ChatResponse(
                reply=f"Vehicle #{extracted.vehicle_id} is out of stock and cannot be reserved.",
                intent=extracted.intent,
                vehicles=[vehicle],
                blocked=True,
                block_reason="out_of_stock",
            )
        audit("reserve", "success", vehicle_id=reserved.id)
        return ChatResponse(
            reply=(
                f"Reserved vehicle #{reserved.id}: {reserved.year} {reserved.make} "
                f"{reserved.model}. Remaining stock: {reserved.stock_count}."
            ),
            intent=extracted.intent,
            vehicles=[reserved],
            reserved_vehicle=reserved,
        )

    if extracted.intent == IntentKind.PURCHASE_INTENT:
        email = extracted.user_email
        if not email:
            return ChatResponse(
                reply="Please include your email so our sales team can follow up on your purchase.",
                intent=extracted.intent,
            )
        vehicle: Vehicle | None = None
        if extracted.vehicle_id is not None:
            vehicle = get_vehicle_by_id(extracted.vehicle_id)
            if vehicle is None:
                return ChatResponse(
                    reply=f"Vehicle #{extracted.vehicle_id} was not found.",
                    intent=extracted.intent,
                )
        elif extracted.make:
            matches = search_vehicles(
                VehicleSearchFilters(
                    make=extracted.make,
                    model=extracted.model,
                    year=extracted.year,
                    in_stock_only=True,
                    limit=1,
                )
            )
            sellable = [v for v in matches if not v.pending_delisting]
            vehicle = sellable[0] if sellable else None

        if vehicle and vehicle.pending_delisting:
            alts = _sellable_alternatives(extracted, limit=3)
            alt_text = _format_vehicle_list(alts) if alts else "Ask us to search 2022+ inventory."
            return ChatResponse(
                reply=(
                    f"We see a {vehicle.year} {vehicle.make} {vehicle.model} in records, but "
                    f"{POLICY_BLOCK_MESSAGE}\n\nEligible alternatives:\n{alt_text}"
                ),
                intent=extracted.intent,
                vehicles=[vehicle] + alts,
                blocked=True,
                block_reason=POLICY_BLOCK_MESSAGE,
            )
        if vehicle:
            try:
                assert_sellable(vehicle)
            except PolicyViolationError as exc:
                audit("purchase_email", "blocked", vehicle_id=vehicle.id, detail=str(exc))
                return ChatResponse(
                    reply=str(exc),
                    intent=extracted.intent,
                    vehicles=[vehicle],
                    blocked=True,
                    block_reason=str(exc),
                )
            if not _purchase_email_allowed(request, str(email), vehicle.id):
                return _purchase_replay_response(extracted.intent)
            email_result = send_purchase_email(str(email), vehicle)
            if email_result.sent:
                audit("purchase_email", "success", vehicle_id=vehicle.id, customer_email=str(email))
                suffix = " Our sales team has been notified by email."
            elif email_result.error == "resend_not_configured":
                suffix = " Purchase interest recorded (Resend API key not configured)."
            else:
                suffix = f" Purchase interest recorded (email failed: {email_result.error})."
            return ChatResponse(
                reply=(
                    f"Thank you. We received your purchase interest for "
                    f"{vehicle.year} {vehicle.make} {vehicle.model} ({email}).{suffix}"
                ),
                intent=extracted.intent,
                vehicles=[vehicle],
                email_sent=email_result.sent,
                email_error=email_result.error,
            )
        if not _purchase_email_allowed(request, str(email), None):
            return _purchase_replay_response(extracted.intent)
        email_result = send_purchase_inquiry_email(
            str(email),
            message,
            make=extracted.make,
            model=extracted.model,
        )
        if email_result.sent:
            audit("purchase_email", "success", customer_email=str(email), detail="inquiry")
            suffix = " Our sales team has been notified by email."
        elif email_result.error == "resend_not_configured":
            suffix = " Purchase interest recorded (Resend API key not configured)."
        else:
            suffix = f" Purchase interest recorded (email failed: {email_result.error})."
        return ChatResponse(
            reply=(
                f"Thank you, {email}. A sales specialist will contact you within one business day.{suffix} "
                "Specify a vehicle id or model for faster matching."
            ),
            intent=extracted.intent,
            email_sent=email_result.sent,
            email_error=email_result.error,
        )

    fallback = (
        "I'm your AI Car Concierge. Ask about inventory (e.g. Tesla under $70000), "
        "policies (refunds, test drives, shipping), hybrid questions (price + refund policy), "
        "or say reserve vehicle #16 / buy vehicle #20 with your@email.com."
    )
    return ChatResponse(reply=fallback, intent=IntentKind.GENERAL_CHAT)
