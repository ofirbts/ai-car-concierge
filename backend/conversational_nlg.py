from __future__ import annotations

from backend.conversation_state import ConversationState, DialoguePhase
from backend.database import Vehicle
from backend.gemini_service import generate_text
from backend.inventory_retrieval import infer_body_type

NLG_SYSTEM = (
    "You are a friendly car sales concierge. Write natural, warm, concise replies. "
    "Use ONLY the vehicle facts and customer context provided. "
    "Never invent prices, stock, features, or policies. "
    "If recommending, explain briefly why each car fits the customer. "
    "Do not use markdown bullet lists; write flowing conversational prose. "
    "Mention vehicle id with # when referencing a specific car."
)


def _vehicle_fact_line(vehicle: Vehicle) -> str:
    stock = f"in stock ({vehicle.stock_count})" if vehicle.stock_count > 0 else "currently out of stock"
    body = infer_body_type(vehicle)
    return (
        f"#{vehicle.id}: {vehicle.year} {vehicle.make} {vehicle.model}, "
        f"{vehicle.color}, {vehicle.fuel_type}, ${vehicle.price:,.0f}, {body}, {stock}"
    )


def _fallback_question(state: ConversationState, question: str) -> str:
    if state.turn_count <= 1:
        return f"Happy to help you find the right car. {question}"
    return question


def _fallback_recommendations(state: ConversationState, vehicles: list[Vehicle]) -> str:
    if not vehicles:
        budget_note = f" under ${state.budget:,.0f}" if state.budget else ""
        return (
            f"I couldn't find in-stock 2022+ matches{budget_note} with what we have so far. "
            "Would you like to adjust your budget or preferred body style?"
        )
    opener = "Based on what you've shared, here are a few options I'd start with:"
    if state.use_case:
        opener = f"Given you're looking for something for {state.use_case}, {opener.lower()}"
    lines = []
    for vehicle in vehicles[:4]:
        why_parts = []
        body = infer_body_type(vehicle)
        if state.body_type and body == state.body_type:
            why_parts.append(f"it's a {body} as you wanted")
        if state.fuel_preference and state.fuel_preference.lower() in vehicle.fuel_type.lower():
            why_parts.append(f"it runs on {vehicle.fuel_type.lower()}")
        if state.budget and vehicle.price <= state.budget:
            why_parts.append("it fits your budget")
        if state.passengers and state.passengers >= 4 and body == "suv":
            why_parts.append("it has the space your group needs")
        why = ", ".join(why_parts) if why_parts else f"it's a solid {body} option"
        lines.append(
            f"I'd look at the {vehicle.year} {vehicle.make} {vehicle.model} "
            f"(#{vehicle.id}) at ${vehicle.price:,.0f} — {why}."
        )
    follow = "Want me to compare any of these, or should we reserve one?"
    return f"{opener}\n\n" + " ".join(lines) + f"\n\n{follow}"


def _fallback_comparison(vehicles: list[Vehicle]) -> str:
    if len(vehicles) < 2:
        return "Tell me which two vehicles you'd like compared (e.g. #16 vs #22)."
    parts = []
    cheapest = min(vehicles, key=lambda v: v.price)
    for vehicle in vehicles:
        body = infer_body_type(vehicle)
        value = "best value" if vehicle.id == cheapest.id else "premium option"
        parts.append(
            f"The {vehicle.make} {vehicle.model} (#{vehicle.id}) at ${vehicle.price:,.0f} "
            f"is a {body} — {value}, {vehicle.fuel_type}, stock {vehicle.stock_count}."
        )
    return " ".join(parts) + " Which matters more to you — lowest price or extra space?"


def _fallback_purchase_prompt(state: ConversationState, vehicle: Vehicle | None) -> str:
    if vehicle:
        return (
            f"Great choice — the {vehicle.year} {vehicle.make} {vehicle.model} (#{vehicle.id}) "
            f"is ${vehicle.price:,.0f}. What email should our sales team use to follow up?"
        )
    return "What email should our sales team use to follow up on your purchase?"


def _build_context(
    state: ConversationState,
    vehicles: list[Vehicle],
    phase: DialoguePhase,
    extra: str = "",
) -> str:
    slots = state.filled_slots()
    lines = [f"Phase: {phase.value}", f"Customer context: {slots or 'minimal'}"]
    if vehicles:
        lines.append("Vehicle facts:")
        lines.extend(_vehicle_fact_line(v) for v in vehicles)
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def generate_clarifying_question(state: ConversationState, question: str) -> str:
    context = _build_context(state, [], DialoguePhase.DISCOVERY, f"Ask exactly: {question}")
    text = generate_text(NLG_SYSTEM, context)
    return text.strip() if text else _fallback_question(state, question)


def generate_recommendations(state: ConversationState, vehicles: list[Vehicle]) -> str:
    context = _build_context(
        state,
        vehicles,
        DialoguePhase.RECOMMENDING,
        "Recommend 2-4 vehicles with brief personalized reasons.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return text.strip() if text else _fallback_recommendations(state, vehicles)


def generate_comparison(state: ConversationState, vehicles: list[Vehicle]) -> str:
    context = _build_context(
        state,
        vehicles,
        DialoguePhase.COMPARING,
        "Compare tradeoffs: price, space, fuel type. Keep it balanced.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return text.strip() if text else _fallback_comparison(vehicles)


def generate_reserve_prompt(state: ConversationState, vehicle: Vehicle) -> str:
    context = _build_context(
        state,
        [vehicle],
        DialoguePhase.RESERVE,
        "Confirm they want to reserve this vehicle and ask for email if missing.",
    )
    text = generate_text(NLG_SYSTEM, context)
    if text:
        return text.strip()
    return (
        f"I can reserve the {vehicle.year} {vehicle.make} {vehicle.model} (#{vehicle.id}) for you. "
        "What email should we send the confirmation to?"
    )


def generate_purchase_prompt(state: ConversationState, vehicle: Vehicle | None) -> str:
    context = _build_context(
        state,
        [vehicle] if vehicle else [],
        DialoguePhase.PURCHASE,
        "Ask for email to connect with sales.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return text.strip() if text else _fallback_purchase_prompt(state, vehicle)


def generate_welcome(state: ConversationState) -> str:
    context = _build_context(
        state,
        [],
        DialoguePhase.DISCOVERY,
        "Welcome the customer and ask how many people usually ride along.",
    )
    text = generate_text(NLG_SYSTEM, context)
    if text:
        return text.strip()
    return (
        "I'm your AI Car Concierge — I'll help you find and buy the right car step by step. "
        "To start, how many people usually ride along with you?"
    )
