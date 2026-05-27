from __future__ import annotations

import re

from backend.conversation_state import ConversationState, DialoguePhase
from backend.database import Vehicle
from backend.gemini_service import generate_text
from backend.inventory_retrieval import infer_body_type

NLG_SYSTEM = (
    "You are a premium automotive advisor. Write calm, confident, consultative replies. "
    "Use ONLY the vehicle facts and customer context provided. "
    "Never invent prices, stock, features, or policies. "
    "Reason about fit for this customer, not generic metadata. "
    "Use memory callbacks like 'since you mentioned...'. "
    "Keep each reply 2-5 concise sentences and avoid repetitive phrasing. "
    "Do not use markdown bullet lists. Mention ids with #."
)


def _vehicle_fact_line(vehicle: Vehicle) -> str:
    stock = f"in stock ({vehicle.stock_count})" if vehicle.stock_count > 0 else "currently out of stock"
    body = infer_body_type(vehicle)
    return (
        f"#{vehicle.id}: {vehicle.year} {vehicle.make} {vehicle.model}, "
        f"{vehicle.color}, {vehicle.fuel_type}, ${vehicle.price:,.0f}, {body}, {stock}"
    )


def _pick(options: list[str], state: ConversationState, shift: int = 0) -> str:
    if not options:
        return ""
    idx = (state.turn_count + shift) % len(options)
    return options[idx]


def _memory_callback(state: ConversationState) -> str:
    callbacks: list[str] = []
    if state.use_case:
        callbacks.append(f"Since you mentioned {state.use_case},")
    if state.space_priority == "space":
        callbacks.append("Because cabin room matters more than efficiency for you,")
    if state.space_priority == "fuel":
        callbacks.append("Since efficiency matters for your use case,")
    if state.passengers and state.passengers >= 4:
        callbacks.append("For a family setup,")
    if callbacks:
        return callbacks[(state.turn_count - 1) % len(callbacks)]
    return "Based on what you've shared,"


def _is_city_focus(state: ConversationState) -> bool:
    return bool(state.use_case and "city" in state.use_case.lower())


def _is_family_focus(state: ConversationState) -> bool:
    return bool((state.passengers or 0) >= 4 or (state.use_case and "family" in state.use_case.lower()))


def _vehicle_reasoning(state: ConversationState, vehicle: Vehicle, rank: int) -> str:
    model = f"{vehicle.year} {vehicle.make} {vehicle.model}"
    body = infer_body_type(vehicle)
    reasons: list[str] = []
    if _is_family_focus(state) and body == "suv":
        reasons.append("you will feel the extra room immediately on family drives")
    if _is_city_focus(state) and "electric" in vehicle.fuel_type.lower():
        reasons.append("it stays smooth and quiet in city traffic")
    if state.space_priority == "space" and body == "suv":
        reasons.append("it gives you stronger cargo flexibility")
    if state.space_priority == "fuel" and any(
        token in vehicle.fuel_type.lower() for token in ("electric", "hybrid")
    ):
        reasons.append("it keeps running costs more predictable")
    if state.budget and vehicle.price <= state.budget:
        reasons.append("it stays comfortably inside your target range")
    if rank == 0:
        reasons.append("this is likely your strongest overall fit")
    core = reasons[0] if reasons else "it feels like a balanced fit for your profile"
    return f"I'd shortlist the {model} (#{vehicle.id}) first because {core}."


def _replace_disallowed_phrases(text: str) -> str:
    replacements = {
        "fits your budget": "stays within your target range",
        "premium option": "more upscale choice",
        "here are a few options i'd start with": "I would shortlist these first",
        "here are a few options i'd start with:": "I would shortlist these first:",
    }
    output = text
    for old, new in replacements.items():
        output = re.sub(re.escape(old), new, output, flags=re.IGNORECASE)
    return output


def _limit_vehicle_mentions(text: str, max_mentions: int = 3) -> str:
    ids = re.findall(r"#\d+", text)
    if len(ids) <= max_mentions:
        return text
    keep = set(ids[:max_mentions])
    lines = [line for line in text.splitlines() if not re.search(r"#\d+", line) or any(k in line for k in keep)]
    return "\n".join(lines).strip()


def polish_response(text: str) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    compact = _replace_disallowed_phrases(compact)
    compact = _limit_vehicle_mentions(compact, max_mentions=3)
    return compact


def _fallback_question(state: ConversationState, question: str) -> str:
    if state.turn_count <= 1:
        return f"Great to meet you. {question}"
    return question


def _fallback_recommendations(state: ConversationState, vehicles: list[Vehicle]) -> str:
    if not vehicles:
        budget_note = f" under ${state.budget:,.0f}" if state.budget else ""
        return (
            f"I couldn't find in-stock 2022+ matches{budget_note} with what we have so far. "
            "Would you like to adjust your budget or preferred body style?"
        )
    head = _memory_callback(state)
    opening = _pick(
        [
            "I'd personally shortlist these first.",
            "These feel like your best next options.",
            "I would start with these three.",
        ],
        state,
    )
    lines = [_vehicle_reasoning(state, vehicle, idx) for idx, vehicle in enumerate(vehicles[:3])]
    closer = _pick(
        [
            "Want me to walk you through the tradeoffs, or hold one now?",
            "I can compare the top two quickly, or reserve your top pick now.",
            "If you want, I can hold your best-fit option while we compare one alternative.",
        ],
        state,
        shift=1,
    )
    return f"{head} {opening}\n\n" + " ".join(lines) + f"\n\n{closer}"


def _fallback_comparison(state: ConversationState, vehicles: list[Vehicle]) -> str:
    if len(vehicles) < 2:
        return "Tell me which two vehicles you'd like compared (e.g. #16 vs #22)."
    ordered = sorted(vehicles, key=lambda v: v.price)
    best_value = ordered[0]
    roomiest = next(
        (v for v in vehicles if infer_body_type(v) == "suv" and v.id != best_value.id),
        next((v for v in vehicles if v.id != best_value.id), vehicles[0]),
    )
    city_friendly = next(
        (
            v
            for v in vehicles
            if v.id not in {best_value.id, roomiest.id}
            and ("electric" in v.fuel_type.lower() or "hybrid" in v.fuel_type.lower())
        ),
        next((v for v in vehicles if v.id not in {best_value.id, roomiest.id}), vehicles[-1]),
    )
    opener = _memory_callback(state)
    return (
        f"{opener} I'd frame it this way: the {best_value.make} {best_value.model} "
        f"(#{best_value.id}) is the value play at ${best_value.price:,.0f}. "
        f"The {roomiest.make} {roomiest.model} (#{roomiest.id}) is stronger on space and long-trip comfort. "
        f"The {city_friendly.make} {city_friendly.model} (#{city_friendly.id}) is the smoother daily choice if you are city-heavy. "
        "I would shortlist the first two unless city driving is your main priority."
    )


def _fallback_purchase_prompt(state: ConversationState, vehicle: Vehicle | None) -> str:
    if vehicle:
        return (
            f"Great choice. I can hold the {vehicle.year} {vehicle.make} {vehicle.model} (#{vehicle.id}) now. "
            "What is the best email for your confirmation?"
        )
    return "I can move this forward right away. What email should we use to send your next-step details?"


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
    return polish_response(text.strip()) if text else polish_response(_fallback_question(state, question))


def generate_recommendations(state: ConversationState, vehicles: list[Vehicle]) -> str:
    context = _build_context(
        state,
        vehicles,
        DialoguePhase.RECOMMENDING,
        "Recommend top 1-3 vehicles with personalized reasoning and premium consultative tone.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return polish_response(text.strip()) if text else polish_response(_fallback_recommendations(state, vehicles))


def generate_comparison(state: ConversationState, vehicles: list[Vehicle]) -> str:
    context = _build_context(
        state,
        vehicles,
        DialoguePhase.COMPARING,
        "Compare practical tradeoffs naturally: comfort, value, city-vs-highway, family fit.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return polish_response(text.strip()) if text else polish_response(_fallback_comparison(state, vehicles))


def generate_reserve_prompt(state: ConversationState, vehicle: Vehicle) -> str:
    context = _build_context(
        state,
        [vehicle],
        DialoguePhase.RESERVE,
        "Use calm premium sales language; offer to hold the vehicle now and ask for email if needed.",
    )
    text = generate_text(NLG_SYSTEM, context)
    if text:
        return polish_response(text.strip())
    return (
        f"This looks like your strongest fit so far. I can hold the {vehicle.year} {vehicle.make} "
        f"{vehicle.model} (#{vehicle.id}) for you now. What email should we send the confirmation to?"
    )


def generate_purchase_prompt(state: ConversationState, vehicle: Vehicle | None) -> str:
    context = _build_context(
        state,
        [vehicle] if vehicle else [],
        DialoguePhase.PURCHASE,
        "Ask for email to connect with sales.",
    )
    text = generate_text(NLG_SYSTEM, context)
    return polish_response(text.strip()) if text else polish_response(_fallback_purchase_prompt(state, vehicle))


def generate_welcome(state: ConversationState) -> str:
    context = _build_context(
        state,
        [],
        DialoguePhase.DISCOVERY,
        "Welcome the customer and ask how many people usually ride along.",
    )
    text = generate_text(NLG_SYSTEM, context)
    if text:
        return polish_response(text.strip())
    return polish_response(
        "Welcome. I can help you narrow this down quickly and confidently. "
        "To start, how many people usually ride with you?"
    )
