from __future__ import annotations

import math
import re
from functools import lru_cache

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.conversation_state import ConversationState
from backend.database import SALES_MIN_YEAR, Vehicle, VehicleSearchFilters, search_vehicles
from backend.gemini_service import embed_query, embed_texts
from backend.intent import ExtractedIntent, extract_price_max

SUV_MODELS = (
    "x5",
    "q7",
    "gle",
    "model y",
    "xc90",
    "cayenne",
    "f-pace",
    "escalade",
    "navigator",
    "range rover",
    "rx",
    "gv70",
)
SEDAN_MODELS = ("a4", "model 3", "3 series", "c-class", "s60", "es", "g80")
SPORTS_MODELS = ("911",)
FAMILY_BOOST_MODELS = (
    "navigator",
    "xc90",
    "escalade",
    "x5",
    "q7",
    "gle",
    "rx",
    "gv70",
    "range rover",
    "model y",
)
FAMILY_PENALIZE_MODELS = ("911",)
FAMILY_SOFT_PENALIZE_MODELS = ("cayenne",)

SEMANTIC_PROFILES: dict[str, dict[str, object]] = {
    "family": {
        "text": "spacious family SUV comfortable seating cargo room kids",
        "body_types": ("suv",),
        "min_passengers": 4,
    },
    "city": {
        "text": "compact efficient city driving easy parking fuel saving",
        "fuel_types": ("electric", "hybrid", "plug-in hybrid"),
        "price_tier": "mid",
    },
    "budget": {
        "text": "affordable value economical low cost practical",
        "price_tier": "low",
    },
    "highway": {
        "text": "comfortable long distance highway cruising quiet stable",
        "body_types": ("suv", "sedan"),
    },
    "couple": {
        "text": "couple two people sedan comfortable stylish",
        "body_types": ("sedan",),
        "min_passengers": 2,
    },
    "baby": {
        "text": "family baby stroller space safe SUV roomy",
        "body_types": ("suv",),
        "min_passengers": 3,
    },
    "commute": {
        "text": "daily commute reliable efficient electric hybrid",
        "fuel_types": ("electric", "hybrid", "plug-in hybrid"),
    },
    "luxury": {
        "text": "premium luxury upscale comfort prestige",
        "price_tier": "high",
    },
    "suv_affordable": {
        "text": "SUV spacious not expensive value family",
        "body_types": ("suv",),
        "price_tier": "mid",
    },
}

NATURAL_QUERY_HINTS: tuple[tuple[str, str], ...] = (
    (r"\bfamily\b|\bkids?\b|\bchildren\b|\bילד", "family"),
    (r"\bcity\b|\burban\b|\bעיר\b|\bחסכונ", "city"),
    (r"\bcheap\b|\bafford\b|\bbudget\b|\bnot expensive\b|\bזול\b|\bתקציב", "budget"),
    (r"\bhighway\b|\blong drive\b|\bcommute\b|\bנסיע", "highway"),
    (r"\bcouple\b|\btwo people\b|\bזוג\b", "couple"),
    (r"\bbaby\b|\binfant\b|\bstroller\b|\bתינוק", "baby"),
    (r"\bsuv\b", "suv_affordable"),
    (r"\belectric\b|\bev\b|\bחשמל", "commute"),
    (r"\bluxury\b|\bpremium\b", "luxury"),
)


class InventoryRetrievalResult(BaseModel):
    vehicles: list[Vehicle] = Field(default_factory=list)
    retrieval_mode: str = "sql"
    matched_profiles: list[str] = Field(default_factory=list)
    query: str = ""


def infer_body_type(vehicle: Vehicle) -> str:
    model_lower = vehicle.model.lower()
    if any(token in model_lower for token in SUV_MODELS):
        return "suv"
    if any(token in model_lower for token in SPORTS_MODELS):
        return "sports"
    if any(token in model_lower for token in SEDAN_MODELS):
        return "sedan"
    if vehicle.price >= 75000:
        return "suv"
    return "sedan"


def vehicle_profile_text(vehicle: Vehicle) -> str:
    body = infer_body_type(vehicle)
    tier = "low" if vehicle.price < 60000 else "mid" if vehicle.price < 80000 else "high"
    stock = "in stock" if vehicle.stock_count > 0 else "out of stock"
    return (
        f"{vehicle.year} {vehicle.make} {vehicle.model} {body} "
        f"{vehicle.fuel_type} ${vehicle.price:.0f} {vehicle.color} {stock} "
        f"family city commute highway luxury spacious efficient affordable"
    )


@lru_cache(maxsize=1)
def _all_sellable_vehicles_cached(db_path: str) -> tuple[Vehicle, ...]:
    vehicles = search_vehicles(
        VehicleSearchFilters(year_min=SALES_MIN_YEAR, in_stock_only=True, limit=100)
    )
    return tuple(v for v in vehicles if not v.pending_delisting)


def _all_sellable_vehicles() -> list[Vehicle]:
    from backend.database import get_db_path

    return list(_all_sellable_vehicles_cached(str(get_db_path())))


def clear_inventory_retrieval_cache() -> None:
    _all_sellable_vehicles_cached.cache_clear()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def detect_semantic_profiles(message: str) -> list[str]:
    lower = message.lower()
    matched: list[str] = []
    for pattern, profile in NATURAL_QUERY_HINTS:
        if re.search(pattern, lower):
            matched.append(profile)
    return matched


def _price_tier(price: float) -> str:
    if price < 60000:
        return "low"
    if price < 80000:
        return "mid"
    return "high"


def _profile_filter_score(vehicle: Vehicle, profiles: list[str]) -> float:
    if not profiles:
        return 0.0
    body = infer_body_type(vehicle)
    tier = _price_tier(vehicle.price)
    fuel = vehicle.fuel_type.lower()
    score = 0.0
    for name in profiles:
        profile = SEMANTIC_PROFILES.get(name, {})
        body_types = profile.get("body_types")
        if body_types and body in body_types:
            score += 2.0
        fuel_types = profile.get("fuel_types")
        if fuel_types and any(ft in fuel for ft in fuel_types):
            score += 1.5
        price_tier = profile.get("price_tier")
        if price_tier == tier:
            score += 1.0
        elif price_tier == "low" and tier == "low":
            score += 1.5
        elif price_tier == "mid" and tier in ("low", "mid"):
            score += 0.5
    return score


def _keyword_score(query: str, vehicle: Vehicle) -> float:
    tokens = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    if not tokens:
        return 0.0
    text = vehicle_profile_text(vehicle).lower()
    overlap = len(tokens & set(text.split()))
    if vehicle.make.lower() in query.lower():
        overlap += 3
    if vehicle.model.lower() in query.lower():
        overlap += 2
    return float(overlap)


def state_to_filters(state: ConversationState) -> VehicleSearchFilters:
    price_max = state.budget
    fuel_type = state.fuel_preference
    if state.space_priority == "fuel" and not fuel_type:
        fuel_type = "Hybrid"
    filters = VehicleSearchFilters(
        year_min=SALES_MIN_YEAR,
        price_max=price_max,
        fuel_type=fuel_type,
        in_stock_only=True,
        limit=40,
    )
    return filters


def is_family_shopping(state: ConversationState | None) -> bool:
    if state is None:
        return False
    if state.use_case and "family" in state.use_case.lower():
        return True
    if (state.passengers or 0) >= 4:
        return True
    if (state.family_size or 0) >= 4:
        return True
    return False


def family_fit_score(vehicle: Vehicle, state: ConversationState | None) -> float:
    if not is_family_shopping(state):
        return 0.0
    model_lower = vehicle.model.lower()
    make_lower = vehicle.make.lower()
    score = 0.0
    if any(token in model_lower for token in FAMILY_PENALIZE_MODELS):
        score -= 3.0
    if any(token in model_lower for token in FAMILY_SOFT_PENALIZE_MODELS):
        score -= 1.5
    if any(token in model_lower for token in FAMILY_BOOST_MODELS):
        score += 2.0
    if infer_body_type(vehicle) == "suv" and make_lower in (
        "lincoln",
        "volvo",
        "cadillac",
        "lexus",
        "land rover",
    ):
        score += 1.0
    if state and state.space_priority == "space" and infer_body_type(vehicle) == "suv":
        if "navigator" in model_lower or "escalade" in model_lower or "xc90" in model_lower:
            score += 1.5
    return score


def body_type_filter(vehicles: list[Vehicle], body_type: str | None) -> list[Vehicle]:
    if not body_type:
        return vehicles
    target = body_type.lower()
    return [v for v in vehicles if infer_body_type(v) == target]


def hybrid_search_inventory(
    query: str,
    state: ConversationState | None = None,
    extracted: ExtractedIntent | None = None,
    limit: int = 4,
) -> InventoryRetrievalResult:
    profiles = detect_semantic_profiles(query)
    if state and state.use_case:
        profiles.extend(detect_semantic_profiles(state.use_case))
    profiles = list(dict.fromkeys(profiles))

    base_filters = state_to_filters(state) if state else VehicleSearchFilters(
        year_min=SALES_MIN_YEAR,
        in_stock_only=True,
        limit=40,
    )
    if extracted:
        if extracted.make:
            base_filters.make = extracted.make
        if extracted.model:
            base_filters.model = extracted.model
        if extracted.price_max is not None:
            base_filters.price_max = extracted.price_max
        elif not base_filters.price_max:
            inferred = extract_price_max(query)
            if inferred is not None:
                base_filters.price_max = inferred

    if state and state.body_type and not base_filters.make:
        pass

    candidates = search_vehicles(base_filters)
    candidates = [v for v in candidates if not v.pending_delisting]
    if state and state.body_type:
        filtered = body_type_filter(candidates, state.body_type)
        if filtered:
            candidates = filtered
    elif profiles:
        body_hint = None
        for name in profiles:
            hint = SEMANTIC_PROFILES.get(name, {}).get("body_types")
            if hint:
                body_hint = hint[0]
                break
        if body_hint:
            filtered = body_type_filter(candidates, str(body_hint))
            if filtered:
                candidates = filtered

    if not candidates:
        candidates = _all_sellable_vehicles()[:40]

    mode = "sql+keyword"
    ranked: list[tuple[float, Vehicle]] = []
    use_embeddings = get_settings().has_google_api()
    query_text = query
    if state:
        parts = [query]
        if state.use_case:
            parts.append(state.use_case)
        if state.body_type:
            parts.append(state.body_type)
        if state.fuel_preference:
            parts.append(state.fuel_preference)
        query_text = " ".join(parts)

    if use_embeddings and candidates:
        texts = [vehicle_profile_text(v) for v in candidates]
        vectors = embed_texts(texts)
        query_vec = embed_query(query_text)
        if vectors and query_vec and len(vectors) == len(candidates):
            mode = "hybrid_semantic"
            for vehicle, vec in zip(candidates, vectors):
                semantic = _cosine_similarity(query_vec, vec)
                profile_score = _profile_filter_score(vehicle, profiles) * 0.15
                family_score = family_fit_score(vehicle, state) * 0.12
                ranked.append((semantic + profile_score + family_score, vehicle))
        else:
            use_embeddings = False

    if not ranked:
        for vehicle in candidates:
            score = _keyword_score(query_text, vehicle)
            score += _profile_filter_score(vehicle, profiles) * 0.2
            score += family_fit_score(vehicle, state)
            if state and state.budget and vehicle.price <= state.budget:
                score += 1.0
            ranked.append((score, vehicle))

    ranked.sort(key=lambda item: (-item[0], item[1].price))
    top = [v for _, v in ranked[:limit]]
    return InventoryRetrievalResult(
        vehicles=top,
        retrieval_mode=mode,
        matched_profiles=profiles,
        query=query_text,
    )
