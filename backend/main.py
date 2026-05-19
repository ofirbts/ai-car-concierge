import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.chat_http import chat_http_status
from backend.config import bootstrap, get_settings
from backend.database import (
    IdempotencyConflictError,
    OutOfStockError,
    PolicyViolationError,
    Vehicle,
    VehicleNotFoundError,
    VehicleSearchFilters,
    VehicleSort,
    count_vehicles,
    get_vehicle_by_id,
    init_db,
    reserve_vehicle,
    search_vehicles,
)
from backend.orchestrator import ChatRequest, ChatResponse, handle_chat
from backend.rag_service import PolicyRAGService, get_policy_rag_service, load_policy_chunks, search_policies
from backend.security import require_api_key

bootstrap()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Car Concierge", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_service() -> PolicyRAGService:
    return get_policy_rag_service()


def chat_rate_limit() -> str:
    return get_settings().chat_rate_limit


@app.exception_handler(PolicyViolationError)
async def policy_violation_handler(_request: Request, exc: PolicyViolationError):
    return JSONResponse(
        status_code=409,
        content={"error": str(exc), "vehicle_id": exc.vehicle_id},
    )


@app.exception_handler(IdempotencyConflictError)
async def idempotency_conflict_handler(_request: Request, exc: IdempotencyConflictError):
    return JSONResponse(status_code=409, content={"error": str(exc)})


@app.exception_handler(VehicleNotFoundError)
async def vehicle_not_found_handler(_request: Request, exc: VehicleNotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(OutOfStockError)
async def out_of_stock_handler(_request: Request, exc: OutOfStockError):
    return JSONResponse(status_code=409, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Invalid request", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/")
def root():
    return {
        "service": "AI Car Concierge API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "chat": "POST /api/chat",
        "ui_hint": "Deploy Streamlit with BACKEND_URL pointing here",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        vehicles = count_vehicles()
        policies = len(load_policy_chunks())
        rag = get_policy_rag_service()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(exc)},
        )
    return {
        "status": "ready",
        "vehicles": vehicles,
        "policy_chunks": policies,
        "rag_mode": rag.retrieval_mode,
    }


@app.get("/vehicles", response_model=list[Vehicle])
def list_vehicles(
    make: str | None = None,
    model: str | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    color: str | None = None,
    fuel_type: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    in_stock_only: bool = False,
    limit: int = Query(default=20, ge=1, le=100),
    sort: VehicleSort = VehicleSort.YEAR_DESC_PRICE_ASC,
    _: None = Depends(require_api_key),
):
    filters = VehicleSearchFilters(
        make=make,
        model=model,
        year=year,
        year_min=year_min,
        year_max=year_max,
        color=color,
        fuel_type=fuel_type,
        price_min=price_min,
        price_max=price_max,
        in_stock_only=in_stock_only,
        limit=limit,
        sort=sort,
    )
    return search_vehicles(filters)


@app.get("/vehicles/{vehicle_id}", response_model=Vehicle)
def get_vehicle(vehicle_id: int, _: None = Depends(require_api_key)):
    vehicle = get_vehicle_by_id(vehicle_id)
    if vehicle is None:
        raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
    return vehicle


class ReserveResponse(BaseModel):
    vehicle: Vehicle
    message: str


@app.post("/vehicles/{vehicle_id}/reserve", response_model=ReserveResponse)
def reserve(
    vehicle_id: int,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    _: None = Depends(require_api_key),
):
    vehicle = reserve_vehicle(vehicle_id, idempotency_key=idempotency_key)
    return ReserveResponse(
        vehicle=vehicle,
        message=f"Reserved. Remaining stock: {vehicle.stock_count}.",
    )


@app.post(
    "/api/chat",
    responses={
        200: {
            "model": ChatResponse,
            "description": "Success or legacy-year informational (check blocked in body)",
        },
        409: {
            "model": ChatResponse,
            "description": "Reserve or purchase blocked",
        },
        401: {"description": "Missing or invalid X-API-Key when API_KEY is configured"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(chat_rate_limit)
def chat(
    request: Request,
    body: ChatRequest,
    rag: PolicyRAGService = Depends(get_rag_service),
    _: None = Depends(require_api_key),
):
    response = handle_chat(body, rag=rag)
    status = chat_http_status(response)
    return JSONResponse(status_code=status, content=response.model_dump(mode="json"))


@app.get("/policies/search")
def policies_search(
    q: str = Query(min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
    _: None = Depends(require_api_key),
):
    return search_policies(q, top_k=top_k)
