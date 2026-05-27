import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.chat_http import chat_http_status
from backend.codebase_packager import PackagerRequest, package_codebase
from backend.config import bootstrap, get_settings
from backend.job_broker import JobBroker, JobState
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
from backend.middleware import RequestContextMiddleware
from backend.output_validation import ValidationVerdict
from backend.orchestrator import (
    ChatRequest,
    ChatResponse,
    handle_chat,
    log_chat_outcome,
    record_chat_governor_result,
)
from backend.rag_service import PolicyRAGService, get_policy_rag_service, load_policy_chunks, search_policies
from backend.request_context import get_request_id
from backend.resilient_iteration_controller import ResilientIterationController
from backend.security import require_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
job_broker = JobBroker()
iteration_controller = ResilientIterationController("data/runtime_journal.jsonl")

def rate_limit_key(request: Request) -> str:
    api_key = (request.headers.get("X-API-Key") or "").strip()
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)


def get_rag_service() -> PolicyRAGService:
    return get_policy_rag_service()


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    bootstrap()
    settings = get_settings()
    cors_raw = settings.cors_origins or os.environ.get("CORS_ORIGINS", "*")
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

    application = FastAPI(title="AI Car Concierge", version="1.0.0", lifespan=lifespan)
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestContextMiddleware)

    def _error_content(message: str, **extra: object) -> dict:
        body: dict = {"error": message, "request_id": get_request_id()}
        body.update(extra)
        return body

    @application.exception_handler(PolicyViolationError)
    async def policy_violation_handler(_request: Request, exc: PolicyViolationError):
        return JSONResponse(
            status_code=409,
            content=_error_content(str(exc), vehicle_id=exc.vehicle_id),
        )

    @application.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict_handler(_request: Request, exc: IdempotencyConflictError):
        return JSONResponse(status_code=409, content=_error_content(str(exc)))

    @application.exception_handler(VehicleNotFoundError)
    async def vehicle_not_found_handler(_request: Request, exc: VehicleNotFoundError):
        return JSONResponse(status_code=404, content=_error_content(str(exc)))

    @application.exception_handler(OutOfStockError)
    async def out_of_stock_handler(_request: Request, exc: OutOfStockError):
        return JSONResponse(status_code=409, content=_error_content(str(exc)))

    @application.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_content("Invalid request", details=exc.errors()),
        )

    @application.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_content("Internal server error"),
        )

    def chat_rate_limit() -> str:
        return get_settings().chat_rate_limit

    @application.get("/")
    def root():
        return {
            "service": "AI Car Concierge API",
            "version": application.version,
            "openapi": "/openapi.json",
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
            "chat": "POST /api/chat",
            "ui_hint": "Deploy Streamlit with BACKEND_URL pointing here",
        }

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/ready")
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

    @application.get("/vehicles", response_model=list[Vehicle])
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

    @application.get("/vehicles/{vehicle_id}", response_model=Vehicle)
    def get_vehicle(vehicle_id: int, _: None = Depends(require_api_key)):
        vehicle = get_vehicle_by_id(vehicle_id)
        if vehicle is None:
            raise VehicleNotFoundError(f"Vehicle {vehicle_id} not found")
        return vehicle

    class ReserveResponse(BaseModel):
        vehicle: Vehicle
        message: str

    @application.post("/vehicles/{vehicle_id}/reserve", response_model=ReserveResponse)
    @limiter.limit("10/minute")
    def reserve(
        request: Request,
        vehicle_id: int,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_api_key),
    ):
        vehicle = reserve_vehicle(vehicle_id, idempotency_key=idempotency_key)
        return ReserveResponse(
            vehicle=vehicle,
            message=f"Reserved. Remaining stock: {vehicle.stock_count}.",
        )

    @application.post(
        "/api/chat",
        responses={
            200: {
                "description": "Success or legacy-year informational (check blocked in body)",
            },
            409: {"description": "Reserve or purchase blocked"},
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
        record_chat_governor_result(body, response)
        log_chat_outcome(response)
        if response.validation_verdict == ValidationVerdict.REJECT:
            content = response.model_dump(mode="json")
            content["request_id"] = get_request_id()
            return JSONResponse(status_code=422, content=content)
        status = chat_http_status(response)
        content = response.model_dump(mode="json")
        content["request_id"] = get_request_id()
        return JSONResponse(status_code=status, content=content)

    @application.get("/policies/search")
    def policies_search(
        q: str = Query(min_length=1),
        top_k: int = Query(default=3, ge=1, le=10),
        _: None = Depends(require_api_key),
    ):
        return search_policies(q, top_k=top_k)

    @application.post("/skills/codebase-packager")
    def skill_codebase_packager(
        request: PackagerRequest,
        _: None = Depends(require_api_key),
    ):
        return package_codebase(request)

    class JobSubmitRequest(BaseModel):
        kind: str
        payload: dict

    class JobSubmitResponse(BaseModel):
        job_id: str
        state: JobState

    def _run_packager_job(payload: dict, context) -> dict:
        packed = package_codebase(PackagerRequest.model_validate(payload))
        if context.is_cancelled():
            return {"cancelled": True}
        return packed.model_dump(mode="json")

    @application.post("/jobs/submit", response_model=JobSubmitResponse)
    def submit_job(
        body: JobSubmitRequest,
        _: None = Depends(require_api_key),
    ):
        if body.kind != "codebase_packager":
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported job kind", "request_id": get_request_id()},
            )
        job_id = job_broker.submit(
            handler=lambda context: _run_packager_job(body.payload, context),
            metadata={"kind": body.kind},
        )
        return JobSubmitResponse(job_id=job_id, state=JobState.PENDING)

    @application.get("/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(require_api_key)):
        record = job_broker.get(job_id)
        if record is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Job not found", "request_id": get_request_id()},
            )
        return record.model_dump(mode="json")

    @application.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, _: None = Depends(require_api_key)):
        cancelled = job_broker.cancel(job_id)
        if not cancelled:
            return JSONResponse(
                status_code=409,
                content={"error": "Job cannot be cancelled", "request_id": get_request_id()},
            )
        return {"job_id": job_id, "cancelled": True}

    class IterationRunRequest(BaseModel):
        run_id: str
        initial_state: dict[str, object] = Field(default_factory=dict)
        steps: list[str] = Field(default_factory=list)

    @application.post("/governor/iteration/run")
    def run_iteration(
        body: IterationRunRequest,
        _: None = Depends(require_api_key),
    ):
        handlers = []
        for name in body.steps:
            handlers.append((name, lambda state, step=name: {**state, "last_step": step}))
        result = iteration_controller.run(
            run_id=body.run_id,
            initial_state=body.initial_state,
            steps=handlers,
        )
        return result.model_dump(mode="json")

    return application


app = create_app()
