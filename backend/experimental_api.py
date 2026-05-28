from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.codebase_packager import PackagerRequest, package_codebase
from backend.job_broker import JobBroker, JobState
from backend.request_context import get_request_id
from backend.resilient_iteration_controller import ResilientIterationController
from backend.security import require_api_key


class JobSubmitRequest(BaseModel):
    kind: str
    payload: dict


class JobSubmitResponse(BaseModel):
    job_id: str
    state: JobState


class IterationRunRequest(BaseModel):
    run_id: str
    initial_state: dict[str, object] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)


def register_experimental_routes(
    application: FastAPI,
    *,
    job_broker: JobBroker,
    iteration_controller: ResilientIterationController,
) -> None:
    def _run_packager_job(payload: dict, context) -> dict:
        packed = package_codebase(PackagerRequest.model_validate(payload))
        if context.is_cancelled():
            return {"cancelled": True}
        return packed.model_dump(mode="json")

    @application.post("/skills/codebase-packager")
    def skill_codebase_packager(
        request: PackagerRequest,
        _: None = Depends(require_api_key),
    ):
        return package_codebase(request)

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
