from backend.database import record_audit
from backend.request_context import get_request_id


def audit(
    action: str,
    outcome: str,
    *,
    vehicle_id: int | None = None,
    customer_email: str | None = None,
    detail: str | None = None,
) -> None:
    record_audit(
        get_request_id(),
        action,
        outcome,
        vehicle_id=vehicle_id,
        customer_email=customer_email,
        detail=detail,
    )
