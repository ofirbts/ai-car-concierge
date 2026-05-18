from __future__ import annotations

import logging

import resend
from pydantic import BaseModel

from backend.config import get_settings
from backend.database import Vehicle

logger = logging.getLogger(__name__)


class EmailResult(BaseModel):
    sent: bool
    error: str | None = None


def send_purchase_email(customer_email: str, vehicle: Vehicle) -> EmailResult:
    settings = get_settings()
    if not settings.resend_api_key.strip():
        return EmailResult(sent=False, error="resend_not_configured")

    resend.api_key = settings.resend_api_key
    subject = f"Purchase interest: {vehicle.year} {vehicle.make} {vehicle.model}"
    html = (
        f"<p>Customer <b>{customer_email}</b> expressed purchase intent.</p>"
        f"<p>Vehicle #{vehicle.id}: {vehicle.year} {vehicle.make} {vehicle.model} "
        f"({vehicle.color}) — ${vehicle.price:,.0f}</p>"
    )
    try:
        resend.Emails.send(
            {
                "from": settings.resend_from_email,
                "to": [settings.resend_to_email],
                "subject": subject,
                "html": html,
                "reply_to": customer_email,
            }
        )
        return EmailResult(sent=True)
    except Exception as exc:
        logger.exception("Resend API failed for purchase email")
        return EmailResult(sent=False, error=str(exc))
