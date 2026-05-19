from __future__ import annotations

import html
import logging

import resend
from pydantic import BaseModel

from backend.config import get_settings
from backend.database import Vehicle

logger = logging.getLogger(__name__)


class EmailResult(BaseModel):
    sent: bool
    error: str | None = None


def _send_email(subject: str, html: str, customer_email: str) -> EmailResult:
    settings = get_settings()
    if not settings.resend_api_key.strip():
        return EmailResult(sent=False, error="resend_not_configured")

    resend.api_key = settings.resend_api_key
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
        logger.exception("Resend API failed")
        return EmailResult(sent=False, error=str(exc))


def send_purchase_email(customer_email: str, vehicle: Vehicle) -> EmailResult:
    safe_email = html.escape(customer_email)
    subject = f"Purchase interest: {vehicle.year} {vehicle.make} {vehicle.model}"
    html_body = (
        f"<p>Customer <b>{safe_email}</b> expressed purchase intent.</p>"
        f"<p>Vehicle #{vehicle.id}: {vehicle.year} {html.escape(vehicle.make)} "
        f"{html.escape(vehicle.model)} ({html.escape(vehicle.color)}) — "
        f"${vehicle.price:,.0f}</p>"
    )
    return _send_email(subject, html_body, customer_email)


def send_purchase_inquiry_email(
    customer_email: str,
    message: str,
    *,
    make: str | None = None,
    model: str | None = None,
) -> EmailResult:
    details = []
    if make:
        details.append(f"Make: {html.escape(make)}")
    if model:
        details.append(f"Model: {html.escape(model)}")
    detail_block = "<br>".join(details) if details else "No specific vehicle selected."
    safe_message = html.escape(message[:2000])
    safe_email = html.escape(customer_email)
    subject = "Purchase inquiry — general interest"
    html_body = (
        f"<p>Customer <b>{safe_email}</b> expressed purchase interest.</p>"
        f"<p>{detail_block}</p>"
        f"<p>Original message:</p><blockquote>{safe_message}</blockquote>"
    )
    return _send_email(subject, html_body, customer_email)
