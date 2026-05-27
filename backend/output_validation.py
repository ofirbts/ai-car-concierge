from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.database import Vehicle
from backend.grounding import reply_prices_grounded
from backend.intent import IntentKind


class FindingSeverity(str, Enum):
    BLOCKER = "BLOCKER"
    RISK = "RISK"
    PREFERENCE = "PREFERENCE"
    CORRECTLY_IDENTIFIED = "CORRECTLY_IDENTIFIED"


class ValidationVerdict(str, Enum):
    PASS = "PASS"
    CALIBRATE = "CALIBRATE"
    REJECT = "REJECT"


class ValidationFinding(BaseModel):
    finding: str
    severity: FindingSeverity
    fix: str | None = None


class ValidationReport(BaseModel):
    findings: list[ValidationFinding] = Field(default_factory=list)
    verdict: ValidationVerdict
    profile: str = "normal"


class ResponseLike(BaseModel):
    reply: str
    intent: IntentKind
    policy_context_used: bool = False
    blocked: bool = False
    reserved_vehicle: Vehicle | None = None
    email_sent: bool = False
    vehicles: list[Vehicle] = Field(default_factory=list)


def _score(reply: str) -> tuple[int, int]:
    blockers = 0
    risks = 0
    if not reply.strip():
        blockers += 1
    if len(reply.strip()) < 12:
        risks += 1
    return blockers, risks


def validate_response_quality(response: ResponseLike) -> ValidationReport:
    profile = (get_settings().validation_profile or "normal").strip().lower()
    if profile not in {"strict", "normal", "relaxed"}:
        profile = "normal"
    findings: list[ValidationFinding] = []
    blockers, risks = _score(response.reply)

    if blockers:
        findings.append(
            ValidationFinding(
                finding="Reply content is empty",
                severity=FindingSeverity.BLOCKER,
                fix="Generate a concrete user-facing response",
            )
        )

    if response.intent == IntentKind.POLICY_QUESTION and not response.policy_context_used:
        findings.append(
            ValidationFinding(
                finding="Policy answer missing policy context",
                severity=FindingSeverity.RISK,
                fix="Attach policy evidence block to response",
            )
        )
        risks += 1

    if response.intent == IntentKind.RESERVE_INTENT and not response.blocked and response.reserved_vehicle is None:
        findings.append(
            ValidationFinding(
                finding="Reserve flow finished without reservation result",
                severity=FindingSeverity.BLOCKER,
                fix="Return reserved vehicle state or explicit block reason",
            )
        )
        blockers += 1

    if response.intent == IntentKind.PURCHASE_INTENT and not response.blocked and not response.email_sent:
        findings.append(
            ValidationFinding(
                finding="Purchase flow did not send email",
                severity=FindingSeverity.RISK,
                fix="Retry notification dispatch with fallback path",
            )
        )
        risks += 1

    grounding_vehicles: list[Vehicle] = list(response.vehicles)
    if response.reserved_vehicle is not None and response.reserved_vehicle not in grounding_vehicles:
        grounding_vehicles.append(response.reserved_vehicle)
    if grounding_vehicles and not reply_prices_grounded(
        response.reply,
        grounding_vehicles,
        reserved_vehicle=response.reserved_vehicle,
    ):
        findings.append(
            ValidationFinding(
                finding="Prices in reply are not grounded in vehicle facts",
                severity=FindingSeverity.BLOCKER,
                fix="Regenerate with template fallback or restrict to known prices",
            )
        )
        blockers += 1

    calibrate_threshold = 3
    if profile == "strict":
        calibrate_threshold = 2
    elif profile == "relaxed":
        calibrate_threshold = 4

    if blockers > 0:
        verdict = ValidationVerdict.REJECT
    elif risks >= calibrate_threshold:
        verdict = ValidationVerdict.CALIBRATE
    else:
        verdict = ValidationVerdict.PASS

    if verdict == ValidationVerdict.PASS:
        findings.append(
            ValidationFinding(
                finding="Response passed validation rubric",
                severity=FindingSeverity.CORRECTLY_IDENTIFIED,
            )
        )

    return ValidationReport(findings=findings, verdict=verdict, profile=profile)

