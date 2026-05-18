from unittest.mock import MagicMock, patch

from backend.automations import send_purchase_email
from backend.database import Vehicle


def _vehicle() -> Vehicle:
    return Vehicle(
        id=16,
        make="Tesla",
        model="Model 3",
        year=2025,
        color="Black",
        price=65000,
        fuel_type="Electric",
        stock_count=2,
    )


@patch("backend.automations.get_settings")
def test_send_purchase_email_skips_without_key(mock_settings):
    mock_settings.return_value.resend_api_key = ""
    result = send_purchase_email("a@b.com", _vehicle())
    assert result.sent is False
    assert result.error == "resend_not_configured"


@patch("backend.automations.resend.Emails.send")
@patch("backend.automations.get_settings")
def test_send_purchase_email_success(mock_settings, mock_send):
    mock_settings.return_value.resend_api_key = "re_test"
    mock_settings.return_value.resend_from_email = "from@test.com"
    mock_settings.return_value.resend_to_email = "sales@test.com"
    result = send_purchase_email("buyer@test.com", _vehicle())
    assert result.sent is True
    assert result.error is None
    mock_send.assert_called_once()


@patch("backend.automations.resend.Emails.send", side_effect=RuntimeError("429 rate limit"))
@patch("backend.automations.get_settings")
def test_send_purchase_email_handles_api_error(mock_settings, _mock_send):
    mock_settings.return_value.resend_api_key = "re_test"
    mock_settings.return_value.resend_from_email = "from@test.com"
    mock_settings.return_value.resend_to_email = "sales@test.com"
    result = send_purchase_email("buyer@test.com", _vehicle())
    assert result.sent is False
    assert "429" in (result.error or "")
