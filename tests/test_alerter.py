"""Tests for alerter — Discord webhook alerting for the autonomous orchestrator.

The alerter has two layers:
  1. format_payload() — pure function, builds the Discord webhook JSON
  2. send_alert()     — async wrapper, performs the POST (mocked in tests)
"""

import pytest

from alerter import (
    Severity,
    format_payload,
    tool_alert_human,
)


class TestSeverity:
    def test_known_levels(self):
        """The three planned severity tiers must exist with stable string values."""
        assert Severity.INFO.value == "info"
        assert Severity.ALERT.value == "alert"
        assert Severity.PANIC.value == "panic"


class TestFormatPayload:
    def test_info_has_no_mention(self):
        """INFO is silent — log-style, no @mention."""
        p = format_payload(Severity.INFO, "Target switched to NGC 7000")
        assert "<@" not in p["content"]
        assert "@everyone" not in p["content"]
        assert "Target switched to NGC 7000" in p["content"]

    def test_alert_mentions_user_when_id_given(self):
        """ALERT @mentions the configured user — they should look at this."""
        p = format_payload(Severity.ALERT, "Autofocus did not converge", user_id="123456789")
        assert "<@123456789>" in p["content"]
        assert "Autofocus did not converge" in p["content"]

    def test_alert_without_user_id_falls_back_to_plain(self):
        """If no user_id is configured, ALERT still sends but without mention."""
        p = format_payload(Severity.ALERT, "Autofocus did not converge")
        assert "<@" not in p["content"]
        assert "Autofocus did not converge" in p["content"]

    def test_panic_uses_everyone_and_siren(self):
        """PANIC wakes the human up — @everyone + 🚨 prefix."""
        p = format_payload(Severity.PANIC, "Safety supervisor aborted session")
        assert "@everyone" in p["content"]
        assert "🚨" in p["content"]
        assert "Safety supervisor aborted session" in p["content"]

    def test_payload_includes_severity_label(self):
        """Severity is visible in the message body so logs in Discord are scannable."""
        p = format_payload(Severity.ALERT, "x")
        # Anything like [ALERT] or **ALERT** is fine — just assert it's there.
        assert "ALERT" in p["content"]

    def test_payload_truncates_overly_long_messages(self):
        """Discord webhook content max is 2000 chars; we must never exceed."""
        long_msg = "x" * 5000
        p = format_payload(Severity.INFO, long_msg)
        assert len(p["content"]) <= 2000

    def test_username_is_set_for_branding(self):
        """The webhook posts as 'NINA Autopilot' so users can identify the source."""
        p = format_payload(Severity.INFO, "hi")
        assert p["username"] == "NINA Autopilot"


class TestToolEnvelope:
    """tool_alert_human() is the entry point the @mcp.tool() wrapper calls.
    It uses dependency injection (http_post) so tests don't hit the network."""

    @pytest.mark.asyncio
    async def test_success_envelope(self):
        captured = {}

        async def fake_post(url, json_payload, image_path=None):
            captured["url"] = url
            captured["payload"] = json_payload
            captured["image_path"] = image_path
            return 204, ""

        env = await tool_alert_human(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            severity="info",
            message="Session started",
            http_post=fake_post,
        )
        assert env["Success"] is True
        assert env["Type"] == "ALERTER"
        assert captured["url"].startswith("https://discord.com/api/webhooks/")
        assert "Session started" in captured["payload"]["content"]
        assert captured["image_path"] is None

    @pytest.mark.asyncio
    async def test_image_attachment_threaded_through(self, tmp_path):
        img = tmp_path / "sub.jpg"
        img.write_bytes(b"fake-jpeg")
        captured = {}

        async def fake_post(url, json_payload, image_path=None):
            captured["image_path"] = image_path
            return 204, ""

        await tool_alert_human(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            severity="alert",
            message="Bad guiding",
            attach_image_path=str(img),
            http_post=fake_post,
        )
        assert captured["image_path"] == str(img)

    @pytest.mark.asyncio
    async def test_invalid_severity_returns_failure(self):
        env = await tool_alert_human(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            severity="screaming",  # not a valid level
            message="x",
            http_post=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")),
        )
        assert env["Success"] is False
        assert env["ErrorType"] == "AlerterError"

    @pytest.mark.asyncio
    async def test_missing_webhook_url_returns_failure(self):
        env = await tool_alert_human(
            webhook_url="",
            severity="info",
            message="x",
            http_post=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")),
        )
        assert env["Success"] is False
        assert env["ErrorType"] == "AlerterError"

    @pytest.mark.asyncio
    async def test_http_failure_returns_failure_envelope(self):
        async def fake_post(url, json_payload, image_path=None):
            return 500, "internal server error"

        env = await tool_alert_human(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            severity="info",
            message="x",
            http_post=fake_post,
        )
        assert env["Success"] is False
        assert env["ErrorType"] == "AlerterError"
        assert "500" in env["Error"]
