"""Discord-webhook alerter for the autonomous astrophotography orchestrator.

Three severity tiers (per the orchestrator plan):
  INFO   — log-style, no @mention. Routine events.
  ALERT  — @mentions the configured human. Fault that was recovered, or one
           the orchestrator can't recover from but is not session-ending.
  PANIC  — @everyone + siren. Safety-supervisor abort, watchdog NINA restart.

The image attachment is optional (e.g. last sub thumbnail when reporting
'guiding lost'). Discord webhook content max is 2000 chars; messages are
truncated to stay under the limit.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiohttp


_DISCORD_CONTENT_LIMIT = 2000
_USERNAME = "NINA Autopilot"


class Severity(str, Enum):
    INFO = "info"
    ALERT = "alert"
    PANIC = "panic"


_PREFIX = {
    Severity.INFO: "**[INFO]**",
    Severity.ALERT: "**[ALERT]**",
    Severity.PANIC: "🚨 **[PANIC]**",
}


def format_payload(
    severity: Severity,
    message: str,
    *,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the JSON body for a Discord webhook POST."""
    mention = ""
    if severity is Severity.ALERT and user_id:
        mention = f"<@{user_id}> "
    elif severity is Severity.PANIC:
        mention = "@everyone "

    content = f"{mention}{_PREFIX[severity]} {message}"
    if len(content) > _DISCORD_CONTENT_LIMIT:
        content = content[: _DISCORD_CONTENT_LIMIT - 1] + "…"

    return {
        "username": _USERNAME,
        "content": content,
    }


HttpPost = Callable[..., Awaitable[tuple[int, str]]]


async def _real_http_post(
    url: str,
    json_payload: dict[str, Any],
    image_path: Optional[str] = None,
) -> tuple[int, str]:
    """Send the webhook. Returns (status_code, response_body)."""
    async with aiohttp.ClientSession() as session:
        if image_path:
            data = aiohttp.FormData()
            data.add_field("payload_json", json.dumps(json_payload), content_type="application/json")
            with open(image_path, "rb") as fh:
                data.add_field(
                    "files[0]",
                    fh.read(),
                    filename=Path(image_path).name,
                    content_type="application/octet-stream",
                )
            async with session.post(url, data=data) as r:
                return r.status, await r.text()
        else:
            async with session.post(url, json=json_payload) as r:
                return r.status, await r.text()


async def tool_alert_human(
    *,
    webhook_url: str,
    severity: str,
    message: str,
    attach_image_path: Optional[str] = None,
    user_id: Optional[str] = None,
    http_post: Optional[HttpPost] = None,
) -> dict[str, Any]:
    """MCP-tool entry point. Returns the standard {Success,...,Type} envelope.

    http_post is injected for testability; production calls _real_http_post.
    """
    envelope_type = "ALERTER"
    try:
        if not webhook_url:
            raise ValueError("webhook_url is required (set DISCORD_WEBHOOK_URL env or pass explicitly)")
        try:
            sev = Severity(severity.lower())
        except ValueError:
            raise ValueError(
                f"Unknown severity '{severity}'. Use one of: info, alert, panic."
            )

        payload = format_payload(sev, message, user_id=user_id)
        sender = http_post or _real_http_post
        status, body = await sender(webhook_url, payload, image_path=attach_image_path)

        if status >= 400:
            raise RuntimeError(f"Discord webhook returned HTTP {status}: {body[:200]}")

        return {
            "Success": True,
            "Message": f"Alert sent ({sev.value})",
            "Details": {"Severity": sev.value, "HttpStatus": status},
            "Type": envelope_type,
        }
    except Exception as e:
        return {
            "Success": False,
            "Error": str(e),
            "ErrorType": "AlerterError",
            "ErrorDetails": {},
            "StatusCode": 500,
            "Type": envelope_type,
        }
