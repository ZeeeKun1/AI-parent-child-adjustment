from __future__ import annotations

import json
from typing import Any

import websocket

from coregulation_poc.providers.websocket_transport import create_websocket_connection
from coregulation_poc.settings import Settings


def check_realtime_connection(settings: Settings, *, timeout_seconds: int = 15) -> dict[str, Any]:
    """Open a native WebSocket and report the first server event without sending media."""
    if settings.dashscope_api_key is None:
        return {"ok": False, "stage": "configuration", "error": "missing_api_key"}
    if not settings.aliyun_workspace_id or not settings.realtime_endpoint:
        return {"ok": False, "stage": "configuration", "error": "missing_workspace_id"}

    headers = [
        f"Authorization: Bearer {settings.dashscope_api_key.get_secret_value()}",
        f"X-DashScope-WorkSpace: {settings.aliyun_workspace_id}",
    ]
    connection: websocket.WebSocket | None = None
    try:
        connection = create_websocket_connection(
            settings.realtime_endpoint,
            header=headers,
            timeout=timeout_seconds,
        )
        raw_message = connection.recv()
        event = json.loads(raw_message) if isinstance(raw_message, str) else {}
        return {
            "ok": event.get("type") == "session.created",
            "stage": "server_event",
            "endpoint_host": settings.realtime_base_url,
            "model": settings.omni_model,
            "event": event,
        }
    except websocket.WebSocketBadStatusException as exc:
        body = exc.resp_body.decode("utf-8", errors="replace") if exc.resp_body else None
        return {
            "ok": False,
            "stage": "http_handshake",
            "status_code": exc.status_code,
            "error": str(exc),
            "response_body": body,
        }
    except (websocket.WebSocketTimeoutException, TimeoutError) as exc:
        return {"ok": False, "stage": "websocket_timeout", "error": str(exc)}
    except (OSError, websocket.WebSocketException, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "stage": "websocket_connection",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if connection is not None:
            connection.close()
