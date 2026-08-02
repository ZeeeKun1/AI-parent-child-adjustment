from __future__ import annotations

import json
import struct
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

import websocket

from coregulation_poc.providers.websocket_transport import create_websocket_connection

PROBE_DEFINITIONS: tuple[tuple[str, dict[str, Any], tuple[str, ...]], ...] = (
    ("empty_session", {}, ()),
    ("text_modality", {"modalities": ["text"]}, ("modalities",)),
    (
        "audio_formats",
        {"input_audio_format": "pcm", "output_audio_format": "pcm"},
        ("input_audio_format", "output_audio_format"),
    ),
    ("manual_turn_detection", {"turn_detection": None}, ("turn_detection",)),
    ("diagnostic_instructions", {}, ("instructions",)),
)

_SECRET_FIELD_NAMES = {
    "api_key",
    "authorization",
    "dashscope_api_key",
    "x-dashscope-workspace",
}


def _event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {
        "event_id": f"event_{uuid4().hex}",
        "type": event_type,
        **payload,
    }


def build_session_update_probes(instructions: str) -> list[dict[str, Any]]:
    """Build cumulative session.update events in the required isolation order."""
    probes: list[dict[str, Any]] = []
    session: dict[str, Any] = {}
    for index, (name, additions, added_fields) in enumerate(PROBE_DEFINITIONS, start=1):
        session.update(additions)
        if name == "diagnostic_instructions":
            session["instructions"] = instructions
        probes.append(
            {
                "index": index,
                "name": name,
                "added_fields": list(added_fields),
                "event": _event("session.update", session=deepcopy(session)),
            }
        )
    return probes


def sanitize_for_log(value: Any) -> Any:
    """Recursively redact credential-bearing fields before persistence or printing."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SECRET_FIELD_NAMES else sanitize_for_log(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    return value


def parse_close_frame(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Decode an RFC 6455 close payload without discarding malformed details."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if not raw:
        return {"status_code": None, "reason": "", "payload_valid": True}
    if len(raw) == 1:
        return {
            "status_code": None,
            "reason": "",
            "payload_valid": False,
            "payload_hex": raw.hex(),
        }
    status_code = struct.unpack("!H", raw[:2])[0]
    try:
        reason = raw[2:].decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        reason = raw[2:].decode("utf-8", errors="replace")
        valid = False
    return {"status_code": status_code, "reason": reason, "payload_valid": valid}


def _decode_text_frame(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload).decode("utf-8")
    raise TypeError(f"Unexpected text-frame payload type: {type(payload).__name__}")


def _receive_until(
    connection: websocket.WebSocket,
    *,
    expected_type: str,
    timeout_seconds: float,
    phase: str,
    server_events: list[dict[str, Any]],
    control_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            opcode, payload = connection.recv_data(control_frame=True)
        except websocket.WebSocketTimeoutException:
            continue
        except (OSError, websocket.WebSocketException) as exc:
            return {
                "status": "FAIL",
                "failure_stage": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        if opcode == websocket.ABNF.OPCODE_TEXT:
            try:
                raw_text = _decode_text_frame(payload)
                event = json.loads(raw_text)
            except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
                return {
                    "status": "FAIL",
                    "failure_stage": phase,
                    "error_type": type(exc).__name__,
                    "error": "Server returned a non-JSON text frame.",
                }
            server_events.append({"phase": phase, "event": sanitize_for_log(event)})
            event_type = event.get("type") if isinstance(event, dict) else None
            if event_type == "error":
                return {
                    "status": "FAIL",
                    "failure_stage": phase,
                    "error": "Server returned an error event.",
                }
            if event_type == expected_type:
                return {"status": "PASS"}
            continue

        if opcode == websocket.ABNF.OPCODE_CLOSE:
            close = parse_close_frame(payload)
            control_frames.append({"phase": phase, "type": "close", **close})
            return {
                "status": "FAIL",
                "failure_stage": phase,
                "error": "Server closed the WebSocket.",
                "close": close,
            }

        if opcode in {websocket.ABNF.OPCODE_PING, websocket.ABNF.OPCODE_PONG}:
            kind = "ping" if opcode == websocket.ABNF.OPCODE_PING else "pong"
            size = len(payload) if isinstance(payload, (str, bytes, bytearray)) else None
            control_frames.append({"phase": phase, "type": kind, "payload_bytes": size})
            continue

        control_frames.append({"phase": phase, "type": f"opcode_{opcode}"})

    return {
        "status": "FAIL",
        "failure_stage": phase,
        "error_type": "TimeoutError",
        "error": f"Timed out waiting for {expected_type}.",
    }


def _probe_one_session_update(
    *,
    endpoint: str,
    api_key: str,
    workspace_id: str,
    definition: dict[str, Any],
    timeout_seconds: float,
    include_workspace_header: bool,
) -> dict[str, Any]:
    event_to_send = sanitize_for_log(definition["event"])
    result: dict[str, Any] = {
        "index": definition["index"],
        "name": definition["name"],
        "added_fields": definition["added_fields"],
        "status": "FAIL",
        "sent_event": event_to_send,
        "server_events": [],
        "control_frames": [],
    }
    headers = [f"Authorization: Bearer {api_key}"]
    if include_workspace_header:
        headers.append(f"X-DashScope-WorkSpace: {workspace_id}")
    connection: websocket.WebSocket | None = None
    try:
        connection = create_websocket_connection(
            endpoint, header=headers, timeout=timeout_seconds
        )
        connection.settimeout(min(1.0, timeout_seconds))
        created = _receive_until(
            connection,
            expected_type="session.created",
            timeout_seconds=timeout_seconds,
            phase="authentication",
            server_events=result["server_events"],
            control_frames=result["control_frames"],
        )
        if created["status"] != "PASS":
            result.update(created)
            return result

        connection.send(json.dumps(definition["event"], ensure_ascii=False))
        updated = _receive_until(
            connection,
            expected_type="session.updated",
            timeout_seconds=timeout_seconds,
            phase="session_update",
            server_events=result["server_events"],
            control_frames=result["control_frames"],
        )
        result.update(updated)
        return result
    except websocket.WebSocketBadStatusException as exc:
        result.update(
            {
                "status": "FAIL",
                "failure_stage": "http_handshake",
                "status_code": exc.status_code,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return result
    except (OSError, websocket.WebSocketException) as exc:
        result.update(
            {
                "status": "FAIL",
                "failure_stage": "websocket_connection",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return result
    finally:
        if connection is not None:
            connection.close()


def _summarize_probes(probes: list[dict[str, Any]]) -> dict[str, Any]:
    first_pass = next((probe for probe in probes if probe["status"] == "PASS"), None)
    first_failure = next((probe for probe in probes if probe["status"] == "FAIL"), None)
    first_failing_addition = None
    if first_failure is not None and first_failure["index"] > 1:
        prior = probes[: first_failure["index"] - 1]
        if all(probe["status"] == "PASS" for probe in prior):
            first_failing_addition = {
                "probe": first_failure["name"],
                "fields": first_failure["added_fields"],
            }
    return {
        "all_passed": bool(probes) and all(probe["status"] == "PASS" for probe in probes),
        "minimal_accepted_probe": first_pass["name"] if first_pass else None,
        "first_failure_probe": first_failure["name"] if first_failure else None,
        "first_failing_addition": first_failing_addition,
    }


def probe_session_update_variants(
    *,
    endpoint: str | None,
    api_key: str,
    workspace_id: str | None,
    instructions: str,
    timeout_seconds: float = 15,
    include_workspace_header: bool = True,
) -> dict[str, Any]:
    """Probe cumulative session.update variants using a fresh connection each time."""
    definitions = build_session_update_probes(instructions)
    if not endpoint or not api_key or not workspace_id:
        probes = [
            {
                "index": definition["index"],
                "name": definition["name"],
                "added_fields": definition["added_fields"],
                "status": "SKIPPED",
                "sent_event": sanitize_for_log(definition["event"]),
                "server_events": [],
                "control_frames": [],
                "reason": "Missing endpoint or authentication prerequisite; event was not sent.",
            }
            for definition in definitions
        ]
        return {"summary": _summarize_probes(probes), "probes": probes}

    probes = [
        _probe_one_session_update(
            endpoint=endpoint,
            api_key=api_key,
            workspace_id=workspace_id,
            definition=definition,
            timeout_seconds=timeout_seconds,
            include_workspace_header=include_workspace_header,
        )
        for definition in definitions
    ]
    return {"summary": _summarize_probes(probes), "probes": probes}
