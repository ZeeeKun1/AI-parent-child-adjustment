from __future__ import annotations

import asyncio
import hmac
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coregulation_poc.capture.media import MediaChunk, MediaFormat
from coregulation_poc.paths import DEFAULT_OUTPUT_DIR, PACKAGE_DIR, resolve_project_path
from coregulation_poc.runtime.session import RealtimeBrowserSession, RealtimeSessionFactory
from coregulation_poc.web.protocol import (
    PROTOCOL_VERSION,
    BrowserCaptureRecorder,
    BrowserProtocolError,
)

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
STATIC_DIR = (PACKAGE_DIR / "web" / "static").resolve()
ChunkHandler = Callable[[MediaChunk], Awaitable[None]]
AUDITED_RUNTIME_EVENTS = {
    "loop_started",
    "analysis_started",
    "state_update",
    "intervention",
    "intervention_held",
    "intervention_outcome",
    "interventions_paused",
    "interventions_resumed",
    "delivery_execution_received",
    "loop_error",
}


@dataclass(frozen=True, slots=True)
class BrowserServerConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    media_format: MediaFormat = field(default_factory=MediaFormat)
    max_image_bytes: int = 750_000
    max_session_seconds: int = 2 * 60 * 60
    hello_timeout_seconds: int = 10
    access_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", resolve_project_path(self.output_dir))
        if self.max_image_bytes < 10_000:
            raise ValueError("max_image_bytes is too small for browser JPEG frames")
        if self.max_session_seconds < 1:
            raise ValueError("max_session_seconds must be positive")
        if self.hello_timeout_seconds < 1:
            raise ValueError("hello_timeout_seconds must be positive")
        if self.access_token is not None and len(self.access_token) < 12:
            raise ValueError("browser capture access token must contain at least 12 characters")


def _parse_control(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BrowserProtocolError("control message must be valid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise BrowserProtocolError("control message must contain a string type")
    return value


def _parse_hello(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if message.get("type") != "hello":
        raise BrowserProtocolError("the first message must be hello")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise BrowserProtocolError("browser protocol version does not match the server")
    session_id = message.get("session_id")
    if not isinstance(session_id, str) or SESSION_ID.fullmatch(session_id) is None:
        raise BrowserProtocolError(
            "session_id must contain 1-80 letters, numbers, underscores or hyphens"
        )
    capabilities = message.get("capabilities")
    return session_id, capabilities if isinstance(capabilities, dict) else {}


def _origin_matches_host(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not origin or not host:
        return True
    return urlsplit(origin).netloc.casefold() == host.casefold()


def _valid_access_token(message: dict[str, Any], expected_token: str | None) -> bool:
    if expected_token is None:
        return True
    supplied_token = message.get("access_token")
    return isinstance(supplied_token, str) and hmac.compare_digest(
        supplied_token,
        expected_token,
    )


def create_browser_capture_app(
    config: BrowserServerConfig | None = None,
    *,
    chunk_handler: ChunkHandler | None = None,
    session_factory: RealtimeSessionFactory | None = None,
) -> FastAPI:
    server_config = config or BrowserServerConfig()
    app = FastAPI(
        title="Parent-child co-regulation browser capture",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.browser_server_config = server_config
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "capture_mode": "browser_camera_microphone",
            "protocol_version": PROTOCOL_VERSION,
            "raw_media_saved": False,
            "api_called": False,
            "api_calls_enabled": session_factory is not None,
            "downstream_chunk_handler_configured": chunk_handler is not None,
            "realtime_loop_configured": session_factory is not None,
            "access_control_required": server_config.access_token is not None,
        }

    @app.websocket("/ws/live")
    async def live_capture(websocket: WebSocket) -> None:
        if not _origin_matches_host(websocket):
            await websocket.close(code=1008, reason="WebSocket origin is not allowed")
            return
        await websocket.accept()
        recorder: BrowserCaptureRecorder | None = None
        runtime_session: RealtimeBrowserSession | None = None
        runtime_stopped = False
        completion_status = "disconnected"
        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)
                if recorder is not None and payload.get("type") in AUDITED_RUNTIME_EVENTS:
                    audited = dict(payload)
                    audio_attached = isinstance(audited.pop("audio_base64", None), str)
                    recorder.store.append_event(
                        {
                            "type": "realtime_loop_event",
                            "event": audited,
                            "audio_attached_to_browser_message": audio_attached,
                            "payload_saved": False,
                        }
                    )

        async def stop_runtime(status: str) -> None:
            nonlocal runtime_stopped
            if runtime_session is None or runtime_stopped:
                return
            runtime_stopped = True
            await runtime_session.stop(status)

        def runtime_metrics() -> dict[str, Any]:
            if runtime_session is None:
                return {}
            metrics = getattr(runtime_session, "runtime_metrics", None)
            if isinstance(metrics, dict):
                return metrics
            return {"api_call_count": runtime_session.api_call_count}

        try:
            try:
                hello_event = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=server_config.hello_timeout_seconds,
                )
            except TimeoutError as exc:
                raise BrowserProtocolError("browser did not send hello before the timeout") from exc
            if hello_event.get("text") is None:
                raise BrowserProtocolError("the first WebSocket message must be a text hello")
            hello_message = _parse_control(hello_event["text"])
            session_id, capabilities = _parse_hello(hello_message)
            if not _valid_access_token(hello_message, server_config.access_token):
                raise BrowserProtocolError("实验访问码无效")
            recorder = BrowserCaptureRecorder(
                output_dir=server_config.output_dir,
                session_id=session_id,
                media_format=server_config.media_format,
                max_image_bytes=server_config.max_image_bytes,
                client_capabilities=capabilities,
            )
            if session_factory is not None:
                runtime_session = session_factory(session_id, send_json)
            await send_json(
                {
                    "type": "ready",
                    "protocol_version": PROTOCOL_VERSION,
                    "media_format": {
                        **asdict(server_config.media_format),
                        "audio_chunk_bytes": server_config.media_format.audio_chunk_bytes,
                    },
                    "max_image_bytes": server_config.max_image_bytes,
                    "max_session_seconds": server_config.max_session_seconds,
                    "raw_media_saved": False,
                    "realtime_loop_enabled": runtime_session is not None,
                }
            )
            session_deadline = time.monotonic() + server_config.max_session_seconds
            while True:
                remaining_seconds = session_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise BrowserProtocolError("capture session exceeded the server time limit")
                try:
                    event = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=remaining_seconds,
                    )
                except TimeoutError as exc:
                    raise BrowserProtocolError(
                        "capture session exceeded the server time limit"
                    ) from exc
                if event.get("bytes") is not None:
                    chunk = recorder.accept_packet(event["bytes"])
                    if chunk_handler is not None:
                        await chunk_handler(chunk)
                    if runtime_session is not None:
                        await runtime_session.accept_chunk(chunk)
                    continue
                raw_text = event.get("text")
                if raw_text is None:
                    raise WebSocketDisconnect(code=1000)
                control = _parse_control(raw_text)
                if control["type"] == "start":
                    recorder.start()
                    await send_json({"type": "started"})
                    if runtime_session is not None:
                        await runtime_session.start()
                elif control["type"] == "stop":
                    completion_status = "completed"
                    await stop_runtime(completion_status)
                    client_metrics = control.get("client_metrics")
                    summary = recorder.finish(
                        status=completion_status,
                        client_metrics=(
                            client_metrics if isinstance(client_metrics, dict) else None
                        ),
                        runtime_metrics=runtime_metrics(),
                    )
                    await send_json({"type": "summary", **summary.as_public_dict()})
                    await websocket.close(code=1000)
                    return
                elif control["type"] == "abort":
                    completion_status = "failed"
                    await stop_runtime(completion_status)
                    client_metrics = control.get("client_metrics")
                    summary = recorder.finish(
                        status=completion_status,
                        error="browser reported a capture failure",
                        client_metrics=(
                            client_metrics if isinstance(client_metrics, dict) else None
                        ),
                        runtime_metrics=runtime_metrics(),
                    )
                    await send_json({"type": "summary", **summary.as_public_dict()})
                    await websocket.close(code=1011)
                    return
                elif control["type"] == "ping":
                    await send_json({"type": "pong"})
                elif runtime_session is not None and await runtime_session.handle_control(control):
                    continue
                else:
                    raise BrowserProtocolError(
                        f"unsupported control message: {control['type']}"
                    )
        except WebSocketDisconnect:
            completion_status = "disconnected"
        except (BrowserProtocolError, ValueError, OSError) as exc:
            completion_status = "failed"
            if recorder is not None:
                recorder.finish(
                    status="failed",
                    error=str(exc),
                    runtime_metrics=runtime_metrics(),
                )
            try:
                await send_json({"type": "error", "message": str(exc)})
                await websocket.close(code=1003)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            await stop_runtime(completion_status)
            if recorder is not None and not recorder.finished:
                recorder.finish(
                    status=completion_status,
                    error=(
                        "browser disconnected before a normal stop"
                        if completion_status == "disconnected"
                        else None
                    ),
                    runtime_metrics=runtime_metrics(),
                )

    return app


def run_browser_capture_server(
    *,
    host: str,
    port: int,
    output_dir: Path,
    access_token: str | None,
    log_level: str = "info",
    session_factory: RealtimeSessionFactory | None = None,
) -> None:
    import uvicorn

    config = BrowserServerConfig(output_dir=output_dir, access_token=access_token)
    uvicorn.run(
        create_browser_capture_app(config, session_factory=session_factory),
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
