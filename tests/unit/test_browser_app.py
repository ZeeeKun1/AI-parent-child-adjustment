from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import MediaChunk, MediaFormat, MediaKind
from coregulation_poc.web.app import BrowserServerConfig, create_browser_capture_app
from coregulation_poc.web.protocol import PROTOCOL_VERSION, encode_binary_packet


def _hello(session_id: str, *, access_token: str = "") -> dict[str, object]:
    return {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "access_token": access_token,
        "capabilities": {
            "audio_worklet": True,
            "media_devices": True,
            "secure_context": True,
            "page_version": "test",
        },
    }


def test_browser_page_and_health_are_served(tmp_path: Path) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))
    client = TestClient(app)

    page = client.get("/")
    health = client.get("/healthz")

    assert page.status_code == 200
    assert "亲子共调节实时采集" in page.text
    assert "/static/app.js" in page.text
    assert 'id="pause-interventions-button"' in page.text
    assert health.json()["raw_media_saved"] is False


def test_browser_websocket_receives_media_as_common_chunks(tmp_path: Path) -> None:
    received: list[MediaChunk] = []

    async def collect(chunk: MediaChunk) -> None:
        received.append(chunk)

    config = BrowserServerConfig(output_dir=tmp_path / "output")
    app = create_browser_capture_app(config, chunk_handler=collect)
    media_format = MediaFormat()
    jpeg = encode_timestamped_jpeg(
        np.full((120, 160, 3), 80, dtype=np.uint8),
        timestamp_ms=101,
        max_bytes=config.max_image_bytes,
    )

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("websocket_test"))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["media_format"]["audio_chunk_bytes"] == 3200
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "started"
        websocket.send_bytes(
            encode_binary_packet(
                MediaChunk(MediaKind.AUDIO, 100, b"\x00" * media_format.audio_chunk_bytes)
            )
        )
        websocket.send_bytes(
            encode_binary_packet(MediaChunk(MediaKind.IMAGE, 101, jpeg))
        )
        websocket.send_json(
            {
                "type": "stop",
                "client_metrics": {
                    "dropped_images": 0,
                    "audio_backpressure_stops": 0,
                    "capture_duration_ms": 250,
                    "device_id": "must-not-be-saved",
                },
            }
        )
        summary = websocket.receive_json()

    assert summary["type"] == "summary"
    assert summary["valid"] is True
    assert [chunk.kind for chunk in received] == [MediaKind.AUDIO, MediaKind.IMAGE]
    run_dirs = list((tmp_path / "output" / "runs").iterdir())
    assert len(run_dirs) == 1
    manifest = (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    metrics = (run_dirs[0] / "metrics.json").read_text(encoding="utf-8")
    assert "must-not-be-saved" not in manifest
    assert "must-not-be-saved" not in metrics
    assert json.loads(metrics)["client_metrics"]["capture_duration_ms"] == 250


def test_browser_disconnect_is_recorded_as_failure(tmp_path: Path) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("disconnect_test"))
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "started"

    run_dir = next((tmp_path / "output" / "runs").iterdir())
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "disconnected"
    assert result["valid"] is False
    assert result["raw_media_saved"] is False


def test_browser_access_token_is_checked_before_creating_a_run(tmp_path: Path) -> None:
    config = BrowserServerConfig(
        output_dir=tmp_path / "output",
        access_token="correct-study-code",
    )
    app = create_browser_capture_app(config)

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("unauthorized_test", access_token="wrong-code"))
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["message"] == "实验访问码无效"
    assert not (tmp_path / "output" / "runs").exists()


def test_browser_rejects_cross_origin_websocket(tmp_path: Path) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/ws/live",
            headers={"origin": "https://untrusted.example"},
        ):
            pass

    assert exc_info.value.code == 1008
    assert not (tmp_path / "output" / "runs").exists()


def test_browser_routes_media_and_delivery_control_to_realtime_session(
    tmp_path: Path,
) -> None:
    sessions: list[object] = []

    class FakeSession:
        def __init__(
            self,
            send_event: Callable[[dict[str, Any]], Awaitable[None]],
        ) -> None:
            self.send_event = send_event
            self.started = False
            self.chunks: list[MediaChunk] = []
            self.controls: list[dict[str, object]] = []
            self.stopped_with: list[str] = []

        @property
        def api_call_count(self) -> int:
            return 0

        async def start(self) -> None:
            self.started = True
            await self.send_event(
                {
                    "type": "intervention",
                    "delivery_id": "delivery-1",
                    "message": "approved test message",
                    "audio_base64": "must-not-be-saved",
                }
            )

        async def accept_chunk(self, chunk: MediaChunk) -> None:
            self.chunks.append(chunk)

        async def handle_control(self, control: dict[str, object]) -> bool:
            if control.get("type") != "delivery_execution":
                return False
            self.controls.append(control)
            await self.send_event({"type": "delivery_execution_received"})
            return True

        async def stop(self, status: str) -> None:
            self.stopped_with.append(status)

    def factory(
        session_id: str,
        send_event: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> FakeSession:
        assert session_id == "loop_websocket_test"
        session = FakeSession(send_event)
        sessions.append(session)
        return session

    config = BrowserServerConfig(output_dir=tmp_path / "output")
    app = create_browser_capture_app(config, session_factory=factory)
    media_format = MediaFormat()
    jpeg = encode_timestamped_jpeg(
        np.full((120, 160, 3), 80, dtype=np.uint8),
        timestamp_ms=101,
        max_bytes=config.max_image_bytes,
    )

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("loop_websocket_test"))
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "started"
        assert websocket.receive_json()["type"] == "intervention"
        websocket.send_bytes(
            encode_binary_packet(
                MediaChunk(MediaKind.AUDIO, 100, b"\x00" * media_format.audio_chunk_bytes)
            )
        )
        websocket.send_bytes(encode_binary_packet(MediaChunk(MediaKind.IMAGE, 101, jpeg)))
        websocket.send_json({"type": "delivery_execution", "delivery_id": "delivery-1"})
        assert websocket.receive_json()["type"] == "delivery_execution_received"
        websocket.send_json({"type": "stop"})
        assert websocket.receive_json()["type"] == "summary"

    session = sessions[0]
    assert isinstance(session, FakeSession)
    assert session.started is True
    assert [chunk.kind for chunk in session.chunks] == [MediaKind.AUDIO, MediaKind.IMAGE]
    assert session.controls[0]["delivery_id"] == "delivery-1"
    assert session.stopped_with == ["completed"]
    run_dir = next((tmp_path / "output" / "runs").iterdir())
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "must-not-be-saved" not in events
    assert '"audio_attached_to_browser_message": true' in events
