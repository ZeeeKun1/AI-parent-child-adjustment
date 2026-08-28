from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from coregulation_poc.acoustics.speaker_enrollment import EnrolledSpeaker, SpeakerEnrollment
from coregulation_poc.acoustics.tencent_voiceprint import (
    TencentEnrolledSpeaker,
    TencentSpeakerEnrollment,
)
from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import MediaChunk, MediaFormat, MediaKind
from coregulation_poc.web.app import BrowserServerConfig, create_browser_capture_app
from coregulation_poc.web.protocol import PROTOCOL_VERSION, encode_binary_packet


def _hello(
    session_id: str,
    *,
    access_token: str = "",
    participant_id: str = "P001",
    experiment_label: str = "技术测试",
    session_round: str = "T1",
) -> dict[str, object]:
    return {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "access_token": access_token,
        "study_context": {
            "participant_id": participant_id,
            "experiment_label": experiment_label,
            "session_round": session_round,
            "basic_info": {
                "parent_age": 36,
                "child_age": 9,
                "child_grade": "三年级",
            },
            "family_roles": {
                "parent": "mother",
                "child": "boy",
            },
            "task_context": {
                "task_name": "数学练习",
                "task_type": "math_calculation",
                "task_difficulty": "moderate",
                "child_grade": "三年级",
            },
        },
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
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["referrer-policy"] == "no-referrer"
    assert page.headers["cache-control"] == "no-store"
    assert "开始前" in page.text
    assert "/static/app.js" in page.text
    assert 'id="pause-interventions-button"' in page.text
    assert 'id="parent-age"' in page.text
    assert 'id="child-age"' in page.text
    assert 'id="child-grade"' in page.text
    assert 'id="device-check-button"' in page.text
    assert 'data-device-status="camera"' in page.text
    assert 'data-device-status="microphone"' in page.text
    assert 'id="participant-id"' not in page.text
    assert 'id="session-round"' not in page.text
    assert 'id="access-code"' not in page.text
    assert health.json()["raw_media_saved"] is False
    assert health.json()["research_console_enabled"] is True

    research_page = client.get("/research")
    assert research_page.status_code == 200
    assert "研究控制台" in research_page.text
    assert "/static/research.js" in research_page.text


def test_research_console_access_token_is_checked(tmp_path: Path) -> None:
    config = BrowserServerConfig(
        output_dir=tmp_path / "output",
        research_access_token="correct-research-code",
    )
    app = create_browser_capture_app(config)

    with TestClient(app).websocket_connect("/ws/research") as websocket:
        websocket.send_json({"type": "research_hello", "access_token": "wrong-code"})
        error = websocket.receive_json()

    assert error == {"type": "error", "message": "研究端访问码无效"}


def test_research_console_lists_connected_family_session(tmp_path: Path) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))

    with TestClient(app).websocket_connect("/ws/live") as family:
        family.send_json(_hello("visible_family"))
        assert family.receive_json()["type"] == "ready"
        family.send_json({"type": "start"})
        assert family.receive_json()["type"] == "started"

        with TestClient(app).websocket_connect("/ws/research") as research:
            research.send_json({"type": "research_hello", "access_token": ""})
            snapshot = research.receive_json()

        assert snapshot["type"] == "research_snapshot"
        assert snapshot["sessions"][0]["session_id"] == "visible_family"
        assert snapshot["sessions"][0]["status"] == "active"
        assert snapshot["raw_media_available"] is False

        family.send_json({"type": "stop"})
        assert family.receive_json()["type"] == "summary"


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
                "type": "device_health",
                "device": "camera",
                "status": "normal",
                "reason": "initial_capture_verified",
                "recorded_at_ms": 102,
            }
        )
        websocket.send_json(
            {
                "type": "device_health",
                "device": "microphone",
                "status": "abnormal",
                "reason": "audio_chunks_not_arriving",
                "recorded_at_ms": 200,
            }
        )
        websocket.send_json(
            {
                "type": "stop",
                "client_metrics": {
                    "dropped_images": 0,
                    "audio_backpressure_stops": 0,
                    "camera_health_failures": 0,
                    "microphone_health_failures": 1,
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
    assert run_dirs[0].name.startswith("P001_")
    assert run_dirs[0].name.endswith("_技术测试_T1")
    manifest = (run_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    metrics = (run_dirs[0] / "metrics.json").read_text(encoding="utf-8")
    events = (run_dirs[0] / "events.jsonl").read_text(encoding="utf-8")
    assert "must-not-be-saved" not in manifest
    assert '"participant_id": "P001"' in manifest
    assert '"experiment_label": "技术测试"' in manifest
    assert "must-not-be-saved" not in metrics
    client_metrics = json.loads(metrics)["client_metrics"]
    assert client_metrics["capture_duration_ms"] == 250
    assert client_metrics["microphone_health_failures"] == 1
    assert '"type": "device_health"' in events
    assert '"status": "abnormal"' in events


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


def test_browser_can_reconnect_same_session_after_transient_disconnect(
    tmp_path: Path,
) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as first:
            first.send_json(_hello("reconnect_test"))
            assert first.receive_json()["type"] == "ready"
            first.send_json({"type": "start"})
            assert first.receive_json()["type"] == "started"

        with client.websocket_connect("/ws/live") as second:
            second.send_json(_hello("reconnect_test"))
            assert second.receive_json()["type"] == "ready"
            second.send_json({"type": "start"})
            assert second.receive_json()["type"] == "started"
            second.send_json({"type": "stop"})
            assert second.receive_json()["type"] == "summary"

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "output" / "runs").glob("*/result.json")
    ]
    assert sorted(item["status"] for item in results) == ["completed", "disconnected"]


def test_invalid_control_is_skipped_without_ending_capture(tmp_path: Path) -> None:
    app = create_browser_capture_app(BrowserServerConfig(output_dir=tmp_path / "output"))

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("nonfatal-control-test"))
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "started"
        websocket.send_text("not-json")
        warning = websocket.receive_json()
        assert warning["type"] == "capture_warning"
        assert warning["session_continues"] is True
        websocket.send_json({"type": "stop"})
        assert websocket.receive_json()["type"] == "summary"


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
    assert error["message"] == "本次会话已失效，请刷新页面后重试"
    assert not (tmp_path / "output" / "runs").exists()


def test_family_session_admission_replaces_visible_access_code(tmp_path: Path) -> None:
    config = BrowserServerConfig(
        output_dir=tmp_path / "output",
        access_token="server-only-study-token",
    )
    app = create_browser_capture_app(config)

    with TestClient(app) as client:
        admission = client.post(
            "/api/session-admission",
            json={
                "session_id": "automatic_admission",
                "basic_info": {
                    "parent_age": 36,
                    "child_age": 9,
                    "child_grade": "三年级",
                },
            },
        )
        assert admission.status_code == 200
        session_token = admission.json()["session_token"]

        with client.websocket_connect("/ws/live") as websocket:
            websocket.send_json(_hello("automatic_admission", access_token=session_token))
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "start"})
            assert websocket.receive_json()["type"] == "started"
            websocket.send_json({"type": "stop"})
            assert websocket.receive_json()["type"] == "summary"


def test_repeated_admission_preserves_existing_speaker_binding(tmp_path: Path) -> None:
    app = create_browser_capture_app(
        BrowserServerConfig(output_dir=tmp_path / "output")
    )
    payload = {
        "session_id": "stable-admission",
        "basic_info": {
            "parent_age": 36,
            "child_age": 9,
            "child_grade": "三年级",
        },
    }
    enrollment = SpeakerEnrollment(
        family_id="stable-admission",
        speakers={
            "parent": EnrolledSpeaker(
                label="parent",
                audio_source="test_fixture",
                duration_ms=3000,
                embedding=tuple([0.0] * 256),
            ),
            "child": EnrolledSpeaker(
                label="child",
                audio_source="test_fixture",
                duration_ms=3000,
                embedding=tuple([0.0] * 256),
            ),
        },
    )

    with TestClient(app) as client:
        first = client.post("/api/session-admission", json=payload)
        app.state.session_enrollments["stable-admission"] = enrollment
        second = client.post("/api/session-admission", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_token"] == second.json()["session_token"]
    assert app.state.session_enrollments["stable-admission"] is enrollment


def test_browser_enrollment_uses_cloud_service_without_exposing_ids(
    tmp_path: Path,
) -> None:
    class FakeVoiceprintService:
        provider_name = "fake_tencent_voiceprint"

        def __init__(self) -> None:
            self.deleted: list[TencentSpeakerEnrollment] = []

        def enroll_speaker(
            self,
            pcm_audio: bytes,
            speaker_label: str,
            family_id: str,
            current: TencentSpeakerEnrollment | None,
        ) -> TencentSpeakerEnrollment:
            speakers = {} if current is None else dict(current.speakers)
            speakers[speaker_label] = TencentEnrolledSpeaker(
                label=speaker_label,
                duration_ms=len(pcm_audio) // 32,
                voiceprint_id=f"secret-{speaker_label}-voiceprint-id",
            )
            return TencentSpeakerEnrollment(
                family_id=family_id,
                group_id=(current.group_id if current else "coreg_TestGroup"),
                speakers=speakers,
            )

        def delete_enrollment(self, enrollment: TencentSpeakerEnrollment) -> None:
            self.deleted.append(enrollment)

    service = FakeVoiceprintService()
    app = create_browser_capture_app(
        BrowserServerConfig(output_dir=tmp_path / "output"),
        voiceprint_service=service,
    )
    pcm = (np.sin(np.arange(5 * 16_000) * 0.1) * 4_000).astype("<i2").tobytes()

    with TestClient(app) as client:
        admission = client.post(
            "/api/session-admission",
            json={
                "session_id": "cloud_binding_test",
                "basic_info": {
                    "parent_age": 36,
                    "child_age": 9,
                    "child_grade": "三年级",
                },
            },
        )
        token = admission.json()["session_token"]
        responses = [
            client.post(
                f"/api/speaker-binding/cloud_binding_test/{speaker}",
                content=pcm,
                headers={
                    "content-type": "application/octet-stream",
                    "x-study-access-token": token,
                },
            )
            for speaker in ("parent", "child")
        ]

        assert [response.status_code for response in responses] == [200, 200]
        final_payload = responses[-1].json()
        assert final_payload["complete"] is True
        assert final_payload["provider"] == "tencent_voiceprint_1n"
        assert final_payload["cloud_voiceprint_registered"] is True
        assert "voiceprint_id" not in json.dumps(final_payload)

    assert len(service.deleted) == 1
    assert service.deleted[0].is_complete


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
        enrollment: SpeakerEnrollment | None = None,
    ) -> FakeSession:
        assert session_id == "loop_websocket_test"
        session = FakeSession(send_event)
        sessions.append(session)
        return session

    config = BrowserServerConfig(output_dir=tmp_path / "output")
    app = create_browser_capture_app(config, session_factory=factory)
    # Pre-register a complete enrollment so the session can start without
    # going through the speaker-binding API.
    app.state.session_enrollments["loop_websocket_test"] = SpeakerEnrollment(
        family_id="loop_websocket_test",
        speakers={
            "parent": EnrolledSpeaker(
                label="parent",
                audio_source="test_fixture",
                duration_ms=3000,
                embedding=tuple([0.0] * 256),
            ),
            "child": EnrolledSpeaker(
                label="child",
                audio_source="test_fixture",
                duration_ms=3000,
                embedding=tuple([0.0] * 256),
            ),
        },
    )
    media_format = MediaFormat()
    jpeg = encode_timestamped_jpeg(
        np.full((120, 160, 3), 80, dtype=np.uint8),
        timestamp_ms=101,
        max_bytes=config.max_image_bytes,
    )

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("loop_websocket_test"))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["speaker_enrollment_complete"] is True
        assert ready["runtime_controls_enabled"] is True
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
    assert '"type": "speaker_enrollment"' in events

    with TestClient(app).websocket_connect("/ws/research") as research:
        research.send_json({"type": "research_hello", "access_token": ""})
        snapshot = research.receive_json()
    monitored = next(
        item for item in snapshot["sessions"]
        if item["session_id"] == "loop_websocket_test"
    )
    assert monitored["closed_loop_was_enabled"] is True
    assert monitored["closed_loop_enabled"] is False


def test_preview_mode_returns_control_unavailable_without_failing_session(
    tmp_path: Path,
) -> None:
    config = BrowserServerConfig(output_dir=tmp_path / "output")
    app = create_browser_capture_app(config)
    media_format = MediaFormat()
    jpeg = encode_timestamped_jpeg(
        np.full((120, 160, 3), 80, dtype=np.uint8),
        timestamp_ms=101,
        max_bytes=config.max_image_bytes,
    )

    with TestClient(app).websocket_connect("/ws/live") as websocket:
        websocket.send_json(_hello("preview_control_test"))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["realtime_loop_enabled"] is False
        assert ready["runtime_controls_enabled"] is False

        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "started"
        websocket.send_bytes(
            encode_binary_packet(
                MediaChunk(MediaKind.AUDIO, 100, b"\x00" * media_format.audio_chunk_bytes)
            )
        )
        websocket.send_bytes(encode_binary_packet(MediaChunk(MediaKind.IMAGE, 101, jpeg)))
        websocket.send_json({"type": "pause_interventions"})
        unavailable = websocket.receive_json()
        assert unavailable == {
            "type": "control_unavailable",
            "control_type": "pause_interventions",
            "reason": "realtime_loop_disabled",
        }
        websocket.send_json({"type": "stop"})
        summary = websocket.receive_json()

    assert summary["type"] == "summary"
    assert summary["valid"] is True
    run_dir = next((tmp_path / "output" / "runs").iterdir())
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"type": "control_unavailable"' in events
