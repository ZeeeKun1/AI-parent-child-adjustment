from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from coregulation_poc.capture.media import (
    MediaChunk,
    MediaFormat,
    MediaKind,
    StrictTimestampNormalizer,
)
from coregulation_poc.storage.run_artifacts import RunArtifactStore

PROTOCOL_VERSION = 1
AUDIO_PACKET = 1
IMAGE_PACKET = 2
PACKET_HEADER = struct.Struct(">BQ")
MAX_SESSION_TIMESTAMP_MS = 8 * 60 * 60 * 1_000
STUDY_TIMEZONE_NAME = "Asia/Shanghai"
STUDY_TIMEZONE = timezone(timedelta(hours=8), name=STUDY_TIMEZONE_NAME)


class BrowserProtocolError(ValueError):
    """Raised when a browser sends an invalid capture message."""


@dataclass(frozen=True, slots=True)
class BrowserCaptureSummary:
    run_dir: Path
    status: str
    valid: bool
    audio_chunk_count: int
    image_chunk_count: int
    audio_bytes_received: int
    image_bytes_received: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    normalized_timestamp_count: int
    api_call_count: int

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_dir.name,
            "status": self.status,
            "valid": self.valid,
            "audio_chunk_count": self.audio_chunk_count,
            "image_chunk_count": self.image_chunk_count,
            "duration_ms": (
                None
                if self.first_timestamp_ms is None or self.last_timestamp_ms is None
                else self.last_timestamp_ms - self.first_timestamp_ms
            ),
            "raw_media_saved": False,
            "api_called": self.api_call_count > 0,
            "api_call_count": self.api_call_count,
        }


def encode_binary_packet(chunk: MediaChunk) -> bytes:
    """Encode one media chunk for protocol tests and non-browser clients."""
    packet_type = AUDIO_PACKET if chunk.kind is MediaKind.AUDIO else IMAGE_PACKET
    return PACKET_HEADER.pack(packet_type, chunk.timestamp_ms) + chunk.payload


def decode_binary_packet(message: bytes, *, max_payload_bytes: int) -> tuple[MediaKind, int, bytes]:
    if len(message) <= PACKET_HEADER.size:
        raise BrowserProtocolError("media packet is missing its header or payload")
    packet_type, timestamp_ms = PACKET_HEADER.unpack_from(message)
    if timestamp_ms > MAX_SESSION_TIMESTAMP_MS:
        raise BrowserProtocolError("media timestamp exceeds the maximum session duration")
    payload = message[PACKET_HEADER.size :]
    if len(payload) > max_payload_bytes:
        raise BrowserProtocolError("media packet exceeds the configured payload limit")
    if packet_type == AUDIO_PACKET:
        kind = MediaKind.AUDIO
    elif packet_type == IMAGE_PACKET:
        kind = MediaKind.IMAGE
    else:
        raise BrowserProtocolError(f"unknown media packet type: {packet_type}")
    return kind, timestamp_ms, payload


class BrowserCaptureRecorder:
    """Validate browser media and retain metadata only, never replayable payloads."""

    def __init__(
        self,
        *,
        output_dir: Path,
        session_id: str,
        media_format: MediaFormat,
        max_image_bytes: int,
        client_capabilities: dict[str, Any] | None = None,
        study_context: dict[str, Any] | None = None,
    ) -> None:
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        self.study_context = dict(study_context or {})
        self.created_at_local = datetime.now(STUDY_TIMEZONE)
        run_name = self._study_run_name() if self.study_context else None
        self.store = RunArtifactStore(output_dir, session_id, run_name=run_name)
        self.media_format = media_format
        self.max_image_bytes = max_image_bytes
        self.normalizer = StrictTimestampNormalizer()
        self.started = False
        self.finished = False
        self.audio_chunk_count = 0
        self.image_chunk_count = 0
        self.audio_bytes_received = 0
        self.image_bytes_received = 0
        self.first_timestamp_ms: int | None = None
        self.last_timestamp_ms: int | None = None
        self.normalized_timestamp_count = 0
        self.api_call_count = 0
        self._write_manifest(client_capabilities or {})

    def _study_run_name(self) -> str:
        return "_".join(
            (
                self.study_context["participant_id"],
                self.created_at_local.strftime("%Y%m%d_%H%M%S"),
                self.study_context["experiment_label"],
                self.study_context["session_round"],
            )
        )

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir

    def _write_manifest(self, client_capabilities: dict[str, Any]) -> None:
        allowed_capabilities = {
            key: value
            for key, value in client_capabilities.items()
            if key
            in {
                "audio_worklet",
                "media_devices",
                "secure_context",
                "page_version",
            }
            and isinstance(value, (str, bool, int, float, type(None)))
        }
        self.store.write_json(
            "manifest.json",
            {
                "source_type": "browser_camera_microphone",
                "protocol_version": PROTOCOL_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "created_at_local": self.created_at_local.isoformat(),
                "study_timezone": STUDY_TIMEZONE_NAME,
                "study_context": self.study_context,
                "media_format": asdict(self.media_format),
                "client_capabilities": allowed_capabilities,
                "privacy": {
                    "raw_media_saved": False,
                    "payload_bytes_saved": False,
                    "device_ids_saved": False,
                    "device_labels_saved": False,
                    "remote_ip_saved": False,
                    "api_credentials_exposed_to_browser": False,
                },
                "engineering_parameters_are_research_findings": False,
            },
        )

    def start(self) -> None:
        if self.finished:
            raise BrowserProtocolError("capture session is already finished")
        if self.started:
            raise BrowserProtocolError("capture session has already started")
        self.started = True
        self.store.append_event(
            {
                "type": "capture_started",
                "recorded_at": datetime.now(UTC).isoformat(),
                "payload_saved": False,
            }
        )

    def accept_packet(self, message: bytes) -> MediaChunk:
        if not self.started or self.finished:
            raise BrowserProtocolError("media is only accepted during an active capture session")
        kind_hint = message[0] if message else None
        payload_limit = (
            self.media_format.audio_chunk_bytes
            if kind_hint == AUDIO_PACKET
            else self.max_image_bytes
        )
        kind, candidate_ms, payload = decode_binary_packet(
            message,
            max_payload_bytes=payload_limit,
        )
        if kind is MediaKind.AUDIO:
            self._validate_audio(payload)
        else:
            self._validate_image(payload)
        timestamp_ms = self.normalizer.normalize(candidate_ms)
        if timestamp_ms != candidate_ms:
            self.normalized_timestamp_count += 1
        chunk = MediaChunk(kind=kind, timestamp_ms=timestamp_ms, payload=payload)
        self._record_chunk(chunk, candidate_timestamp_ms=candidate_ms)
        return chunk

    def _validate_audio(self, payload: bytes) -> None:
        if len(payload) != self.media_format.audio_chunk_bytes:
            raise BrowserProtocolError(
                "audio packet size does not match the negotiated PCM chunk size: "
                f"{len(payload)} != {self.media_format.audio_chunk_bytes}"
            )

    def _validate_image(self, payload: bytes) -> None:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise BrowserProtocolError("image packet is not a decodable JPEG")
        height, width = image.shape[:2]
        if width > self.media_format.image_max_width or height > self.media_format.image_max_height:
            raise BrowserProtocolError(
                "image dimensions exceed the negotiated maximum: "
                f"{width}x{height}"
            )

    def _record_chunk(self, chunk: MediaChunk, *, candidate_timestamp_ms: int) -> None:
        if self.first_timestamp_ms is None:
            self.first_timestamp_ms = chunk.timestamp_ms
        self.last_timestamp_ms = chunk.timestamp_ms
        if chunk.kind is MediaKind.AUDIO:
            self.audio_chunk_count += 1
            self.audio_bytes_received += len(chunk.payload)
        else:
            self.image_chunk_count += 1
            self.image_bytes_received += len(chunk.payload)
        self.store.append_event(
            {
                "type": "media_chunk",
                "kind": chunk.kind.value,
                "timestamp_ms": chunk.timestamp_ms,
                "client_timestamp_ms": candidate_timestamp_ms,
                "payload_bytes": len(chunk.payload),
                "payload_saved": False,
            }
        )

    def finish(
        self,
        *,
        status: str,
        error: str | None = None,
        client_metrics: dict[str, Any] | None = None,
        runtime_metrics: dict[str, Any] | None = None,
    ) -> BrowserCaptureSummary:
        if self.finished:
            return self.summary(status=status)
        self.finished = True
        valid = status == "completed" and self.audio_chunk_count > 0 and self.image_chunk_count > 0
        public_client_metrics = {
            key: value
            for key, value in (client_metrics or {}).items()
            if key
            in {
                "dropped_images",
                "audio_backpressure_stops",
                "camera_health_failures",
                "microphone_health_failures",
                "capture_duration_ms",
            }
            and isinstance(value, (int, float))
        }
        public_runtime_metrics = {
            key: value
            for key, value in (runtime_metrics or {}).items()
            if key
            in {
                "assessment_count",
                "api_call_count",
                "voiceprint_api_call_count",
                "delivery_report_count",
                "analysis_error_count",
                "speaker_binding_count",
                "speaker_binding_success_count",
                "awaiting_post_intervention_response",
                "voice_enabled",
            }
            and isinstance(value, (int, bool))
        }
        api_call_count = public_runtime_metrics.get("api_call_count", 0)
        self.api_call_count = (
            api_call_count
            if isinstance(api_call_count, int) and not isinstance(api_call_count, bool)
            else 0
        )
        metrics = {
            "audio_chunk_count": self.audio_chunk_count,
            "image_chunk_count": self.image_chunk_count,
            "audio_bytes_received": self.audio_bytes_received,
            "image_bytes_received": self.image_bytes_received,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
            "timestamps_strictly_increasing": True,
            "normalized_timestamp_count": self.normalized_timestamp_count,
            "client_metrics": public_client_metrics,
            "realtime_loop": public_runtime_metrics,
            "raw_media_saved": False,
        }
        self.store.write_json("metrics.json", metrics)
        self.store.write_json(
            "result.json",
            {
                "status": status,
                "valid": valid,
                "error": error,
                "api_called": self.api_call_count > 0,
                "api_call_count": self.api_call_count,
                "raw_media_saved": False,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        self.store.append_event(
            {
                "type": "capture_finished",
                "status": status,
                "valid": valid,
                "payload_saved": False,
            }
        )
        return self.summary(status=status)

    def summary(self, *, status: str) -> BrowserCaptureSummary:
        valid = status == "completed" and self.audio_chunk_count > 0 and self.image_chunk_count > 0
        return BrowserCaptureSummary(
            run_dir=self.run_dir,
            status=status,
            valid=valid,
            audio_chunk_count=self.audio_chunk_count,
            image_chunk_count=self.image_chunk_count,
            audio_bytes_received=self.audio_bytes_received,
            image_bytes_received=self.image_bytes_received,
            first_timestamp_ms=self.first_timestamp_ms,
            last_timestamp_ms=self.last_timestamp_ms,
            normalized_timestamp_count=self.normalized_timestamp_count,
            api_call_count=self.api_call_count,
        )
