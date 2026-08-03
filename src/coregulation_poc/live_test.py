from __future__ import annotations

import importlib.metadata
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from coregulation_poc.capture.buffer import BoundedMediaBuffer
from coregulation_poc.capture.devices import (
    DeviceInventory,
    DeviceKind,
    list_windows_media_devices,
    select_device,
)
from coregulation_poc.capture.directshow import (
    DirectShowCaptureConfig,
    DirectShowMediaSource,
)
from coregulation_poc.capture.media import MediaFormat, MediaKind, MediaSource, MediaSourceError
from coregulation_poc.capture.session import MediaCaptureSession
from coregulation_poc.settings import Settings
from coregulation_poc.storage.run_artifacts import RunArtifactStore

InventoryProvider = Callable[[], DeviceInventory]
SourceFactory = Callable[[DirectShowCaptureConfig], MediaSource]


def _validate_image(payload: bytes, media_format: MediaFormat) -> tuple[int, int]:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise MediaSourceError("Captured image payload is not a valid JPEG.")
    height, width = decoded.shape[:2]
    if width > media_format.image_max_width or height > media_format.image_max_height:
        raise MediaSourceError(
            f"Captured image is {width}x{height}, above the configured maximum "
            f"{media_format.image_max_width}x{media_format.image_max_height}."
        )
    return width, height


def run_live_test(
    *,
    session_id: str,
    settings: Settings,
    duration_seconds: float,
    camera_index: int | None = None,
    camera_name: str | None = None,
    microphone_index: int | None = None,
    microphone_name: str | None = None,
    media_format: MediaFormat | None = None,
    max_audio_queue_chunks: int = 100,
    max_image_queue_chunks: int = 10,
    max_image_bytes: int = 190_000,
    requested_camera_width: int | None = None,
    requested_camera_height: int | None = None,
    requested_camera_fps: int | None = None,
    device_open_timeout_seconds: float = 15.0,
    inventory_provider: InventoryProvider = list_windows_media_devices,
    source_factory: SourceFactory = DirectShowMediaSource,
    progress: Callable[[str], None] = print,
) -> tuple[Path, bool]:
    """Capture a short real device session without calling any external API."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if (
        not math.isfinite(device_open_timeout_seconds)
        or device_open_timeout_seconds <= 0
    ):
        raise ValueError("device_open_timeout_seconds must be positive")
    resolved_format = media_format or MediaFormat()

    progress("[1/6] Enumerating Windows camera and microphone devices...")
    inventory = inventory_provider()
    camera = select_device(
        inventory.cameras,
        kind=DeviceKind.CAMERA,
        index=camera_index,
        name=camera_name,
    )
    microphone = select_device(
        inventory.microphones,
        kind=DeviceKind.MICROPHONE,
        index=microphone_index,
        name=microphone_name,
    )
    progress(f"    PASS: camera[{camera.index}] and microphone[{microphone.index}] selected")

    progress("[2/6] Creating a metadata-only run directory...")
    capture_config = DirectShowCaptureConfig(
        camera=camera,
        microphone=microphone,
        media_format=resolved_format,
        max_image_bytes=max_image_bytes,
        requested_camera_width=requested_camera_width,
        requested_camera_height=requested_camera_height,
        requested_camera_fps=requested_camera_fps,
    )
    source = source_factory(capture_config)
    buffer = BoundedMediaBuffer(
        max_audio_chunks=max_audio_queue_chunks,
        max_image_chunks=max_image_queue_chunks,
    )
    capture = MediaCaptureSession(source=source, buffer=buffer)
    store = RunArtifactStore(settings.output_dir, session_id)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "mode": "live_device_capture_dry_run",
        "api_called": False,
        "source": {
            "backend": "windows_directshow",
            "camera": camera.manifest_record(),
            "microphone": microphone.manifest_record(),
        },
        "media": {
            "audio_sample_rate": resolved_format.audio_sample_rate,
            "audio_channels": resolved_format.audio_channels,
            "audio_sample_width_bytes": resolved_format.audio_sample_width_bytes,
            "audio_chunk_ms": resolved_format.audio_chunk_ms,
            "audio_chunk_bytes": resolved_format.audio_chunk_bytes,
            "image_encoding": resolved_format.image_encoding,
            "image_interval_ms": resolved_format.image_interval_ms,
            "image_max_width": resolved_format.image_max_width,
            "image_max_height": resolved_format.image_max_height,
            "image_max_bytes": max_image_bytes,
            "image_timestamp_labels": resolved_format.image_timestamp_labels,
            "requested_camera_width": requested_camera_width,
            "requested_camera_height": requested_camera_height,
            "requested_camera_fps": requested_camera_fps,
            "parameters_are_engineering_defaults_not_research_findings": True,
        },
        "queue": {
            "max_audio_chunks": max_audio_queue_chunks,
            "max_image_chunks": max_image_queue_chunks,
            "audio_overflow_policy": "stop_with_diagnostic_error",
            "image_overflow_policy": "drop_oldest_image_and_record_count",
        },
        "privacy": {
            "raw_audio_saved": False,
            "raw_video_saved": False,
            "payload_bytes_saved": False,
            "metadata_and_metrics_only": True,
        },
        "software": {
            "application": "coregulation-realtime-poc/0.1.0",
            "av": importlib.metadata.version("av"),
            "opencv-python-headless": importlib.metadata.version(
                "opencv-python-headless"
            ),
            "capture_source_type": source.description.source_type,
        },
    }
    store.write_json("manifest.json", manifest)
    progress(f"    PASS: {store.run_dir}")

    progress("[3/6] Opening devices and waiting for the first media chunk...")
    started_ns = time.monotonic_ns()
    first_chunk_ns: int | None = None
    capture_deadline: float | None = None
    duration_reached = False
    source_ended_early = False
    failure: str | None = None
    audio_count = 0
    image_count = 0
    audio_payload_bytes = 0
    image_payload_bytes = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    previous_timestamp_ms: int | None = None
    timestamps_strictly_increasing = True
    image_dimensions: set[tuple[int, int]] = set()
    sequence = 0

    capture.start()
    try:
        open_deadline = time.monotonic() + device_open_timeout_seconds
        while True:
            now = time.monotonic()
            if capture_deadline is None:
                if now >= open_deadline:
                    raise MediaSourceError(
                        "No media arrived before the device-open timeout. "
                        "Check device selection and Windows privacy permissions."
                    )
                timeout = min(0.5, open_deadline - now)
            else:
                if now >= capture_deadline:
                    duration_reached = True
                    break
                timeout = min(0.5, capture_deadline - now)
            try:
                chunk = capture.read(timeout=timeout)
            except TimeoutError:
                continue
            if chunk is None:
                ended_at = time.monotonic()
                duration_reached = (
                    capture_deadline is not None and ended_at >= capture_deadline
                )
                source_ended_early = not duration_reached
                if source_ended_early and failure is None:
                    failure = "Media source ended before the requested capture duration."
                break
            received_ns = time.monotonic_ns()
            if first_chunk_ns is None:
                first_chunk_ns = received_ns
                capture_deadline = time.monotonic() + duration_seconds
                progress("    PASS: both device graph and capture worker are active")
                progress("[4/6] Validating realtime PCM and JPEG chunks...")
            if chunk.kind is MediaKind.AUDIO:
                if len(chunk.payload) != resolved_format.audio_chunk_bytes:
                    raise MediaSourceError(
                        "Captured PCM chunk length does not match the configured "
                        f"{resolved_format.audio_chunk_ms} ms format."
                    )
                audio_count += 1
                audio_payload_bytes += len(chunk.payload)
            else:
                image_dimensions.add(_validate_image(chunk.payload, resolved_format))
                image_count += 1
                image_payload_bytes += len(chunk.payload)
            if first_timestamp_ms is None:
                first_timestamp_ms = chunk.timestamp_ms
            if (
                previous_timestamp_ms is not None
                and chunk.timestamp_ms <= previous_timestamp_ms
            ):
                timestamps_strictly_increasing = False
                raise MediaSourceError("Delivered media timestamps are not strictly increasing.")
            previous_timestamp_ms = chunk.timestamp_ms
            last_timestamp_ms = chunk.timestamp_ms
            store.append_event(
                {
                    "sequence": sequence,
                    "type": f"input.{chunk.kind.value}",
                    "capture_timestamp_ms": chunk.timestamp_ms,
                    "payload_bytes": len(chunk.payload),
                    "payload_saved": False,
                }
            )
            sequence += 1
    except (MediaSourceError, ValueError, OSError) as exc:
        failure = str(exc)
    finally:
        try:
            capture.stop()
        except MediaSourceError as exc:
            if failure is None:
                failure = str(exc)

    finished_ns = time.monotonic_ns()
    metrics = buffer.metrics()
    progress("[5/6] Verifying queue bounds, timestamps and device release...")
    timestamps_valid = (
        first_timestamp_ms is not None
        and last_timestamp_ms is not None
        and last_timestamp_ms >= first_timestamp_ms
        and timestamps_strictly_increasing
    )
    valid = all(
        (
            failure is None,
            duration_reached,
            not source_ended_early,
            audio_count > 0,
            image_count > 0,
            timestamps_valid,
            not capture.worker_alive,
        )
    )
    metrics_record = {
        "valid": valid,
        "error": failure,
        "requested_duration_seconds": duration_seconds,
        "requested_duration_reached": duration_reached,
        "source_ended_early": source_ended_early,
        "device_open_latency_ms": (
            None if first_chunk_ns is None else (first_chunk_ns - started_ns) // 1_000_000
        ),
        "total_wall_ms": (finished_ns - started_ns) // 1_000_000,
        "first_timestamp_ms": first_timestamp_ms,
        "last_timestamp_ms": last_timestamp_ms,
        "timestamps_strictly_increasing": timestamps_valid,
        "audio_chunk_count": audio_count,
        "image_count": image_count,
        "audio_payload_bytes_observed": audio_payload_bytes,
        "image_payload_bytes_observed": image_payload_bytes,
        "image_dimensions": [
            {"width": width, "height": height}
            for width, height in sorted(image_dimensions)
        ],
        "queue": {
            "received_audio_chunks": metrics.received_audio_chunks,
            "received_image_chunks": metrics.received_image_chunks,
            "delivered_audio_chunks": metrics.delivered_audio_chunks,
            "delivered_image_chunks": metrics.delivered_image_chunks,
            "dropped_image_chunks": metrics.dropped_image_chunks,
            "audio_backpressure_waits": metrics.audio_backpressure_waits,
            "max_observed_depth": metrics.max_observed_depth,
            "final_depth": metrics.current_depth,
        },
        "capture_worker_released": not capture.worker_alive,
        "raw_media_files_created": False,
    }
    store.write_json("metrics.json", metrics_record)
    store.write_json(
        "result.json",
        {
            "status": "passed" if valid else "failed",
            "valid": valid,
            "api_called": False,
            "raw_media_saved": False,
            "error": failure,
        },
    )
    progress(f"    {'PASS' if valid else 'FAIL'}: bounded capture worker stopped cleanly")
    progress(f"[6/6] Dry-run result: {'PASS' if valid else 'FAIL'}")
    return store.run_dir, valid
