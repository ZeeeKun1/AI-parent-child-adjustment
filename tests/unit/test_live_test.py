from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import numpy as np

from coregulation_poc.capture.devices import DeviceInventory, DeviceKind, MediaDevice
from coregulation_poc.capture.directshow import DirectShowCaptureConfig
from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import (
    MediaChunk,
    MediaKind,
    MediaSourceDescription,
    MediaSourceError,
)
from coregulation_poc.live_test import run_live_test
from coregulation_poc.settings import Settings


class FakeLiveSource:
    def __init__(self, config: DirectShowCaptureConfig) -> None:
        self.config = config
        self.closed = False

    @property
    def description(self) -> MediaSourceDescription:
        return MediaSourceDescription(
            source_type="synthetic_test_devices",
            source_id="fake-camera-and-microphone",
            media_format=self.config.media_format,
        )

    def iter_chunks(self, stop_event: Event) -> Iterator[MediaChunk]:
        timestamp_ms = 0
        iteration = 0
        while not stop_event.is_set():
            yield MediaChunk(
                MediaKind.AUDIO,
                timestamp_ms,
                b"\x00" * self.config.media_format.audio_chunk_bytes,
            )
            timestamp_ms += self.config.media_format.audio_chunk_ms
            if iteration % 4 == 0:
                image_timestamp_ms = timestamp_ms
                image = np.full((120, 160, 3), iteration, dtype=np.uint8)
                jpeg = encode_timestamped_jpeg(
                    image,
                    timestamp_ms=image_timestamp_ms,
                    max_bytes=self.config.max_image_bytes,
                )
                yield MediaChunk(MediaKind.IMAGE, image_timestamp_ms, jpeg)
                timestamp_ms += 1
            iteration += 1
            time.sleep(0.002)

    def close(self) -> None:
        self.closed = True


def _inventory() -> DeviceInventory:
    return DeviceInventory(
        cameras=(MediaDevice(DeviceKind.CAMERA, 0, "Test Camera", "private-camera-id"),),
        microphones=(
            MediaDevice(DeviceKind.MICROPHONE, 0, "Test Microphone", "private-mic-id"),
        ),
    )


def test_live_dry_run_uses_bounded_capture_without_saving_media(tmp_path: Path) -> None:
    created_sources: list[FakeLiveSource] = []

    def source_factory(config: DirectShowCaptureConfig) -> FakeLiveSource:
        source = FakeLiveSource(config)
        created_sources.append(source)
        return source

    run_dir, valid = run_live_test(
        session_id="live_test",
        settings=Settings(output_dir=tmp_path / "output"),
        duration_seconds=0.05,
        camera_index=0,
        microphone_index=0,
        max_audio_queue_chunks=4,
        max_image_queue_chunks=2,
        inventory_provider=_inventory,
        source_factory=source_factory,
        progress=lambda _message: None,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert valid is True
    assert result["api_called"] is False
    assert result["raw_media_saved"] is False
    assert manifest["privacy"]["payload_bytes_saved"] is False
    assert "private-camera-id" not in json.dumps(manifest)
    assert metrics["audio_chunk_count"] > 0
    assert metrics["image_count"] > 0
    assert metrics["timestamps_strictly_increasing"] is True
    assert metrics["capture_worker_released"] is True
    assert all(source.closed for source in created_sources)
    assert '"payload_saved": false' in events
    assert not list(run_dir.glob("*.wav"))
    assert not list(run_dir.glob("*.pcm"))
    assert not list(run_dir.glob("*.jpg"))
    assert not list(run_dir.glob("*.mp4"))


def test_live_dry_run_writes_diagnostic_result_on_device_failure(tmp_path: Path) -> None:
    class FailingSource(FakeLiveSource):
        def iter_chunks(self, _stop_event: Event) -> Iterator[MediaChunk]:
            raise MediaSourceError("microphone disconnected")
            yield  # pragma: no cover

    run_dir, valid = run_live_test(
        session_id="device_failure",
        settings=Settings(output_dir=tmp_path / "output"),
        duration_seconds=0.01,
        camera_index=0,
        microphone_index=0,
        inventory_provider=_inventory,
        source_factory=FailingSource,
        progress=lambda _message: None,
    )

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert valid is False
    assert result["status"] == "failed"
    assert result["error"] == "microphone disconnected"
    assert metrics["capture_worker_released"] is True
