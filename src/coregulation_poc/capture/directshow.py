from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

import av

from coregulation_poc.capture.devices import DeviceKind, MediaDevice
from coregulation_poc.capture.imaging import encode_timestamped_jpeg
from coregulation_poc.capture.media import (
    MediaChunk,
    MediaFormat,
    MediaKind,
    MediaSourceDescription,
    MediaSourceError,
    StrictTimestampNormalizer,
)


def _quote_directshow_name(name: str) -> str:
    if '"' in name:
        raise ValueError("DirectShow device names containing quotes are unsupported")
    return f'"{name}"'


def build_directshow_input(camera: MediaDevice, microphone: MediaDevice) -> str:
    if camera.kind is not DeviceKind.CAMERA:
        raise ValueError("camera must be a camera device")
    if microphone.kind is not DeviceKind.MICROPHONE:
        raise ValueError("microphone must be a microphone device")
    return (
        f"video={_quote_directshow_name(camera.capture_name)}:"
        f"audio={_quote_directshow_name(microphone.capture_name)}"
    )


def _sanitized_capture_error(
    exc: Exception,
    *,
    camera: MediaDevice,
    microphone: MediaDevice,
) -> str:
    detail = str(exc)
    for device in (camera, microphone):
        if device.alternative_name:
            detail = detail.replace(device.alternative_name, "<hardware-identifier>")
    return detail


@dataclass(frozen=True, slots=True)
class DirectShowCaptureConfig:
    camera: MediaDevice
    microphone: MediaDevice
    media_format: MediaFormat = MediaFormat()
    max_image_bytes: int = 190_000
    requested_camera_width: int | None = None
    requested_camera_height: int | None = None
    requested_camera_fps: int | None = None
    audio_buffer_size_ms: int = 100

    def __post_init__(self) -> None:
        if self.camera.kind is not DeviceKind.CAMERA:
            raise ValueError("camera must use DeviceKind.CAMERA")
        if self.microphone.kind is not DeviceKind.MICROPHONE:
            raise ValueError("microphone must use DeviceKind.MICROPHONE")
        if (self.requested_camera_width is None) != (
            self.requested_camera_height is None
        ):
            raise ValueError("camera width and height must be set together")
        if self.requested_camera_fps is not None and self.requested_camera_fps < 1:
            raise ValueError("requested_camera_fps must be positive")
        if self.audio_buffer_size_ms < 20:
            raise ValueError("audio_buffer_size_ms must be at least 20")
        if self.max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        if self.requested_camera_width is not None and (
            self.requested_camera_width < 1 or self.requested_camera_height < 1
        ):
            raise ValueError("requested camera dimensions must be positive")
        if (
            self.media_format.audio_channels != 1
            or self.media_format.audio_sample_width_bytes != 2
            or self.media_format.image_encoding != "jpeg"
        ):
            raise ValueError("DirectShow capture currently requires mono PCM16 audio and JPEG")


class DirectShowMediaSource:
    """Capture one Windows camera and microphone through a shared DirectShow graph."""

    def __init__(self, config: DirectShowCaptureConfig) -> None:
        self.config = config
        self._container: Any | None = None
        self._container_lock = Lock()

    @property
    def description(self) -> MediaSourceDescription:
        return MediaSourceDescription(
            source_type="windows_directshow",
            source_id=f"camera-{self.config.camera.index}_microphone-{self.config.microphone.index}",
            media_format=self.config.media_format,
        )

    def _open_options(self) -> dict[str, str]:
        options = {"audio_buffer_size": str(self.config.audio_buffer_size_ms)}
        if self.config.requested_camera_width is not None:
            options["video_size"] = (
                f"{self.config.requested_camera_width}x"
                f"{self.config.requested_camera_height}"
            )
        if self.config.requested_camera_fps is not None:
            options["framerate"] = str(self.config.requested_camera_fps)
        return options

    def _register_container(self, container: Any) -> None:
        with self._container_lock:
            self._container = container

    def _release_container(self, container: Any) -> None:
        should_close = False
        with self._container_lock:
            if self._container is container:
                self._container = None
                should_close = True
        if should_close:
            container.close()

    def iter_chunks(self, stop_event: Event) -> Iterator[MediaChunk]:
        if os.name != "nt":
            raise MediaSourceError("DirectShow live capture requires Windows.")
        media_format = self.config.media_format
        input_name = build_directshow_input(self.config.camera, self.config.microphone)
        container = None
        try:
            container = av.open(
                input_name,
                format="dshow",
                options=self._open_options(),
            )
            self._register_container(container)
            if stop_event.is_set():
                return
            if not container.streams.video:
                raise MediaSourceError("Selected DirectShow source opened without video.")
            if not container.streams.audio:
                raise MediaSourceError("Selected DirectShow source opened without audio.")

            audio_stream = container.streams.audio[0]
            video_stream = container.streams.video[0]
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=media_format.audio_sample_rate,
            )
            audio_bytes = bytearray()
            audio_samples_emitted = 0
            audio_anchor_ms: int | None = None
            next_image_due_ms = 0
            timeline = StrictTimestampNormalizer()
            started_ns = time.monotonic_ns()

            for packet in container.demux(audio_stream, video_stream):
                if stop_event.is_set():
                    break
                for frame in packet.decode():
                    if stop_event.is_set():
                        break
                    elapsed_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
                    if isinstance(frame, av.AudioFrame):
                        for converted in resampler.resample(frame):
                            audio_bytes.extend(converted.to_ndarray().tobytes())
                            if audio_anchor_ms is None:
                                audio_anchor_ms = elapsed_ms
                            while len(audio_bytes) >= media_format.audio_chunk_bytes:
                                payload = bytes(audio_bytes[: media_format.audio_chunk_bytes])
                                del audio_bytes[: media_format.audio_chunk_bytes]
                                candidate_ms = (
                                    audio_anchor_ms
                                    + audio_samples_emitted
                                    * 1_000
                                    // media_format.audio_sample_rate
                                )
                                timestamp_ms = timeline.normalize(candidate_ms)
                                yield MediaChunk(MediaKind.AUDIO, timestamp_ms, payload)
                                audio_samples_emitted += (
                                    media_format.audio_chunk_bytes
                                    // media_format.audio_sample_width_bytes
                                    // media_format.audio_channels
                                )
                    elif isinstance(frame, av.VideoFrame) and elapsed_ms >= next_image_due_ms:
                        timestamp_ms = timeline.normalize(elapsed_ms)
                        jpeg = encode_timestamped_jpeg(
                            frame.to_ndarray(format="bgr24"),
                            timestamp_ms=timestamp_ms,
                            max_bytes=self.config.max_image_bytes,
                            max_width=media_format.image_max_width,
                            max_height=media_format.image_max_height,
                        )
                        yield MediaChunk(MediaKind.IMAGE, timestamp_ms, jpeg)
                        while next_image_due_ms <= elapsed_ms:
                            next_image_due_ms += media_format.image_interval_ms
            if not stop_event.is_set():
                raise MediaSourceError("Camera or microphone stream ended unexpectedly.")
        except MediaSourceError:
            raise
        except (av.FFmpegError, OSError, ValueError) as exc:
            if stop_event.is_set():
                return
            detail = _sanitized_capture_error(
                exc,
                camera=self.config.camera,
                microphone=self.config.microphone,
            )
            raise MediaSourceError(
                "Could not capture the selected camera and microphone through DirectShow: "
                f"{detail}"
            ) from exc
        finally:
            if container is not None:
                self._release_container(container)

    def close(self) -> None:
        container = None
        with self._container_lock:
            if self._container is not None:
                container = self._container
                self._container = None
        if container is not None:
            container.close()
