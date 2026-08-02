from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import av
import cv2


class VideoValidationError(ValueError):
    """Raised when a clip cannot support the intended audio-video test."""


class MediaKind(StrEnum):
    AUDIO = "audio"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class MediaChunk:
    kind: MediaKind
    timestamp_ms: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    duration_ms: int
    width: int
    height: int
    frame_rate: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int


@dataclass(frozen=True, slots=True)
class ReplayMedia:
    metadata: VideoMetadata
    chunks: tuple[MediaChunk, ...]
    audio_chunk_count: int
    image_count: int
    audio_sample_rate: int
    audio_chunk_ms: int
    image_interval_ms: int
    image_timestamp_labels: bool


def _frame_timestamp_ms(frame: av.AudioFrame | av.VideoFrame) -> int:
    if frame.pts is None or frame.time_base is None:
        return 0
    return max(0, round(float(frame.pts * frame.time_base) * 1000))


def _rate_as_float(rate: object | None) -> float:
    return float(rate) if rate is not None else 0.0


def inspect_video(path: Path) -> VideoMetadata:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute() or not resolved.is_file():
        raise VideoValidationError(f"Video file does not exist: {resolved}")

    try:
        with av.open(str(resolved)) as container:
            if not container.streams.video:
                raise VideoValidationError("The clip has no video track.")
            if not container.streams.audio:
                raise VideoValidationError("The clip has no audio track.")
            video = container.streams.video[0]
            audio = container.streams.audio[0]
            duration_ms = round(float(container.duration or 0) / av.time_base * 1000)
            return VideoMetadata(
                path=resolved,
                duration_ms=duration_ms,
                width=video.codec_context.width,
                height=video.codec_context.height,
                frame_rate=_rate_as_float(video.average_rate),
                video_codec=video.codec_context.name,
                audio_codec=audio.codec_context.name,
                audio_sample_rate=audio.codec_context.sample_rate,
            )
    except (av.FFmpegError, OSError) as exc:
        raise VideoValidationError(f"Cannot open video: {exc}") from exc


def _resize_for_api(image: object, *, max_width: int = 1280, max_height: int = 720) -> object:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _add_frame_timestamp_label(image: object, timestamp_ms: int) -> object:
    """Add a non-overlapping timestamp strip so visual evidence can cite an exact frame."""
    strip_height = 40
    annotated = cv2.copyMakeBorder(
        image,
        strip_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        annotated,
        f"frame_time_ms={timestamp_ms}",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _encode_jpeg(image: object, *, max_bytes: int, timestamp_ms: int) -> bytes:
    candidate = _resize_for_api(image, max_height=680)
    candidate = _add_frame_timestamp_label(candidate, timestamp_ms)
    for quality in (85, 78, 70, 62, 54, 46):
        ok, encoded = cv2.imencode(".jpg", candidate, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and encoded.nbytes <= max_bytes:
            return encoded.tobytes()
    raise VideoValidationError(
        f"A sampled frame still exceeds {max_bytes} bytes after compression."
    )


def decode_video_for_replay(
    path: Path,
    *,
    image_interval_ms: int = 1000,
    audio_chunk_ms: int = 100,
    audio_sample_rate: int = 16_000,
    max_image_bytes: int = 190_000,
) -> ReplayMedia:
    """Decode one clip into chronological PCM audio and JPEG image chunks."""
    if image_interval_ms < 250:
        raise ValueError("image_interval_ms must be at least 250")
    if audio_chunk_ms < 20:
        raise ValueError("audio_chunk_ms must be at least 20")

    metadata = inspect_video(path)
    chunks: list[MediaChunk] = []
    audio_bytes = bytearray()
    audio_samples_emitted = 0
    bytes_per_audio_chunk = audio_sample_rate * 2 * audio_chunk_ms // 1000
    next_image_ms = 0

    try:
        with av.open(str(metadata.path)) as container:
            audio_stream = container.streams.audio[0]
            video_stream = container.streams.video[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=audio_sample_rate)

            for packet in container.demux(audio_stream, video_stream):
                for frame in packet.decode():
                    if isinstance(frame, av.AudioFrame):
                        for converted in resampler.resample(frame):
                            audio_bytes.extend(converted.to_ndarray().tobytes())
                            while len(audio_bytes) >= bytes_per_audio_chunk:
                                timestamp_ms = audio_samples_emitted * 1000 // audio_sample_rate
                                payload = bytes(audio_bytes[:bytes_per_audio_chunk])
                                del audio_bytes[:bytes_per_audio_chunk]
                                chunks.append(MediaChunk(MediaKind.AUDIO, timestamp_ms, payload))
                                audio_samples_emitted += bytes_per_audio_chunk // 2
                    elif isinstance(frame, av.VideoFrame):
                        timestamp_ms = _frame_timestamp_ms(frame)
                        if timestamp_ms >= next_image_ms:
                            jpeg = _encode_jpeg(
                                frame.to_ndarray(format="bgr24"),
                                max_bytes=max_image_bytes,
                                timestamp_ms=timestamp_ms,
                            )
                            chunks.append(MediaChunk(MediaKind.IMAGE, timestamp_ms, jpeg))
                            while next_image_ms <= timestamp_ms:
                                next_image_ms += image_interval_ms

            for converted in resampler.resample(None):
                audio_bytes.extend(converted.to_ndarray().tobytes())
            if audio_bytes:
                padding = (-len(audio_bytes)) % bytes_per_audio_chunk
                audio_bytes.extend(b"\x00" * padding)
                while audio_bytes:
                    timestamp_ms = audio_samples_emitted * 1000 // audio_sample_rate
                    payload = bytes(audio_bytes[:bytes_per_audio_chunk])
                    del audio_bytes[:bytes_per_audio_chunk]
                    chunks.append(MediaChunk(MediaKind.AUDIO, timestamp_ms, payload))
                    audio_samples_emitted += bytes_per_audio_chunk // 2
    except (av.FFmpegError, OSError) as exc:
        raise VideoValidationError(f"Failed while decoding video: {exc}") from exc

    audio_count = sum(chunk.kind is MediaKind.AUDIO for chunk in chunks)
    image_count = sum(chunk.kind is MediaKind.IMAGE for chunk in chunks)
    if audio_count == 0 or image_count == 0:
        raise VideoValidationError("The clip did not yield both audio and image chunks.")

    chunks.sort(key=lambda chunk: (chunk.timestamp_ms, 0 if chunk.kind is MediaKind.AUDIO else 1))
    return ReplayMedia(
        metadata=metadata,
        chunks=tuple(chunks),
        audio_chunk_count=audio_count,
        image_count=image_count,
        audio_sample_rate=audio_sample_rate,
        audio_chunk_ms=audio_chunk_ms,
        image_interval_ms=image_interval_ms,
        image_timestamp_labels=True,
    )
