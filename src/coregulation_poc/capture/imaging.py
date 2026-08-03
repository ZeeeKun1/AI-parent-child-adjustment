from __future__ import annotations

from typing import Any

import cv2

from coregulation_poc.capture.media import MediaSourceError

TIMESTAMP_STRIP_HEIGHT = 40


def _resize_for_api(
    image: Any,
    *,
    max_width: int,
    max_height: int,
) -> Any:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _add_frame_timestamp_label(image: Any, timestamp_ms: int) -> Any:
    annotated = cv2.copyMakeBorder(
        image,
        TIMESTAMP_STRIP_HEIGHT,
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


def encode_timestamped_jpeg(
    image: Any,
    *,
    timestamp_ms: int,
    max_bytes: int,
    max_width: int = 1_280,
    max_height: int = 720,
) -> bytes:
    """Resize, label and compress one frame for the realtime image buffer."""
    if max_height <= TIMESTAMP_STRIP_HEIGHT:
        raise ValueError("max_height must leave room for the timestamp strip")
    candidate = _resize_for_api(
        image,
        max_width=max_width,
        max_height=max_height - TIMESTAMP_STRIP_HEIGHT,
    )
    candidate = _add_frame_timestamp_label(candidate, timestamp_ms)
    for quality in (85, 78, 70, 62, 54, 46):
        ok, encoded = cv2.imencode(".jpg", candidate, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and encoded.nbytes <= max_bytes:
            return encoded.tobytes()
    raise MediaSourceError(
        f"A sampled frame still exceeds {max_bytes} bytes after compression."
    )
