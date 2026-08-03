from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Condition

from coregulation_poc.capture.media import (
    MediaChunk,
    MediaKind,
    MediaSequenceError,
    MediaSourceError,
)


@dataclass(frozen=True, slots=True)
class BufferMetrics:
    received_audio_chunks: int
    received_image_chunks: int
    delivered_audio_chunks: int
    delivered_image_chunks: int
    dropped_image_chunks: int
    audio_backpressure_waits: int
    max_observed_depth: int
    current_depth: int


class BoundedMediaBuffer:
    """Thread-safe bounded queue that preserves audio and sheds old video under load."""

    def __init__(self, *, max_audio_chunks: int, max_image_chunks: int) -> None:
        if max_audio_chunks < 1 or max_image_chunks < 1:
            raise ValueError("media queue capacities must be positive")
        self.max_audio_chunks = max_audio_chunks
        self.max_image_chunks = max_image_chunks
        self._chunks: deque[MediaChunk] = deque()
        self._condition = Condition()
        self._closed = False
        self._error: MediaSourceError | None = None
        self._audio_depth = 0
        self._image_depth = 0
        self._last_received_timestamp_ms = -1
        self._received_audio = 0
        self._received_images = 0
        self._delivered_audio = 0
        self._delivered_images = 0
        self._dropped_images = 0
        self._audio_waits = 0
        self._max_depth = 0

    def _remove_oldest_image(self) -> bool:
        for index, chunk in enumerate(self._chunks):
            if chunk.kind is MediaKind.IMAGE:
                del self._chunks[index]
                self._image_depth -= 1
                self._dropped_images += 1
                return True
        return False

    def put(self, chunk: MediaChunk, *, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            if chunk.timestamp_ms <= self._last_received_timestamp_ms:
                raise MediaSequenceError(
                    "media timestamps must be strictly increasing: "
                    f"{chunk.timestamp_ms} <= {self._last_received_timestamp_ms}"
                )
            if chunk.kind is MediaKind.AUDIO:
                while self._audio_depth >= self.max_audio_chunks and not self._closed:
                    self._audio_waits += 1
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MediaSourceError(
                            "Audio queue remained full; capture stopped to avoid silent audio loss."
                        )
                    self._condition.wait(remaining)
            else:
                if self._image_depth >= self.max_image_chunks:
                    self._remove_oldest_image()
            if self._closed:
                raise MediaSourceError("Cannot add media after the capture buffer is closed.")

            self._chunks.append(chunk)
            self._last_received_timestamp_ms = chunk.timestamp_ms
            if chunk.kind is MediaKind.AUDIO:
                self._audio_depth += 1
                self._received_audio += 1
            else:
                self._image_depth += 1
                self._received_images += 1
            self._max_depth = max(self._max_depth, len(self._chunks))
            self._condition.notify_all()

    def get(self, *, timeout: float | None = None) -> MediaChunk | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._chunks and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No media chunk became available before the timeout.")
                self._condition.wait(remaining)
            if self._chunks:
                chunk = self._chunks.popleft()
                if chunk.kind is MediaKind.AUDIO:
                    self._audio_depth -= 1
                    self._delivered_audio += 1
                else:
                    self._image_depth -= 1
                    self._delivered_images += 1
                self._condition.notify_all()
                return chunk
            if self._error is not None:
                raise self._error
            return None

    def close(self, error: MediaSourceError | None = None) -> None:
        with self._condition:
            if error is not None and self._error is None:
                self._error = error
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def metrics(self) -> BufferMetrics:
        with self._condition:
            return BufferMetrics(
                received_audio_chunks=self._received_audio,
                received_image_chunks=self._received_images,
                delivered_audio_chunks=self._delivered_audio,
                delivered_image_chunks=self._delivered_images,
                dropped_image_chunks=self._dropped_images,
                audio_backpressure_waits=self._audio_waits,
                max_observed_depth=self._max_depth,
                current_depth=len(self._chunks),
            )
