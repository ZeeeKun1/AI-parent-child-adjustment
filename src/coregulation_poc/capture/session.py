from __future__ import annotations

from threading import Event, Thread

from coregulation_poc.capture.buffer import BoundedMediaBuffer
from coregulation_poc.capture.media import MediaChunk, MediaSource, MediaSourceError


class MediaCaptureSession:
    """Run a blocking media source in one owned producer thread."""

    def __init__(
        self,
        *,
        source: MediaSource,
        buffer: BoundedMediaBuffer,
        join_timeout_seconds: float = 5.0,
    ) -> None:
        self.source = source
        self.buffer = buffer
        self.join_timeout_seconds = join_timeout_seconds
        self.stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("capture session has already started")
        self._thread = Thread(
            target=self._produce,
            name="coregulation-media-capture",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        try:
            for chunk in self.source.iter_chunks(self.stop_event):
                if self.stop_event.is_set():
                    break
                self.buffer.put(chunk)
        except MediaSourceError as exc:
            self.buffer.close(exc)
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            self.buffer.close(MediaSourceError(f"Media source failed: {exc}"))
        else:
            self.buffer.close()
        finally:
            try:
                self.source.close()
            except Exception as exc:  # pragma: no cover - defensive adapter boundary
                self.buffer.close(MediaSourceError(f"Media source close failed: {exc}"))

    def read(self, *, timeout: float | None = None) -> MediaChunk | None:
        if self._thread is None:
            raise RuntimeError("capture session has not started")
        return self.buffer.get(timeout=timeout)

    def stop(self) -> None:
        self.stop_event.set()
        close_error: MediaSourceError | None = None
        try:
            self.source.close()
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            close_error = MediaSourceError(f"Media source close failed: {exc}")
        if self._thread is None:
            self.buffer.close(close_error)
            if close_error is not None:
                raise close_error
            return
        self._thread.join(self.join_timeout_seconds)
        if self._thread.is_alive():
            error = MediaSourceError(
                "Capture worker did not stop after the device close request."
            )
            self.buffer.close(error)
            raise error
        if close_error is not None:
            self.buffer.close(close_error)
            raise close_error

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
