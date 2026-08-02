from __future__ import annotations

import base64
import json
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import websocket

from coregulation_poc.providers.websocket_transport import create_websocket_connection


@dataclass(frozen=True)
class SpeechSynthesisResult:
    pcm_audio: bytes
    model: str
    voice: str
    sample_rate_hz: int
    text: str
    session_id: str | None
    response_id: str | None
    usage_characters: int | None
    first_audio_latency_ms: int | None
    total_latency_ms: int
    events: list[dict[str, Any]]


class QwenRealtimeTTSProvider:
    """Synthesize an approved intervention message without changing its wording."""

    def __init__(
        self,
        *,
        model: str,
        voice: str,
        api_key: str,
        base_url: str,
        language_type: str = "Chinese",
        response_format: str = "pcm",
        sample_rate_hz: int = 24000,
        mode: str = "commit",
        instructions: str,
        optimize_instructions: bool = False,
        workspace_id: str | None = None,
        connection_timeout_seconds: float = 20,
        response_timeout_seconds: float = 90,
    ) -> None:
        if response_format != "pcm":
            raise ValueError("Qwen realtime TTS output must be PCM for traceable WAV storage")
        if mode != "commit":
            raise ValueError("Qwen realtime TTS must use commit mode")
        if optimize_instructions:
            raise ValueError("experiment voice instructions must not be automatically rewritten")
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.base_url = base_url.rstrip("?")
        self.language_type = language_type
        self.response_format = response_format
        self.sample_rate_hz = sample_rate_hz
        self.mode = mode
        self.instructions = instructions
        self.optimize_instructions = optimize_instructions
        self.workspace_id = workspace_id
        self.connection_timeout_seconds = connection_timeout_seconds
        self.response_timeout_seconds = response_timeout_seconds

    @property
    def endpoint(self) -> str:
        separator = "&" if "?" in self.base_url else "?"
        return f"{self.base_url}{separator}{urlencode({'model': self.model})}"

    @staticmethod
    def _event(event_type: str, **payload: Any) -> dict[str, Any]:
        return {
            "event_id": f"event_{uuid4().hex}",
            "type": event_type,
            **payload,
        }

    @property
    def supports_instructions(self) -> bool:
        return "instruct" in self.model

    def _session_update_event(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "voice": self.voice,
            "mode": self.mode,
            "language_type": self.language_type,
            "response_format": self.response_format,
            "sample_rate": self.sample_rate_hz,
        }
        if self.supports_instructions:
            session["instructions"] = self.instructions
            session["optimize_instructions"] = self.optimize_instructions
        return self._event("session.update", session=session)

    def _input_text_append_event(self, text: str) -> dict[str, Any]:
        return self._event("input_text_buffer.append", text=text)

    def _input_text_commit_event(self) -> dict[str, Any]:
        return self._event("input_text_buffer.commit")

    def _session_finish_event(self) -> dict[str, Any]:
        return self._event("session.finish")

    def _headers(self) -> list[str]:
        headers = [f"Authorization: Bearer {self.api_key}"]
        if self.workspace_id:
            headers.append(f"X-DashScope-WorkSpace: {self.workspace_id}")
        return headers

    @staticmethod
    def _receive(connection: websocket.WebSocket) -> dict[str, Any]:
        while True:
            opcode, frame = connection.recv_data_frame(control_frame=True)
            if opcode == websocket.ABNF.OPCODE_CLOSE:
                close_data = frame.data if isinstance(frame.data, bytes) else b""
                close_code = int.from_bytes(close_data[:2], "big") if len(close_data) >= 2 else None
                close_reason = close_data[2:].decode("utf-8", errors="replace")
                raise ConnectionError(
                    "Qwen TTS WebSocket closed before synthesis completed "
                    f"(code={close_code}, reason={close_reason or 'none'})"
                )
            if opcode in {websocket.ABNF.OPCODE_PING, websocket.ABNF.OPCODE_PONG}:
                continue
            if opcode != websocket.ABNF.OPCODE_TEXT:
                raise ConnectionError("Qwen TTS returned an unexpected binary WebSocket frame")
            raw_message = frame.data
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            break
        if not isinstance(raw_message, str) or not raw_message:
            raise ConnectionError("Qwen TTS returned an empty WebSocket event")
        try:
            event = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Qwen TTS returned a non-JSON WebSocket event") from exc
        if not isinstance(event, dict):
            raise ConnectionError("Qwen TTS returned an invalid WebSocket event")
        return event

    @staticmethod
    def _send(connection: websocket.WebSocket, event: dict[str, Any]) -> None:
        # Match the official DashScope SDK wire representation for Chinese text.
        connection.send(json.dumps(event))

    @staticmethod
    def _sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(event)
        if event.get("type") == "response.audio.delta" and isinstance(event.get("delta"), str):
            sanitized["audio_bytes"] = len(base64.b64decode(event["delta"], validate=True))
            sanitized.pop("delta", None)
        return sanitized

    def _wait_for(
        self,
        connection: websocket.WebSocket,
        expected_type: str,
        *,
        deadline: float,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        while time.monotonic() < deadline:
            try:
                event = self._receive(connection)
            except websocket.WebSocketTimeoutException:
                continue
            events.append(self._sanitize_event(event))
            if event.get("type") == "error":
                raise ConnectionError(
                    f"Qwen TTS returned an error: {json.dumps(event, ensure_ascii=False)}"
                )
            if event.get("type") == expected_type:
                return event
        raise TimeoutError(f"Timed out waiting for Qwen TTS event: {expected_type}")

    def synthesize(
        self,
        text: str,
        *,
        on_audio_chunk: Callable[[bytes], None] | None = None,
    ) -> SpeechSynthesisResult:
        if not text.strip():
            raise ValueError("TTS text cannot be empty")

        connection: websocket.WebSocket | None = None
        events: list[dict[str, Any]] = []
        audio = bytearray()
        session_id: str | None = None
        response_id: str | None = None
        usage_characters: int | None = None
        first_audio_latency_ms: int | None = None
        started = time.monotonic()
        try:
            connection = create_websocket_connection(
                self.endpoint,
                header=self._headers(),
                timeout=self.connection_timeout_seconds,
            )
            connection.settimeout(1)
            ready_deadline = time.monotonic() + self.connection_timeout_seconds
            created = self._wait_for(
                connection, "session.created", deadline=ready_deadline, events=events
            )
            session = created.get("session")
            if isinstance(session, dict) and isinstance(session.get("id"), str):
                session_id = session["id"]

            self._send(connection, self._session_update_event())
            updated = self._wait_for(
                connection,
                "session.updated",
                deadline=time.monotonic() + self.connection_timeout_seconds,
                events=events,
            )
            updated_session = updated.get("session")
            if not isinstance(updated_session, dict):
                raise ConnectionError("Qwen TTS session.updated is missing session metadata")
            expected_session = {
                "model": self.model,
                "voice": self.voice,
                "mode": self.mode,
                "response_format": self.response_format,
                "sample_rate": self.sample_rate_hz,
                "language_type": self.language_type,
            }
            mismatches = {
                key: {"expected": expected, "received": updated_session.get(key)}
                for key, expected in expected_session.items()
                if updated_session.get(key) != expected
            }
            if mismatches:
                raise ConnectionError(
                    "Qwen TTS session configuration mismatch: "
                    f"{json.dumps(mismatches, ensure_ascii=False)}"
                )
            self._send(connection, self._input_text_append_event(text))
            self._send(connection, self._input_text_commit_event())

            synthesis_deadline = time.monotonic() + self.response_timeout_seconds
            response_done: dict[str, Any] | None = None
            while time.monotonic() < synthesis_deadline:
                try:
                    event = self._receive(connection)
                except websocket.WebSocketTimeoutException:
                    continue
                events.append(self._sanitize_event(event))
                event_type = event.get("type")
                if event_type == "error":
                    raise ConnectionError(
                        f"Qwen TTS returned an error: {json.dumps(event, ensure_ascii=False)}"
                    )
                if event_type == "response.audio.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        raise ConnectionError("Qwen TTS audio delta is missing base64 audio")
                    try:
                        chunk = base64.b64decode(delta, validate=True)
                    except ValueError as exc:
                        raise ConnectionError("Qwen TTS returned invalid base64 audio") from exc
                    if chunk:
                        if first_audio_latency_ms is None:
                            first_audio_latency_ms = round((time.monotonic() - started) * 1000)
                        audio.extend(chunk)
                        if on_audio_chunk is not None:
                            on_audio_chunk(chunk)
                elif event_type == "response.done":
                    response_done = event
                    break
            if response_done is None:
                raise TimeoutError("Timed out waiting for Qwen TTS response.done")
            if not audio:
                raise ConnectionError("Qwen TTS completed without returning audio")

            response = response_done.get("response")
            if not isinstance(response, dict):
                raise ConnectionError("Qwen TTS response.done is missing response metadata")
            if isinstance(response.get("id"), str):
                response_id = response["id"]
            status = response.get("status")
            if status != "completed":
                raise ConnectionError(f"Qwen TTS synthesis ended with status={status}")
            response_voice = response.get("voice")
            if response_voice != self.voice:
                raise ConnectionError(
                    f"Qwen TTS response voice mismatch: expected={self.voice}, "
                    f"received={response_voice}"
                )
            usage = response.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("characters"), int):
                usage_characters = usage["characters"]
            if usage_characters is None:
                usage = response_done.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("characters"), int):
                    usage_characters = usage["characters"]

            try:
                self._send(connection, self._session_finish_event())
                self._wait_for(
                    connection,
                    "session.finished",
                    deadline=time.monotonic() + min(5, self.connection_timeout_seconds),
                    events=events,
                )
            except (ConnectionError, TimeoutError, websocket.WebSocketException):
                pass

            return SpeechSynthesisResult(
                pcm_audio=bytes(audio),
                model=self.model,
                voice=self.voice,
                sample_rate_hz=self.sample_rate_hz,
                text=text,
                session_id=session_id,
                response_id=response_id,
                usage_characters=usage_characters,
                first_audio_latency_ms=first_audio_latency_ms,
                total_latency_ms=round((time.monotonic() - started) * 1000),
                events=events,
            )
        except (OSError, websocket.WebSocketException) as exc:
            event_types = [str(event.get("type")) for event in events]
            raise ConnectionError(
                f"Qwen TTS WebSocket failed after events={event_types}: {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()


def write_pcm_wav(path: Path, pcm_audio: bytes, sample_rate_hz: int) -> Path:
    """Wrap mono 16-bit PCM from Qwen in a standard, atomic WAV file."""
    if len(pcm_audio) % 2:
        raise ValueError("16-bit PCM byte length must be even")
    path = path.resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm_audio)
    temporary.replace(path)
    return path
