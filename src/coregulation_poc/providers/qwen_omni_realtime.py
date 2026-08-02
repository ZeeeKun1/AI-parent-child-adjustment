from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import websocket

from coregulation_poc.providers.websocket_transport import create_websocket_connection


class QwenOmniRealtimeProvider:
    """Native WebSocket adapter for Alibaba Cloud Qwen-Omni-Realtime."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        workspace_id: str,
        base_url: str,
        instructions: str,
        connection_timeout_seconds: float = 20,
        session_ready_timeout_seconds: float = 15,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.base_url = base_url
        self.instructions = instructions
        self.connection_timeout_seconds = connection_timeout_seconds
        self.session_ready_timeout_seconds = session_ready_timeout_seconds
        self.connection: websocket.WebSocket | None = None
        self.pending_events: list[dict[str, Any]] = []

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}?{urlencode({'model': self.model})}"

    def _require_connection(self) -> websocket.WebSocket:
        if self.connection is None or not self.connection.connected:
            raise ConnectionError("WebSocket connection is not open.")
        return self.connection

    @staticmethod
    def _event(event_type: str, **payload: Any) -> dict[str, Any]:
        return {
            "event_id": f"event_{uuid4().hex}",
            "type": event_type,
            **payload,
        }

    def _session_update_event(self) -> dict[str, Any]:
        """Build only fields documented for the Qwen session.update event."""
        return self._event(
            "session.update",
            session={
                "modalities": ["text"],
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": None,
                "instructions": self.instructions,
            },
        )

    async def _send_event(self, event: dict[str, Any]) -> None:
        message = json.dumps(event, ensure_ascii=False)
        try:
            await asyncio.to_thread(self._require_connection().send, message)
        except (OSError, websocket.WebSocketException) as exc:
            raise ConnectionError(f"WebSocket send failed: {exc}") from exc

    async def _receive_event(self) -> dict[str, Any]:
        connection = self._require_connection()
        while True:
            try:
                raw_message = await asyncio.to_thread(connection.recv)
            except websocket.WebSocketTimeoutException:
                await asyncio.sleep(0)
                continue
            except (OSError, websocket.WebSocketException) as exc:
                raise ConnectionError(f"WebSocket receive failed: {exc}") from exc
            if not isinstance(raw_message, str) or not raw_message:
                raise ConnectionError("WebSocket closed before a complete server event arrived.")
            try:
                event = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                raise ConnectionError("Server returned a non-JSON WebSocket event.") from exc
            return {
                "received_monotonic_ns": time.monotonic_ns(),
                "event": event,
            }

    async def _wait_for_event(self, expected_type: str, *, timeout_seconds: float) -> None:
        async with asyncio.timeout(timeout_seconds):
            while True:
                envelope = await self._receive_event()
                self.pending_events.append(envelope)
                event = envelope["event"]
                event_type = event.get("type") if isinstance(event, dict) else None
                if event_type == "error":
                    raise ConnectionError(
                        "Qwen-Omni-Realtime returned an error: "
                        f"{json.dumps(event, ensure_ascii=False)}"
                    )
                if event_type == expected_type:
                    return

    async def connect(self) -> None:
        headers = [
            f"Authorization: Bearer {self.api_key}",
            f"X-DashScope-WorkSpace: {self.workspace_id}",
        ]
        try:
            self.connection = await asyncio.to_thread(
                create_websocket_connection,
                self.endpoint,
                header=headers,
                timeout=self.connection_timeout_seconds,
            )
        except (OSError, websocket.WebSocketException, ConnectionError) as exc:
            raise ConnectionError(f"WebSocket handshake failed: {exc}") from exc
        self.connection.settimeout(1)
        await self._wait_for_event(
            "session.created", timeout_seconds=self.session_ready_timeout_seconds
        )
        await self._send_event(self._session_update_event())
        await self._wait_for_event(
            "session.updated", timeout_seconds=self.session_ready_timeout_seconds
        )

    async def send_audio(self, pcm_bytes: bytes, timestamp_ms: int) -> None:
        del timestamp_ms
        await self._send_event(
            self._event(
                "input_audio_buffer.append",
                audio=base64.b64encode(pcm_bytes).decode("ascii"),
            )
        )

    async def send_frame(self, jpeg_bytes: bytes, timestamp_ms: int) -> None:
        del timestamp_ms
        await self._send_event(
            self._event(
                "input_image_buffer.append",
                image=base64.b64encode(jpeg_bytes).decode("ascii"),
            )
        )

    async def commit_input(self) -> None:
        await self._send_event(self._event("input_audio_buffer.commit"))
        await self._wait_for_event("input_audio_buffer.committed", timeout_seconds=20)

    async def finish_input(self) -> None:
        await self.commit_input()
        await self._send_event(self._event("response.create"))

    async def events(self) -> AsyncIterator[dict[str, object]]:
        while self.pending_events:
            yield self.pending_events.pop(0)
        while True:
            envelope = await self._receive_event()
            yield envelope
            event = envelope["event"]
            if isinstance(event, dict) and event.get("type") in {"response.done", "error"}:
                break

    async def close(self) -> None:
        if self.connection is not None:
            await asyncio.to_thread(self.connection.close)
            self.connection = None
