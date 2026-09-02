from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from coregulation_poc.intervention import load_strategy_library
from coregulation_poc.runtime.session import RealtimeBrowserSession

ResearchEventSender = Callable[[dict[str, Any]], Awaitable[None]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    safe = dict(event)
    safe.pop("audio_base64", None)
    return safe


@dataclass(slots=True)
class MonitoredSession:
    session_id: str
    runtime: RealtimeBrowserSession | None
    closed_loop_was_enabled: bool
    status: str = "connected"
    connected_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    latest_state: str | None = None
    latest_confidence: str | None = None
    latest_action: str | None = None
    latest_trajectory: str | None = None
    latest_task_process: str | None = None
    latest_support_need: str | None = None
    latest_support_target: str | None = None
    latest_interruptibility: str | None = None
    latest_speaker_binding: dict[str, Any] | None = None
    expert_takeover_active: bool = False
    timeline: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))

    def public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "connected_at": self.connected_at,
            "updated_at": self.updated_at,
            "closed_loop_enabled": self.runtime is not None,
            "closed_loop_was_enabled": self.closed_loop_was_enabled,
            "latest_state": self.latest_state,
            "latest_confidence": self.latest_confidence,
            "latest_action": self.latest_action,
            "latest_trajectory": self.latest_trajectory,
            "latest_task_process": self.latest_task_process,
            "latest_support_need": self.latest_support_need,
            "latest_support_target": self.latest_support_target,
            "latest_interruptibility": self.latest_interruptibility,
            "latest_speaker_binding": self.latest_speaker_binding,
            "expert_takeover_active": self.expert_takeover_active,
            "timeline": list(self.timeline),
        }


class ResearchSessionRegistry:
    """In-memory supervision view. Raw audio and images never enter this registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, MonitoredSession] = {}
        self._subscribers: set[ResearchEventSender] = set()
        self._lock = asyncio.Lock()
        library = load_strategy_library()
        self.strategies = [
            {
                "strategy_id": card.strategy_id,
                "name": card.name,
                "target_actor": card.target_actor.value,
                "repair_target": card.repair_target.value,
                "approved_template": card.approved_template,
            }
            for card in library.cards
        ]

    async def register(
        self,
        session_id: str,
        runtime: RealtimeBrowserSession | None,
    ) -> None:
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is None or existing.status in {"completed", "failed"}:
                self._sessions[session_id] = MonitoredSession(
                    session_id=session_id,
                    runtime=runtime,
                    closed_loop_was_enabled=runtime is not None,
                )
            else:
                existing.runtime = runtime
                existing.closed_loop_was_enabled = (
                    existing.closed_loop_was_enabled or runtime is not None
                )
                existing.status = "connected"
                existing.updated_at = _utc_now()
        await self.broadcast_snapshot()

    async def mark_status(self, session_id: str, status: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.status = status
            session.updated_at = _utc_now()
            if status in {"completed", "failed", "disconnected"}:
                session.runtime = None
                session.expert_takeover_active = False
        await self.broadcast_snapshot()

    async def observe(self, session_id: str, event: dict[str, Any]) -> None:
        safe = _safe_event(event)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            event_type = str(safe.get("type", "event"))
            session.updated_at = _utc_now()
            if event_type == "state_update":
                session.latest_state = safe.get("state")
                session.latest_confidence = safe.get("confidence")
                session.latest_action = safe.get("action")
                session.latest_trajectory = safe.get("trajectory")
                session.latest_task_process = safe.get("task_process")
                session.latest_support_need = safe.get("support_need")
                session.latest_support_target = safe.get("support_target")
                session.latest_interruptibility = safe.get("interruptibility")
            elif event_type == "speaker_enrollment":
                session.latest_speaker_binding = {
                    "enrolled": bool(safe.get("complete")),
                    "bound": None,
                    "method": safe.get("method"),
                }
            elif event_type == "speaker_binding":
                session.latest_speaker_binding = {
                    "enrolled": True,
                    "bound": bool(safe.get("bound")),
                    "method": safe.get("method"),
                    "parent_count": safe.get("parent_segment_count", 0),
                    "child_count": safe.get("child_segment_count", 0),
                    "segment_count": safe.get("segment_count", 0),
                    "low_confidence_count": safe.get(
                        "low_confidence_segment_count", 0
                    ),
                    "continuity_assisted_count": safe.get(
                        "continuity_assisted_segment_count", 0
                    ),
                }
            elif event_type == "expert_takeover_started":
                session.expert_takeover_active = True
            elif event_type == "expert_takeover_ended":
                session.expert_takeover_active = False
            session.timeline.append(
                {
                    **safe,
                    "observed_at": session.updated_at,
                }
            )
        await self._broadcast(
            {
                "type": "session_event",
                "session_id": session_id,
                "event": safe,
            }
        )

    async def subscribe(self, sender: ResearchEventSender) -> None:
        async with self._lock:
            self._subscribers.add(sender)

    async def unsubscribe(self, sender: ResearchEventSender) -> None:
        async with self._lock:
            self._subscribers.discard(sender)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            sessions = [session.public_dict() for session in self._sessions.values()]
        sessions.sort(key=lambda item: item["updated_at"], reverse=True)
        return {
            "type": "research_snapshot",
            "sessions": sessions,
            "strategies": self.strategies,
            "raw_media_available": False,
        }

    async def broadcast_snapshot(self) -> None:
        await self._broadcast(await self.snapshot())

    async def handle_control(self, control: dict[str, Any]) -> bool:
        session_id = control.get("session_id")
        if not isinstance(session_id, str):
            raise ValueError("研究端操作缺少会话编号")
        async with self._lock:
            monitored = self._sessions.get(session_id)
            runtime = None if monitored is None else monitored.runtime
        if monitored is None:
            raise ValueError("目标会话不存在或已经离线")
        if runtime is None:
            raise ValueError("该会话未启用实时闭环，无法进行专家介入")
        forwarded = {key: value for key, value in control.items() if key != "session_id"}
        return await runtime.handle_control(forwarded)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers)
        failed: list[ResearchEventSender] = []
        for sender in subscribers:
            try:
                await sender(payload)
            except (ConnectionError, RuntimeError):
                failed.append(sender)
        if failed:
            async with self._lock:
                for sender in failed:
                    self._subscribers.discard(sender)
