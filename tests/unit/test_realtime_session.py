from __future__ import annotations

import asyncio

from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.models import CoregulationState, StateAssessment
from coregulation_poc.runtime import RealtimeLoopConfig, RealtimeSession, RollingMediaWindow
from coregulation_poc.runtime.window import MediaWindow


def _assessment(
    *,
    session_id: str,
    assessed_at_ms: int,
    state: str,
    performance: str,
    actor: str,
    previous_state: CoregulationState | None,
) -> StateAssessment:
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state=state,
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=[performance],
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": actor,
                        "start_ms": assessed_at_ms,
                        "end_ms": assessed_at_ms,
                        "code": performance,
                        "observation": "Directly observed interaction pattern.",
                        "quote": "我们先看这一步",
                    }
                ],
            },
            "video": {
                "sufficiency": "insufficient",
                "items": [],
                "limitation_reason": "The relevant behavior is outside the frame.",
            },
        },
        previous_state=previous_state,
        reason="The multimodal sequence supports this state.",
    )


class _SequenceRecognizer:
    def __init__(self) -> None:
        self.calls = 0
        self.api_call_count = 0

    async def assess(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment:
        assert history_available is bool(history)
        self.calls += 1
        self.api_call_count += 1
        if self.calls == 1:
            return _assessment(
                session_id=session_id,
                assessed_at_ms=window.end_ms,
                state="dysregulation",
                performance="pace conflict",
                actor="parent",
                previous_state=previous_state,
            )
        return _assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            state="normal",
            performance="normal task progression",
            actor="both",
            previous_state=previous_state,
        )


class _BothActorRecognizer:
    api_call_count = 0

    async def assess(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment:
        del history, history_available
        self.api_call_count += 1
        return _assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            state="dysregulation",
            performance="pace conflict",
            actor="both",
            previous_state=previous_state,
        )


def _audio(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.AUDIO, timestamp_ms, b"\x00\x00" * 1600)


def _image(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.IMAGE, timestamp_ms, b"jpeg")


def test_rolling_window_is_bounded_and_keeps_both_modalities() -> None:
    window = RollingMediaWindow(duration_ms=3_000)
    window.append(_audio(0))
    window.append(_image(1_000))
    window.append(_audio(4_000))

    snapshot = window.snapshot()

    assert snapshot is not None
    assert [chunk.timestamp_ms for chunk in snapshot.chunks] == [1_000, 4_000]
    assert snapshot.has_both_modalities is True


def test_session_runs_four_modules_and_observes_post_intervention_response() -> None:
    asyncio.run(_exercise_four_module_loop())


async def _exercise_four_module_loop() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="closed_loop_test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            window_duration_ms=12_000,
            assessment_interval_ms=20_000,
            post_intervention_observation_ms=1_000,
            voice_enabled=False,
        ),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    await session.accept_chunk(_audio(100))
    await session.accept_chunk(_image(101))
    await session.analyze_now()

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["target_actor"] == "parent"
    assert intervention["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert intervention["voice_expected"] is False

    handled = await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 200,
            "visual": {
                "status": "delivered",
                "started_at_ms": 180,
                "completed_at_ms": 181,
                "provider": "browser_overlay",
            },
            "voice": {"status": "not_attempted"},
        }
    )
    assert handled is True
    assert session.window.snapshot() is None

    await session.accept_chunk(_audio(1_300))
    await session.accept_chunk(_image(1_301))
    await session.analyze_now()

    updates = [event for event in events if event["type"] == "state_update"]
    assert updates[-1]["state"] == "normal"
    assert updates[-1]["recovery_status"] == "recovered"
    assert updates[-1]["post_intervention_response_observed"] is True
    assert session.controller.awaiting_post_intervention_response is False
    assert session.api_call_count == 2
    assert len(session.intervention_outcomes) == 1
    outcome = session.intervention_outcomes[0]
    assert outcome["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert outcome["recovery_status"] == "recovered"
    assert outcome["effect_category"] == "positive"
    assert outcome["observed_interaction_performance"] == ["normal task progression"]
    assert any(event["type"] == "intervention_outcome" for event in events)


def test_failed_delivery_releases_repeat_guard() -> None:
    asyncio.run(_exercise_failed_delivery())


async def _exercise_failed_delivery() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="failed_delivery_test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(assessment_interval_ms=20_000),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    await session.accept_chunk(_audio(100))
    await session.accept_chunk(_image(101))
    await session.analyze_now()
    intervention = next(event for event in events if event["type"] == "intervention")

    await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 200,
            "visual": {
                "status": "failed",
                "started_at_ms": 180,
                "provider": "browser_overlay",
                "error": "render failed",
            },
            "voice": {"status": "not_attempted"},
        }
    )

    assert session.controller.awaiting_post_intervention_response is False
    assert session.pending_delivery is None


def test_pause_and_resume_hold_future_interventions() -> None:
    asyncio.run(_exercise_pause_and_resume())


async def _exercise_pause_and_resume() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="pause_test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(assessment_interval_ms=20_000),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    assert await session.handle_control(
        {"type": "pause_interventions", "recorded_at_ms": 10}
    )
    await session.accept_chunk(_audio(100))
    await session.accept_chunk(_image(101))
    await session.analyze_now()

    assert session.runtime_metrics["interventions_paused"] is True
    assert not any(event["type"] == "intervention" for event in events)
    assert any(event["type"] == "intervention_held" for event in events)
    assert session.controller.awaiting_post_intervention_response is False

    assert await session.handle_control(
        {"type": "resume_interventions", "recorded_at_ms": 200}
    )
    assert session.runtime_metrics["interventions_paused"] is False
    assert any(event["type"] == "interventions_paused" for event in events)
    assert any(event["type"] == "interventions_resumed" for event in events)


def test_held_actor_routing_releases_response_guard() -> None:
    asyncio.run(_exercise_held_actor_routing())


async def _exercise_held_actor_routing() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="actor_hold_test",
        recognizer=_BothActorRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(assessment_interval_ms=20_000),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    await session.accept_chunk(_audio(100))
    await session.accept_chunk(_image(101))
    await session.analyze_now()

    assert not any(event["type"] == "intervention" for event in events)
    held = next(event for event in events if event["type"] == "intervention_held")
    assert held["reason"] == "target_actor_evidence_insufficient"
    assert session.controller.awaiting_post_intervention_response is False
