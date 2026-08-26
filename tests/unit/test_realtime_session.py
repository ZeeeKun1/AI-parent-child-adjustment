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
    interruptibility: str = "natural_pause",
) -> StateAssessment:
    is_dysregulated = state == "dysregulation"
    if performance == "pace conflict":
        task_process = "pace_mismatch"
        support_need = "emotional_support"
    elif is_dysregulated:
        task_process = "sustained_stall"
        support_need = "task_pacing"
    else:
        task_process = "smooth_progress"
        support_need = "none"
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
        trajectory="stable",
        task_process=task_process,
        support_need=support_need,
        support_target=actor,
        interruptibility=interruptibility,
        boundary_signals={
            "task_stall_observed": is_dysregulated,
            "parental_prompt_count": 1 if is_dysregulated else 0,
            "conflict_action_observed": False,
            "child_disengaged_observed": False,
            "regulation_balance": "one_stable" if is_dysregulated else "both_stable",
        },
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
        if self.calls <= 3:
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
            performance="steady coordination",
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


class _PersistentActiveSpeechRecognizer:
    def __init__(self) -> None:
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
        del history, history_available
        self.api_call_count += 1
        return _assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            previous_state=previous_state,
            interruptibility="active_speech",
        )


def _audio(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.AUDIO, timestamp_ms, b"\x00\x00" * 1600)


def _image(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.IMAGE, timestamp_ms, b"jpeg")


async def _reach_confirmed_dysregulation(session: RealtimeSession) -> None:
    """Feed three complete valid windows to satisfy the 30-second boundary."""

    for index in range(3):
        await session.accept_chunk(_audio(index * 10_000 + (1 if index else 0)))
        await session.accept_chunk(_image((index + 1) * 10_000))
        await session.analyze_now()


def test_confirmed_intervention_is_queued_until_live_audio_boundary() -> None:
    asyncio.run(_exercise_queued_intervention())


async def _exercise_queued_intervention() -> None:
    events: list[dict[str, object]] = []
    safe_boundary = False

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    def boundary_detector(_: MediaWindow) -> bool:
        return safe_boundary

    session = RealtimeSession(
        session_id="queued_intervention_test",
        recognizer=_PersistentActiveSpeechRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=100_000,
            voice_enabled=False,
        ),
        turn_boundary_detector=boundary_detector,
    )
    await session.start()
    await _reach_confirmed_dysregulation(session)

    assert session.runtime_metrics["intervention_queued"] is True
    assert not any(event["type"] == "intervention" for event in events)
    assert any(
        event["type"] == "intervention_held" and event.get("queued") is True
        for event in events
    )
    assert session.controller.awaiting_post_intervention_response is False

    safe_boundary = True
    await session.accept_chunk(_audio(30_001))

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert session.runtime_metrics["intervention_queued"] is False
    assert session.controller.awaiting_post_intervention_response is True
    assert any(event["type"] == "intervention_queue_released" for event in events)


def test_queued_intervention_is_cancelled_when_dyad_recovers() -> None:
    asyncio.run(_exercise_queue_cancelled_on_recovery())


async def _exercise_queue_cancelled_on_recovery() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="queue_recovery_test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=100_000,
            voice_enabled=False,
        ),
        turn_boundary_detector=lambda _: False,
    )
    await session.start()
    await _reach_confirmed_dysregulation(session)
    assert session.runtime_metrics["intervention_queued"] is True

    await session.accept_chunk(_audio(30_001))
    await session.accept_chunk(_image(40_000))
    await session.analyze_now()

    assert session.runtime_metrics["intervention_queued"] is False
    assert not any(event["type"] == "intervention" for event in events)
    assert any(event["type"] == "intervention_queue_cancelled" for event in events)


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
    await _reach_confirmed_dysregulation(session)

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["target_actor"] == "parent"
    assert intervention["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert intervention["voice_expected"] is False

    handled = await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 30_200,
            "visual": {
                "status": "delivered",
                "started_at_ms": 30_180,
                "completed_at_ms": 30_181,
                "provider": "browser_overlay",
            },
            "voice": {"status": "not_attempted"},
        }
    )
    assert handled is True
    assert session.window.snapshot() is None

    await session.accept_chunk(_audio(41_300))
    await session.accept_chunk(_image(41_301))
    await session.analyze_now()

    updates = [event for event in events if event["type"] == "state_update"]
    assert updates[-1]["state"] == "normal"
    assert updates[-1]["model_state"] == "normal"
    assert updates[-1]["boundary_rule_applied"] is False
    assert "boundary_signals" in updates[-1]
    assert updates[-1]["recovery_status"] == "recovered"
    assert updates[-1]["post_intervention_response_observed"] is True
    assert session.controller.awaiting_post_intervention_response is False
    assert session.api_call_count == 4
    assert len(session.intervention_outcomes) == 1
    outcome = session.intervention_outcomes[0]
    assert outcome["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert outcome["recovery_status"] == "recovered"
    assert outcome["effect_category"] == "positive"
    assert outcome["observed_interaction_performance"] == ["steady coordination"]
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
    await _reach_confirmed_dysregulation(session)
    intervention = next(event for event in events if event["type"] == "intervention")

    await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 30_200,
            "visual": {
                "status": "failed",
                "started_at_ms": 30_180,
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
    await _reach_confirmed_dysregulation(session)

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
    await _reach_confirmed_dysregulation(session)

    assert not any(event["type"] == "intervention" for event in events)
    held = next(event for event in events if event["type"] == "intervention_held")
    assert held["reason"] == "target_actor_evidence_insufficient"
    assert session.controller.awaiting_post_intervention_response is False


def test_expert_takeover_uses_approved_strategy_and_records_family_response() -> None:
    asyncio.run(_exercise_expert_takeover())


async def _exercise_expert_takeover() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="expert_takeover_test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=20_000,
            post_intervention_observation_ms=1_000,
        ),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    card = session.strategy_cards["PARENT_TONE_AND_PACE"]

    assert await session.handle_control(
        {
            "type": "expert_takeover",
            "operator": "E01",
            "reason": "需要人工帮助家庭放慢互动节奏",
            "recorded_at_ms": 10,
        }
    )
    assert session.runtime_metrics["expert_takeover_active"] is True
    assert session.runtime_metrics["interventions_paused"] is True

    assert await session.handle_control(
        {
            "type": "expert_intervention",
            "operator": "E01",
            "reason": "当前节奏冲突持续",
            "strategy_id": card.strategy_id,
            "message": card.approved_template,
        }
    )
    intervention = [event for event in events if event["type"] == "intervention"][-1]
    assert intervention["source"] == "expert"
    assert intervention["repair_target"] == "parent_regulation"

    assert await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 100,
            "visual": {
                "status": "delivered",
                "started_at_ms": 90,
                "completed_at_ms": 90,
                "provider": "browser_overlay",
            },
            "voice": {"status": "not_attempted"},
        }
    )
    assert await session.handle_control(
        {
            "type": "family_response",
            "response": "self_continue",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 120,
        }
    )
    assert session.family_responses[-1]["response"] == "self_continue"

    await session.accept_chunk(_audio(1_200))
    await session.accept_chunk(_image(1_201))
    await session.analyze_now()
    outcome = [event for event in events if event["type"] == "intervention_outcome"][-1]
    assert outcome["source"] == "expert"
    assert outcome["strategy_id"] == "PARENT_TONE_AND_PACE"

    assert await session.handle_control(
        {
            "type": "expert_release",
            "operator": "E01",
            "reason": "家庭互动已经可以继续",
            "recorded_at_ms": 1_300,
        }
    )
    assert session.runtime_metrics["expert_takeover_active"] is False
    assert session.runtime_metrics["expert_intervention_count"] == 1
