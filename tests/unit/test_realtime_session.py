from __future__ import annotations

import asyncio
from types import SimpleNamespace

from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.models import CoregulationState, StateAssessment, SupportNeed
from coregulation_poc.runtime import RealtimeLoopConfig, RealtimeSession, RollingMediaWindow
from coregulation_poc.runtime.session import VoiceAudio
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
            "video": (
                {
                    "sufficiency": "sufficient",
                    "items": [
                        {
                            "modality": "video",
                            "actor": "child",
                            "start_ms": assessed_at_ms,
                            "end_ms": assessed_at_ms,
                            "frame_timestamp_ms": assessed_at_ms,
                            "code": "child disengagement",
                            "observation": "The child withdraws while the parent keeps pressing.",
                        }
                    ],
                }
                if is_dysregulated
                else {
                    "sufficiency": "insufficient",
                    "items": [],
                    "limitation_reason": "The relevant behavior is outside the frame.",
                }
            ),
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
            "conflict_action_observed": is_dysregulated,
            "child_disengaged_observed": is_dysregulated,
            "regulation_balance": "both_crossed" if is_dysregulated else "both_stable",
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


class _PositiveRecognizer:
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
        assessment = _assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            state="normal",
            performance="task completion",
            actor="both",
            previous_state=previous_state,
        )
        return assessment.model_copy(
            update={"support_need": SupportNeed.POSITIVE_REINFORCEMENT}
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


class _BlockingDysregulationRecognizer:
    """Let media advance while one actionable judgment is still in flight."""

    def __init__(self) -> None:
        self.api_call_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

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
        self.started.set()
        await self.release.wait()
        return _assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            previous_state=previous_state,
        )


class _SlowStagedRecognizer:
    def __init__(self) -> None:
        self.api_call_count = 0
        self.release = asyncio.Event()
        self.judgment_release = asyncio.Event()
        self.two_judgments_started = asyncio.Event()
        self.observed_windows: list[int] = []
        self.judged_windows: list[int] = []
        self.history_lengths: list[int] = []
        self.active_judgments = 0
        self.max_active_judgments = 0

    async def observe(self, *, session_id: str, window: MediaWindow) -> object:
        self.observed_windows.append(window.end_ms)
        await self.release.wait()
        self.api_call_count += 1
        return SimpleNamespace(
            session_id=session_id,
            window=window,
            speaker_binding=None,
            acoustic_features=None,
        )

    async def judge(
        self,
        *,
        observation: object,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment:
        assert history_available is bool(history)
        window = observation.window
        self.judged_windows.append(window.end_ms)
        self.history_lengths.append(len(history))
        self.api_call_count += 1
        self.active_judgments += 1
        self.max_active_judgments = max(
            self.max_active_judgments,
            self.active_judgments,
        )
        if self.active_judgments >= 2:
            self.two_judgments_started.set()
        try:
            await self.judgment_release.wait()
            return _assessment(
                session_id=observation.session_id,
                assessed_at_ms=window.end_ms,
                state="normal",
                performance="steady coordination",
                actor="both",
                previous_state=previous_state,
            )
        finally:
            self.active_judgments -= 1

    async def assess(self, **_: object) -> StateAssessment:
        raise AssertionError("staged recognition should use observe then judge")


class _BlockingVoiceSynthesizer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def synthesize(self, text: str) -> VoiceAudio:
        assert text
        self.started.set()
        await self.release.wait()
        return VoiceAudio(
            pcm_audio=b"\x00\x00" * 160,
            sample_rate_hz=16_000,
            provider="test_tts",
            output_identifier="voice-1",
        )


def _audio(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.AUDIO, timestamp_ms, b"\x00\x00" * 1600)


def _image(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.IMAGE, timestamp_ms, b"jpeg")


async def _reach_confirmed_dysregulation(session: RealtimeSession) -> None:
    """Feed one model-evidenced actionable dysregulation window."""

    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    await session.analyze_now()


def test_confirmed_intervention_is_immediate_during_active_speech() -> None:
    asyncio.run(_exercise_queued_intervention())


async def _exercise_queued_intervention() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    def boundary_detector(_: MediaWindow) -> bool:
        return False

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

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert session.runtime_metrics["intervention_queued"] is False
    assert session.controller.awaiting_post_intervention_response is True
    assert not any(event["type"] == "intervention_queue_released" for event in events)


def test_actionable_intervention_is_not_lost_when_boundary_is_false() -> None:
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
    assert session.runtime_metrics["intervention_queued"] is False
    assert any(event["type"] == "intervention" for event in events)
    assert not any(event["type"] == "intervention_queue_cancelled" for event in events)


def test_stale_actionable_result_is_recorded_without_obsolete_intervention() -> None:
    asyncio.run(_exercise_latency_without_superseding_judgment())


async def _exercise_latency_without_superseding_judgment() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    recognizer = _BlockingDysregulationRecognizer()
    session = RealtimeSession(
        session_id="latency-not-loss-test",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(
            window_duration_ms=10_000,
            assessment_interval_ms=10_000,
            max_assessments_per_session=1,
            max_intervention_staleness_ms=20_000,
        ),
    )
    await session.start()
    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    await recognizer.started.wait()

    # Capture keeps moving while the model is slow. The state remains part of
    # the research record, but its prompt no longer matches the live moment.
    await session.accept_chunk(_audio(40_000))
    await session.accept_chunk(_image(40_001))
    recognizer.release.set()
    await session.wait_for_analysis()

    assert any(event["type"] == "state_update" for event in events)
    assert not any(event["type"] == "intervention" for event in events)
    assert any(
        event.get("reason") == "stale_assessment_record_only" for event in events
    )
    assert session.runtime_metrics["stale_intervention_count"] == 1
    await session.stop("completed")


def test_rolling_window_is_bounded_and_keeps_both_modalities() -> None:
    window = RollingMediaWindow(duration_ms=3_000)
    window.append(_audio(0))
    window.append(_image(1_000))
    window.append(_audio(4_000))

    snapshot = window.snapshot()

    assert snapshot is not None
    assert [chunk.timestamp_ms for chunk in snapshot.chunks] == [1_000, 4_000]
    assert snapshot.has_both_modalities is True


def test_due_windows_are_not_lost_while_perception_is_slow() -> None:
    asyncio.run(_exercise_no_loss_staged_pipeline())


async def _exercise_no_loss_staged_pipeline() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    recognizer = _SlowStagedRecognizer()
    session = RealtimeSession(
        session_id="no-loss-test",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(
            window_duration_ms=10_000,
            assessment_interval_ms=10_000,
            max_parallel_perception=3,
        ),
    )
    await session.start()

    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    await session.accept_chunk(_audio(10_001))
    await session.accept_chunk(_image(20_000))
    await session.accept_chunk(_audio(20_001))
    await session.accept_chunk(_image(30_000))
    await asyncio.sleep(0)

    assert session.runtime_metrics["scheduled_assessment_count"] == 3
    assert session.assessment_count == 0

    recognizer.release.set()
    await recognizer.two_judgments_started.wait()
    assert recognizer.max_active_judgments == 2
    recognizer.judgment_release.set()
    await session.wait_for_analysis()

    assert recognizer.observed_windows == [10_000, 20_000, 30_000]
    assert sorted(recognizer.judged_windows) == [10_000, 20_000, 30_000]
    # Each model call uses a scheduling-time snapshot. The ordered local
    # controller, not a mutable prompt history, owns cross-window continuity.
    assert recognizer.history_lengths == [0, 0, 0]
    assert session.assessment_count == 3
    updates = [event for event in events if event["type"] == "state_update"]
    assert len(updates) == 3
    assert [event["sequence"] for event in updates] == [1, 2, 3]
    assert [event["previous_state"] for event in updates] == [None, "normal", "normal"]
    await session.stop("completed")


def test_timed_out_window_keeps_ordered_slot_and_late_result_is_record_only() -> None:
    asyncio.run(_exercise_timed_out_ordered_slot())


async def _exercise_timed_out_ordered_slot() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    recognizer = _SlowStagedRecognizer()
    session = RealtimeSession(
        session_id="deadline-test",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=10_000,
            analysis_deadline_seconds=0.02,
        ),
    )
    await session.start()
    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    recognizer.release.set()
    await session.wait_for_analysis()

    update = next(event for event in events if event["type"] == "state_update")
    assert update["sequence"] == 1
    assert update["state"] is None
    assert update["evidence_sufficiency"] == "insufficient"
    assert session.runtime_metrics["analysis_timeout_count"] == 1

    recognizer.judgment_release.set()
    for _ in range(20):
        if any(event["type"] == "late_analysis_result" for event in events):
            break
        await asyncio.sleep(0.01)
    late = next(event for event in events if event["type"] == "late_analysis_result")
    assert late["state"] == "normal"
    assert late["record_only"] is True
    assert session.assessment_count == 1
    await session.stop("completed")


def test_analysis_capacity_is_bounded_without_losing_timeline_slots() -> None:
    asyncio.run(_exercise_bounded_analysis_capacity())


async def _exercise_bounded_analysis_capacity() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    recognizer = _SlowStagedRecognizer()
    session = RealtimeSession(
        session_id="capacity-test",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=10_000,
            max_pending_analysis_jobs=2,
        ),
    )
    await session.start()
    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    for end_ms in (20_000, 30_000, 40_000, 50_000):
        await session.accept_chunk(_audio(end_ms - 9_999))
        await session.accept_chunk(_image(end_ms))

    recognizer.release.set()
    await recognizer.two_judgments_started.wait()
    recognizer.judgment_release.set()
    await session.wait_for_analysis()

    updates = [event for event in events if event["type"] == "state_update"]
    event_summary = [
        (event["type"], event.get("message"), event.get("reason")) for event in events
    ]
    assert not [event for event in events if event["type"] == "loop_error"]
    assert [event["sequence"] for event in updates] == [1, 2, 3, 4, 5], event_summary
    assert len([event for event in updates if event["state"] is None]) == 3
    assert session.runtime_metrics["analysis_capacity_skip_count"] == 3
    assert len([event for event in events if event["type"] == "analysis_skipped"]) == 3
    await session.stop("completed")


def test_visual_intervention_is_sent_before_voice_synthesis_finishes() -> None:
    asyncio.run(_exercise_visual_first_delivery())


async def _exercise_visual_first_delivery() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    voice = _BlockingVoiceSynthesizer()
    session = RealtimeSession(
        session_id="visual-first-test",
        recognizer=_SequenceRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=100_000,
            voice_enabled=True,
            voice_synthesis_timeout_seconds=1,
        ),
        voice_synthesizer=voice,
    )
    await session.start()
    analysis = asyncio.create_task(_reach_confirmed_dysregulation(session))
    await voice.started.wait()

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["voice_pending"] is True
    assert not any(event["type"] == "intervention_voice" for event in events)

    voice.release.set()
    await analysis
    voice_event = next(event for event in events if event["type"] == "intervention_voice")
    assert voice_event["delivery_id"] == intervention["delivery_id"]
    assert isinstance(voice_event["audio_base64"], str)
    await session.stop("completed")


def test_shutdown_cancels_a_stuck_analysis_after_bounded_wait() -> None:
    asyncio.run(_exercise_bounded_shutdown())


async def _exercise_bounded_shutdown() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    recognizer = _BlockingDysregulationRecognizer()
    session = RealtimeSession(
        session_id="bounded-shutdown-test",
        recognizer=recognizer,
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=10_000,
            shutdown_drain_timeout_seconds=0.02,
        ),
    )
    await session.start()
    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    await recognizer.started.wait()
    await asyncio.wait_for(session.stop("completed"), timeout=1)

    assert any(
        event.get("stage") == "shutdown" and event["type"] == "loop_error"
        for event in events
    )


def test_session_runs_four_modules_and_observes_post_intervention_response() -> None:
    asyncio.run(_exercise_four_module_loop())


def test_positive_maintenance_is_observed_without_blocking_corrective_control() -> None:
    asyncio.run(_exercise_positive_maintenance_loop())


async def _exercise_positive_maintenance_loop() -> None:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id="positive_maintenance_test",
        recognizer=_PositiveRecognizer(),
        send_event=send_event,
        config=RealtimeLoopConfig(
            assessment_interval_ms=100_000,
            post_intervention_observation_ms=1_000,
            voice_enabled=False,
        ),
        turn_boundary_detector=lambda _: True,
    )
    await session.start()
    await session.accept_chunk(_audio(0))
    await session.accept_chunk(_image(10_000))
    await session.analyze_now()

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["strategy_id"] == "DYAD_POSITIVE_AFFIRM"
    await session.handle_control(
        {
            "type": "delivery_execution",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 10_200,
            "visual": {
                "status": "delivered",
                "started_at_ms": 10_180,
                "completed_at_ms": 10_181,
                "provider": "browser_overlay",
            },
            "voice": {"status": "not_attempted"},
        }
    )

    assert session.controller.awaiting_post_intervention_response is False
    assert session.pending_delivery is None
    assert session.runtime_metrics["awaiting_positive_maintenance_observation"] is True
    receipt = [
        event for event in events if event["type"] == "delivery_execution_received"
    ][-1]
    assert receipt["post_intervention_observation_armed"] is False
    assert receipt["positive_maintenance_observation_armed"] is True

    await session.accept_chunk(_audio(11_300))
    await session.accept_chunk(_image(21_300))
    await session.analyze_now()

    assert session.runtime_metrics["awaiting_positive_maintenance_observation"] is False
    outcome = session.intervention_outcomes[-1]
    assert outcome["outcome_type"] == "positive_maintenance"
    assert outcome["recovery_status"] == "positive_coordination_maintained"
    assert outcome["observed_state"] == "normal"
    assert len([event for event in events if event["type"] == "intervention"]) == 1


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
    assert session.delivery_reports[-1]["delivery_timeline_ms"] == 30_200
    delivery_receipt = next(
        event for event in events if event["type"] == "delivery_execution_received"
    )
    assert delivery_receipt["delivery_timeline_ms"] == 30_200
    assert session.runtime_metrics["recent_intervention_message_count"] == 1

    await session.accept_chunk(_audio(41_300))
    await session.accept_chunk(_image(41_301))
    await session.analyze_now()

    updates = [event for event in events if event["type"] == "state_update"]
    assert updates[-1]["state"] == "fluctuation"
    assert updates[-1]["model_state"] == "normal"
    assert updates[-1]["boundary_rule_applied"] is True
    assert "boundary_signals" in updates[-1]
    assert updates[-1]["recovery_status"] == "partial_recovery"
    assert updates[-1]["post_intervention_response_observed"] is True
    assert session.controller.awaiting_post_intervention_response is False
    assert session.api_call_count == 2
    assert len(session.intervention_outcomes) == 1
    outcome = session.intervention_outcomes[0]
    assert outcome["strategy_id"] == "PARENT_TONE_AND_PACE"
    assert outcome["recovery_status"] == "partial_recovery"
    assert outcome["effect_category"] == "limited"
    assert outcome["observed_interaction_performance"] == ["brief task stall"]
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
    assert await session.handle_control({"type": "pause_interventions", "recorded_at_ms": 10})
    await _reach_confirmed_dysregulation(session)

    assert session.runtime_metrics["interventions_paused"] is True
    assert not any(event["type"] == "intervention" for event in events)
    assert any(event["type"] == "intervention_held" for event in events)
    assert session.controller.awaiting_post_intervention_response is False

    assert await session.handle_control({"type": "resume_interventions", "recorded_at_ms": 200})
    assert session.runtime_metrics["interventions_paused"] is False
    assert any(event["type"] == "interventions_paused" for event in events)
    assert any(event["type"] == "interventions_resumed" for event in events)


def test_both_actor_routing_uses_safe_dyadic_card() -> None:
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

    intervention = next(event for event in events if event["type"] == "intervention")
    assert intervention["target_actor"] == "both"
    assert intervention["strategy_id"] == "DYAD_RELATIONSHIP_RESET"
    assert session.controller.awaiting_post_intervention_response is True


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
