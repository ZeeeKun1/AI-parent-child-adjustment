"""Tests 33-37: Family feedback, research端 events, and state ranking.

Verifies that:
- self_continue suppresses one complete analysis window (33)
- Difficulty feedback updates the task context (34)
- dismissed is not treated as acceptance (35)
- Research端 receives new assessment fields (36)
- Post-intervention fluctuation is not recorded as indeterminate (37)
"""
from __future__ import annotations

import asyncio

from coregulation_poc.capture.media import MediaChunk, MediaKind
from coregulation_poc.control import (
    STATE_RANK,
    StateTrajectoryController,
    load_intervention_policy,
)
from coregulation_poc.models import (
    ControlObservation,
    CoregulationState,
    RecoveryStatus,
    StateAssessment,
)
from coregulation_poc.runtime import RealtimeLoopConfig, RealtimeSession
from coregulation_poc.runtime.window import MediaWindow
from coregulation_poc.web.research import ResearchSessionRegistry

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _insufficient_video() -> dict[str, object]:
    return {
        "sufficiency": "insufficient",
        "items": [],
        "limitation_reason": "Relevant behavior is outside the frame.",
    }


def _dysregulation_assessment(
    *,
    session_id: str,
    assessed_at_ms: int,
    previous_state: CoregulationState | None,
) -> StateAssessment:
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state="dysregulation",
        previous_state=previous_state,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=["pace conflict"],
        task_process="pace_mismatch",
        support_need="task_pacing",
        support_target="parent",
        interruptibility="natural_pause",
        boundary_signals={
            "task_stall_observed": True,
            "parental_prompt_count": 1,
            "conflict_action_observed": False,
            "child_disengaged_observed": False,
            "regulation_balance": "one_stable",
        },
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "parent",
                        "start_ms": max(0, assessed_at_ms - 500),
                        "end_ms": assessed_at_ms,
                        "code": "pace conflict",
                        "observation": "The parent is rushing through the steps.",
                        "quote": "快点写完这一步",
                    }
                ],
            },
            "video": _insufficient_video(),
        },
        reason="The parent is pressing the child to move faster.",
    )


class _DysregulationRecognizer:
    """Always returns a dysregulation assessment with parent evidence."""

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
        return _dysregulation_assessment(
            session_id=session_id,
            assessed_at_ms=window.end_ms,
            previous_state=previous_state,
        )


class _NullRecognizer:
    """Recognizer that does nothing; used when analysis is not triggered."""

    api_call_count = 0

    async def assess(self, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("not used in this test")


def _audio(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.AUDIO, timestamp_ms, b"\x00\x00" * 1600)


def _image(timestamp_ms: int) -> MediaChunk:
    return MediaChunk(MediaKind.IMAGE, timestamp_ms, b"jpeg")


async def _reach_confirmed_dysregulation(session: RealtimeSession) -> None:
    for index in range(3):
        await session.accept_chunk(_audio(index * 10_000 + (1 if index else 0)))
        await session.accept_chunk(_image((index + 1) * 10_000))
        await session.analyze_now()


def _create_session(
    *,
    session_id: str = "feedback-test",
    recognizer: object | None = None,
    config: RealtimeLoopConfig | None = None,
) -> RealtimeSession:
    events: list[dict[str, object]] = []

    async def send_event(event: dict[str, object]) -> None:
        events.append(event)

    session = RealtimeSession(
        session_id=session_id,
        recognizer=recognizer or _NullRecognizer(),  # type: ignore[arg-type]
        send_event=send_event,
        config=config or RealtimeLoopConfig(
            assessment_interval_ms=20_000,
            post_intervention_observation_ms=1_000,
            voice_enabled=False,
        ),
        turn_boundary_detector=lambda _: True,
    )
    session._test_events = events  # type: ignore[attr-defined]
    return session


# ---------------------------------------------------------------------------
# Test 33: self_continue suppresses one complete analysis window
# ---------------------------------------------------------------------------

def test_self_continue_suppresses_one_window() -> None:
    asyncio.run(_exercise_self_continue())


async def _exercise_self_continue() -> None:
    session = _create_session(
        recognizer=_DysregulationRecognizer(),
    )
    await session.start()
    await _reach_confirmed_dysregulation(session)

    # Verify intervention was sent
    events = session._test_events  # type: ignore[attr-defined]
    intervention = next(e for e in events if e["type"] == "intervention")
    assert intervention["strategy_id"] == "PARENT_PACE_RESET"

    # Execute delivery
    await session.handle_control(
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

    # Send self_continue feedback
    assert session._self_continue_suppressed is False
    await session.handle_control(
        {
            "type": "family_response",
            "response": "self_continue",
            "delivery_id": intervention["delivery_id"],
            "recorded_at_ms": 30_300,
        }
    )
    assert session._self_continue_suppressed is True

    # Clear previous plan so selector can re-select the same strategy,
    # isolating the self_continue suppression behavior
    session.previous_plan = None

    # Next analysis cycle: intervention should be held
    await session.accept_chunk(_audio(41_300))
    await session.accept_chunk(_image(41_301))
    await session.analyze_now()

    events = session._test_events  # type: ignore[attr-defined]
    held = [e for e in events if e["type"] == "intervention_held"]
    assert any(h["reason"] == "self_continue_suppressed" for h in held)
    assert session._self_continue_suppressed is False


# ---------------------------------------------------------------------------
# Test 34: Difficulty feedback updates task context
# ---------------------------------------------------------------------------

def test_difficulty_feedback_updates_task_context() -> None:
    asyncio.run(_exercise_difficulty_feedback())


async def _exercise_difficulty_feedback() -> None:
    session = _create_session()
    session.set_task_context(
        {
            "task_name": "math",
            "task_type": "math_calculation",
            "task_difficulty": "moderate",
            "child_grade": "3",
        }
    )

    # task_too_hard -> challenging
    await session.handle_control(
        {
            "type": "family_response",
            "response": "task_too_hard",
            "recorded_at_ms": 100,
        }
    )
    assert session._task_context["task_difficulty"] == "challenging"
    assert session._difficulty_feedback_boost is True

    # task_too_easy -> easy
    await session.handle_control(
        {
            "type": "family_response",
            "response": "task_too_easy",
            "recorded_at_ms": 200,
        }
    )
    assert session._task_context["task_difficulty"] == "easy"
    assert session._difficulty_feedback_boost is False

    # task_just_right -> moderate
    await session.handle_control(
        {
            "type": "family_response",
            "response": "task_just_right",
            "recorded_at_ms": 300,
        }
    )
    assert session._task_context["task_difficulty"] == "moderate"
    assert session._difficulty_feedback_boost is False


# ---------------------------------------------------------------------------
# Test 35: dismissed is not treated as acceptance
# ---------------------------------------------------------------------------

def test_dismissed_does_not_change_state() -> None:
    asyncio.run(_exercise_dismissed())


async def _exercise_dismissed() -> None:
    session = _create_session()
    session.set_task_context(
        {
            "task_name": "math",
            "task_type": "math_calculation",
            "task_difficulty": "moderate",
            "child_grade": "3",
        }
    )

    assert session._self_continue_suppressed is False

    await session.handle_control(
        {
            "type": "family_response",
            "response": "dismissed",
            "recorded_at_ms": 100,
        }
    )

    # dismissed should not change any state
    assert session._self_continue_suppressed is False
    assert session._task_context["task_difficulty"] == "moderate"
    assert session._difficulty_feedback_boost is False

    # Verify the response was recorded
    assert len(session.family_responses) == 1
    assert session.family_responses[0]["response"] == "dismissed"


# ---------------------------------------------------------------------------
# Test 36: Research端 receives new assessment fields
# ---------------------------------------------------------------------------

def test_research_registry_receives_new_assessment_fields() -> None:
    asyncio.run(_exercise_research_fields())


async def _exercise_research_fields() -> None:
    registry = ResearchSessionRegistry()
    await registry.register("research-test", runtime=None)

    event = {
        "type": "state_update",
        "sequence": 1,
        "assessed_at_ms": 1000,
        "state": "dysregulation",
        "previous_state": "normal",
        "trajectory": "worsening",
        "confidence": "high",
        "evidence_sufficiency": "sufficient",
        "action": "intervene",
        "reason_code": "dyad_cannot_self_recover",
        "recovery_status": "not_applicable",
        "post_intervention_response_observed": False,
        "task_process": "pace_mismatch",
        "support_need": "task_pacing",
        "support_target": "parent",
        "interruptibility": "natural_pause",
        "interaction_performance": ["pace conflict"],
    }
    await registry.observe("research-test", event)

    snapshot = await registry.snapshot()
    sessions = snapshot["sessions"]
    session = next(s for s in sessions if s["session_id"] == "research-test")

    assert session["latest_state"] == "dysregulation"
    assert session["latest_trajectory"] == "worsening"
    assert session["latest_task_process"] == "pace_mismatch"
    assert session["latest_support_need"] == "task_pacing"
    assert session["latest_support_target"] == "parent"
    assert session["latest_interruptibility"] == "natural_pause"


# ---------------------------------------------------------------------------
# Test 37: Post-intervention fluctuation is not recorded as indeterminate
# ---------------------------------------------------------------------------

def _fluctuation_observation(
    *,
    assessed_at_ms: int,
    previous_state: CoregulationState | None,
    response_observed: bool = False,
) -> ControlObservation:
    return ControlObservation(
        assessment=StateAssessment(
            session_id="rank-test",
            assessed_at_ms=assessed_at_ms,
            state="fluctuation",
            previous_state=previous_state,
            trajectory="recovering",
            evidence_sufficiency="sufficient",
            confidence="high",
            interaction_performance=["brief task stall"],
            task_process="brief_stall",
            support_need="none",
            support_target="parent",
            interruptibility="natural_pause",
            modality_evidence={
                "audio": {
                    "sufficiency": "sufficient",
                    "items": [
                        {
                            "modality": "audio",
                            "actor": "parent",
                            "start_ms": max(0, assessed_at_ms - 500),
                            "end_ms": assessed_at_ms,
                            "code": "brief stall",
                            "observation": "The parent pauses briefly.",
                            "quote": "等一下",
                        }
                    ],
                },
                "video": _insufficient_video(),
            },
            reason="A brief stall is observed.",
        ),
        natural_turn_boundary=True,
        post_intervention_response_observed=response_observed,
        interaction_history_available=True,
    )


def _dysregulation_observation(
    *,
    assessed_at_ms: int,
    previous_state: CoregulationState | None = None,
) -> ControlObservation:
    return ControlObservation(
        assessment=_dysregulation_assessment(
            session_id="rank-test",
            assessed_at_ms=assessed_at_ms,
            previous_state=previous_state,
        ),
        natural_turn_boundary=True,
        post_intervention_response_observed=False,
        interaction_history_available=False,
    )


def test_post_intervention_fluctuation_not_indeterminate() -> None:
    controller = StateTrajectoryController(load_intervention_policy())

    # First observation: dysregulation -> INTERVENE
    intervene = controller.ingest(_dysregulation_observation(assessed_at_ms=1000))
    assert intervene.action.value == "intervene"
    assert controller.awaiting_post_intervention_response is True

    # Second observation: fluctuation with response observed
    # STATE_RANK: normal=0, fluctuation=1, dysregulation=2, high_risk=3
    # dysregulation(2) -> fluctuation(1): rank decreased -> partial_recovery
    recovery = controller.ingest(
        _fluctuation_observation(
            assessed_at_ms=2000,
            previous_state=CoregulationState.DYSREGULATION,
            response_observed=True,
        )
    )

    assert recovery.recovery_status is not RecoveryStatus.INDETERMINATE
    assert recovery.recovery_status is RecoveryStatus.PARTIAL_RECOVERY
    assert controller.awaiting_post_intervention_response is False


def test_state_rank_includes_fluctuation() -> None:
    """Verify STATE_RANK has all four states including fluctuation."""
    assert STATE_RANK[CoregulationState.NORMAL] == 0
    assert STATE_RANK[CoregulationState.FLUCTUATION] == 1
    assert STATE_RANK[CoregulationState.DYSREGULATION] == 2
    assert STATE_RANK[CoregulationState.HIGH_RISK] == 3
