from __future__ import annotations

from coregulation_poc.control import load_intervention_policy
from coregulation_poc.coordinator import (
    CoordinatorCycleOutcome,
    SessionCoordinator,
    create_coordinator,
)
from coregulation_poc.delivery import DeliveryRuntimeContext, load_delivery_policy
from coregulation_poc.intervention import load_strategy_library
from coregulation_poc.models import (
    Actor,
    EvidenceSufficiency,
    InteractionTrajectory,
    Interruptibility,
    RecoveryStatus,
    StateAssessment,
    SupportNeed,
    TaskProcess,
)

# 100 ms @ 16 kHz, s16, mono = 1600 samples = 3200 bytes
_SILENT_PCM = b"\x00" * 3200


def _insufficient_video() -> dict[str, object]:
    return {
        "sufficiency": "insufficient",
        "items": [],
        "limitation_reason": "Relevant behavior is outside the frame.",
    }


def _assessment(
    *,
    state: str | None,
    performance: str,
    actor: str,
    assessed_at_ms: int,
    session_id: str = "coordinator-demo",
    evidence_sufficient: bool = True,
    confidence: str = "high",
    trajectory: str = "stable",
    interruptibility: str = "natural_pause",
    previous_state: str | None = None,
    support_need: str | None = None,
    task_process: str | None = None,
) -> StateAssessment:
    if not evidence_sufficient:
        return StateAssessment(
            session_id=session_id,
            assessed_at_ms=assessed_at_ms,
            state=None,
            previous_state=None,
            trajectory="unclear",
            evidence_sufficiency="insufficient",
            confidence="low",
            ambiguity_reason="Observable evidence is insufficient.",
            interaction_performance=[],
            task_process=None,
            support_need=None,
            support_target="unknown",
            interruptibility="unclear",
            modality_evidence={
                "audio": {
                    "sufficiency": "insufficient",
                    "items": [],
                    "limitation_reason": "Audio evidence is insufficient.",
                },
                "video": _insufficient_video(),
            },
            reason="No state can be supported.",
        )

    _task_process_defaults = {
        "normal": "smooth_progress",
        "fluctuation": "brief_stall",
        "dysregulation": "pace_mismatch",
        "high_risk": "unclear",
    }
    _support_need_defaults = {
        "normal": "positive_reinforcement",
        "fluctuation": "none",
        "dysregulation": "emotional_support",
        "high_risk": "autonomy_support",
    }
    if task_process is None:
        task_process = _task_process_defaults.get(state, "unclear") if state else None
    if support_need is None:
        support_need = _support_need_defaults.get(state, "unclear") if state else None
    ambiguity_reason = None if confidence == "high" else "The state boundary is uncertain."
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state=state,
        previous_state=previous_state,
        trajectory=trajectory,
        evidence_sufficiency="sufficient",
        confidence=confidence,
        ambiguity_reason=ambiguity_reason,
        interaction_performance=[performance],
        task_process=task_process,
        support_need=support_need,
        support_target=actor,
        interruptibility=interruptibility,
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": actor,
                        "start_ms": max(0, assessed_at_ms - 500),
                        "end_ms": assessed_at_ms,
                        "code": performance,
                        "observation": "A directly observed interaction pattern.",
                        "quote": "我们先看这一步",
                    }
                ],
            },
            "video": _insufficient_video(),
        },
        reason="The observed sequence supports this state and performance.",
    )


def _runtime(prepared_at_ms: int = 0, **kwargs: bool) -> DeliveryRuntimeContext:
    return DeliveryRuntimeContext(
        prepared_at_ms=prepared_at_ms,
        interventions_paused=kwargs.get("interventions_paused", False),
        voice_enabled=kwargs.get("voice_enabled", True),
        voice_available=kwargs.get("voice_available", True),
    )


def _coordinator() -> SessionCoordinator:
    return SessionCoordinator(
        intervention_policy=load_intervention_policy(),
        strategy_library=load_strategy_library(),
        delivery_policy=load_delivery_policy(),
    )


def _feed_silence(coordinator: SessionCoordinator, chunks: int = 5) -> None:
    """Feed silent PCM chunks so the detector reports a turn boundary."""
    for _ in range(chunks):
        coordinator.ingest_audio_chunk(_SILENT_PCM)


def test_normal_state_produces_no_intervention() -> None:
    coordinator = _coordinator()
    _feed_silence(coordinator)
    assessment = _assessment(
        state="normal",
        performance="steady coordination",
        actor="both",
        assessed_at_ms=1000,
    )

    result = coordinator.process(
        assessment=assessment,
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )

    assert result.outcome is CoordinatorCycleOutcome.NO_INTERVENTION
    assert result.selection_result is None
    assert result.delivery_result is None
    assert result.intervention_plan is None
    assert result.post_intervention_response_observed is False
    assert result.natural_turn_boundary is True


def test_dysregulation_delivers_intervention() -> None:
    coordinator = _coordinator()
    _feed_silence(coordinator)
    assessment = _assessment(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
        assessed_at_ms=1000,
    )

    result = coordinator.process(
        assessment=assessment,
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )

    assert result.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED
    assert result.intervention_plan is not None
    assert result.intervention_plan.strategy_id == "PARENT_TONE_AND_PACE"
    assert result.intervention_plan.target_actor is Actor.PARENT
    assert result.delivery_result is not None
    assert result.delivery_result.package is not None
    assert coordinator.awaiting_post_intervention_response is True
    assert coordinator.previous_plan is not None
    assert result.natural_turn_boundary is True


def test_no_turn_boundary_holds_intervention() -> None:
    coordinator = _coordinator()
    # Feed loud audio so no turn boundary is detected
    loud_pcm = (5000).to_bytes(2, "little", signed=True) * 1600
    for _ in range(5):
        coordinator.ingest_audio_chunk(loud_pcm)
    assert coordinator.at_turn_boundary is False

    assessment = _assessment(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
        assessed_at_ms=1000,
        interruptibility="active_speech",
    )

    result = coordinator.process(
        assessment=assessment,
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )

    assert result.natural_turn_boundary is False
    assert result.decision.reason_code.value == "waiting_for_natural_turn_boundary"
    assert result.outcome is CoordinatorCycleOutcome.NO_INTERVENTION


def test_post_intervention_recovery_observed_leads_to_new_decision() -> None:
    coordinator = _coordinator()

    _feed_silence(coordinator)
    first = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )
    assert first.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED

    _feed_silence(coordinator)
    second = coordinator.process(
        assessment=_assessment(
            state="normal",
            performance="steady coordination",
            actor="both",
            assessed_at_ms=2000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=2000),
    )

    assert second.post_intervention_response_observed is True
    assert second.decision.recovery_status is RecoveryStatus.RECOVERED
    assert second.outcome is CoordinatorCycleOutcome.NO_INTERVENTION
    assert coordinator.awaiting_post_intervention_response is False


def test_post_intervention_non_recovery_selects_different_strategy() -> None:
    coordinator = _coordinator()

    _feed_silence(coordinator)
    first = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )
    assert first.intervention_plan is not None
    assert first.intervention_plan.strategy_id == "PARENT_TONE_AND_PACE"

    _feed_silence(coordinator)
    second = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=2000,
            support_need="task_pacing",
            task_process="pace_mismatch",
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=2000),
    )

    assert second.post_intervention_response_observed is True
    assert second.decision.recovery_status is RecoveryStatus.NOT_RECOVERED
    assert second.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED
    assert second.intervention_plan is not None
    assert second.intervention_plan.strategy_id != "PARENT_TONE_AND_PACE"
    assert second.intervention_plan.previous_strategy_id == "PARENT_TONE_AND_PACE"


def test_insufficient_evidence_after_intervention_does_not_observe_response() -> None:
    coordinator = _coordinator()

    _feed_silence(coordinator)
    first = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )
    assert first.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED

    _feed_silence(coordinator)
    second = coordinator.process(
        assessment=_assessment(
            state=None,
            performance="",
            actor="unknown",
            assessed_at_ms=2000,
            evidence_sufficient=False,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=2000),
    )

    assert second.post_intervention_response_observed is False
    assert coordinator.awaiting_post_intervention_response is True
    assert second.outcome is CoordinatorCycleOutcome.NO_INTERVENTION


def test_interventions_paused_holds_delivery_and_preserves_previous_plan() -> None:
    coordinator = _coordinator()

    _feed_silence(coordinator)
    first = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )
    assert first.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED
    first_plan = first.intervention_plan
    assert first_plan is not None

    _feed_silence(coordinator)
    second = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=2000,
            support_need="task_pacing",
            task_process="pace_mismatch",
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=2000, interventions_paused=True),
    )

    assert second.outcome is CoordinatorCycleOutcome.INTERVENTION_HELD
    assert second.delivery_result is not None
    assert second.delivery_result.package is None
    assert coordinator.previous_plan is first_plan


def test_high_risk_requires_interaction_history() -> None:
    coordinator = _coordinator()
    _feed_silence(coordinator)

    result = coordinator.process(
        assessment=_assessment(
            state="high_risk",
            performance="persistent interaction imbalance",
            actor="parent",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )

    assert result.outcome is CoordinatorCycleOutcome.NO_INTERVENTION
    assert result.decision.reason_code.value == "history_required"


def test_complete_cycle_normal_to_dysregulation_to_recovery() -> None:
    coordinator = _coordinator()

    _feed_silence(coordinator)
    cycle_1 = coordinator.process(
        assessment=_assessment(
            state="normal",
            performance="steady coordination",
            actor="both",
            assessed_at_ms=1000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=1000),
    )
    assert cycle_1.outcome is CoordinatorCycleOutcome.NO_INTERVENTION

    _feed_silence(coordinator)
    cycle_2 = coordinator.process(
        assessment=_assessment(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=2000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=2000),
    )
    assert cycle_2.outcome is CoordinatorCycleOutcome.INTERVENTION_DELIVERED
    assert coordinator.awaiting_post_intervention_response is True

    _feed_silence(coordinator)
    cycle_3 = coordinator.process(
        assessment=_assessment(
            state="normal",
            performance="steady coordination",
            actor="both",
            assessed_at_ms=3000,
        ),
        interaction_history_available=False,
        delivery_runtime=_runtime(prepared_at_ms=3000),
    )
    assert cycle_3.post_intervention_response_observed is True
    assert cycle_3.decision.recovery_status is RecoveryStatus.RECOVERED
    assert cycle_3.outcome is CoordinatorCycleOutcome.NO_INTERVENTION
    assert coordinator.awaiting_post_intervention_response is False

    snapshot = coordinator.snapshot()
    assert len(snapshot.points) == 3
    assert snapshot.points[0].state.value == "normal"
    assert snapshot.points[1].state.value == "dysregulation"
    assert snapshot.points[2].state.value == "normal"


def test_create_coordinator_loads_all_configs() -> None:
    coordinator = create_coordinator()

    assert coordinator.session_id is None
    assert coordinator.previous_plan is None
    assert coordinator.awaiting_post_intervention_response is False
    assert coordinator.at_turn_boundary is False


def test_ingest_audio_chunk_returns_boundary_state() -> None:
    coordinator = _coordinator()
    assert coordinator.at_turn_boundary is False

    for i in range(4):
        assert coordinator.ingest_audio_chunk(_SILENT_PCM) is False
    assert coordinator.ingest_audio_chunk(_SILENT_PCM) is True
    assert coordinator.at_turn_boundary is True
