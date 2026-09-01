from __future__ import annotations

import pytest

from coregulation_poc.control.intervention_policy import load_intervention_policy
from coregulation_poc.control.state_tracker import StateTrajectoryController
from coregulation_poc.models import (
    ControlObservation,
    CoregulationState,
    InterventionAction,
    InterventionDecisionReason,
    RecoveryStatus,
    StateAssessment,
)


def _insufficient_modality(reason: str) -> dict[str, object]:
    return {"sufficiency": "insufficient", "items": [], "limitation_reason": reason}


def _assessment(
    state: str | None,
    assessed_at_ms: int,
    *,
    session_id: str = "demo",
    confidence: str = "high",
    evidence_sufficient: bool = True,
    previous_state: str | None = None,
    performance: str | None = None,
    trajectory: str = "stable",
    interruptibility: str = "natural_pause",
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
                "audio": _insufficient_modality("Audio evidence is insufficient."),
                "video": _insufficient_modality("Video evidence is insufficient."),
            },
            reason="No state can be supported.",
        )

    performances = {
        "normal": ["normal task progression"],
        "fluctuation": ["brief task stall"],
        "dysregulation": ["pace conflict"],
        "high_risk": ["persistent interaction imbalance"],
    }
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
    task_process = _task_process_defaults.get(state, "unclear") if state else None
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
        interaction_performance=[performance] if performance else performances[state],
        task_process=task_process,
        support_need=support_need,
        support_target="parent",
        interruptibility=interruptibility,
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "parent",
                        "start_ms": max(0, assessed_at_ms - 500),
                        "end_ms": assessed_at_ms,
                        "code": "observed_interaction",
                        "observation": "The parent produces an observable task-related utterance.",
                        "quote": "再想一想",
                    }
                ],
            },
            "video": _insufficient_modality("Relevant behavior is outside the frame."),
        },
        reason="The observed trajectory supports the selected state.",
    )


def _observation(
    state: str | None,
    assessed_at_ms: int,
    *,
    boundary: bool,
    response_observed: bool = False,
    history_available: bool = False,
    confidence: str = "high",
    evidence_sufficient: bool = True,
    session_id: str = "demo",
    previous_state: str | None = None,
    performance: str | None = None,
    trajectory: str = "stable",
    interruptibility: str = "natural_pause",
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state,
            assessed_at_ms,
            session_id=session_id,
            confidence=confidence,
            evidence_sufficient=evidence_sufficient,
            previous_state=previous_state,
            performance=performance,
            trajectory=trajectory,
            interruptibility=interruptibility,
        ),
        natural_turn_boundary=boundary,
        post_intervention_response_observed=response_observed,
        interaction_history_available=history_available,
    )


def _controller() -> StateTrajectoryController:
    return StateTrajectoryController(load_intervention_policy())


@pytest.mark.parametrize(
    ("state", "expected_action", "expected_reason"),
    [
        ("normal", InterventionAction.NO_INTERVENTION, "normal_coordination"),
        ("fluctuation", InterventionAction.NO_INTERVENTION, "self_recovery_possible"),
    ],
)
def test_normal_and_fluctuation_do_not_intervene(
    state: str,
    expected_action: InterventionAction,
    expected_reason: str,
) -> None:
    decision = _controller().ingest(_observation(state, 1000, boundary=False))

    assert decision.action is expected_action
    assert decision.reason_code == expected_reason
    assert decision.intervention_permitted is False


def test_positive_maintenance_requires_event_pause_and_is_not_a_recovery_guard() -> None:
    controller = _controller()

    waiting = controller.ingest(
        _observation(
            "normal",
            1000,
            boundary=False,
            performance="task completion",
        )
    )
    reinforced = controller.ingest(
        _observation(
            "normal",
            2000,
            boundary=True,
            performance="task completion",
        )
    )
    controller.mark_intervention_delivered(
        CoregulationState.NORMAL,
        delivered_at_ms=2000,
    )
    repeated = controller.ingest(
        _observation(
            "normal",
            3000,
            boundary=True,
            response_observed=True,
            performance="task completion",
        )
    )
    controller.ingest(
        _observation(
            "normal",
            60_000,
            boundary=True,
            performance="steady coordination",
        )
    )
    later_event = controller.ingest(
        _observation(
            "normal",
            122_000,
            boundary=True,
            performance="task completion",
        )
    )

    assert waiting.action is InterventionAction.HOLD
    assert waiting.reason_code is InterventionDecisionReason.WAITING_FOR_NATURAL_TURN_BOUNDARY
    assert reinforced.action is InterventionAction.REINFORCE
    assert reinforced.reason_code is InterventionDecisionReason.POSITIVE_MAINTENANCE_OPPORTUNITY
    assert controller.awaiting_post_intervention_response is False
    assert repeated.action is InterventionAction.NO_INTERVENTION
    assert later_event.action is InterventionAction.REINFORCE


def test_positive_maintenance_has_independent_cooldown_and_never_blocks_dysregulation() -> None:
    controller = _controller()
    reinforced = controller.ingest(
        _observation(
            "normal",
            1000,
            boundary=True,
            performance="task completion",
        )
    )
    assert reinforced.action is InterventionAction.REINFORCE
    controller.mark_intervention_delivered(
        CoregulationState.NORMAL,
        delivered_at_ms=2000,
    )

    within_cooldown = controller.ingest(
        _observation(
            "normal",
            60_000,
            boundary=True,
            performance="positive dyadic exchange",
        )
    )
    dysregulation = controller.ingest(
        _observation("dysregulation", 61_000, boundary=False)
    )

    assert within_cooldown.action is InterventionAction.NO_INTERVENTION
    assert dysregulation.action is InterventionAction.INTERVENE
    assert dysregulation.reason_code is InterventionDecisionReason.DYAD_CANNOT_SELF_RECOVER


def test_self_recovery_from_fluctuation_remains_non_intervention() -> None:
    controller = _controller()
    controller.ingest(_observation("fluctuation", 1000, boundary=False))

    decision = controller.ingest(
        _observation("normal", 2000, boundary=True, trajectory="recovering")
    )

    assert decision.action is InterventionAction.NO_INTERVENTION
    assert decision.reason_code is InterventionDecisionReason.NORMAL_COORDINATION


def test_dysregulation_is_authorized_without_waiting_for_turn_boundary() -> None:
    controller = _controller()

    waiting = controller.ingest(_observation("dysregulation", 1000, boundary=False))
    assert waiting.action is InterventionAction.INTERVENE
    assert waiting.reason_code is InterventionDecisionReason.DYAD_CANNOT_SELF_RECOVER
    assert waiting.strategy_selection_required is True
    assert controller.awaiting_post_intervention_response is True


def test_controller_waits_for_response_then_records_recovery() -> None:
    controller = _controller()
    controller.ingest(_observation("dysregulation", 1000, boundary=True))

    waiting = controller.ingest(_observation("fluctuation", 2000, boundary=False))
    recovered = controller.ingest(
        _observation("normal", 3000, boundary=False, response_observed=True)
    )

    assert waiting.action is InterventionAction.HOLD
    assert waiting.recovery_status is RecoveryStatus.PENDING
    assert recovered.action is InterventionAction.NO_INTERVENTION
    assert recovered.recovery_status is RecoveryStatus.RECOVERED
    assert controller.awaiting_post_intervention_response is False


def test_high_risk_requires_history_before_progressive_support() -> None:
    controller = _controller()

    held = controller.ingest(_observation("high_risk", 1000, boundary=True))

    assert held.action is InterventionAction.HOLD
    assert held.reason_code is InterventionDecisionReason.HISTORY_REQUIRED

    allowed = controller.ingest(
        _observation("high_risk", 2000, boundary=True, history_available=True)
    )
    assert allowed.action is InterventionAction.PROGRESSIVE_SUPPORT


def test_high_risk_escalation_bypasses_dysregulation_episode_interval() -> None:
    controller = _controller()
    first = controller.ingest(_observation("dysregulation", 0, boundary=True))
    assert first.action is InterventionAction.INTERVENE
    controller.mark_intervention_delivered(
        CoregulationState.DYSREGULATION,
        delivered_at_ms=0,
    )

    escalated = controller.ingest(
        _observation(
            "high_risk",
            10_000,
            boundary=False,
            response_observed=True,
            history_available=True,
            previous_state="dysregulation",
        )
    )

    assert escalated.action is InterventionAction.PROGRESSIVE_SUPPORT
    assert escalated.reason_code is InterventionDecisionReason.PERSISTENT_HIGH_RISK_PATTERN
    assert escalated.recovery_status is RecoveryStatus.DETERIORATED


def test_insufficient_or_low_confidence_evidence_holds() -> None:
    controller = _controller()
    insufficient = controller.ingest(
        _observation(None, 1000, boundary=True, evidence_sufficient=False)
    )
    low_confidence = controller.ingest(
        _observation("dysregulation", 2000, boundary=True, confidence="low")
    )

    assert insufficient.reason_code is InterventionDecisionReason.INSUFFICIENT_EVIDENCE
    assert low_confidence.reason_code is InterventionDecisionReason.LOW_CONFIDENCE
    assert low_confidence.intervention_permitted is False


def test_trajectory_rejects_session_or_time_discontinuity() -> None:
    controller = _controller()
    controller.ingest(_observation("normal", 2000, boundary=False))

    with pytest.raises(ValueError, match="same session_id"):
        controller.ingest(_observation("normal", 3000, boundary=False, session_id="another"))
    with pytest.raises(ValueError, match="non-decreasing"):
        controller.ingest(_observation("normal", 1000, boundary=False))


def test_post_intervention_response_timeout() -> None:
    controller = _controller()
    controller.ingest(_observation("dysregulation", 1000, boundary=True))

    waiting = None
    for index in range(1, 12):
        waiting = controller.ingest(
            _observation("dysregulation", 1000 + index * 1000, boundary=True)
        )
        assert waiting.action is InterventionAction.HOLD
        assert waiting.recovery_status is RecoveryStatus.PENDING

    timeout_decision = controller.ingest(
        _observation("dysregulation", 13_000, boundary=True)
    )
    assert timeout_decision.recovery_status is RecoveryStatus.TIMEOUT
    assert timeout_decision.action is InterventionAction.HOLD
    assert timeout_decision.reason_code is (
        InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
    )


def test_same_episode_reintervenes_after_120_seconds_without_escalation() -> None:
    controller = _controller()
    first = controller.ingest(_observation("dysregulation", 0, boundary=True))
    assert first.action is InterventionAction.INTERVENE

    cooldown = controller.ingest(
        _observation(
            "dysregulation",
            10_000,
            boundary=True,
            response_observed=True,
        )
    )
    unchanged = controller.ingest(_observation("dysregulation", 120_000, boundary=True))

    assert cooldown.reason_code is InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
    assert unchanged.action is InterventionAction.INTERVENE


def test_same_episode_wait_starts_when_intervention_is_delivered() -> None:
    controller = _controller()
    first = controller.ingest(_observation("dysregulation", 0, boundary=True))
    assert first.action is InterventionAction.INTERVENE

    # The model and message pipeline take 20 seconds before the browser confirms
    # that the intervention is visible. The family must still receive a complete
    # 120-second observation period after that confirmation.
    controller.mark_intervention_delivered(
        CoregulationState.DYSREGULATION,
        delivered_at_ms=20_000,
    )
    still_waiting = controller.ingest(
        _observation(
            "dysregulation",
            120_000,
            boundary=True,
            response_observed=True,
        )
    )
    ready = controller.ingest(
        _observation(
            "dysregulation",
            140_000,
            boundary=True,
        )
    )

    assert still_waiting.reason_code is (
        InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
    )
    assert ready.action is InterventionAction.INTERVENE


def test_same_episode_can_repeat_beyond_two_interventions_with_spacing() -> None:
    controller = _controller()
    first = controller.ingest(_observation("dysregulation", 0, boundary=True))
    first_wait = controller.ingest(
        _observation(
            "dysregulation",
            10_000,
            boundary=True,
            response_observed=True,
        )
    )
    second = controller.ingest(
        _observation(
            "dysregulation",
            120_000,
            boundary=True,
        )
    )
    second_wait = controller.ingest(
        _observation(
            "dysregulation",
            130_000,
            boundary=True,
            response_observed=True,
        )
    )
    third = controller.ingest(_observation("dysregulation", 240_000, boundary=True))

    assert first.action is InterventionAction.INTERVENE
    assert first_wait.reason_code is InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
    assert second.action is InterventionAction.INTERVENE
    assert second_wait.reason_code is InterventionDecisionReason.SAME_EPISODE_OBSERVATION_PERIOD
    assert third.action is InterventionAction.INTERVENE
