from __future__ import annotations

import pytest

from coregulation_poc.control.intervention_policy import load_intervention_policy
from coregulation_poc.control.state_tracker import StateTrajectoryController
from coregulation_poc.models import (
    ControlObservation,
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
) -> StateAssessment:
    if not evidence_sufficient:
        return StateAssessment(
            session_id=session_id,
            assessed_at_ms=assessed_at_ms,
            state=None,
            evidence_sufficiency="insufficient",
            confidence="low",
            ambiguity_reason="Observable evidence is insufficient.",
            interaction_performance=[],
            modality_evidence={
                "audio": _insufficient_modality("Audio evidence is insufficient."),
                "video": _insufficient_modality("Video evidence is insufficient."),
            },
            previous_state=previous_state,
            reason="No state can be supported.",
        )

    performances = {
        "normal": ["normal task progression"],
        "fluctuation": ["brief task stall"],
        "dysregulation": ["pace conflict"],
        "high_risk": ["persistent interaction imbalance"],
    }
    ambiguity_reason = None if confidence == "high" else "The state boundary is uncertain."
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state=state,
        evidence_sufficiency="sufficient",
        confidence=confidence,
        ambiguity_reason=ambiguity_reason,
        interaction_performance=performances[state],
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
        previous_state=previous_state,
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
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state,
            assessed_at_ms,
            session_id=session_id,
            confidence=confidence,
            evidence_sufficient=evidence_sufficient,
            previous_state=previous_state,
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
        ("fluctuation", InterventionAction.OBSERVE, "self_recovery_possible"),
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


def test_dysregulation_waits_for_natural_turn_boundary() -> None:
    controller = _controller()

    waiting = controller.ingest(_observation("dysregulation", 1000, boundary=False))
    intervention = controller.ingest(_observation("dysregulation", 2000, boundary=True))

    assert waiting.action is InterventionAction.HOLD
    assert waiting.reason_code is (
        InterventionDecisionReason.WAITING_FOR_NATURAL_TURN_BOUNDARY
    )
    assert intervention.action is InterventionAction.INTERVENE
    assert intervention.strategy_selection_required is True
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
        controller.ingest(
            _observation("normal", 3000, boundary=False, session_id="another")
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        controller.ingest(_observation("normal", 1000, boundary=False))
