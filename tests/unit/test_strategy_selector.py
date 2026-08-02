from __future__ import annotations

import json
from pathlib import Path

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import StrategySelectionStatus
from coregulation_poc.models import (
    Actor,
    ControlObservation,
    RecoveryStatus,
    StateAssessment,
    TrajectoryReplayRequest,
)
from coregulation_poc.settings import Settings
from coregulation_poc.strategy_test import run_strategy_test


def _insufficient_video() -> dict[str, object]:
    return {
        "sufficiency": "insufficient",
        "items": [],
        "limitation_reason": "Relevant behavior is outside the frame.",
    }


def _assessment(
    *,
    state: str,
    performance: str,
    actor: str,
    assessed_at_ms: int,
    session_id: str = "strategy-demo",
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


def _observation(
    *,
    state: str,
    performance: str,
    actor: str,
    assessed_at_ms: int,
    response_observed: bool = False,
    history_available: bool = False,
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state=state,
            performance=performance,
            actor=actor,
            assessed_at_ms=assessed_at_ms,
        ),
        natural_turn_boundary=True,
        post_intervention_response_observed=response_observed,
        interaction_history_available=history_available,
    )


def _controller() -> StateTrajectoryController:
    return StateTrajectoryController(load_intervention_policy())


def _selector() -> StrategySelector:
    return StrategySelector(load_strategy_library())


def test_library_separates_parent_child_and_both_cards() -> None:
    library = load_strategy_library()

    assert {card.target_actor for card in library.cards} == {
        Actor.PARENT,
        Actor.CHILD,
        Actor.BOTH,
    }
    assert all(
        state.value not in {"normal", "fluctuation"}
        for card in library.cards
        for state in card.states
    )


def test_module_three_does_not_override_module_two_non_intervention() -> None:
    observation = _observation(
        state="normal",
        performance="normal task progression",
        actor="both",
        assessed_at_ms=1000,
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.status is StrategySelectionStatus.HELD
    assert result.hold_reason == "module_two_did_not_authorize"


def test_parent_pace_evidence_selects_parent_strategy() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
        assessed_at_ms=1000,
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "PARENT_TONE_AND_PACE"
    assert result.plan.target_actor is Actor.PARENT
    assert all(result.plan.validation_checks.values())


def test_unknown_actor_uses_neutral_dyadic_fallback() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="unknown",
        assessed_at_ms=1000,
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "DYAD_PACE_RESET"
    assert result.plan.target_actor is Actor.BOTH


def test_child_task_evidence_selects_child_needs_card() -> None:
    observation = _observation(
        state="dysregulation",
        performance="sustained task stall",
        actor="child",
        assessed_at_ms=1000,
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "CHILD_NEEDS_INQUIRY"
    assert result.plan.target_actor is Actor.CHILD


def test_child_escalation_selects_child_support_without_guessing_parent() -> None:
    observation = _observation(
        state="dysregulation",
        performance="escalating negative interaction",
        actor="child",
        assessed_at_ms=1000,
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "CHILD_AFFECT_SOOTHE"
    assert result.plan.target_actor is Actor.CHILD


def test_high_risk_progresses_only_after_observed_non_recovery() -> None:
    controller = _controller()
    selector = _selector()
    first_observation = _observation(
        state="high_risk",
        performance="persistent interaction imbalance",
        actor="parent",
        assessed_at_ms=1000,
        history_available=True,
    )
    first_decision = controller.ingest(first_observation)
    first = selector.select(
        assessment=first_observation.assessment,
        decision=first_decision,
    )
    assert first.plan is not None
    assert first.plan.strategy_id == "PARENT_AUTONOMY_SPACE"

    second_observation = _observation(
        state="high_risk",
        performance="persistent interaction imbalance",
        actor="parent",
        assessed_at_ms=2000,
        response_observed=True,
        history_available=True,
    )
    second_decision = controller.ingest(second_observation)
    second = selector.select(
        assessment=second_observation.assessment,
        decision=second_decision,
        previous_plan=first.plan,
    )

    assert second_decision.recovery_status is RecoveryStatus.NOT_RECOVERED
    assert second.plan is not None
    assert second.plan.strategy_id == "DYAD_ROLE_RESTART"
    assert second.plan.previous_strategy_id == "PARENT_AUTONOMY_SPACE"


def test_deterioration_uses_approved_dyadic_brake() -> None:
    controller = _controller()
    selector = _selector()
    first_observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
        assessed_at_ms=1000,
    )
    first_decision = controller.ingest(first_observation)
    first = selector.select(
        assessment=first_observation.assessment,
        decision=first_decision,
    )
    assert first.plan is not None

    second_observation = _observation(
        state="high_risk",
        performance="sustained strong resistance or withdrawal",
        actor="child",
        assessed_at_ms=2000,
        response_observed=True,
        history_available=True,
    )
    second_decision = controller.ingest(second_observation)
    second = selector.select(
        assessment=second_observation.assessment,
        decision=second_decision,
        previous_plan=first.plan,
    )

    assert second_decision.recovery_status is RecoveryStatus.DETERIORATED
    assert second.plan is not None
    assert second.plan.strategy_id == "DYAD_AFFECT_BRAKE"


def test_strategy_replay_writes_auditable_plans(tmp_path: Path) -> None:
    observations = [
        _observation(
            state="normal",
            performance="normal task progression",
            actor="both",
            assessed_at_ms=1000,
        ),
        _observation(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=2000,
        ),
    ]
    request = TrajectoryReplayRequest(
        session_id="strategy-demo",
        observations=observations,
    )
    input_path = tmp_path / "strategy.json"
    input_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

    run_dir, valid = run_strategy_test(
        input_path=input_path,
        settings=Settings(output_dir=tmp_path / "output"),
    )

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    plans = json.loads(
        (run_dir / "intervention_plans.json").read_text(encoding="utf-8")
    )
    assert valid is True
    assert result["intervention_plan_count"] == 1
    assert plans[0]["target_actor"] == "parent"
    assert plans[0]["strategy_id"] == "PARENT_TONE_AND_PACE"
