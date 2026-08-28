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
    task_process: str | None = None,
    support_need: str | None = None,
    support_target: str | None = None,
) -> StateAssessment:
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state=state,
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=[performance],
        task_process=task_process,
        support_need=support_need,
        support_target=support_target if support_target is not None else actor,
        interruptibility="natural_pause",
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
    task_process: str | None = None,
    support_need: str | None = None,
    support_target: str | None = None,
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state=state,
            performance=performance,
            actor=actor,
            assessed_at_ms=assessed_at_ms,
            task_process=task_process,
            support_need=support_need,
            support_target=support_target,
        ),
        natural_turn_boundary=True,
        post_intervention_response_observed=response_observed,
        interaction_history_available=history_available,
    )


def _controller() -> StateTrajectoryController:
    return StateTrajectoryController(load_intervention_policy())


def _selector() -> StrategySelector:
    return StrategySelector(load_strategy_library())


def test_library_contains_parent_child_and_dyadic_cards() -> None:
    library = load_strategy_library()

    assert {card.target_actor for card in library.cards} == {
        Actor.PARENT,
        Actor.CHILD,
        Actor.BOTH,
    }
    assert len(library.cards) == 21


def test_module_three_does_not_override_module_two_non_intervention() -> None:
    observation = _observation(
        state="normal",
        performance="steady coordination",
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
        support_need="emotional_support",
        task_process="pace_mismatch",
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


def test_unknown_actor_uses_safe_dyadic_card() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="unknown",
        assessed_at_ms=1000,
        support_target="both",
        support_need="task_pacing",
        task_process="pace_mismatch",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.target_actor is Actor.BOTH
    assert result.plan.strategy_id == "DYAD_RELATIONSHIP_RESET"


def test_child_task_stall_first_clarifies_need() -> None:
    observation = _observation(
        state="dysregulation",
        performance="sustained task stall",
        actor="child",
        assessed_at_ms=1000,
        support_need="need_expression",
        task_process="sustained_stall",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "CHILD_NEEDS_INQUIRY"
    assert result.plan.target_actor is Actor.CHILD


def test_child_task_support_follows_unresolved_needs_inquiry() -> None:
    controller = _controller()
    selector = _selector()
    first_observation = _observation(
        state="dysregulation",
        performance="sustained task stall",
        actor="child",
        assessed_at_ms=1000,
        support_need="need_expression",
        task_process="sustained_stall",
    )
    first_decision = controller.ingest(first_observation)
    first = selector.select(
        assessment=first_observation.assessment,
        decision=first_decision,
    )
    assert first.plan is not None
    assert first.plan.strategy_id == "CHILD_NEEDS_INQUIRY"

    second_observation = _observation(
        state="dysregulation",
        performance="sustained task stall",
        actor="child",
        assessed_at_ms=121_000,
        response_observed=True,
        history_available=True,
        support_need="learning_support",
        task_process="sustained_stall",
    )
    second_decision = controller.ingest(second_observation)
    second = selector.select(
        assessment=second_observation.assessment,
        decision=second_decision,
        previous_plan=first.plan,
    )

    assert second.plan is not None
    assert second.plan.strategy_id == "CHILD_STRATEGY_SUPPORT"


def test_both_actor_evidence_uses_dyadic_card_not_single_actor_card() -> None:
    observation = _observation(
        state="dysregulation",
        performance="misaligned understanding",
        actor="both",
        assessed_at_ms=1000,
        support_need="mutual_understanding",
        task_process="explanation_mismatch",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "DYAD_RELATIONSHIP_RESET"
    assert result.plan.target_actor is Actor.BOTH


def test_both_actor_evidence_uses_closest_dyadic_card() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="both",
        assessed_at_ms=1000,
        support_need="task_pacing",
        task_process="pace_mismatch",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.target_actor is Actor.BOTH
    assert result.plan.strategy_id == "DYAD_RELATIONSHIP_RESET"


def test_positive_event_remains_non_intervention_under_simple_policy() -> None:
    observation = _observation(
        state="normal",
        performance="active child participation",
        actor="child",
        assessed_at_ms=1000,
        support_need="positive_reinforcement",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert decision.action.value == "no_intervention"
    assert result.plan is None
    assert result.hold_reason == "module_two_did_not_authorize"


def test_child_escalation_selects_child_support_without_guessing_parent() -> None:
    observation = _observation(
        state="dysregulation",
        performance="escalating negative interaction",
        actor="child",
        assessed_at_ms=1000,
        support_need="emotional_support",
        task_process="sustained_stall",
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
        support_need="autonomy_support",
        task_process="over_assistance",
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
        support_need="autonomy_support",
        task_process="over_assistance",
    )
    second_decision = controller.ingest(second_observation)
    second = selector.select(
        assessment=second_observation.assessment,
        decision=second_decision,
        previous_plan=first.plan,
    )

    assert second_decision.recovery_status is RecoveryStatus.NOT_RECOVERED
    assert second.plan is not None
    assert second.plan.strategy_id == "PARENT_BOUNDARY_SET"
    assert second.plan.previous_strategy_id == "PARENT_AUTONOMY_SPACE"


def test_deterioration_selects_de_escalation_card() -> None:
    controller = _controller()
    selector = _selector()
    first_observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
        assessed_at_ms=1000,
        support_need="emotional_support",
        task_process="pace_mismatch",
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
        support_need="task_pacing",
        task_process="pace_mismatch",
    )
    second_decision = controller.ingest(second_observation)
    second = selector.select(
        assessment=second_observation.assessment,
        decision=second_decision,
        previous_plan=first.plan,
    )

    assert second_decision.recovery_status is RecoveryStatus.DETERIORATED
    assert second.plan is not None
    assert second.plan.strategy_id == "CHILD_PACE_RESET"
    assert second.plan.target_actor is Actor.CHILD


def test_strategy_replay_writes_auditable_plans(tmp_path: Path) -> None:
    observations = [
        _observation(
            state="normal",
            performance="steady coordination",
            actor="both",
            assessed_at_ms=1000,
        ),
        _observation(
            state="dysregulation",
            performance="pace conflict",
            actor="parent",
            assessed_at_ms=2000,
            support_need="emotional_support",
            task_process="pace_mismatch",
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
    plans = json.loads((run_dir / "intervention_plans.json").read_text(encoding="utf-8"))
    assert valid is True
    assert result["intervention_plan_count"] == 1
    assert plans[0]["target_actor"] == "parent"
    assert plans[0]["strategy_id"] == "PARENT_TONE_AND_PACE"
