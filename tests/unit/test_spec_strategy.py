"""Tests 14-21: Strategy selection multi-condition routing.

Verifies that support_need, task_process, and support_target
actually control strategy selection, not just appear in the prompt.
"""
from __future__ import annotations

from coregulation_poc.control import StateTrajectoryController, load_intervention_policy
from coregulation_poc.intervention import StrategySelector, load_strategy_library
from coregulation_poc.intervention.models import StrategySelectionStatus
from coregulation_poc.models import (
    Actor,
    ControlObservation,
    StateAssessment,
)


def _insufficient_video() -> dict[str, object]:
    return {
        "sufficiency": "insufficient",
        "items": [],
        "limitation_reason": "Relevant behavior is outside the frame.",
    }


def _assessment(
    *,
    state: str,
    performance: str | list[str],
    actor: str,
    assessed_at_ms: int,
    session_id: str = "spec-strategy",
    support_need: str | None = None,
    task_process: str | None = None,
    support_target: str | None = None,
) -> StateAssessment:
    performances = [performance] if isinstance(performance, str) else performance
    return StateAssessment(
        session_id=session_id,
        assessed_at_ms=assessed_at_ms,
        state=state,
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence="high",
        interaction_performance=performances,
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
                        "code": performances[0],
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
    performance: str | list[str],
    actor: str,
    assessed_at_ms: int,
    history_available: bool = False,
    response_observed: bool = False,
    support_need: str | None = None,
    task_process: str | None = None,
    support_target: str | None = None,
) -> ControlObservation:
    return ControlObservation(
        assessment=_assessment(
            state=state,
            performance=performance,
            actor=actor,
            assessed_at_ms=assessed_at_ms,
            support_need=support_need,
            task_process=task_process,
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


# Test 14: support_need=task_pacing selects a task_pacing strategy
def test_support_need_task_pacing_selects_pace_strategy() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
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
    assert result.plan.strategy_id == "PARENT_PACE_RESET"
    assert result.plan.target_actor is Actor.PARENT


# Test 15: support_need=mutual_understanding selects a both relationship strategy
def test_support_need_mutual_understanding_selects_dyad_strategy() -> None:
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


# Test 16: task_process=over_assistance selects parent autonomy support
def test_task_process_over_assistance_selects_autonomy_support() -> None:
    observation = _observation(
        state="high_risk",
        performance="parental over-helping or task takeover",
        actor="parent",
        assessed_at_ms=1000,
        history_available=True,
        support_need="autonomy_support",
        task_process="over_assistance",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.strategy_id == "PARENT_AUTONOMY_SPACE"
    assert result.plan.target_actor is Actor.PARENT


# Test 17: parent target does not select child strategy
def test_parent_target_does_not_select_child_strategy() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="parent",
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
    assert result.plan.target_actor is Actor.PARENT
    assert result.plan.target_actor is not Actor.CHILD


# Test 18: child target does not select parent strategy
def test_child_target_does_not_select_parent_strategy() -> None:
    observation = _observation(
        state="dysregulation",
        performance="pace conflict",
        actor="child",
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
    assert result.plan.target_actor is Actor.CHILD
    assert result.plan.target_actor is not Actor.PARENT
    assert result.plan.strategy_id == "CHILD_PACE_RESET"


# Test 19: unknown target only selects both strategy
def test_unknown_target_only_selects_both_strategy() -> None:
    observation = _observation(
        state="dysregulation",
        performance="misaligned understanding",
        actor="unknown",
        assessed_at_ms=1000,
        support_target="unknown",
        support_need="mutual_understanding",
        task_process="explanation_mismatch",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.target_actor is Actor.BOTH
    assert result.plan.strategy_id == "DYAD_RELATIONSHIP_RESET"


# Test 20: multiple interaction performances do not only use the first routing rule
def test_multiple_performances_use_all_matching_rules() -> None:
    observation = _observation(
        state="dysregulation",
        performance=["pace conflict", "sustained task stall"],
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
    assert "pace conflict" in result.plan.selected_from_interaction_performance
    assert "sustained task stall" in result.plan.selected_from_interaction_performance


# Test 21: no fixed BOTH/PARENT/CHILD priority; support_target determines actor
def test_no_fixed_actor_priority_support_target_determines_selection() -> None:
    observation = _observation(
        state="normal",
        performance="task completion",
        actor="parent",
        assessed_at_ms=1000,
        support_need="positive_reinforcement",
        task_process="completion",
    )
    decision = _controller().ingest(observation)

    result = _selector().select(
        assessment=observation.assessment,
        decision=decision,
    )

    assert result.plan is not None
    assert result.plan.target_actor is Actor.PARENT
    assert result.plan.strategy_id == "PARENT_POSITIVE_AFFIRM"
    assert result.plan.strategy_id != "DYAD_POSITIVE_AFFIRM"
