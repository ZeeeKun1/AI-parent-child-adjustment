from __future__ import annotations

from coregulation_poc.control import BoundaryStateTracker, load_boundary_rule_config
from coregulation_poc.models import CoregulationState, StateAssessment
from coregulation_poc.runtime import RealtimeLoopConfig


def _assessment(
    *,
    state: str = "fluctuation",
    task_stall: bool | None = True,
    prompt_count: int | None = 0,
    conflict: bool | None = False,
    disengaged: bool | None = False,
    balance: str = "one_stable",
    confidence: str = "high",
    task_process_override: str | None = None,
    performance_override: str | None = None,
) -> StateAssessment:
    if state == "normal":
        performance = "normal task progression"
        task_process = "smooth_progress" if task_stall is False else "brief_stall"
    elif state == "dysregulation":
        performance = "sustained task stall"
        task_process = "sustained_stall"
    elif state == "high_risk":
        performance = "persistent interaction imbalance"
        task_process = "over_assistance"
    else:
        performance = "brief task stall"
        task_process = "brief_stall"

    performance = performance_override or performance
    task_process = task_process_override or task_process

    return StateAssessment(
        session_id="boundary-test",
        assessed_at_ms=10_000,
        state=state,
        previous_state=None,
        trajectory="stable",
        evidence_sufficiency="sufficient",
        confidence=confidence,
        ambiguity_reason=(
            "Current evidence is low confidence." if confidence != "high" else None
        ),
        interaction_performance=[performance],
        task_process=task_process,
        support_need="task_pacing" if state != "normal" else "none",
        support_target="unknown",
        interruptibility="natural_pause",
        boundary_signals={
            "task_stall_observed": task_stall,
            "parental_prompt_count": prompt_count,
            "conflict_action_observed": conflict,
            "child_disengaged_observed": disengaged,
            "regulation_balance": balance,
        },
        modality_evidence={
            "audio": {
                "sufficiency": "sufficient",
                "items": [
                    {
                        "modality": "audio",
                        "actor": "unknown",
                        "start_ms": 0,
                        "end_ms": 10_000,
                        "code": "task progress",
                        "observation": "The task sequence is directly observable.",
                        "quote": "我们再看一下",
                    }
                ],
            },
            "video": {
                "sufficiency": "insufficient",
                "items": [],
                "limitation_reason": "Video evidence is not required for this fixture.",
            },
        },
        reason="Observable task and interaction evidence supports the model state.",
    )


def test_default_realtime_cadence_covers_thirty_minutes_at_ten_seconds() -> None:
    config = RealtimeLoopConfig()

    assert config.window_duration_ms == 10_000
    assert config.assessment_interval_ms == 10_000
    assert config.max_assessments_per_session == 180
    assert config.history_assessments == 4


def test_boundary_config_uses_data_derived_cutoffs() -> None:
    config, _ = load_boundary_rule_config()

    assert config.expected_window_ms == 10_000
    assert config.fluctuation_stagnation_minimum_ms == 10_000
    assert config.dysregulation_stagnation_minimum_ms == 30_000
    assert config.spontaneous_recovery_window_ms == 30_000
    assert config.uncertain_evidence_retention_ms == 30_000
    assert config.high_parental_prompt_rate_per_minute_exclusive == 4
    assert config.model_dysregulation_requires_operational_confirmation is True


def test_one_stable_stall_is_reconciled_to_fluctuation_at_ten_seconds() -> None:
    tracker = BoundaryStateTracker.from_codebook()

    result = tracker.resolve(
        _assessment(state="normal"),
        window_start_ms=0,
        window_end_ms=10_000,
    )

    assert result.model_state is CoregulationState.NORMAL
    assert result.assessment.state is CoregulationState.FLUCTUATION
    assert result.rule_applied is True
    assert result.active_stall_duration_ms == 10_000
    assert result.reason_code == (
        "fluctuation_10_to_30_seconds_coordination_disruption"
    )


def test_three_stall_windows_with_high_prompt_rate_reconcile_to_dysregulation() -> None:
    tracker = BoundaryStateTracker.from_codebook()

    first = tracker.resolve(
        _assessment(prompt_count=1),
        window_start_ms=0,
        window_end_ms=10_000,
    )
    second = tracker.resolve(
        _assessment(prompt_count=1),
        window_start_ms=10_000,
        window_end_ms=20_000,
    )
    third = tracker.resolve(
        _assessment(prompt_count=1),
        window_start_ms=20_000,
        window_end_ms=30_000,
    )

    assert first.assessment.state is CoregulationState.FLUCTUATION
    assert second.active_stall_duration_ms == 20_000
    assert third.model_state is CoregulationState.FLUCTUATION
    assert third.assessment.state is CoregulationState.DYSREGULATION
    assert third.active_stall_duration_ms == 30_000
    assert third.rolling_parental_prompt_rate_per_minute == 6
    assert third.corroborating_signals == ("high_parental_prompt_rate",)
    assert third.reason_code == (
        "dysregulation_30_seconds_disruption_with_corroboration"
    )


def test_thirty_second_stall_without_corroboration_does_not_force_dysregulation() -> None:
    tracker = BoundaryStateTracker.from_codebook()
    result = None
    for index in range(3):
        result = tracker.resolve(
            _assessment(prompt_count=0),
            window_start_ms=index * 10_000,
            window_end_ms=(index + 1) * 10_000,
        )

    assert result is not None
    assert result.assessment.state is CoregulationState.FLUCTUATION
    assert result.rule_applied is False
    assert result.corroborating_signals == ()
    assert result.reason_code == "dysregulation_duration_met_waiting_for_corroboration"


def test_progress_return_within_thirty_seconds_records_spontaneous_recovery() -> None:
    tracker = BoundaryStateTracker.from_codebook()
    tracker.resolve(
        _assessment(),
        window_start_ms=0,
        window_end_ms=10_000,
    )

    recovered = tracker.resolve(
        _assessment(
            state="normal",
            task_stall=False,
            balance="both_stable",
        ),
        window_start_ms=10_000,
        window_end_ms=20_000,
    )
    restarted = tracker.resolve(
        _assessment(),
        window_start_ms=20_000,
        window_end_ms=30_000,
    )

    assert recovered.spontaneous_recovery is True
    assert recovered.reason_code == "spontaneous_recovery_within_30_seconds"
    assert recovered.active_stall_duration_ms is None
    assert restarted.active_stall_duration_ms == 10_000


def test_high_risk_history_classification_is_not_overridden() -> None:
    tracker = BoundaryStateTracker.from_codebook()

    result = tracker.resolve(
        _assessment(
            state="high_risk",
            conflict=True,
            balance="both_crossed",
        ),
        window_start_ms=0,
        window_end_ms=10_000,
    )

    assert result.assessment.state is CoregulationState.HIGH_RISK
    assert result.rule_applied is False
    assert result.reason_code == "high_risk_history_rule_preserved"


def test_uncertain_window_pauses_candidate_without_erasing_valid_duration() -> None:
    tracker = BoundaryStateTracker.from_codebook()

    first = tracker.resolve(
        _assessment(prompt_count=1),
        window_start_ms=0,
        window_end_ms=10_000,
    )
    uncertain = tracker.resolve(
        _assessment(task_stall=None, prompt_count=None),
        window_start_ms=10_000,
        window_end_ms=20_000,
    )
    second_valid = tracker.resolve(
        _assessment(prompt_count=0),
        window_start_ms=20_000,
        window_end_ms=30_000,
    )
    confirmed = tracker.resolve(
        _assessment(prompt_count=0),
        window_start_ms=30_000,
        window_end_ms=40_000,
    )

    assert first.active_stall_duration_ms == 10_000
    assert uncertain.active_stall_duration_ms == 10_000
    assert uncertain.reason_code.endswith("candidate_retained")
    assert second_valid.active_stall_duration_ms == 20_000
    assert confirmed.active_stall_duration_ms == 30_000
    assert confirmed.assessment.state is CoregulationState.DYSREGULATION
    assert "high_parental_prompt_rate" in confirmed.corroborating_signals


def test_uncertainty_expiry_discards_stale_candidate() -> None:
    tracker = BoundaryStateTracker.from_codebook()
    tracker.resolve(
        _assessment(),
        window_start_ms=0,
        window_end_ms=10_000,
    )

    expired = None
    for index in range(1, 4):
        expired = tracker.resolve(
            _assessment(task_stall=None, prompt_count=None),
            window_start_ms=index * 10_000,
            window_end_ms=(index + 1) * 10_000,
        )

    assert expired is not None
    assert expired.active_stall_duration_ms is None
    assert expired.reason_code == (
        "coordination_disruption_candidate_expired_after_uncertainty"
    )

    restarted = tracker.resolve(
        _assessment(),
        window_start_ms=40_000,
        window_end_ms=50_000,
    )
    assert restarted.active_stall_duration_ms == 10_000


def test_persistent_pace_conflict_counts_even_when_task_moves_forward() -> None:
    tracker = BoundaryStateTracker.from_codebook()
    result = None
    for index in range(3):
        result = tracker.resolve(
            _assessment(
                state="dysregulation",
                task_stall=False,
                prompt_count=1,
                task_process_override="pace_mismatch",
                performance_override="pace conflict",
            ),
            window_start_ms=index * 10_000,
            window_end_ms=(index + 1) * 10_000,
        )

    assert result is not None
    assert result.active_stall_duration_ms == 30_000
    assert result.assessment.state is CoregulationState.DYSREGULATION
    assert result.reason_code == (
        "dysregulation_30_seconds_disruption_with_corroboration"
    )


def test_model_dysregulation_before_thirty_seconds_is_non_actionable_fluctuation() -> None:
    tracker = BoundaryStateTracker.from_codebook()

    result = tracker.resolve(
        _assessment(
            state="dysregulation",
            prompt_count=2,
            task_process_override="pace_mismatch",
            performance_override="pace conflict",
        ),
        window_start_ms=0,
        window_end_ms=10_000,
    )

    assert result.model_state is CoregulationState.DYSREGULATION
    assert result.assessment.state is CoregulationState.FLUCTUATION
    assert result.rule_applied is True
