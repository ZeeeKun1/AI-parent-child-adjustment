from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.models import (
    ConfidenceLevel,
    CoregulationState,
    EvidenceSufficiency,
    InteractionTrajectory,
    RegulationBalance,
    StateAssessment,
    SupportNeed,
    TaskProcess,
)
from coregulation_poc.paths import STATE_CODEBOOK_PATH


class BoundaryRuleConfig(BaseModel):
    """Typed view of the formative-study operational boundary."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    expected_window_ms: int = Field(gt=0)
    fluctuation_stagnation_minimum_ms: int = Field(gt=0)
    dysregulation_stagnation_minimum_ms: int = Field(gt=0)
    spontaneous_recovery_window_ms: int = Field(gt=0)
    trajectory_window_ms: int = Field(gt=0)
    recovery_confirmation_ms: int = Field(gt=0)
    uncertain_evidence_retention_ms: int = Field(gt=0)
    parental_prompt_rate_window_ms: int = Field(gt=0)
    parental_prompt_rate_minimum_observation_ms: int = Field(gt=0)
    high_parental_prompt_rate_per_minute_exclusive: float = Field(ge=0)
    required_disruption_window_count: int = Field(ge=2)
    required_consecutive_disruption_window_count: int = Field(ge=2)
    required_corroborating_signal_count: int = Field(ge=1)
    model_dysregulation_requires_operational_confirmation: bool
    allow_immediate_dysregulation_for_marked_current_evidence: bool
    marked_current_evidence_requires_independent_corroboration: bool
    coordination_disruption_task_processes: list[TaskProcess] = Field(min_length=1)
    coordination_disruption_performances: list[str] = Field(min_length=1)
    explicit_recovery_task_processes: list[TaskProcess] = Field(min_length=1)
    corroborating_signals: list[str] = Field(min_length=1)
    guidance: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_boundary_order(self) -> BoundaryRuleConfig:
        if self.fluctuation_stagnation_minimum_ms >= self.dysregulation_stagnation_minimum_ms:
            raise ValueError("fluctuation threshold must precede dysregulation threshold")
        if self.spontaneous_recovery_window_ms != self.dysregulation_stagnation_minimum_ms:
            raise ValueError(
                "recovery and dysregulation boundaries must share the 30-second cutoff"
            )
        if self.uncertain_evidence_retention_ms != self.spontaneous_recovery_window_ms:
            raise ValueError(
                "uncertain evidence retention must use the existing 30-second recovery boundary"
            )
        if self.trajectory_window_ms < self.dysregulation_stagnation_minimum_ms:
            raise ValueError("trajectory window must include the dysregulation boundary")
        if self.recovery_confirmation_ms > self.spontaneous_recovery_window_ms:
            raise ValueError("recovery confirmation must fit the recovery window")
        if (
            self.parental_prompt_rate_minimum_observation_ms
            > self.parental_prompt_rate_window_ms
        ):
            raise ValueError("prompt-rate observation must fit its rolling window")
        if (
            self.required_consecutive_disruption_window_count
            > self.required_disruption_window_count
        ):
            raise ValueError("consecutive disruption count cannot exceed total count")
        return self


def load_boundary_rule_config(
    path: str | Path = STATE_CODEBOOK_PATH,
) -> tuple[BoundaryRuleConfig, dict[CoregulationState, tuple[str, ...]]]:
    codebook = load_state_codebook(path)
    payload = codebook.get("operational_boundary")
    if not isinstance(payload, dict):
        raise ValueError("state codebook is missing operational_boundary")
    config = BoundaryRuleConfig.model_validate(payload)
    performances = {
        CoregulationState(state): tuple(definition["interaction_performance"])
        for state, definition in codebook["states"].items()
    }
    return config, performances


@dataclass(frozen=True, slots=True)
class BoundaryResolution:
    assessment: StateAssessment
    model_state: CoregulationState | None
    rule_applied: bool
    reason_code: str
    active_stall_duration_ms: int | None
    rolling_disruption_window_count: int
    consecutive_disruption_window_count: int
    rolling_parental_prompt_rate_per_minute: float | None
    corroborating_signals: tuple[str, ...]
    spontaneous_recovery: bool

    def as_event_fields(self) -> dict[str, Any]:
        return {
            "model_state": None if self.model_state is None else self.model_state.value,
            "boundary_rule_applied": self.rule_applied,
            "boundary_reason_code": self.reason_code,
            "active_stall_duration_ms": self.active_stall_duration_ms,
            "active_coordination_disruption_duration_ms": (self.active_stall_duration_ms),
            "rolling_disruption_window_count": self.rolling_disruption_window_count,
            "consecutive_disruption_window_count": self.consecutive_disruption_window_count,
            "rolling_parental_prompt_rate_per_minute": (
                self.rolling_parental_prompt_rate_per_minute
            ),
            "corroborating_signals": list(self.corroborating_signals),
            "spontaneous_recovery": self.spontaneous_recovery,
            "boundary_signals": self.assessment.boundary_signals.model_dump(mode="json"),
        }


class BoundaryStateTracker:
    """Apply the data-derived fluctuation/dysregulation boundary across windows.

    Only valid windows with directly observed coordination disruption add to an
    episode.  Uncertain windows pause the episode rather than silently treating
    missing evidence as recovery; a candidate expires after the existing
    30-second recovery boundary if uncertainty continues.
    """

    def __init__(
        self,
        config: BoundaryRuleConfig,
        allowed_performances: dict[CoregulationState, tuple[str, ...]],
    ) -> None:
        self.config = config
        self.allowed_performances = allowed_performances
        self._disruption_windows: deque[tuple[int, int]] = deque()
        self._recovery_windows: deque[tuple[int, int]] = deque()
        self._uncertainty_started_at_ms: int | None = None
        self._corroborating_windows: deque[tuple[int, tuple[str, ...]]] = deque()
        self._prompt_windows: deque[tuple[int, int, int]] = deque()
        self._dysregulation_active = False

    @classmethod
    def from_codebook(cls) -> BoundaryStateTracker:
        config, performances = load_boundary_rule_config()
        return cls(config, performances)

    def resolve(
        self,
        assessment: StateAssessment,
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> BoundaryResolution:
        if window_end_ms <= window_start_ms:
            raise ValueError("boundary window end must be greater than its start")

        model_state = assessment.state
        signals = assessment.boundary_signals
        evidence_available = not (
            assessment.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT
            or assessment.confidence is ConfidenceLevel.LOW
            or model_state is None
        )
        self._prune_trajectory(window_end_ms)
        if not evidence_available:
            return self._resolve_uncertainty(
                assessment=assessment,
                model_state=model_state,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                reason_code="boundary_evidence_unavailable",
            )

        self._record_prompt_window(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            prompt_count=signals.parental_prompt_count,
        )
        prompt_rate = self._rolling_prompt_rate(window_end_ms)
        current_corroborating = self._corroborating_signals(assessment, prompt_rate)
        disruption_observed = self._coordination_disruption_observed(assessment)
        explicit_recovery = self._explicit_recovery_observed(assessment)

        if explicit_recovery and not disruption_observed:
            active_duration_ms = self._active_disruption_duration()
            if active_duration_ms is None:
                self._recovery_windows.clear()
                return self._resolution(
                    assessment=assessment,
                    model_state=model_state,
                    reason_code="no_active_coordination_disruption",
                    prompt_rate=prompt_rate,
                    corroborating=current_corroborating,
                )
            recovery_duration_ms = self._record_recovery_window(
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
            if recovery_duration_ms < self.config.recovery_confirmation_ms:
                recovery_assessment = assessment
                if self._dysregulation_active:
                    recovery_assessment = self._replace_state(
                        assessment,
                        CoregulationState.FLUCTUATION,
                        "dysregulation_recovery_provisional",
                    )
                return self._resolution(
                    assessment=recovery_assessment,
                    model_state=model_state,
                    reason_code=(
                        "dysregulation_recovery_provisional"
                        if self._dysregulation_active
                        else "recovery_candidate_retained_until_confirmed"
                    ),
                    stall_duration_ms=active_duration_ms,
                    prompt_rate=prompt_rate,
                    corroborating=self._ordered_episode_corroborating(window_end_ms),
                )
            spontaneous_recovery = (
                active_duration_ms <= self.config.spontaneous_recovery_window_ms
            )
            self._clear_active_disruption()
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="recovery_confirmed_after_stable_coordination",
                prompt_rate=prompt_rate,
                corroborating=current_corroborating,
                spontaneous_recovery=spontaneous_recovery,
            )

        if not disruption_observed:
            if model_state is CoregulationState.DYSREGULATION:
                assessment = self._replace_state(
                    assessment,
                    CoregulationState.FLUCTUATION,
                    "dysregulation_without_current_disruption_downgraded",
                )
            return self._resolve_uncertainty(
                assessment=assessment,
                model_state=model_state,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                reason_code="coordination_recovery_unclear",
                prompt_rate=prompt_rate,
                corroborating=current_corroborating,
            )

        disruption_duration_ms = self._record_disruption_window(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        self._recovery_windows.clear()
        self._uncertainty_started_at_ms = None
        self._record_corroborating_window(window_end_ms, current_corroborating)
        corroborating = self._ordered_episode_corroborating(window_end_ms)
        disruption_window_count = len(self._disruption_windows)
        consecutive_disruption_window_count = self._consecutive_disruption_window_count()

        if model_state is CoregulationState.HIGH_RISK:
            self._dysregulation_active = True
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="high_risk_history_rule_preserved",
                stall_duration_ms=disruption_duration_ms,
                prompt_rate=prompt_rate,
                corroborating=corroborating,
            )

        target_state = model_state
        reason_code = "disruption_below_fluctuation_boundary"
        marked_current = (
            self.config.allow_immediate_dysregulation_for_marked_current_evidence
            and self._marked_current_disruption(assessment, current_corroborating)
        )
        trajectory_confirmed = all(
            (
                disruption_duration_ms
                >= self.config.dysregulation_stagnation_minimum_ms,
                disruption_window_count >= self.config.required_disruption_window_count,
                consecutive_disruption_window_count
                >= self.config.required_consecutive_disruption_window_count,
                len(corroborating) >= self.config.required_corroborating_signal_count,
            )
        )
        if marked_current:
            target_state = CoregulationState.DYSREGULATION
            reason_code = "marked_current_disruption_supports_dysregulation"
        elif self._dysregulation_active:
            target_state = CoregulationState.DYSREGULATION
            reason_code = "active_dysregulation_continues_with_current_disruption"
        elif trajectory_confirmed:
            target_state = CoregulationState.DYSREGULATION
            reason_code = "dysregulation_rolling_trajectory_with_independent_corroboration"
        elif disruption_duration_ms >= self.config.dysregulation_stagnation_minimum_ms:
            if self.config.model_dysregulation_requires_operational_confirmation:
                target_state = CoregulationState.FLUCTUATION
            reason_code = "dysregulation_candidate_waiting_for_balanced_evidence"
        elif disruption_duration_ms >= self.config.fluctuation_stagnation_minimum_ms:
            if (
                model_state is CoregulationState.DYSREGULATION
                and not self.config.model_dysregulation_requires_operational_confirmation
            ):
                reason_code = "model_dysregulation_preserved_with_early_disruption_evidence"
            else:
                target_state = CoregulationState.FLUCTUATION
                reason_code = "fluctuation_10_to_30_seconds_coordination_disruption"
        elif (
            model_state is CoregulationState.DYSREGULATION
            and self.config.model_dysregulation_requires_operational_confirmation
        ):
            target_state = CoregulationState.FLUCTUATION
            reason_code = "dysregulation_candidate_waiting_for_duration"

        if target_state is CoregulationState.DYSREGULATION:
            self._dysregulation_active = True

        resolved = self._replace_state(assessment, target_state, reason_code)
        return self._resolution(
            assessment=resolved,
            model_state=model_state,
            reason_code=reason_code,
            stall_duration_ms=disruption_duration_ms,
            prompt_rate=prompt_rate,
            corroborating=corroborating,
        )

    def _coordination_disruption_observed(self, assessment: StateAssessment) -> bool:
        signals = assessment.boundary_signals
        if signals.task_stall_observed is True:
            return True
        if assessment.task_process in set(self.config.coordination_disruption_task_processes):
            return True
        return bool(
            set(assessment.interaction_performance).intersection(
                self.config.coordination_disruption_performances
            )
        )

    def _explicit_recovery_observed(self, assessment: StateAssessment) -> bool:
        signals = assessment.boundary_signals
        return all(
            (
                signals.task_stall_observed is False,
                assessment.task_process in set(self.config.explicit_recovery_task_processes),
                assessment.trajectory
                in {InteractionTrajectory.STABLE, InteractionTrajectory.RECOVERING},
                signals.conflict_action_observed is False,
                signals.child_disengaged_observed is False,
                signals.regulation_balance is RegulationBalance.BOTH_STABLE,
            )
        )

    def _record_disruption_window(
        self,
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> int:
        self._disruption_windows.append((window_start_ms, window_end_ms))
        self._prune_trajectory(window_end_ms)
        return self._active_disruption_duration() or 0

    def _record_recovery_window(
        self,
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> int:
        if (
            self._recovery_windows
            and window_start_ms
            > self._recovery_windows[-1][1] + self.config.expected_window_ms
        ):
            self._recovery_windows.clear()
        self._recovery_windows.append((window_start_ms, window_end_ms))
        self._uncertainty_started_at_ms = None
        return self._merged_duration(self._recovery_windows)

    def _record_corroborating_window(
        self,
        window_end_ms: int,
        signals: tuple[str, ...],
    ) -> None:
        if signals:
            self._corroborating_windows.append((window_end_ms, signals))

    def _prune_trajectory(self, window_end_ms: int) -> None:
        cutoff = window_end_ms - self.config.trajectory_window_ms
        while self._disruption_windows and self._disruption_windows[0][1] <= cutoff:
            self._disruption_windows.popleft()
        while self._corroborating_windows and self._corroborating_windows[0][0] <= cutoff:
            self._corroborating_windows.popleft()
        if not self._disruption_windows:
            self._recovery_windows.clear()
            self._uncertainty_started_at_ms = None
            self._dysregulation_active = False

    def _consecutive_disruption_window_count(self) -> int:
        """Count the most recent approximately consecutive disrupted windows."""

        if not self._disruption_windows:
            return 0
        count = 1
        windows = tuple(self._disruption_windows)
        for previous, current in zip(
            reversed(windows[:-1]), reversed(windows[1:]), strict=True
        ):
            if current[0] > previous[1] + self.config.expected_window_ms:
                break
            count += 1
        return count

    @staticmethod
    def _merged_duration(windows: deque[tuple[int, int]]) -> int:
        if not windows:
            return 0
        duration_ms = 0
        current_start, current_end = windows[0]
        for start_ms, end_ms in tuple(windows)[1:]:
            if start_ms <= current_end:
                current_end = max(current_end, end_ms)
                continue
            duration_ms += current_end - current_start
            current_start, current_end = start_ms, end_ms
        return duration_ms + current_end - current_start

    def _resolve_uncertainty(
        self,
        *,
        assessment: StateAssessment,
        model_state: CoregulationState | None,
        window_start_ms: int,
        window_end_ms: int,
        reason_code: str,
        prompt_rate: float | None = None,
        corroborating: tuple[str, ...] = (),
    ) -> BoundaryResolution:
        active_duration_ms = self._active_disruption_duration()
        if active_duration_ms is None:
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code=reason_code,
                prompt_rate=prompt_rate,
                corroborating=corroborating,
            )
        if self._uncertainty_started_at_ms is None:
            self._uncertainty_started_at_ms = window_start_ms
        uncertainty_ms = max(0, window_end_ms - self._uncertainty_started_at_ms)
        if uncertainty_ms >= self.config.uncertain_evidence_retention_ms:
            self._clear_active_disruption()
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="coordination_disruption_candidate_expired_after_uncertainty",
                prompt_rate=prompt_rate,
                corroborating=corroborating,
            )
        return self._resolution(
            assessment=assessment,
            model_state=model_state,
            reason_code=f"{reason_code}_candidate_retained",
            stall_duration_ms=active_duration_ms,
            prompt_rate=prompt_rate,
            corroborating=self._ordered_episode_corroborating(window_end_ms),
        )

    def _active_disruption_duration(self) -> int | None:
        if not self._disruption_windows:
            return None
        return self._merged_duration(self._disruption_windows)

    def _ordered_episode_corroborating(self, window_end_ms: int) -> tuple[str, ...]:
        self._prune_trajectory(window_end_ms)
        observed = {
            signal
            for _, signals in self._corroborating_windows
            for signal in signals
        }
        return tuple(
            item
            for item in self.config.corroborating_signals
            if item in observed
        )

    def _clear_active_disruption(self) -> None:
        self._disruption_windows.clear()
        self._recovery_windows.clear()
        self._corroborating_windows.clear()
        self._uncertainty_started_at_ms = None
        self._dysregulation_active = False

    def _marked_current_disruption(
        self,
        assessment: StateAssessment,
        current_corroborating: tuple[str, ...],
    ) -> bool:
        signals = assessment.boundary_signals
        evidence = assessment.modality_evidence.all_items
        actors = {item.actor.value for item in evidence}
        modalities = {item.modality.value for item in evidence}
        independently_corroborated = bool(
            {"parent", "child"}.issubset(actors)
            or "both" in actors
            or len(modalities) >= 2
        )
        explicit_disengagement = bool(
            signals.child_disengaged_observed is True
            and assessment.task_process is TaskProcess.DISENGAGED
            and independently_corroborated
        )
        marked_conflict = bool(
            signals.conflict_action_observed is True
            and (
                signals.child_disengaged_observed is True
                or (
                    signals.regulation_balance is RegulationBalance.BOTH_CROSSED
                    and independently_corroborated
                )
            )
        )
        if not self.config.marked_current_evidence_requires_independent_corroboration:
            return bool(marked_conflict or explicit_disengagement or current_corroborating)
        return marked_conflict or explicit_disengagement

    def _record_prompt_window(
        self,
        *,
        window_start_ms: int,
        window_end_ms: int,
        prompt_count: int | None,
    ) -> None:
        cutoff = window_end_ms - self.config.parental_prompt_rate_window_ms
        while self._prompt_windows and self._prompt_windows[0][1] <= cutoff:
            self._prompt_windows.popleft()
        if prompt_count is not None:
            self._prompt_windows.append((window_start_ms, window_end_ms, prompt_count))

    def _rolling_prompt_rate(self, window_end_ms: int) -> float | None:
        cutoff = window_end_ms - self.config.parental_prompt_rate_window_ms
        eligible = [item for item in self._prompt_windows if item[1] > cutoff]
        observed_ms = sum(end - start for start, end, _ in eligible)
        if observed_ms < self.config.parental_prompt_rate_minimum_observation_ms:
            return None
        prompt_count = sum(count for _, _, count in eligible)
        return round(prompt_count * 60_000 / observed_ms, 3)

    def _corroborating_signals(
        self,
        assessment: StateAssessment,
        prompt_rate: float | None,
    ) -> tuple[str, ...]:
        signals = assessment.boundary_signals
        observed: list[str] = []
        if (
            prompt_rate is not None
            and prompt_rate > self.config.high_parental_prompt_rate_per_minute_exclusive
        ):
            observed.append("high_parental_prompt_rate")
        if signals.conflict_action_observed is True:
            observed.append("conflict_action")
        if signals.child_disengaged_observed is True:
            observed.append("child_disengagement")
        if signals.regulation_balance is RegulationBalance.BOTH_CROSSED:
            observed.append("both_crossed")
        return tuple(item for item in observed if item in self.config.corroborating_signals)

    def _replace_state(
        self,
        assessment: StateAssessment,
        target_state: CoregulationState,
        reason_code: str,
    ) -> StateAssessment:
        if assessment.state is target_state:
            return assessment

        allowed = self.allowed_performances[target_state]
        performances = [item for item in assessment.interaction_performance if item in allowed]
        if not performances:
            performances = [
                "brief task stall"
                if target_state is CoregulationState.FLUCTUATION
                else "sustained task stall"
            ]

        support_need = assessment.support_need
        if support_need is SupportNeed.POSITIVE_REINFORCEMENT:
            support_need = SupportNeed.NONE

        payload = assessment.model_dump(mode="python")
        payload.update(
            {
                "state": target_state,
                "confidence": ConfidenceLevel.MEDIUM,
                "alternative_state": assessment.state,
                "ambiguity_reason": (
                    "The model state was reconciled with the data-derived temporal "
                    f"boundary ({reason_code})."
                ),
                "interaction_performance": performances,
                "support_need": support_need,
                "reason": (f"{assessment.reason} Data-derived boundary rule: {reason_code}."),
            }
        )
        return StateAssessment.model_validate(payload)

    def _resolution(
        self,
        *,
        assessment: StateAssessment,
        model_state: CoregulationState | None,
        reason_code: str,
        stall_duration_ms: int | None = None,
        prompt_rate: float | None = None,
        corroborating: tuple[str, ...] = (),
        spontaneous_recovery: bool = False,
    ) -> BoundaryResolution:
        return BoundaryResolution(
            assessment=assessment,
            model_state=model_state,
            rule_applied=assessment.state is not model_state,
            reason_code=reason_code,
            active_stall_duration_ms=stall_duration_ms,
            rolling_disruption_window_count=len(self._disruption_windows),
            consecutive_disruption_window_count=self._consecutive_disruption_window_count(),
            rolling_parental_prompt_rate_per_minute=prompt_rate,
            corroborating_signals=corroborating,
            spontaneous_recovery=spontaneous_recovery,
        )
