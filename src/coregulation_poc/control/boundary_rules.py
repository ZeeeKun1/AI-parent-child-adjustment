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
    RegulationBalance,
    StateAssessment,
    SupportNeed,
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
    parental_prompt_rate_window_ms: int = Field(gt=0)
    high_parental_prompt_rate_per_minute_exclusive: float = Field(ge=0)
    required_corroborating_signal_count: int = Field(ge=1)
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
    rolling_parental_prompt_rate_per_minute: float | None
    corroborating_signals: tuple[str, ...]
    spontaneous_recovery: bool

    def as_event_fields(self) -> dict[str, Any]:
        return {
            "model_state": None if self.model_state is None else self.model_state.value,
            "boundary_rule_applied": self.rule_applied,
            "boundary_reason_code": self.reason_code,
            "active_stall_duration_ms": self.active_stall_duration_ms,
            "rolling_parental_prompt_rate_per_minute": (
                self.rolling_parental_prompt_rate_per_minute
            ),
            "corroborating_signals": list(self.corroborating_signals),
            "spontaneous_recovery": self.spontaneous_recovery,
            "boundary_signals": self.assessment.boundary_signals.model_dump(mode="json"),
        }


class BoundaryStateTracker:
    """Apply the data-derived fluctuation/dysregulation boundary across windows."""

    def __init__(
        self,
        config: BoundaryRuleConfig,
        allowed_performances: dict[CoregulationState, tuple[str, ...]],
    ) -> None:
        self.config = config
        self.allowed_performances = allowed_performances
        self._active_stall_started_at_ms: int | None = None
        self._prompt_windows: deque[tuple[int, int, int]] = deque()

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
        if (
            assessment.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT
            or assessment.confidence is ConfidenceLevel.LOW
            or model_state is None
        ):
            self._active_stall_started_at_ms = None
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="boundary_evidence_unavailable",
            )

        signals = assessment.boundary_signals
        self._record_prompt_window(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            prompt_count=signals.parental_prompt_count,
        )
        prompt_rate = self._rolling_prompt_rate(window_end_ms)
        corroborating = self._corroborating_signals(assessment, prompt_rate)

        if signals.task_stall_observed is False:
            recovery_elapsed_ms = (
                None
                if self._active_stall_started_at_ms is None
                else max(0, window_start_ms - self._active_stall_started_at_ms)
            )
            spontaneous_recovery = (
                recovery_elapsed_ms is not None
                and recovery_elapsed_ms <= self.config.spontaneous_recovery_window_ms
            )
            self._active_stall_started_at_ms = None
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code=(
                    "spontaneous_recovery_within_30_seconds"
                    if spontaneous_recovery
                    else "no_active_task_stall"
                ),
                prompt_rate=prompt_rate,
                corroborating=corroborating,
                spontaneous_recovery=spontaneous_recovery,
            )

        if signals.task_stall_observed is not True:
            self._active_stall_started_at_ms = None
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="task_stall_unclear",
                prompt_rate=prompt_rate,
                corroborating=corroborating,
            )

        if self._active_stall_started_at_ms is None:
            self._active_stall_started_at_ms = window_start_ms
        stall_duration_ms = max(0, window_end_ms - self._active_stall_started_at_ms)

        if model_state is CoregulationState.HIGH_RISK:
            return self._resolution(
                assessment=assessment,
                model_state=model_state,
                reason_code="high_risk_history_rule_preserved",
                stall_duration_ms=stall_duration_ms,
                prompt_rate=prompt_rate,
                corroborating=corroborating,
            )

        target_state = model_state
        reason_code = "stall_below_fluctuation_boundary"
        if stall_duration_ms >= self.config.dysregulation_stagnation_minimum_ms:
            if len(corroborating) >= self.config.required_corroborating_signal_count:
                target_state = CoregulationState.DYSREGULATION
                reason_code = "dysregulation_30_seconds_with_corroboration"
            else:
                reason_code = "dysregulation_duration_met_waiting_for_corroboration"
        elif (
            stall_duration_ms >= self.config.fluctuation_stagnation_minimum_ms
            and signals.regulation_balance is RegulationBalance.ONE_STABLE
        ):
            target_state = CoregulationState.FLUCTUATION
            reason_code = "fluctuation_10_to_30_seconds_one_stable"
        elif stall_duration_ms >= self.config.fluctuation_stagnation_minimum_ms:
            reason_code = "fluctuation_duration_met_balance_unresolved"

        resolved = self._replace_state(assessment, target_state, reason_code)
        return self._resolution(
            assessment=resolved,
            model_state=model_state,
            reason_code=reason_code,
            stall_duration_ms=stall_duration_ms,
            prompt_rate=prompt_rate,
            corroborating=corroborating,
        )

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
        if observed_ms <= 0:
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
        performances = [
            item for item in assessment.interaction_performance if item in allowed
        ]
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
                "reason": (
                    f"{assessment.reason} Data-derived boundary rule: {reason_code}."
                ),
            }
        )
        return StateAssessment.model_validate(payload)

    @staticmethod
    def _resolution(
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
            rolling_parental_prompt_rate_per_minute=prompt_rate,
            corroborating_signals=corroborating,
            spontaneous_recovery=spontaneous_recovery,
        )
