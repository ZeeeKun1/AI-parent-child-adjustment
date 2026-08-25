from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from coregulation_poc.models import (
    CoregulationState,
    InterventionAction,
    InterventionDecisionReason,
)
from coregulation_poc.paths import INTERVENTION_POLICY_PATH, resolve_project_path


class PolicyPrinciples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_hard_time_or_count_thresholds: bool
    use_single_signal_as_trigger: bool
    require_natural_turn_boundary_for_intervention: bool
    require_post_intervention_response_before_repeat: bool
    post_intervention_max_wait_count: int = Field(ge=1)
    low_confidence_action: InterventionAction
    insufficient_evidence_action: InterventionAction
    post_intervention_full_window_required: bool
    fluctuation_is_non_intervention: bool
    fluctuation_no_reinforcement: bool


class DecisionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: InterventionAction
    reason_code: InterventionDecisionReason
    research_basis: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class StateActionRule(DecisionRule):
    requires_natural_turn_boundary: bool
    history_required: bool


class PositiveMaintenanceRule(DecisionRule):
    allowed_states: list[CoregulationState] = Field(min_length=1)
    explicit_trigger_performances: list[str] = Field(min_length=1)
    recovery_transition_states: list[CoregulationState] = Field(min_length=1)
    requires_natural_turn_boundary: bool


class InterventionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    source: list[str] = Field(min_length=1)
    research_basis: dict[str, str]
    principles: PolicyPrinciples
    guard_actions: dict[InterventionDecisionReason, DecisionRule]
    positive_maintenance: PositiveMaintenanceRule
    state_actions: dict[CoregulationState, StateActionRule]

    @model_validator(mode="after")
    def validate_research_mapping(self) -> InterventionPolicy:
        expected_states = set(CoregulationState)
        if set(self.state_actions) != expected_states:
            raise ValueError("intervention policy must define all four co-regulation states")

        expected_actions = {
            CoregulationState.NORMAL: InterventionAction.NO_INTERVENTION,
            CoregulationState.FLUCTUATION: InterventionAction.NO_INTERVENTION,
            CoregulationState.DYSREGULATION: InterventionAction.INTERVENE,
            CoregulationState.HIGH_RISK: InterventionAction.PROGRESSIVE_SUPPORT,
        }
        for state, expected_action in expected_actions.items():
            if self.state_actions[state].action is not expected_action:
                raise ValueError(f"state {state.value} must map to {expected_action.value}")

        required_guards = {
            InterventionDecisionReason.INSUFFICIENT_EVIDENCE,
            InterventionDecisionReason.LOW_CONFIDENCE,
            InterventionDecisionReason.WAITING_FOR_NATURAL_TURN_BOUNDARY,
            InterventionDecisionReason.WAITING_FOR_POST_INTERVENTION_RESPONSE,
            InterventionDecisionReason.HISTORY_REQUIRED,
            InterventionDecisionReason.SUPPORT_NEED_NOT_IDENTIFIED,
            InterventionDecisionReason.SUPPORT_TARGET_UNIDENTIFIED,
        }
        if not required_guards.issubset(set(self.guard_actions)):
            raise ValueError("intervention policy must define every required controller guard")
        if any(rule.action is not InterventionAction.HOLD for rule in self.guard_actions.values()):
            raise ValueError("controller guards must hold rather than authorize intervention")

        if self.positive_maintenance.action is not InterventionAction.REINFORCE:
            raise ValueError("positive maintenance must use the reinforce action")
        if (
            self.positive_maintenance.reason_code
            is not InterventionDecisionReason.POSITIVE_MAINTENANCE_OPPORTUNITY
        ):
            raise ValueError("positive maintenance must use its dedicated reason code")
        if set(self.positive_maintenance.allowed_states) != {CoregulationState.NORMAL}:
            raise ValueError("positive maintenance is limited to the normal state")
        if set(self.positive_maintenance.recovery_transition_states) != {CoregulationState.FLUCTUATION}:
            raise ValueError("positive maintenance recovery transitions are limited to fluctuation")
        if not self.positive_maintenance.requires_natural_turn_boundary:
            raise ValueError("positive maintenance must wait for a natural turn boundary")

        for state in (CoregulationState.DYSREGULATION, CoregulationState.HIGH_RISK):
            if not self.state_actions[state].requires_natural_turn_boundary:
                raise ValueError(f"state {state.value} must wait for a natural turn boundary")
        if not self.state_actions[CoregulationState.HIGH_RISK].history_required:
            raise ValueError("high-risk progressive support requires interaction history")

        if not self.principles.post_intervention_full_window_required:
            raise ValueError("post-intervention observation must cover a full window")
        if not self.principles.fluctuation_is_non_intervention:
            raise ValueError("fluctuation must be a non-intervention state")
        if not self.principles.fluctuation_no_reinforcement:
            raise ValueError("fluctuation must not be reinforced")

        if self.principles.use_hard_time_or_count_thresholds:
            raise ValueError("hard time or count thresholds are not supported by the research")
        if self.principles.use_single_signal_as_trigger:
            raise ValueError("a single signal cannot trigger intervention")
        if not self.principles.require_natural_turn_boundary_for_intervention:
            raise ValueError("the policy must protect the dyad's natural interaction rhythm")
        if not self.principles.require_post_intervention_response_before_repeat:
            raise ValueError("the policy must observe intervention consequences before repeating")
        if self.principles.low_confidence_action is not InterventionAction.HOLD:
            raise ValueError("low-confidence assessments must hold intervention")
        if self.principles.insufficient_evidence_action is not InterventionAction.HOLD:
            raise ValueError("insufficient evidence must hold intervention")

        referenced_basis = {
            basis
            for rule in [
                *self.guard_actions.values(),
                self.positive_maintenance,
                *self.state_actions.values(),
            ]
            for basis in rule.research_basis
        }
        missing_basis = referenced_basis - set(self.research_basis)
        if missing_basis:
            missing = ", ".join(sorted(missing_basis))
            raise ValueError(f"policy rules reference unknown research basis: {missing}")
        return self


def load_intervention_policy(
    path: str | Path = INTERVENTION_POLICY_PATH,
) -> InterventionPolicy:
    absolute_path = resolve_project_path(path)
    with absolute_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return InterventionPolicy.model_validate(payload)
