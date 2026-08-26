from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coregulation_poc.models import (
    Actor,
    ConfidenceLevel,
    CoregulationState,
    EvidenceReference,
    InterventionAction,
    RecoveryStatus,
)


class RepairTarget(StrEnum):
    PARENT_REGULATION = "parent_regulation"
    CHILD_SUPPORT = "child_support"
    RELATIONSHIP_REPAIR = "relationship_repair"
    TASK_SUPPORT = "task_support"
    TASK_PACING = "task_pacing"
    NEEDS_CLARIFICATION = "needs_clarification"
    AUTONOMY_BOUNDARY = "autonomy_boundary"
    TASK_PAUSE = "task_pause"
    POSITIVE_REINFORCEMENT = "positive_reinforcement"
    POSTURE_ADJUSTMENT = "posture_adjustment"
    ENVIRONMENT_ADJUSTMENT = "environment_adjustment"
    BOUNDARY_SETTING = "boundary_setting"


class MessageSource(StrEnum):
    APPROVED_TEMPLATE = "approved_template"
    CONSTRAINED_LLM = "constrained_llm"
    APPROVED_TEMPLATE_FALLBACK = "approved_template_fallback"


class StrategySelectionStatus(StrEnum):
    READY = "ready"
    HELD = "held"


class StrategySelectionSource(StrEnum):
    EXACT_RULE = "exact_rule"
    BOUNDED_LLM = "bounded_llm"


class StrategyHoldReason(StrEnum):
    MODULE_TWO_DID_NOT_AUTHORIZE = "module_two_did_not_authorize"
    STATE_NOT_SUPPORTED = "state_not_supported"
    ASSESSMENT_DECISION_MISMATCH = "assessment_decision_mismatch"
    NO_ROUTED_STRATEGY = "no_routed_strategy"
    TARGET_ACTOR_EVIDENCE_INSUFFICIENT = "target_actor_evidence_insufficient"
    SEMANTIC_SELECTOR_UNAVAILABLE = "semantic_selector_unavailable"
    SEMANTIC_SELECTOR_NO_MATCH = "semantic_selector_no_match"
    SEMANTIC_SELECTOR_REJECTED = "semantic_selector_rejected"


class StrategyPrinciples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_two_authorization_required: bool
    normal_requires_positive_maintenance_authorization: bool
    target_actor_must_be_explicit: bool
    one_card_one_primary_action: bool
    llm_generates_observation_only: bool
    llm_strategy_selection_is_bounded: bool
    llm_cannot_create_strategy_cards: bool
    approved_template_fallback_required: bool
    do_not_repeat_before_observing_response: bool
    maximum_message_characters: int = Field(gt=0)
    maximum_message_sentences: int = Field(gt=0)


class StrategyRoutingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: CoregulationState
    interaction_performance: str = Field(min_length=1)
    strategy_ids: list[str] = Field(min_length=1)


class StrategyCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    name: str = Field(min_length=1)
    states: list[CoregulationState] = Field(min_length=1)
    target_actor: Actor
    repair_target: RepairTarget
    strategy_family: str = Field(min_length=1)
    support_needs: list[str] = Field(default_factory=list)
    task_processes: list[str] = Field(default_factory=list)
    approved_action_clause: str = Field(min_length=1)
    inactive: bool = False
    priority: int = Field(default=0, ge=0)
    use_when: list[str] = Field(min_length=1)
    avoid_when: list[str] = Field(min_length=1)
    action: str = Field(min_length=1)
    approved_template: str = Field(min_length=1)
    expected_recovery: list[str] = Field(min_length=1)
    research_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_card_scope(self) -> StrategyCard:
        if self.target_actor is Actor.UNKNOWN:
            raise ValueError("strategy cards must explicitly target parent, child or both")
        return self


class StrategyLibraryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    source: list[str] = Field(min_length=1)
    principles: StrategyPrinciples
    banned_phrases: list[str] = Field(min_length=1)
    outcome_mapping: dict[RecoveryStatus, list[str]]
    routing_rules: list[StrategyRoutingRule] = Field(min_length=1)
    cards: list[StrategyCard] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_library(self) -> StrategyLibraryConfig:
        required_principles = {
            "module_two_authorization_required": self.principles.module_two_authorization_required,
            "normal_requires_positive_maintenance_authorization": (
                self.principles.normal_requires_positive_maintenance_authorization
            ),
            "target_actor_must_be_explicit": self.principles.target_actor_must_be_explicit,
            "one_card_one_primary_action": self.principles.one_card_one_primary_action,
            "llm_generates_observation_only": (
                self.principles.llm_generates_observation_only
            ),
            "llm_strategy_selection_is_bounded": (
                self.principles.llm_strategy_selection_is_bounded
            ),
            "llm_cannot_create_strategy_cards": (
                self.principles.llm_cannot_create_strategy_cards
            ),
            "approved_template_fallback_required": (
                self.principles.approved_template_fallback_required
            ),
            "do_not_repeat_before_observing_response": (
                self.principles.do_not_repeat_before_observing_response
            ),
        }
        disabled = [name for name, enabled in required_principles.items() if not enabled]
        if disabled:
            raise ValueError(f"required strategy principles cannot be disabled: {disabled}")

        identifiers = [card.strategy_id for card in self.cards]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("strategy card identifiers must be unique")
        cards = {card.strategy_id: card for card in self.cards}
        for card in self.cards:
            self._validate_template(card)

        for rule in self.routing_rules:
            for strategy_id in rule.strategy_ids:
                if strategy_id not in cards:
                    raise ValueError(f"routing rule references unknown strategy: {strategy_id}")
                if cards[strategy_id].inactive:
                    raise ValueError(
                        f"routing rule references inactive strategy: {strategy_id}"
                    )
                if rule.state not in cards[strategy_id].states:
                    raise ValueError(
                        f"strategy {strategy_id} does not support state {rule.state.value}"
                    )

        expected_routes = {
            CoregulationState.NORMAL: {
                "normal task progression",
                "active child participation",
                "supportive parental guidance",
                "task completion",
                "positive dyadic exchange",
            },
            CoregulationState.DYSREGULATION: {
                "sustained task stall",
                "pace conflict",
                "misaligned understanding",
                "escalating negative interaction",
            },
            CoregulationState.HIGH_RISK: {
                "persistent interaction imbalance",
                "parental over-helping or task takeover",
                "child passive dependence",
                "sustained strong resistance or withdrawal",
            },
        }
        actual_routes = {
            state: {
                rule.interaction_performance
                for rule in self.routing_rules
                if rule.state is state
            }
            for state in expected_routes
        }
        if actual_routes != expected_routes:
            raise ValueError("strategy routing must cover every intervention-state performance")

        required_outcomes = {
            RecoveryStatus.RECOVERED,
            RecoveryStatus.PARTIAL_RECOVERY,
            RecoveryStatus.NOT_RECOVERED,
            RecoveryStatus.DETERIORATED,
            RecoveryStatus.INDETERMINATE,
        }
        if set(self.outcome_mapping) != required_outcomes:
            raise ValueError("outcome_mapping must cover every observable recovery result")
        return self

    def _validate_template(self, card: StrategyCard) -> None:
        template = card.approved_template.strip()
        if len(template) > self.principles.maximum_message_characters:
            raise ValueError(f"strategy {card.strategy_id} template is too long")
        sentence_count = len(
            [part for part in re.split(r"(?<=[。！？!?])", template) if part.strip()]
        )
        if sentence_count > self.principles.maximum_message_sentences:
            raise ValueError(f"strategy {card.strategy_id} template has too many sentences")
        if any(phrase in template for phrase in self.banned_phrases):
            raise ValueError(f"strategy {card.strategy_id} template contains a banned phrase")


class InterventionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    planned_at_ms: int = Field(ge=0)
    state: CoregulationState
    decision_action: InterventionAction
    strategy_library_version: int = Field(ge=1)
    strategy_id: str = Field(min_length=1)
    strategy_name: str = Field(min_length=1)
    target_actor: Actor
    repair_target: RepairTarget
    message: str = Field(min_length=1)
    message_source: MessageSource
    strategy_selection_source: StrategySelectionSource = StrategySelectionSource.EXACT_RULE
    semantic_selection_confidence: ConfidenceLevel | None = None
    semantic_relaxed_dimensions: list[str] = Field(default_factory=list)
    selection_reason: str = Field(min_length=1)
    selected_from_interaction_performance: str = Field(min_length=1)
    evidence_references: list[EvidenceReference] = Field(min_length=1)
    expected_recovery: list[str] = Field(min_length=1)
    outcome_interpretation: dict[RecoveryStatus, list[str]]
    validation_checks: dict[str, bool]
    previous_strategy_id: str | None = None
    progressive_support: bool
    research_codes: list[str] = Field(min_length=1)
    research_sources: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> InterventionPlan:
        if self.target_actor is Actor.UNKNOWN:
            raise ValueError("an intervention plan cannot target an unknown actor")
        if self.decision_action not in {
            InterventionAction.REINFORCE,
            InterventionAction.INTERVENE,
            InterventionAction.PROGRESSIVE_SUPPORT,
        }:
            raise ValueError("an intervention plan requires module-two intervention permission")
        if self.progressive_support != (
            self.decision_action is InterventionAction.PROGRESSIVE_SUPPORT
        ):
            raise ValueError("progressive_support must match the module-two action")
        if not self.validation_checks or not all(self.validation_checks.values()):
            raise ValueError("all message validation checks must pass")
        if self.strategy_selection_source is StrategySelectionSource.BOUNDED_LLM:
            if self.semantic_selection_confidence not in {
                ConfidenceLevel.HIGH,
                ConfidenceLevel.MEDIUM,
            }:
                raise ValueError("bounded LLM selection requires accepted confidence")
        return self


class StrategySelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StrategySelectionStatus
    hold_reason: StrategyHoldReason | None = None
    plan: InterventionPlan | None = None

    @model_validator(mode="after")
    def match_status_and_payload(self) -> StrategySelectionResult:
        if self.status is StrategySelectionStatus.READY:
            if self.plan is None or self.hold_reason is not None:
                raise ValueError("ready strategy selection requires a plan and no hold reason")
        if self.status is StrategySelectionStatus.HELD:
            if self.plan is not None or self.hold_reason is None:
                raise ValueError("held strategy selection requires a hold reason and no plan")
        return self
