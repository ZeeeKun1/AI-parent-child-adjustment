from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Actor(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    BOTH = "both"
    UNKNOWN = "unknown"


class CoregulationState(StrEnum):
    NORMAL = "normal"
    FLUCTUATION = "fluctuation"
    DYSREGULATION = "dysregulation"
    HIGH_RISK = "high_risk"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class EvidenceModality(StrEnum):
    AUDIO = "audio"
    VIDEO = "video"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InterventionAction(StrEnum):
    NO_INTERVENTION = "no_intervention"
    OBSERVE = "observe"
    REINFORCE = "reinforce"
    INTERVENE = "intervene"
    PROGRESSIVE_SUPPORT = "progressive_support"
    HOLD = "hold"


class InterventionDecisionReason(StrEnum):
    NORMAL_COORDINATION = "normal_coordination"
    SELF_RECOVERY_POSSIBLE = "self_recovery_possible"
    POSITIVE_MAINTENANCE_OPPORTUNITY = "positive_maintenance_opportunity"
    DYAD_CANNOT_SELF_RECOVER = "dyad_cannot_self_recover"
    PERSISTENT_HIGH_RISK_PATTERN = "persistent_high_risk_pattern"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_CONFIDENCE = "low_confidence"
    WAITING_FOR_NATURAL_TURN_BOUNDARY = "waiting_for_natural_turn_boundary"
    WAITING_FOR_POST_INTERVENTION_RESPONSE = "waiting_for_post_intervention_response"
    HISTORY_REQUIRED = "history_required"


class RecoveryStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    RECOVERED = "recovered"
    PARTIAL_RECOVERY = "partial_recovery"
    NOT_RECOVERED = "not_recovered"
    DETERIORATED = "deteriorated"
    INDETERMINATE = "indeterminate"
    TIMEOUT = "timeout"


class TimedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_time_order(self) -> TimedEvent:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class TranscriptEvent(TimedEvent):
    actor: Actor
    speaker_id: str
    text: str = Field(min_length=1)
    is_final: bool = True
    volume_db: float | None = None


class VisualEvidence(TimedEvent):
    actor: Actor
    behavior_codes: list[str]
    description: str = Field(min_length=1)
    sufficiency: EvidenceSufficiency


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: EvidenceModality
    actor: Actor = Actor.UNKNOWN
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    code: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    quote: str | None = None
    frame_timestamp_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_modality_specific_source(self) -> EvidenceReference:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if self.modality is EvidenceModality.AUDIO:
            if self.quote is None or not self.quote.strip():
                raise ValueError("audio evidence requires a verbatim quote")
            if self.frame_timestamp_ms is not None:
                raise ValueError("audio evidence cannot contain frame_timestamp_ms")
        if self.modality is EvidenceModality.VIDEO:
            if self.frame_timestamp_ms is None:
                raise ValueError("video evidence requires frame_timestamp_ms")
            if not self.start_ms <= self.frame_timestamp_ms <= self.end_ms:
                raise ValueError("frame_timestamp_ms must fall inside the evidence interval")
            if self.quote is not None:
                raise ValueError("video evidence must use observation, not quote")
        return self


class ModalityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficiency: EvidenceSufficiency
    items: list[EvidenceReference]
    limitation_reason: str | None = None

    @model_validator(mode="after")
    def match_items_to_sufficiency(self) -> ModalityEvidence:
        if self.sufficiency is EvidenceSufficiency.SUFFICIENT and not self.items:
            raise ValueError("sufficient modality evidence requires at least one item")
        if self.sufficiency is EvidenceSufficiency.INSUFFICIENT:
            if self.items:
                raise ValueError("insufficient modality evidence must not contain evidence items")
            if self.limitation_reason is None or not self.limitation_reason.strip():
                raise ValueError("insufficient modality evidence requires limitation_reason")
        return self


class EvidenceByModality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: ModalityEvidence
    video: ModalityEvidence

    @model_validator(mode="after")
    def keep_items_in_their_declared_modality(self) -> EvidenceByModality:
        if any(item.modality is not EvidenceModality.AUDIO for item in self.audio.items):
            raise ValueError("audio evidence bundle contains a non-audio item")
        if any(item.modality is not EvidenceModality.VIDEO for item in self.video.items):
            raise ValueError("video evidence bundle contains a non-video item")
        return self

    @property
    def all_items(self) -> list[EvidenceReference]:
        return [*self.audio.items, *self.video.items]


class StateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    assessed_at_ms: int = Field(ge=0)
    state: CoregulationState | None
    evidence_sufficiency: EvidenceSufficiency
    confidence: ConfidenceLevel
    alternative_state: CoregulationState | None = None
    ambiguity_reason: str | None = None
    interaction_performance: list[str]
    modality_evidence: EvidenceByModality
    previous_state: CoregulationState | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_state_only_with_evidence(self) -> StateAssessment:
        if self.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT and self.state is not None:
            raise ValueError("state must be null when evidence is insufficient")
        modality_sufficient = {
            self.modality_evidence.audio.sufficiency,
            self.modality_evidence.video.sufficiency,
        }
        if (
            self.evidence_sufficiency is EvidenceSufficiency.SUFFICIENT
            and EvidenceSufficiency.SUFFICIENT not in modality_sufficient
        ):
            raise ValueError("sufficient assessment requires at least one sufficient modality")
        if self.evidence_sufficiency is EvidenceSufficiency.INSUFFICIENT:
            if EvidenceSufficiency.SUFFICIENT in modality_sufficient:
                raise ValueError("insufficient assessment cannot contain a sufficient modality")
            if self.confidence is not ConfidenceLevel.LOW:
                raise ValueError("insufficient assessment must use low confidence")
            if self.alternative_state is not None:
                raise ValueError("insufficient assessment cannot select an alternative state")
        if self.evidence_sufficiency is EvidenceSufficiency.SUFFICIENT and self.state is None:
            raise ValueError("sufficient assessment requires a state")
        if self.alternative_state is not None:
            if self.alternative_state is self.state:
                raise ValueError("alternative_state must differ from state")
            if self.ambiguity_reason is None or not self.ambiguity_reason.strip():
                raise ValueError("alternative_state requires ambiguity_reason")
        if self.confidence is not ConfidenceLevel.HIGH:
            if self.ambiguity_reason is None or not self.ambiguity_reason.strip():
                raise ValueError("low or medium confidence requires ambiguity_reason")
        return self


class ControlObservation(BaseModel):
    """One module-one assessment plus event-boundary facts supplied by the runtime."""

    model_config = ConfigDict(extra="forbid")

    assessment: StateAssessment
    natural_turn_boundary: bool
    post_intervention_response_observed: bool = False
    interaction_history_available: bool = False


class TrajectoryReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    observations: list[ControlObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def keep_one_session(self) -> TrajectoryReplayRequest:
        mismatched = [
            item.assessment.session_id
            for item in self.observations
            if item.assessment.session_id != self.session_id
        ]
        if mismatched:
            raise ValueError("all observation assessments must match request session_id")
        return self


class StateTrajectoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    assessed_at_ms: int = Field(ge=0)
    state: CoregulationState | None
    confidence: ConfidenceLevel
    evidence_sufficiency: EvidenceSufficiency
    interaction_performance: list[str]


class InterventionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    decided_at_ms: int = Field(ge=0)
    previous_state: CoregulationState | None
    current_state: CoregulationState | None
    action: InterventionAction
    reason_code: InterventionDecisionReason
    reason: str = Field(min_length=1)
    natural_turn_boundary: bool
    intervention_permitted: bool
    strategy_selection_required: bool
    recovery_status: RecoveryStatus
    evidence_actors: list[Actor]
    interaction_performance: list[str]
    research_basis: list[str] = Field(min_length=1)


class StateTrajectorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    policy_version: int = Field(ge=1)
    points: list[StateTrajectoryPoint]
    decisions: list[InterventionDecision]
    awaiting_post_intervention_response: bool
