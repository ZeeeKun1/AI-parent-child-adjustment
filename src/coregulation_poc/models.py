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
    SAME_EPISODE_OBSERVATION_PERIOD = "same_episode_observation_period"
    SAME_EPISODE_NO_ESCALATION = "same_episode_no_escalation"
    SAME_EPISODE_INTERVENTION_LIMIT = "same_episode_intervention_limit"
    HISTORY_REQUIRED = "history_required"
    SUPPORT_NEED_NOT_IDENTIFIED = "support_need_not_identified"
    SUPPORT_TARGET_UNIDENTIFIED = "support_target_unidentified"


class RecoveryStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    RECOVERED = "recovered"
    PARTIAL_RECOVERY = "partial_recovery"
    NOT_RECOVERED = "not_recovered"
    DETERIORATED = "deteriorated"
    INDETERMINATE = "indeterminate"
    TIMEOUT = "timeout"


class Interruptibility(StrEnum):
    NATURAL_PAUSE = "natural_pause"
    ACTIVE_SPEECH = "active_speech"
    TASK_ENGAGED = "task_engaged"
    UNCLEAR = "unclear"


class TaskProcess(StrEnum):
    SMOOTH_PROGRESS = "smooth_progress"
    BRIEF_STALL = "brief_stall"
    SUSTAINED_STALL = "sustained_stall"
    PACE_MISMATCH = "pace_mismatch"
    EXPLANATION_MISMATCH = "explanation_mismatch"
    OVER_ASSISTANCE = "over_assistance"
    DISENGAGED = "disengaged"
    COMPLETION = "completion"
    UNCLEAR = "unclear"


class SupportNeed(StrEnum):
    NONE = "none"
    POSITIVE_REINFORCEMENT = "positive_reinforcement"
    EMOTIONAL_SUPPORT = "emotional_support"
    NEED_EXPRESSION = "need_expression"
    MUTUAL_UNDERSTANDING = "mutual_understanding"
    TASK_PACING = "task_pacing"
    LEARNING_SUPPORT = "learning_support"
    AUTONOMY_SUPPORT = "autonomy_support"
    UNCLEAR = "unclear"


class InteractionTrajectory(StrEnum):
    STABLE = "stable"
    WORSENING = "worsening"
    RECOVERING = "recovering"
    UNCLEAR = "unclear"


class RegulationBalance(StrEnum):
    """Observable balance between the two participants in the current window."""

    BOTH_STABLE = "both_stable"
    ONE_STABLE = "one_stable"
    BOTH_CROSSED = "both_crossed"
    UNCLEAR = "unclear"


class TaskType(StrEnum):
    CHINESE = "chinese"
    MATHEMATICS = "mathematics"
    ENGLISH = "english"
    MORALITY_AND_LAW = "morality_and_law"
    SCIENCE = "science"
    INFORMATION_TECHNOLOGY = "information_technology"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"

    # Retain legacy values so existing study records and replays remain readable.
    MATH_CALCULATION = "math_calculation"
    MATH_WORD_PROBLEM = "math_word_problem"
    READING_ALOUD = "reading_aloud"
    DICTATION = "dictation"
    READING = "reading"
    OTHER = "other"


class TaskDifficulty(StrEnum):
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    UNKNOWN = "unknown"


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


class TaskContext(BaseModel):
    """Session-level task information provided before the session starts."""

    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1, max_length=200)
    task_type: TaskType
    task_difficulty: TaskDifficulty
    child_grade: str = Field(min_length=1, max_length=50)


class BoundarySignals(BaseModel):
    """Directly observable inputs for the data-derived state boundary rules."""

    model_config = ConfigDict(extra="forbid")

    task_stall_observed: bool | None = None
    parental_prompt_count: int | None = Field(default=None, ge=0)
    conflict_action_observed: bool | None = None
    child_disengaged_observed: bool | None = None
    regulation_balance: RegulationBalance = RegulationBalance.UNCLEAR


class StateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    assessed_at_ms: int = Field(ge=0)

    state: CoregulationState | None
    previous_state: CoregulationState | None
    trajectory: InteractionTrajectory

    evidence_sufficiency: EvidenceSufficiency
    confidence: ConfidenceLevel
    alternative_state: CoregulationState | None = None
    ambiguity_reason: str | None = None

    interaction_performance: list[str]
    task_process: TaskProcess | None
    support_need: SupportNeed | None
    support_target: Actor
    interruptibility: Interruptibility
    boundary_signals: BoundarySignals = Field(default_factory=BoundarySignals)

    modality_evidence: EvidenceByModality
    reason: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_assessment(self) -> StateAssessment:
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
            if self.support_target is not Actor.UNKNOWN:
                raise ValueError("support_target must be unknown when evidence is insufficient")
        if self.evidence_sufficiency is EvidenceSufficiency.SUFFICIENT and self.state is None:
            raise ValueError("sufficient assessment requires a state")
        if self.alternative_state is not None:
            if self.alternative_state is self.state:
                raise ValueError("alternative_state must differ from state")
        # Auxiliary interpretation fields are intentionally not hard validity
        # gates.  The runtime normalises missing ambiguity text, unsupported
        # actor labels and state-incompatible support labels before the
        # assessment reaches the controller.  A minor model formatting or
        # attribution error must not discard an otherwise usable state result.
        return self


class SpeechTurn(BaseModel):
    """One transcribed speech turn from the perception stage."""

    model_config = ConfigDict(extra="forbid")

    speaker: Actor
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_order(self) -> SpeechTurn:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class VisualObservation(BaseModel):
    """One visual behavior observation from the perception stage."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int = Field(ge=0)
    actor: Actor
    description: str = Field(min_length=1)


class PerceptionReport(BaseModel):
    """Stage-1 output: objective multimodal perception without state judgment."""

    model_config = ConfigDict(extra="forbid")

    speech_turns: list[SpeechTurn] = Field(default_factory=list)
    visual_observations: list[VisualObservation] = Field(default_factory=list)
    audio_limitation: str | None = None
    video_limitation: str | None = None


class AcousticSegment(BaseModel):
    """One voiced segment enriched with acoustic features."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: str
    mean_f0_hz: float | None = None
    median_f0_hz: float | None = None
    rms_energy: float = Field(ge=0.0, le=1.0)
    text: str | None = None

    @model_validator(mode="after")
    def validate_time_order(self) -> AcousticSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class SilenceGap(BaseModel):
    """A silence interval between two voiced segments."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class AcousticFeatures(BaseModel):
    """Locally computed acoustic features for the judgment stage."""

    model_config = ConfigDict(extra="forbid")

    segments: list[AcousticSegment] = Field(default_factory=list)
    silence_gaps: list[SilenceGap] = Field(default_factory=list)
    total_speech_ms: int = Field(ge=0, default=0)
    total_silence_ms: int = Field(ge=0, default=0)


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
