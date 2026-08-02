from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coregulation_poc.intervention.models import MessageSource, RepairTarget
from coregulation_poc.models import Actor


class PromptProminence(StrEnum):
    HIGH = "high"


class VisualPlacement(StrEnum):
    PROMINENT_OVERLAY = "prominent_overlay"


class VoiceStyle(StrEnum):
    CALM_NEUTRAL_SUPPORTIVE = "calm_neutral_supportive"


class DeliveryPreparationStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    HELD = "held"


class DeliveryHoldReason(StrEnum):
    INTERVENTIONS_PAUSED = "interventions_paused"


class OutputModality(StrEnum):
    VISUAL_TEXT = "visual_text"
    SPOKEN_VOICE = "spoken_voice"


class OutputExecutionStatus(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class OverallDeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    PARTIAL = "partial"
    FAILED = "failed"


class DeliveryPrinciples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_authorized_intervention_plan: bool
    visual_and_voice_are_primary_modalities: bool
    identical_core_message_across_modalities: bool
    preserve_target_actor_from_strategy: bool
    do_not_repeat_before_observed_response: bool
    record_channels_separately: bool
    never_infer_seen_or_heard_from_successful_output: bool


class VisualDeliveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headings_by_target: dict[Actor, str]
    prominence: PromptProminence
    placement: VisualPlacement
    blocks_primary_task: bool
    dismissible: bool


class VoiceDeliveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)
    language_type: str = Field(min_length=1)
    response_format: str = Field(pattern="^pcm$")
    sample_rate_hz: int = Field(gt=0)
    mode: str = Field(pattern="^commit$")
    instructions: str = Field(min_length=1)
    optimize_instructions: bool
    autoplay: bool
    style: VoiceStyle


class DeliveryFallbackPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hold_all_output_when_interventions_paused: bool
    preserve_visual_when_voice_unavailable: bool
    record_voice_failure_without_claiming_message_was_heard: bool


class DeliveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    source: list[str] = Field(min_length=1)
    principles: DeliveryPrinciples
    visual: VisualDeliveryPolicy
    voice: VoiceDeliveryPolicy
    fallback: DeliveryFallbackPolicy

    @model_validator(mode="after")
    def enforce_research_decisions(self) -> DeliveryPolicy:
        required_principles = self.principles.model_dump()
        disabled = [name for name, enabled in required_principles.items() if not enabled]
        if disabled:
            raise ValueError(f"required delivery principles cannot be disabled: {disabled}")
        if set(self.visual.headings_by_target) != {
            Actor.PARENT,
            Actor.CHILD,
            Actor.BOTH,
        }:
            raise ValueError("visual headings must cover parent, child and both")
        if self.visual.prominence is not PromptProminence.HIGH:
            raise ValueError("the visual intervention must remain prominent")
        if self.visual.blocks_primary_task:
            raise ValueError("the visual intervention must not block the primary task")
        if not self.voice.autoplay:
            raise ValueError("voice is a primary intervention modality and must autoplay")
        if self.voice.optimize_instructions:
            raise ValueError("experiment voice instructions must not be automatically rewritten")
        if not self.fallback.hold_all_output_when_interventions_paused:
            raise ValueError("pausing interventions must hold both output modalities")
        if not self.fallback.preserve_visual_when_voice_unavailable:
            raise ValueError("visual text must remain available when voice fails")
        if not self.fallback.record_voice_failure_without_claiming_message_was_heard:
            raise ValueError("voice failure must remain explicit in the audit record")
        return self


class DeliveryRuntimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepared_at_ms: int = Field(ge=0)
    interventions_paused: bool = False
    voice_enabled: bool = True
    voice_available: bool = True


class VisualPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: OutputModality = OutputModality.VISUAL_TEXT
    target_actor: Actor
    heading: str = Field(min_length=1)
    message: str = Field(min_length=1)
    prominence: PromptProminence
    placement: VisualPlacement
    blocks_primary_task: bool
    dismissible: bool


class VoicePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: OutputModality = OutputModality.SPOKEN_VOICE
    target_actor: Actor
    message: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)
    language_type: str = Field(min_length=1)
    response_format: str = Field(pattern="^pcm$")
    sample_rate_hz: int = Field(gt=0)
    mode: str = Field(pattern="^commit$")
    instructions: str = Field(min_length=1)
    optimize_instructions: bool
    style: VoiceStyle
    autoplay: bool
    enabled: bool


class DeliveryPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    planned_at_ms: int = Field(ge=0)
    prepared_at_ms: int = Field(ge=0)
    delivery_policy_version: int = Field(ge=1)
    strategy_id: str = Field(min_length=1)
    target_actor: Actor
    repair_target: RepairTarget
    message_source: MessageSource
    visual_prompt: VisualPrompt
    voice_prompt: VoicePrompt
    status: DeliveryPreparationStatus
    core_content_identical: bool
    fallback_reason: str | None = None
    research_basis: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dual_channel_contract(self) -> DeliveryPackage:
        if self.target_actor is Actor.UNKNOWN:
            raise ValueError("delivery cannot target an unknown actor")
        if self.visual_prompt.target_actor is not self.target_actor:
            raise ValueError("visual target must match the strategy target")
        if self.voice_prompt.target_actor is not self.target_actor:
            raise ValueError("voice target must match the strategy target")
        if self.visual_prompt.message != self.voice_prompt.message:
            raise ValueError("visual and voice outputs must use the same core message")
        if not self.core_content_identical:
            raise ValueError("core_content_identical must remain true")
        if self.status is DeliveryPreparationStatus.READY:
            if not self.voice_prompt.enabled or self.fallback_reason is not None:
                raise ValueError("ready delivery requires both output modalities")
        if self.status is DeliveryPreparationStatus.DEGRADED:
            if self.voice_prompt.enabled or not self.fallback_reason:
                raise ValueError("degraded delivery requires an explicit voice fallback reason")
        if self.status is DeliveryPreparationStatus.HELD:
            raise ValueError("held preparation cannot contain a delivery package")
        return self


class DeliveryPreparationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DeliveryPreparationStatus
    hold_reason: DeliveryHoldReason | None = None
    package: DeliveryPackage | None = None

    @model_validator(mode="after")
    def match_status_and_payload(self) -> DeliveryPreparationResult:
        if self.status is DeliveryPreparationStatus.HELD:
            if self.hold_reason is None or self.package is not None:
                raise ValueError("held delivery requires a reason and no package")
        else:
            if self.hold_reason is not None or self.package is None:
                raise ValueError("prepared delivery requires a package and no hold reason")
            if self.package.status is not self.status:
                raise ValueError("result and package preparation statuses must match")
        return self


class ChannelExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: OutputModality
    status: OutputExecutionStatus
    started_at_ms: int | None = Field(default=None, ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)
    provider: str | None = None
    output_identifier: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_execution_evidence(self) -> ChannelExecution:
        if self.status is OutputExecutionStatus.DELIVERED:
            if self.started_at_ms is None or self.provider is None:
                raise ValueError("delivered output requires a start time and provider")
            if self.error is not None:
                raise ValueError("delivered output cannot contain an error")
        if self.status is OutputExecutionStatus.FAILED:
            if self.error is None or not self.error.strip():
                raise ValueError("failed output requires an error")
        if self.status is OutputExecutionStatus.NOT_ATTEMPTED:
            if any(
                value is not None
                for value in (
                    self.started_at_ms,
                    self.completed_at_ms,
                    self.provider,
                    self.output_identifier,
                    self.error,
                )
            ):
                raise ValueError("not-attempted output cannot claim execution evidence")
        if (
            self.started_at_ms is not None
            and self.completed_at_ms is not None
            and self.completed_at_ms < self.started_at_ms
        ):
            raise ValueError("completed_at_ms cannot precede started_at_ms")
        return self


class DeliveryExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    recorded_at_ms: int = Field(ge=0)
    visual: ChannelExecution
    voice: ChannelExecution
    overall_status: OverallDeliveryStatus
    user_acknowledged: bool | None = None
    recipient_response_observed: bool | None = None
    start_offset_ms: int | None = None
    interpretation_limit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_execution_report(self) -> DeliveryExecutionReport:
        if self.visual.modality is not OutputModality.VISUAL_TEXT:
            raise ValueError("visual execution must describe visual_text")
        if self.voice.modality is not OutputModality.SPOKEN_VOICE:
            raise ValueError("voice execution must describe spoken_voice")
        delivered_count = sum(
            channel.status is OutputExecutionStatus.DELIVERED
            for channel in (self.visual, self.voice)
        )
        expected = {
            2: OverallDeliveryStatus.DELIVERED,
            1: OverallDeliveryStatus.PARTIAL,
            0: OverallDeliveryStatus.FAILED,
        }[delivered_count]
        if self.overall_status is not expected:
            raise ValueError("overall delivery status must match channel execution")
        if self.visual.started_at_ms is not None and self.voice.started_at_ms is not None:
            observed_offset = self.voice.started_at_ms - self.visual.started_at_ms
            if self.start_offset_ms != observed_offset:
                raise ValueError("start_offset_ms must record the observed channel start offset")
        elif self.start_offset_ms is not None:
            raise ValueError("start_offset_ms requires both channel start times")
        return self
