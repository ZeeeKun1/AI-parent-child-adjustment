from __future__ import annotations

from coregulation_poc.delivery.models import (
    ChannelExecution,
    DeliveryExecutionReport,
    DeliveryHoldReason,
    DeliveryPackage,
    DeliveryPolicy,
    DeliveryPreparationResult,
    DeliveryPreparationStatus,
    DeliveryRuntimeContext,
    OutputExecutionStatus,
    OutputModality,
    OverallDeliveryStatus,
    VisualPrompt,
    VoicePrompt,
)
from coregulation_poc.intervention.models import InterventionPlan


class DeliveryCoordinator:
    """Translate one authorized strategy plan into auditable text and voice output."""

    def __init__(self, policy: DeliveryPolicy) -> None:
        self.policy = policy

    def prepare(
        self,
        *,
        plan: InterventionPlan,
        runtime: DeliveryRuntimeContext,
    ) -> DeliveryPreparationResult:
        if runtime.interventions_paused:
            return DeliveryPreparationResult(
                status=DeliveryPreparationStatus.HELD,
                hold_reason=DeliveryHoldReason.INTERVENTIONS_PAUSED,
            )

        voice_enabled = runtime.voice_enabled and runtime.voice_available
        status = (
            DeliveryPreparationStatus.READY
            if voice_enabled
            else DeliveryPreparationStatus.DEGRADED
        )
        fallback_reason = None
        if not voice_enabled:
            fallback_reason = (
                "voice_disabled_by_runtime"
                if not runtime.voice_enabled
                else "voice_output_unavailable"
            )

        package = DeliveryPackage(
            delivery_id=f"{plan.session_id}:{plan.sequence}:{plan.strategy_id}",
            session_id=plan.session_id,
            sequence=plan.sequence,
            planned_at_ms=plan.planned_at_ms,
            prepared_at_ms=runtime.prepared_at_ms,
            delivery_policy_version=self.policy.version,
            strategy_id=plan.strategy_id,
            target_actor=plan.target_actor,
            repair_target=plan.repair_target,
            message_source=plan.message_source,
            visual_prompt=VisualPrompt(
                target_actor=plan.target_actor,
                heading=self.policy.visual.headings_by_target[plan.target_actor],
                message=plan.message,
                prominence=self.policy.visual.prominence,
                placement=self.policy.visual.placement,
                blocks_primary_task=self.policy.visual.blocks_primary_task,
                dismissible=self.policy.visual.dismissible,
            ),
            voice_prompt=VoicePrompt(
                target_actor=plan.target_actor,
                message=plan.message,
                provider=self.policy.voice.provider,
                model=self.policy.voice.model,
                voice=self.policy.voice.voice,
                language=self.policy.voice.language,
                language_type=self.policy.voice.language_type,
                response_format=self.policy.voice.response_format,
                sample_rate_hz=self.policy.voice.sample_rate_hz,
                mode=self.policy.voice.mode,
                instructions=self.policy.voice.instructions,
                optimize_instructions=self.policy.voice.optimize_instructions,
                style=self.policy.voice.style,
                autoplay=self.policy.voice.autoplay,
                enabled=voice_enabled,
            ),
            status=status,
            core_content_identical=True,
            fallback_reason=fallback_reason,
            research_basis=self.policy.source,
        )
        return DeliveryPreparationResult(status=status, package=package)

    @staticmethod
    def record_execution(
        *,
        package: DeliveryPackage,
        recorded_at_ms: int,
        visual: ChannelExecution,
        voice: ChannelExecution,
        user_acknowledged: bool | None = None,
        recipient_response_observed: bool | None = None,
    ) -> DeliveryExecutionReport:
        delivered_count = sum(
            channel.status is OutputExecutionStatus.DELIVERED
            for channel in (visual, voice)
        )
        overall_status = {
            2: OverallDeliveryStatus.DELIVERED,
            1: OverallDeliveryStatus.PARTIAL,
            0: OverallDeliveryStatus.FAILED,
        }[delivered_count]
        start_offset_ms = None
        if visual.started_at_ms is not None and voice.started_at_ms is not None:
            start_offset_ms = voice.started_at_ms - visual.started_at_ms
        return DeliveryExecutionReport(
            delivery_id=package.delivery_id,
            session_id=package.session_id,
            sequence=package.sequence,
            recorded_at_ms=recorded_at_ms,
            visual=visual,
            voice=voice,
            overall_status=overall_status,
            user_acknowledged=user_acknowledged,
            recipient_response_observed=recipient_response_observed,
            start_offset_ms=start_offset_ms,
            interpretation_limit=(
                "输出成功只表示界面已呈现或语音已播放，不能据此推断参与者已经看到、"
                "听到、理解或采纳；亲子回应由后续观察单独判断。"
            ),
        )


def not_attempted_voice_execution() -> ChannelExecution:
    return ChannelExecution(
        modality=OutputModality.SPOKEN_VOICE,
        status=OutputExecutionStatus.NOT_ATTEMPTED,
    )
