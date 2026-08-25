from __future__ import annotations

import asyncio
from typing import Any, Protocol

from coregulation_poc.acoustics.feature_extraction import (
    extract_acoustic_features,
    merge_perception_into_acoustic,
)
from coregulation_poc.acoustics.speaker_binding import SpeakerBinding, bind_speakers
from coregulation_poc.acoustics.tencent_voiceprint import (
    SpeakerEnrollmentRecord,
    TencentSpeakerEnrollment,
    TencentVoiceprintService,
)
from coregulation_poc.capture.media import MediaKind
from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.fusion.judgment import (
    build_judgment_system_prompt,
    build_judgment_user_prompt,
    parse_judgment_result,
)
from coregulation_poc.fusion.perception import (
    build_perception_prompt,
    parse_perception_report,
)
from coregulation_poc.fusion.response_parser import (
    RealtimeResponseAccumulator,
    validate_assessment_context,
)
from coregulation_poc.models import (
    AcousticFeatures,
    CoregulationState,
    PerceptionReport,
    StateAssessment,
)
from coregulation_poc.providers.qwen_omni_realtime import QwenOmniRealtimeProvider
from coregulation_poc.providers.qwen_text_chat import QwenTextChatProvider
from coregulation_poc.runtime.window import MediaWindow
from coregulation_poc.settings import Settings


class StateRecognizer(Protocol):
    """Runtime boundary for module-one multimodal state recognition."""

    async def assess(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment: ...


class QwenWindowRecognizer:
    """Two-stage assessor: multimodal perception followed by text-based judgment.

    Stage 1 (Perception): The multimodal model observes the audio-video window
    and produces a :class:`PerceptionReport` containing verbatim speech
    transcriptions and visual behavior descriptions. It does NOT classify
    interaction states.

    Stage 2 (Judgment): A text model receives the perception report, locally
    computed acoustic features (F0, RMS energy, silence gaps), the codebook,
    and assessment history, then produces a :class:`StateAssessment` with
    state classification, evidence, and confidence.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        image_interval_ms: int,
        enrollment: SpeakerEnrollmentRecord | None = None,
        voiceprint_service: TencentVoiceprintService | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> None:
        if settings.dashscope_api_key is None:
            raise ValueError("DASHSCOPE_API_KEY is required for the realtime closed loop")
        if not settings.aliyun_workspace_id or not settings.realtime_base_url:
            raise ValueError("ALIYUN_WORKSPACE_ID is required for the realtime closed loop")
        self.settings = settings
        self.image_interval_ms = image_interval_ms
        self.codebook = load_state_codebook()
        self.api_call_count = 0
        self.voiceprint_api_call_count = 0
        self.last_speaker_binding: SpeakerBinding | None = None
        self.last_perception_report: PerceptionReport | None = None
        self.last_acoustic_features: AcousticFeatures | None = None
        self.enrollment = enrollment
        self.voiceprint_service = voiceprint_service
        self.task_context = task_context

        # Stage-2 text chat provider for judgment
        self._judgment_provider = QwenTextChatProvider(
            api_key=settings.dashscope_api_key.get_secret_value(),
            model=settings.judgment_model,
            temperature=settings.judgment_temperature,
            max_tokens=settings.judgment_max_tokens,
            timeout_seconds=settings.judgment_timeout_seconds,
        )
        # Pre-build the judgment system prompt (codebook is static)
        self._judgment_system_prompt = build_judgment_system_prompt(
            codebook=self.codebook,
        )

    async def assess(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment:
        """Run the two-stage assessment pipeline."""
        # --- Speaker binding ------------------------------------------------
        # Formal browser sessions use Tencent 1:N voiceprint matching. Legacy
        # file enrollment remains available for offline/local diagnostics.
        audio_chunks = window.audio_chunks
        if isinstance(self.enrollment, TencentSpeakerEnrollment):
            if self.voiceprint_service is None:
                raise RuntimeError(
                    "Tencent voiceprint enrollment is present but the service is unavailable"
                )
            binding = await asyncio.to_thread(
                self.voiceprint_service.identify_speakers,
                audio_chunks,
                self.enrollment,
            )
            self.voiceprint_api_call_count += binding.provider_request_count
            self.api_call_count += binding.provider_request_count
        else:
            binding = await asyncio.to_thread(
                bind_speakers,
                audio_chunks,
                enrollment=self.enrollment,
                allow_f0_fallback=False,
            )
        self.last_speaker_binding = binding
        speaker_binding_description = (
            binding.to_prompt_description() if binding.bound else None
        )

        # --- Stage 1: Perception (multimodal model via WebSocket) ------------
        perception_report = await self._assess_perception(
            session_id=session_id,
            window=window,
            speaker_binding_description=speaker_binding_description,
        )
        self.last_perception_report = perception_report

        # --- Local acoustic feature extraction -------------------------------
        acoustic_features = await asyncio.to_thread(
            extract_acoustic_features, audio_chunks, binding
        )
        acoustic_features = merge_perception_into_acoustic(
            acoustic_features, perception_report.speech_turns
        )
        self.last_acoustic_features = acoustic_features

        # --- Stage 2: Judgment (text model via HTTP) -------------------------
        assessment = await self._assess_judgment(
            session_id=session_id,
            window=window,
            previous_state=previous_state,
            history=history,
            history_available=history_available,
            perception_report=perception_report,
            acoustic_features=acoustic_features,
        )

        return assessment

    async def _assess_perception(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        speaker_binding_description: str | None,
    ) -> PerceptionReport:
        """Stage 1: Use the multimodal model to observe and report."""
        prompt = build_perception_prompt(
            session_id=session_id,
            window_start_ms=window.start_ms,
            window_end_ms=window.end_ms,
            image_interval_ms=self.image_interval_ms,
            speaker_binding_description=speaker_binding_description,
        )

        provider = QwenOmniRealtimeProvider(
            model=self.settings.omni_model,
            api_key=self.settings.dashscope_api_key.get_secret_value(),
            workspace_id=self.settings.aliyun_workspace_id,
            base_url=self.settings.realtime_base_url,
            instructions=prompt,
            connection_timeout_seconds=self.settings.connection_timeout_seconds,
        )
        accumulator = RealtimeResponseAccumulator()
        self.api_call_count += 1
        try:
            await provider.connect()
            for chunk in window.chunks:
                if chunk.kind is MediaKind.AUDIO:
                    await provider.send_audio(chunk.payload, chunk.timestamp_ms)
                else:
                    await provider.send_frame(chunk.payload, chunk.timestamp_ms)
            await provider.finish_input()
            async with asyncio.timeout(self.settings.response_timeout_seconds):
                async for envelope in provider.events():
                    event = envelope.get("event")
                    if isinstance(event, dict):
                        accumulator.add(event)
            return parse_perception_report(accumulator.response_text)
        finally:
            await provider.close()

    async def _assess_judgment(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
        perception_report: PerceptionReport,
        acoustic_features: AcousticFeatures,
    ) -> StateAssessment:
        """Stage 2: Use the text model to classify the coregulation state."""
        previous_value = None if previous_state is None else previous_state.value
        history_summary = [
            {
                "assessed_at_ms": item.assessed_at_ms,
                "state": None if item.state is None else item.state.value,
                "confidence": item.confidence.value,
                "interaction_performance": item.interaction_performance,
                "task_process": (
                    item.task_process.value if item.task_process is not None else None
                ),
                "support_need": (
                    item.support_need.value if item.support_need is not None else None
                ),
                "trajectory": item.trajectory.value,
            }
            for item in history
        ]

        user_prompt = build_judgment_user_prompt(
            session_id=session_id,
            window_start_ms=window.start_ms,
            window_end_ms=window.end_ms,
            previous_state=previous_value,
            history_summary=history_summary,
            perception_report=perception_report,
            acoustic_features=acoustic_features,
            task_context=self.task_context,
        )

        self.api_call_count += 1
        result = await asyncio.to_thread(
            self._judgment_provider.generate_structured,
            system_prompt=self._judgment_system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.settings.judgment_max_tokens,
            temperature=self.settings.judgment_temperature,
        )

        assessment = parse_judgment_result(result.text)

        # Override runtime-controlled fields
        assessment = assessment.model_copy(
            update={
                "assessed_at_ms": window.end_ms,
                "previous_state": previous_state,
            }
        )

        # Validate against runtime facts
        validate_assessment_context(
            assessment,
            expected_session_id=session_id,
            duration_ms=window.end_ms,
            codebook=self.codebook,
            history_available=history_available,
        )

        # Clamp evidence timestamps to the active window
        if any(
            evidence.start_ms < window.start_ms
            for evidence in assessment.modality_evidence.all_items
        ):
            raise ValueError("model evidence falls before the active realtime window")

        return assessment
