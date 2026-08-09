from __future__ import annotations

import asyncio
import json
from typing import Protocol

from coregulation_poc.capture.media import MediaKind
from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.fusion.prompting import build_state_assessment_prompt
from coregulation_poc.fusion.response_parser import (
    RealtimeResponseAccumulator,
    validate_assessment_context,
)
from coregulation_poc.models import CoregulationState, StateAssessment
from coregulation_poc.providers.qwen_omni_realtime import QwenOmniRealtimeProvider
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
    """Assess one already-captured rolling window with Qwen Omni Realtime."""

    def __init__(
        self,
        *,
        settings: Settings,
        image_interval_ms: int,
    ) -> None:
        if settings.dashscope_api_key is None:
            raise ValueError("DASHSCOPE_API_KEY is required for the realtime closed loop")
        if not settings.aliyun_workspace_id or not settings.realtime_base_url:
            raise ValueError("ALIYUN_WORKSPACE_ID is required for the realtime closed loop")
        self.settings = settings
        self.image_interval_ms = image_interval_ms
        self.codebook = load_state_codebook()
        self.api_call_count = 0

    async def assess(
        self,
        *,
        session_id: str,
        window: MediaWindow,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
        history_available: bool,
    ) -> StateAssessment:
        previous_value = None if previous_state is None else previous_state.value
        history_summary = [
            {
                "assessed_at_ms": item.assessed_at_ms,
                "state": None if item.state is None else item.state.value,
                "confidence": item.confidence.value,
                "interaction_performance": item.interaction_performance,
            }
            for item in history
        ]
        prompt = build_state_assessment_prompt(
            session_id=session_id,
            duration_ms=window.end_ms,
            codebook=self.codebook,
            image_interval_ms=self.image_interval_ms,
            speaker_roles_bound=False,
        )
        prompt += "\n" + "\n".join(
            [
                f"This assessment window covers session time {window.start_ms}-{window.end_ms} ms.",
                f"Set assessed_at_ms exactly to {window.end_ms}.",
                (
                    "Use the session timeline shown in frame labels for all evidence times; "
                    "do not reset timestamps to zero for this window."
                ),
                (
                    "The runtime-provided previous_state is "
                    f"{previous_value}; copy this value exactly, including null."
                ),
                (
                    "Compact prior assessment history (context only; current-window evidence is "
                    "still required): "
                    f"{json.dumps(history_summary, ensure_ascii=False)}"
                ),
            ]
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
            assessment = accumulator.parse_assessment().model_copy(
                update={
                    "assessed_at_ms": window.end_ms,
                    "previous_state": previous_state,
                }
            )
            validate_assessment_context(
                assessment,
                expected_session_id=session_id,
                duration_ms=window.end_ms,
                codebook=self.codebook,
                history_available=history_available,
            )
            if any(
                evidence.start_ms < window.start_ms
                for evidence in assessment.modality_evidence.all_items
            ):
                raise ValueError("model evidence falls before the active realtime window")
            return assessment
        finally:
            await provider.close()
