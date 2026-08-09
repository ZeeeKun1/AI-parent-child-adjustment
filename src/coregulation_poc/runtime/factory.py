from __future__ import annotations

import asyncio

from coregulation_poc.capture.media import MediaFormat
from coregulation_poc.delivery import load_delivery_policy
from coregulation_poc.providers.qwen_tts_realtime import QwenRealtimeTTSProvider
from coregulation_poc.runtime.recognition import QwenWindowRecognizer
from coregulation_poc.runtime.session import (
    RealtimeLoopConfig,
    RealtimeSession,
    RealtimeSessionFactory,
    ServerEventSender,
    VoiceAudio,
)
from coregulation_poc.settings import Settings


class QwenVoiceSynthesizer:
    """Async runtime adapter for the synchronous Qwen Maia TTS client."""

    def __init__(self, settings: Settings) -> None:
        if settings.dashscope_api_key is None:
            raise ValueError("DASHSCOPE_API_KEY is required when realtime voice is enabled")
        policy = load_delivery_policy()
        voice = policy.voice
        self.provider_name = voice.provider
        self.provider = QwenRealtimeTTSProvider(
            model=voice.model,
            voice=voice.voice,
            api_key=settings.dashscope_api_key.get_secret_value(),
            base_url=settings.resolved_tts_base_url,
            language_type=voice.language_type,
            response_format=voice.response_format,
            sample_rate_hz=voice.sample_rate_hz,
            mode=voice.mode,
            instructions=voice.instructions,
            optimize_instructions=voice.optimize_instructions,
            workspace_id=settings.aliyun_workspace_id,
            connection_timeout_seconds=settings.connection_timeout_seconds,
            response_timeout_seconds=settings.response_timeout_seconds,
        )

    async def synthesize(self, text: str) -> VoiceAudio:
        result = await asyncio.to_thread(self.provider.synthesize, text)
        return VoiceAudio(
            pcm_audio=result.pcm_audio,
            sample_rate_hz=result.sample_rate_hz,
            provider=self.provider_name,
            output_identifier=result.response_id,
        )


def build_realtime_session_factory(
    *,
    settings: Settings,
    media_format: MediaFormat,
    config: RealtimeLoopConfig,
) -> RealtimeSessionFactory:
    """Build per-browser sessions without making an API call at server startup."""

    if settings.dashscope_api_key is None:
        raise ValueError("DASHSCOPE_API_KEY is required for the realtime closed loop")
    if not settings.aliyun_workspace_id or not settings.realtime_base_url:
        raise ValueError("ALIYUN_WORKSPACE_ID is required for the realtime closed loop")

    def create(session_id: str, send_event: ServerEventSender) -> RealtimeSession:
        recognizer = QwenWindowRecognizer(
            settings=settings,
            image_interval_ms=media_format.image_interval_ms,
        )
        voice = QwenVoiceSynthesizer(settings) if config.voice_enabled else None
        return RealtimeSession(
            session_id=session_id,
            recognizer=recognizer,
            send_event=send_event,
            config=config,
            voice_synthesizer=voice,
        )

    return create
