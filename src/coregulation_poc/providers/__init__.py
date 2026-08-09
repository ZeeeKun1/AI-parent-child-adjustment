"""Cloud provider adapters for realtime multimodal inference and voice output."""

from coregulation_poc.providers.qwen_text_chat import QwenTextChatProvider, TextChatResult
from coregulation_poc.providers.qwen_tts_realtime import (
    QwenRealtimeTTSProvider,
    SpeechSynthesisResult,
    write_pcm_wav,
)

__all__ = [
    "QwenRealtimeTTSProvider",
    "QwenTextChatProvider",
    "SpeechSynthesisResult",
    "TextChatResult",
    "write_pcm_wav",
]
