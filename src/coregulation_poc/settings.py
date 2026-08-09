from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from coregulation_poc.paths import (
    DEFAULT_CACHE_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
    ENV_FILE,
    resolve_project_path,
)


class Settings(BaseSettings):
    """Environment-backed settings with absolute runtime paths."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dashscope_api_key: SecretStr | None = None
    aliyun_workspace_id: str | None = None
    aliyun_region: str = "cn-beijing"
    omni_model: str = "qwen3.5-omni-flash-realtime"
    tts_realtime_base_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    connection_timeout_seconds: int = Field(default=20, ge=5, le=120)
    response_timeout_seconds: int = Field(default=90, ge=10, le=300)

    text_chat_model: str = "qwen-plus"
    text_chat_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    text_chat_max_tokens: int = Field(default=128, ge=16, le=1024)
    text_chat_timeout_seconds: float = Field(default=10, ge=3, le=60)

    input_dir: Path = Field(default=DEFAULT_INPUT_DIR)
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)
    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR)
    log_dir: Path = Field(default=DEFAULT_LOG_DIR)

    @field_validator("input_dir", "output_dir", "cache_dir", "log_dir", mode="before")
    @classmethod
    def make_absolute(cls, value: str | Path) -> Path:
        return resolve_project_path(value)

    @property
    def realtime_endpoint(self) -> str | None:
        base_url = self.realtime_base_url
        return f"{base_url}?model={self.omni_model}" if base_url else None

    @property
    def realtime_base_url(self) -> str | None:
        """Return the workspace-specific endpoint without the SDK model query."""
        if not self.aliyun_workspace_id:
            return None
        return (
            f"wss://{self.aliyun_workspace_id}.{self.aliyun_region}.maas.aliyuncs.com/"
            "api-ws/v1/realtime"
        )

    @property
    def resolved_tts_base_url(self) -> str:
        """Use the workspace-specific endpoint for TTS when available."""
        if self.aliyun_workspace_id:
            return (
                f"wss://{self.aliyun_workspace_id}.{self.aliyun_region}.maas.aliyuncs.com/"
                "api-ws/v1/realtime"
            )
        return self.tts_realtime_base_url

    def text_chat_api_key(self) -> str | None:
        """Return the API key for text chat, or None if not configured."""
        if self.dashscope_api_key is None:
            return None
        return self.dashscope_api_key.get_secret_value()
