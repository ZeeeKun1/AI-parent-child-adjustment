from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from coregulation_poc.models import Actor, ConfidenceLevel
from coregulation_poc.providers.qwen_text_chat import TextChatResult

if TYPE_CHECKING:
    from coregulation_poc.runtime.recognition import WindowObservation


class BaselineAction(StrEnum):
    NO_INTERVENTION = "no_intervention"
    INTERVENE = "intervene"


class BaselineDecision(BaseModel):
    """A direct generic-facilitator decision with no research-state semantics."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    window_start_ms: int = Field(ge=0)
    window_end_ms: int = Field(ge=0)
    action: BaselineAction
    target_actor: Actor = Actor.UNKNOWN
    message: str | None = None
    reason: str = Field(min_length=1, max_length=160)
    confidence: ConfidenceLevel
    model: str = Field(min_length=1)
    model_latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> BaselineDecision:
        if self.window_end_ms < self.window_start_ms:
            raise ValueError("window_end_ms cannot precede window_start_ms")
        if self.action is BaselineAction.NO_INTERVENTION:
            if self.message is not None:
                raise ValueError("no_intervention cannot contain a message")
            if self.target_actor is not Actor.UNKNOWN:
                raise ValueError("no_intervention cannot claim a target actor")
        else:
            if self.target_actor is Actor.UNKNOWN:
                raise ValueError("an intervention requires a target actor")
            validate_baseline_message(self.message or "")
        return self


class _BaselineModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BaselineAction
    target_actor: Actor | None = None
    message: str | None = None
    reason: str = Field(min_length=1, max_length=160)
    confidence: ConfidenceLevel


class StructuredTextProvider(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> TextChatResult: ...


BASELINE_SYSTEM_PROMPT = """You are a general facilitator for a parent and child doing
homework together.
Use only the recent observations supplied to you. Decide directly whether a brief prompt would
help the interaction right now. If they can continue without help, choose no_intervention. If a
prompt would help, choose intervene, address parent, child, or both, and write one or two short,
natural Chinese sentences that can be shown immediately. Be neutral and actionable. Do not blame,
diagnose, evaluate either person, invent facts, or give the homework answer. Avoid repeating recent
prompts. Return only this JSON object:
{"action":"no_intervention|intervene","target_actor":"parent|child|both|null",
"message":"Chinese prompt or null","reason":"brief factual reason","confidence":"low|medium|high"}
"""

_BANNED_PHRASES = (
    "答案是",
    "你错了",
    "你怎么",
    "你总是",
    "你从不",
    "必须",
    "闭嘴",
    "笨",
)


def validate_baseline_message(message: str) -> None:
    cleaned = re.sub(r"\s+", " ", message).strip()
    if not cleaned:
        raise ValueError("baseline intervention message is empty")
    if len(cleaned) > 90:
        raise ValueError("baseline intervention message is too long")
    sentences = [part for part in re.split(r"(?<=[。！？!?])", cleaned) if part.strip()]
    if len(sentences) > 2:
        raise ValueError("baseline intervention message has too many sentences")
    if any(phrase in cleaned for phrase in _BANNED_PHRASES):
        raise ValueError("baseline intervention message contains unsafe wording")


def _clean_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("baseline response must be a JSON object")
    return value


def build_baseline_user_prompt(
    *,
    observation: WindowObservation,
    task_context: dict[str, Any] | None,
    recent_messages: list[str],
) -> str:
    perception = observation.perception_report.model_dump(mode="json")
    acoustic = observation.acoustic_features.model_dump(mode="json")
    payload = {
        "observation_window_ms": [observation.window.start_ms, observation.window.end_ms],
        "task_context": task_context or {},
        "perception_report": perception,
        "acoustic_features": acoustic,
        "recent_prompts_shown": recent_messages[-3:],
    }
    return (
        "Review this single recent observation window and make the direct facilitation decision.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


class BaselineDecisionGenerator:
    def __init__(self, provider: StructuredTextProvider) -> None:
        self.provider = provider
        self.call_count = 0

    def generate(
        self,
        *,
        observation: WindowObservation,
        sequence: int,
        task_context: dict[str, Any] | None,
        recent_messages: list[str],
    ) -> BaselineDecision:
        user_prompt = build_baseline_user_prompt(
            observation=observation,
            task_context=task_context,
            recent_messages=recent_messages,
        )
        last_error: Exception | None = None
        for attempt in range(2):
            self.call_count += 1
            try:
                result = self.provider.generate_structured(
                    system_prompt=BASELINE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=256,
                    temperature=0.2,
                )
                parsed = _BaselineModelOutput.model_validate(_clean_json(result.text))
                action = parsed.action
                target = parsed.target_actor or Actor.UNKNOWN
                message = None
                if action is BaselineAction.INTERVENE:
                    message = re.sub(r"\s+", " ", parsed.message or "").strip()
                else:
                    target = Actor.UNKNOWN
                return BaselineDecision(
                    session_id=observation.session_id,
                    sequence=sequence,
                    window_start_ms=observation.window.start_ms,
                    window_end_ms=observation.window.end_ms,
                    action=action,
                    target_actor=target,
                    message=message,
                    reason=parsed.reason.strip(),
                    confidence=parsed.confidence,
                    model=result.model,
                    model_latency_ms=result.total_latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            except (ConnectionError, OSError, TimeoutError, ValueError, ValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    user_prompt += (
                        "\nThe previous response was invalid. Return only the required JSON."
                    )
        if last_error is not None:
            raise last_error
        raise RuntimeError("baseline decision failed without an error")
