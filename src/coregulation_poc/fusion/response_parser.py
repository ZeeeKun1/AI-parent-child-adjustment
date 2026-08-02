from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from coregulation_poc.models import StateAssessment


def _find_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The model response did not contain a JSON object.")


def validate_assessment_context(
    assessment: StateAssessment,
    *,
    expected_session_id: str,
    duration_ms: int,
    codebook: dict[str, Any],
    history_available: bool = False,
) -> None:
    """Validate model output against runtime facts that are not expressible in JSON Schema."""
    if assessment.session_id != expected_session_id:
        raise ValueError("Assessment session_id does not match the requested session.")
    if assessment.assessed_at_ms > duration_ms:
        raise ValueError("Assessment time falls outside the source clip.")
    for evidence in assessment.modality_evidence.all_items:
        if evidence.end_ms > duration_ms:
            raise ValueError("Evidence interval falls outside the source clip.")

    if assessment.previous_state is not None and not history_available:
        raise ValueError("Assessment supplied previous_state without available history.")
    if assessment.state is None:
        return

    states = codebook.get("states")
    if not isinstance(states, dict):
        raise ValueError("Codebook states are missing or invalid.")
    state_definition = states.get(assessment.state.value)
    if not isinstance(state_definition, dict):
        raise ValueError("Assessment state is not present in the codebook.")
    allowed_performance = state_definition.get("interaction_performance", [])
    if not isinstance(allowed_performance, list):
        raise ValueError("Codebook interaction_performance is invalid.")
    unknown_performance = set(assessment.interaction_performance) - set(allowed_performance)
    if unknown_performance:
        unknown = ", ".join(sorted(unknown_performance))
        raise ValueError(f"Assessment used interaction performance outside the codebook: {unknown}")
    if state_definition.get("history_required") and not history_available:
        raise ValueError(
            f"State {assessment.state.value} requires unavailable interaction history."
        )


@dataclass(slots=True)
class RealtimeResponseAccumulator:
    text_deltas: list[str] = field(default_factory=list)
    final_text_candidates: list[str] = field(default_factory=list)
    transcript_events: list[dict[str, Any]] = field(default_factory=list)
    server_errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type in {"response.text.delta", "response.audio_transcript.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                self.text_deltas.append(delta)
        elif event_type in {"response.text.done", "response.audio_transcript.done"}:
            final_text = event.get("text", event.get("transcript"))
            if isinstance(final_text, str):
                self.final_text_candidates.append(final_text)
        elif event_type == "response.done":
            response = event.get("response")
            if isinstance(response, dict):
                output = response.get("output", [])
                if isinstance(output, list):
                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        content = item.get("content", [])
                        if not isinstance(content, list):
                            continue
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            final_text = part.get("text", part.get("transcript"))
                            if isinstance(final_text, str):
                                self.final_text_candidates.append(final_text)
        elif event_type.startswith("conversation.item.input_audio_transcription"):
            self.transcript_events.append(event)
        elif event_type == "error":
            self.server_errors.append(event)

    @property
    def response_text(self) -> str:
        if self.text_deltas:
            return "".join(self.text_deltas).strip()
        return "\n".join(self.final_text_candidates).strip()

    @property
    def transcription_failures(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.transcript_events
            if str(event.get("type", "")).endswith(".failed")
        ]

    @property
    def transcription_status(self) -> str:
        event_types = {str(event.get("type", "")) for event in self.transcript_events}
        if any(event_type.endswith(".failed") for event_type in event_types):
            return "failed"
        if any(
            event_type.endswith((".completed", ".done")) for event_type in event_types
        ):
            return "completed"
        if self.transcript_events:
            return "partial"
        return "not_observed"

    @property
    def audit_warnings(self) -> list[str]:
        warnings: list[str] = []
        for failure in self.transcription_failures:
            error = failure.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            warnings.append(f"input_audio_transcription_failed:{code or 'unknown'}")
        return warnings

    @property
    def best_effort_input_transcript(self) -> str:
        candidates: list[str] = []
        for event in self.transcript_events:
            text = event.get("text")
            stash = event.get("stash")
            combined = "".join(part for part in (text, stash) if isinstance(part, str))
            if combined:
                candidates.append(combined)
        return max(candidates, key=len, default="").strip()

    @property
    def input_emotion_observations(self) -> list[dict[str, Any]]:
        """Return compact Qwen ASR emotion labels without treating them as state labels."""
        allowed = {"neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"}
        observations: dict[str, dict[str, Any]] = {}
        for sequence, event in enumerate(self.transcript_events):
            emotion = event.get("emotion")
            if not isinstance(emotion, str) or emotion not in allowed:
                continue
            item_id = event.get("item_id")
            key = str(item_id) if item_id is not None else f"event-{sequence}"
            text = event.get("text")
            stash = event.get("stash")
            combined = "".join(part for part in (text, stash) if isinstance(part, str))
            observations[key] = {
                "item_id": item_id,
                "emotion": emotion,
                "text": combined.strip(),
                "source_event_type": str(event.get("type", "")),
                "interpretation_role": "supporting_observation_only",
            }
        return list(observations.values())

    def parse_assessment(self) -> StateAssessment:
        if self.server_errors:
            raise ValueError(f"Realtime API returned an error: {self.server_errors[-1]}")
        payload = _find_json_object(self.response_text)
        try:
            return StateAssessment.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Model JSON did not match StateAssessment: {exc}") from exc
