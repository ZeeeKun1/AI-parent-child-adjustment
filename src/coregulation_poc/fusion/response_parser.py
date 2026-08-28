from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from coregulation_poc.models import ConfidenceLevel, CoregulationState, StateAssessment

_ACTORS = {"parent", "child", "both", "unknown"}
_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_STATES = {"normal", "fluctuation", "dysregulation", "high_risk"}
_TASK_PROCESSES = {
    "smooth_progress",
    "brief_stall",
    "sustained_stall",
    "pace_mismatch",
    "explanation_mismatch",
    "over_assistance",
    "disengaged",
    "completion",
    "unclear",
}
_SUPPORT_NEEDS = {
    "none",
    "positive_reinforcement",
    "emotional_support",
    "need_expression",
    "mutual_understanding",
    "task_pacing",
    "learning_support",
    "autonomy_support",
    "unclear",
}
_TRAJECTORIES = {"stable", "worsening", "recovering", "unclear"}
_INTERRUPTIBILITY = {"natural_pause", "active_speech", "task_engaged", "unclear"}
_REGULATION_BALANCE = {"both_stable", "one_stable", "both_crossed", "unclear"}


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


def _normalise_evidence_bundle(
    raw_bundle: object,
    *,
    modality: str,
    limitations: list[str],
) -> dict[str, Any]:
    bundle = raw_bundle if isinstance(raw_bundle, dict) else {}
    raw_items = bundle.get("items")
    items: list[dict[str, Any]] = []
    for raw in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(raw, dict):
            continue
        start_ms = raw.get("start_ms")
        end_ms = raw.get("end_ms")
        if (
            not isinstance(start_ms, int)
            or isinstance(start_ms, bool)
            or not isinstance(end_ms, int)
            or isinstance(end_ms, bool)
            or start_ms < 0
            or end_ms < start_ms
        ):
            limitations.append(f"dropped_invalid_{modality}_evidence_interval")
            continue
        quote = raw.get("quote")
        observation = raw.get("observation")
        if modality == "audio":
            if not isinstance(quote, str) or not quote.strip():
                limitations.append("dropped_audio_evidence_without_quote")
                continue
            if not isinstance(observation, str) or not observation.strip():
                observation = quote.strip()
            frame_timestamp_ms = None
        else:
            if not isinstance(observation, str) or not observation.strip():
                limitations.append("dropped_video_evidence_without_observation")
                continue
            frame_timestamp_ms = raw.get("frame_timestamp_ms")
            if (
                not isinstance(frame_timestamp_ms, int)
                or isinstance(frame_timestamp_ms, bool)
                or not start_ms <= frame_timestamp_ms <= end_ms
            ):
                # A missing redundant frame pointer should not invalidate a
                # timestamped visual observation.  Anchor it to its interval.
                frame_timestamp_ms = start_ms
                limitations.append("normalised_video_frame_timestamp")
            quote = None
        actor = raw.get("actor")
        if actor not in _ACTORS:
            actor = "unknown"
            limitations.append(f"normalised_{modality}_evidence_actor")
        code = raw.get("code")
        if not isinstance(code, str) or not code.strip():
            code = "observable interaction"
        items.append(
            {
                "modality": modality,
                "actor": actor,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "code": code.strip(),
                "observation": str(observation).strip(),
                "quote": quote.strip() if isinstance(quote, str) else None,
                "frame_timestamp_ms": frame_timestamp_ms,
            }
        )
    if items:
        return {"sufficiency": "sufficient", "items": items, "limitation_reason": None}
    reason = bundle.get("limitation_reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = f"No usable {modality} evidence was returned for this window."
    return {"sufficiency": "insufficient", "items": [], "limitation_reason": reason}


def normalize_assessment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair non-critical model formatting without inventing a state.

    Runtime facts, timestamps and the selected state remain subject to strict
    validation.  Auxiliary labels and malformed empty evidence entries are
    normalised so one small JSON inconsistency does not erase the whole window.
    """

    raw_limitations = payload.get("limitations")
    limitations = (
        [str(item).strip() for item in raw_limitations if str(item).strip()]
        if isinstance(raw_limitations, list)
        else []
    )
    allowed_fields = set(StateAssessment.model_fields)
    extra_fields = set(payload) - allowed_fields
    if extra_fields:
        limitations.append("ignored_extra_fields:" + ",".join(sorted(extra_fields)))
    data = {key: value for key, value in payload.items() if key in allowed_fields}

    modality_evidence = data.get("modality_evidence")
    modality_evidence = modality_evidence if isinstance(modality_evidence, dict) else {}
    audio = _normalise_evidence_bundle(
        modality_evidence.get("audio"), modality="audio", limitations=limitations
    )
    video = _normalise_evidence_bundle(
        modality_evidence.get("video"), modality="video", limitations=limitations
    )
    data["modality_evidence"] = {"audio": audio, "video": video}

    state = data.get("state")
    if state not in _STATES:
        state = None
        data["state"] = None
        limitations.append("state_missing_or_invalid")
    has_evidence = audio["sufficiency"] == "sufficient" or video["sufficiency"] == "sufficient"
    if state is None or not has_evidence:
        data.update(
            {
                "state": None,
                "evidence_sufficiency": "insufficient",
                "confidence": "low",
                "alternative_state": None,
                "interaction_performance": [],
                "task_process": None,
                "support_need": None,
                "support_target": "unknown",
            }
        )
        # Overall insufficiency cannot carry a modality marked sufficient.
        if state is None and has_evidence:
            limitations.append("state_not_selected_despite_observable_evidence")
            for bundle in (audio, video):
                if bundle["sufficiency"] == "sufficient":
                    bundle["sufficiency"] = "insufficient"
                    bundle["items"] = []
                    bundle["limitation_reason"] = "No state was selected for the evidence."
    else:
        data["evidence_sufficiency"] = "sufficient"
        if data.get("confidence") not in _CONFIDENCE_LEVELS:
            data["confidence"] = "medium"
            limitations.append("normalised_confidence")

    if data.get("trajectory") not in _TRAJECTORIES:
        data["trajectory"] = "unclear"
    if data.get("task_process") not in _TASK_PROCESSES:
        data["task_process"] = "unclear" if state is not None else None
    if data.get("support_need") not in _SUPPORT_NEEDS:
        data["support_need"] = "unclear" if state is not None else None
    if state != "normal" and data.get("support_need") == "positive_reinforcement":
        data["support_need"] = "unclear"
        limitations.append("normalised_state_incompatible_support_need")
    if data.get("support_target") not in _ACTORS:
        data["support_target"] = "unknown"
    if data.get("interruptibility") not in _INTERRUPTIBILITY:
        data["interruptibility"] = "unclear"

    performances = data.get("interaction_performance")
    data["interaction_performance"] = (
        [item.strip() for item in performances if isinstance(item, str) and item.strip()]
        if isinstance(performances, list)
        else []
    )
    alternative = data.get("alternative_state")
    if alternative not in _STATES or alternative == state:
        data["alternative_state"] = None
    if data.get("confidence") != "high":
        ambiguity = data.get("ambiguity_reason")
        if not isinstance(ambiguity, str) or not ambiguity.strip():
            data["ambiguity_reason"] = "部分辅助线索不完整，状态判断保留不确定性。"

    boundary = data.get("boundary_signals")
    boundary = boundary if isinstance(boundary, dict) else {}
    balance = boundary.get("regulation_balance")
    if balance not in _REGULATION_BALANCE:
        balance = "unclear"
    prompt_count = boundary.get("parental_prompt_count")
    if not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count < 0:
        prompt_count = None
    data["boundary_signals"] = {
        "task_stall_observed": (
            boundary.get("task_stall_observed")
            if isinstance(boundary.get("task_stall_observed"), bool)
            else None
        ),
        "parental_prompt_count": prompt_count,
        "conflict_action_observed": (
            boundary.get("conflict_action_observed")
            if isinstance(boundary.get("conflict_action_observed"), bool)
            else None
        ),
        "child_disengaged_observed": (
            boundary.get("child_disengaged_observed")
            if isinstance(boundary.get("child_disengaged_observed"), bool)
            else None
        ),
        "regulation_balance": balance,
    }
    data.setdefault("previous_state", None)
    data.setdefault("reason", "当前窗口已完成结构化状态判断。")
    data["limitations"] = list(dict.fromkeys(limitations))
    return data


def constrain_assessment_evidence_to_window(
    assessment: StateAssessment,
    *,
    window_start_ms: int,
    window_end_ms: int,
) -> StateAssessment:
    """Clamp small model timestamp drift and drop only unusable evidence items.

    The state model sometimes returns an otherwise valid item a few milliseconds
    beyond the active clip.  That auxiliary timing error must not erase the whole
    window.  Items with no overlap are removed; overlapping items are clamped.
    """

    if window_end_ms <= window_start_ms:
        raise ValueError("window end must be greater than window start")
    payload = assessment.model_dump(mode="python")
    limitations = list(assessment.limitations)
    usable_item_count = 0

    for modality in ("audio", "video"):
        source_bundle = payload["modality_evidence"][modality]
        constrained_items: list[dict[str, Any]] = []
        for item in source_bundle["items"]:
            raw_start = int(item["start_ms"])
            raw_end = int(item["end_ms"])
            start_ms = max(window_start_ms, raw_start)
            end_ms = min(window_end_ms, raw_end)
            if end_ms < start_ms:
                limitations.append(f"dropped_out_of_window_{modality}_evidence")
                continue
            if start_ms != raw_start or end_ms != raw_end:
                limitations.append(f"clamped_{modality}_evidence_to_window")
            item["start_ms"] = start_ms
            item["end_ms"] = end_ms
            if modality == "video":
                frame_ms = item.get("frame_timestamp_ms")
                if not isinstance(frame_ms, int):
                    frame_ms = start_ms
                item["frame_timestamp_ms"] = min(max(frame_ms, start_ms), end_ms)
            constrained_items.append(item)

        if constrained_items:
            usable_item_count += len(constrained_items)
            source_bundle.update(
                {
                    "sufficiency": "sufficient",
                    "items": constrained_items,
                    "limitation_reason": None,
                }
            )
        else:
            source_bundle.update(
                {
                    "sufficiency": "insufficient",
                    "items": [],
                    "limitation_reason": (
                        f"No usable {modality} evidence overlaps the active window."
                    ),
                }
            )

    if usable_item_count == 0:
        payload.update(
            {
                "state": None,
                "evidence_sufficiency": "insufficient",
                "confidence": "low",
                "alternative_state": None,
                "interaction_performance": [],
                "task_process": None,
                "support_need": None,
                "support_target": "unknown",
            }
        )
    payload["assessed_at_ms"] = window_end_ms
    payload["limitations"] = list(dict.fromkeys(limitations))
    return StateAssessment.model_validate(payload)


def validate_assessment_context(
    assessment: StateAssessment,
    *,
    expected_session_id: str,
    duration_ms: int,
    codebook: dict[str, Any],
    history_available: bool = False,
) -> StateAssessment:
    """Validate model output against runtime facts that are not expressible in JSON Schema."""
    if assessment.assessed_at_ms > duration_ms:
        raise ValueError("Assessment time falls outside the source clip.")
    for evidence in assessment.modality_evidence.all_items:
        if evidence.end_ms > duration_ms:
            raise ValueError("Evidence interval falls outside the source clip.")

    updates: dict[str, Any] = {}
    limitations = list(assessment.limitations)
    if assessment.session_id != expected_session_id:
        # The session identifier is a runtime fact, not an inferred label. A
        # model copy error must not invalidate an otherwise usable window.
        updates["session_id"] = expected_session_id
        limitations.append("normalised_session_id_to_runtime")
    if assessment.previous_state is not None and not history_available:
        updates["previous_state"] = None
        limitations.append("ignored_previous_state_without_history")
    if assessment.state is None:
        return assessment.model_copy(update={**updates, "limitations": limitations})

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
        updates["interaction_performance"] = [
            item for item in assessment.interaction_performance if item in allowed_performance
        ]
        limitations.append(
            "ignored_performance_outside_codebook:" + ",".join(sorted(unknown_performance))
        )
    if state_definition.get("history_required") and not history_available:
        dysregulation_definition = states.get("dysregulation", {})
        dysregulation_performance = dysregulation_definition.get("interaction_performance", [])
        updates.update(
            {
                "state": CoregulationState.DYSREGULATION,
                "confidence": ConfidenceLevel.MEDIUM,
                "alternative_state": assessment.state,
                "ambiguity_reason": "缺少跨窗口历史，当前证据按失调状态处理。",
                "interaction_performance": [
                    item
                    for item in assessment.interaction_performance
                    if item in dysregulation_performance
                ],
            }
        )
        limitations.append("high_risk_downgraded_without_history")
    updates["limitations"] = list(dict.fromkeys(limitations))
    return assessment.model_copy(update=updates)


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
        if any(event_type.endswith((".completed", ".done")) for event_type in event_types):
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
        payload = normalize_assessment_payload(_find_json_object(self.response_text))
        try:
            return StateAssessment.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Model JSON did not match StateAssessment: {exc}") from exc
