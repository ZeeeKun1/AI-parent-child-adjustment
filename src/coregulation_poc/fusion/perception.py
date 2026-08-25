"""Stage-1 perception prompt construction and response parsing.

The perception stage uses the multimodal model to ONLY observe and report
what is directly seen and heard in the audio-video window. It does NOT
classify interaction states, assess evidence sufficiency, or make judgments.

The output :class:`PerceptionReport` is passed to the stage-2 judgment
model along with locally computed acoustic features.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from coregulation_poc.models import PerceptionReport


def build_perception_prompt(
    *,
    session_id: str,
    window_start_ms: int,
    window_end_ms: int,
    image_interval_ms: int,
    speaker_binding_description: str | None = None,
) -> str:
    """Build the instruction prompt for the stage-1 perception model.

    Parameters
    ----------
    session_id
        Pseudonymous session identifier.
    window_start_ms, window_end_ms
        The session-time range of the current assessment window.
    image_interval_ms
        Approximate interval between sampled video frames.
    speaker_binding_description
        Natural-language description of speaker binding results, or None
        when binding is unavailable.
    """
    schema = PerceptionReport.model_json_schema()

    if speaker_binding_description:
        speaker_instruction = (
            "Speaker roles are externally bound. Use the binding below to assign "
            "speaker=parent or speaker=child in speech_turns. The binding is "
            "probabilistic; if the content clearly contradicts the binding for a "
            "specific utterance, use speaker=unknown and note the discrepancy in "
            "the text field.\n\n"
            + speaker_binding_description
        )
    else:
        speaker_instruction = (
            "No external speaker-role binding is available. Use speaker=unknown "
            "unless the role is directly and unambiguously identifiable from "
            "voice characteristics or visual context."
        )

    return "\n".join(
        [
            (
                "You are observing one parent-child homework episode through "
                "chronological audio and images."
            ),
            (
                "Your ONLY task is to observe and report what you see and hear. "
                "Do NOT classify, judge, assess, or label the interaction state. "
                "Do NOT determine whether the interaction is normal, fluctuating, "
                "dysregulated, or high-risk."
            ),
            (
                "For audio: transcribe what each speaker says, verbatim. "
                "Quote the exact words. Do not correct, translate, paraphrase, "
                "or summarize."
            ),
            (
                "For video: describe only directly visible behavior at each "
                "frame timestamp. Report what the parent, child, or both are "
                "physically doing. Do NOT infer emotion, intention, mental state, "
                "or relationship dynamics."
            ),
            (
                "If audio is unclear, partially inaudible, or missing, set "
                "audio_limitation to a brief explanation and include only the "
                "speech turns you can reliably transcribe."
            ),
            (
                "If video is dark, occluded, or missing, set video_limitation to "
                "a brief explanation and include only the visual observations you "
                "can reliably describe."
            ),
            speaker_instruction,
            f"The pseudonymous session_id is {session_id}.",
            (
                f"This observation window covers session time "
                f"{window_start_ms}-{window_end_ms} ms."
            ),
            (
                f"Images are sampled chronologically about every {image_interval_ms} ms."
            ),
            (
                "Each image includes a frame_time_ms label. Audio timestamps are "
                "approximate offsets from the beginning of the clip in milliseconds."
            ),
            (
                "Use the session timeline shown in frame labels for all timestamps; "
                "do not reset timestamps to zero for this window."
            ),
            "Return exactly one JSON object. Do not use Markdown or add commentary.",
            (
                "Use English enum values for speaker (parent/child/both/unknown) and "
                "actor (parent/child/both/unknown). Transcribed text and descriptions "
                "may be in the original language."
            ),
            "REQUIRED JSON SCHEMA:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def _find_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text string."""
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
    raise ValueError("The perception response did not contain a JSON object.")


def parse_perception_report(response_text: str) -> PerceptionReport:
    """Parse the model's JSON response into a :class:`PerceptionReport`.

    Parameters
    ----------
    response_text
        The accumulated text response from the multimodal model.
    """
    payload = _find_json_object(response_text)
    try:
        return PerceptionReport.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Perception JSON did not match PerceptionReport: {exc}") from exc
