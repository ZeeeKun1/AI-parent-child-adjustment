from __future__ import annotations

import json

from coregulation_poc.models import StateAssessment


def build_state_assessment_prompt(
    *,
    session_id: str,
    duration_ms: int,
    codebook: dict[str, object],
    image_interval_ms: int,
    speaker_roles_bound: bool = False,
    speaker_binding_description: str | None = None,
) -> str:
    schema = StateAssessment.model_json_schema()
    if speaker_roles_bound and speaker_binding_description:
        speaker_instruction = (
            "Speaker roles are bound by offline F0-based clustering. "
            "Use the binding below to assign actor=parent or actor=child in evidence. "
            "The binding is acoustic and probabilistic; if visual or content evidence "
            "clearly contradicts it for a specific utterance, note the discrepancy but "
            "still use the binding as the default.\n\n"
            + speaker_binding_description
        )
    elif speaker_roles_bound:
        speaker_instruction = (
            "Parent and child speaker roles are externally bound; use only the supplied binding."
        )
    else:
        speaker_instruction = (
            "No external speaker-role binding is available. Use actor=unknown unless the role "
            "is directly and unambiguously observable; never guess from content alone."
        )
    return "\n".join(
        [
            (
                "You are observing one parent-child homework episode through chronological "
                "audio and images."
            ),
            "Classify the dyad using only the supplied formative-study codebook.",
            "Assess the interaction sequence and task progression, not a single word or gesture.",
            (
                "If observable evidence is insufficient, set state to null and "
                "evidence_sufficiency to insufficient."
            ),
            (
                "Audio and video evidence are evaluated independently. Do not invent evidence "
                "to make both modalities sufficient; one sufficient modality may support the "
                "overall assessment."
            ),
            (
                "For audio evidence, quote the exact observed words in quote. Do not correct, "
                "translate, or paraphrase the quote."
            ),
            (
                "Speech pace, loudness, or pitch changes may support an observation only when "
                "the change is directly audible across the sequence and the actor is identifiable. "
                "Never treat a raw acoustic value as an emotion or state label, and never classify "
                "the dyad from one acoustic feature alone."
            ),
            (
                "For video evidence, describe only directly visible behavior in observation, "
                "set quote to null, and cite the embedded frame_time_ms label. Do not infer "
                "unobservable emotion or intention."
            ),
            (
                "When a modality is insufficient, set its sufficiency to insufficient, leave "
                "items empty, and explain the limitation in limitation_reason."
            ),
            speaker_instruction,
            (
                "Do not classify high_risk from one isolated turn; it requires a persistent "
                "pattern in the clip."
            ),
            (
                "An isolated wrong answer followed by smooth supportive correction can remain "
                "normal. Use fluctuation only when the sequence shows temporarily uneven "
                "coordination while the dyad still retains recovery capacity."
            ),
            (
                "State boundaries may be ambiguous. Use low, medium, or high confidence; for "
                "low or medium confidence, explain the ambiguity. Include an alternative_state "
                "only when another state is genuinely plausible."
            ),
            f"The pseudonymous session_id is {session_id}.",
            f"The clip duration is approximately {duration_ms} ms.",
            f"Images are sampled chronologically about every {image_interval_ms} ms.",
            (
                "Each image includes a frame_time_ms label. Audio timestamps are approximate "
                "offsets from the beginning of the clip in milliseconds."
            ),
            "Return exactly one JSON object. Do not use Markdown or add commentary.",
            "Use English enum values; reason, observations, and limitation reasons may be Chinese.",
            "FORMATIVE-STUDY CODEBOOK:",
            json.dumps(codebook, ensure_ascii=False, indent=2),
            "REQUIRED JSON SCHEMA:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )
