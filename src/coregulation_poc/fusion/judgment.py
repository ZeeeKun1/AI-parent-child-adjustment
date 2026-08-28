"""Stage-2 judgment prompt construction and response parsing.

The judgment stage takes the structured :class:`PerceptionReport` from
stage 1, locally computed :class:`AcousticFeatures`, the formative-study
codebook, and assessment history, then asks a text model to classify the
parent-child coregulation state and produce a :class:`StateAssessment`.

This separation ensures the multimodal model only perceives (observes and
transcribes), while a text model handles classification based on structured
inputs, avoiding the "forcing" problem where one model must both perceive
and judge simultaneously.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from coregulation_poc.fusion.response_parser import normalize_assessment_payload
from coregulation_poc.models import (
    AcousticFeatures,
    PerceptionReport,
    StateAssessment,
)


def _format_perception_report(report: PerceptionReport) -> str:
    """Render a PerceptionReport as compact text for the judgment model."""
    lines: list[str] = ["=== PERCEPTION REPORT ==="]

    if report.speech_turns:
        lines.append("\nSPEECH TURNS:")
        for i, turn in enumerate(report.speech_turns, 1):
            lines.append(
                f'  {i}. [{turn.start_ms}-{turn.end_ms} ms] {turn.speaker.value}: "{turn.text}"'
            )
    else:
        lines.append("\nSPEECH TURNS: (none transcribed)")

    if report.visual_observations:
        lines.append("\nVISUAL OBSERVATIONS:")
        for i, obs in enumerate(report.visual_observations, 1):
            lines.append(f"  {i}. [{obs.timestamp_ms} ms] {obs.actor.value}: {obs.description}")
    else:
        lines.append("\nVISUAL OBSERVATIONS: (none observed)")

    if report.audio_limitation:
        lines.append(f"\nAUDIO LIMITATION: {report.audio_limitation}")
    if report.video_limitation:
        lines.append(f"\nVIDEO LIMITATION: {report.video_limitation}")

    return "\n".join(lines)


def _format_acoustic_features(features: AcousticFeatures) -> str:
    """Render AcousticFeatures as compact text for the judgment model."""
    lines: list[str] = ["=== ACOUSTIC FEATURES ==="]

    if features.segments:
        lines.append("\nVOICED SEGMENTS:")
        for i, seg in enumerate(features.segments, 1):
            f0_str = (
                f"mean_F0={seg.mean_f0_hz:.0f} Hz, median_F0={seg.median_f0_hz:.0f} Hz"
                if seg.mean_f0_hz
                else "F0=N/A"
            )
            text_str = f', text="{seg.text}"' if seg.text else ""
            lines.append(
                f"  {i}. [{seg.start_ms}-{seg.end_ms} ms] "
                f"{seg.speaker}: {f0_str}, RMS={seg.rms_energy:.3f}{text_str}"
            )
    else:
        lines.append("\nVOICED SEGMENTS: (none detected)")

    if features.silence_gaps:
        lines.append("\nSILENCE GAPS (>= 500 ms):")
        for i, gap in enumerate(features.silence_gaps, 1):
            lines.append(f"  {i}. [{gap.start_ms}-{gap.end_ms} ms] duration={gap.duration_ms} ms")
    else:
        lines.append("\nSILENCE GAPS: (none >= 500 ms)")

    lines.append(
        f"\nSUMMARY: total_speech={features.total_speech_ms} ms, "
        f"total_silence={features.total_silence_ms} ms"
    )

    return "\n".join(lines)


def build_judgment_system_prompt(
    *,
    codebook: dict[str, Any],
) -> str:
    """Build the system prompt containing the classification framework.

    The system prompt establishes the model's role and provides the codebook
    and classification rules. It does not contain assessment-specific data.
    """
    schema = StateAssessment.model_json_schema()

    return "\n".join(
        [
            (
                "You are an expert analyst classifying parent-child homework "
                "interaction states based on structured observation data."
            ),
            (
                "You receive a perception report (speech turns and visual "
                "observations) and acoustic features (F0, RMS energy, silence "
                "gaps) extracted from a 10-second observation window. Your task "
                "is to classify the dyadic coregulation state using the supplied "
                "formative-study codebook."
            ),
            (
                "Classify the interaction sequence as a whole, not a single word, "
                "gesture, gaze direction, silence, compliance response, prompt count, "
                "or acoustic value. A potentially hostile transcribed word is weak "
                "evidence when its speaker, context, or recognition is ambiguous. Never "
                "treat a raw acoustic feature as an emotion or state label."
            ),
            (
                "ASSESS THE INTERACTION TRAJECTORY, NOT ISOLATED EVENTS. An "
                "isolated wrong answer followed by smooth supportive correction "
                "can remain normal. Use fluctuation only when the sequence shows "
                "temporarily uneven coordination while the dyad still retains "
                "observable recovery capacity. A child continuing to answer, look "
                "at the task, or comply does not by itself prove recovery when "
                "repeated prompting, pace conflict, or passive withdrawal persists."
            ),
            (
                "Do NOT classify high_risk from one isolated turn; it requires a "
                "persistent pattern across the observation window."
            ),
            (
                "EVIDENCE RULES:\n"
                "1. For audio evidence: set modality=audio, use the verbatim quote "
                "from the speech turn, copy that quote into observation when no "
                "separate acoustic observation is needed, and set "
                "frame_timestamp_ms to null. Both quote and observation must be "
                "non-empty strings.\n"
                "2. For video evidence: set modality=video, set frame_timestamp_ms "
                "to the observation's timestamp, set quote to null.\n"
                "3. frame_timestamp_ms must fall within [start_ms, end_ms] for "
                "video evidence.\n"
                "4. Audio evidence requires a non-empty quote. Video evidence "
                "requires a non-empty observation.\n"
                "5. The code field should be a short descriptive label (e.g., "
                "'parental guidance', 'child difficulty', 'task progress')."
            ),
            (
                "EVIDENCE SUFFICIENCY:\n"
                "1. Audio and video modalities are evaluated independently.\n"
                "2. One sufficient modality can support an overall sufficient "
                "assessment.\n"
                "3. If a modality has no evidence, mark it insufficient and "
                "explain the limitation.\n"
                "4. If overall evidence is insufficient, set state to null and "
                "confidence to low."
            ),
            (
                "CONFIDENCE:\n"
                "1. Use high confidence only when the state is clearly supported "
                "by multiple evidence items.\n"
                "2. Use medium or low confidence when the state is ambiguous.\n"
                "3. Low or medium confidence requires an ambiguity_reason.\n"
                "4. Include an alternative_state only when another state is "
                "genuinely plausible."
            ),
            (
                "interaction_performance values must come from the codebook "
                "definition of the selected state. Do not invent values."
            ),
            (
                "TRAJECTORY, TASK PROCESS, SUPPORT NEED, SUPPORT TARGET, "
                "AND INTERRUPTIBILITY:\n"
                "You must assess the following fields for every sufficient "
                "assessment. They inform, but do not by themselves determine, "
                "the intervention decision.\n"
                "\n"
                "1. trajectory (compare with the previous window):\n"
                "   - stable: state unchanged and pattern consistent\n"
                "   - worsening: state deteriorating or pattern degrading\n"
                "   - recovering: state improving toward normal\n"
                "   - unclear: insufficient comparison basis\n"
                "\n"
                "2. task_process (how the task is progressing, observable only):\n"
                "   - smooth_progress: steady advancement\n"
                "   - brief_stall: temporary difficulty resolving within the "
                "window\n"
                "   - sustained_stall: task not progressing for most of the "
                "window\n"
                "   - pace_mismatch: parent and child operating at incompatible "
                "speeds\n"
                "   - explanation_mismatch: parent's explanation does not match "
                "child's understanding\n"
                "   - over_assistance: parent providing excessive help that "
                "reduces child autonomy\n"
                "   - disengaged: child withdrawn from the task\n"
                "   - completion: task finished\n"
                "   - unclear: cannot determine from available evidence\n"
                "\n"
                "3. support_need (what type of support would help the dyad):\n"
                "   - none: no support needed\n"
                "   - positive_reinforcement: a positive behavior worth "
                "reinforcing (only in normal state)\n"
                "   - emotional_support: emotional regulation needed\n"
                "   - need_expression: help expressing needs\n"
                "   - mutual_understanding: bridge a parent-child "
                "misunderstanding\n"
                "   - task_pacing: adjust task pace\n"
                "   - learning_support: help with learning content\n"
                "   - autonomy_support: support child's independent effort\n"
                "   - unclear: support need not identifiable\n"
                "\n"
                "4. support_target (who needs the support):\n"
                "   - parent: parent needs guidance\n"
                "   - child: child needs support\n"
                "   - both: both need intervention\n"
                "   - unknown: target not identifiable from evidence\n"
                "\n"
                "5. interruptibility (whether it is semantically appropriate "
                "to interrupt right now):\n"
                "   - natural_pause: a turn just ended and no one is "
                "expressing critical content\n"
                "   - active_speech: the parent or child is currently speaking\n"
                "   - task_engaged: the child is reading aloud, answering, "
                "writing, or concentrating\n"
                "   - unclear: cannot determine\n"
                "\n"
                "If evidence is insufficient, set task_process and support_need "
                "to null. When evidence is sufficient but a specific field "
                "cannot be determined, use the unclear/unknown value for that "
                "field."
            ),
            (
                "DATA-DERIVED BOUNDARY SIGNALS:\n"
                "Return boundary_signals for every window using only directly "
                "observable current-window evidence. These fields are consumed by "
                "a deterministic cross-window boundary tracker.\n"
                "1. task_stall_observed: true only when the task makes no meaningful "
                "progress during the window; false when progress is directly "
                "observed; null when task progress cannot be determined.\n"
                "2. parental_prompt_count: count only direct parental urging, repeated "
                "commands, or repeated questions used to push the child forward. Do "
                "not count neutral explanation, praise, or one supportive question. "
                "Use null when the parent cannot be identified or speech evidence is "
                "insufficient.\n"
                "3. conflict_action_observed: true only for a directly observed "
                "physical conflict action or contextually unambiguous repeated hostile "
                "language. One possibly mistranscribed word is not sufficient. Use false "
                "only when the relevant behavior is observable throughout; otherwise null.\n"
                "4. child_disengaged_observed: true only when the child is directly "
                "observed withdrawing participation through explicit refusal, prolonged "
                "non-response, leaving the task, or sustained passive withdrawal. Do not "
                "equate ordinary thinking silence, looking down, touching the face, or one "
                "slow answer with disengagement.\n"
                "5. regulation_balance: both_stable, one_stable, both_crossed, or "
                "unclear. one_stable means one participant still maintains a rational, "
                "task-oriented response while the other deviates. both_crossed means "
                "both participants show observable loss of coordination."
            ),
            (
                "JUDGMENT BOUNDARIES:\n"
                "1. Do not classify the state from a single pitch, volume, or "
                "facial expression; integrate multiple cues across the window.\n"
                "2. Do not infer unexpressed psychological diagnoses.\n"
                "3. Assess task_process only by how the task advances, not by "
                "answer correctness.\n"
                "4. When the task is not visible or cannot be followed, use "
                "unclear for task_process.\n"
                "5. The support_target must be supported by observable evidence; "
                "do not guess.\n"
                "6. Do not output intervention actions or intervention scripts; "
                "assess only the observable state and its attributes.\n"
                "7. Use the codebook's operational_boundary as a trajectory-level "
                "anchor. Ten to 30 seconds of observed coordination disruption supports "
                "fluctuation. Dysregulation normally requires at least three disrupted "
                "windows in the supplied 60-second trajectory, including two consecutive "
                "windows, and evidence from at least two dimensions. A single prompt "
                "rate, gaze pattern, silence, compliance response, or transcribed word is "
                "not enough. Immediate dysregulation requires marked current evidence "
                "that is independently corroborated across actors, modalities, or repeated "
                "turns. One stable window is provisional fluctuation; 20 seconds of "
                "effective balanced progress confirms normal recovery. The runtime "
                "performs final cross-window calculations, so never invent missing "
                "duration, counts, or corroboration."
            ),
            "Return exactly one JSON object. Do not use Markdown or add commentary.",
            (
                "Use English enum values for state, confidence, evidence_sufficiency, "
                "and actor. Reason, observations, and limitation reasons may be Chinese."
            ),
            "FORMATIVE-STUDY CODEBOOK:",
            json.dumps(codebook, ensure_ascii=False, indent=2),
            "FEW-SHOT CLASSIFICATION EXAMPLES:",
            """
            The following examples are adapted from expert-coded episodes in the
            formative study. They are boundary anchors, not keyword-matching rules.
            Classify the coordination pattern and trajectory rather than matching
            isolated words, gestures, or acoustic values.

            EXAMPLE 1 — NORMAL

            Observed sequence:
            - The child independently chooses a reading passage and reads it fluently.
            - The child asks, "为什么要欢迎台湾的小朋友？"
            - The parent answers the question, waits, and the child continues reading.

            Expected judgment:
            - state: normal
            - task_process: smooth_progress
            - support_need: positive_reinforcement
            - support_target: both
            - boundary_signals: task_stall_observed=false,
              parental_prompt_count=0, conflict_action_observed=false,
              child_disengaged_observed=false, regulation_balance=both_stable

            Reason:
            The child participates actively, the parent responds supportively, and
            the task continues smoothly. A question or temporary uncertainty alone
            does not constitute fluctuation.


            EXAMPLE 2 — FLUCTUATION

            Observed sequence:
            - The child is unsure how to solve a word problem.
            - The parent explains the problem step by step but speaks quickly and
            provides a large amount of information at once.
            - The child remains engaged and continues trying to follow the explanation.
            - The task has slowed down but has not broken down.

            Expected judgment:
            - state: fluctuation
            - task_process: pace_mismatch
            - support_need: task_pacing
            - support_target: parent
            - boundary_signals: task_stall_observed=true,
              parental_prompt_count=0, conflict_action_observed=false,
              child_disengaged_observed=false, regulation_balance=one_stable

            Reason:
            Coordination is temporarily uneven, but task engagement and the capacity
            for within-dyad recovery remain. Do not classify this as dysregulation
            unless the mismatch becomes repeated and unresolved, accompanied by
            withdrawal, conflict, or sustained task stoppage.


            EXAMPLE 3 — DYSREGULATION

            Observed sequence:
            - Recent windows show repeated English-reading errors followed by the
            parent's commands to restart from the beginning.
            - The child proposes reviewing one item, but the parent overrides the
            proposal and repeats the restart instruction.
            - The child still complies at first, then repeatedly stops responding,
            turns away from the task, or explicitly says that they do not want to continue.
            - The prompting-and-withdrawal pattern repeatedly returns without a
            sustained period of balanced task progress.

            Expected judgment:
            - state: dysregulation
            - task_process: pace_mismatch
            - support_need: task_pacing
            - support_target: parent
            - boundary_signals: task_stall_observed=true,
              parental_prompt_count=2, conflict_action_observed=false,
              child_disengaged_observed=true, regulation_balance=one_stable

            Reason:
            The recent trajectory shows a repeated pace conflict and increasingly
            passive participation without confirmed recovery. Continued compliance
            does not make the interaction normal. Explicit physical conflict or total
            disengagement is not required for dysregulation. Do not automatically
            classify it as high_risk because this episode alone does not establish a
            persistent controlling-dependent pattern.


            EXAMPLE 4 — HIGH_RISK

            Observed sequence and history:
            - Across several consecutive prior windows, the parent has supplied nearly
            every step, frequently interrupted the child, and repeatedly corrected
            or completed parts of the task for the child.
            - The child increasingly waits for instructions and rarely initiates or
            explains a solution independently.
            - Earlier support attempts have not changed this interaction pattern.
            - In the current window, the parent again provides the full procedure while
            the child follows passively.

            Expected judgment:
            - state: high_risk
            - task_process: over_assistance
            - support_need: autonomy_support
            - support_target: parent
            - boundary_signals: task_stall_observed=true,
              parental_prompt_count=null, conflict_action_observed=false,
              child_disengaged_observed=false, regulation_balance=one_stable

            Reason:
            Current evidence and interaction history jointly demonstrate a persistent
            imbalance that restricts the child's autonomy and has not recovered.
            One isolated correction, interruption, or instance of parental help is
            not sufficient for high_risk.


            BOUNDARY REMINDERS:
            - A wrong answer or brief difficulty can remain normal when supportive
            coordination and task progress are maintained.
            - Fluctuation means temporary unevenness with observable recovery capacity.
            - Dysregulation means an unresolved or repeatedly returning negative
              interaction cycle across the current window and recent trajectory.
            - High_risk requires a persistent pattern supported by current evidence
            and prior-window history.
            - Acoustic features may corroborate a trajectory, but pitch, RMS energy,
            or silence alone must never determine the state.
            - Never invent behavior, intention, emotion, or history that is absent
            from the supplied evidence.
            """,
            "REQUIRED JSON SCHEMA:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def build_judgment_user_prompt(
    *,
    session_id: str,
    window_start_ms: int,
    window_end_ms: int,
    previous_state: str | None,
    history_summary: list[dict[str, Any]],
    perception_report: PerceptionReport,
    acoustic_features: AcousticFeatures,
    task_context: dict[str, Any] | None = None,
) -> str:
    """Build the user prompt containing assessment-specific data.

    Parameters
    ----------
    session_id
        Pseudonymous session identifier.
    window_start_ms, window_end_ms
        The session-time range of the current assessment window.
    previous_state
        The previous assessment's state value (or None), to be copied exactly.
    history_summary
        Compact list of prior assessments for trajectory context.
    perception_report
        The stage-1 perception output.
    acoustic_features
        Locally computed acoustic features.
    task_context
        Optional session-level task information (name, difficulty, grade).
    """
    return "\n".join(
        [
            f"Session ID: {session_id}",
            f"Observation window: {window_start_ms}-{window_end_ms} ms",
            f"Set assessed_at_ms exactly to {window_end_ms}.",
            (
                f"The runtime-provided previous_state is {previous_state}; "
                "copy this value exactly, including null."
            ),
            (
                "Recent trajectory history (up to the preceding 60 seconds; "
                "current-window evidence is still required). Treat one normal "
                "window as provisional recovery rather than proof that a recurring "
                "pattern ended. Each entry includes state, "
                "confidence, interaction_performance, task_process, "
                "support_need, trajectory, and boundary_signals: "
                + json.dumps(history_summary, ensure_ascii=False)
            ),
            (
                "Task context: " + json.dumps(task_context, ensure_ascii=False)
                if task_context
                else "Task context: (not provided)"
            ),
            "",
            _format_perception_report(perception_report),
            "",
            _format_acoustic_features(acoustic_features),
            "",
            (
                "Based on the perception report and acoustic features above, "
                "classify the coregulation state following the codebook. Select "
                "the most relevant speech turns and visual observations as "
                "evidence. Write a clear reason explaining your classification."
            ),
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
    raise ValueError("The judgment response did not contain a JSON object.")


def parse_judgment_result(response_text: str) -> StateAssessment:
    """Parse the text model's JSON response into a :class:`StateAssessment`.

    Parameters
    ----------
    response_text
        The text response from the judgment model.
    """
    payload = _find_json_object(response_text)
    # Qwen can represent an audio item's redundant human-readable observation
    # as null while still returning the required verbatim quote. Reuse that
    # quote instead of rejecting otherwise traceable evidence; this does not
    # invent or reinterpret any observation.
    modality_evidence = payload.get("modality_evidence")
    if isinstance(modality_evidence, dict):
        audio = modality_evidence.get("audio")
        if isinstance(audio, dict):
            items = audio.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    observation = item.get("observation")
                    quote = item.get("quote")
                    if observation is None and isinstance(quote, str) and quote.strip():
                        item["observation"] = quote.strip()
    try:
        return StateAssessment.model_validate(normalize_assessment_payload(payload))
    except ValidationError as exc:
        raise ValueError(f"Judgment JSON did not match StateAssessment: {exc}") from exc
