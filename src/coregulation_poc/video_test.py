from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coregulation_poc.acoustics import (
    AcousticAnalysis,
    analyze_replay_audio,
    load_acoustic_analysis_config,
)
from coregulation_poc.capture.video_replay import MediaKind, ReplayMedia, decode_video_for_replay
from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.fusion.prompting import build_state_assessment_prompt
from coregulation_poc.fusion.response_parser import (
    RealtimeResponseAccumulator,
    validate_assessment_context,
)
from coregulation_poc.models import Actor, EvidenceSufficiency, StateAssessment
from coregulation_poc.paths import ACOUSTIC_ANALYSIS_PATH, STATE_CODEBOOK_PATH
from coregulation_poc.providers.qwen_omni_realtime import QwenOmniRealtimeProvider
from coregulation_poc.settings import Settings
from coregulation_poc.storage.run_artifacts import RunArtifactStore, sha256_file


def _metadata_dict(media: ReplayMedia) -> dict[str, Any]:
    metadata = media.metadata
    return {
        "duration_ms": metadata.duration_ms,
        "width": metadata.width,
        "height": metadata.height,
        "frame_rate": metadata.frame_rate,
        "video_codec": metadata.video_codec,
        "audio_codec": metadata.audio_codec,
        "source_audio_sample_rate": metadata.audio_sample_rate,
        "replay_audio_sample_rate": media.audio_sample_rate,
        "audio_chunk_ms": media.audio_chunk_ms,
        "image_interval_ms": media.image_interval_ms,
        "image_timestamp_labels": media.image_timestamp_labels,
        "audio_chunk_count": media.audio_chunk_count,
        "image_count": media.image_count,
    }


def _normalized_quote(value: str) -> str:
    return "".join(value.lower().split()).strip("。！？,.!?")


def _build_audit_result(
    *,
    accumulator: RealtimeResponseAccumulator,
    assessment: StateAssessment | None,
    classification_valid: bool,
    acoustic_analysis: AcousticAnalysis,
) -> dict[str, Any]:
    warnings = list(accumulator.audit_warnings)
    transcription_status = accumulator.transcription_status
    transcript = _normalized_quote(accumulator.best_effort_input_transcript)
    quote_checks: list[dict[str, Any]] = []

    if assessment is not None:
        audio = assessment.modality_evidence.audio
        if (
            audio.sufficiency is EvidenceSufficiency.SUFFICIENT
            and acoustic_analysis.quality == "insufficient"
        ):
            warnings.append("audio_evidence_without_measurable_acoustic_signal")
        if (
            audio.sufficiency is EvidenceSufficiency.SUFFICIENT
            and transcription_status != "completed"
        ):
            warnings.append("audio_evidence_without_completed_input_transcript")
        for item in audio.items:
            quote = item.quote or ""
            matched = bool(transcript and _normalized_quote(quote) in transcript)
            quote_checks.append(
                {
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "quote": quote,
                    "found_in_best_effort_transcript": matched,
                }
            )
            if not matched:
                warnings.append("audio_quote_not_found_in_best_effort_transcript")

    warnings = list(dict.fromkeys(warnings))
    return {
        "classification_valid": classification_valid,
        "audit_ready": classification_valid and not warnings,
        "transcription_status": transcription_status,
        "acoustic_measurement_quality": acoustic_analysis.quality,
        "acoustic_interpretation_role": acoustic_analysis.interpretation_role,
        "audit_warnings": warnings,
        "audio_quote_checks": quote_checks,
    }


async def _replay(
    media: ReplayMedia,
    provider: QwenOmniRealtimeProvider,
    store: RunArtifactStore,
) -> None:
    started = time.monotonic()
    for sequence, chunk in enumerate(media.chunks):
        due = started + chunk.timestamp_ms / 1000
        await asyncio.sleep(max(0.0, due - time.monotonic()))
        if chunk.kind is MediaKind.AUDIO:
            await provider.send_audio(chunk.payload, chunk.timestamp_ms)
        else:
            await provider.send_frame(chunk.payload, chunk.timestamp_ms)
        store.append_event(
            {
                "direction": "client",
                "sequence": sequence,
                "type": f"input.{chunk.kind.value}",
                "clip_timestamp_ms": chunk.timestamp_ms,
                "payload_bytes": len(chunk.payload),
            }
        )


async def run_video_test(
    *,
    video_path: Path,
    session_id: str,
    settings: Settings,
    dry_run: bool,
    progress: Callable[[str], None] = print,
) -> tuple[Path, bool]:
    resolved_video = video_path.expanduser().resolve()
    progress("[1/7] Decoding and validating audio/video...")
    media = decode_video_for_replay(resolved_video)
    progress(
        f"    PASS: {media.metadata.duration_ms / 1000:.1f}s, "
        f"{media.audio_chunk_count} audio chunks, {media.image_count} image frames"
    )
    progress("[2/7] Creating traceable run artifacts...")
    store = RunArtifactStore(settings.output_dir, session_id)
    codebook = load_state_codebook()
    acoustic_config = load_acoustic_analysis_config()
    acoustic_analysis = analyze_replay_audio(media, config=acoustic_config)
    prompt = build_state_assessment_prompt(
        session_id=session_id,
        duration_ms=media.metadata.duration_ms,
        codebook=codebook,
        image_interval_ms=media.image_interval_ms,
        speaker_roles_bound=False,
    )
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "source": {
            "filename": resolved_video.name,
            "sha256": sha256_file(resolved_video),
        },
        "model": None if dry_run else settings.omni_model,
        "mode": "dry_run" if dry_run else "manual_realtime_replay",
        "speaker_role_binding": "not_available",
        "media": _metadata_dict(media),
        "research_basis": {
            "codebook_version": codebook.get("version"),
            "codebook_source": codebook.get("source"),
            "codebook_sha256": sha256_file(STATE_CODEBOOK_PATH),
            "acoustic_analysis_version": acoustic_config.version,
            "acoustic_analysis_source": acoustic_config.source,
            "acoustic_config_sha256": sha256_file(ACOUSTIC_ANALYSIS_PATH),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "software": {
            "application": "coregulation-realtime-poc/0.1.0",
            "av": importlib.metadata.version("av"),
            "praat-parselmouth": importlib.metadata.version("praat-parselmouth"),
            "websocket-client": importlib.metadata.version("websocket-client"),
            "transport": "native_websocket",
        },
    }
    store.write_json("manifest.json", manifest)
    store.write_text("prompt.txt", prompt)
    store.write_json("acoustic_summary.json", acoustic_analysis.model_dump(mode="json"))
    progress(f"    PASS: {store.run_dir}")
    if dry_run:
        store.write_json("result.json", {"status": "ready", "api_called": False})
        progress("[3/7] API configuration: SKIPPED (dry run)")
        progress("[4/7] WebSocket session: SKIPPED (dry run)")
        progress("[5/7] Realtime media replay: SKIPPED (dry run)")
        progress("[6/7] Model response: SKIPPED (dry run)")
        progress("[7/7] Result validation: PASS")
        return store.run_dir, True

    progress("[3/7] Checking API configuration...")
    if settings.dashscope_api_key is None:
        store.write_json("result.json", {"status": "blocked", "error": "missing_api_key"})
        return store.run_dir, False
    if not settings.aliyun_workspace_id or not settings.realtime_base_url:
        store.write_json("result.json", {"status": "blocked", "error": "missing_workspace_id"})
        return store.run_dir, False
    progress("    PASS: API key, workspace and endpoint are configured")

    provider = QwenOmniRealtimeProvider(
        model=settings.omni_model,
        api_key=settings.dashscope_api_key.get_secret_value(),
        workspace_id=settings.aliyun_workspace_id,
        base_url=settings.realtime_base_url,
        instructions=prompt,
        connection_timeout_seconds=settings.connection_timeout_seconds,
    )
    accumulator = RealtimeResponseAccumulator()
    started_ns = time.monotonic_ns()
    first_response_ns: int | None = None
    completed = False
    error_message: str | None = None
    assessment: StateAssessment | None = None

    try:
        progress("[4/7] Opening WebSocket and updating the session...")
        await provider.connect()
        progress("    PASS: session.updated received")
        progress("[5/7] Replaying audio and image frames in real time...")
        await _replay(media, provider, store)
        progress("    PASS: media replay completed")
        request_ns = time.monotonic_ns()
        await provider.finish_input()
        progress("[6/7] Waiting for the model response...")
        async with asyncio.timeout(settings.response_timeout_seconds):
            async for envelope in provider.events():
                event = envelope["event"]
                received_ns = int(envelope["received_monotonic_ns"])
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("type", "unknown"))
                if event_type.startswith("response.") and first_response_ns is None:
                    first_response_ns = received_ns
                store.append_event(
                    {
                        "direction": "server",
                        "received_after_start_ms": (received_ns - started_ns) // 1_000_000,
                        "event": event,
                    }
                )
                accumulator.add(event)
                if event_type == "response.done":
                    completed = True
        assessment = accumulator.parse_assessment()
        assessment = validate_assessment_context(
            assessment,
            expected_session_id=session_id,
            duration_ms=media.metadata.duration_ms,
            codebook=codebook,
            history_available=False,
        )
        store.write_json("assessment.json", assessment.model_dump(mode="json"))
        acoustic_evidence = []
        for item in assessment.modality_evidence.audio.items:
            measurement = analyze_replay_audio(
                media,
                config=acoustic_config,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                actor=Actor.UNKNOWN,
                speaker_roles_bound=False,
            )
            acoustic_evidence.append(
                {
                    "evidence_code": item.code,
                    "verbatim_quote": item.quote,
                    "model_observed_actor": item.actor,
                    "measurement_actor": Actor.UNKNOWN,
                    "measurement": measurement.model_dump(mode="json"),
                }
            )
        store.write_json("acoustic_evidence.json", acoustic_evidence)
        progress(f"    PASS: response.done received; state={assessment.state}")
        progress("[7/7] Validating and saving the structured assessment: PASS")
        valid = True
    except (TimeoutError, ValueError, OSError, ConnectionError) as exc:
        error_message = str(exc)
        progress(f"    FAIL: {error_message}")
        valid = False
        request_ns = locals().get("request_ns", time.monotonic_ns())
    finally:
        await provider.close()

    finished_ns = time.monotonic_ns()
    store.write_text("model_response.txt", accumulator.response_text)
    store.write_text("input_transcript_best_effort.txt", accumulator.best_effort_input_transcript)
    store.write_json("transcription_events.json", accumulator.transcript_events)
    store.write_json("input_emotions.json", accumulator.input_emotion_observations)
    audit_result = _build_audit_result(
        accumulator=accumulator,
        assessment=assessment,
        classification_valid=valid,
        acoustic_analysis=acoustic_analysis,
    )
    store.write_json("audit.json", audit_result)
    store.write_json(
        "metrics.json",
        {
            "completed": completed,
            "valid_assessment": valid,
            "classification_valid": valid,
            "audit_ready": audit_result["audit_ready"],
            "transcription_status": audit_result["transcription_status"],
            "acoustic_measurement_quality": audit_result["acoustic_measurement_quality"],
            "audit_warnings": audit_result["audit_warnings"],
            "error": error_message,
            "total_wall_ms": (finished_ns - started_ns) // 1_000_000,
            "response_wait_ms": (finished_ns - request_ns) // 1_000_000,
            "first_response_latency_ms": (
                None if first_response_ns is None else (first_response_ns - request_ns) // 1_000_000
            ),
        },
    )
    return store.run_dir, valid
