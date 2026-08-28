"""End-to-end pipeline test: feed one video clip through all four modules."""

from __future__ import annotations

import asyncio
import importlib.metadata
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coregulation_poc.capture.media import MediaChunk, StrictTimestampNormalizer
from coregulation_poc.capture.video_replay import ReplayMedia, decode_video_for_replay
from coregulation_poc.codebook import load_state_codebook
from coregulation_poc.control import load_intervention_policy
from coregulation_poc.delivery import load_delivery_policy
from coregulation_poc.intervention import load_strategy_library
from coregulation_poc.paths import (
    DELIVERY_POLICY_PATH,
    INTERVENTION_POLICY_PATH,
    STATE_CODEBOOK_PATH,
    STRATEGY_CARDS_PATH,
)
from coregulation_poc.providers.qwen_text_chat import QwenTextChatProvider
from coregulation_poc.runtime.factory import QwenVoiceSynthesizer
from coregulation_poc.runtime.recognition import QwenWindowRecognizer
from coregulation_poc.runtime.session import RealtimeLoopConfig, RealtimeSession
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
        "audio_chunk_count": media.audio_chunk_count,
        "image_count": media.image_count,
    }


def _extract_typed_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group raw session events by type for structured artifact output."""
    grouped: dict[str, list[dict[str, Any]]] = {
        "state_updates": [],
        "interventions": [],
        "intervention_holds": [],
        "delivery_executions": [],
        "intervention_outcomes": [],
        "loop_errors": [],
    }
    for event in events:
        event_type = event.get("type")
        if event_type == "state_update":
            grouped["state_updates"].append(event)
        elif event_type == "intervention":
            grouped["interventions"].append(event)
        elif event_type == "intervention_held":
            grouped["intervention_holds"].append(event)
        elif event_type == "delivery_execution_received":
            grouped["delivery_executions"].append(event)
        elif event_type == "intervention_outcome":
            grouped["intervention_outcomes"].append(event)
        elif event_type == "loop_error":
            grouped["loop_errors"].append(event)
    return grouped


async def run_pipeline_test(
    *,
    video_path: Path,
    session_id: str,
    settings: Settings,
    dry_run: bool = False,
    voice_enabled: bool = False,
    window_seconds: float = 12.0,
    assessment_interval_seconds: float = 12.0,
    max_assessments: int = 0,
    auto_acknowledge: bool = True,
    progress: Callable[[str], None] = print,
) -> tuple[Path, bool]:
    """Replay one video clip through all four modules end-to-end.

    Module 1 (state recognition) calls Qwen Omni Realtime on each rolling window.
    Module 2 (timing control) decides whether to intervene based on the state trajectory.
    Module 3 (strategy selection) picks a matching strategy card.
    Module 4 (dual-channel delivery) prepares visual text and optional Maia voice.

    When *auto_acknowledge* is true, each intervention is automatically marked as
    delivered so the post-intervention observation window can trigger on the
    remaining video content.
    """
    resolved_video = video_path.expanduser().resolve()

    # -- Step 1: decode video ------------------------------------------------
    progress("[1/6] Decoding and validating audio/video...")
    media = decode_video_for_replay(resolved_video)
    progress(
        f"    PASS: {media.metadata.duration_ms / 1000:.1f}s, "
        f"{media.audio_chunk_count} audio chunks, {media.image_count} image frames"
    )

    # -- Step 2: create artifact store and manifest --------------------------
    progress("[2/6] Creating traceable run artifacts...")
    store = RunArtifactStore(settings.output_dir, session_id)
    codebook = load_state_codebook()
    intervention_policy = load_intervention_policy()
    strategy_library = load_strategy_library()
    delivery_policy = load_delivery_policy()

    loop_config = RealtimeLoopConfig(
        window_duration_ms=round(window_seconds * 1000),
        assessment_interval_ms=round(assessment_interval_seconds * 1000),
        max_assessments_per_session=max_assessments,
        voice_enabled=voice_enabled,
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "mode": "pipeline_dry_run" if dry_run else "pipeline_full_replay",
        "source": {
            "filename": resolved_video.name,
            "sha256": sha256_file(resolved_video),
        },
        "media": _metadata_dict(media),
        "loop_config": {
            "window_duration_ms": loop_config.window_duration_ms,
            "assessment_interval_ms": loop_config.assessment_interval_ms,
            "post_intervention_observation_ms": loop_config.post_intervention_observation_ms,
            "max_assessments_per_session": loop_config.max_assessments_per_session,
            "history_assessments": loop_config.history_assessments,
            "max_parallel_perception": loop_config.max_parallel_perception,
            "max_parallel_judgment": loop_config.max_parallel_judgment,
            "max_intervention_staleness_ms": (
                loop_config.max_intervention_staleness_ms
            ),
            "voice_enabled": loop_config.voice_enabled,
        },
        "research_basis": {
            "codebook_version": codebook.get("version"),
            "codebook_sha256": sha256_file(STATE_CODEBOOK_PATH),
            "intervention_policy_version": intervention_policy.version,
            "intervention_policy_sha256": sha256_file(INTERVENTION_POLICY_PATH),
            "strategy_library_version": strategy_library.version,
            "strategy_library_sha256": sha256_file(STRATEGY_CARDS_PATH),
            "delivery_policy_version": delivery_policy.version,
            "delivery_policy_sha256": sha256_file(DELIVERY_POLICY_PATH),
        },
        "speaker_role_binding": "f0_clustering_available",
        "software": {
            "application": "coregulation-realtime-poc/0.1.0",
            "av": importlib.metadata.version("av"),
            "websocket-client": importlib.metadata.version("websocket-client"),
        },
    }
    store.write_json("manifest.json", manifest)
    store.write_json("codebook.json", codebook)
    store.write_json("intervention_policy.json", intervention_policy.model_dump(mode="json"))
    store.write_json("strategy_library.json", strategy_library.model_dump(mode="json"))
    store.write_json("delivery_policy.json", delivery_policy.model_dump(mode="json"))
    progress(f"    PASS: {store.run_dir}")

    if dry_run:
        store.write_json(
            "result.json",
            {
                "status": "ready",
                "api_called": False,
                "mode": "pipeline_dry_run",
                "video_duration_ms": media.metadata.duration_ms,
                "estimated_assessments": max(
                    1, media.metadata.duration_ms // loop_config.assessment_interval_ms
                ),
            },
        )
        progress("[3/6] API configuration: SKIPPED (dry run)")
        progress("[4/6] Session creation: SKIPPED (dry run)")
        progress("[5/6] Pipeline replay: SKIPPED (dry run)")
        progress("[6/6] Result validation: PASS (dry run)")
        return store.run_dir, True

    # -- Step 3: check API configuration -------------------------------------
    progress("[3/6] Checking API configuration...")
    if settings.dashscope_api_key is None:
        store.write_json("result.json", {"status": "blocked", "error": "missing_api_key"})
        progress("    FAIL: DASHSCOPE_API_KEY is not configured")
        return store.run_dir, False
    if not settings.aliyun_workspace_id or not settings.realtime_base_url:
        store.write_json("result.json", {"status": "blocked", "error": "missing_workspace_id"})
        progress("    FAIL: ALIYUN_WORKSPACE_ID is not configured")
        return store.run_dir, False
    progress("    PASS: API key, workspace and endpoint are configured")

    # -- Step 4: create realtime session -------------------------------------
    progress("[4/6] Creating realtime session with all four modules...")
    recognizer = QwenWindowRecognizer(
        settings=settings,
        image_interval_ms=media.image_interval_ms,
    )
    voice_synthesizer: QwenVoiceSynthesizer | None = None
    if voice_enabled:
        voice_synthesizer = QwenVoiceSynthesizer(settings)
    text_chat_provider = QwenTextChatProvider(
        api_key=settings.dashscope_api_key.get_secret_value(),
        model=settings.text_chat_model,
        temperature=settings.text_chat_temperature,
        max_tokens=settings.text_chat_max_tokens,
        timeout_seconds=settings.text_chat_timeout_seconds,
    )
    progress(f"    Module 1: {type(recognizer).__name__}")
    progress("    Module 2: StateTrajectoryController")
    progress("    Module 3: StrategySelector")
    progress("    Module 4: DeliveryCoordinator")
    progress(f"    Voice: {'enabled (Maia)' if voice_enabled else 'disabled (visual-only)'}")

    events: list[dict[str, Any]] = []
    delivery_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    media_time_ms: list[int] = [0]  # mutable holder for current media timestamp

    async def send_event(event: dict[str, Any]) -> None:
        events.append(event)
        store.append_event(event)
        if event.get("type") == "intervention" and auto_acknowledge:
            await delivery_queue.put(event)

    session = RealtimeSession(
        session_id=session_id,
        recognizer=recognizer,
        send_event=send_event,
        config=loop_config,
        voice_synthesizer=voice_synthesizer,
        text_chat_provider=text_chat_provider,
    )

    # -- Step 5: replay video through the pipeline ---------------------------
    progress("[5/6] Replaying video through the full pipeline...")

    async def auto_acknowledger() -> None:
        """Simulate browser-side delivery confirmation for each intervention."""
        while True:
            event = await delivery_queue.get()
            if event is None:
                return
            delivery_id = event["delivery_id"]
            voice_expected = bool(event.get("voice_expected", False))
            voice_error = event.get("voice_error")
            # Brief delay to simulate display and user reading time
            await asyncio.sleep(0.5)
            now_ms = media_time_ms[0]
            started_ms = max(0, now_ms - 300)
            visual: dict[str, Any] = {
                "status": "delivered",
                "started_at_ms": started_ms,
                "completed_at_ms": now_ms,
                "provider": "pipeline_test_auto",
            }
            if voice_expected and not voice_error:
                voice: dict[str, Any] = {
                    "status": "delivered",
                    "started_at_ms": started_ms,
                    "completed_at_ms": now_ms,
                    "provider": event.get("voice_provider", "qwen_tts_realtime"),
                }
            else:
                voice = {"status": "not_attempted"}
            control = {
                "type": "delivery_execution",
                "delivery_id": delivery_id,
                "recorded_at_ms": now_ms,
                "visual": visual,
                "voice": voice,
                "user_acknowledged": True,
            }
            try:
                await session.handle_control(control)
            except (ValueError, RuntimeError) as exc:
                store.append_event(
                    {
                        "type": "auto_acknowledge_error",
                        "delivery_id": delivery_id,
                        "error": str(exc),
                    }
                )

    ack_task: asyncio.Task[None] | None = None
    if auto_acknowledge:
        ack_task = asyncio.create_task(auto_acknowledger())

    started_wall = time.monotonic()
    valid = True
    error_message: str | None = None

    try:
        await session.start()
        normalizer = StrictTimestampNormalizer()
        for sequence, chunk in enumerate(media.chunks):
            due = started_wall + chunk.timestamp_ms / 1000
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            normalized_ts = normalizer.normalize(chunk.timestamp_ms)
            media_time_ms[0] = normalized_ts
            normalized_chunk = MediaChunk(chunk.kind, normalized_ts, chunk.payload)
            await session.accept_chunk(normalized_chunk)
            if sequence % 50 == 0:
                progress(
                    f"    fed {sequence + 1}/{len(media.chunks)} chunks "
                    f"({chunk.timestamp_ms / 1000:.1f}s), "
                    f"assessments={session.assessment_count}"
                )

        # Every due window is retained even while perception is busy.  Drain
        # the ordered analysis queue before finalizing the replay artifacts.
        if session.runtime_metrics["scheduled_assessment_count"] > session.assessment_count:
            progress("    waiting for final analysis to complete...")
            await session.wait_for_analysis()

        # Force a final assessment if none was triggered (short clip)
        if session.assessment_count == 0:
            progress("    forcing a final assessment (clip shorter than interval)...")
            with suppress(ValueError):
                await session.analyze_now()

        # Allow auto-acknowledger to finish pending deliveries
        if ack_task is not None:
            await delivery_queue.put(None)
            await ack_task

        await session.stop("completed")
    except (TimeoutError, ConnectionError, OSError, ValueError, RuntimeError) as exc:
        error_message = str(exc)
        progress(f"    FAIL: {error_message}")
        valid = False
        if ack_task is not None and not ack_task.done():
            await delivery_queue.put(None)
            with suppress(asyncio.CancelledError):
                await ack_task
        with suppress(RuntimeError, ConnectionError):
            await session.stop("error")

    # -- Step 6: save structured results -------------------------------------
    progress("[6/6] Saving structured results...")
    grouped = _extract_typed_events(events)
    metrics = session.runtime_metrics
    finished_wall = time.monotonic()
    if metrics["assessment_count"] == 0:
        valid = False
        error_message = error_message or "no_valid_state_assessments"

    store.write_json("events_all.json", events)
    store.write_json("state_updates.json", grouped["state_updates"])
    store.write_json("interventions.json", grouped["interventions"])
    store.write_json("intervention_holds.json", grouped["intervention_holds"])
    store.write_json("delivery_executions.json", grouped["delivery_executions"])
    store.write_json("intervention_outcomes.json", grouped["intervention_outcomes"])
    store.write_json("loop_errors.json", grouped["loop_errors"])

    # Save full assessment details (evidence, reason, interaction_performance)
    # to enable inspection of whether the model does genuine analysis or
    # surface-level codebook keyword matching.
    full_assessments = [
        item.model_dump(mode="json") for item in session.assessment_history
    ]
    store.write_json("assessments_full.json", full_assessments)

    # Collect speaker binding summary from the recognizer
    speaker_binding_summary: dict[str, Any] = {"bound": False}
    recognizer_binding = getattr(recognizer, "last_speaker_binding", None)
    if recognizer_binding is not None:
        speaker_binding_summary = {
            "bound": recognizer_binding.bound,
            "method": recognizer_binding.method,
            "parent_mean_f0_hz": recognizer_binding.parent_mean_f0_hz,
            "child_mean_f0_hz": recognizer_binding.child_mean_f0_hz,
            "parent_median_f0_hz": recognizer_binding.parent_median_f0_hz,
            "child_median_f0_hz": recognizer_binding.child_median_f0_hz,
            "parent_segment_count": recognizer_binding.parent_segment_count,
            "child_segment_count": recognizer_binding.child_segment_count,
            "separation_hz": recognizer_binding.separation_hz,
            "limitation_reason": recognizer_binding.limitation_reason,
            "segment_count": len(recognizer_binding.segments),
        }
        if recognizer_binding.segments:
            speaker_binding_summary["segments"] = [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "speaker": seg.speaker.value,
                    "mean_f0_hz": seg.mean_f0_hz,
                    "median_f0_hz": seg.median_f0_hz,
                    "voiced_frame_count": seg.voiced_frame_count,
                }
                for seg in recognizer_binding.segments
            ]
    store.write_json("speaker_binding.json", speaker_binding_summary)

    # Save last perception report and acoustic features for debugging
    last_perception = getattr(recognizer, "last_perception_report", None)
    if last_perception is not None:
        store.write_json("perception_report_last.json", last_perception.model_dump(mode="json"))
    last_acoustic = getattr(recognizer, "last_acoustic_features", None)
    if last_acoustic is not None:
        store.write_json("acoustic_features_last.json", last_acoustic.model_dump(mode="json"))

    result = {
        "status": "completed" if valid else "error",
        "valid": valid,
        "error": error_message,
        "mode": "pipeline_full_replay",
        "pipeline_mode": "two_stage_perception_judgment",
        "session_id": session_id,
        "video_duration_ms": media.metadata.duration_ms,
        "total_wall_ms": round((finished_wall - started_wall) * 1000),
        "assessment_count": metrics["assessment_count"],
        "api_call_count": metrics["api_call_count"],
        "state_update_count": len(grouped["state_updates"]),
        "intervention_count": len(grouped["interventions"]),
        "intervention_held_count": len(grouped["intervention_holds"]),
        "delivery_execution_count": len(grouped["delivery_executions"]),
        "intervention_outcome_count": len(grouped["intervention_outcomes"]),
        "loop_error_count": metrics["analysis_error_count"],
        "voice_enabled": voice_enabled,
        "auto_acknowledge": auto_acknowledge,
        "speaker_roles_bound": speaker_binding_summary["bound"],
        "speaker_binding_summary": speaker_binding_summary,
        "final_metrics": metrics,
    }
    store.write_json("result.json", result)

    if valid:
        progress(
            f"    PASS: {metrics['assessment_count']} assessments, "
            f"{len(grouped['interventions'])} interventions, "
            f"{len(grouped['intervention_outcomes'])} outcomes"
        )
    else:
        progress(f"    FAIL: {error_message}")
    return store.run_dir, valid
