from __future__ import annotations

import asyncio
import base64
import io
import re
import time
import wave
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from coregulation_poc.acoustics.tencent_voiceprint import SpeakerEnrollmentRecord
from coregulation_poc.capture.media import MediaChunk
from coregulation_poc.control import (
    STATE_RANK,
    BoundaryStateTracker,
    StateTrajectoryController,
    load_intervention_policy,
)
from coregulation_poc.delivery import (
    ChannelExecution,
    DeliveryCoordinator,
    DeliveryPackage,
    DeliveryRuntimeContext,
    OutputExecutionStatus,
    OutputModality,
    load_delivery_policy,
    not_attempted_voice_execution,
)
from coregulation_poc.intervention import (
    MessageGenerator,
    StrategyChoiceGenerator,
    StrategySelector,
    load_strategy_library,
)
from coregulation_poc.intervention.models import InterventionPlan
from coregulation_poc.models import (
    Actor,
    BoundarySignals,
    ConfidenceLevel,
    ControlObservation,
    CoregulationState,
    EvidenceByModality,
    EvidenceSufficiency,
    InteractionTrajectory,
    Interruptibility,
    InterventionAction,
    ModalityEvidence,
    StateAssessment,
)
from coregulation_poc.runtime.recognition import StateRecognizer, WindowObservation
from coregulation_poc.runtime.window import MediaWindow, RollingMediaWindow

ServerEventSender = Callable[[dict[str, Any]], Awaitable[None]]
TurnBoundaryDetector = Callable[[MediaWindow], bool]


@dataclass(frozen=True, slots=True)
class RealtimeLoopConfig:
    """Engineering cadence controls; none of these values classify dyadic state."""

    window_duration_ms: int = 10_000
    assessment_interval_ms: int = 10_000
    post_intervention_observation_ms: int = 4_000
    max_assessments_per_session: int = 180
    history_assessments: int = 6
    max_parallel_perception: int = 3
    max_parallel_judgment: int = 2
    analysis_deadline_seconds: float = 35.0
    max_pending_analysis_jobs: int = 12
    max_intervention_staleness_ms: int = 35_000
    voice_synthesis_timeout_seconds: float = 8.0
    shutdown_drain_timeout_seconds: float = 30.0
    voice_enabled: bool = False

    def __post_init__(self) -> None:
        if self.window_duration_ms < 3_000:
            raise ValueError("window_duration_ms must be at least 3000")
        if self.assessment_interval_ms < 1_000:
            raise ValueError("assessment_interval_ms must be at least 1000")
        if self.post_intervention_observation_ms < 0:
            raise ValueError("post_intervention_observation_ms cannot be negative")
        if self.max_assessments_per_session < 0:
            raise ValueError("max_assessments_per_session must be non-negative (0 = unlimited)")
        if not 1 <= self.history_assessments <= 20:
            raise ValueError("history_assessments must be between 1 and 20")
        if not 1 <= self.max_parallel_perception <= 4:
            raise ValueError("max_parallel_perception must be between 1 and 4")
        if not 1 <= self.max_parallel_judgment <= 4:
            raise ValueError("max_parallel_judgment must be between 1 and 4")
        if self.analysis_deadline_seconds <= 0:
            raise ValueError("analysis deadline must be positive")
        if not 2 <= self.max_pending_analysis_jobs <= 60:
            raise ValueError("max_pending_analysis_jobs must be between 2 and 60")
        if self.max_intervention_staleness_ms < 5_000:
            raise ValueError("intervention staleness must be at least 5000 ms")
        if self.voice_synthesis_timeout_seconds <= 0:
            raise ValueError("voice synthesis timeout must be positive")
        if self.shutdown_drain_timeout_seconds <= 0:
            raise ValueError("shutdown drain timeout must be positive")


@dataclass(frozen=True, slots=True)
class VoiceAudio:
    pcm_audio: bytes
    sample_rate_hz: int
    provider: str
    output_identifier: str | None = None


@dataclass(slots=True)
class _AnalysisJob:
    """One immutable media window whose result is committed chronologically."""

    snapshot: MediaWindow
    observation_task: asyncio.Task[WindowObservation] | None
    judgment_task: asyncio.Task[tuple[WindowObservation, StateAssessment]] | None
    completed: asyncio.Future[None]
    scheduled_monotonic: float
    skip_reason: str | None = None


class VoiceSynthesizer(Protocol):
    async def synthesize(self, text: str) -> VoiceAudio: ...


class RealtimeBrowserSession(Protocol):
    @property
    def api_call_count(self) -> int: ...

    @property
    def runtime_metrics(self) -> dict[str, Any]: ...

    async def start(self) -> None: ...

    async def accept_chunk(self, chunk: MediaChunk) -> None: ...

    async def handle_control(self, control: dict[str, Any]) -> bool: ...

    async def stop(self, status: str) -> None: ...


RealtimeSessionFactory = Callable[
    [str, ServerEventSender, SpeakerEnrollmentRecord | None], RealtimeBrowserSession
]


def pcm16_wav_bytes(pcm_audio: bytes, sample_rate_hz: int) -> bytes:
    if not pcm_audio or len(pcm_audio) % 2:
        raise ValueError("voice audio must contain non-empty 16-bit PCM samples")
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm_audio)
    return target.getvalue()


def detect_trailing_pcm_silence(window: MediaWindow) -> bool:
    """Conservative technical turn-boundary hint, kept separate from state recognition."""

    chunks = window.audio_chunks[-4:]
    if len(chunks) < 4:
        return False
    root_mean_squares = []
    for chunk in chunks:
        samples = np.frombuffer(chunk.payload, dtype="<i2").astype(np.float64)
        if not samples.size:
            return False
        root_mean_squares.append(float(np.sqrt(np.mean(np.square(samples)))))
    return max(root_mean_squares) <= 450


class RealtimeSession:
    """One browser session coordinating all four modules as a feedback loop."""

    def __init__(
        self,
        *,
        session_id: str,
        recognizer: StateRecognizer,
        send_event: ServerEventSender,
        config: RealtimeLoopConfig | None = None,
        voice_synthesizer: VoiceSynthesizer | None = None,
        turn_boundary_detector: TurnBoundaryDetector = detect_trailing_pcm_silence,
        text_chat_provider: object | None = None,
    ) -> None:
        self.session_id = session_id
        self.recognizer = recognizer
        self.send_event = send_event
        self.config = config or RealtimeLoopConfig()
        self.voice_synthesizer = voice_synthesizer
        self.turn_boundary_detector = turn_boundary_detector
        self.window = RollingMediaWindow(duration_ms=self.config.window_duration_ms)
        self.controller = StateTrajectoryController(load_intervention_policy())
        self.boundary_tracker = BoundaryStateTracker.from_codebook()
        self.strategy_library = load_strategy_library()
        message_generator = None
        strategy_choice_generator = None
        if text_chat_provider is not None:
            message_generator = MessageGenerator(
                provider=text_chat_provider,
                max_characters=self.strategy_library.principles.maximum_message_characters,
                max_sentences=self.strategy_library.principles.maximum_message_sentences,
                banned_phrases=list(self.strategy_library.banned_phrases),
            )
            strategy_choice_generator = StrategyChoiceGenerator(text_chat_provider)
        self.selector = StrategySelector(
            self.strategy_library,
            message_generator=message_generator,
            strategy_choice_generator=strategy_choice_generator,
        )
        self.strategy_cards = {card.strategy_id: card for card in self.strategy_library.cards}
        self.delivery = DeliveryCoordinator(load_delivery_policy())
        self.previous_state: CoregulationState | None = None
        self.previous_plan: InterventionPlan | None = None
        self._recent_intervention_messages: list[str] = []
        self._recorded_message_delivery_ids: set[str] = set()
        self.assessment_history: list[StateAssessment] = []
        self.pending_delivery: DeliveryPackage | None = None
        self._pending_plan: InterventionPlan | None = None
        self.delivery_reports: list[dict[str, Any]] = []
        self.intervention_outcomes: list[dict[str, Any]] = []
        self.expert_interventions: list[dict[str, Any]] = []
        self.family_responses: list[dict[str, Any]] = []
        self.assessment_count = 0
        self._last_scheduled_at_ms: int | None = None
        self._post_response_not_before_ms: int | None = None
        self._positive_maintenance_pending: dict[str, Any] | None = None
        self._positive_maintenance_not_before_ms: int | None = None
        self._scheduled_assessment_count = 0
        self._analysis_queue: asyncio.Queue[_AnalysisJob | None] = asyncio.Queue()
        self._analysis_worker_task: asyncio.Task[None] | None = None
        self._perception_semaphore = asyncio.Semaphore(
            self.config.max_parallel_perception
        )
        self._perception_tasks: set[asyncio.Task[WindowObservation]] = set()
        self._judgment_semaphore = asyncio.Semaphore(self.config.max_parallel_judgment)
        self._judgment_tasks: set[
            asyncio.Task[tuple[WindowObservation, StateAssessment]]
        ] = set()
        self._late_result_tasks: set[asyncio.Task[None]] = set()
        self._latest_media_timestamp_ms = 0
        self._latest_completed_judgment_ms = 0
        self._stale_intervention_count = 0
        self._analysis_timeout_count = 0
        self._analysis_capacity_skip_count = 0
        self._late_analysis_result_count = 0
        self._analysis_error_count = 0
        self._consecutive_analysis_errors = 0
        self._speaker_binding_count = 0
        self._speaker_binding_success_count = 0
        self._boundary_adjustment_count = 0
        self._spontaneous_recovery_count = 0
        self._interventions_paused = False
        self._expert_takeover_active = False
        self._expert_pending: dict[str, Any] | None = None
        self._expert_counter = 0
        self._task_context: dict[str, Any] | None = None
        self._self_continue_suppressed = False
        self._difficulty_feedback_boost = False
        self._started = False
        self._stopped = False

    def set_task_context(self, task_context: dict[str, Any]) -> None:
        self._task_context = task_context
        if hasattr(self.recognizer, "task_context"):
            self.recognizer.task_context = task_context

    @property
    def api_call_count(self) -> int:
        recognition_calls = int(getattr(self.recognizer, "api_call_count", self.assessment_count))
        message_calls = int(getattr(self.selector.message_generator, "call_count", 0))
        strategy_choice_calls = int(
            getattr(self.selector.strategy_choice_generator, "call_count", 0)
        )
        return recognition_calls + message_calls + strategy_choice_calls

    @property
    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "assessment_count": self.assessment_count,
            "scheduled_assessment_count": self._scheduled_assessment_count,
            "analysis_queue_depth": self._analysis_queue.qsize(),
            "perception_inflight_count": sum(
                1 for task in self._perception_tasks if not task.done()
            ),
            "judgment_inflight_count": sum(
                1 for task in self._judgment_tasks if not task.done()
            ),
            "late_result_inflight_count": sum(
                1 for task in self._late_result_tasks if not task.done()
            ),
            "api_call_count": self.api_call_count,
            "strategy_selection_llm_call_count": int(
                getattr(self.selector.strategy_choice_generator, "call_count", 0)
            ),
            "message_generation_llm_call_count": int(
                getattr(self.selector.message_generator, "call_count", 0)
            ),
            "voiceprint_api_call_count": int(
                getattr(self.recognizer, "voiceprint_api_call_count", 0)
            ),
            "delivery_report_count": len(self.delivery_reports),
            "intervention_outcome_count": len(self.intervention_outcomes),
            "expert_intervention_count": len(self.expert_interventions),
            "family_response_count": len(self.family_responses),
            "analysis_error_count": self._analysis_error_count,
            "consecutive_analysis_error_count": self._consecutive_analysis_errors,
            "stale_intervention_count": self._stale_intervention_count,
            "recent_intervention_message_count": len(
                self._recent_intervention_messages
            ),
            "analysis_timeout_count": self._analysis_timeout_count,
            "analysis_capacity_skip_count": self._analysis_capacity_skip_count,
            "late_analysis_result_count": self._late_analysis_result_count,
            "speaker_binding_count": self._speaker_binding_count,
            "speaker_binding_success_count": self._speaker_binding_success_count,
            "boundary_adjustment_count": self._boundary_adjustment_count,
            "spontaneous_recovery_count": self._spontaneous_recovery_count,
            "awaiting_post_intervention_response": (
                self.controller.awaiting_post_intervention_response
            ),
            "awaiting_positive_maintenance_observation": (
                self._positive_maintenance_pending is not None
            ),
            "intervention_queued": False,
            "interventions_paused": self._interventions_paused,
            "expert_takeover_active": self._expert_takeover_active,
            "voice_enabled": self.config.voice_enabled,
            "self_continue_suppressed": self._self_continue_suppressed,
            "difficulty_feedback_boost": self._difficulty_feedback_boost,
        }

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._analysis_worker_task = asyncio.create_task(self._analysis_worker())
        await self.send_event(
            {
                "type": "loop_started",
                "window_duration_ms": self.config.window_duration_ms,
                "assessment_interval_ms": self.config.assessment_interval_ms,
                "max_parallel_perception": self.config.max_parallel_perception,
                "max_parallel_judgment": self.config.max_parallel_judgment,
                "analysis_deadline_seconds": self.config.analysis_deadline_seconds,
                "max_pending_analysis_jobs": self.config.max_pending_analysis_jobs,
                "max_intervention_staleness_ms": (
                    self.config.max_intervention_staleness_ms
                ),
                "voice_enabled": self.config.voice_enabled,
            }
        )

    async def accept_chunk(self, chunk: MediaChunk) -> None:
        if not self._started or self._stopped:
            return
        self._latest_media_timestamp_ms = max(
            self._latest_media_timestamp_ms,
            chunk.timestamp_ms,
        )
        self.window.append(chunk)
        snapshot = self.window.snapshot()
        if snapshot is None:
            return
        if not snapshot.has_both_modalities:
            return
        if (
            self.config.max_assessments_per_session > 0
            and self._scheduled_assessment_count
            >= self.config.max_assessments_per_session
        ):
            return
        if self._last_scheduled_at_ms is None:
            due = snapshot.end_ms >= self.config.assessment_interval_ms
        else:
            due = snapshot.end_ms - self._last_scheduled_at_ms >= self.config.assessment_interval_ms
        if not due:
            return
        await self._schedule_analysis(snapshot)

    async def analyze_now(self) -> None:
        """Run one deterministic analysis cycle; primarily useful for local verification."""

        snapshot = self.window.snapshot()
        if snapshot is None or not snapshot.has_both_modalities:
            raise ValueError("both audio and image media are required before analysis")
        if (
            self._last_scheduled_at_ms is not None
            and snapshot.end_ms <= self._last_scheduled_at_ms
        ):
            await self.wait_for_analysis()
            return
        completed = await self._schedule_analysis(snapshot)
        await completed

    async def wait_for_analysis(self) -> None:
        """Wait until all captured windows have been judged in time order."""

        await self._analysis_queue.join()

    def _supports_staged_recognition(self) -> bool:
        return callable(getattr(self.recognizer, "observe", None)) and callable(
            getattr(self.recognizer, "judge", None)
        )

    async def _schedule_analysis(self, snapshot: MediaWindow) -> asyncio.Future[None]:
        """Start bounded model work while preserving chronological commit order."""

        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()
        observation_task: asyncio.Task[WindowObservation] | None = None
        judgment_task: asyncio.Task[tuple[WindowObservation, StateAssessment]] | None = None
        skip_reason: str | None = None
        if self._analysis_queue.qsize() >= self.config.max_pending_analysis_jobs:
            # Preserve the ten-second timeline slot without retaining another
            # raw media payload or creating an unbounded cloud request queue.
            skip_reason = "analysis_capacity_exceeded"
            snapshot = MediaWindow(
                chunks=(),
                start_ms=snapshot.start_ms,
                end_ms=snapshot.end_ms,
            )
            self._analysis_capacity_skip_count += 1
        elif self._supports_staged_recognition():
            history_snapshot = tuple(
                self.assessment_history[-self.config.history_assessments :]
            )
            previous_state_snapshot = self.previous_state
            observation_task = asyncio.create_task(self._observe_window(snapshot))
            self._perception_tasks.add(observation_task)
            observation_task.add_done_callback(self._perception_tasks.discard)
            judgment_task = asyncio.create_task(
                self._judge_after_observation(
                    observation_task,
                    previous_state=previous_state_snapshot,
                    history=history_snapshot,
                )
            )
            self._judgment_tasks.add(judgment_task)
            judgment_task.add_done_callback(self._judgment_tasks.discard)
        await self._analysis_queue.put(
            _AnalysisJob(
                snapshot=snapshot,
                observation_task=observation_task,
                judgment_task=judgment_task,
                completed=completed,
                scheduled_monotonic=time.monotonic(),
                skip_reason=skip_reason,
            )
        )
        self._last_scheduled_at_ms = snapshot.end_ms
        self._scheduled_assessment_count += 1
        return completed

    async def _observe_window(self, snapshot: MediaWindow) -> WindowObservation:
        async with self._perception_semaphore:
            await self.send_event(
                {
                    "type": "analysis_started",
                    "window_start_ms": snapshot.start_ms,
                    "window_end_ms": snapshot.end_ms,
                }
            )
            staged_recognizer = cast(Any, self.recognizer)
            return await staged_recognizer.observe(
                session_id=self.session_id,
                window=snapshot,
            )

    async def _judge_after_observation(
        self,
        observation_task: asyncio.Task[WindowObservation],
        *,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
    ) -> tuple[WindowObservation, StateAssessment]:
        """Judge independently, then let the ordered worker commit the result."""

        observation = await observation_task
        return await self._judge_observation(
            observation,
            previous_state=previous_state,
            history=history,
        )

    async def _judge_observation(
        self,
        observation: WindowObservation,
        *,
        previous_state: CoregulationState | None,
        history: tuple[StateAssessment, ...],
    ) -> tuple[WindowObservation, StateAssessment]:
        """Judge one window with an immutable scheduling-time history snapshot."""

        async with self._judgment_semaphore:
            staged_recognizer = cast(Any, self.recognizer)
            assessment = await staged_recognizer.judge(
                observation=observation,
                previous_state=previous_state,
                history=history,
                history_available=bool(history),
            )
            self._latest_completed_judgment_ms = max(
                self._latest_completed_judgment_ms,
                assessment.assessed_at_ms,
            )
        return observation, assessment

    async def _analysis_worker(self) -> None:
        while True:
            job = await self._analysis_queue.get()
            if job is None:
                self._analysis_queue.task_done()
                return
            try:
                await self._analyze_job(job)
                if self._consecutive_analysis_errors:
                    recovered_after = self._consecutive_analysis_errors
                    self._consecutive_analysis_errors = 0
                    if recovered_after >= 3:
                        with suppress(ConnectionError, RuntimeError):
                            await self.send_event(
                                {
                                    "type": "analysis_recovered",
                                    "recovered_after_errors": recovered_after,
                                }
                            )
            except asyncio.CancelledError:
                if not job.completed.done():
                    job.completed.cancel()
                raise
            except Exception as exc:
                self._analysis_error_count += 1
                self._consecutive_analysis_errors += 1
                with suppress(ConnectionError, RuntimeError):
                    await self._apply_missing_assessment(
                        job,
                        reason=str(exc),
                    )
                with suppress(ConnectionError, RuntimeError):
                    await self.send_event(
                        {
                            "type": "loop_error",
                            "stage": "analysis",
                            "window_start_ms": job.snapshot.start_ms,
                            "window_end_ms": job.snapshot.end_ms,
                            "message": str(exc),
                            "retryable": True,
                            "consecutive_error_count": self._consecutive_analysis_errors,
                            "service_degraded": self._consecutive_analysis_errors >= 3,
                        }
                    )
            finally:
                if not job.completed.done():
                    job.completed.set_result(None)
                self._analysis_queue.task_done()

    async def _analyze_job(self, job: _AnalysisJob) -> None:
        snapshot = job.snapshot
        if job.skip_reason is not None:
            await self.send_event(
                {
                    "type": "analysis_skipped",
                    "window_start_ms": snapshot.start_ms,
                    "window_end_ms": snapshot.end_ms,
                    "reason": job.skip_reason,
                    "record_only": True,
                }
            )
            await self._apply_missing_assessment(job, reason=job.skip_reason)
            return
        history = tuple(self.assessment_history[-self.config.history_assessments :])
        history_available = bool(history)
        if job.judgment_task is not None:
            elapsed = max(0.0, time.monotonic() - job.scheduled_monotonic)
            remaining = max(0.0, self.config.analysis_deadline_seconds - elapsed)
            try:
                async with asyncio.timeout(remaining):
                    observation, model_assessment = await asyncio.shield(
                        job.judgment_task
                    )
            except TimeoutError:
                self._analysis_timeout_count += 1
                late_recorder = asyncio.create_task(self._record_late_analysis(job))
                self._late_result_tasks.add(late_recorder)
                late_recorder.add_done_callback(self._late_result_tasks.discard)
                raise TimeoutError(
                    "analysis deadline exceeded; chronological slot recorded as unavailable"
                ) from None
            history_available = bool(self.assessment_history)
        else:
            observation = None
            await self.send_event(
                {
                    "type": "analysis_started",
                    "window_start_ms": snapshot.start_ms,
                    "window_end_ms": snapshot.end_ms,
                }
            )
            model_assessment = await self.recognizer.assess(
                session_id=self.session_id,
                window=snapshot,
                previous_state=self.previous_state,
                history=history,
                history_available=history_available,
            )
        # The chronological worker owns trajectory state.  This also protects
        # generic recognizers that may have started with slightly stale context.
        model_assessment = model_assessment.model_copy(
            update={"previous_state": self._trajectory_previous_state()}
        )
        await self._apply_assessment(
            snapshot=snapshot,
            model_assessment=model_assessment,
            history_available=history_available,
            recognition_observation=observation,
        )

    async def _record_late_analysis(self, job: _AnalysisJob) -> None:
        """Persist a late model result without allowing it to change live state."""

        task = job.judgment_task
        if task is None:
            return
        try:
            observation, assessment = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with suppress(ConnectionError, RuntimeError):
                await self.send_event(
                    {
                        "type": "late_analysis_failure",
                        "window_start_ms": job.snapshot.start_ms,
                        "window_end_ms": job.snapshot.end_ms,
                        "message": str(exc),
                        "record_only": True,
                    }
                )
            return
        self._late_analysis_result_count += 1
        perception_report = getattr(observation, "perception_report", None)
        with suppress(ConnectionError, RuntimeError):
            await self.send_event(
                {
                    "type": "late_analysis_result",
                    "window_start_ms": job.snapshot.start_ms,
                    "window_end_ms": job.snapshot.end_ms,
                    "state": None if assessment.state is None else assessment.state.value,
                    "assessment": assessment.model_dump(mode="json"),
                    "perception_report": (
                        perception_report.model_dump(mode="json")
                        if perception_report is not None
                        else None
                    ),
                    "record_only": True,
                }
            )

    async def _apply_missing_assessment(
        self,
        job: _AnalysisJob,
        *,
        reason: str,
    ) -> None:
        """Keep one explicit timeline slot when cloud analysis is unavailable."""

        limitation = f"Window analysis unavailable: {reason}"
        insufficient = ModalityEvidence(
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
            items=[],
            limitation_reason=limitation,
        )
        assessment = StateAssessment(
            session_id=self.session_id,
            assessed_at_ms=job.snapshot.end_ms,
            state=None,
            previous_state=self._trajectory_previous_state(),
            trajectory=InteractionTrajectory.UNCLEAR,
            evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
            confidence=ConfidenceLevel.LOW,
            alternative_state=None,
            ambiguity_reason=limitation,
            interaction_performance=[],
            task_process=None,
            support_need=None,
            support_target=Actor.UNKNOWN,
            interruptibility=Interruptibility.UNCLEAR,
            boundary_signals=BoundarySignals(),
            modality_evidence=EvidenceByModality(
                audio=insufficient,
                video=insufficient.model_copy(deep=True),
            ),
            reason=limitation,
            limitations=[limitation],
        )
        await self._apply_assessment(
            snapshot=job.snapshot,
            model_assessment=assessment,
            history_available=bool(self.assessment_history),
            recognition_observation=None,
        )

    def _trajectory_previous_state(self) -> CoregulationState | None:
        """Return the immediately preceding slot, including an unknown slot."""

        if not self.assessment_history:
            return None
        return self.assessment_history[-1].state

    def _speaker_binding_event(
        self,
        snapshot: MediaWindow,
        recognition_observation: WindowObservation | None = None,
    ) -> dict[str, Any] | None:
        binding = (
            recognition_observation.speaker_binding
            if recognition_observation is not None
            else getattr(self.recognizer, "last_speaker_binding", None)
        )
        if binding is None:
            return None
        self._speaker_binding_count += 1
        if binding.bound:
            self._speaker_binding_success_count += 1

        acoustic = (
            recognition_observation.acoustic_features
            if recognition_observation is not None
            else getattr(self.recognizer, "last_acoustic_features", None)
        )
        acoustic_by_interval = {}
        if acoustic is not None:
            acoustic_by_interval = {
                (segment.start_ms, segment.end_ms): segment for segment in acoustic.segments
            }

        segments = []
        for segment in binding.segments[:50]:
            measured = acoustic_by_interval.get((segment.start_ms, segment.end_ms))
            segments.append(
                {
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "speaker": segment.speaker.value,
                    "parent_cosine": segment.parent_cosine,
                    "child_cosine": segment.child_cosine,
                    "provider_score": segment.provider_score,
                    "confidence": segment.confidence,
                    "forced_assignment": segment.forced_assignment,
                    "mean_f0_hz": (segment.mean_f0_hz if segment.mean_f0_hz > 0 else None),
                    "median_f0_hz": (segment.median_f0_hz if segment.median_f0_hz > 0 else None),
                    "rms_energy": (None if measured is None else measured.rms_energy),
                }
            )

        return {
            "type": "speaker_binding",
            "window_start_ms": snapshot.start_ms,
            "window_end_ms": snapshot.end_ms,
            "bound": binding.bound,
            "method": binding.method,
            "limitation_reason": binding.limitation_reason,
            "parent_segment_count": binding.parent_segment_count,
            "child_segment_count": binding.child_segment_count,
            "parent_mean_cosine": binding.parent_mean_cosine,
            "child_mean_cosine": binding.child_mean_cosine,
            "provider_request_count": binding.provider_request_count,
            "low_confidence_segment_count": binding.low_confidence_segment_count,
            "segment_count": len(binding.segments),
            "segments_truncated": len(binding.segments) > len(segments),
            "segments": segments,
            "total_speech_ms": (None if acoustic is None else acoustic.total_speech_ms),
            "total_silence_ms": (None if acoustic is None else acoustic.total_silence_ms),
            "raw_audio_saved_locally": False,
            "voiceprint_embedding_saved_locally": False,
        }

    async def _apply_assessment(
        self,
        *,
        snapshot: MediaWindow,
        model_assessment: StateAssessment,
        history_available: bool,
        recognition_observation: WindowObservation | None,
    ) -> None:
        analysis_staleness_ms = max(
            0,
            self._latest_media_timestamp_ms - snapshot.end_ms,
        )
        post_response_observed = (
            self._post_response_not_before_ms is not None
            and snapshot.end_ms >= self._post_response_not_before_ms
        )
        positive_maintenance_observed = (
            self._positive_maintenance_not_before_ms is not None
            and snapshot.end_ms >= self._positive_maintenance_not_before_ms
        )
        try:
            boundary_resolution = self.boundary_tracker.resolve(
                model_assessment,
                window_start_ms=snapshot.start_ms,
                window_end_ms=snapshot.end_ms,
            )
            assessment = boundary_resolution.assessment
            if boundary_resolution.rule_applied:
                self._boundary_adjustment_count += 1
            if boundary_resolution.spontaneous_recovery:
                self._spontaneous_recovery_count += 1
            binding_event = self._speaker_binding_event(
                snapshot,
                recognition_observation,
            )
            if binding_event is not None:
                await self.send_event(binding_event)
            observation = ControlObservation(
                assessment=assessment,
                natural_turn_boundary=self.turn_boundary_detector(snapshot),
                post_intervention_response_observed=post_response_observed,
                interaction_history_available=history_available,
            )
            decision = self.controller.ingest(
                observation,
                defer_delivery_timing=True,
            )
            self.assessment_count += 1
            self.assessment_history.append(assessment)
            if assessment.state is not None:
                self.previous_state = assessment.state
            if post_response_observed:
                if self._expert_pending is not None:
                    outcome = self._build_expert_intervention_outcome(assessment)
                else:
                    outcome = self._build_intervention_outcome(
                        assessment=assessment,
                        recovery_status=decision.recovery_status.value,
                    )
                if outcome is not None:
                    self.intervention_outcomes.append(outcome)
                    self.boundary_tracker.record_intervention_outcome(
                        outcome["recovery_status"],
                        observed_at_ms=assessment.assessed_at_ms,
                    )
                    await self.send_event({"type": "intervention_outcome", **outcome})
                self._post_response_not_before_ms = None
                self.pending_delivery = None
                self._pending_plan = None
                self._expert_pending = None
            if positive_maintenance_observed:
                outcome = self._build_positive_maintenance_outcome(assessment)
                if outcome is not None:
                    self.intervention_outcomes.append(outcome)
                    await self.send_event({"type": "intervention_outcome", **outcome})
                self._positive_maintenance_pending = None
                self._positive_maintenance_not_before_ms = None
            await self.send_event(
                {
                    "type": "state_update",
                    "sequence": decision.sequence,
                    "assessed_at_ms": assessment.assessed_at_ms,
                    "state": None if assessment.state is None else assessment.state.value,
                    "previous_state": (
                        None
                        if assessment.previous_state is None
                        else assessment.previous_state.value
                    ),
                    "trajectory": assessment.trajectory.value,
                    "confidence": assessment.confidence.value,
                    "evidence_sufficiency": assessment.evidence_sufficiency.value,
                    "action": decision.action.value,
                    "reason_code": decision.reason_code.value,
                    "recovery_status": decision.recovery_status.value,
                    "post_intervention_response_observed": post_response_observed,
                    "task_process": (
                        assessment.task_process.value
                        if assessment.task_process is not None
                        else None
                    ),
                    "support_need": (
                        assessment.support_need.value
                        if assessment.support_need is not None
                        else None
                    ),
                    "support_target": assessment.support_target.value,
                    "interruptibility": assessment.interruptibility.value,
                    "interaction_performance": list(assessment.interaction_performance),
                    "high_risk_signals": assessment.high_risk_signals.model_dump(
                        mode="json"
                    ),
                    "assessment": assessment.model_dump(mode="json"),
                    "perception_report": (
                        perception.model_dump(mode="json")
                        if (
                            recognition_observation is not None
                            and (perception := getattr(
                                recognition_observation, "perception_report", None
                            )) is not None
                        )
                        else None
                    ),
                    "acoustic_features": (
                        acoustic.model_dump(mode="json")
                        if (
                            recognition_observation is not None
                            and (acoustic := getattr(
                                recognition_observation, "acoustic_features", None
                            )) is not None
                        )
                        else None
                    ),
                    "analysis_staleness_ms": analysis_staleness_ms,
                    "live_action_eligible": (
                        analysis_staleness_ms
                        <= self.config.max_intervention_staleness_ms
                    ),
                    **boundary_resolution.as_event_fields(),
                }
            )
            actionable = decision.action in {
                InterventionAction.INTERVENE,
                InterventionAction.PROGRESSIVE_SUPPORT,
                InterventionAction.REINFORCE,
            }
            if (
                actionable
                and analysis_staleness_ms
                > self.config.max_intervention_staleness_ms
            ):
                await self._hold_stale_intervention(
                    sequence=decision.sequence,
                    staleness_ms=analysis_staleness_ms,
                )
                self._self_continue_suppressed = False
                return
            # Strategy ranking and contextual message generation may make one
            # or two HTTP calls.  Keep them off the event loop so camera/audio
            # capture continues while the result for this original window is
            # pending.  The resulting intervention is still delivered when it
            # becomes ready; it is never reassigned to a newer window.
            selection = await asyncio.to_thread(
                self.selector.select,
                assessment=assessment,
                decision=decision,
                previous_plan=self.previous_plan,
                recent_messages=list(self._recent_intervention_messages),
                task_context=self._task_context,
                difficulty_feedback_boost=self._difficulty_feedback_boost,
            )
            # Consume the boost after it has been applied to selection.
            self._difficulty_feedback_boost = False
            if selection.plan is None:
                if (
                    self.controller.awaiting_post_intervention_response
                    and self.pending_delivery is None
                ):
                    self.controller.mark_intervention_not_delivered()
                    await self.send_event(
                        {
                            "type": "intervention_held",
                            "sequence": decision.sequence,
                            "reason": selection.hold_reason.value,
                        }
                    )
                # Consume self_continue suppression unconditionally:
                # the spec requires exactly one complete analysis window
                # of suppression, regardless of whether this window
                # produced a candidate intervention.
                self._self_continue_suppressed = False
                return
            # self_continue suppresses intervention delivery for one window
            if self._self_continue_suppressed:
                self._self_continue_suppressed = False
                if self.controller.awaiting_post_intervention_response:
                    self.controller.mark_intervention_not_delivered()
                await self.send_event(
                    {
                        "type": "intervention_held",
                        "sequence": decision.sequence,
                        "reason": "self_continue_suppressed",
                    }
                )
                return
            current_staleness_ms = max(
                0,
                self._latest_media_timestamp_ms - snapshot.end_ms,
            )
            if current_staleness_ms > self.config.max_intervention_staleness_ms:
                await self._hold_stale_intervention(
                    sequence=decision.sequence,
                    staleness_ms=current_staleness_ms,
                )
                return
            preparation = self.delivery.prepare(
                plan=selection.plan,
                runtime=DeliveryRuntimeContext(
                    prepared_at_ms=assessment.assessed_at_ms,
                    interventions_paused=self._interventions_paused,
                    voice_enabled=self.config.voice_enabled,
                    voice_available=self.voice_synthesizer is not None,
                ),
            )
            if preparation.package is None:
                self.controller.mark_intervention_not_delivered()
                await self.send_event(
                    {
                        "type": "intervention_held",
                        "sequence": decision.sequence,
                        "reason": preparation.hold_reason.value,
                    }
                )
                return
            self._pending_plan = selection.plan
            await self._deliver(preparation.package)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            self._analysis_error_count += 1
            if (
                self.controller.awaiting_post_intervention_response
                and self.pending_delivery is None
            ):
                self.controller.mark_intervention_not_delivered()
                self._pending_plan = None
            with suppress(ConnectionError, RuntimeError):
                await self.send_event(
                    {
                        "type": "loop_error",
                        "stage": "analysis",
                        "message": str(exc),
                        "retryable": True,
                    }
                )

    async def _hold_stale_intervention(
        self,
        *,
        sequence: int,
        staleness_ms: int,
    ) -> None:
        """Record an actionable old state without showing an obsolete prompt."""

        self._stale_intervention_count += 1
        if self.controller.awaiting_post_intervention_response:
            self.controller.mark_intervention_not_delivered()
        self.pending_delivery = None
        self._pending_plan = None
        await self.send_event(
            {
                "type": "intervention_held",
                "sequence": sequence,
                "reason": "stale_assessment_record_only",
                "analysis_staleness_ms": staleness_ms,
            }
        )

    async def _deliver(self, package: DeliveryPackage) -> None:
        if self._interventions_paused or self._expert_takeover_active:
            if self.controller.awaiting_post_intervention_response:
                self.controller.mark_intervention_not_delivered()
            self._pending_plan = None
            await self.send_event(
                {
                    "type": "intervention_held",
                    "sequence": package.sequence,
                    "reason": (
                        "expert_takeover_active"
                        if self._expert_takeover_active
                        else "interventions_paused"
                    ),
                }
            )
            return
        plan = self._pending_plan
        if plan is not None:
            self.previous_plan = plan
        self.pending_delivery = package
        voice_pending = bool(
            package.voice_prompt.enabled and self.voice_synthesizer is not None
        )
        intervention_event = {
                "type": "intervention",
                "delivery_id": package.delivery_id,
                "sequence": package.sequence,
                "prepared_at_ms": package.prepared_at_ms,
                "target_actor": package.target_actor.value,
                "strategy_id": package.strategy_id,
                "repair_target": package.repair_target.value,
                "source": "ai",
                "heading": package.visual_prompt.heading,
                "message": package.visual_prompt.message,
                "strategy_selection_source": (
                    plan.strategy_selection_source.value if plan is not None else None
                ),
                "semantic_selection_confidence": (
                    plan.semantic_selection_confidence.value
                    if plan is not None and plan.semantic_selection_confidence is not None
                    else None
                ),
                "semantic_relaxed_dimensions": (
                    plan.semantic_relaxed_dimensions if plan is not None else []
                ),
                "selection_reason": plan.selection_reason if plan is not None else None,
                "message_source": (
                    plan.message_source.value if plan is not None else None
                ),
                "message_validation_checks": (
                    plan.validation_checks if plan is not None else {}
                ),
                "dismissible": package.visual_prompt.dismissible,
                "voice_expected": package.voice_prompt.enabled,
                "voice_provider": package.voice_prompt.provider,
                "voice_pending": voice_pending,
                "voice_output_identifier": None,
                "audio_mime_type": None,
                "audio_base64": None,
                "voice_error": None,
            }
        try:
            await self.send_event(intervention_event)
        except (ConnectionError, OSError, RuntimeError):
            self.pending_delivery = None
            self._pending_plan = None
            if self.controller.awaiting_post_intervention_response:
                self.controller.mark_intervention_not_delivered()
            raise
        if not voice_pending:
            return

        audio_base64: str | None = None
        audio_mime_type: str | None = None
        voice_error: str | None = None
        voice_output_identifier: str | None = None
        try:
            voice = await asyncio.wait_for(
                self.voice_synthesizer.synthesize(package.voice_prompt.message),
                timeout=self.config.voice_synthesis_timeout_seconds,
            )
            wav = pcm16_wav_bytes(voice.pcm_audio, voice.sample_rate_hz)
            audio_base64 = base64.b64encode(wav).decode("ascii")
            audio_mime_type = "audio/wav"
            voice_output_identifier = voice.output_identifier
        except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
            voice_error = str(exc)
        await self.send_event(
            {
                "type": "intervention_voice",
                "delivery_id": package.delivery_id,
                "voice_provider": package.voice_prompt.provider,
                "voice_output_identifier": voice_output_identifier,
                "audio_mime_type": audio_mime_type,
                "audio_base64": audio_base64,
                "voice_error": voice_error,
            }
        )

    async def handle_control(self, control: dict[str, Any]) -> bool:
        control_type = control.get("type")
        if control_type == "expert_takeover":
            await self._start_expert_takeover(control)
            return True
        if control_type == "expert_release":
            await self._end_expert_takeover(control)
            return True
        if control_type == "expert_intervention":
            await self._send_expert_intervention(control)
            return True
        if control_type == "family_response":
            await self._record_family_response(control)
            return True
        if control_type in {"pause_interventions", "resume_interventions"}:
            if control_type == "resume_interventions" and self._expert_takeover_active:
                await self.send_event(
                    {
                        "type": "interventions_paused",
                        "reason": "expert_takeover_active",
                        "recorded_at_ms": self._optional_non_negative_int(
                            control.get("recorded_at_ms"), "recorded_at_ms"
                        ),
                    }
                )
                return True
            self._interventions_paused = control_type == "pause_interventions"
            await self.send_event(
                {
                    "type": (
                        "interventions_paused"
                        if self._interventions_paused
                        else "interventions_resumed"
                    ),
                    "recorded_at_ms": self._optional_non_negative_int(
                        control.get("recorded_at_ms"), "recorded_at_ms"
                    ),
                }
            )
            return True
        if control_type != "delivery_execution":
            return False
        if (
            self._expert_pending is not None
            and control.get("delivery_id") == self._expert_pending["delivery_id"]
        ):
            await self._record_expert_delivery(control)
            return True
        package = self.pending_delivery
        if package is None or control.get("delivery_id") != package.delivery_id:
            await self.send_event(
                {
                    "type": "delivery_execution_ignored",
                    "delivery_id": control.get("delivery_id"),
                    "reason": "no_matching_pending_intervention",
                }
            )
            return True
        recorded_at_ms = self._non_negative_int(control.get("recorded_at_ms"), "recorded_at_ms")
        visual = self._channel_execution(
            control.get("visual"),
            modality=OutputModality.VISUAL_TEXT,
            default_provider="browser_overlay",
        )
        if package.voice_prompt.enabled:
            voice = self._channel_execution(
                control.get("voice"),
                modality=OutputModality.SPOKEN_VOICE,
                default_provider=package.voice_prompt.provider,
            )
        else:
            voice = not_attempted_voice_execution()
        report = self.delivery.record_execution(
            package=package,
            recorded_at_ms=recorded_at_ms,
            visual=visual,
            voice=voice,
            user_acknowledged=(
                bool(control["user_acknowledged"])
                if isinstance(control.get("user_acknowledged"), bool)
                else None
            ),
        )
        delivered = any(
            channel.status is OutputExecutionStatus.DELIVERED for channel in (visual, voice)
        )
        delivery_timeline_ms: int | None = None
        if delivered:
            delivery_timeline_ms = max(
                recorded_at_ms,
                self._latest_media_timestamp_ms,
                package.prepared_at_ms,
            )
            pending_plan = self._pending_plan
            if pending_plan is None:
                if self.controller.awaiting_post_intervention_response:
                    self.controller.mark_intervention_not_delivered()
                self.pending_delivery = None
                await self.send_event(
                    {
                        "type": "loop_error",
                        "stage": "delivery",
                        "message": "delivered intervention had no pending plan",
                        "retryable": False,
                    }
                )
                return True
            self.controller.mark_intervention_delivered(
                pending_plan.state,
                delivered_at_ms=delivery_timeline_ms,
            )
            if package.delivery_id not in self._recorded_message_delivery_ids:
                shown_message = re.sub(
                    r"\s+", " ", package.visual_prompt.message
                ).strip()
                if shown_message:
                    self._recent_intervention_messages.append(shown_message)
                    self._recent_intervention_messages = (
                        self._recent_intervention_messages[-3:]
                    )
                self._recorded_message_delivery_ids.add(package.delivery_id)
            positive_maintenance = (
                pending_plan.decision_action is InterventionAction.REINFORCE
            )
            if positive_maintenance:
                self._positive_maintenance_pending = {
                    "delivery_id": package.delivery_id,
                    "strategy_id": pending_plan.strategy_id,
                    "target_actor": pending_plan.target_actor.value,
                    "repair_target": pending_plan.repair_target.value,
                    "expected_recovery": list(pending_plan.expected_recovery),
                }
                self._positive_maintenance_not_before_ms = (
                    delivery_timeline_ms + self.config.post_intervention_observation_ms
                )
                self.pending_delivery = None
                self._pending_plan = None
            else:
                self._post_response_not_before_ms = (
                    delivery_timeline_ms + self.config.post_intervention_observation_ms
                )
            # The next model window must contain only behavior observed after delivery.
            self.window.clear()
            self._last_scheduled_at_ms = delivery_timeline_ms
        else:
            if self.controller.awaiting_post_intervention_response:
                self.controller.mark_intervention_not_delivered()
            self.pending_delivery = None
            self._pending_plan = None
        report_record = report.model_dump(mode="json")
        report_record["delivery_timeline_ms"] = delivery_timeline_ms
        self.delivery_reports.append(report_record)
        await self.send_event(
            {
                "type": "delivery_execution_received",
                "delivery_id": package.delivery_id,
                "overall_status": report.overall_status.value,
                "recorded_at_ms": report.recorded_at_ms,
                "visual_status": report.visual.status.value,
                "voice_status": report.voice.status.value,
                "post_intervention_observation_armed": (
                    delivered
                    and self._post_response_not_before_ms is not None
                ),
                "positive_maintenance_observation_armed": (
                    delivered
                    and self._positive_maintenance_pending is not None
                ),
                "delivery_timeline_ms": delivery_timeline_ms,
            }
        )
        return True

    async def _start_expert_takeover(self, control: dict[str, Any]) -> None:
        if self._expert_takeover_active:
            raise ValueError("当前会话已经处于专家接管状态")
        operator = self._required_text(control.get("operator"), "operator", maximum=80)
        reason = self._required_text(control.get("reason"), "reason", maximum=300)
        self._expert_takeover_active = True
        self._interventions_paused = True
        if self.pending_delivery is not None:
            if self.controller.awaiting_post_intervention_response:
                self.controller.mark_intervention_not_delivered()
            self.pending_delivery = None
            self._pending_plan = None
            self._post_response_not_before_ms = None
        await self.send_event(
            {
                "type": "expert_takeover_started",
                "source": "expert",
                "operator": operator,
                "reason": reason,
                "recorded_at_ms": self._optional_non_negative_int(
                    control.get("recorded_at_ms"), "recorded_at_ms"
                ),
            }
        )

    async def _end_expert_takeover(self, control: dict[str, Any]) -> None:
        if not self._expert_takeover_active:
            raise ValueError("当前会话尚未由专家接管")
        operator = self._required_text(control.get("operator"), "operator", maximum=80)
        reason = self._required_text(control.get("reason"), "reason", maximum=300)
        self._expert_takeover_active = False
        self._interventions_paused = False
        await self.send_event(
            {
                "type": "expert_takeover_ended",
                "source": "expert",
                "operator": operator,
                "reason": reason,
                "recorded_at_ms": self._optional_non_negative_int(
                    control.get("recorded_at_ms"), "recorded_at_ms"
                ),
            }
        )

    async def _send_expert_intervention(self, control: dict[str, Any]) -> None:
        if not self._expert_takeover_active:
            raise ValueError("专家介入前必须先接管当前会话")
        if self._expert_pending is not None:
            raise ValueError("请先观察上一条专家提示后的互动变化")
        operator = self._required_text(control.get("operator"), "operator", maximum=80)
        reason = self._required_text(control.get("reason"), "reason", maximum=300)
        strategy_id = self._required_text(control.get("strategy_id"), "strategy_id", maximum=80)
        card = self.strategy_cards.get(strategy_id)
        if card is None:
            raise ValueError("专家介入必须选择已审核的策略卡")
        message = self._required_text(control.get("message"), "message", maximum=500)
        self._validate_expert_message(message)
        self._expert_counter += 1
        delivery_id = f"{self.session_id}:expert:{self._expert_counter}"
        pending = {
            "delivery_id": delivery_id,
            "strategy_id": strategy_id,
            "target_actor": card.target_actor.value,
            "repair_target": card.repair_target.value,
            "expected_recovery": list(card.expected_recovery),
            "baseline_state": (None if self.previous_state is None else self.previous_state.value),
            "operator": operator,
            "reason": reason,
        }
        self._expert_pending = pending
        self.expert_interventions.append(dict(pending, message=message))
        await self.send_event(
            {
                "type": "intervention",
                "delivery_id": delivery_id,
                "source": "expert",
                "target_actor": card.target_actor.value,
                "strategy_id": strategy_id,
                "repair_target": card.repair_target.value,
                "heading": "专家支持",
                "message": message,
                "dismissible": True,
                "voice_expected": False,
            }
        )
        await self.send_event(
            {
                "type": "expert_intervention_recorded",
                **pending,
                "message": message,
            }
        )

    async def _record_expert_delivery(self, control: dict[str, Any]) -> None:
        assert self._expert_pending is not None
        recorded_at_ms = self._non_negative_int(control.get("recorded_at_ms"), "recorded_at_ms")
        visual = self._channel_execution(
            control.get("visual"),
            modality=OutputModality.VISUAL_TEXT,
            default_provider="browser_overlay",
        )
        delivered = visual.status is OutputExecutionStatus.DELIVERED
        self.delivery_reports.append(
            {
                "delivery_id": self._expert_pending["delivery_id"],
                "source": "expert",
                "recorded_at_ms": recorded_at_ms,
                "visual_status": visual.status.value,
                "overall_status": "delivered" if delivered else "failed",
            }
        )
        if delivered:
            self._post_response_not_before_ms = (
                recorded_at_ms + self.config.post_intervention_observation_ms
            )
            self.window.clear()
            self._last_scheduled_at_ms = recorded_at_ms
        else:
            self._expert_pending = None
        await self.send_event(
            {
                "type": "delivery_execution_received",
                "delivery_id": control.get("delivery_id"),
                "source": "expert",
                "overall_status": "delivered" if delivered else "failed",
                "recorded_at_ms": recorded_at_ms,
                "visual_status": visual.status.value,
                "voice_status": "not_attempted",
                "post_intervention_observation_armed": delivered,
            }
        )

    async def _record_family_response(self, control: dict[str, Any]) -> None:
        response = self._required_text(control.get("response"), "response", maximum=80)
        allowed = {
            "dismissed",
            "self_continue",
            "task_too_easy",
            "task_just_right",
            "task_too_hard",
        }
        if response not in allowed:
            raise ValueError("不支持的家庭端反馈类型")
        record = {
            "response": response,
            "delivery_id": (
                str(control["delivery_id"]) if isinstance(control.get("delivery_id"), str) else None
            ),
            "recorded_at_ms": self._optional_non_negative_int(
                control.get("recorded_at_ms"), "recorded_at_ms"
            ),
        }
        self.family_responses.append(record)

        # Apply feedback effects on system behavior
        if response == "self_continue":
            # Suppress intervention for one complete analysis window
            self._self_continue_suppressed = True
        elif response == "task_too_hard":
            self._update_task_difficulty("challenging")
            self._difficulty_feedback_boost = True
        elif response == "task_too_easy":
            self._update_task_difficulty("easy")
            self._difficulty_feedback_boost = False
        elif response == "task_just_right":
            self._update_task_difficulty("moderate")
            self._difficulty_feedback_boost = False
        # dismissed: no state change, no acceptance, keeps default observation window

        await self.send_event({"type": "family_response_received", **record})

    def _update_task_difficulty(self, difficulty: str) -> None:
        """Update the task context difficulty from family feedback."""
        if self._task_context is None:
            self._task_context = {}
        self._task_context["task_difficulty"] = difficulty
        if hasattr(self.recognizer, "task_context"):
            self.recognizer.task_context = self._task_context

    def _build_expert_intervention_outcome(
        self,
        assessment: StateAssessment,
    ) -> dict[str, Any] | None:
        pending = self._expert_pending
        if pending is None:
            return None
        baseline = pending.get("baseline_state")
        observed = None if assessment.state is None else assessment.state.value
        # Use the unified STATE_RANK (includes fluctuation) to avoid
        # recording indeterminate when the dyad enters fluctuation.
        rank = {state.value: rank for state, rank in STATE_RANK.items()}
        if baseline not in rank or observed not in rank:
            recovery_status = "indeterminate"
        elif rank[observed] < rank[baseline]:
            recovery_status = "recovered" if observed == "normal" else "partial_recovery"
        elif rank[observed] == rank[baseline]:
            recovery_status = "not_recovered"
        else:
            recovery_status = "deteriorated"
        effect_category = {
            "recovered": "positive",
            "partial_recovery": "limited",
            "not_recovered": "no_change",
            "deteriorated": "negative",
        }.get(recovery_status, "indeterminate")
        return {
            "source": "expert",
            "delivery_id": pending["delivery_id"],
            "strategy_id": pending["strategy_id"],
            "target_actor": pending["target_actor"],
            "repair_target": pending["repair_target"],
            "observed_at_ms": assessment.assessed_at_ms,
            "recovery_status": recovery_status,
            "effect_category": effect_category,
            "expected_recovery": pending["expected_recovery"],
            "observed_state": observed,
            "observed_interaction_performance": list(assessment.interaction_performance),
            "interpretation_limit": (
                "The record links expert support to the next observable window; "
                "it does not by itself establish that the intervention caused the change."
            ),
        }

    def _validate_expert_message(self, message: str) -> None:
        principles = self.strategy_library.principles
        if len(message) > principles.maximum_message_characters:
            raise ValueError("专家提示超过策略库允许的最大长度")
        sentence_count = len(
            [part for part in re.split(r"(?<=[。！？!?])", message) if part.strip()]
        )
        if sentence_count > principles.maximum_message_sentences:
            raise ValueError("专家提示包含过多句子")
        if any(phrase in message for phrase in self.strategy_library.banned_phrases):
            raise ValueError("专家提示包含策略库禁止使用的表达")

    @staticmethod
    def _required_text(value: object, field: str, *, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        text = value.strip()
        if len(text) > maximum:
            raise ValueError(f"{field} is too long")
        return text

    def _build_intervention_outcome(
        self,
        *,
        assessment: StateAssessment,
        recovery_status: str,
    ) -> dict[str, Any] | None:
        package = self.pending_delivery
        plan = self._pending_plan
        if package is None or plan is None:
            return None
        effect_category = {
            "recovered": "positive",
            "partial_recovery": "limited",
            "not_recovered": "no_change",
            "deteriorated": "negative",
        }.get(recovery_status, "indeterminate")
        return {
            "source": "ai",
            "delivery_id": package.delivery_id,
            "strategy_id": plan.strategy_id,
            "target_actor": plan.target_actor.value,
            "repair_target": plan.repair_target.value,
            "observed_at_ms": assessment.assessed_at_ms,
            "recovery_status": recovery_status,
            "effect_category": effect_category,
            "expected_recovery": list(plan.expected_recovery),
            "observed_state": (None if assessment.state is None else assessment.state.value),
            "observed_interaction_performance": list(assessment.interaction_performance),
            "observed_evidence": [
                {
                    "modality": item.modality.value,
                    "actor": item.actor.value,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "code": item.code,
                }
                for item in assessment.modality_evidence.all_items
            ],
            "interpretation_limit": (
                "The record links the strategy to the next observable window; "
                "it does not by itself establish that the intervention caused the change."
            ),
        }

    def _build_positive_maintenance_outcome(
        self,
        assessment: StateAssessment,
    ) -> dict[str, Any] | None:
        pending = self._positive_maintenance_pending
        if pending is None:
            return None
        observed_state = None if assessment.state is None else assessment.state.value
        maintenance_status, effect_category = {
            "normal": ("positive_coordination_maintained", "positive"),
            "fluctuation": ("coordination_temporarily_unsteady", "limited"),
            "dysregulation": ("positive_coordination_not_maintained", "negative"),
            "high_risk": ("positive_coordination_not_maintained", "negative"),
        }.get(observed_state, ("indeterminate", "indeterminate"))
        return {
            "source": "ai",
            "outcome_type": "positive_maintenance",
            **pending,
            "observed_at_ms": assessment.assessed_at_ms,
            "recovery_status": maintenance_status,
            "effect_category": effect_category,
            "observed_state": observed_state,
            "observed_interaction_performance": list(
                assessment.interaction_performance
            ),
            "observed_evidence": [
                {
                    "modality": item.modality.value,
                    "actor": item.actor.value,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "code": item.code,
                }
                for item in assessment.modality_evidence.all_items
            ],
            "interpretation_limit": (
                "The record describes coordination in the next observable window; "
                "it does not by itself establish that the reinforcement caused it."
            ),
        }

    @staticmethod
    def _channel_execution(
        raw: object,
        *,
        modality: OutputModality,
        default_provider: str,
    ) -> ChannelExecution:
        if not isinstance(raw, dict):
            raise ValueError(f"{modality.value} execution must be an object")
        status = OutputExecutionStatus(str(raw.get("status")))
        if status is OutputExecutionStatus.NOT_ATTEMPTED:
            return ChannelExecution(modality=modality, status=status)
        started_at_ms = RealtimeSession._non_negative_int(
            raw.get("started_at_ms"), f"{modality.value}.started_at_ms"
        )
        completed_raw = raw.get("completed_at_ms")
        completed_at_ms = (
            None
            if completed_raw is None
            else RealtimeSession._non_negative_int(
                completed_raw, f"{modality.value}.completed_at_ms"
            )
        )
        error = raw.get("error")
        return ChannelExecution(
            modality=modality,
            status=status,
            started_at_ms=started_at_ms,
            completed_at_ms=completed_at_ms,
            provider=(
                str(raw.get("provider"))
                if isinstance(raw.get("provider"), str)
                else default_provider
            ),
            output_identifier=(
                str(raw.get("output_identifier"))
                if isinstance(raw.get("output_identifier"), str)
                else None
            ),
            error=str(error) if isinstance(error, str) else None,
        )

    @staticmethod
    def _non_negative_int(value: object, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    @staticmethod
    def _optional_non_negative_int(value: object, field: str) -> int | None:
        if value is None:
            return None
        return RealtimeSession._non_negative_int(value, field)

    async def stop(self, status: str) -> None:
        del status
        if self._stopped:
            return
        self._stopped = True
        drain_timed_out = False
        try:
            await asyncio.wait_for(
                self.wait_for_analysis(),
                timeout=self.config.shutdown_drain_timeout_seconds,
            )
        except TimeoutError:
            drain_timed_out = True
            self._analysis_error_count += 1
            with suppress(ConnectionError, RuntimeError):
                await self.send_event(
                    {
                        "type": "loop_error",
                        "stage": "shutdown",
                        "message": "analysis drain timed out; pending windows were cancelled",
                        "retryable": False,
                    }
                )
        if drain_timed_out:
            if self._analysis_worker_task is not None:
                self._analysis_worker_task.cancel()
                with suppress(asyncio.CancelledError, ConnectionError, RuntimeError):
                    await self._analysis_worker_task
            for task in (*self._perception_tasks, *self._judgment_tasks):
                if not task.done():
                    task.cancel()
            while True:
                try:
                    pending = self._analysis_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if pending is not None and not pending.completed.done():
                    pending.completed.set_result(None)
                self._analysis_queue.task_done()
        else:
            await self._analysis_queue.put(None)
            if self._analysis_worker_task is not None:
                with suppress(asyncio.CancelledError, ConnectionError, RuntimeError):
                    await self._analysis_worker_task
        for task in (
            *tuple(self._perception_tasks),
            *tuple(self._judgment_tasks),
            *tuple(self._late_result_tasks),
        ):
            if not task.done():
                task.cancel()
        if self._perception_tasks:
            await asyncio.gather(*self._perception_tasks, return_exceptions=True)
        if self._judgment_tasks:
            await asyncio.gather(*self._judgment_tasks, return_exceptions=True)
        if self._late_result_tasks:
            await asyncio.gather(*self._late_result_tasks, return_exceptions=True)
        self.window.clear()
