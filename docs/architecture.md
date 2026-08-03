# Technical Architecture

The prototype contains seven separable paths:

1. Audio and video capture generate timestamped chunks.
2. A local Praat/Parselmouth layer measures pitch and intensity without assigning emotion or state.
3. A realtime multimodal provider converts those chunks into transcript and observable evidence events.
4. The fusion layer combines current evidence, prior state, and the formative-study codebook to produce an auditable state assessment.
5. The deterministic trajectory controller converts consecutive assessments into intervention-timing decisions and post-intervention recovery records.
6. The target-aware strategy selector converts an authorized timing decision into one constrained intervention plan.
7. The delivery coordinator converts that plan into identical prominent-text and spoken-voice outputs, with separate execution evidence for each channel.

The provider interface is intentionally isolated from the state logic. Qwen-Omni-Realtime can therefore be replaced or supplemented without changing the codebook, evidence timeline, or evaluation format.

All filesystem access uses absolute `pathlib.Path` values generated from the repository location. Secrets are read from environment variables and never written to logs.

## Video replay test path

1. `capture/video_replay.py` validates one clip, converts its original audio to 16 kHz mono PCM chunks, and samples compressed JPEG frames at approximately 1 fps.
2. `acoustics/prosody.py` measures the full channel and cited audio intervals. The versioned policy in `config/acoustic_analysis.yaml` prohibits single-feature state decisions and records speaker-binding and microphone limitations.
3. `providers/qwen_omni_realtime.py` sends the chunks in chronological order through Alibaba Cloud's documented native WebSocket protocol in manual-turn mode.
4. `fusion/prompting.py` translates the versioned formative-study codebook and the required state schema into the model instruction. Audio and video sufficiency are assessed independently, and uncertain state boundaries remain explicit.
5. `fusion/response_parser.py` rejects prose or malformed outputs, preserves Qwen ASR emotion labels as supporting observations, and only accepts a `StateAssessment` that passes both the local Pydantic schema and runtime checks.
6. `storage/run_artifacts.py` creates one run folder with hashes, versions, the exact prompt, acoustic measurements, summarized client events, server events, transcription events, classification metrics, audit warnings, and the validated assessment.

Raw audio/image Base64 payloads and API credentials are intentionally excluded from run artifacts.

## Live device capture path

1. `capture/devices.py` uses FFmpeg DirectShow discovery to list Windows cameras and microphones, and requires an explicit device index or exact name.
2. `capture/directshow.py` opens both selected endpoints in one DirectShow graph, resamples audio to 16 kHz mono PCM16, samples timestamp-labelled JPEG frames and places both modalities on one strictly increasing monotonic timeline.
3. `capture/media.py` defines the common `MediaSource` and `MediaChunk` contract used by both decoded local video and live devices. It also reserves a speaker-segment contract without pretending that speaker binding is already implemented.
4. `capture/buffer.py` bounds memory use. Audio backpressure stops with a diagnostic error instead of silently losing speech; video overload drops the oldest frame and records the count.
5. `capture/session.py` owns one producer thread, stop signal and device close lifecycle.
6. `live_test.py` validates a short capture without calling an API and writes only a manifest, payload-free event summaries, metrics and a result.

After real-device dry-run validation, the same media chunks can be forwarded to the existing realtime provider without changing the four-state codebook or downstream intervention modules.

## Continuous trajectory control path

1. Module one supplies consecutive schema-valid `StateAssessment` objects.
2. `control/state_tracker.py` stores the ordered state trajectory without replacing the module-one classification.
3. `config/intervention_policy.yaml` maps normal to no intervention, fluctuation to observation, dysregulation to explicit intervention, and high risk to progressive support.
4. Dysregulation and high-risk actions are held until a natural interaction boundary. There is no fixed time or event-count trigger.
5. After an intervention decision, the controller requires an observable parent-child response before permitting another intervention decision.
6. The controller saves the action, reason, research-basis identifiers, state transition, evidence actors, and recovery status. Strategy selection and utterance generation remain separate downstream responsibilities.

## Target-aware strategy path

1. Module three receives the matching module-one assessment and module-two decision.
2. `intervention/selector.py` refuses to run unless module two authorized strategy selection.
3. `config/strategy_cards.yaml` maps intervention-state performances to versioned parent-, child-, or dyad-targeted cards.
4. Actor-specific cards require actor-specific evidence; unknown or dyadic evidence cannot be used to single out one person.
5. Target actor, repair target, approved template, expected recovery, and research provenance are stored in one `InterventionPlan`; delivery modality is deliberately left to module four.
6. Post-intervention non-recovery or deterioration selects a response-gated next card without overriding the module-two recovery decision.

## Dual-channel delivery path

1. Module four accepts only an authorized `InterventionPlan`; it does not alter state, timing, target actor, repair target or strategy content.
2. `config/delivery_policy.yaml` versions the project decision to use prominent visual text and spoken voice with identical core content.
3. `delivery/coordinator.py` creates one UI-facing `DeliveryPackage`, preserves the module-three target actor, and keeps the visual prompt non-blocking.
4. Voice unavailability produces an explicit degraded result while retaining the visual prompt; pausing interventions holds both outputs.
5. Visual rendering and voice playback are recorded separately. Successful output never implies that a participant saw, heard, understood or adopted the intervention.
6. `delivery-test` replays modules one to four and writes JSON artifacts plus a dependency-free browser preview. With `--synthesize-voice`, the fixed Qwen TTS snapshot synthesizes the approved message using Maia, saves a hashed WAV and sanitized server-event trace, and the preview plays that file instead of a browser-selected system voice.
