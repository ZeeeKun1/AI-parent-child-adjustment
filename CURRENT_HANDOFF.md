# Current Handoff: Parent–Child Co-regulation Realtime PoC

Last updated: 2026-08-03 (Asia/Shanghai)

## 1. Purpose and research context

This repository is a technical feasibility prototype for a CHI-oriented study of AI-mediated parent–child homework co-regulation. The formative study is grounded in parent–child co-regulation theory and dynamic systems theory. The current technical question is deliberately narrow:

> Can Qwen-Omni-Realtime use synchronized speech and sampled video frames to classify a parent–child homework episode into one of four formative-study states and return timestamped, auditable evidence?

The four states are:

- `normal`
- `fluctuation`
- `dysregulation`
- `high_risk`

The final four-state formative-study model is authoritative for implementation. Earlier three-state design drafts in the wider research workspace must not replace it.

Their versioned definitions and evidence policy are in `config/state_codebook.yaml`. The codebook source is recorded as 63 WOZ intervention episodes plus expert interviews.

This is not yet the complete experiment platform. Continuous live capture, parent/child speaker binding, rolling module-one assessments, and the formal participant UI have not been integrated. Modules two through four can replay consecutive assessments, select a constrained strategy, prepare identical prominent-text plus spoken-voice outputs, and generate traceable audio through the fixed Qwen realtime TTS snapshot and Maia voice.

## 2. Repository and environment

Project root: the repository root directory.

Python environment: `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/macOS).

The repository is prepared for publication to `ZeeeKun1/AI-parent-child-adjustment` on the dedicated branch `prototype/local-video-poc`, based on the remote `main` branch. This branch is explicitly the local-uploaded-video technical prototype, not the formal live-camera experiment implementation. Do not delete or reorganize unrelated research files. Raw research media, `.env`, logs, runtime outputs, caches, and credentials are ignored by Git.

Runtime paths are derived from the repository location and converted to absolute `pathlib.Path` values. There are no hard-coded machine paths inside application code.

The `.env` file is already configured with:

- `DASHSCOPE_API_KEY`
- `ALIYUN_WORKSPACE_ID`
- `ALIYUN_REGION=cn-beijing`
- `OMNI_MODEL=qwen3.5-omni-flash-realtime`

Never print, copy, commit, or expose the plaintext API key. The Workspace ID and API key have already been verified to belong to the China (Beijing) region and the same workspace.

## 3. Implemented modules

### Media replay

`src/coregulation_poc/capture/video_replay.py`

- Opens an absolute local video path with PyAV.
- Requires both video and audio tracks.
- Resamples audio to 16 kHz, mono, 16-bit PCM.
- Produces 100 ms audio chunks.
- Samples JPEG frames at approximately 1 fps.
- Resizes frames to at most 1280×720.
- Adds a non-overlapping `frame_time_ms` label to every sampled frame for evidence tracing.
- Compresses each raw JPEG below 190 KB.
- Sorts audio and image chunks on one normalized timeline.

### Theory/formative-study translation

- `config/state_codebook.yaml` version 2: four-state definitions, dynamic decision dimensions, normal–fluctuation boundary guidance, separate classification/action policy, independent modality sufficiency, and uncertainty rules. It deliberately does not impose arbitrary duration or event-count thresholds.
- `src/coregulation_poc/fusion/prompting.py`: embeds the versioned codebook and the required `StateAssessment` JSON schema. Audio evidence must quote observed words; video evidence must cite a labeled frame and describe only visible behavior. Either modality may be insufficient without forcing evidence.
- `src/coregulation_poc/fusion/response_parser.py`: accepts only schema-valid structured output and then validates session ID, clip duration, codebook interaction codes, and history requirements.
- Speaker-role binding is not implemented in this PoC. Evidence actors may be `unknown`; future frontend enrollment and voiceprint matching must provide an external parent/child binding with a low-confidence fallback to `unknown`.

### Qwen realtime transport

`src/coregulation_poc/providers/qwen_omni_realtime.py` and `src/coregulation_poc/providers/websocket_transport.py`

- Uses Alibaba Cloud's documented native WebSocket protocol through `websocket-client`.
- Do not revert to `dashscope.audio.qwen_omni.OmniRealtimeConversation`: its threaded `WebSocketApp` consistently failed to connect on this machine even though the native synchronous WebSocket connection succeeds.
- `websocket_transport.create_websocket_connection` resolves the endpoint hostname to IPv4 only, opens the TCP socket manually, wraps it with SSL, and hands the pre-connected socket to `websocket.create_connection`. This bypasses `websocket-client`'s default `AF_UNSPEC` resolution, which tries IPv6 addresses first and hangs until timeout on networks where IPv6 routes to the Aliyun MaaS host are broken.
- Sends `input_audio_buffer.append` and `input_image_buffer.append` events.
- Uses Manual mode (`turn_detection: null`).
- Commits audio and image buffers before sending `response.create`.
- `response.create` has already been corrected to contain only `event_id` and `type`, matching the official event reference.

### Traceability

`src/coregulation_poc/storage/run_artifacts.py`

Every test creates an isolated run folder containing the input hash, codebook/prompt hash, software versions, summarized client events, server events, transcription events, latency metrics, raw model text, and schema-validated assessment. Real runs additionally save `audit.json` and a best-effort input transcript. Classification validity is distinct from audit readiness, so an ASR failure is preserved as an audit warning rather than hidden by a valid model response. Base64 audio/image payloads and credentials are not logged.

### Diagnostics

`src/coregulation_poc/diagnostics.py`

The `diagnose` command currently checks, in order:

1. configuration;
2. video/audio decoding;
3. DNS resolution;
4. TCP port 443;
5. WebSocket authentication and `session.created`;
6. `session.update` and `session.updated`;
7. one audio chunk plus one image frame and `input_audio_buffer.committed`.

Reports are saved under `data/output/diagnostics/`.

### Continuous state trajectory and intervention timing

`config/intervention_policy.yaml` and `src/coregulation_poc/control/`

- Maps the final four research states to no intervention, observation, explicit intervention, or progressive support.
- Uses no fixed time or count threshold and never treats one signal as an intervention trigger.
- Holds dysregulation/high-risk actions until a natural turn boundary.
- Requires an observable post-intervention response before another intervention decision.
- Requires interaction history before high-risk can enter progressive support.
- Saves ordered state points, decisions, research-basis identifiers, and recovery status.
- `trajectory-test` replays a JSON sequence of module-one assessments and writes auditable artifacts without calling an API.

## 4. Current test media

Local ignored input:

```text
data/input/P01_test_01.mp4
```

Verified properties:

- duration: 37.988 seconds;
- resolution: 1280×720;
- frame rate: 30 fps;
- video codec: H.264;
- audio codec: AAC at 44.1 kHz;
- replay output: 380 PCM audio chunks and 38 JPEG frames.

The clip ends immediately before expert intervention and contains preceding interaction context. It is intentionally excluded from Git.

## 5. What has been verified

The latest diagnostic run (2026-08-02 10:22 Asia/Shanghai) passed all seven checks. All five fresh-connection `session.update` variants, from `session: {}` through the full manual-mode configuration, returned `session.updated`. One audio chunk and one image frame were committed successfully.

Latest successful diagnostic report:

```text
data/output/diagnostics/20260802T022009Z_P01_test_01.json
```

A full schema-version-2 paid video inference also completed successfully and produced a locally valid `fluctuation` assessment with per-modality evidence, high confidence, and a 593 ms first-response latency:

```text
data/output/runs/20260802T022024Z_P01_test_01_787141c2
```

The model returned audio evidence (verbatim quotes of parent speech) and video evidence (labeled-frame observations of child hesitation and parental guidance), classifying the clip as `fluctuation`. The audit warning `audio_quote_not_found_in_best_effort_transcript` indicates that input ASR did not produce a matching transcript for cross-validation, so `audit_ready` is false.

## 6. Active unresolved issue

There is no active transport or model-access failure. The IPv6 timeout that briefly blocked WebSocket connections on 2026-08-02 has been resolved by forcing IPv4 in `websocket_transport.py`.

The current open validation question is whether the schema-version-2 assessment quality holds across more research clips: verbatim audio quotes, labeled-frame visual evidence when observable, per-modality insufficiency when not observable, and explicit uncertainty near state boundaries. The single successful paid run produced a structurally valid assessment, but `audit_ready` is false because input ASR did not produce a cross-validatable transcript.

## 7. Required next action

Run one paid inference of `P01_test_01.mp4` with schema version 2, then inspect:

1. `assessment.json` for per-modality evidence, confidence, alternative state, and ambiguity reason;
2. `audit.json` for transcription status, quote matching, and audit readiness;
3. `metrics.json` for separate `classification_valid` and `audit_ready` values;
4. the original clip at every cited evidence interval.

Do not batch-process research media until this single-run schema and audit behavior are accepted by the research team.

## 8. Commands

Run all local checks:

```powershell
python -m ruff check .
python -m pytest tests/
```

Run full diagnostic:

```powershell
python -m coregulation_poc diagnose --video "data/input/P01_test_01.mp4"
```

Run real inference only after all diagnostic checks pass:

```powershell
python -m coregulation_poc video-test --video "data/input/P01_test_01.mp4" --session-id P01_test_01
```

## 9. Current verification status

Current local verification:

```text
ruff: all checks passed
pytest: 54 passed
diagnostic: 7/7 passed (2026-08-02 10:22 Asia/Shanghai)
schema-v2 paid inference: valid, state=fluctuation, audit_ready=false
Qwen realtime TTS paid synthesis: valid, model=qwen3-tts-instruct-flash-realtime-2026-01-22, voice=Maia
```

Latest successful schema-v2 paid inference artifacts:

```text
data/output/runs/20260802T022024Z_P01_test_01_787141c2
```

A valid `assessment.json` indicates structural and contextual validity, not expert-confirmed classification accuracy. `audit_ready` must also be checked before treating evidence as fully auditable.

## 10. User communication preference

The user prefers concise Chinese responses that answer only the current question. Avoid broad redesigns, excessive branching, and long lists unless the user explicitly requests a complete plan. When debugging, report the concrete evidence first, then the single next action.

## 11. Local acoustic measurement update

Module one now includes a versioned Praat/Parselmouth supporting-measurement layer. It saves full-channel pitch, intensity, dBFS, and voiced-frame measurements to `acoustic_summary.json`, measures every model-cited audio interval in `acoustic_evidence.json`, and preserves Qwen ASR emotion labels separately in `input_emotions.json`. These observations never trigger a state by themselves.

The current mono replay channel mixes parent and child speech, so results are deliberately marked `quality=limited`, `actor=unknown`, and are not interpreted as an individual's emotion. Speech rate remains unavailable until time-aligned text and frontend voiceprint speaker segments exist. The analysis policy and limitations are versioned in `config/acoustic_analysis.yaml` and hashed in each run manifest.

Local verification on `P01_test_01.mp4` passed without an API call. The acoustic layer detected measurable speech and produced a plausible mixed-channel pitch distribution, but this is a pipeline check rather than an accuracy claim. The next ordinary paid inference should be used to confirm creation of `acoustic_evidence.json` and `input_emotions.json`; no dedicated extra paid run is required.

## 12. Module-three strategy selection update

Module three is now implemented in `src/coregulation_poc/intervention/` with a versioned 12-card library in `config/strategy_cards.yaml`. Cards explicitly target parent, child or both and record repair target, use/avoid conditions, approved template, expected recovery, next response-gated strategy and research provenance. Delivery modality is no longer stored in individual cards because module four now applies one versioned dual-channel contract.

The selector cannot override module two: it returns a held result unless the matching decision has both `intervention_permitted=true` and `strategy_selection_required=true`. Actor-specific cards require evidence explicitly assigned to that actor. Unknown or dyadic evidence can use a neutral dyadic fallback but cannot be used to single out one person.

`strategy-test` replays the same observation format through modules two and three without an API call. The verified example is `examples/strategy_replay.json`; its run selected `PARENT_TONE_AND_PACE`, targeted the parent, passed every wording check, and recorded observable recovery indicators.

## 13. Module-four dual-channel delivery update

Module four is implemented in `src/coregulation_poc/delivery/` with the versioned policy `config/delivery_policy.yaml`. It implements the research-team decision to show one prominent non-blocking text prompt and automatically speak the same Chinese message. It preserves the module-three target actor and cannot alter state, timing, repair target, strategy or wording.

The coordinator holds both outputs when intervention is paused. If voice is disabled or unavailable, it retains the visual prompt, marks the package `degraded`, and preserves the failure reason. Visual rendering and voice playback use separate execution records; a successful output is explicitly not treated as evidence that a participant saw, heard, understood or adopted the intervention.

`delivery-test` replays modules one to four without an API call and saves `delivery_policy.json`, `delivery_packages.json`, an empty `delivery_execution_reports.json` contract for the future frontend, and one standalone HTML preview per prepared intervention. `delivery-test --synthesize-voice` uses `qwen3-tts-instruct-flash-realtime-2026-01-22` with `Maia`, saves a hashed 24 kHz mono WAV plus sanitized TTS events and synthesis metadata, and binds the preview to that exact file instead of browser Web Speech. The completed local suite reports `ruff: all checks passed` and `pytest: 54 passed`. A paid synthesis on 2026-08-03 completed successfully with one Maia WAV, zero synthesis failures, 562 ms first-audio latency and 1,734 ms total synthesis latency. Production UI integration remains later work rather than a module-four logic gap.
