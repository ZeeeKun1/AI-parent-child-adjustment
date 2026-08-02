# Module Four: Prominent Text and Spoken Voice Delivery

Module four implements the research team's 2026-08-02 decision to deliver one intervention through two simultaneous representations: a prominent on-screen text prompt and spoken voice. The decision preserves the expert voice intervention used in the formative study and WOZ experiment, while retaining text to reduce missed or misunderstood content.

## Responsibility boundary

Module two decides whether the current moment permits intervention. Module three decides the strategy, target actor and exact constrained message. Module four may only present that message; it cannot reclassify the state, change the target, select another strategy or rewrite the content.

The visual and spoken representations therefore contain the same core message. The target actor remains an explicit data field even though both people may be able to see the screen or hear the speaker in the shared physical setting.

## Output contract

`config/delivery_policy.yaml` requires:

- prominent visual text that does not block the primary task;
- automatic Chinese speech output;
- identical core content across text and voice;
- a complete hold when the user pauses intervention;
- visual fallback with an explicit degraded status when voice is unavailable;
- separate logging of visual rendering and voice playback.

No fixed display duration or forced acknowledgement threshold is introduced because these values are not established by the current formative evidence. The frontend may allow dismissal and must preserve the system's response-gated rule before another intervention.

## Audit boundary

A successful visual render means only that the frontend displayed the prompt. A successful voice result means only that playback occurred. Neither result proves that the parent or child saw, heard, understood or adopted the intervention. User acknowledgement and subsequent parent-child response are separate optional observations, and recovery remains a module-two judgment based on later interaction evidence.

## Fixed experiment voice

The voice path uses Alibaba Cloud Qwen realtime TTS with the fixed snapshot `qwen3-tts-instruct-flash-realtime-2026-01-22` and the fixed system voice `Maia`. The approved module-three message is sent through the text buffer unchanged. A versioned calm, neutral and supportive delivery instruction controls expression, while `optimize_instructions=false` prevents the service from rewriting that experiment instruction. The same voice is retained for parent-, child- and dyad-targeted cards so voice identity is not confounded with intervention target.

PCM output is wrapped as 24 kHz mono 16-bit WAV. Each synthesis record stores the source message, message hash, model, voice, instruction, audio hash, duration, character usage and latency. API keys and base64 audio events are never written to the audit log.

## Verification paths

`delivery-test` replays the existing trajectory input through modules two, three and four without an API call. It saves the versioned policies, decisions, plans, delivery packages and an empty execution-report collection for the future frontend to populate. The standalone HTML preview shows the alert but explicitly reports that no Maia audio has been generated.

`delivery-test --synthesize-voice` additionally calls the fixed Qwen model, saves `delivery_audio_001.wav` and sanitized `tts_events_001.json`, and binds that exact file to the preview. Generating the file is still not recorded as playback: the future frontend must separately report whether visual rendering and audio playback actually occurred.
