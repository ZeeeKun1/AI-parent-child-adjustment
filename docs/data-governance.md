# Data Governance

- Obtain consent before capturing audio, video, transcripts, or voiceprints.
- Store raw participant media outside Git and restrict access to the research team.
- Use pseudonymous session identifiers in generated evidence and logs.
- Keep raw API responses only when required for evaluation and define a deletion schedule.
- Never send identifying participant metadata in model prompts.
- Review cloud-region, retention, and subprocessors before conducting the formal study.
- Live device dry-runs do not save raw camera frames, microphone PCM, JPEG payloads or Base64 content. They save only media type, monotonic timestamp, byte count, bounded-queue metrics, non-sensitive device display information and a hash fingerprint of the raw hardware identifier.
- Device-format parameters are recorded for reproducibility but remain engineering settings until validated. Raw live media may only be recorded through a separate, explicitly approved research-recording workflow covered by participant consent.
