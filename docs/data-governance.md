# Data Governance

- Obtain consent before capturing audio, video, transcripts, or voiceprints.
- Store raw participant media outside Git and restrict access to the research team.
- Use pseudonymous session identifiers in generated evidence and logs.
- Keep raw API responses only when required for evaluation and define a deletion schedule.
- Never send identifying participant metadata in model prompts.
- Review cloud-region, retention, and subprocessors before conducting the formal study.
- Live device dry-runs do not save raw camera frames, microphone PCM, JPEG payloads or Base64 content. They save only media type, monotonic timestamp, byte count, bounded-queue metrics, non-sensitive device display information and a hash fingerprint of the raw hardware identifier.
- Device-format parameters are recorded for reproducibility but remain engineering settings until validated. Browser experiments save one complete activity recording only after explicit participant consent; access, retention and deletion must follow the approved research protocol.
- Browser capture requires an explicit page action and browser permission. A checkbox in the technical UI does not replace an ethics-approved consent process.
- Browser capture run artifacts exclude inference PCM/JPEG, device IDs, device labels, remote IP addresses, access codes and API credentials. They include one consented WebM/MP4 activity recording plus timing, integrity and reliability metadata.
- Non-local browser capture requires a server-side experiment access token and same-origin WebSocket. Public deployment must use HTTPS/WSS and should replace the shared technical-test token with expiring participant-specific invitations before recruitment.
