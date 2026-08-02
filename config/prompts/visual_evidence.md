# Visual evidence extraction

Analyze only observable parent-child homework behavior in the supplied frames.
Do not diagnose emotion, personality, parenting quality, or family relationships.

Return JSON containing:

- actor: parent, child, both, or unknown
- start_ms and end_ms
- behavior_codes drawn from the study codebook
- concise observable description
- evidence_sufficiency: sufficient or insufficient

Examples of observable evidence include leaving the frame, stopping writing,
turning away from the task, repeated visible movement, task takeover, and continued writing.

