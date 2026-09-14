import asyncio
from types import SimpleNamespace

from coregulation_poc.web.app import _RuntimeEventRelay


def test_offline_pcm_does_not_evict_text_or_complete_audio_or_store_raw_payload():
    async def exercise():
        audited, observed, replayed = [], [], []

        async def observe(session_id, payload):
            observed.append(payload)

        async def send(payload):
            replayed.append(payload)

        relay = _RuntimeEventRelay(
            session_id="test",
            recorder=SimpleNamespace(store=SimpleNamespace(append_event=audited.append)),
            registry=SimpleNamespace(observe=observe),
        )
        await relay.send({"type": "intervention", "delivery_id": "test:1"})
        for index in range(100):
            await relay.send({"type": "intervention_audio_chunk", "chunk_index": index,
                              "audio_base64": "AAAA"})
        await relay.send({"type": "intervention_voice", "audio_base64": "AAAA"})
        assert await relay.attach(send)
        assert [e["type"] for e in replayed] == ["intervention", "intervention_voice"]
        assert len(observed) == 2
        assert len(audited) == 102
        assert all("audio_base64" not in row["event"] for row in audited)
        await relay.close()

    asyncio.run(exercise())
