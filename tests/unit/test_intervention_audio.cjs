const { test } = require("node:test");
const assert = require("node:assert/strict");
const { PcmInterventionPlayer } = require("../../src/coregulation_poc/web/static/intervention-audio.js");

function setup() {
  const nodes = [];
  const context = {
    currentTime: 1, state: "running", destination: {},
    createBuffer: (_, n, rate) => ({ duration: n / rate, getChannelData: () => new Float32Array(n) }),
    createBufferSource: () => {
      const node = { connect() {}, disconnect() {}, stop() { this.stopped = true; },
        start(when) { this.when = when; }, end() { this.onended?.(); } };
      nodes.push(node);
      return node;
    },
  };
  return { player: new PcmInterventionPlayer(context, () => 1000), nodes };
}
function chunk(offset, n = 160) {
  return { audio_base64: Buffer.alloc(n * 2).toString("base64"), sample_offset: offset,
    sample_rate_hz: 16000 };
}
function wav(n) {
  const b = Buffer.alloc(44 + n * 2);
  b.write("RIFF"); b.writeUInt32LE(b.length - 8, 4); b.write("WAVEfmt ", 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(16000, 24); b.writeUInt32LE(32000, 28); b.writeUInt16LE(2, 32);
  b.writeUInt16LE(16, 34); b.write("data", 36); b.writeUInt32LE(n * 2, 40);
  return b.toString("base64");
}

test("starts before final audio, ignores duplicate chunks, finishes after playback", async () => {
  const { player, nodes } = setup();
  assert.equal(player.append(chunk(0)), true);
  assert.equal(nodes.length, 1);
  assert.equal(player.append(chunk(0)), false);
  player.append(chunk(160));
  assert.ok(nodes[1].when >= nodes[0].when + nodes[0].buffer.duration);
  player.finish({ audio_sample_count: 320 });
  assert.equal(player.settled, false);
  nodes[0].end(); nodes[1].end();
  assert.equal((await player.result).status, "delivered");
  player.finish({ audio_sample_count: 320 });
  assert.equal(nodes.length, 2);
});

test("reconnect gap repaired from full WAV without replaying received prefix", async () => {
  const { player, nodes } = setup();
  player.append(chunk(0));
  assert.equal(player.append(chunk(320)), false);
  player.finish({ audio_sample_count: 480, audio_base64: wav(480) });
  assert.equal(nodes.length, 2);
  assert.equal(nodes[1].buffer.duration, 320 / 16000);
  nodes.forEach(n => n.end());
  assert.equal((await player.result).status, "delivered");
});

test("whole WAV works when all streaming chunks were offline", async () => {
  const { player, nodes } = setup();
  player.finish({ audio_sample_count: 480, audio_base64: wav(480) });
  assert.equal(nodes.length, 1);
  nodes[0].end();
  assert.equal((await player.result).status, "delivered");
});

test("mid-stream failure and mute stop queued audio without successful receipt", async () => {
  for (const mute of [false, true]) {
    const { player, nodes } = setup();
    player.append(chunk(0));
    if (mute) player.cancel();
    else player.finish({ voice_error: "upstream timeout" });
    assert.equal((await player.result).status, mute ? "not_attempted" : "failed");
    assert.equal(nodes[0].stopped, true);
    assert.equal(player.append(chunk(160)), false);
  }
});

test("invalid PCM and incomplete WAV are failures, not phantom playback", async () => {
  const { player } = setup();
  player.append({ ...chunk(0), audio_base64: "AA==" });
  assert.equal((await player.result).status, "failed");
  const other = setup();
  other.player.finish({ audio_sample_count: 480, audio_base64: wav(160) });
  assert.equal((await other.player.result).status, "failed");
});
