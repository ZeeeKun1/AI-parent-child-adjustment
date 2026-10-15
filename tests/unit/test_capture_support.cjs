const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const support = require("../../src/coregulation_poc/web/static/capture-support.js");
const app = fs.readFileSync(path.join(__dirname, "../../src/coregulation_poc/web/static/app.js"), "utf8");

test("muted and disabled live tracks are not healthy devices", () => {
  assert.equal(support.trackAvailable({ readyState: "live", enabled: true, muted: true }), false);
  assert.equal(support.trackAvailable({ readyState: "live", enabled: false, muted: false }), false);
  assert.equal(support.trackAvailable({ readyState: "ended", enabled: true, muted: false }), false);
  assert.equal(support.trackAvailable({ readyState: "live", enabled: true, muted: false }), true);
});

test("enrollment requests only the selected microphone and never forces a hardware rate", async () => {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const oldStream = globalThis.MediaStream;
  const oldSecure = globalThis.isSecureContext;
  const calls = [];
  const track = { stop() {} };
  try {
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: {
      getUserMedia: async (constraints) => { calls.push(constraints); return { getTracks: () => [track] }; },
    } } });
    globalThis.isSecureContext = true;
    globalThis.MediaStream = class { constructor(tracks) { this.tracks = tracks; } };
    await support.acquire({ audioDevice: "chosen-mic" });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].video, undefined);
    assert.deepEqual(calls[0].audio.deviceId, { exact: "chosen-mic" });
    assert.equal(calls[0].audio.sampleRate, undefined);
  } finally {
    if (oldNavigator) Object.defineProperty(globalThis, "navigator", oldNavigator);
    else delete globalThis.navigator;
    globalThis.MediaStream = oldStream;
    globalThis.isSecureContext = oldSecure;
  }
});

test("an occupied camera is identified separately and the acquired mic is released", async () => {
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const oldSecure = globalThis.isSecureContext;
  let stopped = false;
  try {
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: {
      getUserMedia: async ({ audio }) => {
        if (audio) return { getTracks: () => [{ stop() { stopped = true; } }] };
        throw Object.assign(new Error("device busy"), { name: "NotReadableError" });
      },
    } } });
    globalThis.isSecureContext = true;
    await assert.rejects(support.acquire({ video: true }), (error) => error.device === "camera" && error.message.includes("占用"));
    assert.equal(stopped, true);
  } finally {
    if (oldNavigator) Object.defineProperty(globalThis, "navigator", oldNavigator);
    else delete globalThis.navigator;
    globalThis.isSecureContext = oldSecure;
  }
});

test("native 16k, 44.1k and 48k capture preserve the existing 16k PCM packet format", () => {
  const worklet = fs.readFileSync(path.join(__dirname, "../../src/coregulation_poc/web/static/audio-worklet.js"), "utf8");
  for (const rate of [16000, 44100, 48000]) {
    let Processor;
    const packets = [];
    const sandbox = { sampleRate: rate,
      AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (b) => packets.push(b) }; } },
      registerProcessor: (_, cls) => { Processor = cls; } };
    vm.createContext(sandbox); vm.runInContext(worklet, sandbox);
    const processor = new Processor({ processorOptions: { targetSampleRate: 16000, chunkSamples: 1600 } });
    for (let i = 0; i < rate; i += 128) {
      const input = Float32Array.from({ length: Math.min(128, rate - i) }, (_, n) => Math.sin((i + n) * 2 * Math.PI * 300 / rate) * 0.1);
      processor.process([[input]]);
    }
    assert.ok(packets.length >= 9 && packets.length <= 10);
    assert.ok(packets.every((b) => b.byteLength === 3200));
    assert.ok(support.pcmLevels(packets[0]).rms > 0.06);
  }
});

test("device check rejects zero PCM, accepts speech and releases its audio graph", async () => {
  const source = app.slice(app.indexOf("async function verifyMicrophoneStream("), app.indexOf("function deviceAccessMessage("));
  for (const rms of [0, 0.02]) {
    let disconnected = false;
    let closed = false;
    const sandbox = { window: { setTimeout, clearTimeout }, setTimeout, clearTimeout,
      CaptureSupport: { createContext: () => ({ close: async () => { closed = true; } }), trackAvailable: () => true,
        createPcmCapture: async (_, __, { onChunk }) => {
          queueMicrotask(() => onChunk(new ArrayBuffer(3200), { rms, peak: rms }));
          return { disconnect: () => { disconnected = true; } };
        } }, stream: { getAudioTracks: () => [{}] } };
    vm.createContext(sandbox);
    const result = vm.runInContext(source + "\nverifyMicrophoneStream(stream, 10);", sandbox);
    if (rms) await result;
    else await assert.rejects(result, /没有检测到清晰声音/);
    assert.equal(disconnected && closed, true);
  }
});

test("runtime monitoring catches muted inputs and frozen decoded frames without treating silence as failure", () => {
  const source = app.slice(app.indexOf("function monitorCaptureHealth()"), app.indexOf("async function recoverCaptureDevices("));
  for (const scenario of ["muted", "frozen", "quiet"]) {
    const states = {};
    const sandbox = {
      captureActive: true, captureRecoveryBusy: false, captureRecoveryAttempts: 2,
      nextCaptureRecoveryAt: 0, lastAudioChunkAt: 39999, lastImageFrameAt: 39999,
      lastVideoDecodedAt: scenario === "frozen" ? 0 : 39999,
      lastNonzeroAudioAt: 0, zeroAudioSince: scenario === "quiet" ? 1 : 0,
      audioContext: { state: "running" }, CaptureSupport: support,
      performance: { now: () => 40000 }, negotiated: { media_format: { image_interval_ms: 1000 } },
      mediaStream: { getAudioTracks: () => [{ readyState: "live", enabled: true, muted: scenario === "muted" }],
        getVideoTracks: () => [{ readyState: "live", enabled: true, muted: false }] },
      document: { hidden: true, querySelector: () => ({}) }, hasCaptureDeviceError: () => false,
      setDeviceHealth: (device, state) => { states[device] = state; }, setStatus() {},
    };
    vm.createContext(sandbox); vm.runInContext(source + "\nmonitorCaptureHealth();", sandbox);
    assert.equal(states.microphone, scenario === "muted" ? "error" : "ok");
    assert.equal(states.camera, scenario === "frozen" ? "error" : "ok");
  }
});

test("failed recovery releases partially opened devices and allows another attempt", async () => {
  const source = app.slice(app.indexOf("async function recoverCaptureDevices("), app.indexOf("async function prepareMedia("));
  let stopped = 0;
  let closed = 0;
  const sandbox = {
    captureActive: true, stopping: false, captureRecoveryBusy: false, captureRecoveryAttempts: 0,
    nextCaptureRecoveryAt: 0, capturePipeline: null, mediaStream: null, audioContext: null,
    nextAudioTimestampMs: 0, performance: { now: () => 100 },
    document: { querySelector: () => ({}) }, CaptureSupport: { createContext: () => ({
      close: async () => { closed += 1; },
    }) }, setStatus() {}, sessionTimestampMs: () => 100,
    stopSessionRecording: async () => {}, deviceAccessMessage: (error) => error.message,
    prepareMedia: async () => {
      sandbox.mediaStream = { getTracks: () => [{ stop: () => { stopped += 1; } }] };
      throw new Error("camera initialization failed");
    },
  };
  vm.createContext(sandbox);
  await vm.runInContext(source + "\nrecoverCaptureDevices(true);", sandbox);
  assert.equal(stopped, 1);
  assert.equal(closed, 1);
  assert.equal(sandbox.mediaStream, null);
  assert.equal(sandbox.audioContext, null);
  assert.equal(sandbox.captureRecoveryBusy, false);
});
