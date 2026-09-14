/* PCM playback with contiguous offsets and complete-WAV repair after reconnect. */
class PcmInterventionPlayer {
  constructor(context, clock, onStart = () => {}) {
    this.context = context;
    this.clock = clock;
    this.onStart = onStart;
    this.receivedSamples = 0;
    this.sampleRate = null;
    this.tail = 0;
    this.sources = new Set();
    this.startedAt = null;
    this.inputFinished = false;
    this.settled = false;
    this.result = new Promise((resolve) => { this.resolve = resolve; });
    this.timer = setTimeout(() => this.cancel("failed", "语音播放超时"), 45000);
  }

  static bytes(encoded) {
    const binary = atob(encoded);
    return Uint8Array.from(binary, (char) => char.charCodeAt(0));
  }

  append(message) {
    if (this.settled || this.inputFinished) return false;
    // Duplicate and out-of-order chunks are never played; finish() repairs gaps.
    if (message.sample_offset !== this.receivedSamples) return false;
    try {
      const bytes = PcmInterventionPlayer.bytes(message.audio_base64);
      this.schedule(bytes, message.sample_rate_hz);
      return true;
    } catch (error) {
      this.cancel("failed", error.message);
      return false;
    }
  }

  schedule(bytes, rate) {
    if (!Number.isInteger(rate) || rate < 8000 || rate > 96000 || bytes.length % 2) {
      throw new Error("语音数据格式不正确");
    }
    if (this.sampleRate !== null && this.sampleRate !== rate) throw new Error("语音采样率变化");
    if (!bytes.length) return;
    if (this.receivedSamples + bytes.length / 2 > rate * 120) throw new Error("语音过长");
    this.sampleRate = rate;
    const buffer = this.context.createBuffer(1, bytes.length / 2, rate);
    const data = buffer.getChannelData(0);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < data.length; i += 1) data[i] = view.getInt16(i * 2, true) / 32768;
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    const when = Math.max(this.context.currentTime + 0.06, this.tail);
    this.tail = when + buffer.duration;
    this.sources.add(source);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
      this.maybeComplete();
    };
    source.start(when);
    this.receivedSamples += data.length;
    if (this.startedAt === null) {
      this.startedAt = Math.round(this.clock() + (when - this.context.currentTime) * 1000);
      this.onStart();
    }
    if (this.context.state === "suspended") {
      this.context.resume().catch(() => this.cancel("failed", "浏览器未允许语音播放"));
    }
  }

  static wav(encoded) {
    const bytes = PcmInterventionPlayer.bytes(encoded);
    const view = new DataView(bytes.buffer);
    const tag = (offset) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
    if (bytes.length < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("语音文件格式不正确");
    let rate = null;
    let pcm = null;
    for (let offset = 12; offset + 8 <= bytes.length;) {
      const length = view.getUint32(offset + 4, true);
      const start = offset + 8;
      if (start + length > bytes.length) throw new Error("语音文件不完整");
      if (tag(offset) === "fmt ") {
        if (length < 16 || view.getUint16(start, true) !== 1
          || view.getUint16(start + 2, true) !== 1 || view.getUint16(start + 14, true) !== 16) {
          throw new Error("语音必须为单声道 PCM16");
        }
        rate = view.getUint32(start + 4, true);
      } else if (tag(offset) === "data") pcm = bytes.subarray(start, start + length);
      offset = start + length + (length % 2);
    }
    if (!rate || !pcm) throw new Error("语音文件缺少音频");
    return { rate, pcm };
  }

  finish(message) {
    if (this.settled || this.inputFinished) return this.result;
    try {
      if (message.voice_error) throw new Error(message.voice_error);
      const expected = message.audio_sample_count;
      if (!Number.isInteger(expected) || expected <= 0) throw new Error("语音内容为空");
      if (this.receivedSamples > expected) throw new Error("语音片段长度不一致");
      if (this.receivedSamples < expected) {
        const { rate, pcm } = PcmInterventionPlayer.wav(message.audio_base64);
        if (pcm.length / 2 !== expected) throw new Error("完整语音长度不一致");
        // Keep already queued samples and append only the missing suffix.
        this.schedule(pcm.subarray(this.receivedSamples * 2), rate);
      }
      this.inputFinished = true;
      this.maybeComplete();
    } catch (error) {
      this.cancel("failed", error.message);
    }
    return this.result;
  }

  maybeComplete() {
    if (!this.settled && this.inputFinished && this.sources.size === 0) {
      this.settle({ status: "delivered", started_at_ms: this.startedAt,
        completed_at_ms: Math.round(this.clock()) });
    }
  }

  settle(result) {
    if (this.settled) return;
    this.settled = true;
    clearTimeout(this.timer);
    this.resolve(result);
  }

  cancel(status = "not_attempted", error = null) {
    this.settle({ status, ...(this.startedAt === null ? {} : { started_at_ms: this.startedAt }),
      ...(error ? { error } : {}) });
    for (const source of this.sources) {
      source.onended = null;
      try { source.stop(); } catch (_) { /* already ended */ }
      source.disconnect();
    }
    this.sources.clear();
  }
}

if (typeof module !== "undefined") module.exports = { PcmInterventionPlayer };
