/* Browser capture compatibility. Input stays native-rate; the wire format stays PCM16/16kHz. */
const CaptureSupport = (() => {
  function trackAvailable(track) {
    return Boolean(track && track.readyState === "live" && track.enabled && !track.muted);
  }

  function pcmLevels(buffer) {
    const samples = new Int16Array(buffer);
    let sum = 0;
    let peak = 0;
    for (const value of samples) {
      const normalized = value / 32768;
      sum += normalized * normalized;
      peak = Math.max(peak, Math.abs(normalized));
    }
    return { rms: samples.length ? Math.sqrt(sum / samples.length) : 0, peak };
  }

  function createContext() {
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Context) throw new Error("当前浏览器不支持声音采集，请使用最新版 Chrome、Edge 或 Safari。");
    return new Context();
  }

  function resumeContext(context, timeoutMs = 4000) {
    if (context.state === "running") return Promise.resolve();
    if (context.state === "closed") return Promise.reject(new Error("声音通道已关闭，请重新检查设备。"));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("声音通道尚未恢复，请回到网页后点击恢复设备。")), timeoutMs);
      Promise.resolve(context.resume()).then(() => {
        clearTimeout(timer);
        if (context.state !== "running") reject(new Error("声音通道被暂停，请回到网页后重试。"));
        else resolve();
      }, (error) => { clearTimeout(timer); reject(error); });
    });
  }

  function accessMessage(error, device = "设备") {
    const name = error?.name;
    if (name === "NotAllowedError" || name === "SecurityError") return `未获得${device}权限，请检查浏览器和系统权限，并使用 HTTPS 打开网页。`;
    if (name === "NotFoundError") return `未找到所选${device}，请重新选择或连接设备。`;
    if (name === "NotReadableError" || name === "AbortError") return `${device}暂时无法读取，可能被飞书等程序占用。请切换设备或释放占用后重试。`;
    if (name === "OverconstrainedError" || name === "NotSupportedError") return `${device}当前设置不受支持，请切换设备或使用最新版浏览器。`;
    return error?.message || `${device}检查失败，请重试。`;
  }

  // Audio and video are acquired separately so an occupied camera is not reported as a mic failure.
  async function acquire({ audio = true, video = false, audioDevice = "", videoDevice = "" } = {}) {
    if (!globalThis.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("请通过 HTTPS 或服务器本机 localhost 打开页面。");
    }
    const tracks = [];
    try {
      for (const kind of ["audio", "video"]) {
        if (!(kind === "audio" ? audio : video)) continue;
        const constraints = kind === "audio"
          ? { channelCount: { ideal: 1 }, echoCancellation: { ideal: false }, noiseSuppression: { ideal: false }, autoGainControl: { ideal: false } }
          : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" };
        const id = kind === "audio" ? audioDevice : videoDevice;
        if (id) constraints.deviceId = { exact: id };
        try {
          const stream = await new Promise((resolve, reject) => {
            let expired = false;
            const timer = setTimeout(() => {
              expired = true;
              reject(new Error("等待设备超时，请确认浏览器授权，检查飞书占用后重试。"));
            }, 30000);
            navigator.mediaDevices.getUserMedia({ [kind]: constraints }).then((stream) => {
              clearTimeout(timer);
              if (expired) stream.getTracks().forEach((track) => track.stop());
              else resolve(stream);
            }, (error) => { clearTimeout(timer); reject(error); });
          });
          tracks.push(...stream.getTracks());
        } catch (error) {
          const failure = new Error(accessMessage(error, kind === "audio" ? "麦克风" : "摄像头"));
          failure.name = error.name || "CaptureError";
          failure.device = kind === "audio" ? "microphone" : "camera";
          throw failure;
        }
      }
      return new MediaStream(tracks);
    } catch (error) {
      tracks.forEach((track) => track.stop());
      throw error;
    }
  }

  async function createPcmCapture(stream, context, { targetSampleRate = 16000, chunkSamples = 1600, onChunk = () => {} } = {}) {
    await resumeContext(context);
    if (!context.audioWorklet || !globalThis.AudioWorkletNode) {
      throw new Error("当前浏览器不支持实时声音采集，请在最新版 Chrome、Edge 或 Safari 中打开链接。");
    }
    await context.audioWorklet.addModule("/static/audio-worklet.js?v=2");
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "pcm16-chunk-processor", {
      processorOptions: { targetSampleRate, chunkSamples },
    });
    const mute = context.createGain();
    mute.gain.value = 0;
    node.port.onmessage = ({ data }) => {
      if (data instanceof ArrayBuffer && data.byteLength) onChunk(data, pcmLevels(data));
    };
    source.connect(node).connect(mute).connect(context.destination);
    return { node, source, mute, disconnect() {
      node.port.onmessage = null;
      node.disconnect(); source.disconnect(); mute.disconnect();
    } };
  }

  return { trackAvailable, pcmLevels, createContext, resumeContext, accessMessage, acquire, createPcmCapture };
})();

if (typeof module !== "undefined") module.exports = CaptureSupport;
