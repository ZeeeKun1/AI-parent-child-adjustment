const PROTOCOL_VERSION = 1;
const AUDIO_PACKET = 1;
const IMAGE_PACKET = 2;
const AUDIO_BACKPRESSURE_BYTES = 4 * 1024 * 1024;
const IMAGE_BACKPRESSURE_BYTES = 1024 * 1024;

const elements = {
  consent: document.querySelector("#consent"),
  start: document.querySelector("#start-button"),
  stop: document.querySelector("#stop-button"),
  status: document.querySelector("#status"),
  video: document.querySelector("#preview"),
  audioCount: document.querySelector("#audio-count"),
  imageCount: document.querySelector("#image-count"),
  elapsed: document.querySelector("#elapsed"),
  session: document.querySelector("#session-code"),
  summary: document.querySelector("#summary"),
  accessCode: document.querySelector("#access-code"),
};

let socket = null;
let mediaStream = null;
let audioContext = null;
let audioNode = null;
let muteNode = null;
let imageTimer = null;
let elapsedTimer = null;
let startedAt = 0;
let captureActive = false;
let stopping = false;
let nextAudioTimestampMs = 0;
let audioCount = 0;
let imageCount = 0;
let droppedImages = 0;
let audioBackpressureStops = 0;
let negotiated = null;
let imageCaptureBusy = false;

function makeSessionCode() {
  const randomPart = crypto.randomUUID
    ? crypto.randomUUID().replaceAll("-", "").slice(0, 12)
    : Array.from(crypto.getRandomValues(new Uint8Array(6)), (value) =>
      value.toString(16).padStart(2, "0")).join("");
  return `web_${randomPart}`;
}

function setStatus(message, tone = "neutral") {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
}

function updateCounters() {
  elements.audioCount.textContent = String(audioCount);
  elements.imageCount.textContent = String(imageCount);
  if (captureActive) {
    elements.elapsed.textContent = `${Math.round(performance.now() - startedAt)} ms`;
  }
}

function websocketUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/live`;
}

function waitForSocketOpen(ws, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error("连接服务器超时")), timeoutMs);
    ws.addEventListener("open", () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
    ws.addEventListener("error", () => {
      window.clearTimeout(timeout);
      reject(new Error("无法连接实时采集服务器"));
    }, { once: true });
  });
}

function waitForMessage(ws, acceptedTypes, timeoutMs = 10000) {
  const allowed = new Set(acceptedTypes);
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("等待服务器响应超时"));
    }, timeoutMs);
    const onMessage = (event) => {
      if (typeof event.data !== "string") return;
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (!allowed.has(message.type)) return;
      cleanup();
      if (message.type === "error") {
        reject(new Error(message.message || "服务器拒绝了采集请求"));
      } else {
        resolve(message);
      }
    };
    const onClose = () => {
      cleanup();
      reject(new Error("服务器连接已关闭"));
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
    };
    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose, { once: true });
  });
}

function encodePacket(packetType, timestampMs, payload) {
  const payloadBytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const packet = new ArrayBuffer(9 + payloadBytes.byteLength);
  const view = new DataView(packet);
  view.setUint8(0, packetType);
  view.setBigUint64(1, BigInt(Math.max(0, Math.round(timestampMs))), false);
  new Uint8Array(packet, 9).set(payloadBytes);
  return packet;
}

async function prepareMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: negotiated.media_format.image_max_width },
      height: { ideal: negotiated.media_format.image_max_height },
      facingMode: "user",
    },
    audio: {
      channelCount: { ideal: 1 },
      echoCancellation: { ideal: false },
      noiseSuppression: { ideal: false },
      autoGainControl: { ideal: false },
    },
  });
  elements.video.srcObject = mediaStream;
  await elements.video.play();

  audioContext = new AudioContext();
  if (audioContext.state === "suspended") await audioContext.resume();
  await audioContext.audioWorklet.addModule("/static/audio-worklet.js");
  const source = audioContext.createMediaStreamSource(mediaStream);
  audioNode = new AudioWorkletNode(audioContext, "pcm16-chunk-processor", {
    processorOptions: {
      targetSampleRate: negotiated.media_format.audio_sample_rate,
      chunkSamples: negotiated.media_format.audio_chunk_bytes / 2,
    },
  });
  muteNode = audioContext.createGain();
  muteNode.gain.value = 0;
  source.connect(audioNode).connect(muteNode).connect(audioContext.destination);
  audioNode.port.onmessage = (event) => {
    if (!captureActive || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > AUDIO_BACKPRESSURE_BYTES) {
      audioBackpressureStops += 1;
      void stopCapture(false, "网络积压过大，为避免静默丢失音频，采集已经停止");
      return;
    }
    socket.send(encodePacket(AUDIO_PACKET, nextAudioTimestampMs, event.data));
    nextAudioTimestampMs += negotiated.media_format.audio_chunk_ms;
    audioCount += 1;
    updateCounters();
  };
}

async function captureImage() {
  if (!captureActive || imageCaptureBusy || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  if (socket.bufferedAmount > IMAGE_BACKPRESSURE_BYTES) {
    droppedImages += 1;
    return;
  }
  const sourceWidth = elements.video.videoWidth;
  const sourceHeight = elements.video.videoHeight;
  if (!sourceWidth || !sourceHeight) return;

  imageCaptureBusy = true;
  try {
    const maxWidth = negotiated.media_format.image_max_width;
    const maxHeight = negotiated.media_format.image_max_height;
    const scale = Math.min(1, maxWidth / sourceWidth, maxHeight / sourceHeight);
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(41, Math.round(sourceHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(elements.video, 0, 0, width, height);
    const timestampMs = Math.max(0, Math.round(performance.now() - startedAt));
    context.fillStyle = "rgba(0, 0, 0, 0.78)";
    context.fillRect(0, height - 40, width, 40);
    context.fillStyle = "#ffffff";
    context.font = "600 20px system-ui, sans-serif";
    context.fillText(`frame_time_ms=${timestampMs}`, 12, height - 13);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
    if (!blob) throw new Error("浏览器无法生成 JPEG 图像帧");
    if (blob.size > negotiated.max_image_bytes) {
      droppedImages += 1;
      return;
    }
    const payload = await blob.arrayBuffer();
    socket.send(encodePacket(IMAGE_PACKET, timestampMs, payload));
    imageCount += 1;
    updateCounters();
  } finally {
    imageCaptureBusy = false;
  }
}

function releaseMedia() {
  captureActive = false;
  if (imageTimer !== null) window.clearInterval(imageTimer);
  if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
  imageTimer = null;
  elapsedTimer = null;
  if (audioNode) audioNode.disconnect();
  if (muteNode) muteNode.disconnect();
  audioNode = null;
  muteNode = null;
  if (audioContext) void audioContext.close();
  audioContext = null;
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.video.srcObject = null;
}

async function startCapture() {
  if (!elements.consent.checked) {
    setStatus("请先确认已理解采集说明。", "warning");
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("当前页面不是安全连接。请通过 HTTPS 或服务器本机 localhost 打开。", "error");
    return;
  }

  elements.start.disabled = true;
  elements.summary.hidden = true;
  setStatus("正在连接服务器…", "working");
  try {
    socket = new WebSocket(websocketUrl());
    socket.binaryType = "arraybuffer";
    await waitForSocketOpen(socket);
    socket.send(JSON.stringify({
      type: "hello",
      protocol_version: PROTOCOL_VERSION,
      session_id: elements.session.textContent,
      access_token: elements.accessCode.value,
      capabilities: {
        audio_worklet: Boolean(window.AudioWorkletNode),
        media_devices: Boolean(navigator.mediaDevices),
        secure_context: window.isSecureContext,
        page_version: "0.1.0",
      },
    }));
    negotiated = await waitForMessage(socket, ["ready", "error"]);
    setStatus("请在浏览器提示中允许摄像头和麦克风。", "working");
    await prepareMedia();
    socket.send(JSON.stringify({ type: "start" }));
    await waitForMessage(socket, ["started", "error"]);

    startedAt = performance.now();
    nextAudioTimestampMs = 0;
    audioCount = 0;
    imageCount = 0;
    droppedImages = 0;
    audioBackpressureStops = 0;
    captureActive = true;
    stopping = false;
    elements.stop.disabled = false;
    setStatus("正在采集。原始音视频不会保存在服务器。", "success");
    await captureImage();
    imageTimer = window.setInterval(
      () => void captureImage(),
      negotiated.media_format.image_interval_ms,
    );
    elapsedTimer = window.setInterval(updateCounters, 250);
  } catch (error) {
    releaseMedia();
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    elements.start.disabled = false;
    elements.stop.disabled = true;
    setStatus(error instanceof Error ? error.message : "无法开始采集", "error");
  }
}

async function stopCapture(normal = true, reason = null) {
  if (stopping) return;
  stopping = true;
  releaseMedia();
  elements.stop.disabled = true;
  setStatus(normal ? "正在结束并生成检查结果…" : reason, normal ? "working" : "error");

  try {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: normal ? "stop" : "abort",
        reason,
        client_metrics: {
          dropped_images: droppedImages,
          audio_backpressure_stops: audioBackpressureStops,
          capture_duration_ms: Math.max(0, Math.round(performance.now() - startedAt)),
        },
      }));
      const summary = await waitForMessage(socket, ["summary", "error"]);
      elements.summary.hidden = false;
      elements.summary.textContent = summary.valid
        ? `采集检查通过：${summary.audio_chunk_count} 个音频块，${summary.image_chunk_count} 个图像帧。`
        : `采集未通过：${summary.audio_chunk_count || 0} 个音频块，${summary.image_chunk_count || 0} 个图像帧。`;
      setStatus(summary.valid ? "采集已安全结束。" : "采集结束，但媒体检查未通过。", summary.valid ? "success" : "warning");
      socket.close();
    }
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "结束采集时发生错误", "error");
  } finally {
    socket = null;
    negotiated = null;
    elements.session.textContent = makeSessionCode();
    elements.start.disabled = false;
    stopping = false;
  }
}

elements.session.textContent = makeSessionCode();
elements.start.addEventListener("click", () => void startCapture());
elements.stop.addEventListener("click", () => void stopCapture(true));
window.addEventListener("pagehide", releaseMedia);
updateCounters();
