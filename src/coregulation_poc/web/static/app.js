const PROTOCOL_VERSION = 1;
const AUDIO_PACKET = 1;
const IMAGE_PACKET = 2;
const AUDIO_BACKPRESSURE_BYTES = 4 * 1024 * 1024;
const IMAGE_BACKPRESSURE_BYTES = 1024 * 1024;
const RECONNECT_DELAY_MS = 2000;
const VOICE_DELIVERY_TIMEOUT_MS = 12000;

const elements = {
  consent: document.querySelector("#consent"),
  start: document.querySelector("#start-button"),
  startCard: document.querySelector("#start-card"),
  sessionControls: document.querySelector("#session-controls"),
  pauseInterventions: document.querySelector("#pause-interventions-button"),
  toggleVoice: document.querySelector("#toggle-voice-button"),
  selfContinue: document.querySelector("#self-continue-button"),
  stop: document.querySelector("#stop-button"),
  status: document.querySelector("#status"),
  liveState: document.querySelector("#live-state"),
  video: document.querySelector("#preview"),
  bubbleText: document.querySelector("#bubble-text"),
  stageLabel: document.querySelector("#stage-label"),
  stageTitle: document.querySelector("#stage-title"),
  stageNote: document.querySelector("#stage-note"),
  stageMascot: document.querySelector("#stage-mascot"),
  stageTarget: document.querySelector("#stage-target"),
  stageTargetAvatars: document.querySelector("#stage-target-avatars"),
  stageTargetLabel: document.querySelector("#stage-target-label"),
  cameraEmpty: document.querySelector("#camera-empty"),
  audioCount: document.querySelector("#audio-count"),
  imageCount: document.querySelector("#image-count"),
  elapsed: document.querySelector("#elapsed"),
  session: document.querySelector("#session-code"),
  summary: document.querySelector("#summary"),
  parentAge: document.querySelector("#parent-age"),
  childAge: document.querySelector("#child-age"),
  taskName: document.querySelector("#task-name"),
  taskType: document.querySelector("#task-type"),
  taskDifficulty: document.querySelector("#task-difficulty"),
  childGrade: document.querySelector("#child-grade"),
  bindingCount: document.querySelector("#binding-count"),
  bindingFeedback: document.querySelector("#binding-feedback"),
  bindingPeople: Array.from(document.querySelectorAll(".binding-person")),
  bindingButtons: Array.from(document.querySelectorAll(".binding-button")),
  deviceCheckButton: document.querySelector("#device-check-button"),
  deviceCheckFeedback: document.querySelector("#device-check-feedback"),
  devicePreview: document.querySelector("#device-preview"),
  deviceStatuses: Array.from(document.querySelectorAll("[data-device-status]")),
  roleOptions: Array.from(document.querySelectorAll(".role-option")),
  parentBindingAvatar: document.querySelector("#parent-binding-avatar"),
  childBindingAvatar: document.querySelector("#child-binding-avatar"),
  parentRoleLabel: document.querySelector("#parent-role-label"),
  childRoleLabel: document.querySelector("#child-role-label"),
  startRequirement: document.querySelector("#start-requirement"),
  headerParentAvatar: document.querySelector("#header-parent-avatar"),
  headerChildAvatar: document.querySelector("#header-child-avatar"),
  headerFamilyLabel: document.querySelector("#header-family-label"),
  homeNav: document.querySelector("#home-nav"),
  recordNav: document.querySelector("#record-nav"),
  intervention: document.querySelector("#intervention"),
  interventionSource: document.querySelector("#intervention-source"),
  interventionTarget: document.querySelector("#intervention-target"),
  interventionHeading: document.querySelector("#intervention-heading"),
  interventionMessage: document.querySelector("#intervention-message"),
  interventionChannel: document.querySelector("#intervention-channel"),
  interventionTargetAvatars: document.querySelector("#intervention-target-avatars"),
  interventionMascot: document.querySelector("#intervention-mascot"),
  difficultyOptions: document.querySelector("#difficulty-options"),
  dismissIntervention: document.querySelector("#dismiss-intervention"),
  summaryPanel: document.querySelector("#session-summary"),
  summaryFamily: document.querySelector("#summary-family"),
  summaryDuration: document.querySelector("#summary-duration"),
  summaryInterventions: document.querySelector("#summary-interventions"),
  summaryPositive: document.querySelector("#summary-positive"),
  summaryImprovements: document.querySelector("#summary-improvements"),
  summarySupports: document.querySelector("#summary-supports"),
  newSession: document.querySelector("#new-session-button"),
  wizardTabs: Array.from(document.querySelectorAll(".wizard-tab")),
  wizardPanels: Array.from(document.querySelectorAll(".wizard-panel")),
  wizardPrev: document.querySelector("#wizard-prev"),
  wizardNext: document.querySelector("#wizard-next"),
};

let socket = null;
let mediaStream = null;
let audioContext = null;
let audioNode = null;
let muteNode = null;
let imageTimer = null;
let elapsedTimer = null;
let captureHealthTimer = null;
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
let interventionAudioUrl = null;
let interventionsPaused = false;
let voiceEnabled = true;
let currentDeliveryId = null;
const bindingState = { parent: false, child: false };
const BINDING_RECORDING_MS = 5000;
const BINDING_PROCESSING_TIMEOUT_MS = 20000;
let bindingBusy = false;
let deviceCheckBusy = false;
let checkedMediaStream = null;
const deviceHealth = { camera: "unchecked", microphone: "unchecked" };
let lastAudioChunkAt = 0;
let lastImageFrameAt = 0;
let cameraHealthFailureCount = 0;
let microphoneHealthFailureCount = 0;
let sessionAccessToken = null;
const selectedRoles = { parent: null, child: null };
let latestSessionSummary = null;
let sessionInsights = createSessionInsights();
let currentStep = 1;
let reconnectTimer = null;
let reconnecting = false;
let reconnectCount = 0;
let activeStudyContext = null;
let activeAdmissionToken = null;
let pendingDeliveryExecution = null;

const PHASE_ART = {
  setup: "/static/img/companion-observing.png",
  observing: "/static/img/companion-observing.png",
  intervention: "/static/img/companion-strategy.png",
  positive: "/static/img/companion-positive.png",
  summary: "/static/img/companion-positive.png",
};

const SUPPORT_LABELS = {
  parent_regulation: "情绪与节奏",
  child_support: "儿童支持",
  relationship_repair: "关系沟通",
  needs_clarification: "需求澄清",
  task_pacing: "节奏调整",
  task_support: "学习支持",
  autonomy_boundary: "自主参与",
  posture_adjustment: "姿态调整",
  environment_adjustment: "环境调整",
  boundary_setting: "边界协商",
};

function createSessionInsights() {
  return {
    interventionCount: 0,
    positiveCount: 0,
    improvementCount: 0,
    supportCounts: {},
  };
}

function makeSessionCode() {
  const randomPart = crypto.randomUUID
    ? crypto.randomUUID().replaceAll("-", "").slice(0, 12)
    : Array.from(crypto.getRandomValues(new Uint8Array(6)), (value) =>
      value.toString(16).padStart(2, "0")).join("");
  return `web_${randomPart}`;
}

function setPhase(phase) {
  document.body.dataset.phase = phase;
  const phases = {
    setup: {
      label: "准备中",
      title: "准备开始",
      text: "完成准备后，一起开始今天的任务。",
      note: "页面会保持安静，只在需要时显示简短提示。",
    },
    observing: {
      label: "活动进行中",
      title: "和你一起学习中…",
      text: "按自己的节奏继续就好。",
      note: "只有在时机合适且确有需要时，系统才会提供支持。",
    },
    positive: {
      label: "一条鼓励",
      title: "保持现在的配合",
      text: "",
      note: "这条提示可以忽略，你们也可以直接继续。",
    },
    intervention: {
      label: "一条简短提示",
      title: "一起调整一下",
      text: "",
      note: "你们可以采用、忽略或暂停后继续。",
    },
  };
  const config = phases[phase] || phases.setup;
  elements.stageLabel.textContent = config.label;
  elements.stageTitle.textContent = config.title;
  elements.stageNote.textContent = config.note;
  if (config.text) elements.bubbleText.textContent = config.text;
  if (elements.stageMascot && PHASE_ART[phase]) {
    elements.stageMascot.src = PHASE_ART[phase];
  }
  if (["setup", "observing", "summary"].includes(phase) && elements.stageTarget) {
    elements.stageTarget.hidden = true;
  }
}

function roleFor(group) {
  return selectedRoles[group];
}

function targetRoles(targetActor) {
  if (targetActor === "parent") return roleFor("parent") ? [roleFor("parent")] : [];
  if (targetActor === "child") return roleFor("child") ? [roleFor("child")] : [];
  return [roleFor("parent"), roleFor("child")].filter(Boolean);
}

function renderRoleImages(container, roles, className = "") {
  if (!container) return;
  container.replaceChildren();
  roles.forEach((role) => {
    const image = document.createElement("img");
    image.src = role.avatar;
    image.alt = "";
    if (className) image.className = className;
    container.append(image);
  });
}

function targetLabel(targetActor) {
  const parent = roleFor("parent")?.label || "家长";
  const child = roleFor("child")?.label || "孩子";
  if (targetActor === "parent") return `给${parent}`;
  if (targetActor === "child") return `给${child}`;
  return `给${parent}和${child}`;
}

function updateFamilyDisplay() {
  const parent = roleFor("parent");
  const child = roleFor("child");
  const avatars = [
    [elements.headerParentAvatar, parent],
    [elements.headerChildAvatar, child],
  ];
  avatars.forEach(([element, role]) => {
    if (!element) return;
    element.classList.toggle("placeholder", !role);
    element.style.backgroundImage = role ? `url("${role.avatar}")` : "";
  });
  if (elements.headerFamilyLabel) {
    elements.headerFamilyLabel.textContent = parent && child
      ? `${parent.label}和${child.label}`
      : "共同可见";
  }
  if (parent) {
    elements.parentBindingAvatar.src = parent.avatar;
    elements.parentRoleLabel.textContent = parent.label;
  }
  if (child) {
    elements.childBindingAvatar.src = child.avatar;
    elements.childRoleLabel.textContent = child.label;
  }
}

function selectRole(button) {
  const group = button.dataset.roleGroup;
  if (!group || !(group in selectedRoles)) return;
  const previousValue = selectedRoles[group]?.value;
  selectedRoles[group] = {
    value: button.dataset.role,
    label: button.dataset.label,
    avatar: button.dataset.avatar,
  };
  elements.roleOptions
    .filter((option) => option.dataset.roleGroup === group)
    .forEach((option) => option.setAttribute("aria-pressed", String(option === button)));
  const bindingButton = elements.bindingButtons.find((item) => item.dataset.speaker === group);
  if (bindingButton) bindingButton.textContent = `录${selectedRoles[group].label}声音`;
  if (previousValue && previousValue !== button.dataset.role && bindingState[group]) {
    bindingState[group] = false;
    resetDeviceCheck("角色已更改，录好声音后请重新检查设备。");
    const person = elements.bindingPeople.find((item) => item.dataset.speaker === group);
    if (person) {
      person.dataset.bound = "false";
      const status = person.querySelector("span[id$='binding-status']");
      if (status) status.textContent = "未录音";
    }
    elements.bindingCount.textContent = `${Object.values(bindingState).filter(Boolean).length} / 2`;
    elements.bindingFeedback.textContent = "角色已更改，请重新录这段声音。";
  }
  updateFamilyDisplay();
  updateStartButton();
}

function setStatus(message, tone = "neutral") {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
}

function setLiveState(active, label, state) {
  elements.liveState.dataset.active = String(active);
  elements.liveState.dataset.state = state || (active ? "listening" : "idle");
  elements.liveState.querySelector("strong").textContent = label;
}

function deviceStatusLabel(state) {
  if (state === "checking") return "检查中";
  if (state === "ok") return "正常";
  if (state === "error") return "不正常";
  return "未检查";
}

function sendDeviceHealthEvent(device, state, reason) {
  if (!captureActive || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "device_health",
    device,
    status: state === "ok" ? "normal" : "abnormal",
    reason: reason || null,
    recorded_at_ms: sessionTimestampMs(),
  }));
}

function setDeviceHealth(device, state, reason = null, audit = false) {
  const previous = deviceHealth[device];
  deviceHealth[device] = state;
  elements.deviceStatuses
    .filter((item) => item.dataset.deviceStatus === device)
    .forEach((item) => {
      item.dataset.state = state;
      const value = item.querySelector("strong");
      if (value) value.textContent = deviceStatusLabel(state);
    });
  if (audit && previous !== state && ["ok", "error"].includes(state)) {
    if (state === "error") {
      if (device === "camera") cameraHealthFailureCount += 1;
      if (device === "microphone") microphoneHealthFailureCount += 1;
    }
    sendDeviceHealthEvent(device, state, reason);
  }
}

function devicesReady() {
  return deviceHealth.camera === "ok" && deviceHealth.microphone === "ok";
}

function hasCaptureDeviceError() {
  return deviceHealth.camera === "error" || deviceHealth.microphone === "error";
}

function stopCheckedMediaStream() {
  if (checkedMediaStream) checkedMediaStream.getTracks().forEach((track) => track.stop());
  checkedMediaStream = null;
  if (!mediaStream) elements.video.srcObject = null;
  elements.devicePreview.dataset.active = "false";
}

function resetDeviceCheck(message = "请检查摄像头和麦克风。") {
  stopCheckedMediaStream();
  setDeviceHealth("camera", "unchecked");
  setDeviceHealth("microphone", "unchecked");
  elements.deviceCheckFeedback.textContent = message;
  updateStartButton();
}

function waitForVideoReady(video, timeoutMs = 5000) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = window.setInterval(() => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        window.clearInterval(timer);
        resolve();
      } else if (performance.now() - started >= timeoutMs) {
        window.clearInterval(timer);
        reject(new Error("没有收到摄像头画面"));
      }
    }, 100);
  });
}

async function verifyMicrophoneStream(stream, timeoutMs = 4000) {
  let context = null;
  let source = null;
  let node = null;
  let mute = null;
  try {
    context = new AudioContext();
    if (context.state === "suspended") await context.resume();
    await context.audioWorklet.addModule("/static/audio-worklet.js");
    source = context.createMediaStreamSource(stream);
    node = new AudioWorkletNode(context, "pcm16-chunk-processor", {
      processorOptions: { targetSampleRate: 16000, chunkSamples: 1600 },
    });
    mute = context.createGain();
    mute.gain.value = 0;
    source.connect(node).connect(mute).connect(context.destination);
    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error("没有收到麦克风数据")),
        timeoutMs,
      );
      node.port.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength === 0) return;
        window.clearTimeout(timeout);
        resolve();
      };
    });
  } finally {
    if (node) node.disconnect();
    if (source) source.disconnect();
    if (mute) mute.disconnect();
    if (context) await context.close();
  }
}

function deviceAccessMessage(error) {
  if (!(error instanceof Error)) return "设备检查失败，请重试。";
  if (error.name === "NotAllowedError") return "没有获得摄像头或麦克风权限。";
  if (error.name === "NotFoundError") return "没有找到可用的摄像头或麦克风。";
  if (error.name === "NotReadableError") return "设备可能正被其他程序使用。";
  return error.message || "设备检查失败，请重试。";
}

async function checkDevices() {
  if (deviceCheckBusy) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setDeviceHealth("camera", "error");
    setDeviceHealth("microphone", "error");
    elements.deviceCheckFeedback.textContent = "请通过 HTTPS 或服务器本机 localhost 打开页面。";
    updateStartButton();
    return;
  }
  deviceCheckBusy = true;
  stopCheckedMediaStream();
  setDeviceHealth("camera", "checking");
  setDeviceHealth("microphone", "checking");
  elements.deviceCheckFeedback.textContent = "正在检查画面和声音…";
  updateStartButton();
  try {
    checkedMediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: false },
        noiseSuppression: { ideal: false },
        autoGainControl: { ideal: false },
      },
    });
    elements.video.srcObject = checkedMediaStream;
    await elements.video.play();
    await waitForVideoReady(elements.video);
    setDeviceHealth("camera", "ok");
    elements.devicePreview.dataset.active = "true";
    try {
      await verifyMicrophoneStream(checkedMediaStream);
      setDeviceHealth("microphone", "ok");
    } catch (error) {
      setDeviceHealth("microphone", "error");
      elements.deviceCheckFeedback.textContent = deviceAccessMessage(error);
    }
    if (devicesReady()) {
      elements.deviceCheckFeedback.textContent = "摄像头和麦克风都正常，请在画面中确认双方都在采集范围内。";
    }
  } catch (error) {
    setDeviceHealth("camera", "error");
    setDeviceHealth("microphone", "error");
    stopCheckedMediaStream();
    elements.deviceCheckFeedback.textContent = deviceAccessMessage(error);
  } finally {
    deviceCheckBusy = false;
    updateStartButton();
  }
}

function updateCounters() {
  elements.audioCount.textContent = String(audioCount);
  elements.imageCount.textContent = String(imageCount);
  if (captureActive) {
    const totalSeconds = Math.floor((performance.now() - startedAt) / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    elements.elapsed.textContent = `${minutes}:${seconds}`;
  }
}

function updateInterventionPauseControl() {
  elements.pauseInterventions.dataset.paused = String(interventionsPaused);
  elements.pauseInterventions.textContent = interventionsPaused ? "恢复 AI 提示" : "暂停 AI 提示";
}

function updateVoiceToggleControl() {
  elements.toggleVoice.dataset.enabled = String(voiceEnabled);
  elements.toggleVoice.textContent = voiceEnabled ? "关闭语音" : "开启语音";
}

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  updateVoiceToggleControl();
  if (!voiceEnabled) {
    releaseInterventionAudio();
    setStatus("已关闭语音，只显示文字提示。", "neutral");
  } else {
    setStatus("已开启语音提示。", "success");
  }
}

function toggleInterventions() {
  if (!captureActive || !negotiated?.runtime_controls_enabled || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: interventionsPaused ? "resume_interventions" : "pause_interventions",
    recorded_at_ms: sessionTimestampMs(),
  }));
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
      reject(new Error("无法连接实时服务"));
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
      try { message = JSON.parse(event.data); } catch { return; }
      if (!allowed.has(message.type)) return;
      cleanup();
      if (message.type === "error") reject(new Error(message.message || "服务器拒绝了请求"));
      else resolve(message);
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

function attachSocketLifecycle(ws) {
  ws.addEventListener("message", handleServerMessage);
  ws.addEventListener("error", () => {
    if (captureActive && !stopping && ws === socket) {
      setStatus("实时连接不稳定，正在自动恢复。采集设备仍保持开启。", "warning");
    }
  });
  ws.addEventListener("close", () => {
    if (ws !== socket) return;
    socket = null;
    if (captureActive && !stopping) {
      setStatus("实时连接已中断，正在自动续接。采集设备仍保持开启。", "warning");
      scheduleReconnect();
    }
  });
}

async function openCaptureSocket(studyContext, admissionToken, reconnectIndex = 0) {
  const ws = new WebSocket(websocketUrl());
  ws.binaryType = "arraybuffer";
  socket = ws;
  attachSocketLifecycle(ws);
  await waitForSocketOpen(ws);
  ws.send(JSON.stringify({
    type: "hello",
    protocol_version: PROTOCOL_VERSION,
    session_id: elements.session.textContent,
    access_token: admissionToken,
    study_context: { ...studyContext, reconnect_index: reconnectIndex },
    capabilities: {
      audio_worklet: Boolean(window.AudioWorkletNode),
      media_devices: Boolean(navigator.mediaDevices),
      secure_context: window.isSecureContext,
      page_version: "0.6.0",
    },
  }));
  const ready = await waitForMessage(ws, ["ready", "error"]);
  ws.send(JSON.stringify({ type: "start" }));
  await waitForMessage(ws, ["started", "error"]);
  return ready;
}

function scheduleReconnect() {
  if (reconnectTimer !== null || reconnecting || stopping || !captureActive) return;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    void reconnectCaptureSocket();
  }, RECONNECT_DELAY_MS);
}

async function reconnectCaptureSocket() {
  if (reconnecting || stopping || !captureActive || !activeStudyContext || !activeAdmissionToken) return;
  reconnecting = true;
  try {
    reconnectCount += 1;
    negotiated = await openCaptureSocket(
      activeStudyContext,
      activeAdmissionToken,
      reconnectCount,
    );
    setStatus("实时连接已恢复，活动继续。", "success");
  } catch (error) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    socket = null;
    if (captureActive && !stopping) {
      setStatus("暂时无法连接服务器，系统会继续自动重试。请告知研究人员。", "error");
    }
  } finally {
    reconnecting = false;
    if (!socket && captureActive && !stopping) scheduleReconnect();
  }
}

function sessionTimestampMs() {
  return startedAt ? Math.max(0, Math.round(performance.now() - startedAt)) : 0;
}

function readStudyContext() {
  const parentAge = elements.parentAge.valueAsNumber;
  const childAge = elements.childAge.valueAsNumber;
  const childGrade = elements.childGrade.value.trim();
  if (!Number.isInteger(parentAge) || parentAge < 18 || parentAge > 80) {
    window.alert("请填写 18 至 80 岁之间的家长年龄。");
    elements.parentAge.focus();
    return null;
  }
  if (!Number.isInteger(childAge) || childAge < 5 || childAge > 18) {
    window.alert("请填写 5 至 18 岁之间的儿童年龄。");
    elements.childAge.focus();
    return null;
  }
  if (!childGrade) {
    window.alert("请填写儿童年级。");
    elements.childGrade.focus();
    return null;
  }
  return {
    participant_id: elements.session.textContent,
    experiment_label: "正式实验",
    session_round: "1",
    basic_info: {
      parent_age: parentAge,
      child_age: childAge,
      child_grade: childGrade,
    },
    family_roles: {
      parent: selectedRoles.parent?.value || "",
      child: selectedRoles.child?.value || "",
    },
    task_context: {
      task_name: elements.taskName?.value.trim() || "",
      task_type: elements.taskType?.value || "",
      task_difficulty: elements.taskDifficulty?.value || "",
      child_grade: childGrade,
    },
  };
}

function validateStep(step) {
  if (step === 1) {
    const parentAge = elements.parentAge.valueAsNumber;
    const childAge = elements.childAge.valueAsNumber;
    return Number.isInteger(parentAge) && parentAge >= 18 && parentAge <= 80
      && Number.isInteger(childAge) && childAge >= 5 && childAge <= 18
      && Boolean(elements.childGrade.value.trim())
      && Boolean(elements.taskName.value.trim())
      && Boolean(elements.taskType.value)
      && Boolean(elements.taskDifficulty.value);
  }
  if (step === 2) {
    return Boolean(selectedRoles.parent && selectedRoles.child)
      && bindingState.parent && bindingState.child;
  }
  return true;
}

function canEnterStep(step) {
  for (let prior = 1; prior < step; prior += 1) {
    if (!validateStep(prior)) return false;
  }
  return true;
}

function goToStep(step) {
  if (step < 1 || step > 3) return;
  currentStep = step;
  elements.wizardTabs.forEach((tab) => {
    const tabStep = Number(tab.dataset.step);
    tab.classList.toggle("active", tabStep === step);
    tab.setAttribute("aria-selected", String(tabStep === step));
    tab.disabled = tabStep > step && !canEnterStep(tabStep);
  });
  elements.wizardPanels.forEach((panel) => {
    const panelStep = Number(panel.dataset.step);
    panel.hidden = panelStep !== step;
    panel.classList.toggle("active", panelStep === step);
  });
  elements.wizardPrev.hidden = step === 1;
  const isFinal = step === 3;
  elements.wizardNext.hidden = isFinal;
  elements.start.hidden = !isFinal;
  const wizardBody = document.querySelector(".wizard-body");
  if (wizardBody) wizardBody.scrollTop = 0;
  updateStartButton();
}

function nextStep() {
  if (currentStep < 3 && validateStep(currentStep)) {
    goToStep(currentStep + 1);
  } else if (!validateStep(currentStep)) {
    if (currentStep === 1) {
      window.alert("请先填完基本信息和作业内容。");
    } else if (currentStep === 2) {
      window.alert("请先选择角色并录完两段声音。");
    }
  }
}

function prevStep() {
  if (currentStep > 1) goToStep(currentStep - 1);
}

function updateStartButton() {
  const rolesReady = Boolean(selectedRoles.parent && selectedRoles.child);
  const voicesReady = bindingState.parent && bindingState.child;
  const parentAge = elements.parentAge.valueAsNumber;
  const childAge = elements.childAge.valueAsNumber;
  const informationReady = Number.isInteger(parentAge)
    && parentAge >= 18 && parentAge <= 80
    && Number.isInteger(childAge)
    && childAge >= 5 && childAge <= 18
    && Boolean(elements.childGrade.value.trim());
  const taskReady = Boolean(
    elements.taskName.value.trim()
    && elements.taskType.value
    && elements.taskDifficulty.value
  );
  const deviceReady = devicesReady();
  elements.start.disabled = !(
    informationReady
    && taskReady
    && rolesReady
    && voicesReady
    && deviceReady
    && elements.consent.checked
  );
  elements.deviceCheckButton.disabled = bindingBusy || deviceCheckBusy || !voicesReady;
  if (deviceCheckBusy) {
    elements.deviceCheckButton.textContent = "正在检查…";
  } else if (!voicesReady) {
    elements.deviceCheckButton.textContent = "录完声音后可检查";
  } else if (deviceHealth.camera !== "unchecked" || deviceHealth.microphone !== "unchecked") {
    elements.deviceCheckButton.textContent = "重新检查设备";
  } else {
    elements.deviceCheckButton.textContent = "检查设备";
  }
  elements.bindingButtons.forEach((button) => {
    const roleReady = Boolean(selectedRoles[button.dataset.speaker]);
    button.disabled = bindingBusy || !roleReady;
  });
  if (!informationReady) {
    elements.startRequirement.textContent = "先填写基本信息。";
  } else if (!taskReady) {
    elements.startRequirement.textContent = "请完整填写今天的作业。";
  } else if (!rolesReady) {
    elements.startRequirement.textContent = "先选择家长和孩子。";
    elements.bindingFeedback.textContent = "选好角色后，分别录一段声音。";
  } else if (!voicesReady) {
    elements.startRequirement.textContent = "还需要录完家长和孩子的声音。";
  } else if (!deviceReady) {
    elements.startRequirement.textContent = "请检查摄像头和麦克风。";
    if (!deviceCheckBusy && deviceHealth.camera === "unchecked") {
      elements.deviceCheckFeedback.textContent = "声音已录好，现在可以检查设备。";
    }
  } else if (!elements.consent.checked) {
    elements.startRequirement.textContent = "设备检查正常，请确认采集说明。";
  } else {
    elements.startRequirement.textContent = "都准备好了，可以开始。";
  }
  elements.wizardNext.disabled = !validateStep(currentStep);
  elements.wizardTabs.forEach((tab) => {
    const tabStep = Number(tab.dataset.step);
    tab.disabled = tabStep > currentStep && !canEnterStep(tabStep);
  });
}

async function ensureSessionAdmission() {
  if (sessionAccessToken) return sessionAccessToken;
  const studyContext = readStudyContext();
  if (!studyContext) return null;

  const response = await fetch("/api/session-admission", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: elements.session.textContent,
      basic_info: studyContext.basic_info,
    }),
  });
  if (!response.ok) {
    let detail = "暂时无法建立本次会话，请刷新页面后重试。";
    try {
      const errorBody = await response.json();
      if (errorBody.detail) detail = errorBody.detail;
    } catch {
      // Keep the plain fallback message.
    }
    throw new Error(detail);
  }

  const payload = await response.json();
  if (typeof payload.session_token !== "string" || !payload.session_token) {
    throw new Error("暂时无法建立本次会话，请刷新页面后重试。");
  }
  sessionAccessToken = payload.session_token;
  return sessionAccessToken;
}

async function recordBindingAudio(speaker) {
  if (bindingBusy) return;
  if (!selectedRoles[speaker]) {
    elements.bindingFeedback.textContent = `请先选择${speaker === "parent" ? "家长" : "孩子"}角色。`;
    return;
  }
  if (deviceHealth.camera !== "unchecked" || deviceHealth.microphone !== "unchecked") {
    resetDeviceCheck("声音录制已更新，完成后请重新检查设备。");
  }
  let admissionToken;
  try {
    admissionToken = await ensureSessionAdmission();
  } catch (error) {
    elements.bindingFeedback.textContent = error instanceof Error
      ? error.message
      : "暂时无法建立本次会话，请刷新页面后重试。";
    return;
  }
  if (!admissionToken) return;

  bindingBusy = true;

  const person = elements.bindingPeople.find((p) => p.dataset.speaker === speaker);
  const button = elements.bindingButtons.find((b) => b.dataset.speaker === speaker);
  const statusSpan = person?.querySelector("span[id$='binding-status']");
  const progressBar = person?.querySelector(".binding-progress > span");

  elements.bindingButtons.forEach((btn) => { btn.disabled = true; });
  if (statusSpan) statusSpan.textContent = "正在录制…";
  if (progressBar) progressBar.style.width = "0%";

  let bindingStream = null;
  let bindingContext = null;
  let bindingNode = null;
  let muteGain = null;
  let progressTimer = null;
  const chunks = [];
  const hadBinding = bindingState[speaker];

  try {
    bindingStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: false },
        noiseSuppression: { ideal: false },
        autoGainControl: { ideal: false },
      },
    });

    bindingContext = new AudioContext({ sampleRate: 16000 });
    if (bindingContext.state === "suspended") await bindingContext.resume();
    await bindingContext.audioWorklet.addModule("/static/audio-worklet.js");
    const source = bindingContext.createMediaStreamSource(bindingStream);
    bindingNode = new AudioWorkletNode(bindingContext, "pcm16-chunk-processor", {
      processorOptions: {
        targetSampleRate: 16000,
        chunkSamples: 1600,
      },
    });
    muteGain = bindingContext.createGain();
    muteGain.gain.value = 0;
    source.connect(bindingNode).connect(muteGain).connect(bindingContext.destination);

    bindingNode.port.onmessage = (event) => {
      chunks.push(new Uint8Array(event.data));
    };

    const recordStart = performance.now();
    progressTimer = window.setInterval(() => {
      const elapsed = performance.now() - recordStart;
      const percent = Math.min(100, (elapsed / BINDING_RECORDING_MS) * 100);
      if (progressBar) progressBar.style.width = `${percent}%`;
    }, 100);

    await new Promise((resolve) => window.setTimeout(resolve, BINDING_RECORDING_MS));
    window.clearInterval(progressTimer);
    progressTimer = null;
    if (progressBar) progressBar.style.width = "100%";

    bindingNode.disconnect();
    source.disconnect();
    if (muteGain) muteGain.disconnect();
    if (statusSpan) statusSpan.textContent = "正在确认…";
    if (button) button.textContent = "正在确认…";
    elements.bindingFeedback.textContent = "录音完成，正在确认声音。";

    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    const sessionId = elements.session.textContent;
    const url = `/api/speaker-binding/${sessionId}/${speaker}`;
    const headers = {
      "Content-Type": "application/octet-stream",
      "x-study-access-token": admissionToken,
    };

    const controller = new AbortController();
    const processingTimer = window.setTimeout(
      () => controller.abort(),
      BINDING_PROCESSING_TIMEOUT_MS,
    );
    let response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers,
        body: combined.buffer,
        signal: controller.signal,
      });
    } finally {
      window.clearTimeout(processingTimer);
    }

    if (!response.ok) {
      let detail = `绑定请求失败: ${response.status}`;
      try {
        const errorBody = await response.json();
        if (errorBody.detail) detail = errorBody.detail;
      } catch { /* ignore parse error */ }
      throw new Error(detail);
    }

    bindingState[speaker] = true;
    if (person) person.dataset.bound = "true";
    if (statusSpan) statusSpan.textContent = "已录好";

    const boundCount = Object.values(bindingState).filter(Boolean).length;
    elements.bindingCount.textContent = `${boundCount} / 2`;
    const currentLabel = selectedRoles[speaker]?.label || (speaker === "parent" ? "家长" : "孩子");
    const otherSpeaker = speaker === "parent" ? "child" : "parent";
    const otherLabel = selectedRoles[otherSpeaker]?.label || (otherSpeaker === "parent" ? "家长" : "孩子");
    elements.bindingFeedback.textContent = boundCount === 2
      ? "两段声音已录好，请继续检查设备。"
      : `${currentLabel}已录好，请继续录${otherLabel}的声音。`;

    if (button) button.textContent = "重新录制";
    updateStartButton();
  } catch (error) {
    if (statusSpan) statusSpan.textContent = hadBinding ? "已录好" : "未录音";
    if (person) person.dataset.bound = String(hadBinding);
    const timedOut = error && typeof error === "object" && error.name === "AbortError";
    const errorMsg = timedOut
      ? "声音确认超时，请重新录制。"
      : (error instanceof Error ? error.message : "录制失败，请重试。");
    elements.bindingFeedback.textContent = errorMsg;
    if (button) {
      const roleLabel = selectedRoles[speaker]?.label || (speaker === "parent" ? "家长" : "孩子");
      button.textContent = hadBinding ? "重新录制" : `录${roleLabel}声音`;
    }
  } finally {
    if (progressTimer !== null) window.clearInterval(progressTimer);
    if (bindingNode) bindingNode.disconnect();
    if (muteGain) muteGain.disconnect();
    if (bindingContext) void bindingContext.close();
    if (bindingStream) bindingStream.getTracks().forEach((track) => track.stop());
    if (progressBar) window.setTimeout(() => { progressBar.style.width = ""; }, 600);
    bindingBusy = false;
    updateStartButton();
  }
}

function sendFamilyResponse(response) {
  if (!negotiated?.runtime_controls_enabled || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "family_response",
    response,
    delivery_id: currentDeliveryId,
    recorded_at_ms: sessionTimestampMs(),
  }));
}

function releaseInterventionAudio() {
  if (interventionAudioUrl) URL.revokeObjectURL(interventionAudioUrl);
  interventionAudioUrl = null;
}

function clearPendingDeliveryExecution() {
  if (pendingDeliveryExecution?.timer) window.clearTimeout(pendingDeliveryExecution.timer);
  pendingDeliveryExecution = null;
}

function decodeBase64Audio(encoded, mimeType) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return URL.createObjectURL(new Blob([bytes], { type: mimeType || "audio/wav" }));
}

function supportCategory(repairTarget) {
  return SUPPORT_LABELS[repairTarget] || "其他支持";
}

function trackIntervention(message) {
  if (message.repair_target === "positive_reinforcement") {
    sessionInsights.positiveCount += 1;
    return;
  }
  sessionInsights.interventionCount += 1;
  const category = supportCategory(message.repair_target);
  sessionInsights.supportCounts[category] = (sessionInsights.supportCounts[category] || 0) + 1;
}

function trackInterventionOutcome(message) {
  if (["recovered", "partial_recovery"].includes(message.recovery_status)) {
    sessionInsights.improvementCount += 1;
  }
}

async function playInterventionAudio(message) {
  if (!message.voice_expected) return { status: "not_attempted" };
  if (!voiceEnabled) return { status: "not_attempted" };
  if (!message.audio_base64) {
    return { status: "failed", started_at_ms: sessionTimestampMs(), error: message.voice_error || "语音暂时不可用" };
  }
  releaseInterventionAudio();
  interventionAudioUrl = decodeBase64Audio(message.audio_base64, message.audio_mime_type);
  const audio = new Audio(interventionAudioUrl);
  const startedAtMs = sessionTimestampMs();
  try {
    await audio.play();
    await new Promise((resolve, reject) => {
      audio.addEventListener("ended", resolve, { once: true });
      audio.addEventListener("error", () => reject(new Error("语音播放失败")), { once: true });
    });
    return {
      status: "delivered",
      started_at_ms: startedAtMs,
      completed_at_ms: sessionTimestampMs(),
      provider: message.voice_provider,
      output_identifier: message.voice_output_identifier,
    };
  } catch (error) {
    return {
      status: "failed",
      started_at_ms: startedAtMs,
      provider: message.voice_provider,
      error: error instanceof Error ? error.message : "语音播放失败",
    };
  }
}

async function presentIntervention(message) {
  const visualStartedAtMs = sessionTimestampMs();
  currentDeliveryId = message.delivery_id;

  const phase = message.repair_target === "positive_reinforcement" ? "positive" : "intervention";
  setPhase(phase);
  trackIntervention(message);

  const bubbleText = document.querySelector("#bubble-text");
  bubbleText.textContent = message.message;
  elements.stageTarget.hidden = false;
  renderRoleImages(elements.stageTargetAvatars, targetRoles(message.target_actor));
  elements.stageTargetLabel.textContent = targetLabel(message.target_actor);

  elements.interventionSource.dataset.source = message.source || "ai";
  elements.interventionSource.textContent = message.source === "expert" ? "专家支持" : "AI 支持";
  elements.interventionTarget.textContent = targetLabel(message.target_actor);
  renderRoleImages(elements.interventionTargetAvatars, targetRoles(message.target_actor));
  elements.interventionHeading.textContent = message.heading || "一起调整一下";
  elements.interventionMessage.textContent = message.message;
  elements.intervention.dataset.kind = phase;
  elements.interventionMascot.src = PHASE_ART[phase];
  elements.difficultyOptions.hidden = !["task_support", "needs_clarification", "task_pacing"].includes(message.repair_target);
  elements.interventionChannel.textContent = (message.voice_expected && voiceEnabled) ? "正在同步播放语音" : "你们可以忽略或关闭这条提示";
  elements.intervention.hidden = false;

  if (message.voice_pending && voiceEnabled) {
    clearPendingDeliveryExecution();
    pendingDeliveryExecution = {
      deliveryId: message.delivery_id,
      visualStartedAtMs,
      timer: window.setTimeout(() => {
        const pending = pendingDeliveryExecution;
        if (!pending || pending.deliveryId !== message.delivery_id) return;
        clearPendingDeliveryExecution();
        elements.interventionChannel.textContent = "文字提示已显示，语音暂时不可用";
        sendDeliveryExecution(message.delivery_id, visualStartedAtMs, {
          status: "failed",
          started_at_ms: sessionTimestampMs(),
          error: "语音准备超时",
        });
      }, VOICE_DELIVERY_TIMEOUT_MS),
    };
    elements.interventionChannel.textContent = "文字提示已显示，正在准备语音";
    return;
  }

  const voice = await playInterventionAudio(message);
  if (voice.status === "failed") elements.interventionChannel.textContent = "文字提示已显示，语音暂时不可用";
  sendDeliveryExecution(message.delivery_id, visualStartedAtMs, voice);
}

function sendDeliveryExecution(deliveryId, visualStartedAtMs, voice) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "delivery_execution",
    delivery_id: deliveryId,
    recorded_at_ms: sessionTimestampMs(),
    visual: {
      status: "delivered",
      started_at_ms: visualStartedAtMs,
      completed_at_ms: visualStartedAtMs,
      provider: "browser_overlay",
    },
    voice,
  }));
}

async function completeInterventionVoice(message) {
  const pending = pendingDeliveryExecution;
  if (!pending || pending.deliveryId !== message.delivery_id) return;
  clearPendingDeliveryExecution();
  const voice = await playInterventionAudio({
    ...message,
    voice_expected: true,
  });
  elements.interventionChannel.textContent = voice.status === "delivered"
    ? "文字和语音提示已送达"
    : "文字提示已显示，语音暂时不可用";
  sendDeliveryExecution(message.delivery_id, pending.visualStartedAtMs, voice);
}

function handleServerMessage(event) {
  if (typeof event.data !== "string") return;
  let message;
  try { message = JSON.parse(event.data); } catch { return; }
  if (message.type === "intervention") {
    void presentIntervention(message);
  } else if (message.type === "intervention_voice") {
    void completeInterventionVoice(message);
  } else if (message.type === "interventions_paused") {
    interventionsPaused = true;
    updateInterventionPauseControl();
    setStatus(message.reason === "expert_takeover_active" ? "专家正在提供支持，AI 提示保持暂停。" : "AI 提示已暂停，活动仍在继续。", "warning");
    document.querySelector("#bubble-text").textContent = "已暂停提示，活动仍在继续。";
  } else if (message.type === "interventions_resumed") {
    interventionsPaused = false;
    updateInterventionPauseControl();
    setStatus("AI 提示已恢复。", "success");
    setPhase("observing");
  } else if (message.type === "expert_takeover_started") {
    interventionsPaused = true;
    updateInterventionPauseControl();
    setLiveState(true, "专家支持中", "analyzing");
    setStatus("专业人员已接入，AI 自动提示已暂停。", "working");
  } else if (message.type === "expert_takeover_ended") {
    interventionsPaused = false;
    updateInterventionPauseControl();
    setLiveState(true, "共同活动中", "listening");
    setStatus("专家支持已结束，活动继续。", "success");
  } else if (message.type === "control_unavailable") {
    setStatus("当前未启用实时提示，这项操作不会影响活动记录。", "warning");
  } else if (message.type === "loop_error") {
    setStatus(
      message.service_degraded
        ? "连续几轮分析暂时没有结果，采集仍在继续，请研究人员查看。"
        : "本轮分析暂时没有结果，活动可以继续。",
      message.service_degraded ? "error" : "warning",
    );
  } else if (message.type === "analysis_recovered") {
    setStatus("状态分析已经恢复，活动继续。", "success");
  } else if (message.type === "capture_warning") {
    setStatus("一条采集数据未能处理，活动仍在继续。", "warning");
  } else if (message.type === "stopping") {
    setStatus("正在完成最后一轮分析并保存记录…", "working");
  } else if (message.type === "intervention_outcome") {
    trackInterventionOutcome(message);
  } else if (message.type === "state_update" && captureActive) {
    if (!hasCaptureDeviceError()) {
      setStatus("活动进行中。需要时，系统会给出简短提示。", "success");
    }
    if (elements.intervention.hidden && document.body.dataset.phase !== "observing") {
      setPhase("observing");
    }
  }
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

function waitForFirstAudioChunk(timeoutMs = 4000) {
  if (audioCount > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = window.setInterval(() => {
      if (audioCount > 0) {
        window.clearInterval(timer);
        resolve();
      } else if (performance.now() - started >= timeoutMs) {
        window.clearInterval(timer);
        reject(new Error("没有收到麦克风数据，请重新检查设备"));
      }
    }, 100);
  });
}

function monitorCaptureHealth() {
  if (!captureActive || !mediaStream) return;
  const now = performance.now();
  const videoTrack = mediaStream.getVideoTracks()[0];
  const audioTrack = mediaStream.getAudioTracks()[0];
  const imageStaleMs = Math.max(
    10000,
    Number(negotiated?.media_format?.image_interval_ms || 1000) * 3,
  );
  const cameraOkay = Boolean(
    videoTrack
    && videoTrack.readyState === "live"
    && lastImageFrameAt > 0
    && now - lastImageFrameAt <= imageStaleMs
  );
  const microphoneOkay = Boolean(
    audioTrack
    && audioTrack.readyState === "live"
    && lastAudioChunkAt > 0
    && now - lastAudioChunkAt <= 5000
  );
  const hadError = hasCaptureDeviceError();
  setDeviceHealth(
    "camera",
    cameraOkay ? "ok" : "error",
    cameraOkay ? null : "video_frames_not_arriving",
    true,
  );
  setDeviceHealth(
    "microphone",
    microphoneOkay ? "ok" : "error",
    microphoneOkay ? null : "audio_chunks_not_arriving",
    true,
  );
  if (!cameraOkay || !microphoneOkay) {
    const failed = [
      !cameraOkay ? "摄像头" : null,
      !microphoneOkay ? "麦克风" : null,
    ].filter(Boolean).join("和");
    setStatus(`${failed}采集不正常，请检查设备或联系研究人员。`, "error");
  } else if (hadError) {
    setStatus("摄像头和麦克风已恢复正常。", "success");
  }
}

async function prepareMedia() {
  const checkedTracksAreLive = Boolean(
    checkedMediaStream
    && checkedMediaStream.getVideoTracks().some((track) => track.readyState === "live")
    && checkedMediaStream.getAudioTracks().some((track) => track.readyState === "live")
  );
  if (checkedTracksAreLive) {
    mediaStream = checkedMediaStream;
    checkedMediaStream = null;
  } else {
    stopCheckedMediaStream();
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
  }
  elements.video.srcObject = mediaStream;
  await elements.video.play();
  await waitForVideoReady(elements.video);
  elements.cameraEmpty.hidden = true;

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
    lastAudioChunkAt = performance.now();
    const timestampMs = nextAudioTimestampMs;
    nextAudioTimestampMs += negotiated.media_format.audio_chunk_ms;
    if (!captureActive || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > AUDIO_BACKPRESSURE_BYTES) {
      audioBackpressureStops += 1;
      setStatus("网络暂时拥堵，系统已跳过少量声音数据，活动继续。", "warning");
      return;
    }
    socket.send(encodePacket(AUDIO_PACKET, timestampMs, event.data));
    audioCount += 1;
    updateCounters();
  };
}

async function captureImage() {
  if (!captureActive || imageCaptureBusy || !socket || socket.readyState !== WebSocket.OPEN) return false;
  if (socket.bufferedAmount > IMAGE_BACKPRESSURE_BYTES) {
    droppedImages += 1;
    return false;
  }
  const sourceWidth = elements.video.videoWidth;
  const sourceHeight = elements.video.videoHeight;
  if (!sourceWidth || !sourceHeight) return false;

  imageCaptureBusy = true;
  try {
    const scale = Math.min(1, negotiated.media_format.image_max_width / sourceWidth, negotiated.media_format.image_max_height / sourceHeight);
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(41, Math.round(sourceHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(elements.video, 0, 0, width, height);
    const timestampMs = sessionTimestampMs();
    context.fillStyle = "rgba(0, 0, 0, 0.78)";
    context.fillRect(0, height - 40, width, 40);
    context.fillStyle = "#ffffff";
    context.font = "600 20px system-ui, sans-serif";
    context.fillText(`frame_time_ms=${timestampMs}`, 12, height - 13);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.78));
    if (!blob) throw new Error("浏览器无法生成图像帧");
    if (blob.size > negotiated.max_image_bytes) {
      droppedImages += 1;
      return false;
    }
    socket.send(encodePacket(IMAGE_PACKET, timestampMs, await blob.arrayBuffer()));
    imageCount += 1;
    lastImageFrameAt = performance.now();
    updateCounters();
    return true;
  } finally {
    imageCaptureBusy = false;
  }
}

function releaseMedia() {
  captureActive = false;
  if (imageTimer !== null) window.clearInterval(imageTimer);
  if (elapsedTimer !== null) window.clearInterval(elapsedTimer);
  if (captureHealthTimer !== null) window.clearInterval(captureHealthTimer);
  imageTimer = null;
  elapsedTimer = null;
  captureHealthTimer = null;
  if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
  reconnectTimer = null;
  reconnecting = false;
  if (audioNode) audioNode.disconnect();
  if (muteNode) muteNode.disconnect();
  audioNode = null;
  muteNode = null;
  if (audioContext) void audioContext.close();
  audioContext = null;
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.video.srcObject = null;
  elements.devicePreview.dataset.active = "false";
  elements.cameraEmpty.hidden = false;
  lastAudioChunkAt = 0;
  lastImageFrameAt = 0;
  releaseInterventionAudio();
  clearPendingDeliveryExecution();
}

async function startCapture() {
  const studyContext = readStudyContext();
  if (!studyContext) return;
  if (!selectedRoles.parent || !selectedRoles.child) {
    window.alert("请先选择家长和孩子的角色。");
    return;
  }
  if (!elements.consent.checked) {
    window.alert("请先确认你们已了解采集说明。");
    return;
  }
  // Validate task context before starting
  const taskCtx = studyContext.task_context;
  if (!taskCtx.task_name || !taskCtx.task_type || !taskCtx.task_difficulty || !taskCtx.child_grade) {
    window.alert("请完整填写任务名称、作业学科、任务难度和儿童年级。");
    return;
  }
  if (!devicesReady()) {
    window.alert("请先检查摄像头和麦克风，两项正常后再开始。");
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    window.alert("请通过 HTTPS 或服务器本机 localhost 打开页面。");
    return;
  }
  let admissionToken;
  try {
    admissionToken = await ensureSessionAdmission();
  } catch (error) {
    window.alert(error instanceof Error ? error.message : "暂时无法开始，请刷新页面后重试。");
    return;
  }
  if (!admissionToken) return;


  elements.start.disabled = true;
  latestSessionSummary = null;
  sessionInsights = createSessionInsights();
  elements.summaryPanel.hidden = true;
  elements.recordNav.disabled = true;
  elements.recordNav.classList.remove("active");
  elements.homeNav.classList.add("active");
  elements.homeNav.setAttribute("aria-current", "page");
  elements.stageTarget.hidden = true;
  try {
    reconnectCount = 0;
    activeStudyContext = studyContext;
    activeAdmissionToken = admissionToken;
    negotiated = await openCaptureSocket(studyContext, admissionToken);
    await prepareMedia();

    startedAt = performance.now();
    nextAudioTimestampMs = 0;
    audioCount = 0;
    imageCount = 0;
    droppedImages = 0;
    audioBackpressureStops = 0;
    cameraHealthFailureCount = 0;
    microphoneHealthFailureCount = 0;
    lastAudioChunkAt = 0;
    lastImageFrameAt = 0;
    captureActive = true;
    interventionsPaused = false;
    stopping = false;
    setDeviceHealth("camera", "checking");
    setDeviceHealth("microphone", "checking");
    await waitForFirstAudioChunk();
    const firstImageCaptured = await captureImage();
    if (!firstImageCaptured) throw new Error("没有收到摄像头图像，请重新检查设备");
    setDeviceHealth("camera", "ok", "initial_capture_verified", true);
    setDeviceHealth("microphone", "ok", "initial_capture_verified", true);
    elements.startCard.hidden = true;
    elements.sessionControls.hidden = false;
    const runtimeControlsEnabled = Boolean(negotiated.runtime_controls_enabled);
    elements.pauseInterventions.disabled = !runtimeControlsEnabled;
    elements.toggleVoice.disabled = !runtimeControlsEnabled;
    elements.selfContinue.disabled = !runtimeControlsEnabled;
    elements.stop.disabled = false;
    voiceEnabled = runtimeControlsEnabled;
    updateInterventionPauseControl();
    updateVoiceToggleControl();
    setLiveState(true, "共同活动中", "listening");
    setStatus(
      runtimeControlsEnabled
        ? "活动进行中。需要时，系统会给出简短提示。"
        : "当前仅记录设备与连接状态，不会进行状态判断或给出提示。",
      runtimeControlsEnabled ? "success" : "warning",
    );
    setPhase("observing");
    imageTimer = window.setInterval(() => void captureImage(), negotiated.media_format.image_interval_ms);
    elapsedTimer = window.setInterval(updateCounters, 250);
    captureHealthTimer = window.setInterval(monitorCaptureHealth, 3000);
  } catch (error) {
    console.error("[startCapture] error:", error);
    releaseMedia();
    setDeviceHealth("camera", "error");
    setDeviceHealth("microphone", "error");
    elements.startCard.hidden = false;
    elements.sessionControls.hidden = true;
    elements.deviceCheckFeedback.textContent = error instanceof Error
      ? error.message
      : "设备采集未能开始，请重新检查。";
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    updateStartButton();
    window.alert(error instanceof Error ? error.message : "暂时无法开始，请联系研究人员");
  }
}

function formatDuration(durationMs) {
  const totalSeconds = Math.max(0, Math.round((durationMs || 0) / 1000));
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分钟`;
}

function renderSupportSummary() {
  const entries = Object.entries(sessionInsights.supportCounts)
    .sort((left, right) => right[1] - left[1]);
  elements.summarySupports.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "support-empty";
    empty.textContent = sessionInsights.positiveCount
      ? "本次主要记录到积极配合，系统没有额外提出调整建议。"
      : "本次没有产生额外的支持提示。";
    elements.summarySupports.append(empty);
    return;
  }
  const maximum = Math.max(...entries.map(([, count]) => count));
  entries.slice(0, 5).forEach(([label, count]) => {
    const row = document.createElement("div");
    row.className = "support-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("span");
    track.className = "support-track";
    const fill = document.createElement("span");
    fill.style.setProperty("--support-width", `${Math.round((count / maximum) * 100)}%`);
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = `${count} 次`;
    row.append(name, track, value);
    elements.summarySupports.append(row);
  });
}

function showSessionSummary(summary) {
  latestSessionSummary = summary;
  renderRoleImages(elements.summaryFamily, targetRoles("both"));
  elements.summaryDuration.textContent = formatDuration(summary.duration_ms);
  elements.summaryInterventions.textContent = `${sessionInsights.interventionCount} 次`;
  elements.summaryPositive.textContent = `${sessionInsights.positiveCount} 次`;
  elements.summaryImprovements.textContent = `${sessionInsights.improvementCount} 次`;
  elements.summary.textContent = summary.preview
    ? "这是界面预览总结，未保存实验记录。"
    : summary.valid
      ? "本次实验记录已安全保存。"
      : "活动已经结束，部分技术记录可能不完整。";
  renderSupportSummary();
  elements.startCard.hidden = true;
  elements.summaryPanel.hidden = false;
  elements.sessionControls.hidden = true;
  elements.recordNav.disabled = false;
  elements.recordNav.classList.add("active");
  elements.homeNav.classList.remove("active");
  elements.homeNav.removeAttribute("aria-current");
  elements.recordNav.setAttribute("aria-current", "page");
  setPhase("summary");
}

function resetPreparation() {
  elements.summaryPanel.hidden = true;
  elements.startCard.hidden = false;
  elements.intervention.hidden = true;
  elements.session.textContent = makeSessionCode();
  sessionAccessToken = null;
  elements.parentAge.value = "";
  elements.childAge.value = "";
  elements.consent.checked = false;
  if (elements.taskName) elements.taskName.value = "";
  if (elements.taskType) elements.taskType.value = "";
  if (elements.taskDifficulty) elements.taskDifficulty.value = "";
  if (elements.childGrade) elements.childGrade.value = "";
  bindingState.parent = false;
  bindingState.child = false;
  selectedRoles.parent = null;
  selectedRoles.child = null;
  elements.roleOptions.forEach((option) => option.setAttribute("aria-pressed", "false"));
  elements.bindingCount.textContent = "0 / 2";
  elements.bindingFeedback.textContent = "选好角色后，分别录一段声音。";
  elements.bindingPeople.forEach((person) => {
    person.dataset.bound = "false";
    const statusSpan = person.querySelector("span[id$='binding-status']");
    if (statusSpan) statusSpan.textContent = "未录音";
  });
  elements.parentBindingAvatar.src = "/static/img/avatar-mother.png";
  elements.childBindingAvatar.src = "/static/img/avatar-boy.png";
  elements.parentRoleLabel.textContent = "家长";
  elements.childRoleLabel.textContent = "孩子";
  elements.bindingButtons.forEach((button) => {
    button.textContent = button.dataset.speaker === "parent" ? "录家长声音" : "录孩子声音";
  });
  deviceCheckBusy = false;
  resetDeviceCheck("请先完成家长和孩子的声音录制。");
  elements.recordNav.classList.remove("active");
  elements.recordNav.removeAttribute("aria-current");
  elements.homeNav.classList.add("active");
  elements.homeNav.setAttribute("aria-current", "page");
  updateFamilyDisplay();
  setPhase("setup");
  goToStep(1);
}

async function stopCapture(normal = true, reason = null) {
  if (stopping) return;
  stopping = true;
  const hasLiveSocket = Boolean(socket && socket.readyState === WebSocket.OPEN);
  const localDurationMs = Math.max(0, Math.round(performance.now() - startedAt));
  let finalSummary = {
    valid: hasLiveSocket && normal,
    preview: !hasLiveSocket,
    duration_ms: localDurationMs,
    run_id: elements.session.textContent,
  };
  releaseMedia();
  elements.stop.disabled = true;
  elements.pauseInterventions.disabled = true;
  elements.toggleVoice.disabled = true;
  elements.selfContinue.disabled = true;
  setStatus(normal ? "正在安全结束…" : reason, normal ? "working" : "error");

  try {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: normal ? "stop" : "abort",
        reason,
        client_metrics: {
          dropped_images: droppedImages,
          audio_backpressure_stops: audioBackpressureStops,
          camera_health_failures: cameraHealthFailureCount,
          microphone_health_failures: microphoneHealthFailureCount,
          capture_duration_ms: Math.max(0, Math.round(performance.now() - startedAt)),
        },
      }));
      finalSummary = await waitForMessage(socket, ["summary", "error"], 180000);
      finalSummary.duration_ms = localDurationMs;
      socket.close();
    }
  } catch (error) {
    finalSummary = {
      valid: false,
      duration_ms: localDurationMs,
      run_id: elements.session.textContent,
      error: error instanceof Error ? error.message : "结束时发生错误",
    };
  } finally {
    socket = null;
    negotiated = null;
    activeStudyContext = null;
    activeAdmissionToken = null;
    currentDeliveryId = null;
    elements.intervention.hidden = true;
    elements.sessionControls.hidden = true;
    interventionsPaused = false;
    updateInterventionPauseControl();
    setLiveState(false, "尚未开始", "idle");
    stopping = false;
    showSessionSummary(finalSummary);
  }
}

elements.session.textContent = makeSessionCode();
elements.roleOptions.forEach((button) => {
  button.addEventListener("click", () => selectRole(button));
});
elements.bindingButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const speaker = button.dataset.speaker;
    if (speaker) void recordBindingAudio(speaker);
  });
});
[
  elements.parentAge,
  elements.childAge,
  elements.childGrade,
  elements.taskName,
  elements.taskType,
  elements.taskDifficulty,
].forEach((element) => {
  element.addEventListener("input", () => {
    updateStartButton();
  });
  element.addEventListener("change", () => {
    updateStartButton();
  });
});
elements.consent.addEventListener("change", updateStartButton);
elements.deviceCheckButton.addEventListener("click", () => void checkDevices());
elements.start.addEventListener("click", () => void startCapture());
elements.wizardNext.addEventListener("click", nextStep);
elements.wizardPrev.addEventListener("click", prevStep);
elements.wizardTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const step = Number(tab.dataset.step);
    if (canEnterStep(step)) goToStep(step);
  });
});
const devSkipButton = document.querySelector("#dev-skip-button");
if (["localhost", "127.0.0.1", "::1"].includes(window.location.hostname)) {
  devSkipButton.hidden = false;
  devSkipButton.addEventListener("click", () => {
  console.log("[dev-skip] clicked");
  try {
    // Preview mode: skip WebSocket and getUserMedia, just show the main UI
    elements.startCard.hidden = true;
    elements.sessionControls.hidden = false;
    elements.pauseInterventions.disabled = true;
    elements.toggleVoice.disabled = true;
    elements.selfContinue.disabled = true;
    elements.stop.disabled = false;
    startedAt = performance.now();
    captureActive = true;
    sessionInsights = createSessionInsights();
    voiceEnabled = false;
    updateInterventionPauseControl();
    updateVoiceToggleControl();
    setLiveState(true, "预览模式", "listening");
    setStatus("预览模式：仅展示界面，未连接服务器。", "success");
    setPhase("observing");
  } catch (err) {
    console.error("[dev-skip] error:", err);
  }
  });
}
elements.pauseInterventions.addEventListener("click", toggleInterventions);
elements.toggleVoice.addEventListener("click", toggleVoice);
elements.selfContinue.addEventListener("click", () => {
  sendFamilyResponse("self_continue");
  elements.intervention.hidden = true;
  elements.stageTarget.hidden = true;
  setPhase("observing");
  setStatus("好的，你们按自己的节奏继续。", "success");
});
elements.stop.addEventListener("click", () => void stopCapture(true));
elements.dismissIntervention.addEventListener("click", () => {
  sendFamilyResponse("dismissed");
  elements.intervention.hidden = true;
  elements.stageTarget.hidden = true;
  setPhase("observing");
});
elements.difficultyOptions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-response]");
  if (!button) return;
  sendFamilyResponse(button.dataset.response);
  elements.intervention.hidden = true;
  elements.stageTarget.hidden = true;
  setPhase("observing");
  setStatus("已收到你们的反馈，继续按合适的难度进行。", "success");
});
elements.newSession.addEventListener("click", resetPreparation);
elements.recordNav.addEventListener("click", () => {
  if (latestSessionSummary) showSessionSummary(latestSessionSummary);
});
elements.homeNav.addEventListener("click", () => {
  if (!latestSessionSummary || elements.summaryPanel.hidden) return;
  elements.summaryPanel.hidden = true;
  elements.startCard.hidden = false;
  elements.recordNav.classList.remove("active");
  elements.recordNav.removeAttribute("aria-current");
  elements.homeNav.classList.add("active");
  elements.homeNav.setAttribute("aria-current", "page");
  setPhase("setup");
  goToStep(1);
});
window.addEventListener("pagehide", () => {
  stopCheckedMediaStream();
  releaseMedia();
});
setPhase("setup");
updateFamilyDisplay();
updateCounters();
updateInterventionPauseControl();
updateVoiceToggleControl();
goToStep(1);
