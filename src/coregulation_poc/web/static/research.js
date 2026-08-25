const elements = {
  loginView: document.querySelector("#login-view"),
  loginForm: document.querySelector("#login-form"),
  researchCode: document.querySelector("#research-code"),
  loginError: document.querySelector("#login-error"),
  consoleView: document.querySelector("#console-view"),
  connectionState: document.querySelector("#connection-state"),
  sessionCount: document.querySelector("#session-count"),
  sessionList: document.querySelector("#session-list"),
  noSelection: document.querySelector("#no-selection"),
  sessionDetail: document.querySelector("#session-detail"),
  detailSessionId: document.querySelector("#detail-session-id"),
  detailStatus: document.querySelector("#detail-status"),
  currentState: document.querySelector("#current-state"),
  currentTrajectory: document.querySelector("#current-trajectory"),
  currentTaskProcess: document.querySelector("#current-task-process"),
  currentSupportNeed: document.querySelector("#current-support-need"),
  currentSupportTarget: document.querySelector("#current-support-target"),
  currentSpeakerBinding: document.querySelector("#current-speaker-binding"),
  currentAction: document.querySelector("#current-action"),
  timeline: document.querySelector("#timeline"),
  refresh: document.querySelector("#refresh-button"),
  takeoverState: document.querySelector("#takeover-state"),
  operator: document.querySelector("#operator"),
  reason: document.querySelector("#reason"),
  takeover: document.querySelector("#takeover-button"),
  release: document.querySelector("#release-button"),
  strategy: document.querySelector("#strategy"),
  expertMessage: document.querySelector("#expert-message"),
  sendIntervention: document.querySelector("#send-intervention-button"),
  expertFeedback: document.querySelector("#expert-feedback"),
};

const stateLabels = { normal: "稳定协作", fluctuation: "短暂波动", dysregulation: "需要调节", high_risk: "高风险失衡" };
const trajectoryLabels = { stable: "稳定", worsening: "恶化", recovering: "恢复中", unclear: "不明确" };
const taskProcessLabels = {
  smooth_progress: "顺利推进", brief_stall: "短暂停滞", sustained_stall: "持续停滞",
  pace_mismatch: "节奏不匹配", explanation_mismatch: "讲解不匹配", over_assistance: "过度代劳",
  disengaged: "脱离参与", completion: "任务完成", unclear: "不明确",
};
const supportNeedLabels = {
  none: "无", positive_reinforcement: "正向强化", emotional_support: "情绪支持",
  need_expression: "需求表达", mutual_understanding: "相互理解", task_pacing: "任务节奏",
  learning_support: "学习支持", autonomy_support: "自主性支持", unclear: "不明确",
};
const supportTargetLabels = { parent: "家长", child: "儿童", both: "双方", unknown: "未知" };
const confidenceLabels = { high: "高", medium: "中", low: "低" };
const actionLabels = { no_intervention: "不介入", observe: "继续观察", reinforce: "积极强化", intervene: "提供支持", progressive_support: "渐进支持", hold: "暂不介入" };
const eventLabels = {
  speaker_enrollment: "双方声音已采集",
  speaker_binding: "说话人区分结果",
  device_health: "设备采集状态",
  voiceprint_cleanup: "云端声纹清理",
  control_unavailable: "操作未执行",
  loop_started: "闭环分析已启动",
  analysis_started: "开始新一轮观察",
  state_update: "状态更新",
  intervention: "提示已发送",
  intervention_held: "本轮未发送提示",
  intervention_outcome: "干预后观察",
  interventions_paused: "AI 提示已暂停",
  interventions_resumed: "AI 提示已恢复",
  family_response_received: "家庭主动反馈",
  expert_takeover_started: "专家开始接管",
  expert_takeover_ended: "专家结束接管",
  expert_intervention_recorded: "专家提示已记录",
  loop_error: "本轮分析失败",
};

let socket = null;
let sessions = [];
let strategies = [];
let selectedSessionId = null;

function websocketUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/research`;
}

function selectedSession() {
  return sessions.find((session) => session.session_id === selectedSessionId) || null;
}

function feedback(message, error = false) {
  elements.expertFeedback.textContent = message;
  elements.expertFeedback.dataset.error = String(error);
}

function connect(code) {
  elements.loginError.textContent = "正在连接…";
  socket = new WebSocket(websocketUrl());
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "research_hello", access_token: code }));
  });
  socket.addEventListener("message", handleMessage);
  socket.addEventListener("close", () => {
    elements.connectionState.dataset.offline = "true";
    elements.connectionState.lastChild.textContent = "连接已断开";
  });
  socket.addEventListener("error", () => {
    elements.loginError.textContent = "无法连接研究服务";
  });
}

function handleMessage(event) {
  let message;
  try { message = JSON.parse(event.data); } catch { return; }
  if (message.type === "error") {
    if (elements.consoleView.hidden) elements.loginError.textContent = message.message;
    else feedback(message.message, true);
    return;
  }
  if (message.type === "research_snapshot") {
    sessions = message.sessions || [];
    strategies = message.strategies || [];
    if (!selectedSessionId || !sessions.some((session) => session.session_id === selectedSessionId)) {
      selectedSessionId = sessions[0]?.session_id || null;
    }
    elements.loginView.hidden = true;
    elements.consoleView.hidden = false;
    elements.connectionState.dataset.offline = "false";
    render();
  } else if (message.type === "session_event") {
    window.clearTimeout(handleMessage.refreshTimer);
    handleMessage.refreshTimer = window.setTimeout(() => send({ type: "refresh" }), 80);
  } else if (message.type === "research_control_ack") {
    feedback("操作已记录并同步到家庭端。", false);
    send({ type: "refresh" });
  }
}

function send(payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    feedback("研究端连接已断开，请刷新页面后重试。", true);
    return false;
  }
  socket.send(JSON.stringify(payload));
  return true;
}

function render() {
  elements.sessionCount.textContent = String(sessions.length);
  renderSessionList();
  renderStrategies();
  renderDetail();
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "等待家庭端连接…";
    elements.sessionList.append(empty);
    return;
  }
  for (const session of sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.dataset.selected = String(session.session_id === selectedSessionId);
    button.dataset.active = String(session.status === "active");
    const title = document.createElement("strong");
    title.textContent = session.session_id;
    const status = document.createElement("span");
    status.innerHTML = `<i></i>${session.status === "active" ? "活动中" : session.status}`;
    const state = document.createElement("span");
    state.textContent = session.latest_state ? stateLabels[session.latest_state] || session.latest_state : "等待首次识别";
    button.append(title, status, state);
    button.addEventListener("click", () => {
      selectedSessionId = session.session_id;
      feedback("");
      render();
    });
    elements.sessionList.append(button);
  }
}

function renderStrategies() {
  const currentValue = elements.strategy.value;
  elements.strategy.replaceChildren(new Option("请选择策略", ""));
  for (const strategy of strategies) {
    elements.strategy.append(new Option(`${strategy.name} · ${strategy.target_actor}`, strategy.strategy_id));
  }
  if (strategies.some((strategy) => strategy.strategy_id === currentValue)) elements.strategy.value = currentValue;
}

function renderDetail() {
  const session = selectedSession();
  elements.noSelection.hidden = Boolean(session);
  elements.sessionDetail.hidden = !session;
  if (!session) {
    updateExpertControls(null);
    return;
  }
  elements.detailSessionId.textContent = session.session_id;
  elements.detailStatus.textContent = session.status === "active" ? "活动中" : session.status;
  elements.detailStatus.dataset.active = String(session.status === "active");
  elements.currentState.textContent = stateLabels[session.latest_state] || "尚无结果";
  elements.currentTrajectory.textContent = trajectoryLabels[session.latest_trajectory] || "—";
  elements.currentTaskProcess.textContent = taskProcessLabels[session.latest_task_process] || "—";
  elements.currentSupportNeed.textContent = supportNeedLabels[session.latest_support_need] || "—";
  elements.currentSupportTarget.textContent = supportTargetLabels[session.latest_support_target] || "—";
  elements.currentSpeakerBinding.textContent = speakerBindingLabel(session.latest_speaker_binding);
  elements.currentAction.textContent = actionLabels[session.latest_action] || "观察中";
  renderTimeline(session.timeline || []);
  updateExpertControls(session);
}

function speakerBindingLabel(binding) {
  if (!binding) return "等待录音";
  if (binding.bound === null) return binding.enrolled ? "已录音，等待识别" : "录音未完成";
  if (!binding.bound) return "本轮未能区分";
  const low = Number(binding.low_confidence_count || 0);
  if (binding.parent_count > 0 && binding.child_count > 0) return low > 0 ? `已区分双方（${low}段待复核）` : "已区分双方";
  return low > 0 ? `已识别部分语音（${low}段待复核）` : "已识别部分语音";
}

function describeEvent(event) {
  if (event.type === "state_update") {
    const parts = [stateLabels[event.state] || "未确定"];
    if (event.model_state && event.model_state !== event.state) {
      parts.push(`规则校正: ${stateLabels[event.model_state] || event.model_state} → ${stateLabels[event.state] || event.state}`);
    }
    if (Number.isFinite(event.active_stall_duration_ms)) {
      parts.push(`连续停滞: ${Math.round(event.active_stall_duration_ms / 1000)}秒`);
    }
    if (Number.isFinite(event.rolling_parental_prompt_rate_per_minute)) {
      parts.push(`催促频率: ${event.rolling_parental_prompt_rate_per_minute}/分钟`);
    }
    if (event.spontaneous_recovery) parts.push("30秒内自行恢复");
    if (event.trajectory) parts.push(`轨迹: ${trajectoryLabels[event.trajectory] || event.trajectory}`);
    if (event.task_process) parts.push(`任务: ${taskProcessLabels[event.task_process] || event.task_process}`);
    if (event.support_need) parts.push(`需要: ${supportNeedLabels[event.support_need] || event.support_need}`);
    if (event.support_target) parts.push(`对象: ${supportTargetLabels[event.support_target] || event.support_target}`);
    parts.push(actionLabels[event.action] || event.action);
    return parts.join(" · ");
  }
  if (event.type === "intervention") return `${event.source === "expert" ? "专家" : "AI"}：${event.message || "提示"}`;
  if (event.type === "speaker_enrollment") return event.complete ? "声音样本已建立；本地不保存录音，结束后删除云端声纹。" : "声音样本未完成。";
  if (event.type === "speaker_binding") return `${event.bound ? "本轮已区分" : "本轮未能区分"}；家长 ${event.parent_segment_count || 0} 段，儿童 ${event.child_segment_count || 0} 段，待复核 ${event.low_confidence_segment_count || 0} 段。`;
  if (event.type === "device_health") {
    const device = event.device === "camera" ? "摄像头" : "麦克风";
    return `${device}：${event.status === "normal" ? "正常" : "不正常"}`;
  }
  if (event.type === "voiceprint_cleanup") return event.remote_records_deleted ? "云端声纹已删除。" : "云端声纹删除失败，服务器将在关闭时重试。";
  if (event.type === "intervention_outcome") return `下一观察窗口：${stateLabels[event.observed_state] || "未确定"}；${event.effect_category || "待判断"}`;
  if (event.type === "family_response_received") return `反馈：${event.response}`;
  if (event.reason) return String(event.reason);
  if (event.message) return String(event.message);
  return "事件已记录";
}

function renderTimeline(items) {
  elements.timeline.replaceChildren();
  const reversed = [...items].reverse();
  if (!reversed.length) {
    const item = document.createElement("li");
    item.textContent = "等待首个系统事件…";
    elements.timeline.append(item);
    return;
  }
  for (const event of reversed) {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = event.observed_at ? new Date(event.observed_at).toLocaleTimeString("zh-CN", { hour12: false }) : "—";
    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = eventLabels[event.type] || event.type;
    const detail = document.createElement("p");
    detail.textContent = describeEvent(event);
    body.append(title, detail);
    item.append(time, body);
    elements.timeline.append(item);
  }
}

function updateExpertControls(session) {
  const available = Boolean(session && session.status === "active" && session.closed_loop_enabled);
  const takeover = Boolean(session?.expert_takeover_active);
  elements.takeoverState.textContent = takeover ? "专家已接管" : "未接管";
  elements.takeoverState.dataset.expert = String(takeover);
  elements.takeover.disabled = !available || takeover;
  elements.release.disabled = !available || !takeover;
  elements.strategy.disabled = !available || !takeover;
  elements.expertMessage.disabled = !available || !takeover;
  elements.sendIntervention.disabled = !available || !takeover;
}

function requiredContext() {
  const session = selectedSession();
  const operator = elements.operator.value.trim();
  const reason = elements.reason.value.trim();
  if (!session) return feedback("请先选择一个活动中的会话。", true), null;
  if (!operator || !reason) return feedback("专家标识和操作原因都必须填写。", true), null;
  return { session_id: session.session_id, operator, reason };
}

elements.loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  connect(elements.researchCode.value);
});
elements.refresh.addEventListener("click", () => send({ type: "refresh" }));
elements.takeover.addEventListener("click", () => {
  const context = requiredContext();
  if (context) send({ type: "expert_takeover", ...context });
});
elements.release.addEventListener("click", () => {
  const context = requiredContext();
  if (context) send({ type: "expert_release", ...context });
});
elements.strategy.addEventListener("change", () => {
  const strategy = strategies.find((item) => item.strategy_id === elements.strategy.value);
  elements.expertMessage.value = strategy?.approved_template || "";
});
elements.sendIntervention.addEventListener("click", () => {
  const context = requiredContext();
  if (!context) return;
  const strategyId = elements.strategy.value;
  const message = elements.expertMessage.value.trim();
  if (!strategyId || !message) return feedback("请选择策略并填写提示内容。", true);
  send({ type: "expert_intervention", ...context, strategy_id: strategyId, message });
});
