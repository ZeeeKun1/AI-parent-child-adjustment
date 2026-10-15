/* Hardware-free browser regression: real browser/media pipeline, simulated study/model endpoints. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE_PATH || "playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH,
    headless: true, args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
  try {
    const context = await browser.newContext({ permissions: ["camera", "microphone"], viewport: { width: 1365, height: 900 } });
    const page = await context.newPage();
    const errors = [];
    const controls = [];
    const recordings = [];
    const enrollments = [];
    let captureSocket;
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/session-admission", (route) => route.fulfill({ json: { session_token: "browser-smoke-token", bound_speakers: [] } }));
    await page.route("**/api/speaker-binding/**", async (route) => {
      const speaker = route.request().url().split("/").at(-1);
      const state = await page.evaluate(() => ({
        tracksReleased: window.captureTestStreams.every((stream) => stream.getTracks().every((track) => track.readyState === "ended")),
        contextsClosed: window.captureTestContexts.every((context) => context.state === "closed"),
      }));
      assert.equal(state.tracksReleased && state.contextsClosed, true);
      // Five seconds of real browser PCM should be present; never send it to an external API.
      const bytes = route.request().postDataBuffer();
      assert.ok(bytes.length >= 3 * 16000 * 2);
      enrollments.push({ speaker, bytes: bytes.length, state });
      await route.fulfill({ json: { bound_speakers: enrollments.map((e) => e.speaker), complete: enrollments.length === 2 } });
    });
    await page.route("**/api/session-recording/**", async (route) => {
      recordings.push({ sequence: Number(route.request().url().split("/").at(-1)),
        segment: Number(route.request().headers()["x-recording-segment-start"]), bytes: route.request().postDataBuffer().length });
      await route.fulfill({ json: { ok: true } });
    });
    await page.routeWebSocket("**/ws/**", (ws) => {
      captureSocket = ws;
      ws.onMessage((raw) => {
        if (typeof raw !== "string") return;
        const message = JSON.parse(raw);
        controls.push(message);
        if (message.type === "hello") ws.send(JSON.stringify({ type: "ready", protocol_version: 1,
          media_format: { audio_sample_rate: 16000, audio_chunk_bytes: 3200, audio_chunk_ms: 100,
            image_max_width: 640, image_max_height: 480, image_interval_ms: 1000 },
          max_image_bytes: 750000, session_media_retention_enabled: true, runtime_controls_enabled: true }));
        if (message.type === "start") ws.send(JSON.stringify({ type: "started" }));
        if (message.type === "delivery_visual") ws.send(JSON.stringify({ type: "delivery_visual_received", delivery_id: message.delivery_id }));
        if (message.type === "delivery_execution") ws.send(JSON.stringify({ type: "delivery_execution_received", delivery_id: message.delivery_id }));
        if (message.type === "stop") ws.send(JSON.stringify({ type: "summary", valid: true, run_id: "capture-smoke", experiment_condition: "coregulation" }));
        if (message.type === "ping") ws.send(JSON.stringify({ type: "pong" }));
      });
    });
    await page.goto(process.env.CAPTURE_URL || "http://127.0.0.1:8766/");
    await page.evaluate(() => {
      window.captureTestStreams = []; window.captureTestContexts = [];
      const acquire = CaptureSupport.acquire;
      CaptureSupport.acquire = async (options) => { const stream = await acquire(options); window.captureTestStreams.push(stream); return stream; };
      const create = CaptureSupport.createContext;
      CaptureSupport.createContext = () => { const context = create(); window.captureTestContexts.push(context); return context; };
    });
    await page.locator('[data-condition="coregulation"]').click();
    await page.locator("#parent-age").fill("35");
    await page.locator("#child-age").fill("9");
    await page.locator("#child-grade").fill("三年级");
    await page.locator("#task-name").fill("设备兼容性模拟试运行");
    await page.locator("#task-type").selectOption("mathematics");
    await page.locator("#task-difficulty").selectOption("moderate");
    await page.locator("#wizard-next").click();
    await page.locator('[data-role="mother"]').click();
    await page.locator('[data-role="boy"]').click();
    for (const speaker of ["parent", "child"]) {
      await page.locator(`.binding-button[data-speaker="${speaker}"]`).click();
      await page.waitForFunction((key) => bindingState[key] && !bindingBusy, speaker, { timeout: 20000 });
    }
    console.log(JSON.stringify({ phase: "voice_enrollment", recordings: enrollments.map(({speaker,bytes}) => ({speaker,bytes})) }));
    await page.locator("#wizard-next").click();
    await page.locator("#consent").check();
    await page.locator("#device-check-button").click();
    await page.waitForFunction(() => devicesReady() && !deviceCheckBusy, null, { timeout: 15000 });
    const deviceSettings = await page.evaluate(() => ({ sampleRate: checkedMediaStream.getAudioTracks()[0].getSettings().sampleRate,
      micOptions: document.querySelectorAll('select[data-input-kind="audioinput"] option').length }));
    assert.ok(deviceSettings.micOptions >= 2);
    console.log(JSON.stringify({ phase: "device_check", ...deviceSettings }));
    await page.locator("#start-button").click();
    await page.waitForFunction(() => captureActive && audioCount > 15 && imageCount > 1, null, { timeout: 15000 });
    // Catch regressions where hiding the setup modal stops decoding the sampling video.
    await page.waitForTimeout(12000);
    const running = await page.evaluate(() => ({ audio: audioCount, images: imageCount, health: { ...deviceHealth },
      frameAge: performance.now() - lastVideoDecodedAt }));
    assert.equal(running.health.camera, "ok"); assert.equal(running.health.microphone, "ok");
    assert.ok(running.frameAge < 5000);
    console.log(JSON.stringify({ phase: "running", ...running }));
    for (let i = 0; i < 2; i++) {
      const id = `smoke-delivery-${i}`;
      captureSocket.send(JSON.stringify({ type: "intervention", delivery_id: id, sequence: i + 1,
        target_actor: "both", repair_target: "task_support", heading: "一起试一下", message: "先说说卡在哪一步。",
        voice_expected: true, voice_pending: true, voice_streaming: true }));
      await page.waitForFunction((id) => pendingDeliveryExecution?.deliveryId === id, id);
      captureSocket.send(JSON.stringify({ type: "intervention_voice", delivery_id: id,
        voice_error: "simulated TTS disconnect", audio_sample_count: 0 }));
      await page.waitForFunction((id) => completedDeliveryIds.has(id), id);
      await page.locator("#dismiss-intervention").click();
    }
    assert.equal(controls.filter((m) => m.type === "delivery_execution").length, 2);
    console.log(JSON.stringify({ phase: "tts_failure", visual_receipts: controls.filter((m) => m.type === "delivery_visual").length,
      final_receipts: controls.filter((m) => m.type === "delivery_execution").length }));
    await page.evaluate(() => recoverCaptureDevices(true));
    await page.waitForTimeout(4500);
    await page.evaluate(() => monitorCaptureHealth());
    assert.equal(await page.evaluate(() => deviceHealth.camera), "ok");
    assert.equal(await page.evaluate(() => deviceHealth.microphone), "ok");
    await page.locator("#stop-button").click();
    await page.waitForFunction(() => !captureActive && !stopping, null, { timeout: 20000 });
    assert.ok(new Set(recordings.map((r) => r.segment)).size === 2);
    assert.deepEqual(recordings.map((r) => r.sequence), recordings.map((_, i) => i));
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ phase: "completed", recording_chunks: recordings.length, recording_segments: 2, page_errors: errors }));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
