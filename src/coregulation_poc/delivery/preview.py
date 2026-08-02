from __future__ import annotations

import json

from coregulation_poc.delivery.models import DeliveryPackage


def render_delivery_preview(
    package: DeliveryPackage,
    *,
    audio_filename: str | None = None,
) -> str:
    """Render a dependency-free preview that plays the saved Maia recording."""
    payload = {
        "deliveryId": package.delivery_id,
        "heading": package.visual_prompt.heading,
        "message": package.visual_prompt.message,
        "voiceEnabled": package.voice_prompt.enabled,
        "autoplay": package.voice_prompt.autoplay,
        "targetActor": package.target_actor.value,
        "provider": package.voice_prompt.provider,
        "model": package.voice_prompt.model,
        "voice": package.voice_prompt.voice,
        "audioUrl": audio_filename,
    }
    serialized = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>模块四双通道干预预览</title>
  <style>
    :root {{ color-scheme: light; font-family: "Microsoft YaHei", system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #f4f1ea; color: #1c2924; }}
    .task {{ max-width: 900px; margin: 56px auto; padding: 40px; opacity: .48; }}
    .task h1 {{ margin: 0 0 18px; font-size: 28px; }}
    .task p {{ line-height: 1.8; }}
    .prompt {{
      position: fixed; left: 50%; bottom: 32px; z-index: 20; transform: translateX(-50%);
      width: min(760px, calc(100vw - 32px)); border: 3px solid #173f34; border-radius: 20px;
      background: #fffdf7; box-shadow: 0 18px 60px rgba(19, 54, 45, .24);
      padding: 24px 72px 24px 28px;
    }}
    .prompt h2 {{ margin: 0 0 10px; color: #155844; font-size: 22px; }}
    .prompt p {{ margin: 0; font-size: 24px; line-height: 1.55; font-weight: 650; }}
    .actions {{ display: flex; gap: 10px; margin-top: 18px; }}
    button {{
      border: 0; border-radius: 999px; padding: 10px 16px; font: inherit; cursor: pointer;
    }}
    .replay {{ background: #173f34; color: white; }}
    .close {{ position: absolute; right: 18px; top: 16px; background: #e8eee9; color: #173f34; }}
    .status {{ margin-top: 10px; color: #5a6962; font-size: 13px; }}
  </style>
</head>
<body>
  <main class="task" aria-hidden="true">
    <h1>亲子共同任务界面示意</h1>
    <p>模块四只覆盖干预呈现层。正式前端可在不遮挡主要任务的前提下复用同一份输出载荷。</p>
  </main>
  <section class="prompt" id="prompt" role="alert" aria-live="polite">
    <button class="close" id="close" aria-label="关闭提示">关闭</button>
    <h2 id="heading"></h2>
    <p id="message"></p>
    <div class="actions"><button class="replay" id="replay">再次播报</button></div>
    <div class="status" id="status"></div>
  </section>
  <script>
    const payload = {serialized};
    const status = document.getElementById("status");
    const audio = payload.audioUrl ? new Audio(payload.audioUrl) : null;
    document.getElementById("heading").textContent = payload.heading;
    document.getElementById("message").textContent = payload.message;

    async function speak() {{
      if (!payload.voiceEnabled) {{
        status.textContent = "语音已禁用，文字提示仍然保留。";
        return;
      }}
      if (!audio) {{
        status.textContent = `尚未生成 ${{payload.voice}} 音频；请使用 --synthesize-voice 生成。`;
        return;
      }}
      audio.pause();
      audio.currentTime = 0;
      try {{
        await audio.play();
      }} catch (error) {{
        status.textContent = "浏览器阻止了自动播放，请点击“再次播报”。";
      }}
    }}

    if (audio) {{
      audio.addEventListener("play", () => {{
        status.textContent = `正在播报 · ${{payload.voice}}`;
      }});
      audio.addEventListener("ended", () => {{
        status.textContent = "播报完成；这不代表参与者已经听到、理解或采纳。";
      }});
      audio.addEventListener("error", () => {{
        status.textContent = "Maia 音频加载失败，文字提示仍然保留。";
      }});
    }}

    document.getElementById("replay").addEventListener("click", speak);
    document.getElementById("close").addEventListener("click", () => {{
      if (audio) {{ audio.pause(); audio.currentTime = 0; }}
      document.getElementById("prompt").hidden = true;
    }});
    if (payload.autoplay) window.addEventListener("load", () => setTimeout(speak, 100));
  </script>
</body>
</html>
"""
