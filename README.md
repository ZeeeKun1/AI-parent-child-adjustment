# Parent–Child Co-regulation Realtime PoC

本项目用于验证实时音视频API能否依据形成性研究编码，识别亲子作业陪伴中的四种共调节状态：`normal`、`fluctuation`、`dysregulation`和`high_risk`。

当前仓库只包含独立、可迁移的技术验证骨架，不包含正式前端UI，也不会将参与者音视频或API密钥纳入版本控制。

当前实时开发分支已经加入Windows摄像头与麦克风采集基础层。它通过DirectShow明确枚举和选择设备，将音频转换为16 kHz单声道PCM16、约100 ms一块，将视频压缩为最大1280×720且带时间标签的JPEG，并用有界队列记录视频丢帧和音频背压。首个`live-test --dry-run`只验证设备、格式、同步、队列和释放，不调用API，也不保存原始音视频。

## 实时摄像头与麦克风采集测试

先列出Windows可以访问的设备：

```powershell
python -m coregulation_poc live-test --list-devices
```

从列表中明确选择一个摄像头和麦克风，进行10秒采集：

```powershell
python -m coregulation_poc live-test `
  --camera-index 0 `
  --microphone-index 0 `
  --duration-seconds 10 `
  --session-id P01_live_device_check `
  --dry-run
```

每次运行只在`data/output/runs/<run_id>/`保存`manifest.json`、`events.jsonl`、`metrics.json`和`result.json`。这些文件包含非敏感设备信息、采集参数、时间戳、块大小、队列深度、丢帧数和错误，不包含PCM、JPEG、Base64负载或可回放媒体。参数均为可配置工程初值，不代表形成性研究阈值。详细说明见[实时采集说明](docs/live-capture.md)。

## 单个视频测试

先复制并填写`.env`中的`DASHSCOPE_API_KEY`和`ALIYUN_WORKSPACE_ID`。视频应放在被 Git 忽略的`data/input/`中，详细要求见[视频片段准备规范](docs/video-preparation.md)。

遇到连接或输入错误时，先运行完整诊断。程序会逐项输出配置、媒体解码、DNS、TCP、WebSocket鉴权、会话更新和音视频缓冲区结果，并将报告写入`data/output/diagnostics/`：

```powershell
python -m coregulation_poc diagnose --video "C:\absolute\path\to\clip.mp4"
```

先运行不调用 API 的预检：

```powershell
$projectRoot = (Get-Location).Path
$videoPath = (Join-Path $projectRoot "data\input\P01_normal_01.mp4")
python -m coregulation_poc video-test --video $videoPath --session-id P01_normal_01 --dry-run
```

预检通过后执行真实调用：

```powershell
python -m coregulation_poc video-test --video $videoPath --session-id P01_normal_01
```

每次运行都会在`data/output/runs/<run_id>/`生成独立证据目录，其中包含输入哈希、模型与编码表版本、实际提示词、收发事件、转录事件、本地声学测量、延迟指标和经 schema 校验的状态结果。音视频字节和 API 密钥不会写入日志。

结构化结果将音频和视频证据分别记录。音频证据必须引用观察到的原话；视频证据必须引用画面中的`frame_time_ms`标签并描述直接可见的行为。任一模态均可单独标记为证据不足，不要求每次判断都同时具备两种模态证据。`audit.json`和`metrics.json`会将分类结构有效性与证据审计完整性分开报告；ASR失败不会再被隐藏为完全成功。

本地声学层使用 Praat/Parselmouth 测量音高分布、有声帧比例、强度和 dBFS，不输出情绪或共调节状态。Qwen 输入转录事件中的情绪标签另存为辅助观察。当前回放音轨是未绑定说话者的混合单声道，因此声学结果标记为 `limited` 和 `actor=unknown`；待前端声纹绑定接入后再按家长、儿童片段计算说话者内变化。任何单一声学特征都不能触发状态分类。

主要产物包括：

- `assessment.json`：结构化状态、分类置信度、备选状态、歧义原因和逐模态证据；
- `audit.json`：转录状态、审计警告和音频原话匹配结果；
- `acoustic_summary.json`：完整混合音轨的客观声学测量及解释限制；
- `acoustic_evidence.json`：模型所引音频证据区间的声学测量；
- `input_emotions.json`：Qwen 转录事件返回的情绪标签，明确仅作辅助观察；
- `metrics.json`：分类有效性、审计状态和延迟；
- `input_transcript_best_effort.txt`：从流式转录事件保留的最长可用文本；
- `events.jsonl`、`transcription_events.json`、`manifest.json`和`prompt.txt`：完整追溯材料。

## 连续状态轨迹与介入时机测试

模块二接收模块一连续产生的 `StateAssessment`，将其组织为状态轨迹，并依据形成性研究规则输出：不介入、继续观察、明确介入、渐进支持或暂缓决定。正常和波动状态不触发干预；失调和高风险状态只有在自然话轮边界才允许进入策略选择；介入后必须先观察亲子回应。

测试输入是一个 JSON 文件，包含同一会话的 `observations`。每个 observation 由模块一的 `assessment`、`natural_turn_boundary`、`post_intervention_response_observed` 和 `interaction_history_available` 组成。运行：

```powershell
python -m coregulation_poc trajectory-test --input "C:\absolute\path\to\trajectory.json"
```

运行目录保存 `intervention_policy.json`、`observations.json`、`decisions.json`、`state_trajectory.json` 和 `result.json`。策略生成不属于模块二；只有 `strategy_selection_required=true` 时，后续专家策略模块才可以选择具体策略。

## 干预策略选择测试

模块三只在模块二允许介入后运行，依据模块一的原始证据和互动表现，从版本化专家策略卡中选择干预对象、修复目标、模板话术和预期恢复证据。策略卡分别面向家长、儿童或双方；`unknown` 或笼统的 `both` 证据不能用于单独指向某个人。输出媒介由模块四统一处理，不再由策略卡分别决定。

不调用 API 的完整回放：

```powershell
$inputPath = (Resolve-Path ".\examples\strategy_replay.json").Path
python -m coregulation_poc strategy-test --input $inputPath
```

运行目录保存 `strategy_library.json`、`strategy_selections.json` 和 `intervention_plans.json`，同时保留模块二的决策与状态轨迹。当前使用专家审查模板；以后接入大模型时只允许在卡片约束内调整措辞，校验失败仍回退到模板。详细设计见 [模块三策略选择说明](docs/strategy-selection.md)。

## 文字与语音双通道干预测试

模块四只接收模块三已经授权并选定的话术，将同一条核心内容转换为醒目但不遮挡主要任务的文字提示，并同步生成自动语音播报指令。正常路径要求两种输出同时可用；语音不可用时保留文字并明确标记降级，暂停干预时两种输出都不执行。

不调用 API 的模块一至四回放：

```powershell
$inputPath = (Resolve-Path ".\examples\strategy_replay.json").Path
python -m coregulation_poc delivery-test --input $inputPath

# 真实生成并保存固定模型与 Maia 音色的语音
python -m coregulation_poc delivery-test --input $inputPath --synthesize-voice
```

运行目录保存 `delivery_policy.json`、`delivery_packages.json`、`voice_synthesis_results.json`、待前端填写的 `delivery_execution_reports.json` 和可直接用浏览器打开的 `delivery_preview_001.html`。默认命令只生成离线预览；加 `--synthesize-voice` 后，系统使用固定快照模型 `qwen3-tts-instruct-flash-realtime-2026-01-22` 与固定音色 `Maia` 生成 `delivery_audio_001.wav`，预览页只播放该文件，不再调用电脑自带声线。实际实验前端需要把界面呈现、语音播放结果和参与者后续回应分别回写。详细设计见 [模块四双通道输出说明](docs/intervention-delivery.md)。

## 目录

```text
config/                    四状态编码表、介入规则、声学规则、策略卡与输出策略
data/                      本地输入、输出、缓存和日志（默认不上传GitHub）
docs/                      架构与数据治理说明
scripts/                   独立测试脚本入口
src/coregulation_poc/      可复用Python代码
tests/                     单元测试、集成测试和本地视频夹具
.github/workflows/         GitHub Actions持续集成
```

## 路径约定

配置文件中可以书写相对路径，但进入程序后必须通过`coregulation_poc.paths.resolve_project_path()`转换为绝对路径。项目根目录由当前代码文件的位置自动推导，因此整个文件夹移动后无需修改硬编码盘符。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m coregulation_poc doctor
pytest
```

## GitHub上传前

1. 确认`.env`中已配置密钥，但没有被Git跟踪。
2. 不要提交任何家长或儿童的音频、视频、转录原文和可识别身份信息。
3. 运行`ruff check .`与`pytest`。
4. 使用`git status --ignored`再次检查待上传内容。

项目暂未附带开源许可证；在公开仓库前应由研究团队确定许可证与数据使用政策。
