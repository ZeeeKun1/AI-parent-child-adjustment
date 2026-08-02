# 当前交接：亲子共同调节系统——实时摄像头版本开发

更新时间：2026-08-03（Asia/Shanghai）

## 0. 新对话必须先做什么

请完整读取本文件，然后直接继续“第 11 节：Required next action”。新对话的主要任务是把当前的“本地上传视频测试版”扩展为正式实验可使用的“摄像头与麦克风实时输入版”。

不要重新设计已经完成的四个业务模块，也不要因为追求形式上的完美而逐项扩张需求。所有系统设计必须以本项目的形成性研究、三阶段主题分析、63 个 WOZ 干预片段和专家访谈结果为最高准则；工程参数若没有研究证据，只能标记为可配置的技术参数，不能伪装成研究结论。

用户偏好：严格回答当前问题，避免大段泛化说明和过多分点；先给结论，再给必要依据。

## 1. 仓库、分支与版本边界

本地项目根目录：

```text
H:\Doctor Work\AI 亲子陪伴调节\访谈\二阶段主题分析\coregulation-realtime-poc
```

GitHub：

```text
https://github.com/ZeeeKun1/AI-parent-child-adjustment
```

已经推送的测试版分支：

```text
prototype/local-video-poc
```

测试版提交：

```text
0297f954976674fed019117eaec9a573dd54a955
```

该分支明确对应“用户上传本地视频，再按真实时间回放给模型”的技术验证版本，不是正式实验的实时摄像头版本。远程 `main` 没有被修改。

实时版本开始编码前，建议从 `prototype/local-video-poc` 创建新分支：

```text
feature/realtime-camera-pipeline
```

不要在 `main` 上直接开发，也不要覆盖或删除 `prototype/local-video-poc`。

## 2. 当前系统已经完成什么

### 模块一：多模态共同调节状态识别

- 本地视频通过 PyAV 解码。
- 音频转换为 16 kHz、单声道、16-bit PCM，每块约 100 ms。
- 视频约每秒抽取一张 JPEG，最大 1280×720，并带可追溯的 `frame_time_ms`。
- 音频块和图像帧按照统一时间线实时回放给 Qwen-Omni-Realtime。
- 输出四种形成性研究状态：`normal`、`fluctuation`、`dysregulation`、`high_risk`。
- 音频证据要求保留模型实际观察到的原话；视觉证据只描述可见行为并引用时间帧。
- 某个模态证据不足时明确写证据不足，不强迫每次判断同时具备音频和视觉证据。
- 状态边界允许保留 `alternative_state` 与 `ambiguity_reason`，不制造现实中不存在的绝对边界。
- 本地 Praat/Parselmouth 只记录音高、强度、dBFS、浊音帧等客观声学特征，不单独推断情绪或状态。

关键文件：

```text
config/state_codebook.yaml
config/acoustic_analysis.yaml
src/coregulation_poc/capture/video_replay.py
src/coregulation_poc/providers/qwen_omni_realtime.py
src/coregulation_poc/providers/websocket_transport.py
src/coregulation_poc/fusion/prompting.py
src/coregulation_poc/fusion/response_parser.py
src/coregulation_poc/acoustics/prosody.py
src/coregulation_poc/video_test.py
```

### 模块二：连续状态与干预时机

- 保存连续状态轨迹，不覆盖模块一判断。
- `normal` 不干预；`fluctuation` 继续观察；`dysregulation` 允许显式干预；`high_risk` 进入渐进支持。
- 干预需要等待自然互动边界。
- 干预后必须观察到亲子回应，才能再次决定是否干预。
- 没有将固定秒数或固定事件次数包装成研究结论。

关键文件：

```text
config/intervention_policy.yaml
src/coregulation_poc/control/
src/coregulation_poc/trajectory_test.py
```

### 模块三：策略选择

- 只有模块二授权后才能选择策略。
- 12 张版本化策略卡分别面向家长、孩子或双方。
- 保存目标对象、修复目标、批准话术、预期恢复表现、下一策略和研究来源。
- `unknown` 或笼统的双方证据不能被用来单独指责家长或孩子。
- 模块三只决定干预内容与对象，不决定输出媒介。

关键文件：

```text
config/strategy_cards.yaml
src/coregulation_poc/intervention/
src/coregulation_poc/strategy_test.py
```

### 模块四：文字与语音双通道输出

- 同一条批准话术同时用于醒目文字和语音，不允许两个通道各自改写。
- 文字醒目但不遮挡主要任务。
- 暂停干预时两个通道都暂停。
- 语音失败时保留文字并明确记录降级。
- “成功显示/播放”不等于参与者已经看到、听到、理解或采纳。
- TTS 固定为 `qwen3-tts-instruct-flash-realtime-2026-01-22`，音色固定为 `Maia`。
- `optimize_instructions=false`，避免服务端重写实验语音指令。
- 生成 24 kHz、单声道、16-bit WAV，并保存消息哈希、音频哈希、延迟、字符用量和清理后的事件记录。
- 浏览器预览只播放已生成的 Maia 文件，不再随机调用系统中文声线。

关键文件：

```text
config/delivery_policy.yaml
src/coregulation_poc/delivery/
src/coregulation_poc/providers/qwen_tts_realtime.py
src/coregulation_poc/delivery_test.py
```

## 3. 已验证状态

本地完整检查：

```text
ruff: all checks passed
pytest: 54 passed
```

真实本地视频推理已经成功，样例片段约 38 秒，解码为 380 个音频块与 38 张图像帧。模块一得到结构有效的 `fluctuation` 结果；该结果是管线可行性证据，不是专家确认的准确率证据。当前一次运行的 `audit_ready=false`，原因是最佳努力 ASR 转录没有完整匹配模型引用的音频原话。

真实 TTS 已经全通：

```text
model: qwen3-tts-instruct-flash-realtime-2026-01-22
voice: Maia
sample_rate_hz: 24000
voice_synthesis_count: 1
voice_synthesis_failure_count: 0
first_audio_latency_ms: 562
total_latency_ms: 1734
audio_duration_ms: 4720
```

本地运行目录均在 `data/output/`，已被 Git 忽略，不要上传研究媒体或运行音频。

## 4. 当前版本与实时实验版本的关键差距

当前 `video-test` 会先完整解码一个本地文件，然后按照原始时间戳回放给模型。它证明了模型、提示词、四状态结构、证据审计和后续模块能够运行，但还缺少正式实验需要的以下实时能力：

1. 从摄像头和麦克风持续采集，而不是读取上传文件。
2. 设备发现、设备选择、权限错误和中途断开处理。
3. 有界缓冲与背压，避免长时实验无限占用内存。
4. 持续或滚动地生成模块一评估，而不是每段视频只返回一次。
5. 将连续评估实时送入模块二、模块三和模块四。
6. 将文字提示和 Maia 音频发送到正式实验前端，并接收真实执行回执。
7. 在前端开始阶段让家长和孩子分别确认音频身份，并通过声纹分段绑定说话者。

## 5. 实时版本应采用的总体数据流

```text
摄像头 + 麦克风
        ↓
统一单调时钟与时间戳
        ↓
有界音频/视频队列
        ↓
Qwen-Omni-Realtime 持续会话
        ↓
模块一结构化状态评估
        ↓
模块二连续轨迹与时机控制
        ↓
模块三对象化策略选择
        ↓
模块四醒目文字 + Maia 语音
        ↓
前端执行回执与后续亲子回应
```

实时采集层只负责真实、同步、可追溯地提供媒体，不得在采集层重新定义状态或干预逻辑。

## 6. 实时采集的工程要求

### 音视频格式

第一版应尽量保持与已经验证的本地回放输入一致，以减少同时变化的变量：

- 音频：16 kHz、mono、PCM16、约 100 ms 一块。
- 视频：JPEG，最大 1280×720，初始约 1 fps。
- 音频和视频使用同一个单调时钟生成毫秒时间戳。

100 ms、1 fps、分辨率和评估窗口属于工程参数，不是形成性研究结论。它们必须可配置、写入每次运行清单，并在延迟与识别效果验证后再冻结。

### 缓冲与长时运行

- 使用有界队列，队列满时明确记录丢帧或降采样，不能静默无限堆积。
- 音频优先保持连续；视频在过载时可按规则丢弃较旧帧。
- 所有媒体块必须具有严格递增时间戳。
- 设备断开、API 断开和超时必须产生显式状态，不得伪装成正常评估。
- 原始音视频默认不落盘；只有明确进入经批准的研究记录流程时才能保存。

### 持续评估

模块一的评估频率与媒体窗口长度目前没有形成性研究给出的唯一数值。第一版需要做成可配置调度器，并将窗口起止、触发原因、媒体数量和延迟写入审计记录。

固定技术窗口只能用于系统调度，不能直接成为干预阈值。是否干预仍必须由模块二依据研究政策、自然互动边界和干预后回应决定。

## 7. 说话者绑定的既定方向

正式前端开始会话时，让家长与孩子分别进行一次短音频确认/注册，然后由声纹分段将后续语音片段绑定为 `parent` 或 `child`。

必须遵守：

- 声纹置信不足、重叠说话或环境噪声严重时回退为 `unknown`。
- 不能为了让输出看起来完整而强行指定说话者。
- 客观声学特征按分段保存；语义内容继续交给多模态大模型理解。
- 当前实时采集基础层应预留 `speaker_segment` 接口，但不要在没有选定和验证声纹方案前伪造实现。

## 8. 不得破坏的研究与产品约束

1. 最终四状态主题分析模型是权威来源，不能退回早期三状态草案。
2. 音频证据尽量使用观察到的原话；视觉不可见时写证据不足。
3. 不要求每个判断同时拥有音频和视觉证据。
4. 客观音高、强度、语速等不能单独推断情绪或共同调节状态。
5. `normal` 与 `fluctuation` 不触发干预。
6. 状态边界允许不确定性，不追求现实中不存在的绝对分类边界。
7. 模块二决定是否干预，模块三决定对谁说什么，模块四只负责呈现。
8. 文字与语音必须保持同一条核心话术。
9. 固定使用 Maia，避免音色变化成为实验混杂变量。
10. 不把系统输出成功等同于参与者实际接收或恢复。

## 9. 安全、隐私与 Git 约束

- `.env` 中已配置北京地域的 API Key、Workspace ID 和模型设置。禁止打印、复制、提交或写入日志。
- `.env`、`data/input/`、`data/output/`、音视频文件、缓存和虚拟环境已被 `.gitignore` 排除。
- Base64 音频和图像负载不进入审计文件。
- 新增实时采集后，默认只保存结构化事件、参数、哈希、延迟与错误，不保存原始家庭音视频。
- 未获得覆盖云端处理与音视频记录的知情同意前，不对真实参与者运行。

## 10. 当前可用命令

```powershell
# 配置检查
python -m coregulation_poc doctor

# 完整本地检查
python -m ruff check src tests scripts
python -m pytest

# 本地视频真实推理
python -m coregulation_poc video-test `
  --video "H:\absolute\path\to\clip.mp4" `
  --session-id P01_test_01

# 模块二至四离线回放
python -m coregulation_poc delivery-test `
  --input "H:\absolute\path\to\strategy_replay.json"

# 真实生成 Maia 音频
python -m coregulation_poc delivery-test `
  --input "H:\absolute\path\to\strategy_replay.json" `
  --synthesize-voice
```

实时摄像头命令尚未实现。建议新增：

```text
python -m coregulation_poc live-test --camera-index ... --microphone-index ...
```

参数名称可在实现时根据 Windows 设备枚举方式调整，但不要把机器专属设备名称硬编码进仓库。

## 11. Required next action

新对话应直接执行以下任务：

1. 检查工作区与 Git 状态，以 `prototype/local-video-poc` 为基线创建 `feature/realtime-camera-pipeline`。
2. 先实现实时采集基础层，不先做正式 UI：
   - 定义可复用的媒体源接口，使本地视频源和实时设备源可以输出同一种带时间戳媒体块；
   - 增加 Windows 摄像头与麦克风设备枚举和明确选择；
   - 实现有界音频/视频队列、单调时间戳、停止信号和设备错误；
   - 新增 `live-test` 的短时 `--dry-run`，只验证采集、同步、格式、队列和指标，不调用付费 API；
   - 默认不保存原始音视频。
3. 为实时采集层编写可使用假设备/合成媒体源运行的测试，避免 CI 依赖真实摄像头。
4. 本地采集验证通过后，再把同一媒体块接口接入现有 Qwen Provider，完成一次短时真实摄像头推理。
5. 之后才实现持续评估调度、模块二至四在线串联、前端双通道呈现和声纹绑定。

首个里程碑的验收标准：

- 能列出并明确选择摄像头与麦克风；
- 能连续采集至少一个可配置短时会话；
- 输出音频格式与当前 Qwen 输入一致；
- 音视频时间戳统一且严格递增；
- 队列有界，停止后线程/设备完整释放；
- 设备不可用或断开时给出可诊断错误；
- 运行清单记录设备标识的非敏感部分、采集参数、块数量、丢帧数量和延迟；
- 默认不生成原始音视频文件；
- 不修改四状态代码本、干预政策、策略卡或固定 Maia 输出规则。

## 12. 后续而非首个任务

以下工作重要，但不要与首个实时采集里程碑同时展开：

- 正式参与者前端与视觉样式完善；
- 家长/孩子声纹模型选型和注册流程；
- 长时压力测试和断线重连；
- 多段专家标注视频的分类准确性验证；
- 摄像头角度、遮挡、远场拾音和多人重叠语音实验；
- CHI 稿件所需的用户实验、消融实验和统计分析。

当前最重要的下一步只有一个：在不改变已有科学逻辑的前提下，把经过验证的本地视频媒体接口替换/扩展为可靠的实时摄像头与麦克风媒体接口。
