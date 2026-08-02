# 干预输出与执行方式：文献证据地图

更新时间：2026-08-02

## 当前结论

第二轮针对“亲子互动进行中如何实时输出干预”的精准检索，找到了比一般通知和数字健康研究更直接的证据。但本项目最直接的证据不是外部文献，而是形成性研究和 WOZ 实验中专家已经实际采用语音实施干预。研究团队在 2026-08-02 最终确定首版使用“醒目文字提示 + 同步语音播报”：语音保持与专家干预的设计连续性，文字用于降低漏听和误解风险。

1. 干预渠道、时机和强度都应当作为可调整的干预选项，而不是由策略类别永久绑定。
2. 干预应避开高打断成本的时刻，并考虑接收者当下是否有能力和意愿处理干预。
3. 首版复现 WOZ 的专家语音干预，同时将同一条核心内容醒目显示在屏幕上；两种媒介不能分别生成相互矛盾的内容。
4. 语音内容仍需区分针对家长、儿童或双方；“谁是策略对象”与“现场谁能够听见”必须分别记录。
5. 外部研究提示语音误报可能增加打断并损害信任，因此双通道输出必须受模块二的时机授权、置信与安全回退约束；语音不可用时保留文字并明确记录降级。

## 与本项目最直接的证据

| 研究 | 干预方式 | 主要结果 | 证据判断 |
|---|---|---|---|
| Romanowicz 等，JAMA Network Open 2025，随机试验，50 个家庭 | 儿童佩戴智能手表；AI 预测即将发生的情绪爆发；家长手机收到实时提示，再使用已经学习的 PCIT 技能介入 | 家长打开提示的中位时间为 3.65 秒；数字增强组的平均情绪爆发时长较短，但标准化行为量表的组间差异不显著；研究的首要目标是可行性 | 对“AI 检测 → 家长手机提示 → 家长介入”路径具有目前最高的直接支持；不能据此宣称完整疗效已经确立 |
| Wang 等，Applied Sciences 2025，36 对亲子 | 在模拟家庭作业任务中检测儿童压力；家长侧电脑屏幕显示压力并给出鼓励话术，家长据此与儿童互动 | 加入家长提示后，儿童心理压力指标下降，亲子沟通改善；生理压力变化较弱 | 场景与本项目最接近；但无提示条件固定在前、提示条件固定在后，存在顺序和练习效应，因果证据为中低强度 |
| Song 等，UbiComp 2016，TalkLIME | 在亲子互动中直接比较事后反馈、耳机语音实时反馈和家长手机屏幕实时反馈；实验中特意加入 10%–20% 错误反馈 | 耳机语音即时但所有 9 名家长均认为最具打断性，尤其在误报时；屏幕反馈较少打断且最能容忍误差。随后 8 个家庭的 6 周测试显示儿童主动发起对话比例改善 | 对“屏幕还是耳机”这一输出选择最直接；样本很小，但对自动识别系统的误报风险尤其有价值 |
| Kwon 等，CHI 2022，Captivate! | 结合实时视觉与语音情境，在亲子游戏中向家长平板提供情境相关的短语卡片 | 证明实时、情境化、面向家长的可视语言建议可以嵌入亲子互动 | 支持家长侧短提示的可行性；目标是语言促进，不是情绪失调 |
| Shanley 与 Niec，2010，随机分配 60 名母亲 | 将亲子互动中的实时教练反馈与无教练反馈比较 | 即使没有大量课前讲授，实时教练反馈也能促进家长技能习得 | 支持“互动中给家长即时、可行动反馈”这一机制；实施者是人类教练而非 AI |

这些外部研究说明实时家长提示是一条可行的替代或辅助路径，也提醒我们控制语音的打断与误报风险；但它们不能覆盖本项目已经形成的语音 WOZ 证据。最终系统应优先忠实复现专家语音干预，再通过技术约束降低自动化语音带来的新增风险。

## 证据与可支持结论

| 设计问题 | 代表性证据 | 对本项目的直接性 | 可以支持的结论 | 不能支持的结论 |
|---|---|---:|---|---|
| 是否应固定一种输出渠道 | Nahum-Shani 等将内容、来源、剂量和媒介都定义为 JITAI 的“干预选项”，并指出接收性取决于内容、媒介、剂量和时机 | 间接；理论与跨场景证据 | 渠道应与状态、对象、情境和接收性共同决策 | 某一渠道在所有家庭和状态下最优 |
| 是否等待较自然的介入点 | Park 等在 10 个朋友群体的用餐互动中发现，基于社交断点延迟通知可减少 54.1% 的打断；Mehrotra 等的在场研究显示呈现方式、提醒类型、关系和当前任务都会影响干扰感 | 中等偏低；真实社交互动，但不是亲子作业冲突 | 不应在任意时刻立即插入；需要检测对话/任务断点和接收性 | “长停顿”就是本项目唯一或充分的自然话轮边界 |
| 是否可向家长私密实时指导 | Comer 等的随机试验使用摄像头和家长佩戴的蓝牙耳机，在家实时开展 I-PCIT；远程组与门诊组均取得积极结果，且满意度较高 | 中等；直接涉及亲子互动和私密实时家长指导，但由治疗师实施，儿童年龄和临床情境不同 | 私密、实时、面向家长的指导是一条有依据的候选路径 | AI 自动指导与治疗师指导等效；私密文字优于私密语音 |
| 家长文字干预是否可行 | MyTeen 随机试验中，221 名家长接受每日短信后，家长胜任感、亲子沟通和压力等指标优于对照组；发送时间由家长选择，并可随时停止 | 中等偏低；对象是家长，但属于异步预防项目而非互动中的即时介入 | 简短文字可以低负担地支持家长；应允许用户选择时机和停止 | 在冲突发生时立即显示文字最有效，或一定不会分散家长注意力 |
| 共享语音能否促进亲子互动 | Leech 等对 20 个亲子家庭进行一个月测试，加入会话代理后亲子阅读对话量接近无代理条件的两倍；TaleMate 的 11 对亲子研究也显示共同语音代理可支持参与和故事回忆 | 低；属于平静、结构化的共同阅读 | 共享语音可作为合作性任务中的候选形式，且应补充而非替代亲子互动 | 共享语音适合情绪失调、高风险或正在冲突的时刻 |
| 共享语音会不会产生副作用 | Beneteau 等对 10 个家庭开展 4 周部署，发现智能音箱既可能促进沟通和辅助养育，也会造成偶发冲突和访问干扰 | 中等偏低；家庭原位研究，但不是干预系统 | 共享语音不是中性管道，需要记录冲突、权力和访问后果 | 公开播放天然比私密渠道公平、可信或有效 |
| 语音是否天然更易接受 | 面向家长心理健康支持的 Alexa 可行性研究中，家长认可免手操作和互动性，但社区样本保留率仅 49.1%，并报告安装、识别、机械语音和隐私问题 | 中等偏低；家长与儿童心理健康相关，但非实时亲子互动 | 语音具有便利性，同时必须提供隐私说明、停止控制和失败回退 | 语音优于文字，或家庭会长期稳定使用 |
| 是否需要接收性判断 | Mishra 等在自然环境中部署接收性模型，相比随机发送，接收性最高提升约 40% | 低；数字健康单人场景 | “需要干预”与“现在适合接收”应分开判断 | 该模型或指标可以直接用于亲子作业场景 |

## 对当前系统设计的约束

基于项目内证据、外部文献和研究团队最终决定，模块三继续决定干预对象、修复目标和策略内容；模块四的首版执行路径收敛为“受约束的醒目文字 + 实时语音”。模块四应满足以下约束：

- 屏幕文字和语音使用同一条核心内容，并保存专家模板、显示文本和实际播放内容之间的对应关系；
- 将干预需要、目标对象、实际接收者、输出媒介和介入时机分别记录；
- 在执行前重新检查自然话轮边界、当前接收性以及用户是否暂停或关闭干预；
- 分别记录文字是否成功呈现、语音是否成功播放、用户是否主动确认以及随后是否出现回应；呈现或播放成功不能推断为已经看到、听到、理解或采纳；
- 根据策略卡明确称呼家长、儿童或双方，不让语音对象含混；
- 语音不可用时保留文字提示并记录降级，不把单通道失败伪装成完整成功；
- 在形成性研究没有直接支持的地方，使用可配置或实验分配，而不是固定规则。

当前工程处理是：模块四生成同文的文字与语音载荷，固定使用 `qwen3-tts-instruct-flash-realtime-2026-01-22` 与 `Maia`，将模块三批准的原文直接合成为可审计 WAV；浏览器预览只播放该文件，不再临时选择操作系统声线。正式前端仍需分别回写文字呈现与音频播放结果。真实语音呈现的可接受性、打断感和恢复效果仍需在系统集成与用户实验中验证。

## 证据缺口与后续验证

检索中尚未发现直接研究回答以下问题：自主 AI 在真实家庭作业冲突中，分别通过家长私密文字、家长私密音频或亲子共享语音实时介入，哪一种能更快恢复共同调节且副作用更少。

因此，文字与语音组合的可接受性和效果仍是后续原型测试与正式实验的研究问题。至少测量：感知打断、被责备感、对 AI 的信任、实际采纳、恢复结果、儿童能动性和家长权威关系。该验证属于本项目需要产生的证据，而不是可以完全由外部文献替代的证据。

## 主要文献

1. Nahum-Shani, I., et al. (2018). *Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support*. Annals of Behavioral Medicine, 52, 446–462. https://doi.org/10.1007/s12160-016-9830-8
2. Mishra, V., et al. (2021). *Detecting Receptivity for mHealth Interventions in the Natural Environment*. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 5(2), Article 74. https://doi.org/10.1145/3463492
3. Park, C., et al. (2017). *“Don't bother me. I'm socializing!”: A breakpoint-based Smartphone notification system*. CSCW 2017, 541–554. https://doi.org/10.1145/2998181.2998189
4. Mehrotra, A., et al. (2016). *My Phone and Me: Understanding People's Receptivity to Mobile Notifications*. CHI 2016, 1021–1032. https://doi.org/10.1145/2858036.2858566
5. Comer, J. S., et al. (2017). *Remotely delivering real-time parent training to the home: An initial randomized trial of Internet-delivered parent-child interaction therapy*. Journal of Consulting and Clinical Psychology, 85(9), 909–917. https://doi.org/10.1037/ccp0000230
6. Chu, J. T. W., et al. (2019). *Effect of MyTeen SMS-Based Mobile Intervention for Parents of Adolescents: A Randomized Clinical Trial*. JAMA Network Open, 2(9), e1911120. https://doi.org/10.1001/jamanetworkopen.2019.11120
7. Leech, K., et al. (2025). *Evaluating the Usability of a Conversational Agent to Enhance Parent-Child Shared Reading Interactions*. International Journal of Child-Computer Interaction, 43, 100720. https://doi.org/10.1016/j.ijcci.2024.100720
8. Vargas-Diaz, D., et al. (2025). *Exploring parent involvement in e-book joint reading with voice agents*. International Journal of Human-Computer Studies, 198, 103461. https://doi.org/10.1016/j.ijhcs.2025.103461
9. Beneteau, E., et al. (2020). *Parenting with Alexa: Exploring the Introduction of Smart Speakers on Family Dynamics*. CHI 2020. https://doi.org/10.1145/3313831.3376344
10. Rudd, S., et al. (2024). *A non-randomized feasibility study of a voice assistant for parents to support their children’s mental health*. Frontiers in Psychology, 15, 1390556. https://doi.org/10.3389/fpsyg.2024.1390556
11. Romanowicz, M., et al. (2025). *Feasibility of Digital Augmentation of Parent-Child Interaction Therapy: A Randomized Clinical Trial*. JAMA Network Open, 8(12), e2548869. https://doi.org/10.1001/jamanetworkopen.2025.48869
12. Wang, P., et al. (2025). *Evaluating the Role of Interactive Encouragement Prompts for Parents in Parent–Child Stress Management*. Applied Sciences, 15(1), 256. https://doi.org/10.3390/app15010256
13. Song, S., et al. (2016). *TalkLIME: Mobile system intervention to improve parent-child interaction for children with language delay*. UbiComp 2016, 304–315. https://doi.org/10.1145/2971648.2971650
14. Kwon, T., et al. (2022). *Captivate! Contextual Language Guidance for Parent–Child Interaction*. CHI 2022, Article 219. https://doi.org/10.1145/3491102.3501865
15. Shanley, J. R., & Niec, L. N. (2010). *Coaching Parents to Change: The Impact of In Vivo Feedback on Parents' Acquisition of Skills*. Journal of Clinical Child & Adolescent Psychology, 39(2), 282–287. https://doi.org/10.1080/15374410903532627
