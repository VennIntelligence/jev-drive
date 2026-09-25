# todos/

实验规划与执行记录。简单计划使用 `YYYY-MM-DD-<slug>.md`；需要协作、日志与图表的实验使用同名文件夹。
新建时复制 `TEMPLATE.md`。

用中文写。临时、随手的东西放 `tmp/`（已 gitignore），不要放这里。

- [2026-09-22 B2D controller](2026-09-22-b2d-controller.md)：已完成轨迹控制器实施、反馈迭代与Dev10验收；无合格替代默认，全部结果见目录。

- [2026-09-23 真实TCP与转弯控制](2026-09-23-tcp-controller/README.md)：真实模型纵向对照、静止速度反馈修复，以及按转弯窗口单独优化横向。

- [2026-09-23-lateral-followup/](2026-09-23-lateral-followup/README.md)：逐弯控制诊断、Hermite负面闭环、固定后轴传播补偿及全部中间图表。

- [2026-09-23 横向 v2](2026-09-23-controller-next/lateral-v2-report.md)：后轴速度系 pursuit + Ackermann 反解的预注册闭环（[协议](2026-09-23-controller-next/lateral-v2-protocol.md)）；CTE 主目标与 held-out 全部改善，动作保护量失败，候选不通过。

- [2026-09-23 W2：TFv6 + 我们的控制器](2026-09-23-tfv6-controller/report.md)：TFv6 waypoint 交给 production 控制器与作者 waypoint PID 的四臂预注册闭环（[协议](2026-09-23-tfv6-controller/protocol.md)）；route+speed 表示比 waypoint 高约 14 DS；W2 的 C/D 被静止时的坐标变换缺陷污染；修正后重跑（[W2b](2026-09-23-tfv6-controller/report-w2b.md)），C−B 合并 +1.0 [−12, +15]，仍未检出。
- [2026-09-23 P4 CARLA 特征差距](2026-09-23-p4-carla-feature-gap.md)：Waymo 训的 head 在 CARLA 帧上能不能用（domain AUC、词表覆盖、head 迁移），预登记判据决定 P5 的前提。
- [2026-09-24 P5 CARLA 配对考卷 v0](2026-09-24-p5-carla-pairs-v0.md)：Bench2Drive 10 个 family 造 165 对 ego 逐帧相同的反事实配对 + 55 个天气 null，expert 两侧重跑出标签，考 TFv6 与 CARLA 内训的 Qwen 薄 head（结论见 decisions 第 32 条）。
- [2026-09-24 openpilot smoke run](2026-09-24-openpilot-smoke/README.md)：small / Cinque v3 / Lebowski 三个 openpilot 驾驶模型在 box 上用 onnxruntime + TensorRT 跑通（batch 1 p50 1.0 / 2.3 / 3.3 ms），数值与 CPU 参考等价，comma1M 4 段真实视频上 plan 明显好于 constant velocity；附 zero-shot 上 WOD-E2E / NAVSIM 的缺口。
- [2026-09-24 Alpamayo 1.5 smoke run](2026-09-24-alpamayo-smoke/README.md)：10B reasoning VLA 在 box 上跑通（权重走 ModelScope、clip 按 zip 成员稀疏拉取），batch 1 默认 n=1 p50 983 ms / n=6 2722 ms，SDPA+compile+flow 5 步降到 697 / 2184 ms 且输出基本不变；31 个 clip 上 minADE_6 0.74 m（card 0.916 m）；Cosmos-Reason2-8B gate 未通过需用户 accept；附 zero-shot WOD-E2E / NAVSIM 的缺口。
- [2026-09-24 zero-shot 考试：Bench2Drive](2026-09-24-zeroshot-exam/bench2drive.md)：Alpamayo 1.5 与 openpilot Lebowski 按各自原生相机在 CARLA 里闭环，预注册 rig / nav / 控制器 / 规划频率；5 条 smoke 上 Alpamayo DS 60.8、SR 2/5，openpilot DS 2.7、SR 0/5；全量 220 条估计 4.3–6.6 h（Alpamayo server-bound），等批准。
- [2026-09-24 openpilot 迁移计划](2026-09-24-zeroshot-exam/openpilot-migration.md)：B2D smoke 里 openpilot 失败的根因是适配 bug（plan 原点在相机 → 静止时 7.1 m/s 速度指令；相机 sensor_tick 抖动），已修、复跑等控制器定版；comma1M 8 段 + WOD 479 帧上的 44 种 rig / 安装 / 图像 / 时序变体：单相机、鱼眼、多相机拼接几乎无损，yaw 标定、帧抖动和 NAVSIM 的 2 Hz 时间轴伤得最重，真实数据上安装高度不重要；附各目标的接法与全量估计。
- [2026-09-25 Alpamayo B2D 全量停车诊断](2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md)：全量 117 条里 82% 的仿真时间停着，91% 的长停车始于碰撞；根因是适配 bug（Zoo PID 把 Alpamayo 静止时的倒车 plan 按距离绝对值读成前进速度，68% 踩油门）加 2 Hz 控制保持的起步过冲；全量已暂停在 119/220，建议 forward-only 修复后 re-smoke 并从头跑。
- [2026-09-24 zero-shot 考试：WOD-E2E](2026-09-24-zeroshot-exam/wod-e2e.md)：Alpamayo 1.5（WOD 7 路相机重投影成它的 4 路 f-theta）与 openpilot small/Cinque/Lebowski 在 val 479 个 rater 帧上零样本打分；RFS Cinque 8.01、Lebowski 7.89、Alpamayo 7.86（medoid 8.03），全部高于 cv 7.10 和我们 train 训的 `cls ego` 7.31，logged future 8.13；nav 文本无效果；失败集中在停车起步。
- [2026-09-24 zero-shot 考试：NAVSIM](2026-09-24-zeroshot-exam/navsim.md)：Alpamayo 1.5（nuPlan 8 路重投影成 4 路 f-theta，4 帧放 t0）与 openpilot 三个模型（F0 渲染、2 Hz sample-and-hold）在 navtest 全量 12146 上零样本打官方分：EPDMS Alpamayo 43.2、Lebowski 45.5、Cinque 46.2、small 42.5（CV 25.9、human 94.5、TransFuser 76.7），PDMS 44.3 / 50.9 / 52.1 / 47.4；nav 文本与 turn desire 都拖分；navhard 全部 ≈ CV（10–11.5）；附 numpy 1.23.4 OpenBLAS 在本机算错的发现、适配验证与折中清单。
- [2026-09-24 榜单 hack 审计](2026-09-24-hack-audit/README.md)：7 个榜单各抽 3–5 个公开代码的方法，只读代码、对照论文，找出分数里不属于驾驶能力的部分，分类由审计自己长出来；Codex 在 Tokyo box 上执行。
- [2026-09-24 zero-shot 考试：nuScenes 开环 + PhysicalAI-AV 反向检查](2026-09-24-zeroshot-exam/nuscenes-physicalai.md)：UniAD/VAD 口径的 L2 / collision。两类模型的 L2 都比匀速直行差（quarter n = 1159：Alpamayo 0.97 m、openpilot 0.86–1.09 m，CV 0.72 m；我们的 CV 复现 BEV-Planner 的 GoStraight），原因是纵向开得比 nuScenes 司机远；转弯样本和 collision 上比 CV 好（Alpamayo 0.40%、openpilot small 0.17%，CV 1.04%）；nav 文本无效。PhysicalAI-AV 31 个 clip 上 openpilot ADE@6.4 s 2.35–2.86 m，Alpamayo 单条采样 1.77 m，配对差的 CI 都在 0 以上。
- [2026-09-24 zoo 进 CARLA](2026-09-24-zoo-in-carla.md)：已 dropped，并入进行中的 zeroshot-exam；只有「无 scenario 路段」这一项移到 R 层测量。
- [2026-09-24 R 层测量](2026-09-24-r-layer-routine.md)：去掉 scenario 的路线上测 routine 驾驶能力，看排序是否等于榜单排序，并把 TFv6 的分数拆成接口、手写规则、网络三部分。
- [2026-09-24 P5 v1：E 层测量](2026-09-24-p5-v1-e-layer.md)：在 v0 上加绕行类 family 和横向标签、剔除规则兜住的题、接入 PDM-Lite 与 zoo 考生，测突发事件时反应的方向和方式。
- [2026-09-24 榜单文本分析（第二轮）](2026-09-24-leaderboard-text-analysis/README.md)：沿用 hack 审计的 36 个条目，只读文本收集正面归因账本、跨榜一致性、GitHub issue 里的复现差距、榜单指标攻击面和 C 档方法的论文披露；深度判读留给综合阶段。
- [2026-09-24 榜单综合判读（第 6 项）](2026-09-24-leaderboard-synthesis.md)：把两轮文本材料交给专家做综合判读，回答高分从哪里来、分数与驾驶能力的关系、对 R/E 两层和我们自己考卷的含义。
- [2026-09-25 TFv6 规则 × 接口，B2D 按 hazard family 拆分](2026-09-25-tfv6-rules-interface/README.md)：第 35 条的实验 1（关掉作者规则后拿分通道会不会反应，闭环配对）和实验 6（22 个公开条目的逐路线结果按 family 拆、用重复评测量噪声）；公开部分已出：相邻名次总分差全在噪声内（单次 SD 0.8 DS），突发 hazard 上只有 BLUE 分得出来（decisions 第 38 条）。
- [2026-09-25 reactivity 计划](2026-09-25-reactivity-program.md)：D0（openpilot vision 层行人 probe）、M-C（双流 reaction head）、I1（P5 v1：PDM-Lite + 101 路线，[子文档](2026-09-25-reactivity-program/i1-p5v1.md)）、I3（HUGSIM 3DGS 开环配对，[子文档](2026-09-25-reactivity-program/i3-hugsim-pairs.md)）、I4（worldmodel-4B，[子文档](2026-09-25-reactivity-program/i4-worldmodel.md)）。已出：行人信息在 openpilot 的 vision 层就没有（D0 AUC 0.52）；配对差分在 cut-in 上 +13 pp 而 hard-example 重加权 0，行人仍 0（decisions 第 42 条）。
- [2026-09-25 融合前诊断](2026-09-25-fusion-diagnostics.md)：选融合架构（openpilot `temporal`、Qwen 视频 token、SAM 3.1 感知 schema、typed decision head）之前的 9 项预登记诊断：互补矩阵、WOD 损失解剖、SAM 召回与 BEV 误差、规则门 floor、模式词表、反应时间窗、空间 token 读出；一张卡 + 20 核约 4–8 GPU·h、1.5–2 天；SAM 3.1 权重的许可待用户确认。
- [2026-09-25 闭环基础设施验收](2026-09-25-closed-loop-infra-acceptance.md)：CARLA harness 瓶颈与布局（每卡 6 个 server，box 上限是线程数 pids.max）、B2D 控制器用 PDM-Lite 专家轨迹做验收（没有控制器通过，固定控制器 2 Hz 最接近，Zoo PID 不可用）、HUGSIM 控制器验收（只有 fixed2 通过）、不明 SIGKILL 取证（不是 kernel OOM，最可能是平台内存执法）。
- [2026-09-26 激发计划](2026-09-26-elicitation-program.md)：不拼接、不重训，冻结组件 + 配对差分把能力激发出来，并从 CARLA 走到真实开环数据：E1 M-C head 零样本套 WOD（迁移检查）、E2 真实帧反事实编辑对（SAM 离线造对）、E3 从 log 挖 ego 匹配的孪生帧对（新）、E4 PDM-Lite 集的计分窗口、E5 M-C 蒸馏到 20 Hz student（等 fast-perception 延迟表）、E6 可选（NAVSIM 上离 TransFuser 的 6 分是不是配方）。2026-09-26 按开环对比最终结果修订：E2 / E3 主数据集改 navtrain（GT 轨迹 + PDM scorer），E1 加 NAVSIM 列。
- [2026-09-26 夜间队列](2026-09-26-overnight-queue.md)：5–6 h 空卡只排预登记过的活：E2 批量、全部 head 补 3 seed、E3 挖掘后定向抽特征、D-depth 接地点、I3 配对上跑现有考生、E4c latency 曲线、E5；早上按汇总回填各 todo。
- [2026-09-26 快通道感知](2026-09-26-fast-perception.md)：SAM 3.1 的同模型提速、EfficientSAM3、Grounding DINO、YOLOE-26、YOLO26-seg 在 Q4 同一批帧上比延迟 × 召回；YOLO26x-seg 640 三路 20–30 ms、行人召回不劣于 SAM 3.1（decisions 第 45 条），召回缺口在 BEV 放置而非检测器。
