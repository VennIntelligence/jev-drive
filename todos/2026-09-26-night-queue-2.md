# 夜间队列 2（2026-09-26 晚）：第三层量具、榜单 head × 反应、快通道去 lift、backbone 补行

状态: registered（写于任何数字之前；执行员按节领活，每节结果写回本节末尾，结论回填 decisions）

上一轮的四条真实数据任务（G0 / G1 / G3 / R40，`2026-09-26-real-data-transfer.md`）由中央调度员派出，正在卡上跑；本队列只吃剩余容量，
每张卡 96 GB，现占 23–32 GB。**开工前 `nvidia-smi` 选最空的卡，run 目录里记下卡号；不碰别人的进程。**

依据的三份调研（都在 `tmp/`，2026-09-26）：`behavior-layer-survey.md`（第三层量具与 P6 草案）、`nohack-mechanisms.md`（no-hack 高分方法的机制）、
`ablation-matrix.md`（矩阵空格与口径不一致）。要点：

- 第三层（判断后的行为：bypass、negotiation、recovery）我们目前没有量具；公开开环榜都不按行为模式计分。P5 v1 的路线里横向避让类几乎没出题。
- PDM-Lite 会绕，但靠 `active_scenarios` 的特权登记触发（50 m 内平移路线，对向 gap 用真值速度）；BehaviorAgent 只停不绕，static prop 检测不到。
- Hydra 打分头（NAVSIM 84.2）从没上过 P5 / I3；pair-Δ head 上过 WOD / NAVSIM 且有害（第 44 条）。榜单最优 head 保不保反应，还没量。
- ego-only 在 P5 上的 6–8% 是 τ = 0 的标签伪影，不是反应；I3 上是 0。
- V-JEPA 2 权重在 box 上但没上过 P5；DINOv3 没权重（申请被拒）；SigLIP2、DINOv2、openpilot small 在 P5 上没数。
- 第 45 条：行人召回缺口在 20–40 m 的 flat-ground 放置（图像上检出 46–61%，BEV 只剩 12–22%）；E5 的检测 embedding 走的正是这条 lift。

## 通用规则

1. 判据写在数字之前；看了数再想改，只能新登记一行「事后」，不改原判格。
2. 不跑 CARLA 闭环。P6 是开环配对：录像 + 事后评测。
3. 估时后再跑；超过估计 2 倍先停下写日志。> 1 min 的活进 tmux `jev`，写 `log.txt` / `events.jsonl`。
4. 主表 3 seed（seed 定义沿用 `2026-09-26-overnight-queue.md` [SEEDS]）；单 seed 的数只能写「同一水平」，不能写「更好」。
5. 口径：RFS 报 cluster mean（frame mean 只在与旧表对照时并列）；P5 翻转率沿用 `p5_exam` 的 τ = 考生自身 null p95；I3 judge 不改。
6. 结果小文件到 `research/results/night2/<节>/`，图到 `research/figs/`，commit + push；结论回填 decisions（新条或就地修正，标**待定**）。

## N1. P6 v0：行为模式考试的数据与 expert 统计（GPU，约 4 h × 2 卡）

**目标**：造出第三层的第一版考卷，并先量 expert 自己在上面怎么做；本节不读任何考生。

**配对**（按 `behavior-layer-survey.md` §3.1，只用 220 集路线，expert = PDM-Lite，3 个 TM seed）：

| 世界 | 障碍 | 对向车流 | 预期 expert 动作 |
|:--|:--|:--|:--|
| x₀₀ | 藏 actor **并删掉 `active_scenarios` 里的登记** | 删 | keep |
| x₁₀ | 在 | 删 | bypass |
| x₁₁（只 TwoWays 类） | 在 | 在（对向车道确定车流，spawn 表写死，TM seed 固定） | 等 gap 再绕 |
| x₀₁（只 TwoWays 类） | 删 | 在 | keep（对向车本身不应引起反应） |

- scenario：1W = Accident、ConstructionObstacle、ParkedObstacle、HazardAtSideLane；2W = 四类的 TwoWays 版 + VehicleOpensDoorTwoWays；另加 InvadingTurn、YieldToEmergencyVehicle（救护车 {有, 无}）。
  unprotected turn 六类作第二期，本节不做。
- null：天气 null（同 P5）；**放置 null**（障碍移到路肩不占车道，expert 应 keep）；**镜像题**（相邻车道不可用：实线 + 护栏或对向车流不断，expert 应停）。
- 记录：P5 v1 的 recorder（`scripts/p5_pair_agent.py`，`driver: pdm_lite`，5 Hz Waymo 标定相机，`cams/<cam>/<frame>.jpg`），加 expert 的未来轨迹、GT actor、
  `active_scenarios` 状态、对向车 GT 速度；TFv6 考生要的传感器（LiDAR）若 P5 v1 已录则照录。
- 规模：约 580 个世界（1W 120、2W 300、InvadingTurn + Emergency 60、null 约 100）；P5 v1 单 run 中位 276 s、11 实例并行 → 约 4 h。两张卡各 6 个 CARLA server。

**开工前的 smoke（必过，10 个世界）**：x₀₀ 里 PDM-Lite 的横向偏移 |Δ_lat(3 s)| 相对 x₁₀ 同帧 < 0.3 m 的比例 ≥ 95%；不过就是删登记失败，先修再批量。
另一条 smoke：对向车流在 x₀₁ / x₁₁ 里确实出现在 expert 变道窗口内（每个 2W 世界至少 1 辆在 50 m 内）。

**expert 统计（本节的交付，写死）**：

| 量 | 读法 |
|:--|:--|
| 每类 scenario 的 expert 模式分布（keep / stop / bypass_L / bypass_R / wait-then-bypass），按 §2.1 的判定规则 | 有没有出题：bypass 类里 bypass 比例 ≥ 70% 才算这类可用 |
| 横向分叉时刻 t_div 相对障碍首次可见 t_vis（同 P5 的 t_vis 定义） | t_div ≥ t_vis 的对才计入主读数；PDM-Lite 靠登记，可能提前，提前的对单列 |
| x₁₁ − x₁₀ 的纵向 Δv 与横向起动延迟 | negotiation 有没有被造出来：x₁₁ 里 wait 比例 ≥ 50% |
| 放置 null 与镜像题上 expert 的模式 | 放置 null keep ≥ 90%；镜像题 stop ≥ 80%；不过就修生成器 |

**另两条 CPU 准备**（同一执行员）：(i) box 上数 `cls_late` K = 1024 词表里 bypass 形状的 anchor 数与 oracle 覆盖率（判定同 §2.1，先写阈值：3 s 处横向 ≥ 1.0 m
且 5 s 内回到 ±0.5 m）；(ii) 220 集 xml 里上述 scenario 各多少条路线。两个数进本节结果。

**判读考生**（登记在此，但**本节不执行**，等 N1 数据齐、expert 统计过门后另开一节）：回归类考生读 Δ_lat(2 s / 3 s) 的横向翻转率 + 4 × 4 模式混淆；
`cls_late` 类读词表模式质量 Δm_bypass(x₁₀ − x₀₀)、Δm_stop(x₁₁ − x₁₀)；null 地板 = 天气 null p95、放置 null 的 bypass 率、x₀₁ − x₀₀。
考生：openpilot `ridge_late` / `cls_late`（Cinque、Lebowski）、M-C、E5 student、TFv6 waypoint（若传感器已录）、Qwen `L18_last`、ego-only。

- 2026-09-26 09:52 CST [A] 开工，分步估时（写于任何 N1 / N2 数字之前）：
  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | 1 | 生成器（x₀₀ / x₁₀ / x₁₁ / x₀₁、放置 null、镜像题、天气 null 的 variant XML）+ hook（删登记、对向车流开关、障碍平移）+ recorder 补记（prop、登记状态）+ 分析模块 | 2 h（到约 12:00） | Mac + box CPU |
  | 2 | smoke 10 个世界 = profiling pass（逐 tick 分项、每 run 墙钟），两条 smoke 判据 | 0.5 h | 1 卡 |
  | 3 | 批量约 605 个世界（todo 的 580 + 镜像题 25），3 卡 × ≤ 6 server（受 pids.max 限制） | 约 3 h（P5 v1 PDM-Lite 实测每 run 均值 244 s；绕行类预计 5–6 min / run） | GPU 0–2，约 9 卡·h |
  | 4 | 建 index + expert 统计表 + 两条 CPU 准备（词表 bypass anchor、220 集路线数） | 0.5 h | CPU |
  | N2-a | probe a / b 在 P5 v1 帧上 + desire 执行器检查（与步 1–3 并行，子执行员） | 1.5–2 h | CPU + < 1 GPU·h（GPU 2 空档） |
  | N2-b | N1 数据到后重跑 probe a / b、跑 probe c、x₁₀ 帧补 desire | 1 h | CPU + 少量 GPU |
  合计墙钟约 7–8 h（预计 17:30–18:00 CST 收尾）。任何一步超估计 2 倍停下写日志。盘：box 剩 167 GB，P5 v1 每 run 约 70 MB → 本节约 45 GB。

## N2. openpilot 里有没有绕行需要的信息 + desire 执行器检查（CPU + 少量 GPU，< 1 h）

激发的前提是冻结特征里有信息。行人那一轮 openpilot vision 层 AUC 0.51，只能外接。绕行需要三样，逐样 probe：

| probe（线性，CARLA GT 标签，5 fold，AUC） | 标签 | 数据 |
|:--|:--|:--|
| a. 本车道前方 ≤ 30 m 有静止障碍 | GT actor 在 ego 车道、v < 0.5 m/s | P5 v1 现有帧先跑（障碍类少，报 n）；N1 数据到后重跑 |
| b. 相邻车道 ±20 m 内有车 | GT | 同上 |
| c. 对向车道 50 m 内有来车 | GT | N1 的 2W 世界 |

特征：openpilot `temporal`（Cinque、Lebowski）、`driving_vision` 输出、YOLO26x-seg image-plane token 集（对照）。
**判据**：AUC ≥ 0.70 记「有信息，可激发」；≤ 0.60 记「没有，要外接」；之间如实报。

**desire 执行器检查**（零训练）：在 P5 v1 直行帧（v ≥ 5 m/s，按速度分档 5–10 / 10–15 / > 15 m/s）上重跑 openpilot，desire = laneChangeLeft / Right 的上升沿脉冲
（`OPModel` 已有 rising-edge），读原生 plan 3 s 处相对不带 desire 的横向偏移。**判据**：每档 |Δ_lat(3 s)| 中位 ≥ 1.5 m 且方向正确 ≥ 90% → 执行器可用，
第三层可以走「模式头 → desire → openpilot plan」；中位 < 0.8 m → 记「openpilot 不按 desire 变道，执行器另找」。N1 的 x₁₀ 帧到后再补一遍（有障碍时）。
09-25 的 2b 是拿 desire 当导航（有害），这里是反应式触发，两者分开写。

## N3. 榜单 head × 反应：能力包的 2 × 2（CPU 为主，< 2 GPU·h）

同一个冻结 `temporal` 上，回答「榜单最优的 head 保不保反应、能力 head 掉不掉榜单分、叠起来两边能不能都留住」。

| head | NAVSIM PDMS / EPDMS（navtest） | WOD RFS | P5 v1 BA 行人 / cut-in 翻转 | I3 |
|:--|:--|:--|:--|:--|
| `ridge_late` | 有 | 有 | 有 | 有 |
| `cls_late` | 有 | 有 | **补** | **补** |
| Hydra 打分头 | 有（单 seed）→ **补 3 seed + Lebowski 行** | 不适用（WOD 无 PDM 子分，写「不适用」） | **补** | **补** |
| pair-Δ（M-C）不加 gate | 有（有害） | 有（有害） | 有 | 有 |
| Hydra + gated Δ（gate 取 G1 出的主 arm；G1 没出就用 P2(e) gate） | **补** | — | **补** | **补** |

- Hydra 头零样本上 P5 / I3 之前先做**兼容检查**：Hydra 训练用的 NAVSIM `temporal` 是 2 Hz 输入下抽的，P5 / I3 是 5 Hz；对照 = 同一 head 在 P5 null 帧上的
  输出分布与 navtest 上的分布（候选 argmax 的 top-10 重叠、走廊内 DAC 子分均值差）。差得离谱（top-10 重叠 < 30%）就写「不可比」，这格不读。
- **判据**（写死）：Hydra 头 P5 BA 行人翻转 CI 上界 < `ridge_late` 的点估计 → 「榜单 head 压掉反应」；CI 重叠 → 「同一水平」；
  Hydra + gated Δ 的 NAVSIM PDMS 掉 ≤ 1.0 且 P5 行人翻转 ≥ M-C 的 80% → 「能力包成立（CARLA 内）」。
- 顺带补两格：nuScenes 冻结 head 的 collision 率；E5 student 的 E4c latency 曲线。特征、标签、预测都在 box 上。

## N4. 快通道去 lift：E5-b image-plane token（CPU + < 0.5 GPU·h）

第 45 条把召回缺口归到 flat-ground 放置；E5 的 embedding 又用同一条 lift 加手写走廊筛检测。改成不做几何、让配对差分自己学：

| arm | 检测 token（每路相机前 k = 8） |
|:--|:--|
| A（现 E5） | lift → 走廊筛 → (类别, x, y, 尺寸, score) |
| B | image-plane：(相机 id, u, v, w, h, 类别, score, 检测 embedding)，不 lift、不筛 |
| C | B ⊕ A |

P5 v1 BA 集，配对差分，3 seed，Cinque 与 Lebowski。**判据**：B 的行人翻转 CI 与 A 重叠或更高，且 DOC 非反应误翻不比 A 多 3 pp 以上 → 快通道改为 image-plane，
lift 只留在测量里；B 明显更低（CI 不重叠）→ 记「几何先验在这里有用」。cut-in 对 prior 的 Δ 并列报。端到端延迟同 E5 口径重测一次。

## N5. 放置修复只用于测量（GPU 1–2 h）

第 45 条要求登记的一项。metric depth（Depth Anything 3 metric 或 UniDepth v2，给相机内参）替代 flat-ground lift，只改**召回测量**，不进模型。
数据同第 45 条：P5 hazard 行人帧 + nuScenes 子集。**判据**：20–40 m 行人 BEV 召回（2 m 容差）从 0.12–0.22 升到 ≥ 0.40 → 第 45 条「缺口在放置」得到修法；
仍 < 0.30 → 记「单目 metric depth 也不够，需要地面高度」。第 45 条就地补一行。

- 2026-09-26 09:55 CST [C] 开工（执行员 C，N5 → N6，GPU 3；N6 抽特征时若 GPU 4 空着借来分片）。分步估时（写于任何 N5 / N6 数字之前）：

  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | N5-1 | `envs/depth`（torch 2.13 cu130 同 `envs/sam3`）+ UniDepth v2 ViT-L 与 DA3METRIC-LARGE 权重（各约 1.3 GB） | 30–45 min | CPU、网络 |
  | N5-2 | 子集 S 上逐图 metric depth（P5 9 918 张 + nuScenes 6 024 张 × 2 个模型），在 YOLO26x-640 与 SAM 3.1 的已存检测接地点上取深度 | 30–45 min | GPU 3，≤ 20 核 |
  | N5-3 | `fastperc.evaluate --contact depth` 原样（2 模型 × 2 检测器 + 固定 2 m 门 side）+ 表 | 20–30 min | CPU |
  | N6-1 | 抽特征模块 + 与 `features.py` 原类的逐位等价检查 + 吞吐 smoke | 45 min | Mac + GPU 3 |
  | N6-2 | V-JEPA 2 / DINOv2 / SigLIP2：46 703 行 × 3 路（去重后约 14 万个 clip），解码与推理重叠 | 30–60 min | GPU 3（+ GPU 4），≤ 48 核 |
  | N6-3 | openpilot small `temporal`：P5 v1 BA 的 858 个 stream、7.4 万帧，6 个分片（与 N6-2 并行） | 20–40 min | GPU 3，≤ 30 核 |
  | N6-4 | `ridge_late` + pair-Δ：5 个 backbone（含 Qwen 复现行）× 3 seed | 30 min | GPU 3 |
  | N6-5 | 表、图、decisions、todo 结果 | 60 min | Mac |

  合计墙钟约 5 h（到约 15:00），GPU 约 2.5 GPU·h。特征体积：三个图像 backbone 主 + 副 tap 共约 1.7 GB（float16），small `temporal` < 0.2 GB；数据盘现剩 168 GB（不是 260 GB），够。
  任何一步超估计 2 倍就停下写日志。

- 2026-09-26 09:55 CST [C] **N5 的操作性选择**（写于任何深度数字之前）：
  - **两个模型都测，主读数 = UniDepth v2**（`lpiccinelli/unidepth-v2-vitl14`，作者仓库 `infer(rgb, K)`，给针孔内参 fx / fy / cx / cy，输出沿光轴的 z 深度）；
    副读数 = Depth Anything 3 `DA3METRIC-LARGE`（作者 API，默认处理分辨率，按作者 README 的焦距换算成米）。理由：N5 写「给相机内参」，UniDepth v2 把内参当输入条件，
    DA3 metric 只在输出上按焦距缩放。两个权重 2026-09-26 都核实可下载（HF 未 gated）。判格按主读数下；副读数不一致时两个都写。
  - P5 相机有 Waymo k1 / k2 畸变，给深度模型的 K 不含畸变；接地点的射线仍由 `fusion_q4.lift` 去畸变，深度沿这条射线放（`fastperc._depth_place` 原样，射线光轴分量为 1，所以深度即 z 深度）。
  - 检测不重跑：主 = YOLO26x-seg 640（快通道检测器，E5 用的也是它）的已存检测，副 = SAM 3.1 原样（第 43 / 45 条基线）。取深度的规则沿用 `fastperc.depth_sample`：接地点上方 3 px 起 5 × 5 窗口的中位数，深度图在原图分辨率上。
  - 「2 m 容差」按第 45 条那组 0.12–0.22 实际用的匹配门理解：`fusion_q4.gate` = max(2 m, 0.1 d)（20–40 m 处 2–4 m），不改；另报固定 2 m 门作 side（平地与深度两边都算）。
  - 判读对象：行人 recall (ii)（≤ 40 m、在图内，P5 = 背景 actor，nuScenes = visibility ≥ 3）20–40 m 档，P5 与 nuScenes 各判：≥ 0.40 → 修法成立，< 0.30 → 不够，之间如实报。
    本条判格：两个数据集都 ≥ 0.40 才写「得到修法」，都 < 0.30 才写「需要地面高度」，否则分数据集写。并列报 0–10 / 10–20 m、hazard 行人（全部 / ≤ 20 m）和深度 / 平地放置距离比的中位数，
    因为 YOLO26x-depth 那次近处被弄坏（第 45 条），修远处不能以坏近处为代价，这一条只描述、不进判格。

## N6. backbone 行补到 P5 v1 BA（GPU 2–4 h）

`ablation-matrix.md` 的空格：V-JEPA 2（权重在 box）、DINOv2、SigLIP2、openpilot small 在 P5 v1 BA 集上抽特征（29 757 张，三路），拟合 `ridge_late` 和 pair-Δ，
读行人 / cut-in 翻转与 null。回答「视频 / 图像自监督特征里有没有 E 层信号」，与 WA-JEPA 的 +6 对照（`nohack-mechanisms.md` 说那是微调 encoder，与我们冻结不同口径，报时注明）。
**判据**：pair-Δ 行人翻转 CI 下界 > 该 backbone 自身 null p95 + 10 pp → 「有 E 层信号」。DINOv3 无权重，写「未测」。

- 2026-09-26 09:55 CST [C] **N6 的操作性选择**（写于任何 N6 特征或数字之前）：
  - **行数**：P5 v1 BA 索引 46 703 行（P4 训练 9 529 + P5 训练 17 746 + 观测 19 428），每行三路相机；本节写的「29 757 张」与索引对不上，按索引全量抽
    （`ridge_late` 与 pair-Δ 的训练行都要）。按 JPEG 真实路径去重后约 14 万个（行, 相机）单元。
  - **输入与主 tap**（每个 backbone 一个主 tap 进判格，副 tap 只并列）：V-JEPA 2 ViT-L（`vjepa2-vitl-fpc64-256`）每路相机用索引里同一个 4 帧 clip（5 Hz、跨 0.6 s，
    与 WOD 阶梯的 4 帧 × 0.2 s 相同），256² 拉伸，主 `mean`（WOD 的主 tap）、副 `last_mean`；DINOv2-base 当前帧，保持宽高比缩到 350 × 322（25 × 23 patch，
    约等于 nuScenes 配方 252 × 448 的 576 个 patch；原配方是横图，直接用会把 972 × 1079 的竖图压扁），主 `patch_mean`、副 `cls`；SigLIP2 so400m 当前帧 384² 拉伸（原生），
    主 `patch_mean`、副 `pooled`；openpilot small `temporal`（`scripts/p5_openpilot.py` 原样，5 Hz 每帧 hold 4 步，同 Cinque）。前三个的三路按 front / front_left / front_right 拼接。
    预处理用 `jevdrive/features.py` 里的原类（`VJepaFeatures`、`DinoFeatures`、`SiglipFeatures`），只换外面的循环。
  - **head**：`ridge_late <bb>` = `p5_exam.heads` 原样；**pair-Δ 主读数 = 单流**：`reactivity_mc.fit_fold` 原样、把 Qwen 流换成该 backbone，prior = openpilot Cinque `ridge_late`，
    Δ 只看该 backbone（= 第 42 条 M-C「只 Qwen」那一格，Qwen 行人 42.1%）；双流（backbone ⊕ Cinque `temporal`）与 Lebowski prior 并列作 side。λ 网格、μ、fold 规则一字不改。
    Qwen `L18_last` 作为第 5 个「backbone」在同一驱动里重跑一遍，seed 0 必须逐位复现已存的 M-C「pair qwen [cinque]」，不复现就先查驱动。
  - **seed**：route fold 排列 0 / 1 / 2（`2026-09-26-overnight-queue.md` [SEEDS]）。
  - **判据的操作化**：τ = 考生自身 null |Δ| 的 p95（`p5_exam` 原样），所以「null p95」对应的翻转率就是该考生的样本外 null false-flip（构造上约 5%）；
    判据读作：行人翻转的路线 bootstrap CI 下界 > 该考生（同一 seed）样本外 null false-flip + 10 pp。3 个 seed 都过 → 「有 E 层信号」，都不过 → 「没有」，混合 → 「不稳定」如实写。
    cut-in 报翻转率与对 prior 的配对 Δ（M-C `criteria` 口径），不进判格。

## N7. state-space policy（另一执行员，已派，`todos/2026-09-26-state-space-policies.md`）

找公开权重、装机、预登记后量 bypass / negotiation。与本文件其余节不共享 GPU 以外的东西。

## 分派与顺序

| 执行员 | 节 | 卡 |
|:--|:--|:--|
| A | N1 → N2（N2 的 probe 先在 P5 v1 上跑，N1 数据到后重跑） | 两张最空的卡（CARLA 6 server / 卡） |
| B | N3 → N4 | CPU 为主，零散 GPU |
| C | N5 → N6 | 一张卡 |
| D | N7 | 一张卡 |

顺序图：N2(P5 v1 部分) / N3 / N4 / N5 立即；N1 smoke 过后批量；N6 在 N5 后；N2 的 N1 部分最后。

- 2026-09-26 09:55 CST [main] A / B / C 由中央调度员（real-data-transfer 那个 session）派出，D 已在跑。卡：G0 / G3 已收工、5 张卡全空；
  A 用 GPU 0–2（每卡 ≤ 6 个 CARLA server，全 box ≤ 30，与 D 的 CARLA 合计），C 用 GPU 3，B 的零散 GPU 用 GPU 4；空出来的卡谁需要谁用，开工前看 `nvidia-smi`。

## 结果

（按节追加，每条带出处路径。）
