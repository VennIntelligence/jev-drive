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

## N6. backbone 行补到 P5 v1 BA（GPU 2–4 h）

`ablation-matrix.md` 的空格：V-JEPA 2（权重在 box）、DINOv2、SigLIP2、openpilot small 在 P5 v1 BA 集上抽特征（29 757 张，三路），拟合 `ridge_late` 和 pair-Δ，
读行人 / cut-in 翻转与 null。回答「视频 / 图像自监督特征里有没有 E 层信号」，与 WA-JEPA 的 +6 对照（`nohack-mechanisms.md` 说那是微调 encoder，与我们冻结不同口径，报时注明）。
**判据**：pair-Δ 行人翻转 CI 下界 > 该 backbone 自身 null p95 + 10 pp → 「有 E 层信号」。DINOv3 无权重，写「未测」。

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
