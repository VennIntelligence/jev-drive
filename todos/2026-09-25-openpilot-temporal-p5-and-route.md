# openpilot temporal 特征：P5 配对考卷上有没有长尾信息；route 进 head / 进 backbone 买不买得回静止起步与路口

状态: done（预登记写于任何抽取之前；1 与 2a 结果 2026-09-25 14:20，2b 结果 15:50）
上游: 第 40 条及后续 (iii)（[driving-backbones](2026-09-24-driving-backbones/README.md)）、第 32 条（[P5 配对考卷 v0](2026-09-24-p5-carla-pairs-v0.md)）、第 34 条（[WOD zero-shot](2026-09-24-zeroshot-exam/wod-e2e.md)）
背景: openpilot 的 512 维 `temporal` 是项目里最强的冻结表征（第 40 条），但它的视觉端有约 700 bit 的 information bottleneck（arXiv 2504.19077），
长尾信息在不在里面要测；它输的地方是静止帧和 Intersections（第 34 条），comma 自己加 route 输入的尝试（Navigate on openpilot，0.9.4 → 0.9.7 删除）在闭环里失败过。

## 三个实验，各自否掉一条路

| # | 问题 | 回答方式 | 判据 |
|---|---|---|---|
| 1 | `temporal` 里有没有第 3 层（突发 hazard）的信息 | P5 的 165 对 x⁺ / x⁻ + 55 null 上抽 Cinque、Lebowski 的 `temporal`，按 P5 协议报 hazard probe AUC、配对差分训的 head 的定向翻转率、null false-flip | probe AUC 与 Qwen 的 0.635 [0.58, 0.69] 配对比；翻转率与 `ridge_late` 的 0% 和 TFv6 waypoint 的 39% 并排 |
| 2a | head 层面加 route（intent 已在 ego 特征里）买不买得回静止帧和路口 | 零 GPU：heads_train run 的逐帧 preds 按子集重切 | 120 个起始速度 < 0.5 m/s 的 rater 帧、Intersections / Multi-Lane cluster 上，`cls_late` 对 cv、`cls ego`、原生 plan 的配对 RFS Δ |
| 2b | route 进 backbone（WOD intent → desire 脉冲）有没有用 | 带 desire 重抽 train + val，重拟合 `ridge_late` / `cls_late`；同一次抽取给出带 desire 的原生 plan | 原生 plan RFS 对 8.005 / 7.886 的配对 Δ；head 在 2a 的两个子集上的 Δ；全集主表不能变差 |

## 硬约束

- 协议、head、judge、统计全部沿用第 40 条和 P5 v0，一字不改；新增的只是 examinee / arm。
- 不重跑 CARLA：P5 的三路 Waymo 标定相机 JPEG 已在盘上（`runs/p5_pairs/gen`，5 Hz，约 30 GB）。5 Hz 够用：rate study 里 Cinque `ctx5-hold` 与 native 逐位相同，Lebowski context 步本来就是 0.2 s。
- 三路 → road / wide model frame 用 WOD 抽取的同一渲染器（`jevdrive/openpilot/frames.py`），只换 CARLA 内参（fov 由 render_w / f 算）；先做 16 行等价性检查再批量。
- desire 脉冲打在 intent 上升沿（`jevdrive/openpilot/model.py` 已实现 rising-edge），方向映射按 HUGSIM 清单第 7 项验证过的那套。
- 不占 GPU 2（Alpamayo 考试）；GPU 0 / 1 各一张，显存 < 10 GB。
  **偏离（2026-09-25 用户决定，跑之前）**：改用 GPU 2（Alpamayo B2D 暂停），与 HUGSIM baseline、decision40-* 共卡，本实验 ≤ 40 GB；CPU 是 box 的瓶颈（CARLA 占约 68/75 核），全部进程钉在 12 核（184–195）、nice 10。slot 名 `p5route-*`，入口 `scripts/p5route.sh`。
- 结论进 decisions.md 第 40 条下面，不另开条目；小表进 `research/results/driving-backbones/`。

## 预算

| 步骤 | 估计 | 依据 |
|---|---|---|
| 1：解码 + 重投影 127k 张 JPEG | 进程池 < 10 min | 考试 78 ms·核/张 |
| 1：抽 42k 帧 × 2 模型 | 15–20 min 单卡 | WOD 11 ms/帧·进程 |
| 1：probe + head + judge | CPU 分钟级 | p5_exam 加 examinee |
| 2a | CPU 分钟级 | `heads_train/20260925-110819/*_preds_dir0.npz` |
| 2b：重抽 52 万帧 × 2 模型 | 约 70 min 两卡 | 上一轮实测 |
| 2b：cls_late L-BFGS | 每模型 15–30 min GPU | 上一轮实测 |
| 工程 | 1 和 2b 各半天到一天 | |

顺序：2a 先出（今天）；1 与 2b 各占一张卡并行。任何一步超预算 2 倍先停下来报。

## 预期（跑之前写下）

- 1：AUC 高于 Qwen 但翻转率仍接近 0 → 信息在、readout 不在，配对监督的 head 有戏；AUC 低于 Qwen → bottleneck 滤掉了长尾，openpilot 只做第 2 层 backbone。
- 2a：静止帧上 `cls_late` 不比 cv 好（第 34 条的模式延续）。
- 2b：原生 plan 在 Intersections 上 +，静止帧不动；comma 的先例说闭环起步不是输入问题。

## 结果

代码：`jevdrive/p5_openpilot.py` + `scripts/p5_openpilot.py`（实验 1 的抽取）、`p5_exam run --op cinque,lebowski`（考试，只加 examinee）、
`jevdrive/op_route.py`（2a / 2b 的读数与 desire 映射）、`scripts/drive_backbones_openpilot.py --desire`（2b 抽取）。小表在
[research/results/driving-backbones/op-p5/](../research/results/driving-backbones/op-p5/) 与 [op-route/](../research/results/driving-backbones/op-route/)。

### 读法上的一处澄清（跑之前定的）

表里实验 1 写的「配对差分训的 head」与硬约束「head 一字不改」冲突。按硬约束执行：examinee 是 P5 v0 的 `ridge_late`
（在 CARLA expert 帧上按路线 5 折交叉拟合，**不是**配对差分监督），只把特征从 Qwen `L18_last` 换成 openpilot `temporal`。
配对监督的 head 是预期里「有戏」那一支的下一步，本轮没做。

### 等价性与成本

| 检查 | 结果 |
|:--|:--|
| 标定记录对 P4 录制端相机模型（独立实现：CARLA 左手系、居中 pinhole 渲染、Waymo 畸变的不动点反解） | road / wide 每个 model frame 像素落到 render 上的误差中位 0.40 px、最大 0.74 px（最近邻取整的量级）；road 全部取自 FRONT，wide 74% FRONT、13% / 13% 左右 |
| 16 行（实际 33 行：4 个 stream × 最后 4 个 target × 2 模型）：进程池渲染 + 整条 stream 对 主进程重渲染 + 从零重跑到该帧 | 渲染逐位相同；`temporal` 最大差 **0.0** |
| 2b 抽取器加 desire 参数后，不给 desire 重跑 2 条 WOD stream（459 帧）对第 40 条已存的 trainval 特征 | `temporal` 与原生 plan 最大差 **0.0** |
| 考试管线复现 | 已有 examinee 逐位复现 P5 v0：TFv6 waypoint 39.4% [28.5, 50.1]、`ridge_late` L18 0%、hazard probe 0.635；ego 输入逐位相同的 3883 帧 38.4% |

| 步骤 | 预算 | 实测 |
|:--|:--|:--|
| 1：536 条 stream（151 条 P4 训练路线 + 385 个 P5 世界）、48 046 帧 × 2 模型，渲染与抽取重叠 | 10 + 15–20 min | 24 min（两进程 × 5 渲染 worker，12 核） |
| 1：probe + head + judge | 分钟级 | 17 min（与共卡的 CARLA / HUGSIM 同卡） |
| 2a | 分钟级 | 1 min |

Cinque 每个 5 Hz 帧在 20 Hz 时钟上保持 4 步（rate study 的 `ctx5-hold`），Lebowski 每帧一个 context 步；每条 run 从零状态开始流到最后一个索引帧。
P4 训练路线也必须抽：head 的训练行里有 9529 行来自 P4（协议不变）。

### 实验 1：`temporal` 里有没有第 3 层的信息

合并 6 个 family、490 个 reactive 帧、25 条路线；CI 是路线 bootstrap。

| 考生 | τ_model (m/s) | 定向翻转率 [95% CI] | family 等权 | 反方向 | 样本外 null false-flip | non-reactive 帧上翻转 | ego 输入逐位相同的 458 帧 |
|:--|--:|:--|--:|--:|--:|--:|:--|
| TFv6 waypoint 2 s 速度（参照） | 2.22 | 39.4% [28.5, 50.1] | 38.3% | 0 | 7.1% | 12.1% | 38.4% [27.4, 49.0] |
| `ridge_late` Qwen `L18_last`（参照） | 0.51 | 0.0% [0.0, 0.0] | 0.0% | 0 | 5.3% | 0.0% | 0.0% |
| **`ridge_late` Cinque `temporal`** | 1.49 | **46.7% [31.7, 60.0]** | 32.7% | 0 | 5.1% | 4.7% | 47.8% [34.1, 60.5] |
| **`ridge_late` Lebowski `temporal`** | 1.38 | **41.6% [25.5, 56.3]** | 29.6% | 0 | 5.5% | 4.8% | 42.4% [28.0, 57.1] |

| family（reactive 帧） | TFv6 waypoint | Cinque `temporal` | Lebowski `temporal` | hazard probe AUC：Qwen `L18_last` / Cinque / Lebowski |
|:--|--:|--:|--:|:--|
| HighwayCutIn (102) | 46% | **89%** | **86%** | 0.84 / 0.77 / 0.76 |
| StaticCutIn (123) | 42% | 54% | 59% | 0.53 / 0.56 / 0.56 |
| ParkingCutIn (135) | 35% | 53% | 32% | 0.56 / 0.56 / 0.57 |
| DynamicObjectCrossing (45) | 38% | 0% | 0% | 0.62 / 0.50 / 0.52 |
| ParkingCrossingPedestrian (76) | 36% | 0% | 0% | 0.54 / 0.51 / 0.52 |
| Light (9) | 33% | 0% | 0% | — |
| PedestrianCrossing（不进合并，3 帧） | 0% | 0% | 0% | 0.77 / 0.53 / 0.52 |

| hazard probe（x⁺ 对 x⁻，3744 帧 / 37 路线） | AUC [CI] | 对 Qwen `L18_last` 的配对 Δ [CI] |
|:--|:--|:--|
| Qwen `L18_last`（P5 v0） | 0.635 [0.581, 0.690] | — |
| Cinque `temporal` | 0.540 [0.517, 0.579] | **−0.096 [−0.143, −0.042]** |
| Lebowski `temporal` | 0.543 [0.523, 0.575] | **−0.092 [−0.138, −0.042]** |

（训练路线外的训练帧上 probe AUC：Qwen 0.93、Cinque 0.93、Lebowski 0.91，特征本身能学会这个标签。light probe 四个 tap 都是 0.50，和 v0 一样不下结论。）

**判定：两条预期都没有原样出现。** 按预登记的第二支，probe AUC 低于 Qwen（配对 Δ −0.09，CI 不跨零）→「bottleneck 滤掉了长尾」；
但同一特征上的**未改动的** `ridge_late` 翻转率 42–47%，高于 TFv6 waypoint，而 Qwen 在同一个 head 下是 0。两件事按 family 拆开就不矛盾：

- **车辆 cut-in（前车插入）上信息在、而且普通 ridge 就读得出来**：HighwayCutIn 86–89%，StaticCutIn / ParkingCutIn 32–59%，从不往反方向翻，
  null false-flip 5%（与 Qwen 相同，门槛 τ 是各自在 null 上定的）。符号与 expert 一致 85–88%（Qwen 66%），Δ 中位数 −1.1 到 −1.4 m/s（Qwen −0.004）。
  这是 openpilot 自己的训练分布（前车、lead），它的 `temporal` 把「前面有车插进来 → 要减速」编码成了纵向计划的方向。
- **行人和红绿灯上什么也没有**：两个行人 family 与 Light 翻转全是 0，行人 probe 0.50–0.53（Qwen 0.62–0.88）。
  合并 AUC 的 3744 帧里约一半来自四个行人类 family（观测帧 725 + 896 + 199 + 179），它们全在 0.50–0.53，合并值因此被拉到 0.54；cut-in family 上 openpilot 的 probe 与 Qwen 接近（0.56–0.77 对 0.53–0.84）。
- 所以更准确的读法是：**openpilot 的 bottleneck 保留了它训练分布里的第 3 层（车辆 cut-in），滤掉了分布外的（行人、横穿物、灯）**；
  第 3 层里 cut-in 那一半不需要配对监督，冻结特征 + ridge 就过 TFv6；行人那一半在特征里就没有，配对监督的 head 在这个特征上救不回来。
- 限定：合并 CI 很宽（25 条路线），family 等权只有 30–33%；只有一个 expert（BehaviorAgent），行人 family 的 reactive 帧只有 45 + 76 个。

### 实验 2a：head 层面（intent 已在 ego 特征里）买不买得回静止帧和路口

第 40 条 (iii) 的 `heads_train/20260925-110819` 逐帧预测按子集重切，零拟合；frame-mean RFS，sequence bootstrap。

| 子集 | n | `cls_late` − cv：Cinque / Lebowski | `cls_late` − `cls ego` | `cls_late` − 原生 plan | 参照：原生 − cv | 参照：`ridge_late` − cv |
|:--|--:|:--|:--|:--|:--|:--|
| 全部 rater 帧 | 479 | +0.61 [+0.37, +0.85] / +0.66 [+0.44, +0.89] | +0.38 / +0.43 | −0.29 [−0.47, −0.10] / −0.10 [−0.29, +0.09] | +0.90 / +0.76 | +0.41 / +0.43 |
| **起始速度 < 0.5 m/s** | 120 | **−0.02 [−0.39, +0.34] / +0.00 [−0.32, +0.34]** | +0.31 [+0.00, +0.61] / +0.34 [+0.03, +0.62] | −0.18 [−0.42, +0.02] / +0.06 [−0.22, +0.33] | +0.16 [−0.18, +0.52] / −0.06 | **−0.74 [−1.22, −0.28] / −0.64 [−1.10, −0.18]** |
| **Intersections + Multi-Lane** | 158 | +0.58 [+0.19, +0.95] / +0.61 [+0.23, +0.96] | +0.37 [+0.13, +0.62] / +0.40 [+0.13, +0.65] | −0.07 [−0.40, +0.26] / +0.19 [−0.17, +0.54] | +0.65 [+0.25, +1.06] / +0.42 [−0.03, +0.84] | +0.40 / +0.36（CI 跨零） |

（两个 cluster 分开：Intersections 116 帧 +0.62 / +0.57，Multi-Lane 42 帧 +0.48 / +0.71，后者 CI 跨零。`cls ego` − cv 在静止帧上 −0.33 [−0.55, −0.12]，`ridge ego` −0.63。全表 `route2a_paired.csv`。）

**判定：预期成立**——静止帧上 `cls_late` 不比 cv 好（Δ ≈ 0，半宽 0.35），第 34 条的模式延续。但它也不比 cv 差，这是新的：
回归 head（`ridge ego`、`ridge_late`）在静止帧上比 cv 低 0.6–0.7，分类头把这块亏损补平了（对 `cls ego` +0.3，CI 下端 ≈ 0），
所以「静止帧输给 cv」在我们的 head 家族里是**回归平均掉「继续停」这一 mode** 的代价，不是 route 信息的问题；分类头之后剩下的只是追平，不是超过。
路口 / 多车道上 `cls_late` 已经和原生 plan 打平（−0.07 / +0.19，CI 跨零），比 cv 高 0.6：在 intent 已作为 ego 输入的前提下，head 层面没有可见的 route 缺口可补。

### 实验 2b：route 进 backbone（WOD intent → desire 脉冲）

映射：GO_LEFT → turnLeft、GO_RIGHT → turnRight，直行 / UNKNOWN → none；one-hot 每步都给，OPModel 按 modeld 取上升沿脉冲；
81 386 / 522 023 帧的 intent 是转弯。run：box 上 `drive_backbones/heads_train/20260925-154105`（head）、`op_route/2b/20260925-154719`（读数），
表 `route2b_{arms,paired,ade}.csv`。统计同 2a（frame-mean RFS，sequence bootstrap）；每一行都是「+desire − 同一 arm 不带 desire」的配对差。

| 子集 | n | Cinque 原生 | Lebowski 原生 | Cinque `cls_late` | Lebowski `cls_late` | Cinque `ridge_late` | Lebowski `ridge_late` |
|:--|--:|:--|:--|:--|:--|:--|:--|
| 全部 rater 帧 | 479 | **−0.13 [−0.23, −0.05]** | −0.04 [−0.11, +0.03] | −0.06 [−0.16, +0.03] | −0.08 [−0.17, +0.02] | −0.03 [−0.12, +0.05] | +0.00 [−0.09, +0.08] |
| 起始速度 < 0.5 m/s | 120 | **−0.66 [−0.99, −0.35]** | −0.21 [−0.50, +0.06] | −0.03 [−0.23, +0.16] | −0.00 [−0.15, +0.14] | −0.01 [−0.23, +0.21] | −0.09 [−0.33, +0.14] |
| Intersections + Multi-Lane | 158 | −0.08 [−0.22, +0.07] | +0.01 [−0.13, +0.14] | −0.08 [−0.24, +0.10] | −0.03 [−0.20, +0.13] | **−0.15 [−0.29, −0.00]** | −0.06 [−0.18, +0.05] |
| 其中 Intersections | 116 | −0.11 [−0.30, +0.07] | +0.06 [−0.10, +0.24] | −0.08 [−0.30, +0.14] | −0.02 [−0.23, +0.19] | −0.19 [−0.38, +0.01] | −0.10 [−0.26, +0.04] |
| 转弯 intent 帧（追加，未预登记） | 52 | −0.16 [−0.72, +0.39] | +0.22 [−0.30, +0.72] | −0.08 [−0.51, +0.35] | +0.14 [−0.16, +0.49] | +0.02 [−0.18, +0.22] | +0.17 [−0.19, +0.57] |

带 desire 的绝对值（cluster mean）：Cinque 原生 7.87（不带 8.00）、Lebowski 原生 7.82（7.89）、`cls_late` 7.55 / 7.62（7.64 / 7.73）。
不读特征的 `ridge ego` 逐位不变（|Δ| < 1e-5），`cls ego K1024` 只差 −0.005 [−0.06, +0.04]（L-BFGS 重拟合的噪声），说明两个 run 的行、split、judge 完全配对。

第 22 条的 ADE 读数（全部 val 帧，s_ego 第 1–9 档，对不带 desire 的同一 arm）：

| arm | pre-onset（n = 1291） | 全部帧（95 724） | 直行帧（42 497） |
|:--|:--|:--|:--|
| Cinque `ridge_late` | **−0.227 [−0.309, −0.149]** | −0.025 [−0.033, −0.017] | +0.010 [+0.006, +0.014] |
| Lebowski `ridge_late` | **−0.213 [−0.282, −0.143]** | −0.022 [−0.028, −0.017] | +0.007 [+0.004, +0.010] |
| Cinque `cls_late` | −0.190 [−0.406, −0.032] | −0.040 [−0.059, −0.022] | −0.040 |
| Lebowski `cls_late` | −0.072 [−0.154, −0.001] | −0.013 [−0.025, −0.001] | −0.006（CI 跨零） |

**判定：route 进 backbone 没有买回静止帧和路口，预期两条都不成立。**

- 原生 plan：预登记的主读数是对 8.005 / 7.886 的配对 Δ。Cinque **变差** −0.13（CI 不跨零），Lebowski 不变；「全集主表不能变差」这一条对 Cinque 原生不成立。
  预期的「路口上 +」没有出现（Intersections −0.11 / +0.06，CI 都跨零；转弯 intent 帧 52 个，CI 半宽 0.5，读不动）。
  预期的「静止帧不动」也不对：Cinque 在静止帧上 **−0.66**，比 cv 还低 0.50 [0.06, 0.94]——desire 脉冲让停着的车起步，而 rater 在这些帧上偏好继续等
  （第 34 条失败案例的同一模式，被 desire 放大）。
- head：`cls_late` / `ridge_late` 在 RFS 的任何预登记子集上都没有变好（全部 CI 跨零或为负），路口上 Cinque `ridge_late` 还低 0.15。
- ADE 上有一个真实但方向不同的效应：pre-onset 帧（机动开始前、ego 状态还看不出要转的帧）上 `ridge_late` 的 ADE 降 0.21–0.23 m（CI 不跨零），
  即 `temporal` 里确实带进了 intent 的方向信息，回归 head 读得出来。它换不成 RFS：pre-onset 帧里只有很少是 rater 帧，而且 RFS 看的是「选对 mode」，
  ADE 看的是「往正确方向偏一点」（第 2 / 10 条）。另外 intent 本来就在 ego 特征里（2a 的前提），pre-onset 上的 ADE 增益说明 late fusion 下
  特征通道比 ego 通道用得更好，不说明 openpilot「理解了路线」。

合起来和 comma 的先例一致：openpilot 的 desire 是低速转弯 / 变道的触发信号，不是路口导航；把 WOD 的 routing intent 当 desire 喂进去，
在开环上不带来 RFS，在停着的帧上反而有害（Cinque）。NAVSIM 考试里 turn desire 一致拖分（−1.0 到 −3.4 EPDMS，第 37 条）是同一件事。

## 三个实验合起来（给第 40 条）

| # | 结论 | 对主线的含义 |
|---|---|---|
| 1 | `temporal` 保留了车辆 cut-in 类的突发 hazard（冻结特征 + ridge 翻转率 42–47%，高于 TFv6 waypoint 的 39%），行人 / 横穿物 / 灯上是零 | openpilot 可以直接做第 3 层里「车」的那一半；行人那一半要别的表征或别的数据 |
| 2a | intent 已在 ego 输入时，分类头在路口上已追平原生 plan；静止帧上「不如 cv」是回归平均掉「继续停」的代价，分类头补平 | head 层面没有 route 缺口可补；静止帧要靠多模态读出，不是 route |
| 2b | intent → desire 进 backbone：RFS 不涨，Cinque 原生 −0.13、静止帧 −0.66；只有 pre-onset ADE −0.2 m | 不采用 desire 作 route 输入；route 继续走 ego 侧 |
