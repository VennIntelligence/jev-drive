# 融合前诊断：谁看得见什么、谁对什么起反应、缺口在哪（9 项，1–2 张卡）

状态: 待排（预登记，写于任何新拟合、新抽取之前，2026-09-25 18:00；用户 2026-09-25 拍板三项，见「用户决定」；等中央调度员分卡后开跑）
用户决定（2026-09-25）: (1) SAM 3.1 权重直接用 ModelScope 同名副本，SAM License 已接受；(2) 新建 `envs/sam3`；(3) 卡由中央调度员分配，可能是 1 张也可能是 2 张，本文件不指定卡号，两种排程都写在下面。
上游: 第 22 条（judge 口径）、第 24 条（P3 阶梯）、第 25 条（continuation prior + reaction decoder）、第 32 条（P5 v0）、第 34 条（WOD zero-shot）、第 35 条（榜单 E 层）、
第 40 条及后续 (iii)（[driving-backbones](2026-09-24-driving-backbones/README.md)）、第 42 条（[reactivity](2026-09-25-reactivity-program.md) 的 D0 / M-C）、
[op-temporal-p5-and-route](2026-09-25-openpilot-temporal-p5-and-route.md)（实验 1、2a、2b）
主线: openpilot 冻结；闭环考试仍暂停。本文件全部是开环、离线，读已有特征和已有帧。

## 目的

融合架构有四个候选零件：openpilot 冻结的 `temporal` 特征（512 维，进 plan head 之前的时序 token）、Qwen3-VL-4B 原生视频 token（`L18_last` / `L18_mean`）、
SAM 3.1（Meta 的开放词表分割 / 跟踪模型，文本概念 prompt）作为结构化感知 schema（「这帧里有哪些物体、在 BEV 的哪里」的 JSON 状态）的来源、
Jev 式的 typed decision head（输出离散的动作类型 + 参数，而不是整条轨迹回归）。在选之前，先用盘上已有的数据量清楚三件事：
**谁看得见什么**（表征 / 感知的覆盖）、**谁对什么起反应**（读出之后的行为）、**缺口落在哪一层**（感知、决策、路线）。

已经知道的（本文件不重做，只当先验）：

| 来源 | 事实 |
|:--|:--|
| 第 40 条 | openpilot `temporal` + `ridge_late`（先拟合 ego ridge，再在 ego + 特征上拟合它的残差）在 WOD 上是全阶梯最强的冻结表征：pre-onset（机动开始前的帧）−0.29 / −0.32 m，RFS（Rater Feedback Score，0–10）cls 头 7.64 / 7.73 |
| 实验 1 | 同一 `ridge_late` 在 P5 上翻转率 42–47%，全部来自车辆 cut-in（HighwayCutIn 86–89%）；行人 family 与 Light 为 0%，行人 probe AUC 0.50–0.53，Qwen `L18_last` 0.62–0.88 |
| 实验 2a | 起始速度 < 0.5 m/s 的 120 个 rater 帧上 `cls_late` 与 cv（constant velocity）打平；Intersections 上与原生 plan 打平 |
| 第 42 条 D0 | 行人信息在 openpilot 的 vision encoder 输出里就没有（AUC 0.515 / 0.519），不是时序模块丢的 |
| 第 42 条 M-C | 配对差分监督的线性 reaction head（prior = openpilot `ridge_late`，输入 Qwen `L18_last` ⊕ `temporal`）：cut-in +13 pp，行人仍 0；**只 Qwen 流也是 0**；训练对上行人拟合斜率中位 0.12，瓶颈在 pooled 输入 |
| 第 22 条 | judge：主判 s_ego（ego-only readout 的残差）第 1–9 档上的 pre-onset ADE Δ，RFS 并排，DiD（pre-onset Δ − straight Δ），第 10 档单独 |
| 第 32 条 | P5 judge：定向翻转率（expert 在 x⁺/x⁻ 两侧的 2 s 速度差 Δ_expert 与模型 Δ 同号且 \|Δ_model\| > τ_model），τ 由各自 null（只换天气的对）定，报样本外 null false-flip |

所以 Q5 已完成、Q9 的 pooled 版本已完成（见下），Q1 在 P5 行人上的结果大概率是 0。本文件的新信息主要来自 Q1 的 WOD 半边、Q2、Q4、Q6、Q7、Q8 和 Q9 的空间 token 版本。

## 盘上已有什么（2026-09-25 17:55 在 box 上只读核对）

| 数据 | 路径（`$DATA_DIR` 下） | 规模 |
|:--|:--|:--|
| WOD index（含 `cluster`、`intent`、`n_pref`） | `processed/waymo_e2e/index.parquet` | train 415 663 / val 106 360 / test 738；val rater 帧 479（Interections 116、Foreign Object Debris 78、Cyclist 71、Pedestrian 52、Multi-Lane 42、Single-Lane 38、Special Vehicles 25、Others 22、Cut_ins 20、Construction 15；`Interections` 是 WOD 原拼写） |
| openpilot 特征，子集 | `features/op_{small,cinque,lebowski}_p3` | 20 237 行，`temporal` / `vision` / `hidden` / `plan` |
| openpilot 特征，train + val | `features/op_{cinque,lebowski}_p3_trainval` | 522 023 行，只有 `temporal` |
| Qwen3-VL-4B 视频，子集 | `features/qwenvid_p3` | 19 663 行，`L18_last` 2560 / `L18_mean` / `vis_mean` / `vit_mean` |
| Qwen3-VL-4B 视频，train thin=4 | `features/qwenvid_train_t4`（356 个 shard 目录，39 GB） | train 137 533 + val 19 663 行；另有 **`L18_grid`（4×4 空间网格）** |
| V-JEPA 2 | `features/vjepa2_p3`、`vjepa2_p3_trainval` | 19 663 / 506 517 行（前视） |
| 逐帧预测（第 40 条 (iii)） | `runs/drive_backbones/heads_train/20260925-110819/p3drive_heads_preds_dir0.npz` | val 106 360 帧：`ridge ego`、`ridge_late` × 2、`cls ego K1024`、`cls_late` × 2，外加 `fut`、`speed`、`s_ego`；**没有原生 plan 和 cv** |
| 原生 plan / cv 的 rater 帧打分 | `runs/wod_zeroshot/score/*/per_frame.npz`（479 帧，含 `cluster`、`intent`、三条 rater 轨迹与分数、cv） | openpilot 原生逐帧轨迹在 `runs/wod_zeroshot/openpilot`（执行时确认具体 run） |
| P5 v0 index | `processed/carla_p5/{index,obs,null}.parquet`、`pairs.csv` | 29 827 行（obs 9919 + train 19 908；P5 20 298、P4 9529）；165 对（`t_trig`/`t_vis`/`t_div`/`t_last`，单位 20 Hz tick） |
| P5 特征 | `processed/carla_p5/features`（Qwen，19 块 × 1500）、`op_{cinque,lebowski}`（`temporal`）、`op_*_vis`（`temporal`/`vision`/`hidden`） | 29 827 行；**没有 V-JEPA 2，没有 Qwen 空间网格** |
| P5 原始帧 | `runs/p5_pairs/gen/attempts/<run>/<attempt>/`：`cams/{front,front_left,front_right}/*.jpg`（972×1079）、`actors.npz`（frame/id/xyz/yaw/v）、`actor_kinds.json`（蓝图、scenario/background、bbox 尺寸）、`frames.jsonl`、`pose.jsonl`、`lights.json` | 420 个 attempt 目录（385 个完成的 run），128 328 张 JPEG（前视 42 776），27 GB；相机标定在 `processed/carla_p5/op_plan.json` 的 `calib` |
| openpilot lead 输出 | `runs/p5_pairs/lead_save` | **空**：P5 上 openpilot 的 lead 输出没存，Q4c 要重跑一次 stream |
| nuScenes | `datasets/nuscenes`（148 GB，`samples/CAM_*` 各 34 149 张 + `v1.0-trainval` 标注） | val 150 scene / 6019 个 keyframe；前三路 18 057 张 |
| P5 v1（I1） | `runs/p5v1/gen-pdm` 在生成（178 / 1029 done） | Q9a、Q1-P5 在 v1 出来后复跑 |
| SAM 3.1 | **不在盘上**。HF `facebook/sam3.1` 是 manual gated，我们的 token（`taitanpascal`）403；ModelScope `facebook/sam3.1` 可直接拉（`sam3.1_multiplex.pt` 3.50 GB，SAM License 2025-11-19） | 见 Q4 的阻塞项 |

box 状态：GPU 3 在 17:54 空闲（其余四张 40–57 GB 在用），load 约 90 / 125 核，数据盘剩 438 GB。

## 九项诊断

共同规则见文末「硬约束」。每项的资源估计在下面的汇总表里。

### Q1 互补矩阵：openpilot 和 Qwen 加在一起，比单个好多少

- **问题**：两路表征是冗余、互补在感知（行人存在），还是互补在决策（pre-onset）？
- **做法**：head 与 judge 一字不改（`ridge_late`、`cls_late` = K = 1024 anchor 的线性 softmax 分类头，以 `cls ego` logits 为冻结 offset）。arm：
  `ego`；Cinque / Lebowski `temporal`；Qwen `L18_last`、`L18_mean`；V-JEPA 2 `mean`；concat（Cinque `temporal` ⊕ Qwen `L18_last`，Lebowski 同）；late fusion（两个单 arm 的样本外预测等权平均；cls 头为 logits 相加）。
  concat 的两路先按训练行标准化、再各乘 1/√d，使两路总方差相等（与 M-C 相同）。
  - WOD (a)：`p2p3_v1` 的 19 663 帧，第 40 条的半/半 cross-fit，每个 rater 帧恰评一次。
  - WOD (b)：train 训、val 评，训练行 = `qwenvid_train_t4` 的 137 533 行 ∩ op trainval ∩ V-JEPA 2 trainval，评估行 = val 的 19 663 行（Qwen val 只覆盖这些）。
  - P5：29 827 行，`p5_exam` 按路线 5 折，一字不改；V-JEPA 2 在 P5 上没抽，列为可选（见资源表）。
- **输出**：每个 arm 的 pre-onset Δ（第 1–9 档）、RFS Δ（cluster mean 与 frame mean）、DiD、P5 按 family 的翻转率与 null false-flip；
  **concat − best single 的配对 Δ 与 95% CI**（WOD 按 sequence、P5 按路线 bootstrap）。best single 在 fit 半 / 训练 fold 上按同一读数选，不在评估行上选，避免选择偏差。
  另报每个 arm 按 WOD cluster 的 RFS，作为能力矩阵的一列。
- **已有数据**：全部特征都在；单 arm 的 WOD 数字大多已有（第 40 条），这里重算只为在同一批帧上配对。P5 上 Qwen 单 arm 0%、openpilot 42–47%、M-C 双流行人 0% 已知。
- **预期**：P5 行人上 concat 仍 0；WOD 上 concat 对 openpilot 的增益 ≤ 0.05 m，CI 大概率跨零。

### Q2 WOD 损失解剖：openpilot 输的帧，是没看见、看见了决策错，还是路线本身有歧义

- **问题**：在 WOD 的 rater 帧上，openpilot 输分的原因分布。
- **做法**：
  - 损失帧：rater 帧上 Cinque 原生 plan 或 Cinque `cls_late` 的 RFS ≤ max(三条 rater 轨迹的分) − 1。按 cluster 分组：Interections、Special Vehicles、Multi-Lane、Pedestrian、Cut_ins、Foreign Object Debris（其余 cluster 并成「其他」，只描述）。
  - **Q2a（纯 CPU，先做）**：rater_best 对 log 与对模型的差按方向分类：纵向（2 s / 5 s 纵向位移差 > 3 m，分停 / 走两支）、横向（终点横向差 > 2 m 或朝向差 > 15°）。
    横向差异的帧按 intent 判：intent 是转弯且 rater_best 朝 intent 方向、模型没有 → 「决策错（intent 解释得了）」；intent 是直行 / UNKNOWN 而分支是转弯或并道 → 「路线歧义」。
  - **Q2b（依赖 Q4 的 SAM 环境）**：纵向损失帧上，「原因物体」由 SAM 3.1 在前视图上标出（固定 prompt 列表，地面接触点经 WOD 标定抬到 BEV，ego 未来 5 s logged 路径 ±1.5 m 走廊内、40 m 以内）。
    hazard-presence probe：标签 = 走廊内是否有该 cluster 对应类别的物体（Pedestrian → pedestrian，Cyclist → cyclist，Cut_ins / Multi-Lane → vehicle，Foreign Object Debris → cone ∪ debris，Special Vehicles → emergency vehicle），
    在 19 663 帧子集上按 sequence 4 折 cross-fit 的 logistic probe，两份：openpilot `temporal` 上一份、Qwen `L18_last` 上一份；阈值取各自 90% specificity 的工作点（在训练折上定）。
  - 每个纵向损失帧归入：**看见了但决策错**（有原因物体且 openpilot probe 过阈值）/ **没看见**（有原因物体、openpilot probe 不过）/ **无可见原因**（SAM 在走廊内没标出物体；灯、规则、让行等）。
    另加一列「Qwen 看见而 openpilot 没看见」的比例，这是融合能补的那部分。
- **输出**：cluster × 类别的比例表（带按 sequence bootstrap 的 CI）；逐帧 CSV（帧名、cluster、方向、类别、两份 probe 分数）。
- **已有数据**：Q2a 全在盘上（per_frame.npz + preds npz）；Q2b 需要 SAM 在 WOD 前视 ~20 100 张图上跑（479 个 rater 帧 + 19 663 帧子集，probe 需要训练行）。
- **限定**：同一 cluster 里 rater 帧只有 15–116 个，损失帧更少，比例的 CI 会很宽；这是描述性读数，决策规则只用合并比例。

### Q3 真实数据镜像：P5 上的 family 结论在 WOD 的对应 cluster 上有没有影子

- **问题**：实验 1 里「openpilot 对 cut-in 有反应、对行人没有」在真实数据的 cluster 上是否同样成立。
- **做法**：零拟合。已有逐帧预测按 cluster 切：Cut_ins、Pedestrian、Cyclist、Foreign Object Debris（对照：Interections、全部）。
  两个配对差：`cls_late` − `cls ego`（表征的贡献，val 全部帧的 ADE 第 1–9 档 + rater 帧 RFS），原生 − cv（rater 帧 RFS；ADE 在 20 237 帧子集上用 `op_*_p3/plan`）。
- **输出**：cluster × 读数的 Δ 与 CI（sequence bootstrap）。
- **已有数据**：全部在盘上。

### Q4 感知 schema 的召回：SAM 3.1 看得见多少，抬到 BEV 准不准

- **问题**：如果行人这条线交给结构化感知（SAM → JSON 状态 → decision head），它在我们关心的距离和天气上召回够不够；openpilot 自己的 lead 输出又召回多少。
- **做法**：
  - **Q4a P5**：观测帧（x⁺ / x⁻ / null 两侧，obs 角色 9919 行 × 3 路 = 29 757 张图），固定 prompt 列表：pedestrian、cyclist、vehicle、cone、debris、emergency vehicle，image 模式逐帧检测。
    每个实例取 mask 最低处的地面接触点，用 `op_plan.json` 里的 CARLA 标定（intrinsic 含畸变，extrinsic 高 1.806 m）按平地假设抬到 ego BEV。
    GT = `actors.npz` + `actor_kinds.json`（同一 frame 的全部 actor，bbox 底面中心）。匹配：同类别、BEV 距离 ≤ max(2 m, 0.1 × 距离) 的 Hungarian。
    **可见性**：CARLA 没存深度，被遮挡的 actor 也在 GT 里，所以召回分两层报：(i) hazard actor 在 `factor_px` 大于阈值（obs.parquet 已有，因素可见的像素数）的 x⁺ 帧上的召回，这是主读数；
    (ii) 背景 actor 在视锥内、≤ 40 m 的召回，明确标注「含遮挡，偏低」。
  - **Q4b nuScenes val**：6019 个 keyframe × 前三路 = 18 057 张，同一 prompt，GT 是 devkit 的 3D box（visibility token ≥ 3，即 ≥ 40% 可见），抬 BEV 用 `calibrated_sensor` + `ego_pose`。
  - **Q4c openpilot lead**：在 P5 的同一批 stream 上重跑 Cinque / Lebowski（实验 1 的抽取器，只多存 `leads_v3` 输出），GT lead = ego 车道走廊（未来 route ±1.5 m）内最近的车辆，≤ 80 m。
  - **Q4d 延迟**：在我们的卡上量 batch 1 的 image 模式（1 路 / 3 路，6 个 prompt）和 video 模式（5 Hz stream，约 5 个对象），报 p50 / p95 与峰值显存。
- **输出**：召回、precision、BEV 位置误差中位数 / p90，按 类别 × 距离档（0–10、10–20、20–40、40–80 m）× 天气（白天 / 夜晚按 `sun_altitude` < 0 / 雨按 `precipitation` > 30；nuScenes 按 scene description）；
  openpilot lead 的召回与距离误差；延迟表。
- **已有数据**：帧和 GT 全在；SAM 3.1 权重、环境都没有。
- **权重与环境（已决定，2026-09-25）**：HF 上 `facebook/sam3.1` 是 manual gated（我们的 token 403），**直接拉 ModelScope 同名副本**（3.50 GB，走 Alpamayo 那条路），SAM License（Meta 自定义许可，2025-11-19 版）用户已接受。
  官方要求 Python ≥ 3.12、PyTorch ≥ 2.7、CUDA ≥ 12.6，没有 transformers 集成，走 `facebookresearch/sam3` 仓库；box 上的 `envs/jevdrive` 是 3.11，所以新建 `envs/sam3`（torch cu128 或 cu130，sm_120 已支持）。
  文本 prompt 在 video 模式下官方支持（`handle_request(start_session / add_prompt)`，SAM 3.1 的 release note 有 video PCS with text prompt 一栏）。未核实：SAM 3.1 的 multiplex checkpoint 是否也提供单独的 image 检测接口；若不提供，image 模式改用同一仓库的 SAM 3 检测器（`facebook/sam3`，ModelScope 同样有镜像），这一条写进偏离日志。
- **等价性 / 坐标检查（批量之前必须过，16 帧）**：挑 16 个 hazard 清楚可见的 P5 x⁺ 帧（8 个行人、8 个车辆，覆盖三路相机和 5–40 m）：
  (a) GT actor 底面中心投到图上，落在匹配的 SAM mask 外接框内，像素误差中位 ≤ 15 px；(b) SAM 地面点抬到 BEV 与 GT 的距离，≤ 20 m 的中位 ≤ 1.0 m；
  (c) 同一张图 batched 与单张推理的输出一致（mask IoU ≥ 0.99，分数差 ≤ 1e-3）；(d) nuScenes 取 4 帧做 (a)(b)。任何一条不过，先修再批量。

### Q5 vision 层 probe

已完成：[reactivity 计划的 D0](2026-09-25-reactivity-program.md)，第 42 条。行人合并 AUC 0.515 / 0.519，不过 0.60；信息在 openpilot 的 vision encoder 就没有。
本文件只引用，不重做；I1 的 P5 v1 出来后由 reactivity 计划复跑。

### Q6 反应有多少是几何：规则门在 GT 状态和 SAM 状态上各能拿多少

- **问题**：P5 里 expert 的反应，多少能被三条几何规则解释；换成 SAM 感知到的状态后还剩多少。
- **做法**：三条规则门，参数在 P4 训练路线（role = train，不含 P5 pair 路线）上定，之后冻结：
  (i) TTC 门：走廊内前方物体的 TTC < 3.0 s；(ii) 车道侵入：任一物体 BEV 位置进入 ego 未来 3 s 路径 ±1.2 m；(iii) 行人接近：行人朝走廊方向的速度分量 > 0.5 m/s 且距走廊 < 4 m、距离 < 30 m。
  任一门触发 → 预测「制动」方向，量级取 expert 在 P4 训练帧上触发时的中位减速。按 P5 协议算定向翻转率（x⁺ 触发、x⁻ 不触发、方向与 Δ_expert 同号）与 null false-flip。
  先在 GT 状态（`actors.npz` + `pose.jsonl`）上算，再在 SAM 状态（Q4a 的 BEV 点，5 Hz 最近邻 Hungarian 关联出速度）上算。
- **输出**：每个 family 的 rule floor（GT）与 perception-limited floor（SAM），以及二者之差。
- **已有数据**：GT 部分全在；SAM 部分依赖 Q4a。
- **限定**：BehaviorAgent 本身就是规则 expert（不提前减速），GT 规则门在 v0 上接近 expert 是半同义反复；I1 的 PDM-Lite 标签出来后复跑，决策规则以 PDM-Lite 那次为准，v0 只描述。

### Q7 模式词表：数据里到底有几种反应

- **问题**：typed decision head 的输出词表应该有哪些类型；openpilot 的 desire 输入能覆盖多少。
- **做法**：
  - P5：每个 reactive 帧的 expert 两侧差（`future.npy` 的 x⁺ 减 x⁻）取三维描述：2 s 纵向位移差、最大横向位移差、onset 时间（`t_div` − `t_vis`）。标准化后 GMM，k = 1…8 按 BIC 选。
  - WOD：rater 帧里 s_ego 第 10 档（约 48 帧）以及全部 rater 帧，rater_best − log 的 20 × 2 差，取同样三维（onset 取纵 / 横差首次超过 0.5 m 的时刻），同样聚类。
  - 每个聚类映射到 openpilot 的 desire（none、turnLeft、turnRight、laneChangeLeft、laneChangeRight、keepLeft、keepRight；第 8 位保留未用）：横向类能映射，纵向类（停、让、蠕行、提前减速）映射为「无」。
- **输出**：模式词表表（模式、定义、P5 与 WOD 各自的占比、desire 覆盖与否）；覆盖率 = 能映射到 desire 的帧占比。
- **已有数据**：全在盘上，CPU。

### Q8 反应时间窗：慢通道来得及吗

- **问题**：因素可见到 expert 必须分叉之间的时间，减去感知 + 决策延迟，还剩多少；哪些 family 必须留在快通道（20 Hz 的 policy）。
- **做法**：每对取 `t_div` − `t_vis`（20 Hz tick → s）的分布，按 family；减去三档延迟：openpilot Cinque 2.3 ms（p50，smoke）、SAM 3.1 在我们卡上的实测（Q4d）+ decision head、openjev 470 ms p50（3 路，docs/baselines.md，作为 VLM 慢通道的代表）。
  **「必须留在快通道」**：某 family 的窗口 p25 < 慢通道延迟 + 0.5 s（执行余量）。
- **输出**：family × 延迟档的剩余预算表与标记。
- **已有数据**：窗口全在（`pairs.csv`）；SAM 延迟依赖 Q4d。
- **限定**：BehaviorAgent 反应晚，`t_div` 偏晚，窗口偏宽；PDM-Lite（I1）出来后给第二个估计，判定取两者中较短的。

### Q9 读出能不能救 Qwen 的行人

- **Q9a（pooled，已完成）**：就是第 42 条 M-C 的「配对差分，只 Qwen」arm：行人翻转 0%，cut-in +1.1 pp，null 5.3%。按下表规则已经落在「still 0」。
  I1 的 P5 v1 出来后在加固版上复跑同一 arm 与 hard-example 重加权对照（分钟级，CPU），这是 Q9a 的最终读数。
- **Q9b（空间 token，新）**：M-C 的诊断说 pooled 输入连训练对都拟合不了，所以把输入换成 Qwen `L18_grid`（4×4 空间网格，`qwenvid_train_t4` 已用同一 recipe 抽过）。
  在 P5 obs 行（9919 行）上重抽 `L18_grid`；head = 在 16 个网格 token 上的单层 attention pooling + 线性输出（与 P2 的 attention readout 同构），配对差分 loss（同 M-C），
  对照 = 同结构的 hard-example 重加权；μ 项（直行帧上修正为零）由 null 对承担，不再读 role = train 行（因为 train 行没有网格特征；这一点与 M-C 不同，预先写明）。
  按路线 5 折，`p5_exam.exam` 一字不改。
- **输出**：行人合并翻转率 [CI]、cut-in 对 prior 的配对 Δ、样本外 null false-flip；训练对上的拟合斜率（同 M-C 偏离 5 的诊断）。
- **判据**（沿用 M-C）：行人 ≥ 20% 且路线 bootstrap CI 下端 > null false-flip，且 null ≤ 7%。

## 预登记的决策规则

「CI 不跨零」都是 95%，配对 bootstrap（WOD 按 sequence，P5 按路线，500 次）。

| 观测 | 操作化 | 含义 | 方向 |
|:--|:--|:--|:--|
| Q1 concat 不加分 | concat − best single 在 WOD pre-onset、RFS、P5 合并翻转率三个读数上 CI 都跨零（或偏负） | 冗余 | 只调 openpilot，去掉 Qwen |
| Q1 只在行人 family 上加分，且 Q4 SAM 召回够 | P5 行人合并翻转 concat − best single CI > 0，WOD 两个读数 CI 跨零；SAM：hazard 行人召回 ≥ 0.9（≤ 30 m、白天）且 ≥ 0.8（全部），≤ 20 m BEV 误差中位 ≤ 1 m，nuScenes 行人 ≤ 30 m 召回 ≥ 0.8 | Qwen 只贡献「存在」 | SAM → JSON 状态 → decision head；去掉 Qwen |
| Q1 在 pre-onset 上也加分 | WOD pre-onset concat − best single CI 整体 < 0（(a) 与 (b) 两个协议都成立） | Qwen token 带决策相关信息 | token 进 decision head（M-C）；SAM 只做离线标注 |
| Q2 损失多数是「看见了但决策错」 | 六个 cluster 合并的纵向损失帧里该类 > 50%，加上横向里「intent 解释得了」的帧 | 缺决策 | Jev 式 head 有理由，词表取自 Q7 |
| Q2 多数是「没看见」 | 合并比例 > 50% | 缺感知 | 先建 Q4 的感知层，decision head 其后 |
| Q6 GT 规则 floor 接近 expert | PDM-Lite 标签下，GT 规则门的 family 合并翻转率 ≥ 80%，null false-flip ≤ 7% | 反应主要是几何 | 规则 + 学习的混合就够；VLM 的价值必须在 Q2 的语义帧（无可见原因 / 看见但决策错）上单独证明 |
| Q9 翻转离开 0，null 不动 | Q9b 过 M-C 判据 | 是读出问题 | Qwen 保留为行人通道 |
| Q9 仍为 0 | Q9a 与 Q9b 都不过 | 是表征问题 | 行人走 SAM 通道 |

规则冲突时（比如 Q1 说「冗余」而 Q9b 说「读出问题」），按「越靠近行为的读数越优先」：Q9b > Q1 的 P5 行 > Q1 的 WOD 行；冲突本身写进「架构方向」一页。
Q2 两类都不过半，报三类比例，架构页写「混合」，不硬判。

## 交付物

1. **能力矩阵**：模型（ego、Cinque、Lebowski、Qwen `L18_last` / `L18_mean`、V-JEPA 2、concat、late fusion、原生 plan、SAM 规则门）× family / cluster，WOD（RFS、pre-onset Δ）与 P5（翻转率）并排。
2. **感知 schema 盲区表**：类别 × 距离档 × 天气的 SAM 召回 / precision / BEV 误差（P5 与 nuScenes 并排），加 openpilot lead 召回一行。
3. **模式词表表**：Q7 的聚类、占比、desire 覆盖。
4. **「架构方向」一页**：逐条填上面的决策规则，写明落在哪一格、证据是哪个数字。
5. 小 CSV 放 `research/results/fusion-diagnostics/`（每项一个子目录）；图按 CLAUDE.md 的规则由一个 doc 收录；结论进 decisions.md 新开一条（第 43 条，或当时的下一个号）。

## 资源估计与排程

预算：**一张专用 RTX PRO 6000（96 GB）+ 约 20 核**，其余卡和核不碰。box CPU 目前被 CARLA 压到 cgroup 上限附近（load 约 90 / 125），
D0 的考试就因此从「分钟级」拖到 60 min，所以下面 CPU 项的墙钟按 20 核、有争用估，上界放宽了。

| 项 | GPU·h | 峰值显存 | 核 | 墙钟 | 新增磁盘 | 依据 |
|:--|--:|--:|--:|:--|--:|:--|
| Q3 | 0 | — | 4 | 10–20 min | < 10 MB | 纯切片，预测都在 npz 里 |
| Q7 | 0 | — | 8 | 20–40 min | < 10 MB | GMM 在几千行上 |
| Q2a | 0 | — | 4 | 20–30 min | < 10 MB | 479 帧 |
| Q6（GT 部分） | 0 | — | 12 | 30–60 min | < 50 MB | 385 run 的 actors + pose，向量化 |
| Q1 ridge 全部 arm（WOD a/b + P5） | 0 | — | 12 | 30–60 min | < 200 MB | 最宽 3072 维，137.5k 行的 XᵀX 秒级；P5 `p5_exam` 上次 17 min |
| Q1 `cls_late` | 1.0–1.5 | ≤ 20 GB | 8 | 1.5–2 h | < 300 MB | 415k × 512 上 L-BFGS 每次约 8 TFLOP、15–30 min；137.5k × 3072 每次约 2.6 TFLOP → 每 arm 5–15 min，约 7 个特征 arm；19.6k 子集每 arm 1–2 min |
| Q1 可选：V-JEPA 2 在 P5 上抽 | 0.4–0.6 | ≤ 3 GB | 8 | 30 min | 150 MB | 48 ms / 4 帧 clip，29.8k 行 |
| Q4 环境 + 权重 | 0 | — | 4 | 工程半天；下载约 10 min | 约 12 GB（env 8 + 权重 3.5） | ModelScope 镜像 |
| Q4 等价性 16 帧 + 200 帧 profiling | 0.2 | ≤ 30 GB | 8 | 30 min | < 100 MB | 批量前必做 |
| Q4 SAM 批量（P5 29.8k + nuScenes 18.1k + WOD 前视 20.1k = 68k 张） | **1.5–3.8** | ≤ 30 GB（估） | 10（JPEG 解码 + 缩放） | 1.5–4 h | 3–6 GB（实例表 + RLE mask） | 下面单独说明 |
| Q4c openpilot lead 重跑 | 0.2–0.4 | < 10 GB | 12 | 15–25 min | < 100 MB | D0 同一批 stream 实测 11 min（12 核） |
| Q4d 延迟 | 0.1 | ≤ 10 GB | 2 | 10 min | — | |
| Q4 匹配 / 统计，Q6（SAM 部分），Q8，Q2b probe | 0 | — | 12 | 1–1.5 h | < 100 MB | Hungarian 按帧并行 |
| Q9a（I1 之后） | 0 | — | 8 | 5–20 min | — | M-C 实测 4 min（有争用时 20 min） |
| Q9b Qwen `L18_grid` 在 P5 obs 上重抽 + head | 0.8–1.7 | ≤ 12 GB | 8 | 1–2 h | < 1 GB | qwenvid train 抽取 compile b8 292 ms/帧，P5 上次 eager b2 619 ms/帧；9919 行 |
| **合计** | **约 3.8–7.7 GPU·h**（含可选的 V-JEPA 2 为 4.2–8.3；I1 v1 上 Q9b 再抽一次另加 1–2） | 同时 ≤ 60 GB | ≤ 20 | 计算关键路径 7–10 h；含工程约 2 个工作日 | 约 20 GB（盘上剩 438 GB） | |

**SAM 吞吐的估法**（最大的不确定性，所以批量前 200 帧实测，以实测为准）：官方数字是 SAM 3 在 H200 上 100+ 个对象每张 30 ms、SAM 3.1 在 H100 上约 5 个对象 32 fps（video）；
我们的卡按 bf16 算力和显存带宽比 H100/H200 慢约 1.5–2.5 倍，单 prompt 每张 45–75 ms。6 个文本 prompt 共享一次图像编码，每多一个 prompt 付一次融合编码器 + DETR decoder（估编码占单 prompt 时间的约 60%），
所以每张 80–200 ms（中位估 120 ms），68k 张 1.5–3.8 h。等效只算 P5 观测帧（29.8k 张）是 0.7–1.7 h。video 模式只用于 Q4d 的延迟和 Q6 的速度关联的抽查，不做全量。
**停机线**：200 帧 profiling 外推全量 > 7.6 h（估计上界的 2 倍）就停下来报，考虑的降档依次是：WOD 只做 479 个 rater 帧 + 5000 帧 probe 训练行、nuScenes 只做 CAM_FRONT、P5 只做 x⁺ 与 null。

**排程 A：分到一张卡**（并发与先后）：

1. **第 0 阶段（CPU，立刻）**：Q3、Q7、Q2a、Q6（GT）并行，总 ≤ 20 核，约 1 h 墙钟。同时装 `envs/sam3`、拉 ModelScope 权重。
2. **第 1 阶段（GPU 两路并发）**：
   - 路 A：Q4 的 16 帧检查 → 200 帧 profiling → SAM 批量（P5 → nuScenes → WOD 的顺序，逐数据集落盘、可续跑）。≤ 30 GB、10 核。
   - 路 B：Q1 的 `cls_late`（WOD a、b 与 P5）和 ridge，≤ 20 GB、8 核；它是 L-BFGS 的小矩阵乘，与 SAM 共卡会互相拖慢约 20–40%，但总墙钟仍短于串行。
   - Q4c（lead 重跑，< 10 GB、12 核）排在路 B 之后，避免两路同时吃 CPU 超过 20 核。
3. **第 2 阶段**：SAM 批量结束后，Q4 统计、Q6（SAM）、Q8、Q2b 在 CPU 上跑（约 1.5 h）；同时 GPU 跑 Q9b 的网格重抽（≤ 12 GB）。
4. **第 3 阶段**：I1 的 P5 v1 出来后（约 10–12 h 后），Q9a、Q1-P5、Q6 在 v1 上复跑（CPU 为主，约 1 h；Q9b 需在 v1 obs 帧上再抽一次网格，约 1–2 h GPU）。
5. 汇总四张表与架构页，写 decisions.md。

整个包：不等 I1 的部分约 **1.5–2 天**（其中计算关键路径 7–10 h，其余是 SAM 接入、BEV 抬升和匹配的工程）；加上 I1 v1 的复跑再加半天。

**排程 B：分到两张卡**。GPU·h 总量不变（3.8–7.7），变的是关键路径：
- 卡 1 只跑 SAM 链（16 帧检查 → 200 帧 profiling → 批量 → Q4d 延迟），不与任何 L-BFGS 共卡，批量不再被拖慢 20–40%，估 1.2–3 h。
- 卡 2 依次跑 Q1 `cls_late`（1.5 h）→ Q4c lead 重跑（20 min）→ Q9b 网格重抽 + head（1–2 h）→ 可选 V-JEPA 2 抽取（30 min）。
- CPU 仍是 20 核上限，是两张卡时的瓶颈：SAM 的 JPEG 解码 10 核、Q1 8 核已到顶，Q4c 与 Q9b 只能排在 Q1 之后；第 0 阶段的 CPU 项照旧先跑。
- 计算关键路径从 7–10 h 缩到约 **4–5 h**，整包仍约 1.5 天（工程时间不变）；I1 v1 后的复跑同 A。
- 两张卡时显存合计 ≤ 60 GB 的约束不变，每张 ≤ 30 GB。

## 硬约束

- judge 一字不改：WOD 用第 22 条口径（`waymo_ladder.rejudge`），P5 用 v0 的定向翻转率 + 样本外 null false-flip（`p5_exam.exam`）。新增的只是 arm、examinee 和描述性读数。
- 训练与考试按路线（P5）或 sequence（WOD）分开；所有阈值（probe 工作点、规则门参数、best single 的选择）只在训练行上定。
- SAM 部分必须先过上面的 16 帧坐标 / 等价性检查，才能批量。
- 只用中央调度员分给本任务的卡（1 张按排程 A，2 张按排程 B；卡号以 `runs/schedule.md` 为准，本文件不指定）和约 20 核；其他卡上什么都不跑。每个 > 1 min 的作业进 tmux，写 `log.txt` / `events.jsonl` / `tb/`，并在 `runs/zeroshot-exam/gpu-plan.md` 登记。
- 任何一步超出本表估计的 2 倍就停下来报，不自行加资源。
- 偏离写进下面的偏离日志，时间戳早于受影响的数字。

## 偏离与澄清日志

（执行时追加。）

- 2026-09-25 18:20 CST [Q3] 原生 plan 用 decision 40 (iii) `heads_readout` 读的同一份：`processed/drive_backbones/op_{cinque,lebowski}_native.npz`（20 237 帧子集的流式抽取，`wod` 即 `op_*_p3/plan` 换算成 WOD 后轴 waypoint 的版本，覆盖全部 479 个 rater 帧）。
  ADE 的 native − cv 按登记在这 20 237 帧上算，s_ego 分档取 `heads_train/20260925-110819` 在 val 全部 106 360 帧上的十分位（与 `rejudge` 同一口径），子集只是取其中的行。
  另加一行描述性读数：native − cv 在 val 全部帧上的 ADE（`op_*_trainval_native.npz`，实验 2a 用的那份），标为 side，不进判据。`cls_late − cls ego` 另报 pre-onset 第 1–9 档一列，同样标 side。
  CI 用 `waymo_p1.paired`（sequence bootstrap，1000 次）。
- 2026-09-25 18:20 CST [Q2a] (1) 原生 plan 同 [Q3]（`op_cinque_native.npz`）；`cls_late` = `heads_train/20260925-110819` 的 `cls_late op-cinque temporal`。
  (2) 损失帧 = 两者之一 RFS ≤ max(rater 分) − 1 的并集；方向按「输了的那个模型」判，两个都输时按原生 plan 判（问题问的是 openpilot 输的帧）；另附按模型分开的两张表。
  (3) 差值 = 模型 − rater_best，在 rater_best 自己的纵 / 横坐标系里量（与 RFS 同一个 `_rater_frames`）：纵向取 2 s（第 8 个点）与 5 s（第 20 个点）的纵向投影，任一 |·| > 3 m 算纵向，
  rater_best 在后（模型走得更远）为「停」支、rater_best 在前为「走」支；横向取 5 s 的横向投影 |·| > 2 m，或朝向差 > 15°（朝向 = 最后 0.5 s 的位移方向，两条轨迹这段都 ≥ 0.5 m 才判）。
  同时满足横向与纵向的帧归横向（因为横向要再按 intent 拆），另记一个 `both` 标记；两者都不满足记「无方向」（例如低速 trust region 缩小导致的输分）。
  (4) intent 拆分：intent = GO_LEFT / GO_RIGHT，且 rater_best 在 5 s 的弦方位角朝 intent 一侧超过 5°（`waymo.ONSET_BEARING`）而模型没有 → 「决策错（intent 解释得了）」；
  intent = GO_STRAIGHT / UNKNOWN → 「路线歧义」；intent 是转弯但不满足前一条（例如两者都转、只是车道不同）→ 「横向其他」。
  (5) rater_best − log 用同一套分类另存为描述列。比例的 CI：损失帧上的类别指示变量，sequence bootstrap（`traj.boot_ci`，1000 次）。Q2b 的列（SAM 原因物体、两份 probe 分数、类别）留空。
- 2026-09-25 18:20 CST [Q7] (1) P5 的 reactive 帧 = `p5_exam.exam` 的定义（|Δ_expert| > τ_exp，τ_exp = max(null |Δ_expert| 的 95 分位, 0.5)）。
  (2) 三维描述：2 s 纵向差 = x⁺ − x⁻ 在第 8 个点（2.0 s）的 x；最大横向差 = |Δy| 最大那一点的带符号 Δy（左正，desire 映射需要方向）；onset 按登记 P5 取 (t_div − t_vis) × 0.05 s（逐对常数），
  WOD 取 |Δx| 或 |Δy| 首次 > 0.5 m 的时刻（0.25–5 s；从不超过记 5.25 s，删失）。两边 onset 含义不同，所以**分开聚类**（各自标准化、各自 BIC 选 k），词表通过同一套命名规则对齐。
  (3) GMM：sklearn `GaussianMixture`，full covariance，n_init = 10，reg_covar = 1e-4，seed 0，k = 1…8 取 BIC 最小。WOD 聚类在全部 479 个 rater 帧上拟合，s_ego 第 10 档（val 全部帧的十分位）另报各簇占比，并单独拟合一次。
  (4) 簇 → 模式 → desire 的命名规则（按簇内中位数，物理单位，预先写死）：|最大横向差| ≥ 1.0 m 为横向类，≥ 6 m → turnLeft/Right，2.5–6 m → laneChangeLeft/Right，1.0–2.5 m → keepLeft/Right；
  否则为纵向类（desire「无」）：2 s 纵向差 < −1 m 为「减速 / 停」，> +1 m 为「加速 / 走」，其余为「近零差」。覆盖率 = 落在横向类簇的帧占比。
- 2026-09-25 18:20 CST [Q6] (1) **P4 的 attempt 目录里没有 `actors.npz`**（P4 录制器不存 actor），GT 状态下 P4 训练路线上算不出门是否触发，登记的「在 P4 训练路线上定参数」做不到。
  处理：三条门的阈值就用登记里写明的数值（TTC 3.0 s；±1.2 m；0.5 m/s、4 m、30 m），不做任何拟合；唯一要拟合的量是制动量级 m（门触发时 expert 的中位减速 v0 − v(2 s)），
  改在 P5 role = train 帧上按 `p5_exam.folds` 的路线 5 折**样本外**估（评估第 f 折时只用其余 4 折路线的训练帧，与 `p5_exam.heads` 的训练行口径相同，路线不相交）。
  注意：Δ_model = −m_f ·（门(x⁺) − 门(x⁻)），取值只有 {−m, 0, +m}，定向翻转率与 null false-flip 对 m 不敏感，m 只影响量级列。
  (2) 「ego 未来 3 s 路径」用路线中心线（`route.json`，从 ego 最近点向前 max(v0 × 3 s, 5 m)），**不用 logged 未来**：x⁺ 的 logged 未来已经含 expert 的制动，会把答案漏进门里。
  TTC 门的走廊 = 同一中心线向前 60 m、±1.2 m，TTC = 纵向间距（减 ego 前悬 = bbox 半长）/ 接近速度（v0 − 物体速度在路径切向的分量，> 0 才算）；行人门的走廊 = 中心线向前 30 m，
  「距走廊」= 到中心线横距 − 1.2 m，「朝走廊方向的速度分量」= 物体速度在指向中心线的法向上的分量；只对 walker 计（自行车按 vehicle 类进 (i)(ii)）。
  (3) 物体 = 同一 tick 的全部 actor（除 ego），与 ego 高差 > 8 m 的剔除（x⁻ 的 hazard 被藏在地下 500 m、生成前也在地下）。坐标用 CARLA 左手系取 y 反号转成右手系，ego BEV 原点在车辆位置（bbox 中心地面），x 前 y 左。
  (4) 考官：`p5_exam.exam` 一字不改，examinee = `gate any`（主读数，rule floor）以及三条门各自（描述）。

## 结果

（跑完再填；box 上 run dir：`$DATA_DIR/runs/fusion_diag/<item>/<time>`。）
