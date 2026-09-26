# 消融矩阵盘点（2026-09-26，只读）

状态: 调研报告，2026-09-26 由 Opus 子代理只读写成，主会话落盘；数字以所引出处为准，结论已进 [decisions.md](decisions.md) 第 46、47 条（就地修正见第 35 条）。

范围：decisions.md 第 20–45 条、research/results/**、todos 2026-09-24 driving-backbones、2026-09-25-*、2026-09-26-*。只抄已有数字；找不到写「未见」；「进行中」= 已登记、box 上在跑、没出数（G0 / G1 / G3b）。
Box 核对：2026-09-26 只读 `ls` / `du`，没跑任何东西。

## 0. 协议代号（同一格多个数时靠它区分）

| 代号 | 数据 / 切分 | 读数 | 出处 |
|:--|:--|:--|:--|
| W-half | WOD `p2p3_v1` 子集 20 237 帧，半/半 val 两方向 | 第 22 条：pre-onset 第 1–9 档 ΔADE（方向 0 / 1），RFS Δ | 第 23、24 条 |
| W-xfit | 同子集 19 663 帧，两方向 cross-fit 合并 | pre-onset 1–9 档（n=1265）/ 全部帧 1–9 档 / RFS Δ（478 帧，frame mean） | driving-backbones README |
| W-train | WOD train 41.6 万帧训、完整 val 评 | pre-onset 1–9 档（n=1291；V-JEPA 行 n=1249）/ RFS（479，cluster mean） | 第 24 d‴、40 条，heads-train |
| W-zs | 零样本原生 plan，479 rater 帧 | RFS cluster mean | 第 34 条，openpilot-openloop-standing |
| N-head | navtrain 10.3 万 token 拟合，navtest 12 146 官方 devkit | PDMS v1.1 / EPDMS main@0a380a9 / navhard two-stage EPDMS（无 CI） | 第 37 条修正、第 40 条 6、E6 |
| N-zs | 原生 plan，2 Hz × 1.5 s 输入 | 同上 | 第 37 条 |
| U-zs | nuScenes val quarter n=1159，UniAD/VAD 口径 | L2 均值（1/2/3 s 前各步平均）、collision % | 第 39 条 |
| U-head | nuScenes 700 train / 150 val scene，CAM_FRONT | ADE 3 s，s_ego 1–9 档，Δ vs `ridge ego`（0.514 m），ego 带 command | 第 40 条 4 |
| P5v0 / BA / PDM | P5 v0（165 对）/ v1 BehaviorAgent 集 / v1 PDM-Lite 集 | 逐帧定向翻转率 %、样本外 null false-flip | 第 32、42 条，E4 `e4_all.csv` |
| I3 | HUGSIM 3DGS 65 场景车辆配对，P5 v1 BA 上拟合后零样本 | 定向翻转率 %（1 832 reactive 帧） | i3-hugsim-pairs 子文档 |

## 1. WOD-E2E（W-*）：backbone × head

Δ 均为 arm − `ridge ego`，ADE 负好、RFS 正好。

| backbone | `ridge_late`（主判 pre-onset；全部帧；RFS） | `cls_late` K=1024 | gated residual | hard-example | pair-Δ（CARLA 训、加到真实 prior） |
|:--|:--|:--|:--|:--|:--|
| ego-only | base：`ridge ego` RFS 7.056（P0）/ 7.065（heads-train）；cv 7.10 | `cls ego` 7.262（heads-train）/ 7.311（P0）；3 seed 对 `ridge ego` +0.22 / +0.23 / +0.18 | — | — | — |
| Qwen3-VL-4B `L18_mean` | W-half −0.021 / −0.039，RFS +0.001 / −0.040；W-xfit −0.030 [−0.064, +0.006]，−0.044，RFS −0.020；W-train −0.005 [−0.055, +0.047]，RFS 6.942 | P0 train：7.30（对 `cls ego` −0.05 [−0.16, +0.05]） | W-half pooled-mlp +0.113 / +0.055、attn grid −0.000 / −0.024（RFS 全负） | L0 surprise a1（旧全集口径，train）+0.026，RFS 6.781 | — |
| Qwen 4B 原生视频 `L18_last`（d″） | W-half −0.053 [−0.092, −0.018] / −0.045 [−0.075, −0.016]；W-xfit −0.049 [−0.073, −0.028]，RFS +0.005 | 未见 | 未见 | 未见 | M-C 的 Qwen 流即它（见最后一行） |
| Qwen 2B 视频 `L14_last` | W-half −0.024 / −0.036（两向 CI 不跨零） | 未见 | 未见 | 未见 | — |
| Qwen 32B（L32/L50 × mean/last） | W-half 8 个数在 −0.036…+0.006；RFS 7/8 为负（到 −0.151） | 未见 | 未见 | 未见 | — |
| V-JEPA 2 ViT-L | W-half +0.005 / −0.101；W-xfit −0.049 [−0.095, +0.001]，RFS +0.012；W-train −0.030 [−0.104, +0.041]，RFS Δ −0.205 | 未见 | 未见 | 未见 | — |
| V-JEPA 2 ViT-g / V-JEPA ⊕ Qwen late | W-train −0.027 / −0.036，RFS Δ −0.201 / −0.234 | 未见 | 未见 | 未见 | — |
| Wan2.2-TI2V-5B DiT | W-half 四个 tap 全 ≈ 0（−0.003…+0.005） | 未见 | 未见 | 未见 | — |
| DINOv2 / DINOv3 / SigLIP2 | **未见**（WOD 上从未抽过） | 未见 | 未见 | 未见 | — |
| openpilot Cinque `temporal` | W-xfit **−0.141** [−0.247, −0.027]，−0.322，RFS +0.288；W-train **−0.294** [−0.424, −0.168]，RFS 7.449；3 seed 逐位相同 | W-train 7.637（3 seed 7.64 / 7.50 / 7.50），pre-onset +0.21 / +0.05 / +0.20；对原生 −0.29 / −0.46 / −0.42 | 未见 | 未见 | 见最后一行 |
| openpilot Lebowski `temporal` | W-xfit −0.109 [−0.232, +0.015]，−0.316，RFS +0.260；W-train **−0.318** [−0.449, −0.197]，RFS 7.522 | 7.734（3 seed 7.73 / 7.46 / 7.57）；对原生 −0.10 / −0.35 / −0.23 | 未见 | 未见 | 见最后一行 |
| openpilot small `temporal` | W-xfit −0.055 [−0.141, +0.033]，−0.201，RFS −0.060；W-train 未见 | 未见 | 未见 | 未见 | — |
| Alpamayo 1.5 `L18_mean` / `L27_last` | W-xfit −0.003 / **−0.163** [−0.254, −0.078]，−0.106 / −0.275，RFS −0.123 / −0.076；W-train 未见 | 未见 | 未见 | 未见 | — |
| YOLO26x-seg embedding | 未见（G0 进行中） | 未见 | — | — | E5 student：G0 进行中 |
| Qwen ⊕ openpilot 均匀 concat | W-xfit Cinque‖A −0.100，−0.248，RFS +0.144；Q1：WOD pre-onset 8 个 Δ 全跨零 | Q1(b) 对 best single RFS −0.14 / −0.15 | — | — | — |
| M-C 双流（Qwen ⊕ op，P5 v1 BA 配对训） | — | — | G1（三种 gate）进行中 | — | E1：RFS **−1.02** [−1.21, −0.82]（Cinque）/ **−1.52**（Leb），ADE 1–9 档 +1.39 / +2.07，直行激活 16.7% / 40.7%；E2 navtrain 编辑对训 −2.16，只 WOD 编辑对 +0.01 |

原生 plan（W-zs）：Cinque 8.005、Lebowski 7.886、small 7.640、Alpamayo nav 期望 7.857 / medoid-of-6 8.034、logged 8.131、cv 7.103。Hydra 打分头在 WOD：**未见**。
旁注（E 层判断的零样本读数，WOD pre-onset 1458 帧，纵向 meta-action 答对率）：Qwen 4B main 0.519、32B text 0.618，低于 ego 运动学外推 0.658（`p5-vlm-metaaction/final/accuracy.csv`）。

## 2. NAVSIM（N-*）

| 行 | PDMS [CI] | EPDMS [CI] | navhard | seed / 出处 |
|:--|:--|:--|--:|:--|
| cv / ctrv | 20.7 / 41.1 | 25.9 / 43.9 | 11.5 / 11.4 | standing §3 |
| `ridge ego` / `cls ego` | 62.7 / 68.4 [67.7, 69.1] | 64.2 / 67.8 | 13.3 / 13.6 | `cls ego` 3 seed 68.4 / 67.7 / 68.4 |
| Cinque `ridge_late` / `cls_late` | 73.5 / 77.9 [77.2, 78.5] | 73.9 / 77.4 | 16.8 / 19.8 | 3 seed：ridge 极差 0.13，cls 77.9 / 77.4 / 77.8 |
| Lebowski `ridge_late` / `cls_late` | 72.4 / 77.3 | 72.9 / 76.7 | 17.0 / 17.5 | 3 seed：cls 77.3 / 77.2 / 76.8 |
| **Cinque Hydra 式打分头（E6）** | **84.2 [83.7, 84.7]** | **82.6** | **25.7** | 单 seed；同候选集对 `cls_late` +6.3 [+5.7, +6.9] |
| Lebowski Hydra | 未见 | 未见 | 未见 | — |
| 原生 plan Cinque / Leb / small / Alpamayo | 52.1 / 50.9 / 47.4 / 44.3 | 46.2 / 45.5 / 42.5 / 43.2 | 9.3 / 10.2 / 10.2 / 10.8 | N-zs，第 37 条（2 Hz 协议读数） |
| M-C Δ 加到 Cinque / Leb `ridge_late`（E1） | −8.2 [−8.9, −7.5] / −11 | −13.0 [−13.7, −12.3] / −16 | +4–5（无 CI） | 走廊有行人 897 token：Cinque −6.0 / −11.2 |
| 文献 TransFuser / DiffusionDrive | 84.0 / 88.1 | 76.7 / 84.5 | 23.1 / 27.5 | devkit 不同，只作量级 |

未见：Qwen / V-JEPA / DINO / SigLIP / Alpamayo 特征 + head（box 上只有 navtest / navhard 的 Qwen，navtrain 没有）；openpilot small 特征 + head；gate 与 student（G1、G0 进行中）。

## 3. nuScenes

| 行 | U-zs：L2 均值 m / collision % | U-head：全部帧 1–9 档 Δ（pre-onset n=47） |
|:--|:--|:--|
| CV | 0.72 / 1.04 | — |
| `ridge ego`（base） | — | 0 = ADE 0.514 m |
| Qwen 4B `L18_mean` | — | −0.015 [−0.022, −0.008]（−0.035） |
| openpilot Cinque / Lebowski `temporal` | 原生 0.92 / 1.09，col 0.46 / 0.49 | **−0.091 / −0.087**（−0.077 / −0.063，跨零）；不带 command 时 pre-onset −0.402 / −0.411 |
| openpilot `vision` tap（Cinque / Leb） | — | −0.065 / −0.064（−0.099 / −0.103） |
| fusion Cinque ‖ A | — | −0.082（−0.099） |
| openpilot small | 原生 0.86 / 0.17 | 未见 |
| Alpamayo 1.5 nav | 0.97 / 0.40 | 未见（特征） |
| DINOv2 | 早期 planner v0（2 Hz 单帧、图像单独、另一套 ADE）：3.592 对 Qwen L19 3.250 | 未见（现行协议） |
| 文献 UniAD / VAD-Base / Ego-MLP | 0.66 / 0.37 / 0.35；col 0.62 / 0.33 / 0.37 | — |

未见：任何冻结特征 + head 的 **collision**；`cls_late`、Hydra、pair-Δ 在 nuScenes；V-JEPA / SigLIP2 在 nuScenes。

## 4. P5 配对考卷（逐帧定向翻转 %，[CI]；null = 样本外 null false-flip）

| 考生 | P5v0 合并 | BA 合并 / 行人 / cut-in | PDM 合并 / 行人 | null（BA） | 备注 |
|:--|:--|:--|:--|--:|:--|
| `ridge ego` | 6.5 [3.6, 10.3] | 6.0 / 7.6 / 3.8 | 7.9 / 11.7 | 0.5% | τ = 0 的标签伪影（E4 已说明），不是反应 |
| Qwen `ridge_late L18_last` | 0 | 0 / 0 / 0 | 0 / 0 | 5.1% | probe AUC 0.635（v0）/ 行人 0.578（v1 BA） |
| openpilot Cinque `ridge_late`（= prior） | 46.7 [31.6, 60.0] | 48.2 / 0.2 / 76.9 | 0.5 / 0.6 | 5.0% | 3 seed 随路线分折变 |
| openpilot Lebowski `ridge_late` | 41.6 | 45.1 / 2.7 / 70.6 | 2.6 / 0.3 | 5.1% | — |
| openpilot `vision` tap（Cinque） | 未见 | 48.7 / 4.9 / 74.9 | 0.1 / 0 | 5.0% | D0：行人 AUC 0.509 |
| M-C 配对双流 Cinque / Leb | —（v0 行人 0） | **66.3 / 43.3 [35.0, 50.7] / 80.0**；Leb 63.9 / 41.6 / 77.2 | 1.4 / 1.0；Leb 4.1 / 0.3 | 5.1% | 3 seed 行人 43.3 / 43.6 / 42.4；DOC 非反应误翻 11.7% |
| M-C 只 Qwen / 只 op（Cinque） | — | 60.5 / 42.1 / 71.4；55.8 / 6.9 / 85.1 | 0.5 / 0.6；1.1 / 0.6 | 4.9 / 5.0% | — |
| M-C hard-example / 均匀（Cinque） | — | 48.5 / 0.0 / 77.5；48.8 / 0.0 / 78.0 | 0.4 / 0.3；0.5 / 0.6 | 4.9 / 5.0% | 3 seed 行人 0–3.7% |
| Q9b Qwen 4×4 网格 + attention，配对 | 行人 0（v0） | 行人 46.1 / 52.2 / 45.3（3 seed） | ≤ 4.5 | — | hard 对照 3.4 / 2.5 / 0.0 |
| Qwen ⊕ op 均匀 concat（Q1-v1） | 合并 −20 / −25 pp vs best | 行人 0.2–1.2 | 贴地板 | — | 第 43 条修正 |
| E5 student A / B（op ⊕ YOLO，配对） | — | 行人 **52.7 [41.7, 62.1]** / 51.5；cut-in Δ −1.3 / +2.4 pp | 未见（不做） | 5.1 / 5.0% | 3 seed 51–57%；p95 约 30 ms；DOC 12.7 / 12.3% |
| TFv6 waypoint 2 s / target speed | 39.4 [28.5, 50.1] / 2.0 | 30.4 / 29.3 / 32.5；target 0 | 12.5 / 10.4；6.1 | 5.5% | 按对 73%（v0） |
| Q6 GT 规则门（any） | GT 42.9 / SAM 状态 29.2 | 50.8 / 80.0 / 34.2 | 4.2 / 10.7 | 0 | 特权几何，非传感器考生 |
| Hydra / `cls_late` / V-JEPA 2 / DINO / SigLIP2 / Alpamayo / op small | **未见** | 未见 | 未见 | — | 见第 7 节 |

## 5. I3（HUGSIM 3DGS 车辆配对，P5 v1 BA 上拟合后零样本；65 场景，单次 5 fold 平均）

| 考生 | 翻转 [CI] | 反方向 | 非反应误翻 | null |
|:--|:--|--:|--:|--:|
| `ridge ego` | 0.0（两侧 ego 相同，sanity） | 0 | 0 | 0 |
| Qwen `ridge_late L18_last` | 2.8 [1.3, 4.7] | 0.1 | 1.2 | 5.4 |
| op Cinque / Leb `ridge_late` | **70.0 [65.5, 74.5]** / 70.5 | 9.2 / 7.2 | 18.0 / 19.1 | 4.4 / 4.9 |
| M-C 配对双流 Cinque / Leb | 58.7 [54.2, 63.1] / 67.4 | 5.0 / 5.6 | 4.6 / 9.4 | 5.8 / 5.2 |
| M-C 只 Qwen / 只 op（Cinque） | 59.6 / 68.7 | 5.3 / 8.5 | 5.7 / 16.2 | 5.1 / 5.1 |
| M-C hard / 均匀（Cinque） | 64.6 / 69.8 | 13.2 / 9.8 | 18.2 / 18.2 | 4.5 / 4.8 |

双流 − prior（Cinque）：static −9.9 [−12.7, −7.2]、cutin −20.4 [−29.1, −12.4]、oncoming −10.2 pp。
未见：E5 student、Hydra、`cls_late`、V-JEPA / DINO / SigLIP / Alpamayo、G1 gate（已登记「在 I3 上先过一遍」，进行中）、G2（I3 上重训，等 G0 / G1）。

## 6. E4c latency、闭环

- **E4c**（BA 行人，A − null 地板 [CI]，[L, 10 s]）：M-C Cinque +0.38 [+0.22, +0.55]、Leb +0.48、TFv6 waypoint +0.40、op prior −0.20、Qwen −0.09、GT 规则门 +0.83；A₃（[L, 3 s]）M-C +0.25 [+0.07, +0.43]。PDM-Lite 集上所有传感器考生都在地板下。首翻比 BA onset 晚 0.8 s（行人）。未见：student、Hydra、V-JEPA 的曲线。
- **Bench2Drive 闭环（暂停，只记已有）**：Alpamayo 1.5 零样本 DS 60.8、RC 70.1、SR 2/5（n=5 smoke）；openpilot DS 2.7 作废（适配 bug）；TFv6 控制器 16 路线 × 2 seed DS 94.1（A）/ 95.0（B）。冻结特征 + head 的闭环：未见。
- **HUGSIM 闭环**：cv HD-Score 0.04 / 0.29（official / fixed 控制器）；openpilot 64 场景逐场景分在 `hugsim-exam/scored_op.csv`，文档未汇总（本报告不代算）。

## 7. 空格与补齐成本

Box 上已有（`ls` / `du` 核过）：
- WOD `processed/waymo_e2e/features/`：`op_{small,cinque,lebowski}_p3`、`alpamayo15_p3`（1.5 G）、`qwen_front3`、`qwenvid_p3`、`qwenvid2b_p3`、`qwen32b_front3_p3`、`vjepa2*_p3`、`wan22_dit_p3`；train：`op_{cinque,lebowski}_p3_trainval`、`vjepa2(g)_p3_trainval`、`qwenvid_train_t4`（39 G，train 每 4 帧取 1）。**没有** `op_small` / Alpamayo 的 trainval，没有任何 DINOv2 / SigLIP2。
- P5 v1：`processed/carla_p5v1_{ba,pdm}/features/c*/{L18_last,L18_mean,vis_mean,vit_mean}.npy`、`op_{cinque,lebowski}_vis/{temporal,vision,hidden}.npy`（1.6 G）；**没有** V-JEPA / DINO / SigLIP / Alpamayo / op small。
- I3：`processed/hugsim_pairs/features`（Qwen）、`op_{cinque,lebowski}`（temporal）、`op_streams_lead`；YOLO `processed/real_transfer/yolo/i3` 只有 `frames.parquet`（G0 检测在跑）。
- NAVSIM：`runs/navsim_zs/openpilot/navtrain/{cinque,lebowski}_temporal.npz`（468 M）、navtest 18 G；E6 逐 anchor 子分标签 `runs/elicitation/e6-prep/20260926-003758/`（477 M）；E6 只存了预测，没存 head 权重。Qwen 只有 `processed/navsim_qwen/{navtest,navhard_two_stage,e2*}`，没有 navtrain。
- nuScenes：`processed/nuscenes/v1.0-trainval/features/{dinov2,qwen,qwen_w800}`、`processed/drive_backbones/nusc_op/{cinque,lebowski}`、预测 `runs/nusc_backbones/ladder/20260925-144240/nusc_preds.npz`。
- 权重（HF cache）：dinov2-base、siglip2-so400m、vjepa2-vitl / vitg、Alpamayo-1.5、Qwen3-VL 2B/4B/8B/32B、Wan2.2；**DINOv3 没有**（第 12 条：访问申请被拒）。

| 空格 | 级别 | 依据 |
|:--|:--|:--|
| NAVSIM 训的 Hydra / `cls_late` / `ridge_late` 零样本上 P5 v1 BA 与 I3 | CPU 分钟级 + 胶水半天 | op Cinque `temporal` 两边都在盘；Hydra 需用 e6-prep 标签重拟合（E6 记约 1 min GPU）；ego 输入要从 `past.npy` 造 NAVSIM 的 32 维 |
| Lebowski Hydra（NAVSIM）+ Hydra 补 2 个 seed | CPU 分钟级 + devkit 约 30 min / 次 | 子分标签与特征无关、已缓存 |
| nuScenes 冻结 head 的 collision | CPU 分钟级 | `nusc_preds.npz` 在盘；collision 代码是第 39 条考试那套 |
| E4c 曲线加 E5 student | CPU 分钟级 | `e5-fit/.../preds_obs.npz` 在盘 |
| op small 在 P5 / I3 / W-train | GPU 小时级 | P5 / I3 / WOD trainval 都没有 small 的 `temporal` |
| V-JEPA 2 在 P5 v1 与 I3（probe + `ridge_late` + 替换 M-C 的 Qwen 流） | GPU 小时级（WOD 上 8.1 ms/帧、front 一路） | 权重在盘，P5 / I3 没有抽 |
| DINOv2 / SigLIP2 在 W-half、P5 | GPU 小时级 | 权重在盘，从未抽；W-half 可直接进 P3 阶梯代码 |
| Qwen 特征 + head 在 NAVSIM | GPU 小时级 | navtrain 没有 Qwen |
| Alpamayo `L27_last` W-train 复现 / 上 P5 | GPU 约 20 h（第 40 条估计） | trainval 未抽 |
| DINOv3 任意一格 | 要先拿权重 | 第 12 条 |
| Hydra 在 WOD / P5 的原生训练（非零样本） | 要造新数据 | WOD train 无 rater 分、CARLA / I3 无 PDM 子分标签 |
| 行人版 I3 | 要造新数据 | HUGSIM 只有 3DRealCar 车辆资产 |
| gate（G1）、student 上真实数据（G0）、I3 重训（G2） | 进行中 / 已登记 | real-data-transfer todo |

## 8. 对论文最值钱的 9 个空格

| # | 空格 | 回答的「hack 还是能力」问题 | 成本 |
|:--|:--|:--|:--|
| 1 | **NAVSIM 上训的 Hydra 打分头零样本上 P5 v1 BA 与 I3**，与同特征的 NAVSIM `cls_late` / `ridge_late` 并排 | 榜单最优 head（84.2 PDMS，训练标签就是评测 scorer）在配对考卷上保不保反应：翻转若 ≤ `ridge_late`，E6 的 +6.3 就是纯 metric 对准 | CPU 分钟级 + 半天胶水；限定：2 Hz 抽的 NAVSIM 特征对 5 Hz 抽的 P5 / I3 特征有域差，要先报 `ridge_late` 跨域的 sanity |
| 2 | **pair-Δ head 加 gate 后在 NAVSIM / WOD 的榜单分**（G1 主 arm）与 student 零样本（G0） | 能力 head 掉不掉榜单分。已有的无 gate 读数是大幅掉分（WOD RFS −1.02 / −1.52，NAVSIM PDMS −8.2 / −11）；有 gate 若「无害」，论文能写「能力 head 与榜单分不冲突」 | 进行中；不另派 |
| 3 | **同一 backbone 上 Hydra 的 3 seed + Lebowski 行** | 84.2 是论文里 R 层配方的主数，现在单 seed、λ 碰网格下沿 1e-5 | CPU 分钟级 + devkit 约 1–2 h |
| 4 | **V-JEPA 2 上 P5 v1 BA（probe + ridge + 替换 Qwen 流的 M-C）** | V-JEPA 在 WOD 顶档朝 rater 偏好挪（−0.31 到 −0.34，train split 活下来），是反应能力，还是多模态的 metric 偏好？P5 不看 rater | GPU 约 1 h 以内抽取 + CPU 分钟 |
| 5 | **DINOv2 / SigLIP2 进 W-half 与 P5** | 单帧自监督 / 语言对齐的对照行，AD-MLP 式论文的 backbone 表缺这两行；DINOv3 只能写「权重不可得」 | GPU 小时级 |
| 6 | **nuScenes 冻结 head 的 collision**（ego / Qwen / openpilot） | AD-MLP 那一问的原版：L2 降 18% 是 continuation 还是真少撞 | CPU 分钟级 |
| 7 | **op small 进 P5 / I3** | 30 M 的 small 在 WOD 只靠全部帧过判；它在 cut-in 上翻不翻，决定「车辆反应来自规模还是来自驾驶视频预训练」 | GPU 小时级 |
| 8 | **E4c 曲线加 E5 student** | 快通道 student 是看见后反应还是提前刹（DOC 非反应误翻 12.7%） | CPU 分钟级 |
| 9 | **Alpamayo `L27_last` 上 P5 或 W-train 复现** | VLA 深层 last-token 是全阶梯唯一两向过门槛的 arm，但只是 10 个次要 array 之一；上 P5 能区分「驾驶微调给了反应」还是多重比较 | GPU 约 20 h（W-train）；P5 更便宜但未估 |

四个点名问题的现状：
- **Hydra 上过 P5 / I3 吗**：没有。Hydra 只有 NAVSIM Cinque 一行（E6）。
- **同一 backbone 上 pair-Δ head 上过 NAVSIM / WOD 吗**：上过（无 gate，Cinque / Lebowski 的 M-C）。WOD RFS −1.02 [−1.21, −0.82] / −1.52 [−1.74, −1.30]，NAVSIM PDMS −8.2 / −11、EPDMS −13.0 / −16；编辑对训的也有害（WOD −2.16）。gate 版和 student 版在跑。
- **V-JEPA 2、DINOv3 上过 P5 吗**：都没有。V-JEPA 权重在盘上但 P5 没抽特征；DINOv3 没有权重。
- **ego-only 在 P5 / I3 上的读数**：P5 v0 6.5% [3.6, 10.3]，v1 BA 6.0%，PDM 7.9%（行人 11.7%），null 0.2–0.6%，全部来自 τ = 0 的标签伪影；I3 0.0%（两侧 ego 相同，按构造）。所以 ego-only 的「反应」读数实际是零。

## 9. 口径不一致

| # | 指标 / 对象 | 不一致在哪 | 影响 |
|:--|:--|:--|:--|
| 1 | RFS | 榜单口径 cluster mean vs 配对 Δ 用 frame mean（如 Cinque `ridge_late` 7.449 cluster / 7.458 frame）；n = 479 vs 478（W-xfit 少 1 帧） | 同一格两个数，引用时要带口径 |
| 2 | `cls ego` | 7.311（P0）vs 7.262（heads-train，同配方、k-means 行集略不同） | 差在 CI 半宽 1/4 内，但表里不能混用 |
| 3 | WOD pre-onset 的 n 与切分 | W-half 625 / 640（两方向分报）、W-xfit 1265、W-train 1291（V-JEPA 行 1249）、L0 旧口径 1510（全集 ADE，非第 22 条） | 跨协议的 Δ 不能并列比大小（第 24 条 d″ 的就地修正就是这个） |
| 4 | s_ego 分档 | 来自 `ridge ego` 自己的残差，循环；第 22 条要求每次引用带限定 | 所有「第 1–9 档」数 |
| 5 | EPDMS devkit | main@0a380a9，human 94.5，文献常引 90.3；PDMS 用 v1.1 | EPDMS 对文献只能写「同量级」 |
| 6 | navhard | two-stage EPDMS 无 CI；seed 极差 0.3–1.6 分，与行间差同量级 | navhard 行间比较不可靠 |
| 7 | seed 数 | M-C、E5、NAVSIM / WOD 薄 head、Q9b 已补到 3 seed；**E6 Hydra、I3 考试、E1 / E2、Alpamayo 零样本（seed 42）、W-xfit 全部 arm 是单 seed**；`ridge_*` 确定性；cls 头 GPU k-means / L-BFGS 同 seed 不可复现（WOD 差 0.00–0.05） | Hydra 与 `cls_late` 的比较一边单 seed、一边 3 seed |
| 8 | Lebowski `cls_late`「够到原生」 | seed 0 够到，seed 1 / 2 没够到；G 跨 −0.06 到 0.70 | 第 40 条已就地修正，中期 artifact 若引「0.4–0.7」要改 |
| 9 | nuScenes L2 | U-zs 是 UniAD/VAD 的 1/2/3 s 前各步平均（CV 0.72）；U-head 是 3 s ADE + s_ego 分档（`ridge ego` 0.514）；planner v0 又是另一套 ADE（3.25 m 量级） | 三套数不能放同一列；U-head 带 command 时 pre-onset 被泄露 |
| 10 | P5 翻转率 | 逐帧（BA 主判）/ 按对 / 可见门控 (a) / 曲线面积 A、A₃；PDM-Lite 集只报按对 | 同一考生 TFv6 39.4%（v0 逐帧）/ 73%（v0 按对）/ 30.4%（v1 BA 逐帧） |
| 11 | τ 与 null | P5 的 τ 由 P5 null（天气互换）定；I3 的 τ 由 24 个同车同道 null 定；`ridge ego` τ = 0 | I3 上 M-C 的 τ 从 0.53 抬到 1.52 m/s 是它掉分的机制 |
| 12 | expert / 标签 | P5 v0/v1 BA（BehaviorAgent，晚反应）、PDM-Lite（特权提前）、I3 规则 expert（匀速外推 + 只纵向）、WOD log / rater | 「翻转率」在四张卷上不是同一个量 |
| 13 | Qwen 的 `L18_last` | P5 / I3 / M-C 用的是 P3(d″) 视频 clip 的 `L18_last`；WOD arm A 是单帧 `L18_mean`；NAVSIM E1 用 0.5 s 间隔 clip | 「Qwen 行」跨考卷不是同一个特征 |
| 14 | openpilot 输入时钟 | WOD / nuScenes 原生 10 Hz 流；NAVSIM 2 Hz sample-and-hold；HUGSIM 闭环 4 Hz dilate（1.25×）；I3 5 Hz 无拉伸 | NAVSIM 原生 plan 分数是协议读数（第 37 条修正） |
| 15 | E2 的「1.44」 | 原写成特征位移比，实为 R1（head Δ 幅值比）；同统计量的特征位移是 1.21（G3 澄清） | G3a 对照要用 1.21 |

## 10. 给主会话的三条建议

1. **先派第 8 节 #1 + #3（CPU 分钟级，一个 agent 做完）**：Hydra 的 3 seed + Lebowski 行，再把 NAVSIM 训的 Hydra / `cls_late` / `ridge_late` 零样本上 P5 v1 BA 与 I3。这是整张矩阵里唯一直接回答「榜单最优 head 保不保反应」的格，特征和标签都在盘上；登记时把 2 Hz → 5 Hz 的特征域差写成先决 sanity（NAVSIM `ridge_late` 在 P5 上的 cut-in 翻转应接近 WOD / P5 训的 prior，否则整格不读）。
2. **backbone 行一次补齐，统一用 W-half + P5 v1 BA 两张卷**：V-JEPA 2、DINOv2、SigLIP2、op small 各抽一次，走现成的 P3 阶梯代码和 `p5_exam`，GPU 合计小时级。DINOv3 在论文里写「权重不可得」，不要再为它排期。Alpamayo W-train（约 20 h）排在这之后、由用户决定。
3. **论文主表之前先定口径**：RFS 全表统一 cluster mean（配对 Δ 另列 frame mean）、WOD 只用 W-train 作主协议（W-half / W-xfit 进附录）、P5 主判只用 BA 逐帧 + A₃ 并列、所有 head 行标 seed 数；把 ego-only 在 P5 的 6–8% 标成 τ = 0 伪影（否则审稿人会读成「不看路也能反应」）。G0 / G1 出数后，把第 1、2 节的「进行中」格按同一口径回填。
