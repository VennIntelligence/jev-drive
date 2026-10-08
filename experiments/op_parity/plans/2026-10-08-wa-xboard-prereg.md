# op_parity / wa-xboard 预登记：NAVSIM 专家 WA-JEPA 在 WOD-E2E val 上零样本（2026-10-08，任何 RFS 读出之前写定并提交）

## 问题

论文对照的是「轻适配的工业驾驶模型（openpilot Cinque）」对「benchmark 专家」。正向已有：shipped Cinque 没见过 WOD，RFS 8.005；WP2 / WLG（WOD train 上训的 adapter）8.111 / 8.187（第 34、163、169 条）。缺反向对照：NAVSIM 专家 WA-JEPA（navtest 91.71、navhard 35.41、HUGSIM 64 HD 0.451，第 138 条等，均已测）在它不是为之建的板上零样本多少分。本线补 WOD-E2E val。

**不重测**：shipped / WP2 / WLG / P2H10 的存档 WOD 预测（`processed/wod_zeroshot/preds/op_cinque*`）、日志与 cv 行、RFS 端口（bit-exact）、分层标签，全部原样使用。WA-JEPA 的 navtest / navhard / HUGSIM 读数不重跑。

## 硬约束

- WA-JEPA 原样跑：释放权重（`models/wajepa/model_state_dict.pt`），官方 repo 的 `eval/navsim_agent.py` 特征构造经 `experiments/top10/lib/top10_t2/wajepa_run.py`（fp32 = 其 NAVSIM 路径，4 步 flow，batch 1），不微调、不改模型代码；只写 WOD → NAVSIM 请求的包装。
- 只在 val 上读；不提交、不上传。变体全部报告，**不按 val 分数挑**。

## 目标集与打分口径（同 `results/wod_parity.md`）

479 个 rater 帧（478 条序列）算 RFS（cluster 均值），ADE@3s / @5s 在 rater + extra 共 1 437 帧上对日志未来；序列配对 bootstrap，B 4 000，seed 0（`wod_launch_report.Ctx` 的 K 矩阵，与 wod_parity 一致）。对照臂：shipped（`op_cinque`）、WP2（WP2-full-s0/s1 逐帧均值）、WLG（WLG-full-s0/s1）、P2H10（P2H10-F-s0/s1）、log（日志未来作预测）、cv 行。

## 输入映射（一律写在这里；数据只用 WOD front3，slim 分片无后视）

| 项 | 映射 | 开放? |
|---|---|---|
| 帧率 / 历史 | NAVSIM 2 Hz × 4 帧：目标帧 f 与 f−5 / f−10 / f−15（10 Hz 真实帧，间隔 0.5 s，时间序最旧在前） | 固定（WOD 帧是真实时间，无需 warp） |
| 位姿历史 | `pp_wod.wod_ego` 的 4 个位姿（past_states 步 9 / 11 / 13 / 15 = −1.5 / −1.0 / −0.5 / 0 s，弦向航向，t0 为原点、航向 0），即 `hist (4, 3)` 本身就是相对当前位姿的形式 | 固定（第 155 条同） |
| ego 状态 | `[cmd one-hot(4), vx, vy, ax, ay]`：vx = 位置推出的速度（`past_kinematics.v`），vy = 0，ax / ay = past_states 最后一步给出的加速度 | 固定（第 162 条：此映射不是 P2H 的差距来源） |
| 指令 | intent GO_LEFT / GO_STRAIGHT / GO_RIGHT → NAVSIM one-hot 序 [left, straight, right, unknown] 的 0 / 1 / 2；UNKNOWN → 3（val 不出现） | 固定（一一对应） |
| 后视 B0 | 黑图（−1），所有变体 | 固定（HUGSIM 档同法，第 138 条） |
| 图像 | 见下三个变体 | **开放，三个都跑** |
| 输出 | 8 × 0.5 s 路点（0.5 … 4.0 s，后轴系，x 前 y 左，与 WOD 同系）→ 20 × 0.25 s | 见「4 s → 5 s」 |

### 图像变体（三个，全部报告）

NAVSIM 相机：K = 1545 px、1920×1080、带 Brown-Conrady 畸变（D = −0.356, 0.173, −0.002, 0, −0.052）；L0 / R0 偏航约 ±55°，F0 约 0°（`sensor2lidar`，nuPlan 读出）；WA-JEPA 把图以 cv2 INTER_AREA 缩到 512×256。

- **V1 `reproj`**（先验主变体，因为最接近它训练时看到的几何，不由分数选）：对每个 NAVSIM 相机 L0 / F0 / R0 构造虚拟相机（NAVSIM 的 K、畸变、`sensor2ego` 旋转，纯旋转无视差），每个像素的射线落到 WOD FRONT / FRONT_LEFT / FRONT_RIGHT 中光轴最近且看得见的那个（`camgeom.choose_sources` + `waymo_project`），双线性取样；先 2× 超采样（1024×512）再 INTER_AREA 到 512×256 以匹配其缩放。看不到的像素为黑。
- **V2 `crop`**（朴素包装）：WOD FRONT → F0，FRONT_LEFT → L0，FRONT_RIGHT → R0，居中裁成 2:1，INTER_AREA 缩到 512×256；不做视场 / 畸变对齐。
- **V3 `front`**：V1 的 F0，L0 / R0 / B0 黑图。回答侧视图这件事（映射最不确定的部分）占多少。

### 控制

- **IMG0**（ego-only）：四路图全黑，其余输入（指令、位姿历史、ego 状态）与 V1 相同。
- **STATE0**：V1 图像，指令 = unknown（one-hot 第 4 位）、vx = vy = ax = ay = 0、位姿历史全 0。
- 两者合起来读：图像贡献（V1 − IMG0）、状态与指令贡献（V1 − STATE0）。

### 4 s → 5 s

WA-JEPA 只出 4.0 s。5 s 判定处用：先把 (0, 0) 与 8 个路点线性插值到 0.25 s 格，4.25 … 5.0 s 由 3.5 → 4.0 s 段的速度**匀速外推**（`wod_launch` 的 `xcv`，同一算子）。灵敏度：`xca`（匀加速外推，速度不为负）。**同口径行**：shipped、WP2、WLG、log 的预测同样截到 4.0 s 后做同一外推（`-x4` 行），把外推算子本身的代价单独列出，主表仍用它们的原生 5 s 预测。ADE@3s 不受外推影响，是最干净的列。

## 比较与读法

WA-JEPA 各变体对 shipped、WP2、WLG、P2H10、log 的 ΔRFS / ΔADE@3s / ΔADE@5s，序列配对 bootstrap。分层（同 `wod_launch_report.Ctx.st` 并加转向）：stopped（v0 < 0.5）/ moving；速度档 0.5–5、5–12、≥ 12；intent 直行 / 转向（左 + 右）；夜 / 昼（luma 标签）；10 个 scenario cluster（val 无 Spotlight）逐簇 RFS。stopped 与 launch 子集沿用第 169 条定义。

判读规则（读分前定）：ΔRFS 的 CI 下界 > 0 记「胜」，上界 < 0 记「负」，其余「不可分」；|Δ| < 0.10 另注「接近」。每个变体各自判，**不合并成单一数**；若三个变体的方向或显著性不一致，结论写「受输入映射影响」，并给范围。IMG0 与 V1 的差：CI 不含 0 记图像有贡献。

## 流程与闸门

1. **G0（映射管线）**：虚拟相机的内参 / 畸变 / 旋转往返测试：把渲染器的目标相机设成 WOD FRONT 自己的内参，渲染结果与原图的平均绝对灰度差 < 2；同一请求两次推理输出逐位相同；请求路径对 `wajepa_run` 的等价沿用 `--check`（该 runner 已验证）。
2. **分级发射**：1 个目标（看图 + 看输出范围）→ 10 个目标（拼图人看 V1 / V2 / V3 的三路图，输出落在合理范围）→ 全部 1 437 目标 × 3 变体 + 2 控制。每级写 STATUS，错误写 ERROR，完成写 DONE；一个链式脚本；GPU 任务只走 pool。
3. 预测落盘后才打 RFS。

## 预算

约 2 卡时、半天墙钟。输入映射在预算内跑不出原样的 WA-JEPA 即停并报告阻塞点。

## 只读估算（不跑）：WA-JEPA 在 Bench2Drive 闭环

写进 `results/wa_xboard.md` 末节：每次决策的单样本延迟（本线实测）、B2D 的 220 路线 × 平均步数、需要补的东西（B2D 相机 → NAVSIM 4 相机的映射、指令来源、2 Hz 历史缓存、位姿 / 加速度来源、控制器），对照 `docs/closed-loop-runbook.md` 的卡时。

## 不验证的东西（预先说明）

单 split、开环、无视差纯旋转重投影、无后视、侧视覆盖不全、noise seed 为 repo 默认（另在 V1 上做一个 noise seed 的灵敏度，不并入主表）。
