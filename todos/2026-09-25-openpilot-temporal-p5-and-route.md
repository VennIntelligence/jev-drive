# openpilot temporal 特征：P5 配对考卷上有没有长尾信息；route 进 head / 进 backbone 买不买得回静止起步与路口

状态: 待开始（预登记，写于任何抽取之前）
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

（待填）
