# 第 9 轮 NU-PAIBASE：PAI 混合底座走 nuPlan 候选 B2 的配方（预登记，10-22，任何训练与读数之前写定）

代码：[../scripts/nupb.sh](../scripts/nupb.sh)（整条链一次提交）、[../scripts/nupb.py](../scripts/nupb.py)（冒烟 / 汇总 / 判定）。
盒上 `$DATA_DIR/runs/round9/nu_paibase/`；结果 [../results/nu_paibase/](../results/nu_paibase/)（英文）。不碰 R9DL 的作业与文件；不做镜像、不提交。

## 1. 问题与假设

第 449 条的 PAI 混合底座（[pai_base](../results/pai_base/arena_base.md)）在 nuPlan 大考场 held 原生上，底座阶段 RB0-P70-s0 对 RB0-F-s0：走廊零分
353 对 392（−39），offroad 94 对 55（+39），AF-front 37 对 11（+26），均分 −0.012 [−0.034, +0.011]。走廊（静止窗滚出，第 349 / 393 条）是 nuPlan
最难的丢分；offroad 与 AF-front 是 L18 S 段历来修的（RB0-F → RSG held 零分 59 → 34，第 286 条）。
**假设**：PAI 混合底座走 nuPlan 候选的配方后保住走廊上的好处、S 段修掉 offroad / AF-front，胜过 B2。**反证要看**：L18 以前把底座学到的等待洗掉过（第 346 条）。

## 2. 臂（同一命令，只换 init）

B2 = `STANDFMXH-L18-s0 / s1`（第 389 / 391 条）。盒上 meta.json 的 argv（git 5858f899，clean）与 `runs/defect_fix/stand_b/cmd_FH(S).txt` 一致：
JTN 配方从 RB0-F-s0、池 on1_JTX（JTX00，on-k 70）、STAND W_S 0.0068、FREE-M move 0.001 frac 0.95、9 500 步、warm-up 500，s1 只差 `--seed 1`。

| 臂 | tag | 与 B2 同 seed 的命令差 |
|---|---|---|
| P70 s0 / s1 | `NPB70-L18-s0 / s1` | `--tag`、`--init RB0-F-s0` → `--init RB0-P70-s0` |
| P30 s0 / s1 | `NPB30-L18-s0 / s1` | `--tag`、`--init RB0-F-s0` → `--init RB0-P30-s0` |

底座只有 s0（第 449 条）；s1 臂与 B2-s1 一样是同一底座换 seed。四个训练一次 `cl fanout`，编译（与 B2 相同），HEAD 代码。
`nupb.sh check` 逐 token 比对：每臂对同 seed 的 B2 命令恰好 4 个新 token（tag 与 init 各两处的值），不过不训练。

## 3. 训练前的闸（不过 = 停线报告）

1. **恒等**：B2-s0 的命令（init = RB0-F-s0）60 步 eager，warm-up 50：B2 的训练代码 5858f899（`git worktree`，从旧工作树运行）对 HEAD，
   `df.py bitcmp --extra-ok`（每个标量、每个权重、dev ADE、on_hash_60）逐位相同 → `results/nu_paibase/identity.json`。
2. **冒烟**：P70-s0 的命令 60 步 eager（HEAD）：loss 标量全有限、`loss/on/lon_free_pos` 有（`stand_b.py smoke --finite-only`，对 1 的恒等 run）；
   meta argv 的 init = RB0-P70-s0；60 步后 dev ADE 与权重都和恒等 run（init RB0-F-s0）不同（错 init 对照：init 若没换，两者逐位相同）
   → `results/nu_paibase/smoke.json`。
两闸是队列里的 CPU 作业，训练 `--after` 它们（推测提交）。

## 4. 读数（每臂，一次提交，原生）

- 大考场 held 18 块原生（`jt.sh aread <tag> off c`，2 228 窗，第 355 / 371 条的选模型尺子）；
- held 460 @18（`rm_chain.sh read <tag> i 6`，6 块）。
- 汇总（盒上，facts 缓存）：`jevdrive.facts table` 大考场与 held 460，每臂对 B2-s0、B2-s1、JTN-L18-s1（存档读数）；零分分类（corridor / offroad /
  AF-front …）；静止层 = 静止窗（arena 窗 type）上的走廊零分，另报运动窗；窗口表 `arena_windows.csv.gz`（tag / id / score / type）。
- Mac 上：冻结配池 PCS（`pcs_local.py score`，第 382 条 board-mix，2 106 题，全集）各臂与 B2 s0 / s1、JTN-L18-s1；易题表（`easy_items.py subject`，
  第 387 / 390 条，ref JTN-L18-s1）各臂。底座阶段的对照（RB0-P70 对 RB0-F）照报。

## 5. 采用线（读之前写定）

一个份额（P70 或 P30）**过** = 两个 seed 都满足，每个 seed 对同号 B2：
- (a) 大考场 held 原生配对均值差（臂 − B2，共同窗，facts table 口径）≥ 0；
- (b) 走廊零分（held 窗，零分类里含 corridor 的全部：corridor、AF-front+corridor、offroad+corridor、AF-side+corridor）< B2；
- (c) held 460 零分 ≤ B2 + 2；
- (d) 冻结配池 PCS ≥ B2（同一脚本、同一冻结题目重新打 B2 的窗口表）。
两个份额都过：取两 seed 平均大考场差大者，并报另一个。都不过 = 不采用，写明哪条。对 JTN-s1、PAI、易题表照报不判。
**预期方向（读之前）**：若假设对，静止层走廊零分低于 B2（B2 s0 / s1 大考场零分 92 / 101）；若第 346 条重演，走廊零分回到 B2 水平、offroad / AF-front
仍高于 B2，均分 < B2。种子散布：B2 s0 / s1 大考场差 0.001，|差| < 0.005 记作「分不开」。

## 6. 成本、停线与分段

- 训练 4 × 约 0.4 卡时（编译，每卡 2 个）；读数每臂 18 + 6 块，约 5.7 作业小时 / 4 每卡 ≈ 1.4 卡时，4 臂 ≈ 5.7；汇总 CPU。合计约 7.5 卡时
  （停线 15）。墙钟：闸约 15 分钟，训练约 35 分钟，读数约 1.5 小时 → 约 2.5 小时。
- 分段：闸（60 步）先于训练；训练是同一配方的已知长度（B2 跑过两次），不另做 1 → 10 分段。任何闸失败、或累计超过 15 卡时：停线报告。
- 优先级：训练 20、读数 19（低于 PAI-BASE 的 21，不抢 R9DL 已排的作业）。
- 决定行只在核实的判定之后写。

## 补记 1（10-22 18:10，恒等闸作业失败后，任何训练与读数之前）

恒等闸的旧代码 run（5858f899 工作树）在读池时报错：`on1_JTX@warp` 已于 10-21 经用户批准换成零拷贝索引（`front.parts.json`，没有 `front.npy`），
5858f899 的 `pp_train.open_front` 打不开（找 `ticks.npy`）。这不是数值不一致，是旧代码读不了现在的盘。改法：旧工作树里只把 `open_front` 对
`front.parts.json` 的分支接到 HEAD 的 `jevdrive.tokens.open_front`（零拷贝检查 `cl_infra/results/zero-copy` 已证每行、每种 gather、60 步训练逐位相同），
B2 训练逻辑其余全是 5858f899。闸的含义不变（B2 的命令在旧代码与 HEAD 上 60 步逐位相同）。整条链用 `RETRY=1 nupb.sh launch` 重新提交（第一次的训练、
读数作业都已随闸失败，没有任何输出）。
