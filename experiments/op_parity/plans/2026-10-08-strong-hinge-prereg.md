# op_parity strong-hinge 预登记（2026-10-08，任何 SH 臂打分之前写定）

## 问题

第 166 条 / turn_oracle：给 plan 真值边界几何不减少 > 45 deg 切内角；本线唯一看到的杠杆是目标函数。pilot 规模（seed 0，3 000 步）hinge lambda 30、margin 0.5 m
相对 lambda 10 / margin 0.3：EPDMS +0.68 [+0.44, +0.94]，全 token DAC 失败 -0.58 pp，> 20 deg 切内角 -0.9 pp，oracle 臂和 shuffle 对照一样（与几何输入无关）。
本线不带任何 memory / oracle token，在全量 P2H10 配方上跑这一个设置，问：它在全量 / 两个 seed 上还成立吗，是真的路径改善还是像第 161 条那样只是 scorer 特定。

## 设置（唯一，不做任何其它超参搜索）

- 配方：`pp_train.py --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`，无 `--mem`；其余与基线完全相同。默认 P2H 路径不改。
- pilot 臂 `SHP-F-s0`：H0（`RH0-F-s0`，= P2H pilot）配方：`navsim/op-parity-s234`，3 000 步 x 64，warmup 100，seed 0；参照 `RH0-F-s0`（复用）。
  核对：turn_oracle 阶段 B 的 `TOSH` 臂用同一组 flags（`--hinge-lam 30 --hinge-margin 0.5`），仅多 `--mem sdf_shuf`。
- 全量臂 `SH30-F-s{0,1}`：P2H10 配方（12 个 navtrain shard，`navsim/op-parity-full`，10 000 步 x 128，warmup 300，seed 0 / 1，`--eval-every 1000`）。
- 基线（不重训）：`P2H10-F-s{0,1}`（lambda 10 / 0.3）、`RMH10-F-s{0,1}`（第 161 条 replay hinge），均为已存储的 bench 结果。

## 闸门（seed 0 pilot，先于全量；打分前写定）

SHP-F-s0 对 RH0-F-s0，navtest 12 146 token，按 log 聚类配对 bootstrap：
- 停：EPDMS 增益点估计 < +0.3，**或** 直行 token（逻辑 4 s 航向变化 < 5 deg，S5）EPDMS 下降超过 0.2（差值 < -0.2）。
- 任一条停 -> 不跑全量，报告 pilot 读数后结束。两条都过 -> 全量 2 seed。

## 读数（全量，navtest / navhard / HUGSIM 64）

1. navtest EPDMS 与子分数（NC / DAC / DDC / TLC / EP / TTC / LK），分层 all / S5 / T20 / T45；对 P2H10 和 RMH10 的配对差，log 聚类 bootstrap（B 4 000，jevdrive.stats），seed 均值。
2. DAC 失败率 T20 / T45；T45 切内角率、T45 转不过去率（four_dirs 口径，`turn_oracle.py replay` + `report`）；sharp R < 15 m 子集。
3. **raw plan 出界 vs 仅 replay 出界**（第 161 条的拆分，`rh.py proxy --name` 的几何回放 + `geomtab`，对**所有** token）。
4. 直行 token EPDMS 与 EP（margin 变大可能用更慢 / 更宽的行驶换 DAC）。
5. navhard（GIMM 帧，combined / stage 1 / stage 2）。HUGSIM 64 `spec_plan_smooth`（HD、碰撞、turn23 转弯路线子集、stuck），与 P2H10 / RMH10 同 preset。

## 什么算「真实的路径改善」（打分前写定）

同时满足：(a) navtest EPDMS 对 P2H10 的 CI 排除 0 且为正；(b) **raw plan 出界率下降**（全体或 T20，CI 排除 0），而不只是「仅 replay 出界」下降；(c) T45 切内角率点估计下降。
(a) 成立但 (b) 不成立 = 与第 161 条同类，scorer 特定。HUGSIM / navhard 作为跨 scorer 的检查，不作闸门。

## 预算与约束

约 3 卡时（pilot ~10 分钟，全量 2 x ~45 分钟并行）加 HUGSIM。所有 GPU / CPU 作业走池（`jevdrive.cl submit`），navtest / navhard / HUGSIM 读数只走 `jevdrive.bench`。
不改 `research/decisions.md`，不写 HTML。
