# op_parity off-track rows 预登记：SH30 配方加约 10% 静态扰动的重投影行，navhard 涨多少、navtest 掉不掉（2026-10-09，任何本线分数读数之前写定）

任务书：[research/next-round/overnight.md](../../../research/next-round/overnight.md) 的 R1；判定线出自 [plan.md](../../../research/next-round/plan.md) 第 3 节，下文照抄，不改。
决定编号 198。代码 `scripts/ot_rows.py`（prep / probe / train / gate / report）、`scripts/ot_rows_chain.sh`。`pp_train.py` 不改（O1 同时在改它），`Store` / `PModel` / `Losses` 原样 import。

## 问题

navhard stage 2 与 AlpaSim 闭环的共同点是 ego 不在日志轨迹上（d189：AlpaSim 第 3 次决策 online plan 离日志 1.73 m，离线 0.58 m）。SH30 的训练行全部在日志位姿上。
问：不做任何 on-policy rollout，只把约 10% 的训练行换成「同一真实帧从横向偏 ±0.5 m、带小幅 yaw 的位姿看出去」的行，navhard two-stage 涨多少，navtest 掉不掉。

## 与既有条目的重叠，这次不同在哪

grep 过 `research/decisions.md`（离轨、扰动、重投影、DAgger、on-policy、漂移、恢复、mixed）。近邻 d132、d141、d143、d146；另有 d170（SH30 基线）、d174（多域混训不赚）。没有条目测过「navtrain 上的静态离轨行 + SH30 配方 + navhard 读数」。

| 条目 | 它做的 | 本线 |
|---|---|---|
| d132 重投影 DAgger | WOD；状态来自 shipped 在引擎里**闭环走出**的访问分布；标签是平滑的**恢复路径**；微调 shipped（无 adapter），action 头一起动（曲率增益符号翻转）；读数在引擎内 + HUGSIM 11 | navtrain；状态是**脚本化静态偏移**（d132 的对照臂 st1 的状态分布，不是 dg1 的）；目标是**日志未来在扰动位姿下的重表达**，不构造恢复路径；SH30 配方（冻结 Cinque token、P2 adapter、hinge 30 / 0.5 m），action 头仍蒸馏到 shipped；读数是 navhard / navtest |
| d141 G0 | 定义并验证了引擎（plane 重投影，深度不更好）；失败域 |dy| 1.5 m / 15° | 引擎原样用（`jevdrive.op_interp.warp_frame`）；扰动幅度 |dy| ≤ 0.5 m（历史起点 ≤ 1.0 m）、|dψ| ≤ 2°，远在有效域内；不重测引擎 |
| d143 G1 | 精确自车引擎 on-policy（S3：横纵闭环、8 s、3 轮 DAgger、时间同步源帧）；HUGSIM 起步停滞 18 → 48，机制是「画面在动才走」；横向 2 s 的 S2 闭环最好但也不过线 | 无 rollout、**无纵向扰动**：每个历史帧的源帧就是同一时刻的真实帧，画面里的运动与 ego 速度都是日志的；只取 t0 速度 > 3 m/s 的行，起步 / 静止状态不进扰动行 |
| d146 P2 起步 on-policy | P2 + WOD 引擎行 on-policy 3 轮；navhard +2.58 [−0.62, +5.97] n.s.、navtest −0.29；HUGSIM-12 起步停滞 1 → 9；off-policy 对照 C 也 1 → 3 | 无 WOD 行、无 on-policy；d146 的 navhard 点估计是本线期望效应的参照（那次不显著）；HUGSIM 起步停滞进护栏 |

不重测：引擎选型（d141）、on-policy（d143 / d146）、hinge 强度（d170 / d172）。

## 设计

### 离轨行怎么造（`ot_rows.py prep`，唯一一组参数，不搜索）

- **取行**：navtrain 全量 shard 里 t0 速度 > 3 m/s 且有日志未来的 token，每个 token 一份扰动（预计约占 shard 的四分之三）。
- **扰动**：dy ~ U(−0.5, +0.5) m，dψ ~ U(−2°, +2°)，独立；按 shard 固定 seed。历史 10 帧（4 个 2 Hz key + 6 个 lattice 帧）的偏移用现成的脚本漂移
  `op_adapt_h.drift`（layer-3 O pair 的同一个函数）：航向误差从 −1.6 s 的 0 线性到 t0 的 dψ，横向误差按 v·ψ 积分、在 t0 等于 dy；dψ = 0 时就是恒定横向偏移。
  若 −1.5 s 处横向偏移超过 1.0 m 则重抽（高速行的 dψ 因此偏小）。没有纵向偏移。
- **帧**：每一帧都是「最近的真实 key 帧」经 plane 引擎一次重投影到 `日志位姿(t) ∘ (0, y(t), ψ(t))`；零偏移时与 pp_prep 的 W 协议逐帧相同（链里有零偏移等值闸门）。之后同 pp_prep：冻结 Cinque encoder 出 8 个 slot 的 token。
- **ego 输入**：4 个历史位姿重表达到扰动后的 t0 系；速度、加速度、command 用日志值（漂移下车沿自身航向走，体坐标速度一阶不变）。
- **目标**：日志未来 8 个位姿重表达到扰动后的 t0 系（任务书规定）；不做平滑恢复路径。
- **hinge**：照加（λ 30 / margin 0.5 m）。SDF raster 仍是该 token 日志系的那张；plan 先变回日志系再采样，0.1 s 插值的起点是扰动位姿。
- **teacher**：shipped Cinque 在扰动 token 上的输出，非 plan 头照旧蒸馏；离轨行不作 anchor 行。

### 训练（`ot_rows.py train`）

- 每个 batch 固定 round(0.1 × batch) 行来自离轨行（64 → 6 行，9.4%；128 → 13 行，10.2%），其余行与 anchor 比例（0.25）同 SH30。步数、lr、warmup、split 全同参照配方。
- 训练循环是 ap2_train 那样的独立循环；`--ot-mass 0` 时行流与 pp_train 相同，作为 trainer check 臂。
- **pilot**（`navsim/op-parity-s234`，3 000 × 64，warmup 100，seed 0）：`OTP-F-s0`（10% 离轨行）；`OTC-F-s0`（0%，同一循环）。
- **全量**（12 shard，`navsim/op-parity-full`，10 000 × 128，warmup 300，seed 0 / 1）：`OT30-F-s{0,1}`。参照 `SH30-F-s{0,1}`（已存，不重训）。

### 训练前的两个闸门（不是分数读数）

1. 零偏移等值：64 行零偏移的 token 对已存 W token 的相对平均差 < 1e-3，fut / ego 差 < 1e-4。
2. 符号 / 几何：P0 与 SHP-F-s0 在离轨行上，plan 横向位置对 yaw 偏移的最小二乘响应（4 s）≥ 0.3。对横向偏移的响应只报告、不设线（d132：shipped 的偏移保留比 0.93，响应本来就小）。

## 步骤（照抄任务书）

**步骤**：先 pilot 规模 1 seed 读一次 navhard；点估计 ≥ +1.0 再放全量 2 seed。

- pilot 的参照：pilot 参照配方（SH30 配方、无离轨行）现有的三个 checkpoint 的逐 group 均值：`SHP-F-s0`、`SHP-F-s1`（pp_train）、`OTC-F-s0`（本线循环，0%）。
  用三个而不是一个，是因为 navhard 相近臂之间的可分辨效应约 ±1.5–2（plan.md 第 1 节），单个参照 seed 的噪声与 +1.0 的线同量级。对每个参照单独的差也报告。
- 点估计 < +1.0：记为 pilot 负，停，不放大；报告 pilot 的 navhard（分阶段）、navtest、离线读数、HUGSIM 护栏读数。

## 读数（照抄任务书，下面是口径）

**读数**：navhard two-stage（G 帧口径，SH30 是 33.67，S1 75.90 / S2 44.68）分阶段报；navtest；AP2 输入标准下的离线读数（`experiments/alpasim/scripts/ap2_offline.py` 的口径，m = 1..4）。

- navhard：`jevdrive.bench run --model <tag>@gimm --bench navhard`，combined / stage 1 / stage 2，225 组，按 stage-1 log 聚类的配对 bootstrap（B 10 000），seed 均值。
- navtest：`jevdrive.bench run --model <tag> --bench navtest`，EPDMS 与子分；同样的配对 bootstrap（按 log）。
- 离线读数：`ap2_offline.py plans / report`，NAVSIM 标准训练的 checkpoint 按 AlpaSim 输入标准（冷启动规则 backwarp，即 SH30 现在的 serving 方式）在 navtest 上读 m = 1..4 的 ADE，
  与 2 000 token 子集的 EPDMS（无 EC，non-reactive，`score-poses`）；全量对 `SH30-F-s0`，pilot 对 `SHP-F-s0`（pilot 的 EPDMS 只打 m = 1 与 m = 4，ADE 报全部 m）。只是读数，不设线。
- 附带诊断（不设线）：留出 token 的离轨行上，各 checkpoint 的 ADE 与对横向 / yaw 偏移的响应系数（`ot_rows.py probe`）。

## 判定线（照抄任务书）

**判定线**：navhard 对 SH30 ≥ +2.0 且 CI 下界 > 0；navtest ≥ −0.2。两条都过记为进配方候选。
HUGSIM 的起步停滞与 HD ≥ −0.03 是进配方前的护栏，今晚卡有空再跑，没空留到明天，但没跑之前不写「进配方」。

- 全量判定用 `OT30-F-s{0,1}` 的 seed 均值对 `SH30-F-s{0,1}` 的 seed 均值。任一条不过：不进配方（plan.md 的「判负之后」）。
- HUGSIM 护栏口径：64 场景、`spec_plan_smooth`（SH30 的口径），每场景一次。起步停滞数（bench 的 `launch_stall`）不高于参照；HD 差的点估计 ≥ −0.03。
  pilot checkpoint 上的护栏读数（`OTP-F-s0` 对 `SHP-F-s0`）在卡空时先跑，是早期读数；算数的护栏是判定所针对的那组 checkpoint（跑了全量就是 `OT30` 对 `SH30`）。
- 本线不训 AP2 变体，不向 AlpaSim 提交。

## 会让结论作废或要披露的事

- 零偏移闸门或符号闸门不过：修到过为止，修之前不训练；修法写进结果。
- 取行后离轨行不足全 shard 的一半，或 hinge 标签覆盖率在离轨行上明显低于普通行：披露。
- OTC-F-s0 与 SHP-F-s0 在 navtest 上的差超出 seed 噪声（|差| > 0.3）：独立循环不等价，披露，并把 pilot 参照改述为各参照分别的差。
- 看到任何 OT 臂分数之后不改线、不改扰动参数、不改参照。

## 预算与运行

- 估时：prep 每 shard 约 5–10 分钟（CPU 重投影为主，20 核一个 job，smoke 的 64 行先量 tokens/s 再放 3 个 shard）；pilot 两臂各约 10 分钟；navhard / navtest 打分各约 5–10 分钟 CPU；
  HUGSIM 64 每个 checkpoint 约 0.3 卡时。pilot 合计约 1 卡时。全量：9 个 shard 的 prep 约 0.5 小时，两个 seed 各约 30–45 分钟，加读数与 HUGSIM，约 2.5–4 卡时。单个 job 不超过 1 小时，墙钟不超过 3 小时。
- 训练读的是缓存好的 token（离轨行在 prep 里一次算完），训练期没有在线 warp，dataloader 与 SH30 相同；全量前仍用 20 步 smoke 看 it/s。
- 所有 GPU / CPU job 走 pool（`jevdrive.cl submit`，owner `op_parity-r1`），读数走 `jevdrive.bench`；链在 tmux `jev:ot-rows`，状态文件 `$DATA_DIR/runs/op_parity/ot_rows/chain-run/{STATUS, DONE, ERROR, GATE_STOP}`。

## 修订 1（2026-10-08 23:5x box 时间，任何 OT 臂分数读数之前）

只改训练前的符号 / 几何闸门（第 2 条），判定线、pilot 线、扰动参数、取行、参照都不动。

- **原来**：P0 与 SHP-F-s0 对 yaw 偏移的响应（4 s）≥ 0.3；对横向偏移的响应不设线。
- **现在**：两个模型对 yaw 偏移和对横向偏移的响应（4 s）都 ≥ 0.1。
- **为什么**：64 行 smoke 的读数是 P0 横向 0.03 / 0.15 / 0.52、yaw −0.13 / 0.03 / 0.35（1 / 2 / 4 s），SHP-F-s0 横向 0.03 / 0.12 / 0.32、yaw −0.10 / 0.01 / 0.29。
  两个偏移的响应都为正并随时域增大，符号错误会在 4 s 给出负值；Mac 上对 `warp_map` 的数值核对也是对的（dy = +0.5 m 时路面点右移，dψ = +2° 时源像素左移 32 px = 2° × 910 px/rad）。
  0.3 是写预登记时凭感觉定的幅度线，量的是「现有模型本来修多少」，不是闸门要查的符号与几何；0.29 对 0.3 的差在 64 行的抽样误差内（斜率标准误约 0.05）。
  闸门改成查符号（两个分量都明显为正）。1 s 处 yaw 响应略负与漂移历史一致：历史里航向在向 dψ 一侧转，模型短时域延续这个转向，而目标要求转回来。
  这也说明现有模型在离轨状态下几乎不修（与 d132 的偏移保留比 0.93 一致），是本线要量的东西，不是数据的错。
- 这组读数只来自训练前的探针（P0 与已有的 SHP-F-s0 在 64 个离轨行上的 plan），不是任何 OT 臂的分数。
