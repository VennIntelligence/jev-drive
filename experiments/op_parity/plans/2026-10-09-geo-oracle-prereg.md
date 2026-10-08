# op_parity geo-oracle 预登记：真值几何（可行驶 SDF + agent 占用）经 adapter memory 通道喂给 policy，「完美几何支路」的上限是多少（2026-10-09，任何本线分数读数之前写定）

任务书：[research/next-round/overnight.md](../../../research/next-round/overnight.md) 的 O1；判定线出自 [plan.md](../../../research/next-round/plan.md) 第 3 节，下文照抄，不改。
决定编号 197。代码 `scripts/geo_oracle.py`、`scripts/geo_oracle_chain.sh`，`pp_train.py` 只在 `MEM_KINDS` 里加四个 kind。

## 问题

plan.md 2.1 的方法是一条轻量前视支路，经 adapter memory 通道进 openpilot temporal policy，由评价器的后果标签（可行驶 SDF、agent 占用）监督。
在训练支路之前要知道它的上限：假如支路把这两样几何读得完全正确，policy 在 navtest 上能涨多少。
本线用 oracle 回答：memory token 直接由**真值**可行驶 SDF 与**真值** agent 占用生成，其余同 SH30 的 pilot 配方。
真值地图与真值 agent 在这里是特权输入，只作 oracle 探针，不是方法，不进任何可报告的推理路径；本线不产生可报告的 driver。不用 WA-JEPA 的任何权重或特征。

## 与既有条目的重叠，这次不同在哪

grep 过 `research/decisions.md`（memory、oracle、真值、占用、agent hinge）。相关：d147、d158、d160、d165、d166、d170、d192。

- **d160**（MW 臂）：WA-Cf 的 32 × 512 前视 token 走 `ParityAdapter` 的 side 通道（`pp_train --mem wa_cf`），pilot navtest +0.95 [+0.47, +1.37]，测试时屏蔽 −0.60。
  本线复用同一通道、同一训练子集（`navsim/op-parity-s234`，25 415 train / 408 dev）、同一步数与读数；d160 的 +0.95 只作参照行，WA-Cf 不进任何新 run。
  与 d160 不同的一处：基线配方是 SH30 的 pilot 配方（hinge λ 30 / margin 0.5 m，即 `SHP-F`），d160 的 H0 是 P2H（λ 10 / 0.3 m）。
- **d166 需要更正任务书里的一句话。** 任务书写「d166 把真值边界直接喂 plan 通路时没有被读，所以这次必须走 memory 通道」。读 d166 的预登记与结果后：
  d166 Step 1 的 OG 臂**已经**走的就是这条 memory 通道（`pp_train --mem sdf_gt`，同一个 side 通道），Stage B 的 OGh / OSh 一对还用了 λ 30 / 0.5 m，即本线的配方：
  OGh 88.58 对 OSh 88.48（seed 0），屏蔽 memory 后 88.32。所以「真值 SDF 走 memory 通道」不是没测过，测过的是**一种编码**：
  把 128 × 96 的 0.5 m raster 原样切成 32 条（每 token 384 个原始 SDF 值），只经通道自带的 LayerNorm + Linear 读入。该编码下通道几乎不被读（dev ADE 0.615 对 0.621，WA-Cf 是 0.421）。
  d166 自己把「卷积编码器」登记为范围外。Stage A 则说明几何本身可用：只给真值 SDF + ego 的小 decoder，> 20° DAC 失败 6.82%，与 WA-Cf 的 6.88% 持平。
- **本线与 d166 的差别**（不是重测同一机制）：
  1. **编码**：token 不再是原始 raster，而是一个几何 tokenizer 的输出（见下）。它对应 plan.md 里「支路输出 latent token 进 memory」的形态，也对应 WA-Cf 的来历（被规划目标监督过的 encoder 的 token）。d166 的原始 raster 编码不重跑。
  2. **agent 占用**：d166 只有地图。d158 的 agent hinge 在冻结特征上 loss 不降；这里把真值 agent 直接作为输入。
  3. **主读数**：navtest EPDMS 对无 memory 同配方臂（上限有多大）；d166 的主量是 > 45° 切内角闭合（视觉还是 plan head）。
  4. **2 seed**（过 seed-0 闸门时）；d166 是单 seed。
- 不重测：d166 的结论「几何不减切内角」；d154、d156、d159 的 head 侧项；d145 的解冻。

## 设计

### 编码：几何 tokenizer（本线的设计决定）

**token 口径与 WA-Cf 相同：每个 token 一份 32 × 512，fp16，4 × 8 = 32 格**，存成 `runs/op_parity/mem/<kind>/<data>.npy`，`pp_train --mem` 与 bench 原样读入，adapter 一行不改。

- **输入 raster**（ego 后轴系，x −8..56 m，y ±24 m，0.5 m，128 × 96，即 d148 hinge 标签的网格；只含当前时刻几何，不含 logged path 与路线）：
  - S：可行驶 SDF，`runs/op_probe/labels/{navtrain_all, navtest}.npz`，clip(±6 m) / 3。1 通道。
  - A：agent 占用，`runs/op_parity/agent_labels/{navtrain_all, navtest}.npz`（d158 的标签：vehicle / generic_object / pedestrian / bicycle，t0 最近的 K = 16 个）。3 通道：
    到最近 agent 框的有符号距离 clip(±6 m) / 3（无 agent 为 +2）；最近框的速度 vx / 10、vy / 10（ego 轴，绝对速度，由 t0 与 t0 + 0.5 s 两帧标注差分；距离 > 2 m 的格为 0）。
    **不含**未来占用：前视支路能给出的上限是当前位置与速度，不是 4 s 的未来轨迹；「真值未来占用」是另一个 oracle（预测），不在本线内。
- **tokenizer**：4 层 stride-2 卷积（32 / 64 / 128 / 256，GroupNorm + GELU）+ 1 层 3 × 3，1 × 1 投到 512，自适应平均池化到 8 × 4 → 32 个 token（x 每 8 m 一条、y 每 12 m 一格）。约 1.3 M 参数。
- **tokenizer 的训练**（一次，冻结后出 bank；对应 WA-Cf「在 navtrain 上被规划目标监督过」的来历）：token 接一个只在训练时用的 thin head（d147 / d160 的 decoder 形态：
  [32 × 512 token 展平, ego 20 维] → dropout 0.1 → 2 × 1024 MLP → 8 个位姿），损失 = 模仿（`rep.py decode` 的 Huber）+ 后果项：
  S 用 drivable 足迹 hinge（λ 30 / margin 0.5 m，SH30 的设置）；A 用 d158 的 agent hinge（λ 10 / margin 0.5 m）；B 两项都加。head 用完即弃，bank 里只有 token。
  6 000 步 × batch 256，AdamW lr 1e-3、wd 1e-2，warmup 300 + cosine，seed 0。
- **tokenizer 的训练行**：navtrain 的 s0、s1、s5–s11（约 7.7 万 token）中标签有效、有未来、且 log 不在 `navsim/op-parity-full-dev` 里的行。
  与 pilot 的 s2–s4 **token 不相交**（log 有重叠：shard 是按 token 划的）；与 dev **log 不相交**；与 navtest 无关。split 注册进 `jevdrive.data.splits`（`navsim/op-parity-geotok-train`）。
  这样 pilot 训练行上的 token 不是 tokenizer 自己拟合过的行，减小「训练行 token 过准、navtest 行 token 较差」的错配；剩余的错配用下面的检查量出来。
- **三个 tokenizer**：S（1 通道）、A（3 通道）、B（4 通道），各自训练，各出一套 bank。

### 臂（tag `<stem>-F-s{seed}`）

| 臂 | stem | `--mem` | 含义 |
|:--|:--|:--|:--|
| H0 | `GH0` | 无 | 无 memory 的同配方臂（SH30 pilot 配方，用当前代码重训；`SHP-F-s{0,1}` 是同配方的旧 run，只作复现核对） |
| GS | `GOS` | `geo_s` | 只 SDF |
| GA | `GOA` | `geo_a` | 只 agent 占用 |
| GB | `GOB` | `geo_b` | 两者（判定臂） |
| GX | `GOX` | `geo_x` | 打乱对照：`geo_b` 的 bank 在每个 data dir 内固定置换，每行换到别的 log（`turn_oracle.derange`，navtest 同样置换）；token 数、边缘分布、参数量相同，内容与当前帧错配 |

配方（全部臂相同）：`pp_train --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`，`navsim/op-parity-s234`（s2–s4），3 000 步 × batch 64，warmup 100，d_frac 0.25；
memory 臂每行以 p = 0.25 屏蔽 memory（独立 rng 流，所以各臂行序、anchor 行与 H0 相同）。seed 0，过闸门后补 seed 1。

### 读数

navtest 全部 12 146 token，`python -m jevdrive.bench`（v2 devkit）。每 token 取 seed 均值，按 log 聚类的配对 bootstrap（`jevdrive.stats.paired`，B 4 000，95% CI），与 d160 / d166 的表同一做法。

- **主量**：navtest EPDMS，各臂 − H0；判定臂是 GB。
- > 20°（T20，3 154）与 > 45°（T45，1 517）的 DAC 失败率；T45 的切内角率与转不过去率（four_dirs 插桩回放，`turn_oracle.py replay`，d166 的同一口径）；NC + TTC 失败率。
- 分桶 EPDMS（< 5°、5–20°、20–45°、> 45°）与各子分（NC、DAC、DDC、TLC、EP、TTC、LK、HC、EC）。
- **通道被读的证据**（三项都报）：(a) GB − GX 的 EPDMS 与 T20 DAC 失败差；(b) GB `:noside`（测试时屏蔽 memory）相对 GB 的变化，GS、GA 同读；(c) dev ADE 对 H0（d160：H0 0.62 m，MW 0.42 m；d166 的原始 raster：0.615 m）。
  判「通道被读」的规则：(a) 的 EPDMS 差 CI 不含 0，或 (b) 的 EPDMS 变化 CI 不含 0。
- 参照列（不重跑）：d160 的 MW − H0（P2H 配方，+0.95）；`SHP-F` 的旧读数。

## 判定线（照抄任务书，不改）

- 两者齐用的臂 navtest ≥ +0.7 且 CI 下界 > +0.3：几何足够，支路的问题收敛成「从前视图预测这些几何」。参照：WA-Cf 是 +0.95 [+0.47, +1.37]。
- < +0.3：缺的不是几何，记为负，轻量几何支路这条路要重估。
- 之间：记为部分，报告哪个子分、哪个转角桶拿到了、哪个没拿到。

「两者齐用的臂」= GB，「navtest」= EPDMS 的 GB − H0。过 seed-0 闸门时按 2-seed 均值判；闸门停时按 seed 0 判并标明单 seed。

**对判定的限定规则（现在写定，不改线）**：若 GB 落在「< +0.3」而通道按上面的规则**未被读**，则记为「负，但通道未被读」：它重复的是 d166 的状态，说明这种 tokenizer 编码经 adapter 也没学会，
不能读成「几何的上限 < +0.3」；报告里两句话都写。若通道被读而 < +0.3，才是任务书意义上的「缺的不是几何」。

## 步骤、闸门、分级启动

一条自推进的链（`geo_oracle_chain.sh`，tmux `jev:geo-oracle`，状态文件 `$DATA_DIR/runs/op_parity/geo_oracle/chain/{STATUS, DONE, ERROR, GATE_STOP}`），所有 GPU / CPU job 走 pool。

1. **tokenizer × 3 与 H0-s0 同时提交**（四个互不依赖的 job）。每个 tokenizer job 先以 `--steps 30` 作 preflight。
2. **tokenizer 检查（读分数之前，不过则停并报告，不是判定）**：在 dev（408 token，log 不相交）上，thin head 用真 token 的 ADE 比用打乱 token 的 ADE 低 ≥ 10%（token 确实带有与当前帧对应的规划信息）；
   bank 无 NaN、RMS 在三个 shard 与 navtest 间相差 < 25%；`geo_x` 没有一行来自自己的 log。另记录（不设门）：tokenizer 训练行 / pilot 训练行 / dev 三处的 head ADE 与足迹出界率，量 in-sample 错配。
3. **seed 0 四个 memory 臂同时提交**（各带 3 步 preflight），训练完过 `pp_full_check.py train`，navtest：5 个臂 + GS / GA / GB 的 `:noside`，回放。
4. **seed-0 闸门（明确阴性）**：GB − H0 的 EPDMS 点估计 < +0.3 **且** CI 上界 < +0.7 → 明确阴性，不跑 seed 1，按 seed 0 出报告。否则五个臂同时补 seed 1。
5. 报告。navhard 不跑（合成帧没有 SDF / agent 标签，d166 同）；HUGSIM、全量都不跑；plan.md 的 L1 及之后不启动。

## 预算

估计：tokenizer 3 × 约 10 分钟；训练 10 × 约 5–7 分钟；bench plan 导出 16 次 × 约 1–2 分钟：GPU 合计约 2 卡时，在 5 卡时以内。
CPU：16 次 navtest 打分（每次约 10 分钟、72 核，与 N1 / R1 共享 75 核配额）与 2 次回放，是墙钟主项，估 2–3 小时。

## 限定（开跑前已知）

- pilot 规模（25 k token、3 000 步）；tokenizer 单 seed、单一结构；agent 只取最近 16 个、只有当前时刻与速度。
- tokenizer 见过 navtrain 其余 shard 的日志未来（模仿项），所以 token 里除几何外还带「这种几何下人怎么开」的先验；这与 WA-Cf 同性质，也是支路训练后会有的形态，但它使本线量的是「几何 + 规划监督的 tokenizer」的上限，不是原始几何数值的上限。
- 上限针对这条 memory 通道与 pilot 配方；通道容量（32 token、2 层 decoder、零初始化的 bias）本身是上限的一部分。
- WA-Cf 的 +0.95 是在 P2H（λ 10）基线上量的，本线基线更强（λ 30），两个增量不是同一基线上的数。

## 开跑后的声明（2026-10-09，任何分数读数之前）

- **seed 1 的训练提前排队。** 协调方要求在会话暂停期间保持卡不空：五个臂的 seed-1 训练已用 `--when-exists <seed-0 训练的 DONE>`、低优先级排进 pool（job 名与 log dir 同链内的 `train`，链不会重复提交）。
  这只是训练，不读分数；seed-0 闸门不变：明确阴性时 seed 1 不打分、不进判定，按 seed 0 报告。多出的 GPU 约 0.5 卡时，仍在预算内。
- tokenizer 的 fit 行已注册：`navsim/op-parity-geotok-train@v1:905892337c03`，76 084 token。
