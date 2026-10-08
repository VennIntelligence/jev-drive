# op_parity geo-e2e 预登记：几何 tokenizer 放进 adapter 训练端到端学，真值几何经 memory 通道能不能被读、读了值多少（2026-10-09，任何本线分数读数之前写定）

lane O1b，接 O1（[geo-oracle 预登记](2026-10-09-geo-oracle-prereg.md)，d197）。决定编号 200。
代码 `scripts/geo_e2e.py`（在线 tokenizer、收尾导出、报告）、`scripts/geo_e2e_chain.sh`，`pp_train.py` 加 `--mem-e2e` / `--mem-init`，`geo_oracle.py tok` 加 `--weights`。

## 问题

d197 的两者臂 +0.10 [−0.14, +0.34]，按登记判负，但带登记的限定「通道未被读」：冻结的、预训练过的几何 token 自己带几何（thin head 真 / 乱 ADE 1.27 / 4.57 m），
policy 却没用（对打乱对照 +0.18 [−0.06, +0.42]，屏蔽 memory 无显著变化，dev ADE 不变）。几何支路的上限因此没量到。
d197 写明会推翻它的证据：tokenizer 放进 adapter 训练端到端学，看两者臂是否与打乱对照分开。本线做这一件事，pilot 规模。

真值可行驶 SDF、真值 agent 框、以及 JP 臂的日志未来路径，在这里都是特权输入，只作 oracle 探针，不是方法，不进任何可报告的推理路径；本线不产生可报告的 driver。
不用 WA-JEPA 的任何权重或特征（WA-Cf 也不用）；d197 提到的第二条证据（把 WA-Cf token 里的专家路径信息去掉）按硬约束不做。

## 与 d197、d166 的差别

grep 过 `research/decisions.md`（memory、oracle、真值、tokenizer、端到端）。相关：d160、d165、d166、d170、d192、d197。

- **d166**：原始 0.5 m raster 切成 32 条直接进通道（只经通道自带的 LayerNorm + Linear），没有可训练的编码器；通道未被读。
- **d197**：卷积 tokenizer 先在别的 shard 上用 thin head（模仿 + hinge）训好再**冻结**出 bank；policy 的 loss 到不了 tokenizer；通道未被读。
- **本线**：同一个 tokenizer 结构、同一份 raster、同一通道、同一 pilot 配方与读数，差别只有一处：tokenizer 的权重在 adapter 训练里**由 policy 的 loss 更新**
  （plan 模仿项与 λ 30 hinge 的梯度经 policy、adapter 的 cross-attention、`side_in` 回到卷积）。token 不再是固定 bank，而是每步由当前 tokenizer 现算。
  另加一个 d166 / d197 都没有的东西：同一装置的阳性对照（JP），用来区分「几何对这个 policy 没用」与「这条通道 / 这种训练装置吸收不了 raster 信号」。
- 不重测：d197 的 S / A 单独臂（agent token 无任何可测作用、SDF token 只有登记规则之外的微弱痕迹）；d166 的原始 raster 编码；任何解冻 Cinque 的臂（d145）。

## 设计

### 共同配方（全部臂相同，= d197）

`pp_train --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`，`navsim/op-parity-s234`（25 415 train / 408 dev），3 000 步 × batch 64，warmup 100，d_frac 0.25；
memory 每行以 p = 0.25 屏蔽（独立 rng 流，行序、anchor 行、屏蔽行与 d197 各臂逐 seed 相同）。**seed 0 与 1 从一开始并行**。

- **基线 H0 复用** `GH0-F-s0` / `GH0-F-s1`（d197 训好的无 memory 同配方臂；s1 当时训了没打分，本线打分）。配方没有变，不重训。
- tokenizer：`geo_oracle.build_net` 的卷积部分（4 层 stride-2 卷积 + 1 层 3 × 3 + 1 × 1 到 512，池化到 8 × 4 = 32 token，约 1.1 M 参数），输入 raster 与 d197 的 B 相同
  （SDF 1 通道 + agent 距离 / vx / vy 3 通道，x −8..56 m、y ±24 m、0.5 m，只含 t0 的几何）。训练时前向 fp32。
- tokenizer 进优化器的方式：单独一个参数组，lr = adapter 的 `lr_new` 3e-4、同一 warmup + cosine、wd 0.01；梯度**单独** clip 到 1.0，policy 与 adapter 的 clip 不受它影响
  （与 H0、d197 各臂的更新规则保持一致）。tokenizer 在 `PModel` 之后构造，adapter 的初始化抽样与 d197 的 memory 臂相同。
- adapter 的输出层是零初始化，所以第 0 步 tokenizer 的梯度为 0，等 `out` 权重离开 0 后才有；这是通道的性质，不改。每 25 步记 tokenizer 的梯度范数。
- 测试：训练结束后同一个 job 用训好的 tokenizer 对 navtest 真值 raster 出 bank（`runs/op_parity/mem/ge_<tag>/lb_navtest.npy`，32 × 512 fp16，bench 原样读，`:noside` 屏蔽）。
  不写新 runner，`jevdrive.bench` 一行不改；`PModel` 认 `P2+ge_<tag>` 这个 arm 名。

### 臂（tag `<stem>-F-s{0,1}`）

| 臂 | stem | `--mem-e2e` | 含义 |
|:--|:--|:--|:--|
| H0 | `GH0` | 无 | 无 memory（复用） |
| **JB** | `GEB` | `b` | **判定臂**：tokenizer 随机初始化，与 adapter 联合训练，真值 SDF + agent |
| JX | `GEX` | `x` | 打乱对照：与 JB 完全相同的训练，但每行读到的是别的 log 的 raster（d197 `geo_x` 的同一固定置换，navtest 同样置换） |
| JW | `GEW` | `b` + `--mem-init` | 加臂 1：JB，但 tokenizer 从 d197 配方预训练的权重起步（同 recipe、seed 0 重训一次存权重；thin head 丢弃），再联合训练 |
| JP | `GEP` | `p` | 加臂 2：装置的阳性对照。同一 tokenizer、同一联合训练，输入换成 1 通道「到**日志未来路径**折线的距离场」（clip 6 m / 3，与 SDF 通道同一缩放；只有空间路径，没有时间与速度）。泄漏标签，不是几何臂 |
| GB / GX | `GOB` / `GOX` | 冻结 bank | 参照行：d197 的冻结 tokenizer 两者臂与打乱对照，s0 已有分数，s1 的 checkpoint 已训好、本线打分，使「冻结 vs 端到端」在同样 2 seed 上可比 |

**为什么是这两个加臂**（探过代码后选的，按每卡时的信息量）：

- 一次训练约 6.5 分钟、0.1 卡时，训练不是成本主项；能加的臂里要挑的是哪两个最能把 d197 留下的歧义拆开。
- **JP** 直接回答「装置能不能吸收」。d160 已证明通道能读 WA-Cf，但那是冻结的 512 维 encoder token，不是「raster → 随机初始化卷积 → 通道」这条链。
  若 JB 仍未被读，没有 JP 就分不清是几何没用还是这条链 3 000 步学不动。路径距离场与可行驶 SDF 是同一类输入（距离场，卷积要从中取出一条走廊），内容对模仿项的价值却是确定的。
  JP 被吸收而 JB 没有 → 装置没问题，是几何本身对这个 policy 没有可取的东西；JP 也没被吸收 → 装置问题，上限仍未量到。
  它同时给 d197 的推测（「通道读的是专家路径信息」）一个直接读数。比「更长训练」便宜：9 000 步要连基线、对照一起重训，是 3 × 3 倍的卡时，且 d166 已见过 9 000 步时对照与 oracle 同步上涨。
- **JW** 管初始化。随机初始化要在零初始化的输出层后面从稀疏的 hinge 梯度里学出卷积；预训练起步的 token 第 0 步就带几何（d197 已示），policy 的 loss 只需改造它。
  两个起点里只要有一个被读，就说明「可以被读」；JB 读而 JW 不读则说明 d197 的预训练 token 本身是个坑。lr 变体不做：与 adapter 同 lr 是默认且自然的选择，信息量低于这两个。
- 判定只看 JB。JW、JP 的结果照登记报告，不改判定；若 JW 满足「通道被读」而 JB 不满足，报告里写明并把 JW 对登记线的读数作为变体列出。

### 读数

navtest 全部 12 146 token，`python -m jevdrive.bench`（v2 devkit），每 token 取 2 seed 均值，按 log 聚类的配对 bootstrap（`jevdrive.stats.paired`，B 4 000，95% CI），同 d197。

- **主量**：EPDMS，JB − H0。
- 分桶 EPDMS（< 5°、5–20°、20–45°、> 45°）；> 20°（3 154）与 > 45°（1 517）DAC 失败率；> 45° 切内角率与转不过去率（four_dirs 回放，`turn_oracle.py replay`）；NC + TTC 失败率；各子分（NC、DAC、DDC、TLC、EP、TTC、LK、HC、EC）。全部臂对 H0，JB / JW 对 JX，JB / JW 对 GB。
- **通道被读**：JB − JX 的 EPDMS；JB`:noside` − JB 的 EPDMS（JW 同读）；dev ADE（训练 DONE 里的 on / 屏蔽 / 错配三个读数，及原始足迹出界率）。
- JP − H0 的 EPDMS 与 dev ADE。
- 参照行：GB − H0、GB − GX 的 2-seed 读数（d197 当时只有 seed 0）；d160 的 WA-Cf +0.95（不重跑）。
- 逐 seed 的 JB − H0、JB − JX 点估计（报告，不判）。

打 15 个分：GH0-s1、GOB-s1、GOX-s1；每个 seed 的 GEB、GEX、GEW、GEP、GEB`:noside`、GEW`:noside`。navhard 不跑（合成帧没有标签）。

## 判定线

**照抄 O1（任务书），不改**：

- 两者齐用的臂 navtest ≥ +0.7 且 CI 下界 > +0.3：几何足够，支路的问题收敛成「从前视图预测这些几何」。参照：WA-Cf 是 +0.95 [+0.47, +1.37]。
- < +0.3：缺的不是几何，记为负，轻量几何支路这条路要重估。
- 之间：记为部分，报告哪个子分、哪个转角桶拿到了、哪个没拿到。

「两者齐用的臂」= JB，「navtest」= EPDMS 的 JB − H0，2-seed 均值。

**另登记「通道被读」判据（比 d197 的「任一」更严，两条都要）**：JB − JX 的 EPDMS 差 CI 整体高于 0，**且** JB 屏蔽 memory − JB 的 EPDMS 变化 CI 整体低于 0。

**装置吸收判据（JP）**：JP − H0 的 EPDMS CI 整体高于 0，且 JP 的 dev ADE ≤ 0.8 × H0 的 dev ADE（2-seed 均值）。

**两个判据与判定线的组合，现在写定**：

| 线 | 通道被读 | JP 吸收 | 记为 |
|:--|:--|:--|:--|
| ≥ +0.7 且下界 > +0.3 | 是 | – | 几何足够；上限即此读数 |
| 部分 | 是 | – | 部分；上限即此读数，按桶 / 子分报 |
| < +0.3 | 是 | – | 负，任务书意义上的「缺的不是几何」；d197 的「待定」改判 |
| < +0.3 | 否 | 是 | 负：装置能吸收同类 raster 信号，真值几何却没被取用 → pilot 规模下几何对这个 policy 没有可取的增量；d197 的「上限未量到」收紧为此 |
| < +0.3 | 否 | 否 | 负但装置未证明能吸收：上限仍未量到，d197 的状态不变 |
| ≥ +0.3 | 否 | – | 不按线记：增益不能归于几何内容（对照没分开），照实报两项 |

第一次读数就是全部 2 seed；读数明确为负（上表后三行任一）即停，不加训、不放大。

## 步骤与启动

一条自推进的链（`geo_e2e_chain.sh`，tmux `jev:geo-e2e`，状态 `$DATA_DIR/runs/op_parity/geo_e2e/chain/{STATUS, DONE, ERROR}`），所有 job 走 pool，不选卡。

1. 启动前：单个 30 步、batch 64 的 smoke job（走 pool）核对代码路径并量真实 VRAM 峰值，按实测申报；smoke 不产生任何 navtest 分数。
2. 一次提交：预训练 tokenizer（`geo_oracle.py tok --kind b --weights`，带 smoke preflight，不写 bank、不动 d197 的 bank）+ 8 个训练（4 臂 × 2 seed，各带 3 步 preflight；JW 以 `--when-exists` 等预训练权重）。
3. 每个训练过 `pp_full_check.py train`；检查 navtest bank 存在。
4. navtest 15 个读数（bench）→ four_dirs 回放 → 报告。
5. 分级启动的检查单（看到任何 navtest 分数之前）：8 个训练 loss 有限、dev ADE ≤ 1.2 m、inputs-off drift ≤ 0.30 m（`pp_full_check`）；tokenizer 梯度范数在 warmup 后非零；bank 无 NaN；JX 的置换无一行来自本 log。

## 预算

GPU：预训练约 3 分钟 + 8 个训练 × 约 7 分钟 + 15 次 plan 导出 × 约 1–2 分钟 ≈ 1.5 卡时，上限 4 卡时。CPU：15 次 navtest 打分与 1 次回放，墙钟约 1 小时。

## 限定（开跑前已知）

- pilot 规模（25 k token、3 000 步）；2 seed；一种 tokenizer 结构、一种 lr。
- agent 只有 t0 的 16 个最近框与速度，不含未来占用（同 d197）。
- 基线已带 λ 30 hinge，同一份 SDF 已经作为训练标签用过；量到的是「几何作输入」在此之上的增量。
- JP 泄漏标签，只说明装置能否吸收一个 raster 信号以及通道对路径信息的反应，不是任何意义上的方法或上限。
- 上限针对这条 memory 通道（32 token、2 层 decoder、零初始化输出）与 pilot 配方。
- GB / GX 的 seed 1 在 d197 的闸门下未打分；本线把它作为参照行打分，不改 d197 按 seed 0 作出的登记判定，只在 d197 的状态行里补记。

## 开跑前的声明（2026-10-09，任何分数读数之前）

- smoke（pool，30 步 × batch 64，b / x / p 三种 kind 各一个）通过：loss 有限，dev 诊断与 navtest bank 导出都跑通，bank (12 146, 32, 512)，token RMS 0.48。
  30 步时 on / 屏蔽 / 错配的 dev ADE 相同（1.46 m），符合零初始化输出层的预期。smoke 不产生 navtest 分数；它的 bank 已删。
- VRAM 实测：smoke 含 dev eval 与 navtest 导出的峰值 19.9 GB（b）/ 18.2 GB（p）。d197 的冻结臂 30 步 11.9 GB、3 000 步 17.4 GB，全程会再涨，训练 job 按 26 GB 申报；RAM 40 GB、6 核（同 d197）。
