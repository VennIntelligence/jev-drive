# op_parity / mixed-domain 预登记：「停车 / 起步由画面决定，不由 adapter bias 的按域常数决定」这条原则在各榜自己的配方里是否成立（2026-10-08，任何新臂打分之前写定）

## 问题

第 162 / 167 条：navtrain 训的 P2H10 的 adapter bias 里有一个与输入无关的常数项，在 navtest / HUGSIM 上它就是起步能力，在 WOD 真实帧上变成停车前溜；serving 侧关掉它（停车门、减平均 bias）会毁掉 P2H10 在本榜的起步（navtest −1.50 / −6.71，HUGSIM 卡死 0 → 11.5）。wod-launch 线 Step 1：WP2 在 WOD 停车帧的缺口有一半来自只读 ego 的 adapter 在停车时给出与画面无关的偏移，修法是把规则训练进去（`pp_train --stop-gate 0.5`，WLG，今晚在训）。

**问的是**（按 coordinator 的改题，写于任何打分之前）：「停车 / 起步由画面决定」是不是一条在每个榜自己的配方里都有用的原则。不要求一个配方通吃所有榜，不为统一而牺牲某个榜的分数。

- **主读数（按榜形式）**：navtrain 配方里把停车规则训练进去（**P2HG** = P2H10 配方 + `--stop-gate 0.5`），navtest 与 HUGSIM 64 上对 P2H10。第 167 条说明 serving 侧的门不行是因为常数项在起步；这里读的是训练进去之后 plan 通路能否从画面学会起步。
- **WOD 一侧**：就是 wod-launch 线的 WLG，不重跑，只引用其结果。
- **MX（次读数，一个臂）**：navtrain + WOD r2-train 混合训练的同一个 adapter，检验「混合是否让常数项变得域中性」。不按「一个 driver 通吃」判。

## 不重测的既有结论

第 148 / 149 / 153 条（P2H10 的 navtest 88.67、HUGSIM 0.431）、第 155 / 162 / 163 / 164 条（P2H10 / WP2 在 WOD 的读数与分层）、第 167 条（serving 侧门与 biasdenav）、wod-launch 的 WLG：全部用存档结果，不重训不重跑。

## 臂

| 臂 | 训练 | 说明 |
|:--|:--|:--|
| P2H10 | 存档 P2H10-F-s0 / s1 | navtrain 档现役 driver，参照 |
| WP2 / WLG | 存档（WLG 由 wod-launch 线产出） | WOD 档参照，只引用 |
| shipped | 存档 | 地板 |
| **P2HG** | P2H10 配方原样（navtrain 全量 12 片，W 帧，hinge λ 10 / margin 0.3，10 000 步 × 128，warmup 300，2 seed）+ `--stop-gate 0.5` | 喂入速度 vx < 0.5 m/s 的行整条 ego 输入置零（present = 0，bias 恰为 0），训练与 serving 同一规则；阈值即 wod-launch 的固定值，不调 |
| **MX** | P2H 配方用在 navtrain 全量 + WOD r2-train 的并集上，10 000 步 × 256（每批 128 navtrain + 128 WOD，精确各半），其余超参同 P2H10 | 见下「MX 的域处理」 |

**P2HG 的 serving**（与训练同一规则）：navtest `P2HG-F-s*:sg`（`jevdrive.bench` 的 `sg` 开关，阈值 0.5）；HUGSIM `--opts '{"parity": {"stop_gate": 0.5}}'`（读的是喂给 adapter 的模型时钟速度，与第 167 条相同）；WOD `wod_launch.py gbias`。

**MX 的域处理**（逐项写明）：
- 帧协议与输入映射各用各的：navtrain 行 = W 协议 warp 帧（`navtrain_full.s*of12@warp`）+ pp_prep 的 ego；WOD 行 = `cache/wod_r2` 的真实帧 + `pp_wod.wod_ego`。切分：navtrain 行按 token（`navsim/op-parity-full`），WOD 行按序列（`wod/r2`）。
- hinge 只作用在有可行驶标签的行（navtrain）；WOD 行无 hinge。锚行（0.25）两域都有，各自蒸馏回 shipped 在本域帧上的输出。蒸馏的归一化标准差在两域训练行合并后算。
- **slot 数（8 + 零 对 9 个真实）**：navtrain 缓存只有 8 个 slot，最老 slot 为零且无效；WP2 用 9 个真实 slot。若各用各的，「最老 slot 是否为零」就是 plan 通路可以直接读到的域标记，混合训练可以借它学出按域的起步先验而看起来像「按画面」。所以 **MX 主臂把 WOD 行的最老 slot 也置零（`--wod-slots 8`，两域都是 8 + 零）**，WOD 行的教师用同样 8 个 slot 重算（`mixed_domain.py teacher8`）。serving 不变：navtest 8 + 零；WOD harness 与 HUGSIM 9 个真实 slot（与 P2H10 现在的 serving 条件完全相同）。对照 **MX9**（WOD 行保留 9 个真实 slot）只在 pilot 跑，用来量这个选择的代价，不进全量。
- 每个域每步看到的样本数与单域配方相同（128），所以 batch 是 256；这是相对 P2H10 / WP2 的偏离（总 batch 翻倍，lr 不变）。

**第二个 MX 臂**不预设；只有 pilot 给出明确动机且预算够才加，加之前在本文件补记。

## 小步与闸门

- **P2HG pilot**：`P2HG-P-s0` = H0 pilot 配方（s2–s4 三片，3 000 步 × 64，hinge 10 / 0.3，seed 0）+ `--stop-gate 0.5`，对照存档 `RH0-F-s0`（同配方无门）。读 navtest 全部 12 146 token（`:sg` serving）。**闸门 G-NG**：EPDMS 差 ≥ −0.50 且 v0 < 0.5 分层差 ≥ −5.0（serving 侧门是 −1.50 / −19.4）→ 进全量；否则停，报告诊断（分层、EP / EC、停车 token 的计划位移分布）。
- **MX pilot**（wod_parity 的 pilot 规模）：`MX-P-s0` = `lb_navtrain + lb_h1train`（W 帧，`navsim/op-parity-pilot`）+ `wod_pilot`，600 步 × 128（64 + 64），warmup 100，无 hinge（两个单域 pilot 都无 hinge），`--wod-slots 8`；`MX9-P-s0` 同上但 9 slot。单域 pilot 对照：`WP2-pilot-s0`（WOD）、`P2-W-s0`（navtrain，W 帧）；两者在对方榜上的读数（`P2-W-s0` 上 WOD harness、`WP2-pilot-s0` 上 navtest）现跑，只是推理。**闸门 G-MX**：清晰负结果 = MX-P 在 WOD val RFS 与 navtest EPDMS 上都低于两个单域 pilot（四个差的 CI 上界全部 < 0）→ 停并写诊断；否则进全量。MX9-P 只报不判。
- 预算紧时的优先级：P2HG 先，MX 后。

## 读数（全量）

1. **navtest**（12 146 token，W 帧，按 log 分簇配对 bootstrap；口径同 stop_gate_xboard）：EPDMS 与 NC / DAC / DDC / TLC / EP / TTC / LK / HC / EC；分层 v0 < 0.5、起步（v0 < 2 且日志 4 s > 5 m）、stay（v0 < 0.5 且日志 4 s ≤ 5 m）、行进（v0 ≥ 0.5）、转弯 > 20°。P2HG − P2H10、MX − P2H10、各自 − shipped（P0，W 帧）。
2. **HUGSIM 64**（`spec_plan_smooth`，每 seed 一次，按场景配对）：HD、起步停滞、卡死、打转、fg / bg 碰撞、完成；P2HG 另从 `zs_steps.jsonl` 读门的触发场景数与是否锁死（第 167 条的 latch）。对存档 P2H10 `rr1 / rr2` 均值。
3. **WOD val**（第 155 / 163 条 harness，479 rater 帧 cluster 平均 RFS、1 437 帧 ADE，按序列配对 bootstrap B 4 000）：分层 standstill（v0 < 0.5）、SS、SL、launch、moving、turn intent。MX 对 WP2 / shipped / P2H10；P2HG 对 P2H10 / shipped（navtrain driver 加了门之后在 WOD 上还溜不溜，顺带读）；WLG 的行直接引用。
4. **bias 分解**（第 162 条方法）：各臂 adapter bias 在 WOD val 帧与 navtest token 上的 rms、常数项（帧平均）rms、残差 rms，停车帧上同样三项；两域常数项之差的 rms 与余弦；MX 另在 WOD 上跑 serving 变体 zero / biasmean / biasresid。
5. navhard（G 帧）便宜则给 P2HG 与 MX 各读一次，只报。

## 判读（写在结果之前）

- 每榜对该榜现役 driver，标签沿用 stop-gate-xboard 的规则。navtest（全集 EPDMS 差的 CI）：helps = 下界 > 0；hurts = 上界 < 0；harmless = CI 整体在 [−0.10, +0.10]；其余 inconclusive。HUGSIM（HD 差的 CI）：harmless = CI 整体在 [−0.03, +0.03]；helps = 下界 > 0 且起步停滞 + 卡死不增；hurts = 上界 < 0，或卡死数比基线多 ≥ 2（臂均值）。WOD：RFS 差的 CI 下界 > 0 为 helps，上界 < 0 为 hurts，否则 inconclusive（479 帧的 CI 半宽约 0.17）。
- **原则在某榜成立** = 该榜自己的配方加上这条原则后 helps 或不 hurts（harmless / inconclusive 且点估计不低于 −0.10 EPDMS / −0.03 HD / −0.10 RFS）。navtrain 档看 P2HG 对 P2H10（navtest 与 HUGSIM 两个都要过）；WOD 档看 WLG 对 WP2（引用）。一个榜成立另一个不成立就照实写。
- **P2HG 学会从画面起步** 的机制读数：v0 < 0.5 与起步分层对 P2H10 的差（serving 侧门 −19.4 / −8.9）、EP、EC；HUGSIM 卡死数（serving 侧门 11.5）。分层只描述。
- **MX 的常数项是否域中性**：(a) MX 的 WOD 停车帧常数项与 navtest 停车帧常数项之差的 rms 相对常数项本身 rms 的比例（同一个 adapter 只读 ego，停车时两域输入几乎相同，这个差按构造应很小：这一项回答的是「adapter 已无法按域区分」，不等于行为域中性）；(b) 行为：MX 在 WOD SS 帧的 5 s 位移对 P2H10（前溜）与 WP2，MX 在 navtest v0 < 0.5 / 起步分层对 P2H10；(c) WOD 上 biasmean / biasresid / zero 变体相对 MX 的 RFS 差。MX 的全表对 P2H10、WP2、shipped 都给，带 CI，不设「通吃」判据。

## 预算

P2HG：pilot 训练约 7 min + navtest（CPU）；全量 2 × 约 22 min（独占时 7.4 it/s）+ navtest + WOD harness 2 × 约 15 min，约 1.3 卡时。MX：teacher8 约 3 min；pilot 训练 2 × 约 3 min + harness 3 × 约 15 min；全量 2 × 约 45 min（batch 256，约 45 GB）+ harness 2 × 约 20 min，约 2.8 卡时。合计约 4.5 卡时，另加 HUGSIM 4 次 64 场景（各约 0.6 卡时）。单个作业都 < 1 h；全量之前都有 pilot。全部 GPU 作业走 pool。

## 限定（预先写明）

WOD 开环；HUGSIM 一个 preset、每场景一次；阈值 0.5 m/s 固定；P2HG 的 pilot 只读 navtest；MX 的 batch 是单域配方的两倍；MX 在 WOD / HUGSIM 的 serving 用 9 个真实 slot 而训练是 8 + 零（P2H10 同此）；画面本身带域信息（warp 合成帧对真实帧），「按画面」与「按域外观」在这套实验里分不开，HUGSIM 是唯一不在两个训练域里的读数。val 的 rater 标签只用于评测；不提交 WOD test。
