# op_parity / wod-launch 预登记：WP2 在 WOD 停车帧起步不足的归因与配方修正（2026-10-08，任何新臂打分、任何场景标签读出之前写定）

## 问题

第 163 / 164 条：WP2（Cinque 冻结视觉 + P2 输入对齐，WOD r2-train 训练）在 WOD val 479 个 rater 帧上 RFS 8.111，停车帧（v0 < 0.5 m/s，120 帧）7.598，对日志 −0.762 [−1.206, −0.337]、对 shipped −0.356 [−0.696, −0.057]（弱）；「停车且日志起步」95 帧 5 s 中位少走 5.6 m；停车 + 起步占 WP2 gap 的 36–40%。离线速度缩放拿不回（第 164 条 C）。

**问的是**：WP2 起步不足的来源在哪一层（场景类型 / ego 输入通道 / 帧协议 / 训练目标与损失 / 监督时域 / 底模），能否在同一个单模型配方里修掉且不丢行进帧的增益；顺带回答「红绿灯在 WOD 上丢不丢分」。

## 不重测的既有结论（只引用）

- 第 162 条：navtrain 训的 P2H 在 WOD 停车帧**前溜**，来源是 adapter bias 常数项；本线只把同一分解套在 WP2 上，不重跑 P2H。
- 第 163 / 164 条的存档预测（shipped、WP2-full-s0/s1、WP1-full-s0/s1、WP2-pilot-s0）与分层口径原样使用，不重跑。
- 第 111、119、141、90 / 100 / 110 条：openpilot 起步的横向环路增益、action 加速度路径起步失败、重投影引擎里 t0 加速度 −0.09 的起步停滞，都是闭环 / 其他榜的性质，这里不重测；本线只读 WOD 开环 5 s 计划。
- 第 89 条：冻结特征上灯态可读（AUC 0.91–0.98）、停止线距离不可读；Step 1 若判「不可修」，收尾探针沿用这个口径。
- 第 84 条：Qwen3-VL-4B 四选一读灯在 CARLA 帧上 93–94%；这里换到 WOD 真实帧，一致率要自己手查。

## Step 1 诊断（离线 + 推理）

口径：479 rater 帧，cluster 平均 RFS，按序列配对 bootstrap（B 4 000）；WP2 = 两 seed 逐帧平均。主分层：`stopped`（v0 < 0.5，120 帧）、`SL` = stopped 且日志 5 s ≥ 1 m（95 帧）、`SS` = stopped 且日志 < 1 m（25 帧）、`launch`（v0 < 2 且日志 5 s > 5 m，111 帧）、`moving`（v0 ≥ 0.5）。参照缺口 **R = RFS(log) − RFS(WP2) 在 stopped 上**（第 164 条 0.762）。距离量 d_t = 轨迹在 t 秒处到原点的位移。

每个候选的恢复率 r_X = [RFS(X) − RFS(WP2)] / R（stopped 上），CI 为 X − WP2 的配对 bootstrap。**carries**：r ≥ 0.5 且 CI 不含 0；**part**：0.25 ≤ r < 0.5 且 CI 不含 0；否则 **not**。同时报 SL 上 5 s 位移中位数与日志的差恢复了多少。

### (a) 场景类型（描述性 + 一个判读）

- 自动标注：Qwen3-VL-4B（box 上 HF 缓存，zero-shot，选项首 token 打分），输入 t0 的 FRONT 图（lead 一题给 t0 − 1.0 s 与 t0 两张）。四题：`light`（red_or_yellow_for_ego / green_for_ego / no_light_for_ego）、`stop_sign`（yes / no）、`lead`（no_lead / lead_stopped / lead_moving_away）、`cross`（行人或横穿车辆在 ego 路径上 / 无）。479 帧全标。
- 手查：stopped 帧里随机 40 帧（seed 0），我先看图独立写标签（落盘后才打开 VLM 标签），报逐题一致率与混淆。任一题一致率 < 80% 时，该题在表里标「不可靠」，stopped 帧上用手标（必要时把 120 帧全部手标）。
- 上下文归类（stopped 帧，优先级从高到低）：`red`（ego 红 / 黄灯）> `green`（ego 绿灯）> `stop_sign` > `lead_moving` > `lead_stopped` > `cross` > `open`。
- 读数：每类 n、日志起步比例、RFS top / log / WP2 / shipped / WP1、5 s 位移中位数、WP2 相对 top 丢的 points。全部帧上另报 `light` 三档 × （stopped / moving）的各臂 RFS：回答红绿灯丢不丢分。
- 判读：WP2 在 stopped 上丢的 points 里，「起步线索在 t0 可见」的类（green、lead_moving、open）占比 ≥ 60% → 欠起步主要是**可见线索下不走**；「线索不可见」的类（red、lead_stopped、stop_sign、cross）占比 ≥ 60% → 主要是**该不该提前走的预判**（日志 / rater 在 5 s 内放行而 t0 画面还在等）；其间为混合。每类帧数小，CI 只作弱读。

### (b) ego 输入通道（WP2 两 seed，harness 一次 warm-up 多个 bias，只跑 rater 479 帧）

bias 变体（改动加在全部帧上，按分层读）：`main`（复现存档）、`zero`（bias = 0）、`biasmean`（全帧平均 bias）、`biasresid`（减平均）、`cmd0`、`acc0`、`vx1` / `vx3`（只把 vx 加 1 / 3 m/s，位姿历史不动）、`cv1` / `cv3`（vx 加 1 / 3 m/s 且位姿历史换成与之一致的匀速直线历史）。

- **ego 通道 carries / part**：`zero` 的 r 达标（把输入关掉就恢复）。子归因：`biasmean` 对 `biasresid`（常数项 vs 随 ego 变化项，第 162 条同法）；`vx*` / `cv*` 给出 5 s 位移对速度输入的剂量（m per m/s）与哪一个子通道（速度标量 / 位姿历史）在起作用。
- 等价检查：`main` 对存档 WP2 预测的 plan xy 差（第 162 条为 0.010 m）。

### (c) 帧协议

- **协议替换**：同一批停车 rater 帧（落在 val token 缓存 slot ≥ 9 的那部分，约一半）上，WP2 用训练协议的 token 路径（图像对 (f − 2, f)，9 个 0.2 s slot；`wod_parity.py check` 的路径）对 harness（每帧喂两次）的 5 s 位移与 RFS。token 路径在这部分帧上的 r ≥ 0.5 → 帧协议 carries。
- **起步预判曲线**（描述性，val 全部偶数帧，对日志）：停车行按「日志起步时刻 τ」（未来第一个步速 > 0.5 m/s 的时刻）分档 0–0.5 / 0.5–1 / 1–2 / 2–3 / 3–5 s / 不起步，加上刚起步的行（v0 0.5–2 m/s），各臂（shipped、WP1、WP2）计划 3 s / 5 s 位移对日志的比值。WP2 只在 ego 已经动了之后才给出起步、而 shipped / WP1 更早 → 指向 (b)；三臂同样晚 → 指向底模 / 预判。

### (d) 训练目标与损失

- 基率：r2-train 停车行（v0 < 0.5）里日志 4 s 内起步（d4 ≥ 2 m）的比例 p_go，对 val rater 停车帧的同一比例（第 164 条：95 / 120 在 5 s 口径上起步）。
- 分布内读数（r2-dev 停车行，token 路径）：M = mean d4(WP2) / mean d4(log)；起步行（d4_log ≥ 2 m）上 R4 = median(d4_WP2 / d4_log) 与起步召回（d4_WP2 ≥ 0.5 d4_log 的比例）；不起步行上的误起步率（d4_WP2 ≥ 2 m）；WP2 d4 分布对日志双峰的直方图。imitation 损失是归一化残差的 Huber（δ = 1，4 s 处 σ_x = 1.1 m）：误差 > 1.1 m 时是 L1，取条件中位数，起步概率 < 0.5 的行整条判停。
- **分布内欠起步**：M < 0.8 或 R4 < 0.75 → yes；M ≥ 0.9 且 R4 ≥ 0.9 → no；其间 partial。
- 可读性：停车行上 logistic 探针（冻结 token 的 slot 平均，r2-train 训、r2-dev 读）预测「4 s 内起步」的 AUC，对 WP2 / shipped 自己的 d4 当分数的 AUC。探针 AUC < 0.75 → 冻结视觉不提供起步信号；探针 AUC − 计划 AUC ≥ 0.05 → 信号在而计划没用足。
- plan 头只有一条假设（均值 + 方差）；报停车行上 4 s 处纵向 σ（shipped 与 WP2）在起步 / 不起步行上的差别，作为「头是否知道不确定」的读数。

### (e) 定位到输入 / 微调 / 底模（存档预测，无新推理）

R 拆成 (log − shipped) + (shipped − WP1) + (WP1 − WP2)，各带 CI；SL / SS 上各臂 3 s、5 s 位移中位数与 RFS 的 3 s / 5 s 分量。WP1 − WP2 占 R ≥ 50% → 输入通道；log − shipped ≥ 50% → 底模本身。

### (f) 监督时域（读代码时发现，打分前登记）

`pp_train` 的 imitation 目标只有 0.5–4.0 s 的 8 个位姿，对应 plan 网格 i ≤ 21（4.31 s）；WOD 输出的 4.5 / 4.75 / 5.0 s 路点由 i = 22、23（4.73、5.17 s）插值，这两个点在 imitation 行上没有任何损失（只在 anchor 行上蒸馏回 shipped）。RFS 在 3 s 与 5 s 两处判。

- 离线干预 `x45`：WP2 的 4.25–5 s 路点换成它自己 3.5–4.0 s 段的外推：`xcv` 匀速（主）、`xca` 匀加速（次）。对 shipped、log 做同样替换作对照（算子本身的代价）。
- 描述：SL 帧上各臂逐步速度剖面（0.25 s 一步），看 WP2 在 4 s 之后有没有相对日志 / 自己前段的折点；d5 / d4 比值 WP2 对 log。
- `xcv`（或 `xca`）在 stopped 上 r 达标 → 监督时域 carries / part。

### 多重比较

479 帧、约 10 个 bias 臂 × 5 个分层 + 7 个场景类 × 5 臂：单个贴边 CI 只算弱；结论落在两 seed 同向、且 r 达标的臂上。

## Step 2 分支规则（Step 1 读完后不再改）

按 r（stopped 上）从大到小取**一个**候选；并列取便宜的。另一个 ≥ part 的候选只在第一个的 pilot 不过闸门时补一个 pilot。

| Step 1 结果 | Step 2 的单一配方改动（WOD r2-train，其余同 WP2） |
|:--|:--|
| (f) 监督时域 carries / part | **WL-h5**：imitation 目标加 4.5、5.0 s 两个位姿（WOD 未来到 5 s），`pp_train --fut5`，新 tab 另存，不动原缓存 |
| (b) ego 通道 carries / part（`zero` 恢复） | **WL-ego**：停车行（v0 < 0.5）上以 0.5 概率把速度与位姿历史输入置零（present 保留、命令保留），逼计划在停车时用画面；若子归因是常数项则改为训练末减去停车帧平均 bias 的常数修正 |
| (d) 分布内欠起步 yes / partial 且探针 AUC ≥ 0.75 | **WL-bal**：停车行内起步 / 不起步行按 1 : 1 采样质量重加权（权重只由 r2-train 统计得出，不看 val） |
| (c) 帧协议 carries | **WL-proto**：停车帧的服务协议改为训练协议（或训练里加入 harness 协议的 token） |
| (d) no（分布内已校准）而 val 仍短，探针 AUC ≥ 0.75 | 同 WL-bal（明确写成先验修正：rater 帧比 train 行更偏起步），pilot 闸门照常 |
| 都不达标，或探针 AUC < 0.75 且 (a) 判「线索不可见」≥ 60% | **停在 Step 1**：报告证据；收尾读数 = 冻结特征探针（灯态 / 4 s 内起步）在 val 停车帧上的 AUC |

**pilot**（wod_parity 的 pilot 规模：`cache/wod_pilot` 4 874 行，600 步 × 64，seed 0；对照 = 存档 WP2-pilot-s0）。闸门 **G-L**：stopped 上 d RFS(WL-pilot − WP2-pilot) ≥ +0.10，且 moving 上 ≥ −0.05。不过 → 停（或按上面补另一个候选的 pilot，至多一次）。Step 1 报告里先给出 WP2-pilot-s0 在 stopped 上的读数作参照；若 pilot 规模下 WP2-pilot 本身没有欠起步（对 shipped 在 stopped 上 ≥ −0.10），pilot 闸门改为 stopped 上 ≥ 0 且 moving ≥ −0.05，并写明闸门弱。

**全量**（过闸门后）：2 seed，10 000 步 × 128，同 WP2-full 的行序与 seed。读数（WOD val，第 155 / 163 条 harness 与 RFS port，按序列配对 bootstrap B 4 000，对 WP2-full）：
1. 主：stopped、SL、SS、launch 上 d RFS 与 5 s 位移；
2. 全部帧；moving（v0 ≥ 0.5）及 slow / mid / fast 三档：不许丢超过噪声（**不丢** = moving 上 d RFS 的 CI 含 0 或在 0 以上，且点估计 ≥ −0.05）；
3. 夜 / 昼；ADE@3s / @5s（1 437 帧）；对 shipped、对 log 的同表。
**判读**：stopped 上 WL − WP2 的 CI 下界 > 0 且 moving 不丢 → 修掉；stopped 点估计 > 0 但 CI 含 0 → 弱 / 未定；SS 上明显变差（前溜，第 162 条的失败型）单独报。

val 的 rater 标签只用于评测：不在 val 上训练，不在 val 上拟合参数（WL-bal 的权重、WL-ego 的概率、常数修正都只用 train）；不提交 test。

## 预算与实现

- Step 1：VLM 标注约 10 min 一卡；harness 2 个作业 × 约 10 min（479 目标，10 个 bias）；token 路径（val 流 + r2-dev，4 个模型）约 10 min；其余 CPU。合计 < 1 卡时。
- Step 2：pilot 训练约 2 min + 服务约 15 min；全量 2 × （训练 25–45 min + 服务 15–25 min）≈ 2–2.5 卡时。单个作业都 < 1 h；全量前先有 pilot（1 → 全量的分级）。
- 代码：`scripts/wod_launch.py`（label / bias / tok / report / figs）、`scripts/wod_launch_chain.sh`（一条自推进链，STATUS / DONE / ERROR / GATE_STOP 在 `$DATA_DIR/runs/op_parity/wod/launch/`）；新臂全部在开关后面（`pp_train` 新参数默认关），不改存档预测与其他线用的 P2 路径。GPU 作业全部走 pool。

## 限定（预先写明）

开环；479 帧，停车分层 120 帧；场景标签来自 zero-shot VLM + 40 帧手查；token 路径只覆盖偶数帧流（约一半 rater 帧）；(d) 的分布内读数对的是日志（训练目标），不是 rater 偏好；`x45` 外推在静止计划上等于不动（只检验已经在动的计划）。

## 补记（2026-10-08，Step 1 读完之后、任何 Step 2 臂训练之前写定；属对上文的偏离，报告里照此标注）

Step 1 读数（详见 results/wod_launch/）：R = 0.762 = (log − shipped) 0.406 [0.033, 0.761] + (shipped − WP1) −0.029 + (WP1 − WP2) 0.384 [0.087, 0.735]。(b) `zero` r = 0.47（**part**，+0.357 [+0.051, +0.714]，两 seed +0.377 / +0.336）；`biasmean`、`biasresid` 的 CI 都含 0，子归因不成立；(f) `xcv` r 0.09、(c) token 路径 r 0.004，**not**；(d) 分布内欠起步 yes（R4 0.63，但 shipped / WP1 同为 0.65 / 0.64），探针 AUC 0.76 低于计划自身的 0.85。按分支规则取 (b) → WL-ego；(d) → WL-bal 为备选。

1. **WL-ego 的形式改动**。登记的形式是「停车行上以 0.5 概率把速度与位姿历史输入置零」。Step 1 显示停车帧上这些输入本来就约为 0（停车帧 bias 的 rms 1.07 中，停车帧均值占 1.00，围绕均值的变化只有 0.38），登记的形式等于不改。同一意图（停车时不让只看 ego 的通道说话，让计划用画面）改为：**喂入速度 vx < 0.5 m/s 的行整条 ego 输入置零（present = 0，bias 恰为 0），训练（全部此类行，不是 0.5 概率）与服务同一规则**，`pp_train --stop-gate 0.5`，臂名 **WLG**。阈值 0.5 m/s 是预登记的停车分层边界，不调。机制依据（不依赖 val rater 标签）：adapter 只读 ego，停车时输入无信息，它在停车帧上只能给一个与画面无关的偏移；r2-dev 停车行上 WP2 的 4 s 位移直方图把 < 0.5 m 的份额从日志的 48%（shipped 51%）压到 31%，0.5–2 m 从 13% 升到 37%。
2. **只在服务端加同一规则的参照**（不训练，离线可算）：WP2 现有权重 + 停车帧 bias 置零 = `zero` 臂在门内帧、`main` 臂在门外帧的拼接。这个开关是看了 val 之后选的，所以按约束报**序列级 5 折 × 20 次的 out-of-fold**（每折在训练折上决定开不开门）与 in-sample 两个数；它只作参照，结论以训练出来的 WLG 为准。
3. **pilot 闸门**。Step 1：WP2-pilot-s0 在 stopped 上对 shipped +0.015，pilot 规模下没有要修的欠起步；而 WLG 在停车帧上按构造就是「输入关掉」的模型，对 WP2-pilot 的期望差约为 −0.015 加噪声。登记的弱闸门（stopped ≥ 0）会被这个按构造的零差随机判负，所以改为**只查伤害**：stopped 上 d RFS(WLG-pilot − WP2-pilot) ≥ −0.05 且 moving 上 ≥ −0.05。这是在看到参照数之后放宽的闸门，写明；全量的判读规则不变。
4. 备选 WL-bal 只在 WLG 的 pilot 判负时补一个 pilot（规则不变）。
5. 场景标注的偏离：VLM 的 `lead` 一题与手标一致率 52.5%（判不可靠），改用 shipped 的 lead 头（prob > 0.5 且距离 < 20 m，规则在 40 个手标帧上定，一致率 95% 为样本内）且不再分「起步 / 静止」；ego 自己的 stop sign 多数不在 FRONT 画面内，事后加一题 FRONT + FRONT_RIGHT 的 `stop_ctrl`（对手标推断 85.7%）。「线索可见」类相应改为 green + open，lead 两边都不算。
