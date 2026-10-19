# REDLINE-PROBE 预登记（2026-10-19 晚，写在任何探针输出、任何新 dump 读数之前）

## 问题

第 372 条：JTN-L18-s1 离 0.93 差 87 分，其中 85 分是「人站在红灯停止线前、没有前车」的静止窗（大考场 held，239 窗）。
同一模型在近前车后（477 窗 5 个零分）、无灯无车（38 窗 1 个）都站得住；红灯处 k = 0 的 plan 4 s 弧长反而更长。
要决定的是：这是**输入问题**（CAM_F0 2 Hz serve 路径上、冻结 token 里根本没有「红灯在线前 / 该站住」）还是**训练问题**
（token 里有，头没学会用）。规则（用户 / memory）：可见性只靠探针模型特征，不靠人看。机制沿用第 362 条 PAI-STOP-PROBE
（`experiments/route_follow/scripts/rf_stopprobe.py` 的 `boot_auc` / `fit_lin`；token 探针同构，步数按样本量改，见下）。

## 窗口与标签（全部来自已有标签：大考场 set + census 的灯标签，`native_gap_report.common` 的 2 228 个共有 held 窗）

类别按 `native_gap_report.cue` 的先后规则（lead <= 8 m，lead 8-30 m，红灯线前，未知灯，无灯无车），只取静止层（stratum = static）：

| 类 | 角色 | 窗 | log 数 | 定义 |
|---|---|--:|--:|---|
| P 红灯线前无前车 | 正例 | 239 | 18 | static，`red_stop & red_cls == C`，无前车 |
| N1 近前车 | 负例 | 477 | 27 | static，前车 <= 8 m |
| N2 无灯无车 | 负例 | 38 | 6 | static，无前车、无灯线 |
| N3 行驶中驶向绿灯 / 未知灯 | 负例 | 65 | 25 | 非 static，v0 >= 2 m/s，灯线在车头前 0-60 m，窗内从不为红（`red_frac == 0`），非 `red_stop` |
| G 绿灯起步（渲染里看不到绿） | 单独报告 | 40 | | `red_stop & red_cls == G`，无前车（d340：渲染的灯色不变） |
| U 未知灯静止 / F 远前车静止 | 单独报告（描述） | 75 / 49 | | static，cue 为「未知灯」/「lead 8-30 m」 |

主比较：P 对 N = N1 + N2 + N3（819 负例）。注意：有效样本是 log（P 只有 18 个 log，同一路口的多个 3 s 窗高度相关）。

## 输入（策略自己算的东西，sh30 2 Hz serve 路径，合成槽在内）

- 帧：JTN-L18-s1 原生（无 hold）大考场闭环、`SH30_DUMP` 全部会话（`native_gap.sh dump` 同一条命令），新 dump 放
  `$DATA_DIR/runs/defect_fix/redline_probe/dump/c<shard>/run`；N3 里已在 `native_gap/dump/JTN1` 的 13 窗直接复用。
- **主决策 k = 0**：force-GT，自车就在人的位置上，帧与 checkpoint 无关（第 372 条：成败主要在 k = 0 决定）。k = 0 只有 1 个关键帧：
  槽 7（t0）是真关键帧，槽 0-6 是它沿回推轨迹的 backwarp 合成。
- **次决策 k = 3**（4 个关键帧到齐）：槽 2（t0 - 1.0 s）与槽 7 是真关键帧，其余 6 个合成；这是 JTN1 自己的闭环状态（它若滚出，
  帧里带运动），只作逐槽对照，不进判读。
- token = Cinque 冻结视觉编码器 `view_39` 在 8 对（槽 - 0.2 s, 槽）上的输出 (8, 32, 512)，与 serve 同一函数（`native_gap.py cf` 的调用）。
- 读出（门：k = 0 前向重算的 plan 与存档原生读数 dec.parquet 的 k = 0 plan 一致，max 误差 <= 0.15 m，否则停下查）：
  RSG-L18-s0 与 JTN-L18-s1 各自在 k = 0 的前向（喂各自存档读数 k = 0 的 route memory），取 lead 头均值（72）+ lead_prob（3）与 plan。

## 特征集与探针

- **KIN** = 喂给 adapter 的 ego 行（20）。
- **LEAD-RSG / LEAD-JTN1** = KIN + 该模型 lead 均值 + lead_prob。判读用两者中合并 AUC 较高的那个（保守）。
- **ARC0-RSG / ARC0-JTN1** = 该模型 k = 0 plan 的 4 s 弧长，取负直接当分数（不拟合）。
- **TOK**（主）= KIN + 槽 7 的 32 x 512 token：逐 token LN → 线性到 16 维 → MLP 256（rf_stopprobe 同构）。偏差：rf_stopprobe 有 2.3 万行、
  3 epoch；这里约 800 行，改成 batch 64、30 epoch（约 400 步），3 个种子分数取平均；不调参。
- **TOK-lin** = KIN + 槽 7 token 均值（512），逻辑回归。
- **TOK-s j**（j = 0..7）= 同 TOK，换成槽 j；k = 0 与 k = 3 各一套（逐槽：真关键帧对合成槽）。
- 线性集：标准化逻辑回归，C = 0.1，balanced。
- 交叉拟合：5 折，按 sha1(log) 分折；每个窗由没见过它所在 log 的折模型打分；只用 P 与 N 训练，G / U / F 窗由同折模型打分。
- 95 % CI：log 聚类 bootstrap，B = 2000；配对差用同一组抽样。

## 判读（主比较 P 对 N，k = 0，TOK）

- **可读**：TOK AUC >= 0.85，且 TOK − LEAD >= +0.05、该差的 95 % CI 下界 > 0（LEAD = 两模型中较强者）。
  → token 里有「红灯在线前 / 该站住」，策略没用上：**训练问题**。
- **不可读**：TOK AUC < 0.85。→ **输入问题**（灯在 CAM_F0 2 Hz 路径的 token 里读不出：分辨率 / 裁切 / 编码器）。
- **只到头的水平**：TOK >= 0.85 但差值条款不过 → token 里的信息不超过 lead 头 + 运动学已有的（靠「没前车 / 没在走」分开，不是靠灯）；
  记为「灯不可读」，按输入问题处理，并看 N2 / N3 分项确认。

## 次要（全部预先列出，只描述，不改判读）

1. 分项：P 对 N1、N2、N3 各自的 AUC（每个集，含 TOK − LEAD）。N2（无灯无车静止）是最干净的「灯」对照：同为静止、无前车。
2. 同 log 配对 AUC：只数同一 log 内的（正, 负）对，排除城市 / 场景外观混杂。
3. 逐槽：k = 0 与 k = 3 的 TOK-s j AUC，真关键帧槽对合成槽。
4. 到线距离分箱：P 按 `red_dfront`（车头到线，m）分 < -1.5 / -1.5 至 -1 / -1 至 -0.5 / >= -0.5，各箱 P 对全部 N；
   N3 按灯线距离 0-20 / 20-40 / 40-60 m，各箱 N3 对全部 P。
5. G（40 窗）：在各折训练负例 10 % FPR 阈值下被判为「该站住」的比例（TOK / LEAD）；G 对 P 的 AUC（灯色不变时应接近 0.5）；
   U、F 同样报判正比例。
6. ARC0 两模型的 AUC（第 372 条的 0.86 / 0.89 是对零分的，这里是对类别）。

## 产出

`experiments/defect_fix/scripts/redline_probe.{py,sh}`；盒上 `$DATA_DIR/runs/defect_fix/redline_probe/`；
结果 `experiments/defect_fix/results/redline_probe.md`，decisions.md 下一个空号。所有 GPU 步骤走 `python -m jevdrive.cl submit`，priority 12，
一条 `--after` 链一次提交，不轮询。
