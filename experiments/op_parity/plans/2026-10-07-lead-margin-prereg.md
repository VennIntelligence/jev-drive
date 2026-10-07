# op_parity：plan 速度剖面的前车静止余量（lead margin，执行层规则）预登记草稿（2026-10-07，未启动，待审）

## 问题与依据

[results/four_dirs.md](../results/four_dirs.md) 方向 3、[four_dirs/hugsim.md](../results/four_dirs/hugsim.md)：HUGSIM 64 上 P2H `spec_plan_smooth` 的前景碰撞 31 场景，其中前方静止 / 慢车（D3b）10 场景，替换上限 (a) 3.8 [0.2, 8.5]、(b) 3.5。lead 头看到了（0.93），plan 也减速（0.67），但 plan 在最后 1.5 s 里避开静止车只有 0.13；lead 头近距离测距偏远：真实间距 < 3 m 时 lead_x 多 +1.95 m [1.66, 2.47]（接触前 1 s 中位：真实 2.2 m，lead_x 4.4 m）。plan 要停但执行没停只 0.31。即模型把停车点定在它以为还远 2 m 的车上。navtest 前方静止车碰撞 0.50 分（P2H 特有）。

第 140 条的教训：只加谨慎的规则风险是停住不走（resume 规则起步后撞上 lead 头没看到的目标，闸门判停）。本规则只在已在减速 / 有前车时收紧停车点，不负责起步。

## 规则（写代码前定死，一个 driver 两榜同用）

在 plan 交给执行层之前（HUGSIM 客户端与 NAVSIM 导出共用同一函数，`jevdrive/openpilot` 接口层）：

- 触发：lead_prob > 0.5。
- 校正距离 g = lead_x − b(lead_x)，b 取 four_dirs 的标定表（lead_x < 6 m 时 b = 2.0 m，6–10 m 线性降到 0，> 10 m 为 0）。
- 允许位移：s_max(t) = max(0, g + max(lead_v, 0)·t − d_min)，d_min = 2.5 m（ego 后轴到车头约 4 m 已含在 lead_x 的口径里，按 four_dirs 的口径核对一次）。
- plan 的累计弧长 s(t) 超过 s_max(t) 的时刻起，把该点及之后的位姿沿 plan 路径回拉到 s_max(t)（形状不变，只改速度剖面）；不触发时 plan 原样。
- 数值不调；b 表来自 HUGSIM（`hugsim_lead_calib.csv`），NAVSIM 上若 lead 输出可得则用同一表。

## 第一步：离线（CPU，无模拟）

- HUGSIM：在已存 D3b 10 场景的逐步 trace 上（`$DATA_DIR/runs/op_parity/four_dirs/hugsim_traces.json.gz`），对接触前 3 s 内每一步的 plan 套规则，按 actor 的已记录运动判断「规则后的 plan 在 1.5 s 内是否避开 actor 框」。另在 27 个完成场景的全部步上统计触发率与被回拉的步数（停住风险的代理）。
- NAVSIM：先确认 navtest bench 的 plans 阶段是否导出 lead 头（`P2H10` 的 torch port 输出列）；若无，NAVSIM 读数延后到加上该列（记为偏离）。若有，在已存 navtest plans 上套规则后 CPU 重打分（`fd_navsim.py replay` 的 in-process devkit）。
- **离线过**：D3b 场景中 ≥ 7 / 10 的接触前 plan 被规则改成避开；完成场景里规则改动的步 < 5%；navtest（若可读）EPDMS ≥ −0.2 且 NC + TTC 失败降。否则停，报告。

## 第二步：闭环小读（仅离线过时；GPU，需再确认）

HUGSIM `spec_plan_smooth` + 规则，P2H10-F-s0 / s1，先 5 个场景（0254-extreme-00、0138-extreme-00、0411-medium-00、032-medium-00、3000_3200-medium-00），经 `jevdrive.bench`（`--opts` 记录规则，独立 run key）。

- **过**：5 个中 ≥ 3 个接触消失，且无新增 max_steps / 起步停滞 → 跑全 64（2 seed），判据全 64 HD 对 `spec_plan_smooth` 差 ≥ 0、无新增卡死；navtest EPDMS ≥ −0.2。
- **停**：< 3 个消失，或出现新的卡死 → 停，报告。

## 代价与时间

离线约 1–2 h CPU；闭环 5 场景约 15 min，全 64 两 seed 约 30–40 min（第 149 条重复实测）；navtest 重打分约 10 min。

## 限定

这是执行层的补丁，修的是测距偏差的症状；根本修复在近距离测距表征（four_dirs 修复 4）或训练目标（agent hinge，另一份登记）。HUGSIM 的 D3a 对向脚本车（21 场景）不在覆盖内，WA-JEPA 停住也被撞。HUGSIM 每场景单次，场景级差 < 0.2 不算证据。
