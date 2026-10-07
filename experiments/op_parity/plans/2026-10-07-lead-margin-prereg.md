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

## 执行中的声明与偏离（2026-10-07，写于任何打分之前；用户已批准本登记）

1. **实现**：规则只有一份，`jevdrive/openpilot/lead_margin.py`（`apply` / `LeadMargin`）。HUGSIM：`experiments/hugsim/lib/zs_agent.py` 选项
   `lead_margin`（`{}` = 登记值），位置在 forward_only 之后、straight_stop 之前，记录 `rec["lm"]` 与改动前的 plan；横向不动（op_ctrl 曲率仍来自
   模型）。NAVSIM：op_interp 导出 adapter `lm`（相机系 plan 上套规则，再走 lever 臂转后轴），bench 模型选项 `:lm`（如 `P2H10-F-s0:lm`）。
   接口记为 trick `lead_margin`（两榜都在已声明的 tricks 里，semi）。不开选项时两榜行为不变。
2. **「回拉」的读法**：s'(t) = min(s(t), s_max(t))。字面「越界时刻起全部置为 s_max」在 plan 之后又短于 s_max 时会把车往前推，违背「只加谨慎、
   不负责起步」的意图，故取 min（越界段等于 s_max，其余不变）。
3. **lead_v 单位**：HUGSIM dilate 时钟下模型看到的速度是真实的 1.25 倍，plan 时间已换成模拟器秒，所以规则用 lead_v / 1.25；lead_x 是距离，不变。
   NAVSIM 无时间缩放。lead 头取 HUGSIM 服务器同一解码：selection 0、t = 0 的 x / v，lead_prob[0]。
4. **d_min 口径核对**（不改数值）：HUGSIM plan 原点 = 相机 = ego 框中心，车头在前 1.5 m（框长 3.0 m），four_dirs 的标定也是相机到前车近端，
   口径一致：停住时车头离前车尾 1.0 m。NAVSIM CAM_F0 在后轴前 1.665 m，Pacifica 后轴到车头 4.049 m，相机到车头 2.38 m：d_min 2.5 只留 0.12 m。
   按登记值跑，NAVSIM 的小余量写进限定。
5. **离线 HUGSIM 判据细化**：D3b run = spec_plan_smooth 中以 fg_collision 结束且接触类型为 lead_stopped / lead_moving、所在 (臂, 场景) 格被
   four_dirs 判为 D3b 的 run。被撞 actor 按 four_dirs 规则（与最终 ego 框相交，否则最近）。窗口 = 接触前 12 步（3 s）每一步；「避开」= 规则后
   plan 在 0.5 / 1.0 / 1.5 s 的 ego 框（3.0 × 1.6 m，朝向取 plan 路径）不与该 actor 在 k+2 / k+4 / k+6 步的已记录框相交（超出 run 末尾按最后一步
   速度外推）。**run 改成避开（主）= 窗口内每一步都避开**；次要 = 至少一半步避开。场景改成避开 = 其 D3b run 中至少一半改成避开；
   **闸门：主口径 ≥ 7 / 10 场景**。同口径报告不加规则的基线。完成场景：被判 complete 的格里以 complete 结束的 run 的全部步；「改动的步」=
   有 plan 点被回拉 > 0.05 m；闸门 < 5 %；另报触发率与「规则把移动的 plan 变成停车」（终点 < 1 m 而原 ≥ 1 m）的比例。
6. **NAVSIM 偏离**：已存 navtest plans（P2H 的 op_lb / bench plan 文件）只有 plan，没有 lead 列。按登记「加上该列」：bench 的 parity plans 阶段
   加存 lead_prob / lead_x / lead_v，以 `P2H10-F-s{0,1}:lm` 重出 plans（同一 torch port、同 batch；与原 plan 文件逐位比对 plan_pos），导出用
   adapter `lm`，打分用 bench 的全 navtest v2 devkit（复现 WA-JEPA 91.71 的 harness），不用 fd_navsim replay 的失败 token 子集打分：bench 是规定
   的评测路径，且直接给出全 navtest EPDMS。对照 = 已存 P2H（同 harness）每 token 分。「NC + TTC 失败」= NC < 1 或 TTC < 1 的 token 数，两 seed
   合计。EPDMS 差用 bench report 的 log 聚类 bootstrap，闸门读点估计 ≥ −0.2。
7. **闭环小读判据**：场景「接触消失」（主）= 两个 seed 都不以 fg_collision 结束；次要按 run 计。「新增卡死 / 起步停滞」= 规则臂出现 max_steps
   或 launch_stall（bench 定义：前 40 步峰值速度 < 1.6 m/s），而对照 spec_plan_smooth 的三次存档（r0 / rr1 / rr2）该场景该臂都没有。规则臂单次
   （HUGSIM 近确定，第 149 条），独立 run key（`--opts '{"lead_margin": {}}'`）。
8. **全 64 判据**：HD 差 = 规则臂（单次）− spec_plan_smooth 三次存档均值，两 seed 平均，场景 bootstrap；闸门读点估计 ≥ 0，CI 同报。无新增卡死 =
   没有对照三次都不卡、规则臂卡住（max_steps）的 (臂, 场景)，launch stall 同理。navtest 闸门用第 6 条的结果。navhard 不在登记里：若跑，只作次要
   读数，不设闸门。
