# HLEAD 预登记：把 openpilot 的 lead 通路放进 HUGSIM 闭环（2026-10-10，先于任何带开关的 run）

接第 237 条（HUGSIM 64 上 38 个失败里 9 个 L1：前车在行驶带内、刹停可免，lead 头 9 / 9 亮、6 / 9 要求 ≥ 1.5 m/s²）、第 226 条（AlpaSim 的开关 (b) `JEV_LEAD`）、第 219 条（driver 丢弃 lead 输出）。问题只有一个：同一条 lead 通路放进 HUGSIM 的环里，前车碰撞是否消失，HUGSIM 64 的 HD 怎么变。

**实现（不新写律）。** `experiments/alpasim/lib/serve_fix.py` 里的 radard 概率滤波 / vision lead / lead MPC / planner tick 原样搬到 `jevdrive/openpilot/lead_long.py`（alpasim、op_parity、hugsim 三个主题引用），serve_fix 重新导出全部名字。HUGSIM agent 新增选项 `op_lead`（默认关）：forward_only 之后、straight_stop 之前，plan 的点沿自己的 path 后拉到「plan 速度与 lead MPC 速度解逐点取小」，只减速、无 latch，横向曲率不动。planner 跑在模型时钟里（速度 × 1.25，时间 / 1.25，每个仿真步 4 个 tick），输入是同一次前向的 lead 输出、仿真器报告的 ego 速度；device 到车头 1.5 m（HUGSIM ego 框长 3 m、以 ego 位置为中心，与 `lbd_hugsim.py` 相同）。不读框、route、actor。

**臂（单 seed `SH30-F-s0`，preset `spec_plan_smooth`，`python -m jevdrive.bench`）。**
- A0：第 237 条用的已存 run（HD 0.4432）。
- A0r：改完代码、开关关，repeat `hlead-off`，all64。用途：先在 1 个 scenario 上核对与 A0 逐位相同，再给出同配置重跑的逐 scenario 差（配对噪声的底）。
- A1：`--opts '{"op_lead": {}}'`。
- 不跑 `JEV_VCONT` 臂：1.0 s 来自 AlpaSim MPC 的跟踪窗口，HUGSIM 的 iLQR 没有对应的推导。不扫参数、不跑第二个 seed、不训练。

**顺序。** 阶段 1：9 个 L1 scenario + 5 个 A0 中 complete 且 lead 头亮的步数占比最高的 scenario（053-medium-02、095-medium-01、150623512729-medium-01、102751446607-medium-01、132384196576-medium-01；选法先于 run 固定，095-medium-01 是第 157 条的卡死风险场景）。看逐步日志后：若 L1 修好 0 / 9，或 5 个干净 scenario 里 ≥ 3 个不再 complete，停，不跑 64；否则 A1 跑 all64。

**读法。**
- 单位 = scenario。**L1 修好** = A1 的结局不是碰撞也不是 off_route（complete 或 max_steps）；分开报 complete 与停在前车后面耗尽步数（max_steps），以及各自的 HD 差。**不变** = 仍以 fg_collision 结束。**新失败** = 其他结局。4 个入弯过快（L2）单位用同一三分法，只作描述（lead 通路不针对它）。
- L1 的线：修好 ≥ 5 / 9 记为「lead 通路去掉了前车碰撞」（lead 头要求减速的是 6 / 9，上限约 6）；≤ 2 / 9 为否；3–4 为部分。
- HD：A1 − A0 的逐 scenario 配对差，scenario bootstrap 95% CI（B 10000，`bench report`）。旁列基线 seed 间差：SH30-F `spec_plan_smooth` s0 0.4432 / s1 0.4338（差 0.009）；P2H10-F `spec` 0.388 / 0.402（差 0.014，`experiments/body1/results/other_boards.md`；第 170 条 SH30 两 seed 均值 0.439）。**有收益** = CI 下界 > 0 且均值 > 0.014；**有害** = CI 上界 < 0；其余 = 无可分辨差异。
- 变差的 scenario = HD 差 ≤ −0.05 且 A0r 与 A0 在该 scenario 上一致（差 < 0.05）。逐个归因：stall（max_steps，或结束前限速在作用且车速 < 0.1 m/s ≥ 10 s）、too slow（complete 但 rc / ttc 以外的分项因慢而降）、被追尾（fg_collision 且接触在车尾）、其他。

**已知的坑，直接核对。** 第 157 条：近距 lead 偏远（真实 1.9 m 时多读 4.4 m）、侧偏目标 lead 头看不到；在未修好的 L1 单位上报限速是否触发、停住时到被撞物体的真实间距（仿真框只作标签）。第 140 条：本规则不放车只减速，但 lead 概率掉到 0.5 以下限速即解除，在未修好的单位上看是否属于此类。第 124 条：5 s 静止预热与 dilate 时钟保留；跟车距离 d(v) 用的是 × 1.25 的模型速度，比面值长，属披露的偏离。

**停止规则。** L1 ≤ 2 / 9 或 HD 有害，本线结束：不调跟车距离、不加 danger-zone-only 变体、不换 seed。预算 ≤ 3 card-hour。
