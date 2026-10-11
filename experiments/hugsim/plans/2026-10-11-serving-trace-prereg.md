# HUGSIM serving trace：微调臂在 HUGSIM 里每个 slot 实际吃到什么、时钟差在哪里（预登记）

2026-10-11。只诊断，不改 serving 行为，不训练。起因：[served_plan_length.md](../../body1/results/served_plan_length.md) 第 7 点量到 PAI 上
warp 帧训练的 `P2H10` 吃真实 10 Hz 帧后 plan 短约 10%，并写明 HUGSIM 的帧路径没有追过；docs/hugsim.md 写 openpilot 每 0.2 s context step
直接吃一张渲染帧、HUGSIM 4 Hz、模型时钟快 1.25 倍。

## 已定、不重测（只引用）

- 第 142 条：全量协议定 warp（2 Hz key + ego-motion warp 的 0.2 s lattice）；`P2H10` 家族按它训练。
- 第 116 条：shipped 模型上历史帧质量不是瓶颈（真实 10 Hz 对 GIMM +0.51，CI 含 0）。
- docs/hugsim.md 与 `interface.py` 的声明偏离：HUGSIM 用 dilate 时钟（一个 0.25 s 仿真步当作一个 0.2 s context step，速度 ×1.25 进、plan
  时间 /1.25 出），shipped 模型在 comma1M 上离线代价是 2 s 横向误差 +25–38%；5 s 静态预热是声明偏离（第 100、118、124 条）。
  这些是「设计如此」，本次不重测其代价，只核对它在微调臂的每个消费者处是否真的被一致地补偿。

## 两个问题，各自的判据

**1. 帧来源。** 对象：`P2H10-F-s0`，preset `spec`（`bench` 的默认 serving 路径：`zs_agent.py` → `hugsim_zs_server.py` 里的 queued ONNX
`pp-P2H10-F-s0.onnx`，其 `state_img_q` (2, 5, 6, 128, 256) 与 `state_feat_q` (128, 1, 16384) 是 ONNX 内部队列）。

- 记为「存在」：warm 决策（仿真步 n ≥ 9）上，policy 读到的每个 slot 的帧对都是 agent 当步直接打包的渲染帧（CRC 与第 n−j、n−j−1 步送出的
  `img2` 相同），不是由 2 Hz key 经 `op_interp` warp 得到的 lattice 帧；即与训练协议 W 的像素来源不同。
- 记为「不存在」：slot 帧是 key + warp 的 lattice（与 `sh30_core.lattice_gpu` 同构）。
- 同时报告、不设判据：每个 slot 相对 t0 的真实时间偏移与模型以为的偏移；有效 slot 数（训练 8 个有效 + 最老一格为零）；n < 9 时 slot 里是
  什么（预热重复帧）。

**2. 时钟。**

- 事实量：相邻决策的仿真时间差（`info["timestamp"]` 与 HUGSIM `kinematic.yaml` 的 dt）、每个决策模型走了几个 20 Hz 步（= 模型以为过了多久）。
- 记为「存在未补偿的时钟错」：下列任一消费者把模型时间当仿真时间用（或反之），表现为系统性的 1.25 或 0.8 倍：
  (a) 喂给 policy server 的速度 / 仿真速度；(b) 喂给 bias server 的 vx、ax 与 4 个位姿历史的时间点；(c) 送出的 plan 第 k 点 = 模型 plan 在
  τ = 0.5k / 1.25 处的插值；(d) `op_ctrl.hugsim_steer` 的 dt 与 delay；(e) 执行结果：v > 3 m/s 的步上，下一步实际位移 / (plan 第一点距离 × 0.5)
  与 plan 第一点距离 / (0.5 s × 当前速度) 的中位数偏离 1 超过 0.05。
- 记为「不存在（时钟是声明的 dilation，各处一致）」：以上全部在容差内。此时 1.25 倍只作为已声明的输入分布偏移报告（模型看到的世界快 1.25 倍），
  不算 bug。
- 另报：plan 相对喂入速度的长度（模型时钟下 arc(τ) / (v_model × τ)，warm 决策，v_model > 3 m/s），与 served_plan_length.md 的 nuPlan warm
  0.99 / PAI 0.949 并排。这是描述性读数：场景与相机不同，不能单独归因到帧来源。

不去找「证实怀疑」的读数：任一项读出来不是 bug 就照实写。

## 记录什么

1. 代码阅读（主体）：`file:line` 级别的路径。
2. 已有 rollout 日志（不占卡）：`runs/bench/hugsim/P2H10-F-s0_spec-rr{1,2}` 的 `zs_steps.jsonl`（每步 t、v、喂入的 ego 特征、reps、模型
   plan、送出的 plan）→ 时钟各项与 plan 长度。
3. 仪表化短跑（默认关，agent opt `trace`，只读不写状态）：每个决策记 仿真时间、送出帧的 CRC、reps、喂入速度、dilation；server 端每个 20 Hz
   步读回 `state_img_q` 5 帧的 CRC 与新入队 feature 行的 CRC，决策末读回 `state_feat_q` 128 行的 CRC。离线把队列里每一行对回产生它的
   帧对（哪一步的渲染帧、重复还是新帧）。写 `<scenario>/zs_trace.jsonl`。
4. ONNX 影响探针（一个 session，1 张卡几分钟）：逐个扰动 `state_img_q` 的 5 帧与 `state_feat_q` 的 128 行，看输出是否变，定出 vision 读哪两帧、
   policy 读哪几行。配合第 3 项得到 slot → 帧的完整表。

不变性检查：trace 开着的 run 与已有未开的 rr1 / rr2 在同一场景第 0 步（输入逐位相同）的 plan 比较，差应在 rr1 对 rr2 的差以内。

## 场景与预算

4 个场景，四个数据集各一个，都在 rr1 / rr2 里跑过：`nuscenes/scene-0013-medium-00`、`kitti360/scene-570_770-easy-00`、
`waymo/scene-113792265837-easy-00`、`pandaset/scene-040-easy-00`。`python -m jevdrive.bench run --model P2H10-F-s0 --bench hugsim
--preset spec --opts '{"trace": true}' --jobs 1 --workers 4`，bench 自己把 stage 交给 GPU pool（VRAM 8 + 8.5 × 4 GB，CPU 11）。探针另用
`cl submit`（约 6 GB、4 核）。合计 1 张卡远低于 1 小时。约 2 小时内跑不通就只交代码阅读结果并写明未验证项。

## 交付

`experiments/hugsim/results/serving_trace.md`（英文）：两问各自的路径、证据表、判定；对 served_plan_length.md 的第二读者核对；每项若要修会动
哪里（不做）。小表放 `results/serving_trace/`。
