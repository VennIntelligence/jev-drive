# HUGSIM 闭环：`P2H10-F-s0` 用 stateless 训练协议 serve 一次，HD-Score 动不动（预登记）

2026-10-11。接第 249 条（[serving_trace.md](../results/serving_trace.md)）与第 250 条（[frame_ab.md](../results/frame_ab.md)）。
这是 mismatch 的闭环测试，不是改 shipping 的 serving：新模式是 agent opt，默认关；关着时输出必须与现在逐位相同。

## 已定、不重测

- serving 的帧与时钟路径（第 249 条）；1.25× 是声明的 dilation，各 consumer 已补偿，不是 bug。
- 离线、开环：同一批 decision 改走训练协议（arm `b` / 映射 M2），plan 在 3 s 处移 2.28 m、长 12%，v < 2 m/s 时长 44%，step 0 处 served plan
  短 4.2 倍（第 250 条）。**这不是分数预测**：`b` 是在 served 臂走到的状态上评估的。
- 去掉 static warm-up 的 queued ONNX cold start 起不来（第 124 条，shipped Cinque，`spec_cold`）。那是另一条路径（queue 从零状态起），
  但说明 launch 是这次要先看的风险。
- HUGSIM 的 run-to-run：同一 checkpoint 的 `spec` 重复 `rr1` / `rr2`（2026-10-07）HD 0.38825 / 0.38800，各类计数相同；docs/bench.md 记的
  between-server noise 是 HD ±0.005。

## 问题

`P2H10-F-s0`（preset `spec`）在 HUGSIM 64 个 scenario 上，每个 decision 改走 stateless 训练协议（8-slot warp lattice、真实时钟、无 dilation），
HD-Score 及其分量相对现有 served `spec` run 怎么变；第 250 条预测的起步差异在闭环里是否出现、是否换成分数。

## 臂与 reference

- **臂 `stateless`**：agent opt `stateless`（`{}`）。每个 sim step n：key = step n、n−2、n−4、n−6 的 packed frame（各在自己的 rear-axle pose 上），
  速度 / 加速度 / pose history 不乘 1.25，`sh30_core.Core.plan` 原样调用（`synth = "cpu"`、`cold = "backwarp"`，与训练、bench navtest、
  nuPlan driver 同一份代码）。输入由 `lib/parity_hugsim.core_inputs` 构造：就是 frame_ab_torch.py 里 arm `b` 用的那个函数，挪进 lib 让离线脚本与
  agent 共用，不重写。server：`experiments/hugsim/scripts/stateless_server.py`（op-train env，一个 `Core`，一把锁），由 bench 的 worker 启动，
  代替 ONNX policy server 与 bias server。
- 与离线 arm `b` 的两处差别（都写明）：(1) `b` 一律 `lht = False`；闭环里 `lht` 取 scene 的 traffic flag（Singapore scene 为 True），与 reference run
  喂给 ONNX 的 traffic 输入一致、也与训练行一致。(2) `Core.plan` 只返回 plan；`spec` 用 action head 的 curvature 转向，所以 server 在同一组
  token / ego feature 上再跑一次 policy forward 取完整输出（frame_ab_torch.py 的做法），并断言两次的 plan 相差 < 1e-3 m。
- desire：训练行 desire 恒 0，`Core.plan` 没有 desire 输入，command 只走 adapter 的 one-hot。第 250 条的第四处不同（desire pulse）因此一并去掉。
- **reference**：同一 checkpoint 的 served `spec`。主对照 = 在本次代码上重跑一次的 `spec`（`--repeat r3`，64 scenario，option 关），
  同时充当 off-path identity check；`rr1` / `rr2` 作为次对照与 run-to-run spread。seed 0（HUGSIM 这条路径没有随机种子，「seed」指 checkpoint seed）。
- plan 之后的一切（forward_only、straight_stop、iLQR 纵向、op-ctrl patch、preset `spec`）不动，除了下面的 time base。

## Time base：跟着真实时钟改的 consumer（第 249 条的 consumer 清单逐项）

| consumer（第 249 条） | dilate 下 | stateless 下 | 在哪改 |
|:--|:--|:--|:--|
| 帧进模型：一个 render = 4 个 model step = 0.2 model s | `reps` = 4，dil 1.25 | 不 step queue；key 每 2 个 sim step 一张 = 0.5 s，lattice 0.2 s 在真实时钟上 warp | `zs_agent.py` `stateless()`，`parity_hugsim.Stateless` |
| 给 policy 的速度（decode 里 action curvature 的 `max(1, v)^2`、plan curvature） | `ego_velo × 1.25` | `ego_velo` | `Stateless.step` 的 `speed` |
| ego feature：vx × 1.25、ax × 1.25²、pose 在 sim −1.875 / −1.25 / −0.625 / 0 s | scale 1.25 | scale 1.0：sim −1.5 / −1.0 / −0.5 / 0 s，原值 | `core_inputs(..., scale = 1.0)` |
| plan 输出：sim 0.5 k s = model 0.5 k / 1.25 s | `openpilot_to_plan(dil = 1.25)` | `openpilot_to_plan(dil = 1.0)` | `zs_agent.py` `stateless()` |
| 横向：`op_ctrl.hugsim_steer` dt 0.25 sim s、sim 速度；`OP_CTRL.delay` 0.25 sim s = 0.2 model s | delay 0.25 | dt 与速度不变（本来就是 sim 单位）；**delay 0.2 s**（lateralDelay 原值，`spec_hold` 的先例） | `--controller-env '{"OP_CTRL": {"delay": 0.2}}'`；agent 在 opt 开着而 delay ≠ 0.2 时报错拒跑 |
| 纵向：iLQR 跟踪已在 sim 时间上的 plan | 不变 | 不变 | — |
| interface.json | `history.clock = dilate` 等 | `history.frames = warp-lattice`、`rate_hz = 2`、`clock = real`、`warmup = backwarp`、`lateral.delay_s = 0.2`、`command.channel = onehot` | `interface.resolve_hugsim` |

不需要换算、现在按原值生效的：`MIN_SPEED` 1 m/s、模型的 lateral action time 0.275 s。与 `stateless` 不兼容、agent 直接拒绝的 opt：
`parity`（bias 由 `Core` 自己算）、`derot_*`、`lstab`、`op_lead`、`lead_margin`、`op_clock = hold`。代码行号在结果 note 里列。

## Cold start

用训练协议自带的：`sh30_core` 的 `backwarp`。没有 5 s static warm-up（stateless 协议里不存在 queue，也就没有「预热」这个操作）。
step 0–1：1 张 key；2–3：2 张；4–5：3 张；n ≥ 6：4 张。缺的 history pose 由最老的已知状态按恒定 body velocity 与 yaw rate 倒推，
缺的 slot 是最老 key 沿倒推 pose 的 back-warp（8 个 slot 都 valid）。车以 1.0 m/s 出生。已知限制：这个 checkpoint 没用 backwarp cold start 训过
（[served_plan_length.md](../../body1/results/served_plan_length.md)：nuPlan 上 decision 0 是减速 plan；frame_ab：HUGSIM step 0 上 `b` 的 plan 反而比
served 长 4.2 倍）。不另做 warm-up 变体：那是第二个臂，超出本次一个 run 的范围。

## Gate（先过再跑闭环）

1. **serving 模式 = 离线 arm `b`**（`scripts/stateless_gate.py`）：第 250 条录下的 16 个 scenario / 989 个 decision（box
   `runs/bench/hugsim/P2H10-F-s0_spec-c11011ea9047d/…/zs_frames/`），按 step 顺序把录下的 frame 与 state 喂进 serving 代码
   （`parity_hugsim.Stateless` → wire → `stateless_server.py`），与 `runs/hugsim_frame_ab/out/torch/*.npz` 的 `pos_b` 比 4 s 以内的 plan 点
   （真实时钟，`T_IDXS ≤ 4 s`）。容差 0.03 m（docs/long-runs.md 的 fp16 规则）。过线：超容差的 decision ≤ 1%，每个 decision 的 key 数与 `b` 相同，
   且 wrong-arm control（同一批输出对 `pos_wd`，dilated clock 的 lattice）超容差 ≥ 50%。gate 用 `lht = False`（`b` 的设置）；Singapore scene 再用
   自己的 flag 重放一次，只报不 gate。不过 → 停，报告。
2. **off-path identity**：opt 关着时 agent 走的代码路径没有分支变化（`self.sl is None`）。闭环核对照 serving_trace.md 的做法：`r3`（新代码、opt 关）
   对 `rr1` / `rr2`，step-0 plan 相同的 scenario 数、全程每个 sent plan 相同的 scenario 数、end class 与 HD；已知同代码重复之间也只有部分 scenario
   全程逐位相同（serving_trace：3 / 4，第 4 个差 8 mm），所以判据是「`r3` 对 `rr1` 的差不大于 `rr1` 对 `rr2` 的差」，至少看 2 个 scenario（实际 64 个）。

## 读什么

- HD-Score 与分量（rc、nc、dac、ttc、c、pdms）：`stateless − reference`，按 scenario 配对，`jevdrive.stats.paired`（B = 10 000，seed 0）。
- fixed / broken：HD 升 / 降超过 0.05 的 scenario 数；end class 转移矩阵。
- 失败分类（bench `units.csv` 的口径）：complete、fg_coll、bg_coll、off_route、stuck（max_steps）、spin（heading 误差 ≥ 60°）、
  launch stall（前 40 step 峰值速度 < 1.6 m/s）。
- 起步：到 2 m/s 的时间（首个 v ≥ 2 m/s 的 sim 时间；到不了的记为 censored 并计数）；v < 2 m/s 的 decision 上 sent plan 的 0–3 s arc
  （每臂均值与比值，按 scenario cluster bootstrap；注意两臂的状态不同，这是闭环读数不是 A / B）；step 0 的 plan arc（同一初始状态，可直接比）。
- plan 长度（context）：warm、v > 3 m/s 上 arc(0–3 s) / (3 s × v)，对 frame_ab 的 1.029（served）/ 1.127（`b`）。
- latency：每个 decision 的 server `infer_ms`（分 lattice / encode / policy / action）、agent `rtt_ms`、`agent_ms`，off 对 on；scenario wall。
- run-to-run spread：`rr1` 对 `rr2`、`r3` 对 `rr1` 的配对 HD 差与 |ΔHD| 分布，作为判断配对差的底噪。

## 什么结果意味着什么（先写死）

底噪 = max(0.005, `rr1` / `rr2` / `r3` 两两配对 HD 均值差的最大绝对值)。

| 结果 | 读法 |
|:--|:--|
| ΔHD 的 CI 不含 0、为正、且 > 底噪；fixed > broken | mismatch 在闭环里扣分。只有一个 seed、一个 checkpoint：值得补 seed 1 / 其他 parity 臂，再谈默认 serving |
| 起步更快 / plan 更长（到 2 m/s 更早，低速 arc 更大），但 ΔHD 的 CI 含 0 | 第 250 条的 plan 差异在闭环里存在，但不是 HD 的限制项；看 class 转移（例如起步更快换成更多前景碰撞，第 142 / 239 条的 stop-for-crash 交换）。mismatch 作为 HUGSIM 掉分原因降级 |
| 起步更快，ΔHD 为负且 CI 不含 0 | 去掉 dilation / 换协议在这个 checkpoint 上是净损失；served 路径的「慢」在保护分数。照实报哪一类变坏 |
| 闭环里 plan 并不更长（起步指标在底噪内） | 开环的 12% / 44% 在自己走到的状态上不出现 |
| launch stall 大面积出现（pilot ≥ 6 / 8 起不来，或全量 launch stall ≥ 16） | cold start（没训过的 backwarp、无 warm-up）主导，mismatch 测试被混淆；停在 pilot，报告，不跑全量 |
| ΔHD 在底噪内、各类计数基本不变 | mismatch 对 HD 无可测影响（一个 seed 的精度内） |

限制（事先声明）：一个 checkpoint seed、一次 run；stateless 同时改了帧来源、slot 数、时钟、desire、cold start、lateral delay 的取值，
闭环差是这一揽子的合计，不做归因；`lht` 与 action head 的两处与离线 `b` 的差别见上。

## Staged launch 与 sanity checklist

同一个 run identity（bench 会 union scenario，已完成的跳过）：

1. **1 个**：`scene-0051-easy-00`。看：有 `eval.json`、end class 正常；`interface.json` 是上表的值；step 0 的 model plan 与 gate 里同一 scenario
   step 0 的 `b` plan 相差 ≤ 0.03 m（同一初始帧；若该 scene 是 lht 则对 gate 的 scene_lht 重放）；plan / kappa 全部有限；每步 latency。
2. **8 个**（每个数据集 2 个，取自录帧的 16 个；reference 里 4 complete、2 fg、1 spin、1 launch stall）：`scene-2800_3000-easy-00`、
   `scene-8440_8640-easy-00`、`scene-0051-easy-00`、`scene-0013-medium-00`、`scene-021-extreme-02`、`scene-039-easy-00`、
   `scene-398895700423-easy-00`、`scene-881121006469-hard-00`。checklist：
   - crash / 缺失 0；每个 scenario 有 HUGSIM 的 end class；
   - launch stall ≤ 5 / 8（reference 1 / 8；≥ 6 → 停，按上表报）；complete 数照实记（reference 4）；
   - warm、v > 3 m/s 的 arc / (3 s × v) 在 [0.9, 1.4]（第 250 条：served 1.03、`b` 1.13）；step-0 plan arc 是 reference step 0 的 2–8 倍（第 250 条 4.2 倍）；
   - 每个 decision 的 agent 侧 latency median ≤ 1 s、p99 ≤ 5 s（HUGSIM 循环阻塞等 plan，没有实时约束；bench 的 watchdog 是 sim.log 静默 420 s、
     单 scenario 1 500 s），单 scenario wall ≤ reference 同步数的 2 倍；
   - plan、kappa 无 NaN / inf。
   不过 → 停，不跑全量。
3. **64 个**（`all64`），与 reference 重跑 `r3` 同时提交。

## 预算与耗时估计

录帧 run：16 scenario、1 张卡、6 slot，8 min。64 scenario ≈ 0.6–0.8 card-hour；bench 默认每张卡一个 worker job，池里有 4 张以上空卡时
wall 约 10–15 min。stateless 每个 decision 多出 lattice（CPU，8 线程）+ encoder + 两次 policy，gate 里先量；按 ~0.1–0.2 s / decision、
每 scenario 100–400 step 估，wall 增加 < 20%。合计：gate ~10 min（1 卡）+ pilot 1 + 8（~15 min，1 卡）+ 全量两个 run（各 ≤ 1 card-hour）
≈ 2–2.5 card-hour，在 2–4 card-hour 的预算内；人力 3–4 h。全部经 `python -m jevdrive.bench run`（bench 自己交给 GPU pool），gate 与 report 各一个
`cl submit`，都在 `jevdrive.run.Run` 里。

    B=".venv/bin/python -m jevdrive.bench"
    $B run --model P2H10-F-s0 --bench hugsim --preset spec --opts '{"stateless": {}}' --controller-env '{"OP_CTRL": {"delay": 0.2}}' --scenarios <1 | 8 | all64>
    $B run --model P2H10-F-s0 --bench hugsim --preset spec --repeat r3 --scenarios all64

## 交付

`experiments/hugsim/results/stateless_cl.md`（英文）：gate、staged-launch 检查、配对表（HD 与分量、CI、fixed / broken、失败分类、起步指标）、
reference spread、latency、time-base 改动与代码行、limits；小表在 `results/stateless_cl/`；topic README 里挂在 frame_ab.md 旁边。
