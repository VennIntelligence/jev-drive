# HUGSIM frame A/B：同一批 decision 走 serving 路径与走训练协议，plan 差多少（预登记）

2026-10-11。只量 mismatch，不改 serving 路径，不跑 protocol (b) 的闭环，不训练。接第 249 条（[serving_trace.md](../results/serving_trace.md)）：
`P2H10-F-s0`（preset `spec`）在 HUGSIM 里每个 slot 吃两张相邻 raw render（间隔 0.25 s，9 格全填），训练是 8 格 warp lattice。
effect size 在 HUGSIM 帧上没量过，只有 navtest stand-in（real lattice 移 1.20 m、arc / log 0.895；填第 9 格移 0.36–0.42 m 不变短）。

## 已定、不重测

- 1.25× 时钟是声明的 dilation，各 consumer 已补偿（第 249 条）。不再查它是不是 bug；下面只把它当作一个需要单独记账的分量。
- slot → 帧的对应关系（slot j =（render n−j−1，render n−j），n < 9 时旧 slot 是（render 0，render 0））：第 249 条已逐帧核对，直接用。

## 问题

在已记录的 HUGSIM render 上离线重放：同一个 decision（同一 sim step、同一 ego 状态、同一 command），输入从 (a) 换到 (b)，plan 移多少、
变长还是变短、横向 / curvature 变多少；差异集中在 rollout 的哪一段（launch、低速、转弯）。
开环：(b) 在 (a) 闭环走到的状态上评估，不是 (b) 自己会走到的状态，结果不是分数预测。

## 臂（全部在同一批 decision 上，逐个只改一处）

| 臂 | 输入 | 路径 | 用途 |
|:--|:--|:--|:--|
| `a` | rollout 里 log 的 plan | 线上 queued ONNX（TensorRT） | serving 本身 |
| `a_off` | dump 的 `img2` + `intent_bias` + desire，按 `reps` step 同一个 ONNX | `jevdrive.openpilot.model.OPModel`（trt） | gate (i) |
| `a_nod` | 同 `a_off`，desire 全 0 | 同上 | serving 喂 desire pulse（`command.channel = desire-sim+onehot`），训练行 desire 恒 0：第 249 条没列的第四处不同，单独记 |
| `a_zb` | 同 `a_off`，bias 置 0 | 同上 | gate (i) 的 wrong-model control |
| `a_t` | 9 格 raw render pair，serving 的 ego feature（model clock），desire 0 | torch port：`pp_train.PModel`（`sh30_core.Core` 里那份权重）+ 它的 vision encoder | 把 torch 路径锚到 serving（应 ≈ `a_nod`） |
| `s8` | `a_t` 去掉最老一格（8 格 raw pair） | 同上 | 9 → 8 格 |
| `s8z` | `s8` 且最老一对的前一帧换成 zero image（训练行的 slot 结构） | 同上 | 9 → 8 格（含 zero-image pair） |
| `wd` | 8 格 warp lattice，**model clock**（映射 M1，见下），ego feature 与 serving 逐位相同 | `sh30_core.Core.plan`，原样调用 | 像素来源（时钟不变） |
| `b` | 8 格 warp lattice，**real clock**（映射 M2），ego feature 不带 1.25 | `sh30_core.Core.plan`，原样调用，cold start 用它自带的 `backwarp` | 训练协议 = (b) |

lattice 用 `Core` 的默认 `synth = "cpu"`（`op_interp.synth_cpu` `warp`，全部 checkpoint 训练所用的 reference；`lattice_gpu` 是它的 GPU 等价实现，
LAT1 已核对）。不重写 warp、不重写 policy。`a_t` / `s8` / `s8z` 只是把 raw pair 送进同一个 encoder 和同一个 `PModel.forward`（n = 9 或 8）。

三段归因（顺序固定，order-dependent，照实写）：desire `a → a_nod`；port 残差 `a_nod → a_t`（应在噪声内）；**9 对 8 格** `a_t → s8z`；
**像素来源** `s8z → wd`；时钟 / key 网格 `wd → b`。**static warm-up** 不是链上的一步，而是按 decision 分层：n ≤ 8（旧 slot 是首帧重复；
(b) 在 n < 6 是 cold start）与 n ≥ 9（warm）分开报，warm-up 的效应 = 同一条链在 n ≤ 8 上比 n ≥ 9 上多出来的部分。

## keyframe 怎么落到 0.25 s 的 render 网格上（设计选择）

训练 lattice：key 在 −1.5 / −1.0 / −0.5 / 0 s，slot 在 −1.4 .. 0 s（0.2 s 一格）。HUGSIM 一个 sim step = 0.25 s 真实时间 = serving 当作的 0.2 s model time。

- **主选 M2（real clock）**：key = render n、n−2、n−4、n−6（真实 0 / −0.5 / −1.0 / −1.5 s，都在网格上，pose 就是该 step 的 rear-axle pose，
  不插值）；lattice 的 0.2 s slot 按真实时钟从这些 key warp 出来；速度、加速度、pose history 都不乘 1.25；plan 的时间就是真实时间。
  理由：这是训练协议本身，也是 serving_trace.md「fix」一节描述的做法；每张 key 都是在它声称的 pose 上拍的；`Core.plan` 不用改一行。
  代价：`b − a` 里除了像素来源和格数，还含「去掉 dilation」这一项，所以加 `wd` 把它拆开。
- **最接近的备选 M1（model clock）**：保留 serving 的时钟，key 取 model time −1.5 / −1.0 / −0.5 / 0 s = 7.5 / 5 / 2.5 / 0 个 step 之前，
  半格的两张取较新的 render（7 / 5 / 2 / 0，沿用 `lib/parity_hugsim.key_steps` 给 side camera 定的规则），pose 用 model clock 上插值的
  pose（与 bias server 收到的相同），速度 ×1.25。缺点：两张 key 的图像比它声称的 pose 新 0.125 s（10 m/s 时 1.25 m），影响 8 格里的 3 格；
  训练里没有这种错位。所以 M1 只作为桥（`wd`）和敏感性读数，不做 headline。

两种映射都报 `vs a` 的数；headline 是 `b`（M2）。相机位置 `cam_t` =（`rear_offset`，0，`interface.HUGSIM_HEIGHT[dataset]`）。

## 读什么

每个 decision、每个臂：HUGSIM plan 坐标系（x 右、y 前、原点在相机）下真实时间 0.5 .. 4 s 的 plan 点（`a` 系与 `wd` 用
`hugsim_zs.openpilot_to_plan(dilation = 1.25)`，`b` 用 `dilation = 1`；board 用到 3 s）。

- 位移：|Δp| 在 1 / 2 / 3 / 4 s，以及 0.5–3 s 的平均。
- arc-length ratio（0–3 s，另报 0–4 s）：Σ arc(臂) / Σ arc(`a`)，pooled 与 per scenario。
- 横向：3 s 处横向位移差（带符号与绝对值）；`spec` 实际用来转向的 action curvature 的差，`plan_smooth` curvature 的差。
- CI：`jevdrive.stats`，按 scenario 做 cluster bootstrap（B = 10000，seed 0）；ratio 用同一组 resample 的 ratio of sums。
- 分层：warm（n ≥ 9）与 warm-up（n ≤ 8）分开；rollout 位置：sim 速度 < 2 / 2–5 / > 5 m/s，转弯（|`a` 的 action curvature| > 0.01 1/m 或
  |hyaw15| > 5°）对直行，episode 的前 2.25 s / 2.25–5 s / 之后；按 scenario 的结局（launch stall、spin、off route、collision、complete）。

## Gate（先过再读）

1. **(i) 离线 = serving**：`a_off` 对 dump 的 full-precision `pos`，只看 model time ≤ 3.2 s（= 真实 4 s）的点；容差 0.03 m（docs/long-runs.md 的
   fp16 规则）；超容差的 decision 占比 ≤ 1 % 且 `a_zb`（control）超容差占比 ≥ 90 % 才算过。不过 → 离线重放不是 serving 路径，停，报告。
2. **(ii) (b) = 训练协议**：`Core.plan` 在 32 个 navtest token（真实 CAM_F0 JPEG + index 的 ego 状态）上对 bench 的预测
   `bench/ol/lb_navtest/preds/P2H10-F-s0-warp__base.npz`：8 个 pose 的最大差 ≤ 0.05 m 的 token ≥ 95 %（sh30_check 之前读到 max 0.045 m）；
   control：同一 token 去掉 adapter bias 的 plan 应超容差。
3. **(iii) HUGSIM 侧几何约定（自加）**：把 render n−2 用 `op_interp.warp_frame` 按记录的 ego motion warp 到 render n−1 的 pose，与真实的
   render n−1 比 Y 平面 MAE：应低于不 warp 的 MAE，且低于 yaw / 平移符号翻转的 warp。只在 v > 3 m/s 的 decision 上算。不过 → `wd` / `b` 的
   pose 约定有错，停。

## 判据（先写死）

在 warm decision（n ≥ 9）、`b` 对 `a` 上：

- 「与 PAI 的 10 % 同量级」：pooled arc ratio（0–3 s）在 [0.95, 1.05] 之外且 CI 不含 1；或 3 s 处平均位移 ≥ 1.0 m
  （navtest stand-in 是 4 s 内平均 1.20 m）。
- 「可忽略」：arc ratio 在 [0.98, 1.02] 之内，且 3 s 处平均位移 < 0.3 m（填第 9 格的 stand-in 是 0.36–0.42 m；`a` 对 `a_t` 的 port 残差是地板，
  若它 ≥ 0.3 m 则以它为准并写明）。
- 两者之间：存在但小于 PAI。方向（变长 / 变短）照实报，不预设。

## Scenario 与录帧

已有 run（`runs/bench/hugsim/P2H10-F-s0_spec-{rr1,rr2,c85e22e1821d5}`）没有存 render（没有 `zs_dump`）。录一次：agent 新 opt `trace_frames`
（默认关；开着时 plan 与 `zs_steps.jsonl` 不变），每个 decision 写 `<scenario>/zs_frames/<step>.npz`（`img2`、`intent_bias`、full-precision ego
feature、ego pose / 速度 / 加速度 / command / desire / reps、模型输出的 `pos` / `yaw` / `vel`、action curvature）。

16 个 scenario，四个数据集各 4 个，按 rr1 / rr2 的结局挑（两次结局相同）：

| 数据集 | scenario（rr1 结局） |
|:--|:--|
| kitti360 | `scene-8440_8640-easy-00`（spin）、`scene-570_770-easy-00`（bg collision，heading 误差 50°）、`scene-5980_6180-easy-00`（off route）、`scene-2800_3000-easy-00`（complete） |
| nuscenes | `scene-0041-medium-00`（off route）、`scene-0051-easy-00`（complete）、`scene-0930-hard-00`（complete）、`scene-0013-medium-00`（fg collision） |
| pandaset | `scene-021-extreme-02`（64 个里唯一的 launch stall）、`scene-039-easy-00`（complete）、`scene-040-easy-00`（complete，长）、`scene-034-hard-00`（fg collision） |
| waymo | `scene-144248042870-extreme-00`（spin）、`scene-398895700423-easy-00`（complete）、`scene-113792265837-easy-00`（complete，heading 误差 20°）、`scene-881121006469-hard-00`（fg collision） |

rr1 里合计约 990 个 decision。launch stall 只有 1 个、spin 只有 2 个：这两类的读数是个例，不做 CI 结论。

## 预算

录帧：`python -m jevdrive.bench run --model P2H10-F-s0 --bench hugsim --preset spec --opts '{"trace_frames": true}' --jobs 1 --workers 6`
（bench 自己交给 GPU pool），约 10 min。离线两段各一个 `cl submit`：ONNX 重放（envs/openpilot，约 6 GB、4 核，几分钟）与 torch 臂
（envs/op-train，约 8 GB、12 核，约 10 min），都在 `jevdrive.run.Run` 里；report 是 CPU。合计 1 张卡 < 1 h，约 2–3 h 人力。
超出就交已成立的部分并写明缺什么。

## 交付

`experiments/hugsim/results/frame_ab.md`（英文）：gate、表、三段归因、集中在哪、limits；小表在 `results/frame_ab/`；topic README 里挂在
serving_trace.md 旁边。
