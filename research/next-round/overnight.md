# 今晚的任务（2026-10-09 夜）

给执行会话的任务书，整段贴进去即可。方向与判定线的出处是 [plan.md](plan.md)，以下各项与它第 3 节的编号对应。

## 背景，三句话

1. 我们的 driver 是 openpilot 冻结 encoder（Cinque）加 P2 adapter 加强 hinge（SH30；按 AlpaSim 输入标准重训的版本叫 AP2）。
   navtest 上对 WA-JEPA 差 2.16，四分之三在占 26% 的转弯 token 上，子分差在 DAC 与 NC；急弯「转不过去」（> 45° 的 2.47%）至今没动。
2. 证据指向冻结表征是上限（d147、d160、d165、d166、d192）。下一步的方法是一条轻量前视支路，经 adapter memory 通道进 policy，由评价器的后果标签监督。
3. 今晚只做 navsim / navhard 上有基准的三件事，它们同时服务论文和 AlpaSim 参赛（NC 对应 at-fault 碰撞，DAC 对应 offroad，navhard stage 2 对应闭环里的离轨状态）。

开工前读：`research/next-round/plan.md` 第 1、3 节；`research/decisions.md` 里 d143、d146、d158、d160、d165、d166、d170、d179、d186 – d193 的全文；
`experiments/op_parity/results/strong_hinge.md`；`docs/lib.md`、`docs/bench.md`、`docs/long-runs.md`。

## 硬约束

- 不用 WA-JEPA 的任何权重或特征做新实验（WA-Cf、整模型都不行）。已有的 WA-JEPA 逐 token 分数只作对照列。
- 真值地图与真值 agent 只能作 oracle 探针和训练标签，不进任何可报告的推理路径。
- 每项开跑前在 `experiments/op_parity/plans/` 写一份预登记（问题、臂、读数、判定线），提交之后才读分数；线照抄下文，不改。
- 所有 GPU job 走 pool（`python -m jevdrive.cl submit`），不手选卡；打分与评测走 `python -m jevdrive.bench`，不写新 runner。
- 先小后大：每项先在 pilot 规模读一次，明确为负就停，不放大。
- 开跑前 grep `research/decisions.md`，已有定论的机制不重测；与既有条目重叠时在预登记里写清这次不同在哪。
- 结果进 `experiments/op_parity/results/`，每项一条 decision；代码、注释、日志全英文。
- 不动 `jev:alpasim-dl` 与 `jev:alpasim-dl2` 两个下载窗口。不向 AlpaSim 提交任何东西，注册与提交是用户的动作。
- 汇报：每项出判定时报一次，出错或卡住立即报，其余时间不报。

## N1. NC 失败分类（不占卡，先做）

**目标**：说清 SH30 在 navtest 上的 NC 失败（约 170 个 token）和 TTC 失败是哪几类，各占多少，哪些是 WA-JEPA 通过的，以及只改纵向能救回多少。

**要交付的读数**：
- 按类型分的表：前方静止或慢车、切入、路口横穿、转弯时侧碰、其他；每类再按转角桶（< 5°、5–20°、20–45°、> 45°）和 ego 速度段分。
- 每类里 WA-JEPA 通过的比例。
- 同路径纵向缩放族的 oracle（例如 0.7 到 1.1 倍速若干档，路径不变，non-reactive 打分）：能把 NC 失败救回多少，同时 EP 掉多少。
- navhard stage 1 上同样的分类，量小的话只报计数。

**为什么**：AlpaSim 本地的零分全是 at-fault 碰撞，SH30 的两个都发生在减速到 3 m/s 以下时；d158 与 d179 说明这一块在冻结特征上难修，要先知道是哪一类。

**不是 go / no-go**，交付的是分类表和纵向 oracle 的回收率，以及一句话：下一步的 NC 修法该打哪一类。

## O1. 真值几何 oracle 走 memory 通道（论文线的第一闸门）

**目标**：量出「一条完美的几何支路」的上限。把真值可行驶 SDF、真值 agent 占用编成 token，经 d160 的 MW 臂用过的 memory 通道喂给 policy，其余同 SH30 的 pilot 配方。

**臂**：只 SDF；只 agent 占用；两者；打乱 memory 的对照（同 token 数、内容错配）。编码方式由你定，原则是 token 数与 WA-Cf 的口径可比，并在预登记里写明。

**读数**：d160 的 pilot 协议（同一训练子集、同一 navtest 读数）。navtest EPDMS 对无 memory 的同配方臂；> 20° 与 > 45° 的 DAC 失败；
> 45° 的切内角与转不过去两个比例；NC + TTC。

**判定线**：
- 两者齐用的臂 navtest ≥ +0.7 且 CI 下界 > +0.3：几何足够，支路的问题收敛成「从前视图预测这些几何」。参照：WA-Cf 是 +0.95 [+0.47, +1.37]。
- < +0.3：缺的不是几何，记为负，轻量几何支路这条路要重估。
- 之间：记为部分，报告哪个子分、哪个转角桶拿到了、哪个没拿到。

**注意**：d166 把真值边界直接喂 plan 通路时没有被读，所以这次必须走 memory 通道，并在报告里用打乱对照证明通道被读了。

**预算**：约 5 卡时以内。超出先报。

## R1. 离轨 warp 行

**目标**：SH30 配方里加入约 10% 的离轨行，看 navhard 涨多少、navtest 掉不掉。

**做法约束**：真实帧经现有重投影引擎做横向 ±0.5 m、小幅 yaw 的静态扰动，目标是日志未来在扰动位姿下的重表达，hinge 照加；
只取 v > 3 m/s 的行（d143 的起步伪影）。不做 on-policy、不做纵向扰动。d132、d141、d143、d146 是近邻，预登记里写清这次与它们的差别。

**步骤**：先 pilot 规模 1 seed 读一次 navhard；点估计 ≥ +1.0 再放全量 2 seed。

**读数**：navhard two-stage（G 帧口径，SH30 是 33.67，S1 75.90 / S2 44.68）分阶段报；navtest；
AP2 输入标准下的离线读数（`experiments/alpasim/scripts/ap2_offline.py` 的口径，m = 1..4）。

**判定线**：navhard 对 SH30 ≥ +2.0 且 CI 下界 > 0；navtest ≥ −0.2。两条都过记为进配方候选。
HUGSIM 的起步停滞与 HD ≥ −0.03 是进配方前的护栏，今晚卡有空再跑，没空留到明天，但没跑之前不写「进配方」。

**预算**：pilot 约 1 卡时，全量约 2–6 卡时。

## 顺序与资源

1. N1 立刻开，纯 CPU。
2. O1 与 R1 的预登记写完后同时提交到 pool，互不依赖。
3. 卡空着而上面三项都在等结果时，补 R1 的 HUGSIM 护栏；不要去起 plan.md 里 L1 之后的任何训练。

## 三项都有判定之后

1. 看 AlpaSim 公开集落了多少：`ls -a $DATA_DIR/datasets/alpasim_nuplan | grep .done`（10-08 23:15 是 part001、002、003、005）。
2. 用已落盘 shard 的全部 scene 建 scene list，按 `docs/alpasim.md` 的 pool 命令跑 SH30-F-s0、AP2-AB-s0，AP2 再重复一次量噪声（每 400 scene 约 0.35 卡时）。
3. 交付：每个 driver 的零分分类表（at-fault 碰撞、offroad、出 corridor）、偏慢 scene 数、at-fault 事件数、两次 AP2 之间的逐 scene 差、SH30 与 AP2 的 best-of-two。
4. 若 R1 过线：用 `ap2_train.py` 按 AlpaSim 输入标准把带离轨行的配方重训一个 AP2 变体，离线读数过了再进这批 scene 对比。没过线就不训。

## 收尾

- 每项一条 decision（先写结论，再写限定和会推翻它的证据），`experiments/op_parity/README.md` 与 `experiments/INDEX.md` 各补一行。
- 最后给一份总报告：三项的判定、数字、遗留问题。结果在前，问题在后，不提问。
