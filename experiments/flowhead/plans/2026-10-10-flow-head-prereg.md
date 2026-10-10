# FLOW1：冻结 encoder 上把 plan 的回归换成 flow-matching 轨迹 head（预登记，2026-10-10）

lane FLOW1，topic flowhead。本文件在任何读数之前提交并 push。输入：第 144、148、172、204、207、240、243、244 条。只动 loss 形式与读出 head；没有新数据、没有闭环、不用 WA-JEPA 的权重或代码（只取做法）。

## 1. 问题

SH30（冻结 Cinque encoder + temporal policy + P2 adapter，λ 30 / margin 0.5 的 footprint hinge，navtest EPDMS 89.55）的 plan 是回归训出来的。第 207 条：转弯差距是路径形状，等弧长曲线偏移是双侧的，|C(4 s)| 0.71 m 对 WA-JEPA 0.40 m（> 45° 0.88 对 0.46），这正是对两个 mode 取平均会给出的形态，第 207 条自己的限定里也写了分不开。WA-JEPA 的轨迹 head 是 x-prediction flow matching，它在自己的联合模型里报告 flow 比回归高约 +1.0 EPDMS。

问题：其余全按 SH30，只把 plan 的回归换成 flow-matching head，navtest 大弯（> 45°、> 20°）的 off-road（DAC 失败）会不会少。

先验（写在读数之前）：低。第 204、243 条：同一份冻结特征上再读一遍的新 head 加得很少。预期 FM − RG 的 EPDMS 在 ±0.3 内、> 45° DAC 失败率的 CI 含 0；预期样本间散布远小于 plan 误差（head 实际上是确定性的）。

## 2. 臂

trainer：`experiments/op_parity/scripts/pp_train.py --thead <kind>`（本 lane 加的开关，默认关；关着时代码路径与原来相同），head 在 `lib/traj_head.py`。命令 = SH30 的原命令加 `--thead`：

`--arm P2 --seed s --frames warp --host --data navtrain_full.s{0..11}of12 --split navsim/op-parity-full --steps 10000 --batch 128 --warmup 300 --eval-every 1000 --hinge-lam 30 --hinge-margin 0.5 --thead <kind> --tag <tag>`

| 臂 | tag | head | 说明 |
|:--|:--|:--|:--|
| FM | `FMH-F-s{0,1}` | `fm` | x-prediction flow matching，4 步 Euler，固定 noise 的单个样本作为 plan |
| RG | `RGH-F-s{0,1}` | `rg` | 同一个网络（x_t = 0，t = 0），同一个 loss，确定性回归；容量、步数、其余一切相同 |
| SH30（参照） | `SH30-F-s{0,1}` | 原 plan 读出 | 已存档，不重训 |

head 的定义（两臂共用，一组写死的超参，不调）：

- 输入 = `select_4`（512 维），即原 plan 读出解码的那个 hidden state；条件化方式与原 plan 相同（ego / history / command 经 adapter 的 bias 进入 context frame）。head 取代 plan 读出：模仿项与 hinge 都作用在 head 的输出上，梯度照常进 temporal summarizer 与 adapter（encoder 冻结，与 SH30 相同）。原 plan 列在模仿行上不再有 loss，只保留 anchor 行的 cons 与 distill。anchor 行上 head 也跟教师的 plan（权重 lam_c）。
- 网络：残差 MLP（宽 512、4 个 block，4.79 M 参数），条件 = LN(h) 的投影 + 时间 embedding，每个 block 前重加；输出零初始化。输出 8 × 3（x, y, yaw；后轴系，0.5–4 s），即模仿目标与导出文件的坐标系，不再经过 lever arm 与重采样（SH30-F-s0 上两条路径的差：均值 1.5 mm，最大 2.6 cm）。
- FM：x_t = (1 − t)·noise + t·x，t = sigmoid(N(−0.2, 1.6)) 截到 [1e-4, 1 − 1e-4]，x-prediction；采样 4 步 Euler（x += dt·(x̂ − x) / max(1 − t, 1e-3)），noise 取一个固定 bank 的第 0 行（每个 token 相同，plan 不依赖 batch）。
- 与 WA-JEPA 做法的两处有意偏离：
  1. flow 在 z-score 空间里走（按训练行日志轨迹的逐维均值 / 标准差），但 loss 算在还原后的 x̂ 上、按 SH30 的逐时刻 σ 归一：0.5·Σ((x̂ − x)/σ)²。这是 SH30 那个 σ 归一 Huber 的 L2 形式（1σ 以内两者相同），目的是让 hinge 的 λ 30 保持原来的相对强度（z-score MSE 的量级约是它的 1/20–1/30）。逐维加权不改变最小点（条件均值），flow 仍是 flow。
  2. hinge 作用在采样出来的轨迹上：每步对模仿行用新 noise 做 4 步 Euler 采样（带梯度），对样本算 footprint hinge。RG 臂的 hinge 作用在回归输出上。所以 hinge 两臂都有，不需要另加「无 hinge」对照。
- RG 与 SH30 的差别：新 head（从零训）取代预训练的 plan 读出，L2 取代 Huber（只差在 1σ 以外的尾部）。

## 3. 放量前的检查

1. tag 全新：提交前在 box 上确认 `runs/op_parity/runs/<tag>`、`runs/op_parity/train-<tag>`、`runs/bench/*/<tag>*` 都不存在，链脚本里有 guard。
2. 默认路径不变：同一条不带 `--thead` 的 60 步命令，在本次提交与它的父提交（git worktree）上各跑一次，比较 checkpoint 的全部张量；不同则再跑一次新代码判断是否是 box 的 run-to-run 不确定性。
3. 两个 kind 各 60 步 smoke（shard s0）：loss 有限，`thead.pt` 写出且能被 `jevdrive.bench` 的 `poses` stage 读回。
4. 全量训练后：DONE 有限，head 在 dev 行上的 ADE ≤ 1.2 m（SH30 0.56）。

## 4. 读数

全部 2 seed 按 token 平均后按 log 做 cluster bootstrap（B 10 000），主对比 FM − RG 配对；另报 RG − SH30、FM − SH30。评测只走 `jevdrive.bench`（新加的 `poses` stage 把 head 的 8 个位姿直接写成导出文件，其后与其他模型同一条打分链）。

1. navtest 全板：EPDMS 与 9 个子项；< 5°、5–20°、20–45°、> 45° 四桶 EPDMS。
2. > 45° 与 > 20°：DAC 失败率、切内角率、转不过去率（第 153 / 240 条的定义，`turn_oracle.py replay` 的同一读法）、桶 EPDMS、等弧长 |C(4 s)|（第 207 条的定义，`pt_swap.Curve`；先复现 SH30 的 0.71 / 0.88 m）。
3. navhard two-stage（G 帧）：combined / stage 1 / stage 2。
4. 样本多样性（只 FM 臂）：每个 token 16 个 noise 行的 plan，4 s 处的横向（沿日志路径法向）与 heading 的样本标准差、极差；> 45° 上极差 > 1 m 的 token 占比；样本标准差对「服务样本的 4 s 横向误差」的比（标定）；相邻样本最大间隔 / 极差（双峰指标）。对照桶 < 5°。
5. best-of-N oracle（只作分析，特权挑选）：> 20° token × 8 个 noise 行，`score-poses --traffic non_reactive`（no-EC），逐 token 取最高分对第 0 行；同时报 8 行的平均（第 0 行是否走运）。恒等闸门：第 0 行的 8 个子分与 bench 的逐 token 相同。

## 5. 判定线

- **S（sanity，先看）**：RG 的 2-seed navtest EPDMS 在 89.55 ± 0.30 内（SH30 两个 seed 89.47 / 89.63）。不在：FM − RG 仍是 loss 形式的对比，但水平不是 SH30 的，照实写；并补 `rh` 臂（同一 head、SH30 原来的 Huber，2 seed，tag `RHH-F-s{0,1}`）把 head 与 L2 尾部拆开。在：不跑 `rh`。
- **主判定（> 45° DAC 失败率，FM − RG）**：点估计 ≤ −1.0 pp 且 CI 上界 < 0 记「有效」；CI 含 0 或 |点估计| < 1.0 pp 记「无效」；点估计 ≥ +1.0 pp 且 CI 下界 > 0 记「变差」。
- **全板护栏**：navtest EPDMS FM − RG 的 CI 下界 ≥ −0.3，且 < 5° 桶点估计 ≥ −0.2。任何一条不过，即使主判定「有效」也不建议采用。
- **是否值得继续**：主判定有效 + 护栏过 + 全板 EPDMS FM − RG 点估计 ≥ +0.3。否则本支路关闭，结论写进 decisions。
- **机制读数**（不参与判定）：若「回归对两个 mode 取平均」成立，FM 的 > 45° |C(4 s)| 应比 RG 低 ≥ 0.10 m（CI 上界 < 0），且样本散布在 > 45° 上明显大于 < 5°、与误差同量级。两者都不出现 → 双侧偏移不是 mode 平均（至少不是这份 hidden state 里可分的 mode）。

## 6. 不做的事 / 限定

一种 head 结构、一组超参、不扫采样步数与 noise 尺度；不做 best-of-N 的可部署选择器；不上 HUGSIM / AlpaSim（head 没有接进闭环的服务路径）；每臂 2 seed；RG 与 SH30 的差别含两样（新 head、L2 尾部）。预算：4 次训练约 2.7 卡时（同时最多 2 个，约一张卡的显存），评测与读数为 CPU 作业。
