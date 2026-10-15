# 预登记：ON-K pilot，当前基座配方的 on-policy 行占比，2026-10-15

用户的问题（2026-10-15）：每个 batch 13 行 on-policy 行是不是太少？这个数从来没变过：它继承自 yaw-rate 行的占比 round(0.1 x 128)
（[onpol 预登记 2026-10-11](../../onpol_nuplan/plans/2026-10-11-onpol-prereg.md) "Arms"）。本 pilot 在 RSG-L18-s0 的原命令上只改
`--on-k`（以及一个控制臂的 `--ho-w`），按 lr x steps pilot 的方式读 held 460 / 18 决策并对 RSG-L18-s0 配对。
本文件在任何读数被看到之前提交并 push。补记追加在后，不就地改。

## 配方（从 box 上读出，不是重建的）

`$DATA_DIR/runs/route_mem/train-RSG-L18-s0/20261013-023026/meta.json` 的 argv（git d4864575, dirty）：

```
rm_train.py train --tag RSG-L18-s0 --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank
  --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode self
  --on on1_L18RB0 --seed 0
```

它的 log：每 batch 102 normal + 13 yaw-rate + 13 on-policy（pool on1_L18RB0，17 429 行，28 个 log）；dev ADE @0 0.5276；step 100 imit 0.7552；
382 s，18.2 GB。

## 臂

| 臂 | tag | 改动 | 每 batch 行（normal / yaw-rate / on-policy） |
|:--|:--|:--|:--|
| k13（基准，不重训） | RSG-L18-s0 | 无（存档 checkpoint 和存档 held 读数） | 102 / 13 / 13 |
| k26 | RSGk26-L18-s0 | `--on-k 26` | 89 / 13 / 26 |
| k51 | RSGk51-L18-s0 | `--on-k 51` | 64 / 13 / 51 |
| k51w1 | RSGk51w1-L18-s0 | `--on-k 51 --ho-w 1` | 64 / 13 / 51 |

其余参数与上面的命令完全一致，`--init RB0-F-s0`（只从这个 checkpoint 起）。on-policy 行从 normal 行里扣（on_train.setup：
nN = 128 - k_yr - k_ho - k_on，k_yr = round(128 x 0.1) = 13，k_ho = 0）。注意：k 变了，normal 行的 rng 流和 pool 行的 orng 流
（[seed, 0, 53]）都会和 RSG-L18-s0 不同（不再是同一行序列前缀），所以臂间差异混有 seed 量级的行序列噪声；RSG-L18-s1 对 s0 的
held 零分差 +0.0 [-4.7, +4.9]（lrsteps.md）是这个噪声的参照。

## 压力算术（读 on_train.py 的 LossesOn 和 body1/lib/loss43.py 的 hinge-only 路径得出）

`--on-mode self` 下 on-policy 行是 Losses43 的 hinge-only 行（family `on`）：agent hinge 和 road-and-lane hinge 各自在 on-policy 行上
**求和**，除以 batch 的 imitation 行数 n_imit，再乘 lam（agent_lam 10 / road_lam）和 ho_w：
`total += ho_w x (agent_lam x sum_on(agent) + road_lam x sum_on(road)) / n_imit`。
n_imit = 非 hinge-only、非 anchor、有未来轨迹的行 = normal 行里非 anchor 的部分 + yaw-rate 行（anchor 只从 normal 行里抽，
d_frac 0.25；yaw-rate 和 on-policy 行永远不是 anchor）。期望值 n_imit ≈ 0.75 nN + 13（假设训练行都有未来轨迹）。
imitation 项本身是 imitation 行上的 mean（lam_i 1），其权重不随 k 变；distill_ho 除以非 hinge-only 行数（115 / 102 / 77）。

| 臂 | k | ho_w | nN | n_imit（期望） | 每行权重 ho_w / n_imit | 每 batch 压力 ho_w x k / n_imit | 对 k13 的倍数 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| k13 | 13 | 3 | 102 | 89.5 | 0.0335 | 0.436 | 1.00 |
| k26 | 26 | 3 | 89 | 79.75 | 0.0376 | 0.978 | 2.24 |
| k51 | 51 | 3 | 64 | 61.0 | 0.0492 | 2.508 | 5.76 |
| k51w1 | 51 | 1 | 64 | 61.0 | 0.0164 | 0.836 | 1.92 |

所以 k51 的 on-policy hinge 压力是 k13 的 **5.8 倍**（不是 3.9 倍：n_imit 跟着 k 缩小）。控制臂 k51w1 把压力降到 k13 的 **1.9 倍**，
并没有压平（压平要 ho_w ≈ 0.52）。按派活要求保留 `--ho-w 1`；读法因此是：k51w1 对 k51 隔开的是 ~3 倍压力（5.8 → 1.9），
k51w1 对 k13 混有占比 x 4 和压力 x 1.9 两样。k26（2.2 倍压力）与 k51w1（1.9 倍）压力接近、占比差一倍，是占比的第二个对照。

## 读数

held 460，18 决策（rm_chain.sh `read`，6 chunk，`CL_HIST_KEY=alpasim-read4`，所有 RSG-L18 held 读数的同一路径）。
报告同 rf_lrsteps.sh `report`：`held.py` 在 460 个公共场景上（`held_raw.txt`），`ol_confirm.py` 在 452 个有 rollout 的场景上配对
（39 个 log，log 聚类 bootstrap B = 10 000），base RSG-L18-s0；`rf_follow.py`（24 个 held 转弯）。
- **主指标：** held-452 零分，对 RSG-L18-s0 的配对差和 95 % CI。
- 次要（报告，不用于判定之外的选择）：均分、at-fault 零分、碰撞 / 出路 / 走廊拆分、GT 距离、dev ADE、转弯完成率。
- 本 pilot 不读 PAI、不读公开 1491。

## 登记线（照抄 lrsteps 的口径）

1. **候选线**（seed 0，每个臂单独）：零分对 RSG-L18-s0 的 CI 上界 < 0，**或**均分对 RSG-L18-s0 的 CI 下界 > 0。达到 = "候选"，
   否则 = "持平 / 无增益"。CI 整体 > 0 的臂报告为"更差"。
2. **停止线**（决定 seed 1）：零分对 RSG-L18-s0 的 CI 下界 > +5 的臂停，不跑 seed 1；其余每个臂跑 seed 1。
3. **2 seed 读法**：每个跑了 seed 1 的臂报告 2 seed 均值对 RSG-L18（s0、s1 均值；`ol_confirm --mean`）和同 seed 对
   （armk-s1 对 RSG-L18-s1）。"复现" = 2 seed 均值上达到候选线（零分 CI 上界 < 0 或均分 CI 下界 > 0）且 s1 对同号。
   采纳与否、公开 1491 护栏由主会话决定。
4. 三个臂各比一次，95 % 下偶然出一个"候选"并不罕见；seed 1 是对它的防线。

## 已知的（不是本 pilot 的读数）

- k 从未变过：13 = round(0.1 x 128)，onpol 预登记 2026-10-11 沿用 yaw-rate 行占比。
- lr x steps pilot（同配方 3 x 3，results/lrsteps/lrsteps.md）：**持平**；最少零分 (0.03, 5000) 23 对 26，-3.0 [-7.6, +1.0]；
  短 / 弱训练更差。RSG-L18-s1 对 s0 零分 +0.0 [-4.7, +4.9]。
- 决策 286 补记：RSGall-L18（pool 行 +45 %，每 batch 仍 13 行）公开 1491 对 RSG 均分 -0.0034 [-0.0073, +0.0001]、零分
  +3.0 [-2.1, +8.6]，无增益。即"更多不同的 on-policy 行"没帮上，"每 batch 更多 on-policy 行"没测过。
- 决策 302：臂 a / b（RSG-L18 上续训）无害但未达转弯线。

## 代价估计（发起前写）

- 训练：382 s / 2500 步 + ~1 min 启动；3 个臂 ≈ 0.4 job-h（每卡一个训练，~18 GB）。
- held 读：6 chunk x ~11 min ≈ 1.1 job-h / checkpoint，3 个 ≈ 3.3 job-h；每卡 4 个读打包 ≈ 1 card-h。
- seed 1（最多 3 个臂）同样 ≈ 3.7 job-h。合计最坏 ≈ 7.4 job-h，约 2.5–3.5 card-h。
- 发起时 `cl top`：3 张卡全空闲，队列为空。pool 优先级 -1（低于默认 0、route-v2 的 4、route_follow 的 14），只填空闲卡，不抢占任何人，也不让任何别的 lane 的
  作业排在后面。

## 分步发起与停止规则

1. `first`：先只提交 k51 的训练（最极端的臂）和它的 6 个读 chunk（`--after`）。在 `rest` 之前检查训练 log：
   每 batch 行 = "64 normal, 13 yaw-rate, 51 on-policy"；pool 行 17 429；init 的 dev ADE @0 = 0.5276；step 100 损失有限，imit 与
   0.7552 在 ~5 % 内。任何一项不符：取消它的读 chunk，不再提交任何东西，报告（训练器拒绝或行流出错时不改代码硬上）。
   （这就是"dry check"：训练器的 `draws` 子命令只报 pool 抽样份额，不报 batch 账，batch 账只在 train 的第一行 log 里。）
2. `rest`：k26、k51w1 的训练和它们的读 chunk，加一个 report 作业 `--after` 全部读 chunk。
3. seed-0 报告出来、按停止线定下哪些臂跑 seed 1 后：`seed1 <arms>`，再一个 2 seed report。
4. 训练失败（非有限损失、崩溃）或读 chunk 失败：立刻报告；读 chunk 可重交一次，训练不静默重试。

链：[scripts/rf_onk.sh](../scripts/rf_onk.sh)（`first | rest | seed1 | report | train`）。
结果：`results/onk/`（`onk.md` 表和按上面的线给的结论，`held18/confirm.md`，`held_raw.txt`，`follow.md`）。
