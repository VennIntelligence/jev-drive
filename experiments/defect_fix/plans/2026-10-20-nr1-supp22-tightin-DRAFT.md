# NR1 预登记补记 22 DRAFT（TIGHT-IN：急弯（R ≤ 9.3 m）里别跑到人的路线外侧；从 B2 出发）

**状态：草稿，未登记、未实现、未启动。** 依据是 TURN-SEP 诊断（[results/turn_sep.md](../../alpasim_proxy/results/turn_sep.md)，第 397 条）。要不要做由用户定；
做的话先把本稿改成正式登记（写死行、阈值、剂量、闸与护栏），再写代码。

## 22.0 为什么

- 第 396 条：按速度比设上限的 TURN-CAP 在 B2 能通过的弯里也会触发（得分决策的 5.9 %），离线闸 G3 不过。
- TURN-SEP 预登记了 10 个 B2 自身的状态量（速度比、弧长比、等弧长外偏、曲率亏欠、入弯滞后、横向加速度、当前外偏）和它们的组合，
  没有一个在失败与通过的急弯之间达到窗级 AUC 0.85，最好的是弧长比 0.78 / 0.73。
- 唯一过线的是道路状态：人路径前方 20 m 内的最大曲率（AUC s1 0.85、s0 0.92，JTN / RSG 0.81–0.91），阈值 0.108 /m（R ≈ 9.3 m）。
  它在得分决策上触发 1.0 % / 0.9 %，覆盖失败窗 79 % / 90 %。在这些紧弯里，B2 的状态同样分不开失败与通过（AUC 0.38–0.72）。
  所以失败就是紧弯里的基础失败率（约 30–40 %），不是某个可辨认的坏状态。
- garage 0014 在同样的急弯里**提前入弯**（比人早 1.2–1.6 m 站位）、**顶点在人路线内侧** 0.3–0.9 m、**从不跑到外侧**（最大外偏 ≈ 0）。
  B2 晚入弯 0.7–1.3 m，在每个急弯里都跑宽约 1 m，路窄的地方就出界。因此本项推的方向是「不许外偏」，速度只作为次要部分。

## 22.1 项（草稿）

- **行**：池的 S 行；自车 t0 速度 ≥ 2.0 m/s；人路径（rows meta `gpath`，1 m 间距，GI = 15 为自车站位）在 [s_e, s_e + 20] m 上：
  航向变化 ≥ 15°，且 `kmax_1m`（两段 1 m 航向差的 2 m 平均，见 `turn_sep.kmax_1m`）≥ K_TI = 0.108 /m。
- **侧向部分（主）**：plan 每个 pose 投到 `gpath` 折线上，取有符号横向偏移 d_j（左正）。σ = 转弯方向。
  外偏 o_j = −σ·(d_j − d_0)，d_0 = 自车（原点）的偏移。超过 M_TI = 0.2 m 的部分用 smooth-L1 罚，只罚外侧，内侧（garage 的线）不罚。
- **纵向部分（次）**：第 396 条的弧长上限，K 1.05 × max(arc(fut), arc(fut_reg))，只在本项的行上。

```python
def tightin_terms(poses, gpath, ok, sigma, m=M_TI, k_arc=1.05, cap=None, beta=BETA):
    """poses (n, 8, 3) ego frame; gpath (n, 86, 2) logged path at 1 m (ego frame); ok (n,) rows (speed, turn, kmax_1m >= K_TI);
    sigma (n,) +1 left / -1 right turn; cap (n, 8) the d396 arc cap or None -> (n,) term, fire mask."""
    xy = poses[..., :2]
    seg_a, seg_b = gpath[:, :-1], gpath[:, 1:]                                   # (n, 85, 2)
    ab = seg_b - seg_a
    u = ((xy[:, :, None] - seg_a[:, None]) * ab[:, None]).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-6)[:, None]
    c = seg_a[:, None] + u.clamp(0, 1)[..., None] * ab[:, None]                 # nearest points (n, 8, 85, 2)
    dist = (xy[:, :, None] - c).norm(dim=-1)
    j = dist.argmin(-1, keepdim=True)
    cross = ab[:, None, :, 0] * (xy[:, :, None, 1] - seg_a[:, None, :, 1]) - ab[:, None, :, 1] * (xy[:, :, None, 0] - seg_a[:, None, :, 0])
    d = (torch.sign(cross) * dist).gather(-1, j).squeeze(-1)                    # signed offset of each pose (n, 8), left +
    d0 = signed_offset_of_origin(gpath)                                         # (n,), same rule at (0, 0)
    out = torch.relu(-sigma[:, None] * (d - d0[:, None]) - m)                   # outward beyond m only
    v = F.smooth_l1_loss(out, torch.zeros_like(out), beta=beta, reduction="none").sum(1)
    fire = ok & (out.detach().amax(1) > 0)
    if cap is not None:                                                         # optional longitudinal part on the same rows
        va, pa = NF.stand_terms(poses, cap * (k_arc / NF.K_TC), ok, parts=True)
        v, fire = v + va, fire | pa["fire"]
    return v * ok.to(v.dtype), fire
```

## 22.2 预期触发（存档 plan，闭环决策上的估计；不是闸）

| 读数 | 得分决策触发 | 得分窗触发 | 失败急弯窗命中 | a2 目标 |
|:--|--:|--:|--:|--:|
| B2 s1 | 0.77 %（侧向 0.37 %，弧长 0.62 %） | 2.0 % | 15 / 19 | 6 / 9 |
| B2 s0 | 0.72 % | 1.8 % | 18 / 21 | 6 / 7（在急弯集内） |
| JTN s1 / s0、RSG s0 | 0.86 / 0.90 / 0.97 % | 2.2–2.5 % | 13–15 / 16–20 | |

- **满足 d396 的 1 % / 5 % 线，但原因是紧弯本身稀少（占得分决策约 1 %），不是本项挑出了坏状态。**
  紧弯里的得分窗（s1 38 窗、s0 32 窗）也会被作用，这是已知的附带作用，不是误报。
- **池**：on1_JTX 域 3 502 行（S 的 2.7 %，JTX00 抽样 1.2 %，约每 70 行 batch 0.8 行；24 个 log；右转 2 066）。
  存档 served plan 在 4 m 弧长处外偏 > 0.2 m 的有 1 273 行（36 %）。

## 22.3 登记前要定的

- 剂量：按补记 21 的定标规则，对 FREE-M 的梯度比；两档。
- 闸：G1 人路径作 plan 不触发（侧向部分按构造为 0）；G2 / G3 用上表的口径。
- pilot 护栏：沿用补记 21（held 零分 / 前撞、慢窗、k = 9 速度比、等待集起步）。另加**紧弯得分窗的零分 ≤ 对照 + 2**，以及**内切出界（向内侧出路）计数**。
  第 286 条 RBSG2 / route hinge 的教训是「只用底座自身状态会教出右转切内侧」；garage 的内侧线是在路面内的，本项只罚外侧、不奖内侧，但仍要量。
- 期望收益未量：第 395 条按观察性零分率比估的 a2 价格是 +17 / +30（board-mix / g15，s1），上限 +27 / +49。
