# 等待能力的训练方案（设计稿，未启动；2026-10-16，接第 326 条）

给用户批改的方案。只改纵向（沿 plan 自己路径的弧长），不碰形状；一个联合臂 + 一个对照；先 1 000 步剂量 pilot。

## 1. 为什么这么设计（读数里来的约束）

- 第 326 条：出厂 Cinque（P0）在人等的帧上 0.978 等、人走的帧上 0.334 等（D = 0.64，读得出该不该走）；**第一个 adapter 阶段 P2-F 就把它降到 0.012**，之后各阶段不再变；
  没有任何阶段横向绕行（W_L + S_L 绕行 ≤ 0.004）。病是「静止即起步」，不是「绕过去」。
- 来源是数据：navtrain_full 103 288 行里静止 8 421 行，人 4 s 不动只有 82 行（1 %），2 s 内起步 7 505 行；adapter 的只读 ego 常数项把「静止 → 起步」学成先验（第 162、167、218 条）。
- 第 174 条：只把「静止关 adapter」训进去（P2HGA）不赚，起步先验搬进 plan 权重；navtrain + WOD 混合（MX）学成两套按域行为。所以等待行必须**同域**（nuPlan），
  而且静止行里「等」和「走」要配平，让只读 ego 的那部分拿不到信息、只能靠画面分（P0 的画面通路已经会分）。
- 第 303 条补记 5、第 286、321 条：自身状态上的形状项伤闭环；速度项可以复用。第 315 条：开环纵向是自身运动的趋势跟随者，S 行若喂人的减速历史，模型照历史就能满足标签。

## 2. 数据

- **不能用** NAVSIM test 的 147 个 log（公开 1491 与 held 的 scene、以及 waitcap_test 评估行都在这里）。
- trainval 的帧在盒上只下了 navtrain 用到的那些：人在等、带 4 帧历史的只有 26 帧（盘点：trainval 723 019 帧，CAM_F0 在盘 152 495，等待帧 228 215）。
- **第 0 步 = 下载** OpenScene trainval 相机归档的一个子集（每个归档是整 log 的 8 相机，不能只取 CAM_F0）：按 test 的 127.9 GB / 75 122 帧（1.7 MB / 帧）估，
  全量 trainval ≈ 1.2 TB（约 21 h，16 MB/s；盘剩 1.5 TB）；取约 10 % 的 log ≈ 120 GB、约 2 h，就有约 2 万个等待帧。先拉归档索引核对每个归档覆盖哪些 log，只下覆盖
  等待帧最多、且不在任何 held / 公开集合里的那些。
- 行：同一个 `wc.py select`（格子定义不变）跑在下载到的 trainval log 上 → `idx_waitcap_train`：W_L、W_N（等）、G_L、G_N（起步）、S_L、S_N（停下），
  同 log 同格隔 2 s，每格上限 5 000。index（navsim2）→ `pp_prep.py --frames warp --no-side` → `rm_build.py nav`，与本次评估行同一条链（约 0.3 卡时）。
- 打包成 on-policy 池格式（`cache/on1_WAIT/`，tab.npz 多存 `s_h` (n, 8) 人的弧长、`cell`、`mode = "lon"`），这样复用 `--on` 的合并池、`--on-weight` 与 weight 0 的逐位对照性质。

## 3. 损失（只有纵向）

```python
# on_train.losses_on().__call__ 里，mode == "lon" 的池行：不进 imit、不进 agent / road hinge，只加这一项
def lon_arc_loss(x, y, s_h, cell, beta=0.5):
    """x, y: (B, 8) 后轴系 plan 位姿（T.rear，0.5..4 s）；s_h: (B, 8) 人的弧长；cell: (B,) 行格子。
    plan 自己路径上的弧长 s_p 对人的 s_h；梯度只沿 plan 切向（弧长对旋转不变），曲率 / 横向不受直接作用。"""
    seg = torch.hypot(torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1])), torch.diff(y, dim=1, prepend=torch.zeros_like(y[:, :1])))
    s_p = seg.cumsum(1)
    d = s_p - s_h
    two_sided = cell_in(cell, ("W_L", "W_N", "G_L", "G_N"))     # 静止行：等要等住，走要走起来
    over_only = cell_in(cell, ("S_L", "S_N"))                    # 停下行：只罚比人走得远（d303 唯一能迁移的那一半）
    d = torch.where(over_only[:, None], torch.relu(d - 1.0), d)
    return F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta, reduction="none").sum(1)   # / n_imit x lam_lon，与其他池项同一归一
```

- S 行喂**匀速历史**（rf_lonol 的 F：vx 保留，vy = ax = ay = 0，历史位姿按 v0 重排；`vhist.respace`），标签仍是人的减速弧长：去掉「照历史减速」的捷径，
  要停只能看画面。W / G 行的历史本来就是静止的，不改。
- 静止行抽样配平：`--on-weight cell=W_L:1,cell=W_N:1,cell=G_L:a,cell=G_N:a`，a 使「等 : 走」= 1 : 1（自然比例约 16 : 1，配平是本方案的一个可批改的选择）。

## 4. 臂

起点与配方都照 RSG-L18-s0 的原命令（RB0-F-s0 起、温和一轮 2 500 步、lr × 0.1、k 13 个池行 / batch、只 S 两个 hinge），池换成 `on1_L18RB0 ⊕ on1_WAIT`：

| 臂 | 池权重 | 说明 |
|:--|:--|:--|
| **J**（联合） | WAIT 行占 13 个池行的 q（剂量），L18RB0 行占其余 | 唯一的处理臂 |
| **C**（对照） | WAIT 行权重 0 | 按 rm_train 的 weight-0 性质，抽样与 RSG-L18 逐位相同；同时重跑以吸收 cudnn 噪声 |

- **剂量 pilot**（1 000 步，seed 0）：q ∈ {2 / 13, 5 / 13} × λ_lon 10（= `--on-lon-lam` 的量级），加 C；共 3 个 1 000 步。
- 看 pilot 选一个剂量后，J、C 各 2 seed × 2 500 步。

## 5. 判据（先登记再读；读数在 waitcap_test 上，那些 log 从不进训练）

- **pilot 闸（开环，CPU 读）**：W 上等 ≥ 0.60（RSG 0.005，P0 0.978）；G 上起步 ≥ 0.85（等 ≤ 0.15；RSG 1.00，P0 0.67）；W_L 撞上前车 ≤ 0.05（RSG 0.31）；
  S_N 超出人的中位数 ≤ C；navtrain dev ADE ≤ C + 0.01 m；行进中 dev 行 2 s 处 plan 横向相对 C 的中位差 ≤ 0.05 m（形状不动的检查）。两个剂量都不过 → 停，不做全量。
- **闭环（全量 J 对 C，2 seed）**：held 452 / 18 决策零分 ≤ C + 3；公开 1491 / 10 决策护栏：均分 ≥ C − 0.005、at-fault 零分 ≤ C + 3；公开唯一的静止 scene 与 N 层 24 个：碰撞；
  PAI 281：17 个 `cor_route_end` 的个数与停车点前 3 s 的减速意图（RSG 13 / 14 无意图），以及 PAI 全集均分（护栏 ≥ C − 0.01）。
- 能力三项（主报）：人等时等的比例、人走时走的比例、PAI stop17；分数是护栏不是目标。

## 6. 成本（卡时，估计）

| 项 | 卡时 | 依据 |
|:--|--:|:--|
| 下载 ~120 GB（无 GPU） | 0（墙钟 ~2 h） | 16 MB/s |
| 行（index / pp_prep / 路线记忆） | 0.3 | 本次 4 674 行 pp_prep 2 min |
| pilot 3 × 1 000 步 + 开环闸 | 0.2 | RSG-L18 2 500 步 382 s |
| 全量 2 臂 × 2 seed × 2 500 步 | 0.5 | 同上 |
| held 452 / 18 × 4 | ~6 | 估 |
| 公开 1491 / 10 × 4 | ~10 | d325：318 scene 0.5 卡时 |
| PAI 281 × 2（s0） | ~5 | 估 |
| **合计** | **~22** | pilot 不过只花 ~0.5 |

## 7. 风险

1. 起步迟滞：P0 在人走时也有 0.33 不走；配平后可能把 HUGSIM / AlpaSim 的起步卡住带回来（第 167 条门控：卡死 0 → 11.5）。闸里有「走 ≥ 0.85」，闭环看 progress 与卡死。
2. 第 174 条的反例：门控把起步先验搬进 plan 权重。区别在于这里有同域、配平的反例行；若 pilot 里 W 等上去而 G 走下来同样多（D 不涨），就是同一个失败，停。
3. 等待帧里有路边停车 / 上下客，人「等」是对的，但 AlpaSim 长 scene 里人起步而车停着会不会被罚（第 325 条：5.5 s 内不罚，18 决策没读过）。
4. S 行匀速历史是分布外输入（闭环里也会出现）；若它让 dev ADE 掉得多，退回只用 W / G 行。
5. 下载的归档边界与覆盖要先核；全量需要 ~1.2 TB，盘只剩 1.5 TB。
