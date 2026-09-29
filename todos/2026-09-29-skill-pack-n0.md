# Skill pack N0：原生 plan ⊕ Hydra 打分头的切换器 + 纵向速度候选（NAVSIM）

状态: running（预登记 2026-09-29，写于任何 N0 分数之前）
主题: [../research/leaderboard-skill-pack.md](../research/leaderboard-skill-pack.md) 第 3、5 节（组件 B1 + B2）；decisions 第 40 条第 6 点（E6）、第 53 条
协调: 导航 lane [2026-09-29-op-leaderboard.md](2026-09-29-op-leaderboard.md)。N0 只读它的 navtrain 子集 plan，不碰 desire、不跑 GPU、不写它的 run dir。

## 问题

openpilot Cinque 原生 plan（GIMM-VFI 补帧，navtest 84.2 PDMS）与 E6 的 Hydra 式打分头（hold 输入的 `temporal`，84.2）逐 token 互补（取较好者 oracle 91.2）；
原生 plan 系统性偏慢（EP 丢 8.7 分，4 s 终点比 log 短 3.4 m）。一个只有两个离散参数的切换器，加上沿原路径拉长的速度候选，能不能在 navtest 上稳定超过原生 plan？

## Setup（写死）

- **调参集 T**：导航 lane 的 navtrain 子集（`runs/op_lb/lb_navtrain`，seed 0，每个 command 1000 token）去掉 E6 子分头训练用的 2 万个 token（`e6-prep/20260926-003758/tokens.txt`）后剩下的 token。
- **打分头**：E6 配方原样重拟合在 CPU 上：五个子分头（NC、DAC、EP、TTC、C）用 E6 的 2 万 token 与 E6 选出的 λ；模仿项 `cls ego` → `cls_late` 用 E6 的 λ，
  但拟合行**去掉 T 的 token**（E6 原来用全部 navtrain，T 对它是样本内）；聚合权重固定为 E6 选出的 (w_im, w_mul, w_TTC, w_EP, w_C) = (0.1, 1, 1, 2, 0)，不再调。
  sanity：navtest 上重拟合打分头的选择与 E6 存档选择同 anchor 的比例（预期 ≥ 0.9，只报告，不设门）。
- **原生候选**：Cinque `none`、GIMM 补帧、base adapter（navtrain：lane 的 `plans/gimm@cinque.npz` 经 op_interp 的 `adapt` 导出；navtest：`op_interp/navfull` 的 84.2 那份 pose 文件）。
  速度候选 n_s：沿原生 plan 的路径按弧长拉长 s 倍（s ∈ {1.00, 1.05, 1.10, 1.15}；位置按弧长插值，超出末点沿末段方向外推，heading 同步插值），时间轴不变。
- **切换规则**：S_k = E6 的对数分加权和；Hydra 选 k* = argmax S_k。原生候选 n 的分 = 它最近 anchor（8 个点 xy 最大绝对差最小）的 S。
  D = S_{k*} − S_{a(n)} ≥ 0；D ≤ δ 输出 n，否则输出 anchor k*。
- **两个参数**（只在 T 上按平均 PDMS 选；并列取更靠前的格）：
  - 模式 m ∈ {s = 1.00, s = 1.05, s = 1.10, s = 1.15, auto}，auto = 四个 n_s 里 S_{a(n_s)} 最大者（并列取小 s）；
  - δ ∈ {−1（永远 Hydra）, 0, 0.1, 0.25, 0.5, 1, 2, 4, 8, ∞（永远原生）}。
  网格 5 × 10 = 50 格。T 上需要的 devkit 分：n_1.00 / n_1.05 / n_1.10 / n_1.15 / k* 五条，每条一次 v1.1 打分（lane 的 `v1_navtrain_oplb` metric cache）。
- **navtest**：选定的 (m*, δ*) 在 12 146 个 token 上生成一个 pose 文件，**只打一次分**（v1.1 PDMS）；不在 navtest 上看任何别的臂。navhard 不在 N0 范围。

## 判读（写死）

1. **成立**：navtest 上 N0 − 原生 plan（84.2 那一跑，逐 token 配对，10 000 次 token bootstrap）的 PDMS Δ 95% CI 下界 > 0。
2. 另报（描述，不设门）：N0 − E6 Hydra 存档（84.2）的配对 Δ；五个子分；按 command（直行 / 左 / 右 / 起步）的 PDMS；输出原生候选的 token 比例；
   选中 s > 1 的比例；EP 超过 human 的 token 比例（human 用 devkit 的 human agent 在同一 split 上的已有分数，若有）。
3. 若 (m*, δ*) 落在网格边缘（δ* = −1 或 ∞），照样上 navtest，但写明「切换器退化为单一来源」。
4. 预期（写于分数之前）：T 上 auto 模式、δ 在 0.5–2；navtest +2 到 +4（研究文档第 3 节的推测）。

## 资源

CPU only，`sch_table` 行 `skill-pack-n0`，cores 155–167（13 核），nice 19；打分头重拟合 CPU 估计 < 30 min，六次 devkit 打分每次几分钟。GPU 0。
run dir：box `$DATA_DIR/runs/skill_pack/n0/`；代码 `jevdrive/skill_pack_n0.py`，链式脚本 `scripts/skill_pack_n0.sh`（tmux `jev`，DONE / ERROR / STATUS）。

## 步骤

- [ ] 预登记（本文件）提交
- [ ] navtrain 原生候选导出 + 速度拉长；打分头 CPU 重拟合；T 与 navtest 上的候选
- [ ] T 上 5 条候选的 devkit 打分；网格选 (m*, δ*)
- [ ] navtest 一次打分；报告；写回研究文档与 decisions

## 结果

跑完再填。
