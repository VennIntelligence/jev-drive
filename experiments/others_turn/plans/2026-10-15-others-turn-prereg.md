# OTHERS-TURN 预登记：第三方策略在 held 24 个转弯上转不转（2026-10-15，结果留空）

问题：WA-JEPA、AlpaSim 官方 sample（LTF、DiffusionDrive）、主办方 py123d-garage（latent TransFuser resnet34）这些「没做转弯专项训练」的策略，
在我们同一把尺子（`rf_follow.py`，held 452 / 18 决策，24 个转弯）上闭环完成几个弯？对照 RSG-L18-s0（16 / 24，零分 26，均分 0.881，同一 held 读）。
只测量，不训练，不改它们的代码（third-party 原样跑）。research 笔记：`research/others-turn.md`。

## 数据与运行

- M1 WA-JEPA：已存的 `$DATA_DIR/runs/alpasim_long/fam/wajepa{0..3}`（alpasim_long 补记 3，WAJ_AMP=0，c0b_chain.OVERRIDES + after160，18 决策）；
  driver 自己记了每个决策的 anchor / t0 / plan（drive.jsonl），rf_follow 直接读，0 GPU。
- M2 LTF / DD / garage：存档 run 没有 anchor、得分非零的 rollout.asl 已删 → 只在 24 个转弯 scene（`results/turn24.txt`，= RSG-L18-s0@18 读出的 24 个，
  APP0C / APP2W05 同一组）上按 al_families.sh 的同一命令重跑（pool，`run.sh <dir> ltf|dd|garage`，保留 rollout.asl），
  anchor 从 rollout.asl 的 EGO actor_poses 在每个决策 t0 取（`scripts/ot_asl_records.py`，写一个只含 start / drive 记录的 drive.jsonl 视图目录）。
- 对照：RSG-L18-s0@18（`$DATA_DIR/runs/route_mem/read/RSG-L18-s0/c*/run`），桥接 YR10m10-F-s0@18（alpasim_long held/read_apy 的同一 run，follow_ref 读 9 / 24）。
- 零分 / 均分：ol_confirm.py held18，WA-JEPA / LTF / DD / garage 用存档的 460 run，配对 RSG-L18-s0（log 聚类 bootstrap）。

## 闸（先过闸再读）

- G1 asl 转换：在 WA-JEPA 存档里同时有 drive.jsonl 和 rollout.asl 的零分 rollout 上，asl 取的 anchor 与 driver 记的 anchor 位置差中位 ≤ 0.05 m、航向 ≤ 0.5°；
  并且对这些 rollout，用 asl 记录跑 rf_follow 的 turn / completed 与用 driver 记录的逐条一致。不过 → M2 只报零分 / 均分，不报转弯。
- G2 桥接：YR10m10-F-s0@18 读出 9 / 24 ± 1，turn 集与 RSG 的 24 个相同（不同就报差异，按各自 turn 集报）。
- G3 重跑一致：M2 重跑在这 24 个 scene 上的得分与存档 460 run 的同 scene 得分，零 / 非零一致 ≥ 20 / 24（AlpaSim 有 cm 级非确定性，d215）。

## 判读（写死）

- 「它们不经专项训练就会转」成立 = 某第三方策略完成率 ≥ RSG 的 16 / 24，且配对差 CI 下界 > −10 pp；强成立 = ≥ 20 / 24。
- 「它们也不会转」= 完成率 ≤ 12 / 24（YR10m10 水平 9 附近），或配对差 CI 上界 < 0。
- 中间 = 不确定；按 outcome（completed / event / off / short）和左右分开描述，不下结论。
- 零分 / 均分只作背景：已知 WA-JEPA 0.7525 / 93 零分（460），远差于 RSG。

## 附读（score-only，PAI）

主办方 PAI 参考 subject（alpamayo1、vavam-linear / nonlinear、alternative_1..5，各 3 rollout，441 curated val）的逐 scene 得分 / offroad / 走廊，
在我们 PAI 281 的 36 个严格转弯上：通过率（score > 0）对 RSG-L18-s0（同 36，path 0 与 gate 各报）。不是完成率（无轨迹），配置不同（主办方 run），只作方向。

## 结果

（留空）
