# P2 追平读数（WA-JEPA vs 当前谱系）预登记

2026-10-22 21:55 写，HUGSIM 全量读数出来之前（只看过单场景冒烟：WA-JEPA 0.881、P0 0.059 与历史逐位相同）。论文骨架 research/paper-outline 3.2 / 五 P2，决策 138、144、148、170、452。

## 问题

当前谱系（Cinque 382M + 路线记忆 + S 行 / DAgger 训练）在不污染的榜上离 WA-JEPA 多远；同一 runner、同一控制器、同一天配对重读。

## 驱动与榜

- HUGSIM 64（`all64`，preset `exam` = PR #57 控制器树 fixed，WA-JEPA 只支持这个 preset），全部 `--repeat p2`，同一 commit 950003af 起。
  - 参照：WA-JEPA（出厂 checkpoint，自带 client，bf16）。
  - 候选：nuPlan B2 = `STANDFMXH-L18-s0` / `-s1`（主对，两 seed 平均）；PAI 最好的两个单 checkpoint `R6DAG-L18-s1`、`R8K-L18-s2`（第 443 条）；PAI 已提交候选的 checkpoint `NTV0-L18-s0`。
  - 无身体图式参照：`RB0-F-s0` / `-s1`（无 S、无 on-policy 行的路线记忆底座）。
  - 旧底座锚点：`P0`（出厂 Cinque 走 parity 通路，对第 138 条）、`SH30-F-s0` / `-s1`（对第 170 条）。
  - 重复噪声：WA-JEPA 与 `STANDFMXH-L18-s1` 各加一次 `--repeat p2b`。
- 路线记忆：HUGSIM 没有 AlpaSim 的 route 消息，路线记忆 checkpoint 喂空记忆（训练里 25 % 行就是空记忆，在分布内）；WA-JEPA 同样只拿 command。这是输入对等的读法，不是我们模型的最好读法。
- AlpaSim：不新跑。用已存读数：WA-JEPA 10-12 held 460 @18（`alpasim_long/fam/wajepa0-3`，eval / 场景配置与我们的 held 460 读数相同，只差并发）对候选已存的 held 460 @18 读数，按 log 聚类配对。注意 AlpaSim nuPlan 是我们谱系的训练域（S 行来自 train 1031 场景，held 460 不训），对 WA-JEPA 是域外，只作旁证。
- 不报：navtest / navhard（S 行来自 navtest log，且 navsim 通路不服务路线记忆分支；registry 已把这些 checkpoint 设为 HUGSIM-only）；WOD（数据已删）；B2D（关闭的线）。

## 读出线（先写死）

主指标 HD-Score，配对差 = 驱动 − 参照，按场景（scene，同一场景的难度变体一起重采样）聚类 bootstrap 95 % CI（B 10000）。

1. 差距（候选 vs WA-JEPA）：CI 下界 > 0 → 「超过」；CI 上界 < 0 → 「落后」；否则若 CI 下界 ≥ −0.05 → 「追平（非劣，δ = 0.05）」，下界 < −0.05 → 「分不开但不能说追平」。主对是 B2 两 seed 平均 vs WA-JEPA；PAI checkpoint 和 NTV0 是次要行。
2. 身体图式贡献（B2 两 seed 平均 − RB0-F 两 seed 平均）：CI 下界 > 0 → HUGSIM 上有贡献；CI 跨 0 → 「HUGSIM 上看不出」；上界 < 0 → 有害。B2 不只加了 S（还有 STAND 行、FREE-M 等），写作「当前配方相对无 S 底座」，不单归 S。
3. 分项：RC、NC、DAC、TTC、COM 的均值差，终局类（complete / 前景碰撞 / 背景碰撞 / off_route / spin / 卡死 max_steps / 起步停滞）计数，写在表里，不另设线。
4. 噪声：seed 散布（B2 s0 vs s1，RB0 s0 vs s1）与重复散布（p2 vs p2b，WA-JEPA 与 B2-s1）。若某个结论的差小于重复散布的 2 倍，降为弱。
5. 锚点：P0 − WA-JEPA 应接近第 138 条的 −0.173；SH30 应接近 0.439（第 170 条）。偏离 > 0.05 则先查 harness 再下结论。
6. AlpaSim held 460：同样的 (1)(2) 线，单位 = 场景，按 log 聚类。

## 成本

每个驱动 64 场景约 15 卡分钟；13 个读约 3–4 卡时；输出只留 units / summary / 表（ply 每集约 54 MB，跑完由 bench 收）。
