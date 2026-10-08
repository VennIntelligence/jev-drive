# op_parity sh30-crossfit 预登记：navtrain 转弯 token 的 held-out SH30 plan 与 simulator 标签（2026-10-08，任何本线新分数读数之前写定）

## 问题

第 178 条：navtest > 20° 转弯 token 上，SH30 plan 附近 33 条轨迹的小族里有好得多的轨迹（best-of-19 特权上限 +12.20）。下一阶段要在 navtrain 转弯 token 上用
simulator 分数训练 selector / 重标 target。nt-cache（results/nt_cache.md）已建好 28 323 个 navtrain 转弯 token 的 v2 metric cache，并给 SH30 的 plan 打了分，
但**那些 plan 是 in-sample**：SH30 训练在 op-parity-full-train 上，27 892 / 28 323 转弯 token 在其中。in-sample DAC 失败率 4.5%、no-EC EPDMS 88.7，
navtest > 20° 是 7.31% 与 83.7。按 in-sample plan 造的标签，失败更少、也不同于模型测试时的失败。

本线：用 K 折按 log 不相交训练 SH30 配方，让每个 navtrain 转弯 token 的 plan 来自没见过它那条 log 的模型，打分；回答 4.5% 对 7.31% 的差是 in-sample 造成的，
还是 navtrain 转弯本来就容易，还是折模型训练数据变少所致；之后在 held-out plan 周围建 33 候选族并打分，给出 navtrain 上的特权上限表。不训练任何基于这些标签的东西。

## 设计

**K = 5。** 理由：(1) 一次完整 SH30 训练 ≈ 40 min 一张卡（seed 0 实测 2 399 s，步数固定 10 000 × 128，与数据量无关），K 次训练 = 3.3 card-h，加上 navtest 与 navtrain 出 plan，
总预算约 5 card-h；K = 10 要 6.7 card-h 超预算，K = 5 是预算内最大的折数。(2) K = 5 时每个折模型训练在 op-parity-full-train 的 ~80%（约 80–82.5k 个 token，SH30 是 101.5k），
数据量损失小于 K = 3、4。(3) 一个 seed（seed 0），预算不留第二个 seed 的余量；SH30 两个 seed 差 0.16 EPDMS。

**折的构造（`jevdrive.data.splits`，按 log）。** navtrain 共 1 192 个 log（103 288 token）。fold(log) = `int(sha256("cf5|" + log), 16) % 5`。登记 10 个 unit = log 的 split：
`navsim/op-parity-cf5f{j}-dev`（折 j 的所有 log，即折模型 j 的 held-out）与 `navsim/op-parity-cf5f{j}-train`（其余 log 去掉 op-parity-full 的 dev log，即 sha256(log) % 50 == 0 的 log）。
折模型 j 的训练 token 恰为 op-parity-full-train 去掉折 j 的 token（`sh30_crossfit.py folds` 里断言）。op-parity-full 原本的 1 789 个 dev token（431 个转弯）不在任何折模型的训练集里，
按哈希落在各折，其 plan 同样来自对应折模型（对它们 SH30 本身也是 held-out）。`pp_train.split_rows` 加了 unit = log 的分支（行为对已有 split 不变）。
实际折：训练 token 80 481 / 82 493 / 82 513 / 80 946 / 79 563，held-out token 21 438 / 19 304 / 19 095 / 21 168 / 22 283，held-out 转弯 token 5 696 / 5 580 / 5 532 / 5 594 / 5 921（共 28 323）。

**配方。** 与第 170 条完全一致：`pp_train.py --arm P2 --frames warp --host --data navtrain_full.s{0..11}of12 --steps 10000 --batch 128 --warmup 300 --eval-every 1000 --hinge-lam 30 --hinge-margin 0.5 --seed 0`，
只换 `--split navsim/op-parity-cf5f{j}`。模型名 `CF5f{j}-F-s0`。dev 评估集 = 折 j 的 held-out（约 2 万 token，只算 ADE / drift，不参与选模）。

**流程。** 一条自走链（`sh30_crossfit_chain.sh`，tmux `jev:sh30-crossfit`，STATUS / DONE / ERROR 在 `$DATA_DIR/runs/op_parity/sh30_crossfit/chain/`）：折 split 检查 → 折 0 的 3 步 preflight 冒烟 →
折 0 单独起、量吞吐 → 折 1–4 起（均经 pool，各一个 job）→ 每折训练完：`jevdrive.bench run --bench navtest`，navtrain 12 个 shard 的 plans + nav-export（nt_labels 的阶段，换模型）→
全部折完：闸门 → held-out plan 装配 → `score-poses --traffic non_reactive --mcache v2_navtrain`（key `h`，28 323 token）→ 诊断。步骤 4 只在诊断写完、闸门通过后跑。

**成本停止线。** 折 0 单独跑时看吞吐（it/s）与 VRAM：折 0 的外推墙钟 > 70 min（SH30 实测 40 min 的 1.75 倍）→ 停下报告。总账（card-h、core-h）按 `cl usage` 与 job 的 wall 记；
预算 ≈ 5 card-h、40 core-h，实测外推超过 1.5 倍（7.5 card-h / 60 core-h）就停下报告。

## 闸门（折配方是否复现 SH30）

读 bench navtest 全量 12 146 token 的 EPDMS（含 EC，与第 170 条同口径，SH30 两 seed 均值 89.55，P2H10 88.67）。

- **通过**：5 个折模型的 navtest EPDMS 均值 ≥ 89.05（SH30 均值 − 0.5：允许少 ~20% 训练数据的损失，约三倍的 seed 间差），**且**每个折模型 ≥ 88.67（不低于没有强 hinge 的 P2H10）。
- 不通过：诊断照读照报，但标注「折配方没有复现 SH30」，不跑步骤 4，停下报告。
- 另做训练健全性检查（`pp_full_check.py train --tag CF5f{j}-F-s0`，如在 SH30 链上）。

## 诊断（预登记读数）

对象：|dyaw| ≥ 20° 的 navtrain 转弯 token（28 323），分桶 20–45°（17 550）、≥ 45°（10 773），navtest 对应 3 154 个 token（bench 的 20–45 / > 45 档）。三个量，全部 no-EC 口径
（`score-poses --traffic non_reactive`；navtest 一侧由 bench 逐 token 子分重算，EC 去掉）：**DAC 失败率**（DAC = 0，主读数）、NC 失败率（NC = 0）、no-EC EPDMS × 100。

四个格子：
- **B** = SH30 在 navtrain 转弯上的 in-sample plan（nt-cache，两 seed 均值）；
- **A** = 折模型在 navtrain 转弯上的 held-out plan（每 token 一个模型）；
- **D** = 5 个折模型在 navtest 转弯上（逐 token 对 5 个模型取均值，类比 SH30 的 2 seed 均值）；
- **C** = SH30 在 navtest 转弯上（两 seed 均值；DAC 失败 7.31%、no-EC 83.7，第 178 条）。

差 G = C − B（4.5% 对 7.31%，同号方向：navtest 更差）按望远镜式恒等分三项：
mem = A − B（in-sample 效应：同一批 token 上，见过 log 与没见过）、dom = D − A（navtrain 转弯对 navtest 转弯的难度差，同一类模型）、mod = C − D（折模型比完整 SH30 弱的效应，预期 ≤ 0）。
CI：按 log 聚类的 bootstrap（navtrain 与 navtest 各自独立重采样 log，B = 10 000，seed 0，比值之和；A 与 B 同在一批 token 上、C 与 D 同在一批 token 上，所以 mem 与 mod 是配对差）。

**判定（只用 ≥ 20° 合并桶的 DAC 失败率；固定，看到分数后不改）：**
1. mem 与 dom 都是 CI 下界 > 0 才算「显著正」。两者都显著正：mem / (mem + dom) ≥ 0.67 → 判「in-sample 解释」；≤ 0.33 → 判「navtrain 转弯本来更容易」；之间 → 判「两者都有」。
2. 只有一个显著正，判那一个。两个都不显著 → 「未分辨」。
3. 「held-out 复现 navtest 水平」= dom 的 CI 含 0（或 |dom| < 0.5 pp）：即同一类模型在 navtrain 与 navtest 转弯上的失败率没有可分辨的差。
4. 若 |mod| > 0.25 |G|，标注「折模型变弱」是 A 与 B 差的一部分来源，并并排报告 mod。
分桶（20–45、≥ 45）与 NC / EPDMS 的同样分解只描述，不进判定。另报：每折的 A（held-out）、该折模型在 navtest 转弯上的值、该折 token 上的 B；431 个 op-parity-full dev 转弯 token 上的 A 对 B（SH30 本身 held-out 的那批）。

## 步骤 4：held-out plan 周围的 33 候选族（诊断写完之后）

`turn_ceiling.py candidates / transform / families` 原样（不改），对 28 323 个 held-out plan 建 c00–c32（c00 = 恒等 = held-out plan 本身，逐位相同），打
`score-poses --traffic non_reactive --mcache v2_navtrain`。每个 token 一条 plan（没有 seed 轴）。

- **stage 0**：300 token（20–45 与 ≥ 45 各 150，seed 0）× 33 key：恒等候选（c00）的 NC / DAC / DDC / TLC / TTC / LK / HC / EP / no-EC 分与第一次打的 key `h` 逐 token 完全相等（EP 与分数绝对差 ≤ 1e-6）才继续。量 core-s / token。
- **成本阶梯**：全量成本 = 实测 core-s / token × 28 323；预算余量 = 40 core-h − 已用（训练、plans、导出、打分按 `cl usage` 记）。取第一个外推 ≤ 余量的族：F33（33 key）→ F27 → F19（只剩判线用的族）；F19 也超余量 1.5 倍则停下报告，不打。削了什么写进结果。
- **报告**（第 178 条同款，无 seed 轴）：各族 best-of-K 对恒等的 no-EC EPDMS 增益，桶 ≥ 20° / 20–45 / ≥ 45 / 左 / 右，log 聚类 bootstrap CI；自由度（只偏移 O3、只曲率 K3、只速度 V3、任一单轴 F7、联合 F27 − F7、O3 − K3 / O3 − V3 / K3 − V3）；
  恒等与 oracle 的 DAC / NC 失败率；best of {恒等, 速度 × 0.8, 速度 × 0.6}（nt-cache 的 in-sample 同一数：+4.89 / +5.00）。第 178 条的 +4.0 线在此不作停止线（本线不是判线读数）。
  best-of-K 是特权上限，不是方法；速度 × 0.6 之类放慢的候选会把出界 / 碰撞推到 4 s 视野之外（第 178 条限定同样适用）。

## 不做

不训练 selector / 重标 target；不改 turn_ceiling.py、nt-cache 的输出、v2_navtrain cache、SH30-F-s0/s1 的 checkpoint；不碰其他 lane 的文件与进程；不手选卡与端口。

## 产物

`experiments/op_parity/results/sh30_crossfit.md`（表 + `figs/sh30_crossfit/`），新的 research/decisions 条，脚本 `scripts/sh30_crossfit.py`、`scripts/sh30_crossfit_chain.sh`，
标签在盒子上的 `$DATA_DIR/runs/op_parity/sh30_crossfit/labels/`（poses.npz、score.csv、fold_of_token.csv、poses_f33.npz、cscore_*.csv）。
