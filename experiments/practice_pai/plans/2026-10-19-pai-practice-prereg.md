# PRACTICE-PAI 预登记：PAI 专用练习池，小池 pilot + 闭环闸（2026-10-19，写于任何输出之前）

练习计划第 3 项（[research/next-round/practice-plan.md](../../../research/next-round/practice-plan.md)）。另两条并行：第 1 项（trainval 抽样，优先级 12/13）、
第 2 项（nuPlan 换起点，`experiments/practice_nuplan/`）。本条优先级 11。

## 问题

RSG-L18-s0（PAI 部署模型）在 PAI 自己的场景上、按 PAI 提交 serve 滚出来的状态做 S 行（只带后果 hinge），短训 1 000 步后，PAI 281 闭环上是否
**不比**同样 1 000 步但没有这批行的对照差？过闸才值得为 1 761 个 curated-train 场景下载 3.2 TB 并全量滚。第 333 条的 PAI 行（15 万行，含本池同样的
RSG-L18-s0 × train160）只过了离线闸、从没进闭环，这次直接用闭环判。

## 场景（第 1 步）：为什么是 160 个，不是 200 个

- box 上 `datasets/nurec/` 的 1 272 个 `.usdz` 只有 **441 个实体文件**（`all-usdzs/`，753 GB），另 830 个是 `scenesets/` 里指向它们的符号链接、1 个是半截临时文件；
  441 个实体 = curated val 441 = test281 + train160。**curated-train 1 761 个场景一个都不在盘上**；Tokyo 也只有同样的 441 个。
- 下载：hf-mirror 今晚限流（box 合计 ≈ 3 MB/s，持续 429，共享断路器 limit 2；同一镜像上另有 navsim trainval 相机下载在跑，134 / 183 包）。
  200 × 1.84 GB = 368 GB ≈ 34 h，今晚做不完；再开下载还会挤占那条线。**今晚不开下载。**
- 所以 pilot 用盘上唯一可用的训练场景：**train160**（第 3.1 节污染规则：可用；与 281 交集 0）。登记为 `physical_ai_av/practice-pai@v1:8e693482a597`（160，= train160）。
  它是 curated val 441 的一部分（另一部分就是 281 考题），所以不能与 `curated-val` 不相交；已核：与 `physical_ai_av/test281@v1:4c6b4371a40f` 不相交、
  与 `curated-train@v1:08d0822165c1` 不相交、`test281 ∪ practice-pai = curated-val@v1:0eb692d91c4f`，`hard_exam` / `control` / `exam_turn26` 子集都不在其中（`scripts/pp.py split`）。
- 后果：本池与 GAP 拟合（第 330 条）、第 333 条用的是同一批 160 个场景，不是「新场景」。pilot 只回答「PAI 状态的 S 行在闭环里是帮是害」，不回答「新场景多样性」。

## 初始化：RSG-L18-s0，不是 JTN

- RSG-L18-s0 是 PAI 部署模型（+ g10 + ACC + GAP，私榜 PCS 2092）；练的是它自己在提交 serve 下的状态，参照读数（`acc_gap/read/s2`）已存，serve 完全相同。
- JTN + ACC + GAP 方向上更好（+0.023 [−0.008, +0.057]，第 359 条），但 1 seed、CI 含 0；JTN 的配方带 trainval 起步 / 刹车模仿行，第 363 条刚定位为起步偏置的来源，
  第 1 项今晚就在改它。在 JTN 上练会把第 1 项的待定改动混进来。RSG 的配方简单（只有 S 行，无 trainval 行、无横向栅格），一处变量。

## 滚动与行（第 3 步）

- RSG-L18-s0，PAI 探针 serve 原样：`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`，`controller.gains.rel_front_steering_angle_weight=10`，`PAI_DUMP=100000`
  （帧 dump），`KEEP_ASL=1`，CONC 4，ALPASIM_FAST=1（默认）。`pai_box.sh submit`，每个 chunk 一个 pool 作业（5 个，第 333 条同一份 32 场景 chunk 表，
  所以逐场景分数可与第 333 条的同一滚动比，作 serve 路径跨代码版本的复现检查，描述性）。
- 行：`onp_rows.py rows / cold` → `onp_cache.py build` → `rf_pai_pool.py --name PPAI160`（S 行：车 hinge + route 标签；PAI 无道路栅格，道路 hinge 在这些行上关）。
  RSG 的配方不用 `--rm-anchor`，所以不建 `.a` 锚定特征。
- 输入路径检查 R1（第 333 条补记 1 的线）：在池输入上重算 RSG-L18-s0 的 plan，对 rollout 记录的 `poses_model`，8 点（≤ 4 s）；中位行最大差 ≤ 0.02 m 且超过
  max(0.03 m, 4 fp16 ulp) 的行 ≤ 10 %，错模型（RB0-F-s1）超出 ≥ 90 %。不过 → 链停（不训练）。
- 合池：`rf_pait.py merge --pools on1_L18RB0:s:self on1_PPAI160:p:self --out PPM160`（RSG 自己的池在前，PAI 行在后）。

## 训练（第 4 步）：pilot 与对照，只差 `--on-weight`

```
rm_train.py train --tag PPAIP-L18-s0 / PPAIC-L18-s0 --init RSG-L18-s0 --allow-init --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10
  --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --lr-scale 0.1 --on-k 13 --on-mode self --on on1_PPM160
  --on-weight pool=p:<W> --steps 1000 --seed 0
```

- pilot：W = n_s / n_p，即每批 13 个池行里期望一半是 PAI 行（第 333 条 b 格的做法：PAI 行没有道路项，均匀抽会稀释道路 hinge）。
- 对照：W = 0；PAI 行在池尾、其余权重相等时，抽样与只用 on1_L18RB0 逐位相同（rm_train 文档的权重 0 规则）。
- 其余（init、seed、步数、学习率、on-k、所有 flag）相同；`pp.sh dry` 打印两条命令并 diff，只允许 `--on-weight` 不同（输出存 results/）。
- 剂量说明：1 000 步 × 13 = 13 000 次池抽取，PAI 约 6 500 次，对约 2.5 万 PAI 行约 0.25 遍。剂量小：闸不过时只说明「这个剂量下没有可见好处」，弱证据。

## 闸（第 4 步，读之前写定）

- 读：两臂各读 PAI 281 的 **4 个固定 chunk：c000、c002、c004、c006**（`acc_gap/read/s2/chunks/` 的偶数号，36 × 4 = 144 个场景；规则写于读之前，按编号取偶数，
  不看内容），PAI 探针 serve（同上，无 dump），CONC 4，`CHUNKS_FROM` 用同一份 chunk 文件。
- 场景集合：两臂都打出分、且无 driver RPC 错误的场景（第 327 条规则）。
- **线（两条都过才 PASS）**：(1) pilot 零分数 ≤ 对照零分数 + 2；(2) 配对均值 pilot − 对照 ≥ 0（报场景聚类 bootstrap 95 % CI，10 000 次，seed 0；
  PAI clip 没有 log id，每个场景是自己的簇）。
- 参照（只报告，不判）：存档读数 `$DATA_DIR/runs/route_follow/acc_gap/read/s2`（RSG-L18-s0 + ACC + GAP，第 330 条阶段 2）。serve 与 chunk 完全相同
  （config.txt：同一 SERVE / WIZ / CONC / ALPASIM_FAST；同一份 chunk 文件）；代码 94711ca9 → 现在：`pai_driver.py` / `pai_core.py` / `sh30_core.py` / `rmem.py`
  的改动都是默认关的开关（JEV_DESIRE；锚定布局只在 sidecar 写了 `anchor` 时开，RSG-L18-s0 没写），不重读。两臂对参照的配对差和零分一并报告。
- 闸在队列里：CPU 作业（`--vram 0.1`）在 8 个读数之后跑 `pp.py gate`，写 `$DATA_DIR/runs/practice_pai/gate/{gate.json, gate.md, VERDICT}` 和
  `pai281_4ch.md`（pai281.py 的完整分类表，pilot / 对照 / 参照三列）。
- PASS → 第 5 步：全量估算（下面）并分阶段 1 → ~10 → 全部；今晚只排盘上已有的部分（= 0 个 curated-train 场景，所以今晚不排）。
  FAIL → 停第 3 项，写结论（第 333 条之外的第二个反证）。

## 全量估算（第 5 步，写在读之前）

- curated-train 1 761 个场景：盘上 0、缺 1 761；每个 1.84 GB，合计 ≈ 3.24 TB，盘上空余 1.3 TB → 必须分批「下 → 滚 → 建行 → 删 usdz」（每批 ≤ 约 500 个、0.9 TB）。
- 下载是瓶颈：hf-mirror 不限流时两个场景并行 6–15 MB/s（docs/alpasim.md）→ 60–150 h；今晚的 3 MB/s → 约 300 h。
- 滚动：第 333 条同类 chunk（32 个场景、CONC 4、22 GB）19.5 min → 55 个 chunk ≈ 18 作业小时，每卡 3 个 ≈ 6 卡时；建行（rows / cache / pool / R1）每 160 个场景约 15 min
  → 约 3 h（CPU 为主，可与滚动重叠）。产出约 1 761 × 160 ≈ 28 万 PAI 行。
- 本 pilot 的花费预计：滚动 5 × 0.33 = 1.6 作业小时，读数 8 × 约 0.4 = 3.2 作业小时，训练 2 × 0.1，其余 < 0.5；墙钟约 1.5–2 h（box 空闲时）。

## 链（`scripts/pp.sh chain`，一次全部入队，优先级 11）

roll ×5 → rows → cache → pool → fwd ×2 → r1 → (clean, merge) → train pilot / control → subreads（读数要求 checkpoint 存在才能提交，所以由这个队内作业提交 8 个读数
和闸）→ gate。作业 id 记在 `$DATA_DIR/runs/practice_pai/jobs.txt` 与 `stages.txt`。
