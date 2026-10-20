# PAI 人类行的道路栅格（第 5 轮配套，2026-10-21 登记，先于任何构建 / 读数）

目的：让 PAI 训练行带上 nuPlan 行已有的道路 hinge 所需的可行驶区栅格，使现有道路 hinge（`--road-lam` / `--road-margin`，
on_train LossesOn 的 `both` / `mixed` 路径）能作用在 PAI 人类行（on1_PAIH / PAIH05 / PAIH15）上，以后也能作用在 PAI S 行上。
不训练，不新造 hinge / 损失。依据：决策 410（PAI-ROWS）、414（ROUND3）、415（D1：853 / 853 训练场景有 NuRec 矢量地图）。

## 1. 格式（与 nuPlan road.npz 逐项相同）

- 训练器读的布局 = `runs/body1/labels_road` / `<POOL>/road.npz`：每行一张 128 x 96 的有符号距离栅格（m，面内为正），0.5 m 格，
  x -8 .. 56 m（前），y -24 .. 24 m（左），行在 t0 的后轴坐标系；格心 x = -8 + 0.5 (i + 0.5)，y = -24 + 0.5 (j + 0.5)；float16，
  截断到 ±20 m。hinge 用 `lib/drivable_hinge.Hinge.margins`（8 个计划点插值到 0.1 s，Pacifica 车身 4 角，双线性采样）。
- 栅格代码走 nuPlan 的同一条路：`experiments/op_probe/scripts/opb_labels.py` 的画布（0.25 m 填充，两侧 EDT，2 x 2 平均）。
  只把它的「多边形 -> 栅格」部分抽成函数 `canvas_sdf(geom, pose)`，`sdf_raster` 调它，nuPlan 行为逐位不变（Mac 上合成多边形对比旧实现）。
- 面：NuRec 矢量地图的车道多边形 ∪ road area（评测器 offroad 的面，`rf_tax.Map`，与 D1 mapcov 同一个对象）。
- 坐标系：PAI 行的 `anchor`（行文件，场景局部坐标系的 rig 位姿 = 后轴）；`fut` / `log_fut` 就在这个 rig 系里（onp_rows `rig_n`），
  与 PAI 驱动导出的计划同一系。PAI 车 5.21 x 2.16 m，训练器 hinge 仍用 Pacifica 车身（5.18 x 2.30 m，现有代码路径，不改）。

## 2. 存储（零拷贝式）

- 拉伸行（PAIH05 / PAIH15）与其 `base` 平常行同一 t0 位姿，栅格相同：每个 (场景, k) 只存一张。
  `$DATA_DIR/runs/round5/road_pai/sdf.npy` (U, 128, 96) float16 + `keys.npz`（scene, k, anchor）。U 约 13.6 万，约 3.3 GB。
- 每个池一个小索引 `road_<pool>.npz`：tokens = 池的行名，`ok`，`ix`（行 -> sdf.npy 的行），`sdf_file`，x0 / y0 / res / layers，
  另带诊断列（t0 最小 SDF、人路径最小 SDF、人路径 hinge 值、回退原因）。合并池（on1_JTPH 及以后第 5 轮的合并池）用同一命令建索引，
  按行名里的 `paih|paih05|paih15:<scene>:<k>` 找栅格，`fut` 取该池自己的 tab。
- `lib/drivable_hinge.Hinge`：标签文件含 `ix` 时按 `ix` 读 memmap 的 `sdf_file`；不含 `ix` 的文件（现有全部）走原代码，不变。
- 盘：sdf.npy 约 3.3 GB + 压缩分片约 1 GB + 索引 < 0.2 GB + G3 三个 60 步 checkpoint；上限 10 GB。

## 3. 训练器接口（默认关）

- `--on-road-pai FILE`（on_train / rm_train）：池行道路 hinge 的第三个栅格来源，排在 navtrain 道路栅格与 `<POOL>/road.npz`
  （或 `--on-road`）之后；前两个来源已标的行不动，只有它们没标的行（PAI 行 ok 本来全 False）取这个文件。
- `--on-road-pai-w W`（默认 1）：这些行道路 hinge 项的权重（乘在每行的 hinge 值上；W = 0 时该行也不计入 `on/n_hinge`）。
- 不给 `--on-road-pai` 时代码路径与现在完全相同。

## 4. 回退规则（照 nuPlan note (x) 的思路）

nuPlan 池行：t0 车身已在道路栅格外的行改用更宽的 drivable 栅格（bd4_train note (x)）。PAI 只有一个面（评测器的面已是最宽的），
没有更宽的栅格可换，所以：

- (a) t0 车身（零计划，Pacifica 4 角）任一角 SDF < 0 -> 该行不挂 hinge（ok False）；
- (b) 人自己的标签路径（t0 + 8 个 `fut` 点，插值到 0.1 s，4 角）最小 SDF < -0.5 m -> 不挂（D1：约 10 % 的在动场景有这种段）；
- (c) 该场景没有地图 / 栅格全外 -> 不挂。
- 合并池里同一 (场景, k) 的平常行与拉伸行各按自己的 `fut` 判 (b)。

## 5. 闸（训练前必须看）

- **G1（人路径不触发）**：每个池里挂 hinge 的行中，人自己的 4 s 标签路径触发道路 hinge（`--road-margin 0.3`：164 个角点样本里
  任一 SDF < 0.3 m，即 hinge 值 > 0）的份额 **≤ 1 %**。同时报全部行口径、按速度分档，并在 nuPlan 的对应行上（on1_JTPH 的 `tv`
  部分：nuPlan trainval 人类行，`both` 模式，带道路栅格；t0 在栅格外的按 note (x) 换 drivable）量同一个数作参照。
  不过时：不调阈值；登记的备选 = 再加回退 (d)「人路径本身触发 hinge 的行不挂」（ok 写成 (a) - (d)，`ok_primary` 保留 (a) - (c)），
  G1 记为「主规则不过，靠排除行通过」，并报覆盖率掉多少，交主会话决定用哪个。
- **G2（能抓到 R2J 的出路）**：R2J-L18-s0 281 ACC + GAP 存档读数的 offroad 零分场景（pai_fix facts `fclass == offroad`），事件时刻按
  D1（首个 offroad 样本，否则 corridor 事件，否则 facts `t_ev_s`），取事件前 2.5 s 的那次决策（`searchsorted(t0, te - 2.5 s)`），
  在该时刻自车（控制器轨迹，后轴）位姿建同格式栅格，量 R2J 自己的模型计划（`poses_model`）的道路 hinge：触发份额，线 **> 50 %**
  （「多数」）。同时报：其中 t0 已出面（规则 (a) 会不挂）的个数、计划最小 SDF < 0 的个数；对照 = R2J 得分 ≥ 0.98 的场景每 2 s
  一次决策的触发率（误报底，不设线）。
- **G3（挂上但权重 0 = 逐位不变）**：R2J 的训练命令（r2.sh `cmd J`，on1_JTPH，JEV_RM_EAGER=1）跑 60 步，三次：旧代码（本改动前的
  提交，独立 worktree）无新 flag；新代码无新 flag；新代码 `--on-road-pai road_on1_JTPH.npz --on-road-pai-w 0`。df.py bitcmp：
  全部 scalar、全部权重、dev ADE、on_hash_60 逐位相同。两对都要过（新代码不带 flag = 旧代码；带 flag 权重 0 = 不带）。
  这是 60 步的恒等测试，不是训练；checkpoint 不作任何用途。

## 6. 执行

- 构建与 G1 / G2 是池里的 CPU 作业（`--vram 0.1`，PRIO 5，核数由提交时的 `jevdrive.common.n_cpus()` 推出，脚本内用 `n_cpus()`）；
  G3 是三个 GPU 60 步作业（PRIO 5）+ 一个 CPU 比对作业。用 `--after` 串成一条链，最后一个作业写 `DONE`；只用一个后台等待者。
- 代码：`experiments/round5/scripts/road_pai.py`（build / index / g1 / g2 / report）、`road_pai.sh`（chain / wait / 作业体）。
- 结果：`experiments/round5/results/road_pai.md` + `road_pai.json`。不写 decisions.md（主会话写）。
