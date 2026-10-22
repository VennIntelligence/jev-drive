# NUPLAN-DAG：PAI 的 DAgger + 地图横向教师搬到 nuPlan（10-22 预登记；任何训练输出之前写定，探针滚动之前提交）

依据：第 434 条（PAI R6DAG：DAgger + 地图横向教师，s1 在 281 上第一次明确超过 NTV0）、第 438 / 443 条（第 2 轮与 R8K 都不比第 1 轮好，方法本身成立）、
第 391 条（nuPlan 原生候选 B2 = `STANDFMXH-L18-s1`）、第 393 / 395 条（B2 剩下的易题侧向零分大多是过速：急右转里 1.1–1.65 倍人速、停车冲过；
正常速度下的路径错只有 1 窗）、第 264 条（nuPlan 车道中心 hinge 被日志自己违反 46 %，所以教师加了人的横偏）。
代码：[../scripts/nd.sh](../scripts/nd.sh)（链）、[../scripts/nd_teach.py](../scripts/nd_teach.py)（教师 / 闸 / 池 / 抽取）、
[../scripts/nd.py](../scripts/nd.py)（读数表）。盒上 `$DATA_DIR/runs/nuplan_dag/`。结果 [../results/](../results/)。

**问题。** 第 434 条在 PAI 上有效的「学生自己的状态 + 特权地图教师只教侧向」，在 nuPlan（B2 学生、B2 自己的配方）上是否也有效；
第 395 条说 nuPlan 的侧向零分多是过速，所以同时回答：**只教侧向能不能动这些侧向零分**。

## 1. 教师（与 PAI b3 / 第 434 条相同，只换地图来源）

- **路线**：AlpaSim 运行时的 MAP 路线生成器（`RouteGeneratorMap`，`route_start_offset_m 40`，与 nuPlan 滚动的 `generated-user-config` 相同），在场景记录的
  rig 轨迹上重建 = driver 收到的 route 消息切出来的那条折线。地图 = AlpaSim 场景加载器给的 trajdata VectorMap（由 nuPlan 地图的 lane / lane connector
  建成）。**路线车道** = 生成器自己的选道规则：每个记录位姿取 4 m 内、航向差 < π/8 的车道；上一条仍在候选里就保持，否则它的后继里唯一的候选，
  否则对后面最多位姿都成立的候选——即人实际开过的 lane / lane connector 序列（路口里就是人走的那条 connector）。侧向换道台阶按
  clip(3 s × 人速, 15, 60) m 五次 smoothstep 过渡；日志之后用生成器自己的 80 m 延长。代码就是 round6 `dag.py` 的 `route_ref` / `human_ref` /
  `teacher_path` / `place`，原样 import，不复制。冒烟（10-22）：nuPlan 场景上加载与重建可用，k = 0 的 anchor 与加载器的记录位姿相差 < 1e-6 m（帧一致）。
- **参考线**：每 0.25 m 重采样，σ = 3 m 高斯平滑；**加上人在同一站位相对车道中心的横偏**（σ = 10 m 低通，饱和 ±0.5 m，从不替换成 0）。
- **路径**：学生位姿投到参考线 → s0、d0（左正）、ψ0；Frenet 五次多项式在 clip(3 s × v0, 6, 80) m 内并回；|d0| > 5 m 或 |ψ0| > 60° 不出标签。
- **纵向 = 学生自己 serve 出去的 plan 的弧长**（行 meta 的 `poses`，原生 serve 无 hold），标签 8 个点放在教师路径上同一时刻 0.5 … 4 s 的弧长处，
  航向 = 路径切向，转到决策 anchor 的行坐标系。只改形状，不改快慢。
- **为什么不加 d443 的保持步调底线**：底线只会把慢的标签推快；B2 在 nuPlan 上已经偏快（k = 9 速度比 1.044，第 391 条），侧向零分的主因是过速
  （第 395 条 a 类 16 / 20），推快与目标相反。**为什么也不用第 438 条的 min(学生, 人) 弯道上限**：PAI 上第 2 轮没有比第 1 轮好；更重要的是本线要
  回答「只教侧向能不能动侧向零分」，混入纵向修正就分不开了。纵向留给 TURN-CAP / STAND-R 线。

## 2. 学生、滚动与行

- 学生 **B2 = `STANDFMXH-L18-s1`**，原生 serve（`sh30`，`JEV_HOLD` 不设，= B2 采用时的读数 serve），`SH30_DUMP=1000000` 帧 dump。
- 场景：`alpasim_nuplan/onpol-train` 的 **1031 个训练场景**（`experiments/onpol_nuplan/results/split/train.txt`，与 held 460 无交集，脚本检查），
  18 个决策，长缓存 `alpasim_nuplan_long/after160`（与 onpol_long / L18 系列池同一设置）。分块：c000 = 6 个探针场景（按已存 L18RB0 行的人轨迹：
  急右转 2、左转 1、刹停 1、起步 1、直路 1，固定种子），其余 1025 个分 24 块。
- 行管线与 onpol_long / practice_nuplan 相同、不改：`on_rows.py geo`（行、actor 框、路面 / 可行驶栅格）→ `on_rows.py tok`（token + 闸 R1 / R2，
  对照 smoke10e，线同 pn.sh）→ `rm_build pool / anchor`（路线特征）→ strip。池 **`on1_NDAG`**（探针池 `on1_NDP` 只用于检查，不合并）。
- **可抽行** = 策略决策（k ≥ 2；k = 0、1 是日志回放状态）且喂入车速 ≥ 2 m/s 且教师有标签，且 T4 不丢；其余行 fut = served、权重 0。
  可抽行的 `tab.fut` := 教师标签；`dagger.npz` 记全部标记。

## 3. 闸（线在读数之前写定）

- **T1 教师几何对人自己的路径**：1031 个训练场景的记录日志，每 0.5 s、v ≥ 2 m/s、有完整 4 s 未来的人的状态；教师从人的位姿出发，点按人自己的弧长放；
  误差 = 人的点减教师点、沿教师左法向。**线：除变道形状外全部状态，|误差| 中位 2 s ≤ 0.30 m、4 s ≤ 0.50 m**；转弯（人 4 s 内转 ≥ 30°）超过线的
  2 倍就先检查转弯几何。纯车道中心教师的同一读数并列报（几何检查，人相对 nuPlan 车道中心的偏移）。教师带人横偏，所以这一闸主要查路线 / 车道选择与帧。
- **T2 教师在可行驶栅格内**：池的 `drivable.npz`（评测器 drivable_area：ROADBLOCK | INTERSECTION | CARPARK_AREA，行坐标系 SDF，0.5 m）上，
  nuPlan 自车框（5.176 × 2.297 m，后轴到中心 1.461 m）沿 8 个标签位姿的轮廓最小 SDF。**线：可抽候选行里，自车当前在栅格内、标签 ≥ 6 / 8 个位姿在
  56 m 画布内的行，教师标签全程在内（SDF ≥ 0）的份额 ≥ 0.90**。并列报学生 served plan 的同一读数，以及自车已在栅格外的行里教师比 served 更靠里的份额。
- **T3 恒等**：B2 的原命令（meta.json argv）60 步 eager 两次：on1_JTX / JTX00 对 on1_JTXD / NDAG0（DAgger 行权重 0），`df.py bitcmp` 必须 identical。
- **T4 丢掉不可行的标签**：标签横向加速度（位置估计量，8 点 0.5 s，与 PAI 第 434 条补记 3 同一读法）> 4 m/s²，或自车当前在栅格内而标签离开栅格
  （SDF < 0）——这些行权重 0。**线：丢掉的 ≤ 可抽候选行的 10 %**，否则停（VERDICT）。
- 另外（不是闸，不过就停）：池闸 R1 / R2（pn.sh 的修正线）、合并后前缀与 on1_JTX 每个文件逐位相同（`nd_teach.py append` 自检）、烟测
  （每批 45 normal + 13 yaw-rate + 78 池行，各组份额与抽取文件一致、标量有限）。
- **探针（c000，6 个场景）的检查 chk1**：计分 ≥ 5 / 6、行 ≥ 60、k ≥ 2 且 ≥ 2 m/s 的决策里教师有标签 ≥ 80 %、帧检查 p99 ≤ 0.10 m、T1（1031 个日志）、
  T2、T4 在探针行上都过；外加教师 BEV 图（`results/figs/probe/`）。过了就自动提交全量（主会话 10-22 指示：不等）；不过写 VERDICT 停。

## 4. 臂

- **NDAG（`NDAG-L18-s0` / `s1`）** = B2 的原命令（`STANDFMXH-L18-s1` 的 meta.json argv）只改：`--on on1_JTXD --on-wfile runs/nuplan_dag/w/NDAG.npz
  --on-k 78 --batch 136 --yr-mass 0.095`（45 normal + 13 yaw-rate + 70 原池行不变，另加 **8 行 DAgger**），`--tag`，`--seed`（0 / 1）。从 RB0-F-s0 起
  （与 B2 同一起点、同 9 500 步），不是在 B2 上续训——这样 T3 才有意义、对照就是 B2 自己（s1）与它的 seed 复本 `STANDFMXH-L18-s0`（s0）。
- 抽取文件 NDAG = JTX00 的权重 × 70 / 78 + 可抽 DAgger 行 8 / 78 均匀；落在配方 `--on-hold-pct 15` 留出日志（S 行日志 crc32 % 100 < 15）里的
  DAgger 行权重 0（池监控读数不被污染）。DAgger 行在混合池里是 `both` 模式（模仿行），不进 STAND / FREE-M（只作用于 `self` 行），侧向栅格
  L18NR1v2 只覆盖前缀（`--lat-host`），新行没有它。
- 剂量：每批 8 行，约等于 PAI；nuPlan 每场景 18 个决策，可抽行预计约 1 万，9 500 步里每行被抽约 7 次（PAI < 1 次）——登记在此，不改剂量；
  过拟合迹象看训练里的 pool_held 读数。
- 对照：已存的 B2 s1 / s0 读数（同 serve、同考场），不重跑。

## 5. 考场与判定

- 读数与 B2 采用时相同：大考场 held 划分 18 块（`jt.sh aread <tag> off`，原生）+ held 460 @18（`rm_chain.sh read`）；`free_m.py ext / arena`
  按窗配对、`df.cl_read` 按场景配对（按 log 聚类 bootstrap）；冻结配池 PCS（`pcs_local.py score`，Mac）。
- 侧向零分 = 大考场 held 窗里 offroad / corridor 零分，按第 395 条预登记的规则（`lat_zero.classify`）分：**a 过速**（出界前 3 s 同站点速度比 ≥ 1.05
  且人减速 ≥ 10 % 或曲率 ≥ 0.02）、**b 正常速度下的路径错**、**c 其他**；a 再按人在出界前段转向 ≥ 28° 分 a2 转弯 / a1 停车类。held 460 的侧向零分
  （offroad / corridor / AF-side）按类计数。
- **adopt-candidate**（两个 seed 都满足，各对同 seed 的 B2）：(1) held 460 零分 ≤ B2 + 2（两个 seed 的 B2 都是 41；+2 是同族 seed 噪声，第 366 / 391 条）；
  (2) held 460 配对均分点估计 ≥ 0（报 CI）；(3) 大考场配对均分点估计 ≥ 0（报 CI）；(4) 大考场侧向零分 < B2。
- **「方法迁移成立」** = 两个 seed 都是 adopt-candidate。**「机制成立、未过护栏」** = 两个 seed 侧向零分都比同 seed B2 少 ≥ 15 %，但 (1)–(3) 有一条不过。
  其余 = 不迁移。另外按 a / b 分：若 b 类下降而 a 类不动，记为「只教侧向修不了过速」（第 395 条的预期）；若 a2 也下降，说明形状标签在急弯里
  也有约束力。
- 不改 serve、不提交、不推镜像；不碰别的线的作业与文件。决策行写在 decisions.md 底部，由本线在判定后写。
