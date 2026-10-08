# op_parity nc-taxonomy 预登记：SH30 的 NC / TTC 失败是哪几类，只改纵向能救回多少（2026-10-09，任何本线逐 token 读数之前写定）

任务书 [research/next-round/overnight.md](../../../research/next-round/overnight.md) 的 N1。纯 CPU，不训练，不跑模型。**不是 go / no-go**：
交付分类表、纵向 oracle 的回收率与 EP 代价，以及一句话「下一步的 NC 修法该打哪一类」。

## 问题

SH30（第 170 条，`SH30-F-s{0,1}`，navtest EPDMS 89.55）在 navtest 上 NC 98.60、TTC 98.00，WA-JEPA 是 99.40 / 98.89；NC 是乘性项，
也是 AlpaSim at-fault 碰撞零分的对应项（本地 48 scene 上 SH30 的两个零分都发生在减速到 3 m/s 以下时）。要回答：
(1) 这些失败按碰撞对象与几何分成哪几类、各在哪个转角桶与速度段；(2) 每类里 WA-JEPA 过了多少（即多少是 SH30 特有的）；
(3) 路径不变、只缩放纵向，能把多少失败救回，EP 掉多少。

写本稿之前已经看过的数：plan.md 第 1 节与第 170 条里的 SH30 / WA-JEPA 全榜子分均值（NC、TTC、EP、DAC），以及 `bench` 存档 `summary.json` 的 seed 0 均值。
没有看过 SH30 的任何逐 token NC / TTC、任何按类型的拆分、任何纵向缩放后的分数。

## 与已有条目的差别（不重测的部分）

- **第 153 条**（four_dirs）对 **P2H**（λ 10 的 hinge）做过一次撞车分类：266 个 NC 或 TTC 失败合并成一组，前方静止车 36%、49% 的 plan 比日志快 ≥ 10%，WA 同 token 只挂 30%。
  本线的不同：(a) driver 换成现配方 SH30（hinge 30 / 0.5 m，DAC 与路径都变了）；(b) NC 与 TTC-only 分开报；(c) 类型 × 转角桶 × ego 速度段的三维表，
  并把「侧向接触」从「前方静止车」里按几何拆出来（第 153 条里 64% 的静止车撞击在转弯 token 上，但没有区分是车头撞上还是车身擦上）；
  (d) 纵向缩放 oracle 是新读数。分类规则沿用第 153 条的事件提取代码（`fd_navsim.py` 的 scorer hook），不重写；P2H 的数不重算，只作对照行引用。
- **第 158 条**（agent hinge）与**第 179 条**（模型自己的 lead 输出）是修法与信号的负结果，本线不测任何修法，只给它们之后「该打哪一类」的依据。
- **第 178 条**（turn_ceiling）量过 > 20° token 上速度缩放 0.8 / 1.2 / 0.6 / 1.4 对 no-EC EPDMS 的 best-of-K，读数是总分，没有按 NC 失败、也没有覆盖直行 token。
  本线复用它的 `speed()` 变换，读数换成 NC / TTC 失败的回收率与 EP 代价，覆盖全部 token。

## 对象与数据

- navtest 12 146 token（split `navsim/navtest`），SH30-F-s0 / s1 的 bench 存档：逐 token 子分 `runs/bench/navtest/SH30-F-s?@warp/units.csv`，
  导出位姿 `bench.compat.pred_file`（后轴，0.5–4 s 共 8 个）。两个 seed 各自分类，表里报 s0 / s1 计数与 seed 均值。
- navhard stage 1（split `navsim/navhard_two_stage`，G 帧口径，`runs/bench/navhard/SH30-F-s?@gimm/harness/harness_tokens.csv` 里 `stage == 1` 的行）。
  只报计数表；stage 2 不做。
- WA-JEPA 只用存档的逐 token 子分（navtest CSV、navhard harness_tokens.csv）作对照列。不重放它的轨迹、不用它的权重或特征。
- 真值 agent 框与地图只经 devkit scorer 进入读数（oracle 探针），不进任何推理路径。

## 失败集合

- **NC 失败**：NC < 1（含撞静态物体的 0.5）。
- **TTC-only 失败**：TTC < 1 且 NC = 1。
- 两者之并 = 第 153 条的 D3 口径，只作对照行。

## 事件提取与分类规则

事件提取：`fd_navsim.py` 的 instrumented devkit 重放（scorer 的 hook，读 scorer 自己的状态），对 SH30 两个 seed 在失败 token 上各重放一次。
navtest 用 non-reactive traffic（与 bench navtest 同），navhard 沿用存档 harness 的口径。重放的 8 个子分必须与 bench 存档逐 token 相同
（最大绝对差报在结果里；> 1e-6 的 token 单列并从分类表剔除）。NC 失败取第一个 at-fault 碰撞事件，TTC-only 取第一个 TTC 事件。

细类沿用第 153 条的 `ctype_class`（static object / VRU / stopped vehicle ahead / sideswipe / oncoming / crossing / cut-in / lead vehicle / unresolved）。
在它之上加一个几何标记 **side**：事件时刻对象质心在 ego 后轴系里 dx < 4.05 m（不超过车头）且 |dy| ≥ 1.0 m，即对象在车身侧面而不是车头前方。
TTC 事件的位置取事件步上 ego 的实际仿真状态（不是匀速外推后的框），这一标记对 TTC-only 只是近似。

任务书要的五类，按下面顺序取第一条命中的：

| 大类 | 规则 |
|:--|:--|
| E 其他：静态物体 | 对象不是 agent（锥桶、隔离栏等） |
| E 其他：VRU | 行人 / 自行车 |
| D 侧碰 | 车辆，且 (细类 = sideswipe) 或 (细类 = stopped vehicle ahead 且 side 标记成立) |
| A 前方静止或慢车 | 细类 = stopped vehicle ahead（非 side）→ A1 静止；细类 = lead vehicle → A2 同向行驶的前车 |
| B 切入 | 细类 = cut-in |
| C 路口横穿 | 细类 = crossing / turn conflict（相对航向 30–150°） |
| E 其他：对向 / 未解析 | 细类 = oncoming（> 150°）或 unresolved |

「转弯时侧碰」= D 类落在 ≥ 20° 两个转角桶里的部分；D 类本身不用转角定义，这样类型 × 转角表不是循环定义。细类计数同时全表给出，五类的合并不丢信息。

分桶：
- 转角桶：bench strata 的 `turn`（logged 4 s 航向变化 |dyaw|：< 5°、5–20°、20–45°、> 45°）。navhard 没有 logged future，用第 153 条存下的 PDM-Closed 参考路径算同一个量。
- ego 速度段：bench strata 的 `speed`（t0 速度 < 2、2–5、5–10、≥ 10 m/s）。另报两列描述量：事件时刻 ego 速度 < 3 m/s 的比例（对应 AlpaSim 的观察），
  plan 4 s 弧长 / 日志 4 s 弧长 > 1.1 的比例（第 153 条的「比日志快」）。

## WA-JEPA 对照列

每个（集合、类）格子里：WA-JEPA 在同一 token 上该项通过的比例（NC 集合看 WA 的 NC = 1；TTC-only 集合看 WA 的 TTC = 1 且 NC = 1）。
通过 = 这一格是 SH30 特有的失败；不通过 = 两个模型共有。另给 WA-JEPA 自己的 NC / TTC-only 失败总数与它们在转角桶上的分布（无类型，因为不重放）。

## 纵向缩放 oracle

- 族：同一条路径，每个时刻走过的弧长 × a（`turn_ceiling.speed`，超出 4 s 终点按末段曲率外延）。
  **主族 a ∈ {0.7, 0.8, 0.9, 1.0, 1.1}**（任务书的范围）；**扩展族**再加 {0.5, 0.6}，单独一行报，用来看主族是不是太窄。a = 1.0 是存档数组本身。
- 打分：`python -m jevdrive.bench score-poses --traffic non_reactive`（bench navtest 的 traffic 口径），不带 EC，分数口径是 no-EC EPDMS。
  a = 1.0 的 8 个子分必须与 bench 存档逐 token 相同（identity gate，不过就停，不读其余分数）。
- 先小后大：stage 0 只打失败 token（两个 seed 的 NC 或 TTC 失败之并）× 7 档 × 2 seed，出回收率；stage 1 打全部 12 146 token × 7 档 × 2 seed，
  出「全体统一缩放」的板级代价。stage 0 的回收率不设停线（本项不是 go / no-go），stage 1 只在 CPU 有余量时跑，没跑就如实写。
- 读数（逐 seed 算，再取 seed 均值；比例的 CI 用按 log 聚类的 bootstrap，`jevdrive.stats`，B 10 000）：
  1. **NC 回收率**：NC 失败 token 里，族内存在某档使 NC = 1 的比例；主族与扩展族各一行；按大类拆；按所需最小改动的累计（只许 0.9–1.1、0.8–1.1、0.7–1.1、0.5–1.1）。
     只减速（a < 1）与只加速（a = 1.1）分开报。
  2. **干净回收率**：同上，但要求该档的 no-EC EPDMS 高于 a = 1.0 的（即没有用别的乘性项失败换 NC）。
  3. **TTC-only 回收率**：TTC-only token 里存在某档使 TTC = 1 且 NC = 1 的比例，同样的拆法。
  4. **EP 代价（token 级）**：被回收的 token 上，取回收档里 no-EC EPDMS 最高的一档 a*（并列取离 1.0 最近的），EP(a*) − EP(1.0) 的均值、a* 的分布。
  5. **板级上限**：只在 SH30 的 NC 或 TTC 失败 token 上换成 a*（其余 token 不动）时，全榜 NC、TTC、EP、no-EC EPDMS 的变化（点数）。
     这是「知道哪里该减速、减多少」的特权上限，不是方法。
  6. **统一缩放的代价**（stage 1）：每个 a 对全部 token 一刀切时全榜 NC、TTC、DAC、EP、no-EC EPDMS 的变化；以及新增的 NC 失败数（原来通过、缩放后失败）。

## 一句话结论的取法（事先写定）

「该打哪一类」= SH30 特有失败数（NC 失败数 ×（WA-JEPA 通过比例））最大的大类；并注明它的主族纵向回收率。
如果最大的两类相差不到各自 bootstrap CI 的半宽，就两类并列写。这只决定那句话怎么写，不是判定线。

## 资源与流程

- CPU only。重放用 `n_cpus() // 3` 个进程；score-poses 走 pool 的 CPU 作业，`--jobs` 限到 3（36 核），给同时在跑的 O1 / R1 / C0 留核。
  估计：重放 < 5 min；stage 0 约 500 token × 14 key，几分钟；stage 1 约 17 万 (token, key)，non-reactive 下估 0.5–1 h（turn_ceiling 的实测量级），
  超过 3 h 就停在 stage 0 的读数。
- 一条自推进的链（tmux `jev:nc-tax`，`nc_tax_chain.sh`），状态文件 `$DATA_DIR/runs/op_parity/nc_tax/chain/{STATUS, DONE, ERROR, log.txt}`。
- 代码 `experiments/op_parity/scripts/nc_tax.py`（`replay` / `family` / `gate` / `report`），结果 `experiments/op_parity/results/nc_taxonomy.md` 与 `nc_taxonomy/*.csv`，第 196 条。

## 已知的限定（事先写下）

- 类型是 scorer 碰撞状态上的启发式（相对航向、t0 侧向偏移、质心位置）；side 标记用质心而不是接触点，大车会误分。
- non-reactive：他车按日志走，不对 ego 的减速 / 加速作反应；缩放后被后车追尾不算 at-fault，所以「减速救回」在 reactive / 闭环里偏乐观。
- 纵向缩放改的是 plan，devkit 的 LQR 跟踪会让实际路径略变，DAC 可能翻转；「干净回收率」就是为此设的。
- oracle 用了 navtest 的 simulator 分数来选档，只是上限。
- navhard stage 1 量小，只报计数，不下结论。
