# long_oracle 预登记：当前 driver 有没有比 served 更好的纵向工作点，纵向修法该打哪一类碰撞（2026-10-11，任何本线逐 token 读数之前写定）

lane LONG-ORACLE。任务书 = [research/next-round/overnight.md](../../../research/next-round/overnight.md) 的 N1，改到当前 checkpoint。
纯 CPU（navtest 部分），不训练，不跑模型，不改第三方 scorer。**不是 go / no-go**：交付分类表、纵向缩放的回收率与代价、
「有没有比 served 更好的工作点」的读数，以及一句话「下一步的纵向修法该打哪一类」。

## 问题

1. `P2H10S-F`（第 232 / 244 / 247 条，当前最好的 checkpoint，PAI 对 base +0.09）与 base `P2H10-F` 在 navtest 上的 NC / TTC 失败，
   按类型 × 转角桶 × ego 速度段分是什么，每类 WA-JEPA 通过多少。
2. 路径不变、速度按 a 缩放（a ∈ {0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15}，另带第 196 条的 0.5 / 0.6 扩展档），
   non-reactive 官方打分：每档全体一刀切的净效应（NC、TTC、DAC、EP、no-EC EPDMS），每类失败被救回多少、新增多少；
   有没有一档净正；只用推理时拿得到的输入（t0 ego 速度段、plan 自己的转角）按段选档、按 log 交叉拟合，有没有净正。
3. 若 (2) 有余地：PAI 闭环里同一问题（见「PAI 条件」）。

## 写本稿之前已经看过的数

- 第 196、203、226、230、232、241、244、247、249–251 条全文或摘要；`P2H10S-F` / `P2H10-F` 的 bench navtest 聚合（`tables.load`）：
  EPDMS 89.01 / 89.09（S s0 / s1）、88.58 / 88.77（base s0 / s1）；NC 失败 181 / 167 / 188 / 176；TTC-only 90 / 92 / 86 / 81。
- 没有看过这四个臂的任何逐 token 失败类型、任何缩放后的分数。

## 与已有条目的差别（不重测的部分）

- **第 196 条**（N1，SH30）已经给出：NC 失败 53% 是本车道前方静止 / 慢车，同路径 0.7–1.1 特权 oracle 救回 74%，一刀切减速没有净正档
  （× 0.9 −0.33）。本线**不重测这个机制**，引用它；新的部分只有：(a) 换成当前 checkpoint（`P2H10S` 的 hinge 只看形状、速度剖面回到 base，
  第 232 条），以及 base 作对照，看分类与工作点在 hinge 前后是否一样；(b) 在 1.0 附近加细档 0.85 / 0.95 / 1.05 / 1.15：第 249–251 条显示
  闭环里 plan 长 10–12% 就多撞、短了就丢 progress，问的是「刀刃」两侧 5% 的地方有没有更好的点，第 196 条只有 10% 的步长；
  (c) 非特权的条件化工作点（按 t0 速度段选档，log 交叉拟合）；(d) 按第 203 条的教训，把增益拆成 NC / TTC 改变的 token、只有 DAC 改变的
  token（放慢推出 4 s 视野）、其余 token。
- **第 203 条**（学出来的选档头）已判负，本线不训练任何选档器；(c) 只是 4 个格子的查表，用来回答「有没有一个可服务的工作点」，不是方法。
- **第 157 / 179 / 158 条**（固定余量、模型 lead 读数、agent hinge）是修法的负结果，本线不测修法。
- **第 226 条**：PAI served = `JEV_VCONT=1.0 JEV_LEAD=1`。`JEV_VCONT` **不是速度倍率**，是速度剖面从 ego 当前速度并入 plan 的 1.0 s 时间常数；
  serving 里没有现成的速度倍率旋钮。第 226 条量过 VCONT 开 / 关（+a +0.058，CI 含 0）与 LEAD（+b +0.086），没有量过速度倍率。

## 对象与数据

- navtest 12 146 token（split `navsim/navtest`），`P2H10S-F-s0 / s1` 与 `P2H10-F-s0 / s1` 的 bench 存档（warp 帧，`runs/bench/navtest/<arm>@warp`），
  导出位姿 `bench.compat.pred_file`。navhard stage 1 只报计数（同 bench 默认帧）。
- WA-JEPA 只用存档逐 token 子分作对照列。真值只经 devkit scorer 进读数。
- 代码：`experiments/op_parity/scripts/nc_tax.py`（第 196 条的分类与打分，加环境变量参数化，默认行为不变）跑每个臂（两 seed 一组），
  `experiments/long_oracle/scripts/lo_report.py` 出本线新增的读数，链 `experiments/long_oracle/scripts/lo_chain.sh`。
  打分 = `python -m jevdrive.bench score-poses --traffic non_reactive`（pool）。

## 分类（照抄第 196 条，不改）

失败集合 NC < 1；TTC-only = TTC < 1 且 NC = 1。事件提取 `fd_navsim.py`（第 153 条的 scorer hook），类：E 静态物体 / VRU → D 侧碰
（sideswipe，或静止车且质心在车身侧 dx < 4.05 m、|dy| ≥ 1.0 m）→ A1 前方静止车 → A2 同向前车 → B 切入 → C 路口横穿 → E 对向 / 未解析。
桶：bench strata 的转角桶（< 5 / 5–20 / 20–45 / > 45°）与 t0 速度段（< 2 / 2–5 / 5–10 / ≥ 10 m/s）。「arm 特有」= 该臂失败而 WA-JEPA 通过。
校验：重放 8 子分与 bench 存档逐 token 相同（> 1e-6 剔除并报数）；a = 1.0 的打分行与存档逐 token 相同（identity gate，不过就停）。

## 读数

1. 分类表（每臂两 seed 计数与均值，份额的 log 簇 bootstrap CI）：NC 与 TTC-only 各自按类、类 × 转角、类 × 速度；WA-JEPA 通过比例；
   `P2H10S` 减 base 的逐类计数差（描述性）。
2. 特权 oracle（第 196 条口径，逐 token 用 simulator 分数选档）：每类回收率（主族 0.7–1.1 与新增的 0.95–1.05、0.85–1.15、全部档）、
   回收 token 上的 EP 代价、板级上限。
3. **一刀切**：每档 a 对 a = 1.0 的 ΔNC、ΔTTC、ΔDAC、ΔEP、Δno-EC EPDMS（逐 token 配对、按 log 簇 bootstrap，B 10 000，`jevdrive.stats.paired`），
   NC / TTC-only 失败的修好数与新增数（按类：修好数按 a = 1.0 的类；新增数按转角桶 × 速度段，新增失败不重放）；Δno-EC EPDMS 拆成
   (i) NC 或 TTC 改变的 token、(ii) 只有 DAC 改变的 token、(iii) 其余 token 三部分。
4. **条件化工作点**（非特权）：按 t0 速度段（4 格）为主、t0 速度段 × plan 自身 4 s 转角桶（16 格）为次，在一半 log 上每格选
   no-EC EPDMS 均值最大的档（平局取离 1.0 近的），在另一半 log 上读；两折交换后合并成全 12 146 token 的 out-of-fold 读数，
   与 a = 1.0 配对、log 簇 bootstrap。折 = log 名的 sha1 奇偶（固定）。同样拆 (i) / (ii) / (iii)。
5. 每臂每 seed 分开报；`P2H10S` 是主臂，base 是对照。

## 判定线（先写后读，不改）

**「navtest 上有余地」** = 主臂 `P2H10S-F` 的两个 seed **都**满足下面任一条：
- (U) 某个 a ≠ 1.0 的一刀切 Δno-EC EPDMS 的 95% CI 下界 > 0；
- (C) 主条件化工作点（速度段 4 格）out-of-fold Δno-EC EPDMS 的 95% CI 下界 > 0；
并且对 a < 1 的档（或条件化里选了 a < 1 的格），增益里 (i) NC / TTC 改变的 token 至少占一半（否则按第 203 条记为 DAC 视野效应，不算余地）。
a > 1 的档不加这一条（增益来自 EP）。

不满足 = **明确为负**：没有比 served 更好的纵向工作点（在同路径缩放族里），按停止规则不做 PAI，不再加档。

「该打哪一类」：主臂两 seed 的 NC 失败里「arm 特有」计数最大的类；若前两类相差不到 10%，按特权 oracle 回收率高的取。TTC-only 单独报，不进这一句。

## PAI 条件（只在 navtest 有余地时）

- 需要一个 serving 旋钮：在 `experiments/alpasim/lib/serve_fix.py` 加 `JEV_VSCALE=<a>`（路径不变、弧长 × a，与 `turn_ceiling.speed` 同一变换，
  默认 1.0 = 逐位不变），在 VCONT / LEAD 之前施加；条件化工作点则按 t0 速度段查表。
- 臂：`P2H10S-F-s0`，served（`JEV_VCONT=1.0 JEV_LEAD=1`）加 navtest 选出的 2 档（最好的档，以及它与 1.0 之间的一半），33 scene 的 `s_b1a`，
  `pai_eval.sh`，Tokyo 双卡，在 `/data/runs/alpasim/keywarp/STATUS` 写 DONE 之后才开，不碰那里的任何进程；上限 2 card-hour。
  读数：对同 checkout 的 served a = 1.0 逐 scene 配对（按 scene bootstrap），at-fault 碰撞零分、slow scene 数、progress。
- 闭环判定（描述性，33 scene 功效低）：Δ均分 CI 下界 > 0 记为「PAI 上有更好的点」；点估计 > 0 但 CI 含 0 记为「未分出」；点估计 ≤ 0 记为负。

## 限定（事先写明）

non-reactive：减速的回收偏乐观、减速新增的追尾不计；no-EC EPDMS（缩放后的 EC 未定义）；特权 oracle 是上限不是方法；
类型是 scorer 状态上的启发式；条件化工作点只有 4 / 16 个格子，交叉拟合按 log 折，不调别的参数；PAI 的 served 带 LEAD，
navtest plan 不带，二者纵向通路不同。

## 预算

navtest：box CPU 约 1 h（第 196 条 14 键全量打分 8 分钟，本线每臂 11 键 × 2 seed）。PAI：≤ 2 card-hour，只在有余地时。
