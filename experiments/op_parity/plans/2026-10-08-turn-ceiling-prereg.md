# op_parity turn-ceiling 预登记：SH30 自己的 plan 附近一个小轨迹族里，有没有分数高得多的轨迹（2026-10-08，任何本线读数之前写定）

## 问题

navtest 上 SH30（第 170 条）对 WA-JEPA 差 2.16 EPDMS，其中 1.65 在转弯 token：> 45° 79.49 对 87.41（n 1 517），20–45° 83.35 对 88.23（n 1 637）。
hinge 强度轴已饱和（第 172 条），真值边界输入对 P2 通路无用（第 166 条），转弯平衡采样不闭合（第 154 条），表征线暂停（第 160 条）。
剩下的想法是转弯上不再模仿日志轨迹，直接拿 simulator 分数当目标（按分数重标的 target，或按分数监督的 selector）。训练之前先量一个特权上限：
SH30 plan 附近三个自由度（横向偏移、曲率增益、速度缩放）的小族里，best-of-K 能比 SH30 自己高多少，增益在哪个自由度上。
所有 oracle 读数都用了 navtest 的 simulator 分数来选，只是上限，不是方法。本线只做 stage 0 和 stage 1，纯 CPU，不训练。

## 已有的、不重测

- 第 153 条：急弯失败 54% 切内角（前角约 3.4 s 切入 0.40 m）、26% 转不过去，航向增益 1.04；第 148 条第 5 点：曲率 / 航向增益不封顶（斜率 0.94）。
  本线不再量增益本身，只量「换一条轨迹能拿回多少分」。
- 第 170 / 172 条：SH30 的 checkpoint、navtest plan 与逐 token 分数直接复用（`SH30-F-s{0,1}`，bench 存档），不重训、不重打 identity 之外的分。
- 第 168 条：WOD 上同类读数（F20 +1.068，8 条全局锚点就 +0.59）；本线沿用它的拆法（单轴 / 联合）和小 K 读数，换成 navtest 的 EPDMS。
- 打分只走 `python -m jevdrive.bench score-poses`；转角分桶用 bench 的 strata（logged 4 s 航向变化）。

## 候选族

对象：navtest 上 |dyaw| ≥ 20° 的 3 154 个 token（bench 的 20–45 与 > 45 两档），SH30 两个 seed 各自的 plan 各建一族，读数取每 token 的 seed 均值
（与第 170 条的表同口径）。输入是 bench 的导出位姿（后轴，0.5–4 s 共 8 个，`preds/SH30-F-s?-warp__base.npz`）。

三个自由度，顺序固定为 曲率 → 偏移 → 速度；某一轴取恒等值时该步跳过，恒等候选就是存档数组本身（逐位相同）：

| 轴 | 定义 | 内层 | 外层 | 取值理由 |
|:--|:--|:--|:--|:--|
| 曲率增益 g | 所有航向（位姿 yaw 与弦方向，相对 t0 航向）× g，段长不变，所以速度剖面不变 | 0.85 / 1 / 1.15 | 0.7 / 1.4 | 「转不过去」口径是航向增益 < 0.9，1.15 把 0.87 拉回 1；外层 1.4 对应 shipped 级别的欠转（0.63–0.77，第 141 条） |
| 横向偏移 d | 沿位姿法向平移 d（左为正），按行驶距离前 6 m 的 smoothstep 渐入（静止不偏移），yaw 跟随平移后路径的切向 | −0.5 / 0 / +0.5 m | ±1.0 m | SH30 的 hinge margin 是 0.5 m；切内角深度 0.40 m，宽弯擦边 62% < 0.3 m（第 153 条）；1.0 m 已到车道保持扣分的量级 |
| 速度缩放 a | 同一条路径，每个时刻走过的弧长 × a；超出 4 s 终点的部分按末段曲率（上限 0.3 / m）外延 | 0.8 / 1 / 1.2 | 0.6 / 1.4 | 撞车 plan 49% 比日志快 ≥ 10%（第 153 条）；EP 差 0.22 需要更快的一侧 |

候选共 33 条（`turn_ceiling.py candidates`）：c00 恒等；c01–c06 内层单轴；c07–c18 内层两轴；c19–c26 内层三轴；c27–c32 外层单轴。
网格左右对称，所以候选集合与转向无关；「内侧 / 外侧」只在分析时按 logged 航向变化的符号标注。

读数用的族：

| 族 | K | 内容 |
|:--|--:|:--|
| O3 / K3 / V3 | 3 | 只动偏移 / 只动曲率 / 只动速度（内层） |
| F7 | 7 | 三个单轴的并集 |
| **F19** | 19 | 内层网格去掉三轴同时动的 8 个角（恒等 + 单轴 6 + 两轴 12）：**判线用这一族** |
| F27 | 27 | 内层 3 × 3 × 3 全乘积 |
| O5 / K5 / V5、F13、F33 | 5 / 13 / 33 | 加上外层单轴点 |

为什么是 33 而不是 20：(1)「必须联合」只有在乘积网格上才有定义，3 × 3 × 3 是最小的对称乘积；(2) 用户同意的线是 best-of-20，
所以判线族固定为 K = 19 的 F19（比 20 少一条，只会更保守），F27 / F33 是次读数，不参与判线；(3) 外层 6 条只用来看单轴的剂量（内层是不是太窄）。
成本上一个 token 的 k 条候选只多跑 k + 1 次 IDM（bench 的 memo），66 个 key 估 ~69 core-s / token。
已知的近似：速度缩放沿弦线插值，R 12 m、5 m/s 的圆弧上位置误差 ≤ 6 cm（偏内）；合成轨迹上验过恒等逐位不变、静止 plan 所有候选不动。

## 分数口径

score-poses 给的是不带 extended comfort 的逐 token EPDMS（EC 要相邻帧的 plan，候选族里没有定义）。所以本线所有上限、差值都用 **no-EC EPDMS × 100**；
SH30 = 恒等候选，WA-JEPA 用存档的 8 个子分按同一公式重算。带 EC 的存档数（79.49 / 87.41 / 83.35 / 88.23）并排列出，只作对照。
oracle 逐 seed 取族内最高分（并列取编号最小的候选：恒等优先，其次动的轴少的），再对两个 seed 取均值；CI 是按 log 聚类的配对 bootstrap（`jevdrive.stats.paired`，B 10 000）。

## 读数

1. **上限**：各族 best-of-K − SH30，分桶 > 20°、20–45°、> 45°、左 / 右（> 20° 与 > 45°）。
2. **自由度拆分**：只偏移（O3）、只曲率（K3）、只速度（V3）、任一单轴（F7）、必须联合 = F27 − F7、三轴同动 = F27 − F19；O3 − K3、O3 − V3、K3 − V3 的配对差。
3. **小 K**：在 > 20° token 上贪心前向选 K 条（含恒等，K = 2 / 3 / 4 / 6 / 8 / 12 / 20，候选池 F33），族内 oracle；in-sample 与按 log 两折 cross-fit
   （另一半 log 上选族）都报。另报「一条固定变换用在所有 token 上」（不带 oracle）的增益：每条候选的值，以及 cross-fit 选出的最优一条。
   best-of-K 天然乐观，固定变换那一行是唯一不含逐 token 特权选择的读数。
4. **oracle 选了什么**：各候选被选中的比例；按轴的边际（偏移朝内 / 不动 / 朝外，曲率 < 1 / = 1 / > 1，速度 < 1 / = 1 / > 1）及其承载的增益份额。
5. **子分拆分**：SH30 对 WA-JEPA 的差在 > 45° 与 20–45° 上按 Shapley 拆到 DAC / NC / TTC / EP / LK / 其余（DDC + TLC + HC + EC）：
   带 EC 的存档口径（对上 7.92 / 4.88）与 no-EC 口径各一份；oracle（F19、F27）− SH30 同样拆，每个子分的回收比 = oracle 拿回的 / WA 领先的（比值的 log bootstrap）。
   另列各臂的 NC / DAC / TTC / LK 失败率与 EP。

## 闸门与停止规则（现在定死）

- **Stage 0（冒烟）**：先 24 个 token，再 300 个 token（两档各 150，seed 0 随机），都打全部 66 个 key。闸门脚本**只读恒等候选的行**：
  两个 seed 的 NC / DAC / DDC / TLC / TTC / LK / HC 与存档逐 token 完全相等，EP 与 no-EC 分数绝对差 ≤ 1e-6，300 个 token 全部满足才过。
  不过就停，修工具，其他分数一概不读（对其余 key 只做「有限且在 [0, 1]」的范围检查，不输出任何值）。
- **成本**：300 token 那次量出每 token 的 core-s，按 池 CPU 预算 // 12 × 12 个核外推 stage 1 墙钟。削减阶梯（取第一个 ≤ 1.5 h 的）：
  F33 × 2 seed → F27 × 2 seed → F33 × seed 0 → F27 × seed 0；都 > 1.5 h 时取 ≤ 3 h 的最小一档；都 > 3 h 则停下报告。削了什么写进结果。
- **Stage 1（全部 3 154 token）判线**：**F19 在 > 20° 桶上的 best-of-K 增益点估计 < +4.0 EPDMS（≈ navtest 全量 +1.0）→ 这条线结束**，归 (c)：
  小族里没有好得多的轨迹，选择不是杠杆。看到分数后不移动这条线，也不换族。
- **≥ +4.0 时的分支**（> 20° 桶上的单轴增益）：(a) 横向偏移：O3 − K3 的 CI 下界 > 0 且 O3 > V3 → 目标函数问题，走按分数重标的 target；
  (b) 曲率增益：O3 − K3 的 CI 上界 < 0 且 K3 > V3 → 基座带来的欠转，走更长训练 / 放松 anchor；
  (d) O3 − V3 与 K3 − V3 的 CI 上界都 < 0 → 增益在速度剖面，两个横向分支都不成立（如第 168 条）；其余 → 混合，照实报告。
  方向性佐证（报告，不进规则）：(b) 要求 oracle 选 g > 1 明显多于 g < 1、且固定 g > 1 的 cross-fit 增益为正；(a) 看偏移朝外是否多于朝内。
- **不做**：stage 2（重标 navtrain、任何训练）、GPU 作业。另只读地查两件事：box 上有没有 navtrain 的 metric cache、建一个要多少成本；
  navtrain 里 |dyaw| ≥ 20° 的 token 数（main 按 navtest 占比估 ~27 k，未核）。

## 执行

`scripts/turn_ceiling.py`（family / gate / report / navtrain，各自在 `jevdrive.run.Run` 里，token 成员取自 `jevdrive.data.splits` 的 `navsim/navtest`），
`scripts/turn_ceiling_chain.sh`（tmux `jev:turn-ceiling`，一条自推进链，`$DATA_DIR/runs/op_parity/turn_ceiling/chain/{STATUS, DONE, ERROR}`）。
分阶段 24 → 300 → 3 154。并行只有 score-poses 自己的 claim 队列，本线没有自己的并行循环，所以不用 `jevdrive.par`；poses 文件走 `jevdrive.cache`。
预算：stage 1 约 1 h 墙钟（估 66 key × ~1.03 core-s × 3 154 ≈ 60 core-h，72 核约 50 min）。

---

## 补记 A：stage 0a 闸门不过，修工具（2026-10-08，除恒等候选外任何分数读出之前写定）

24 token 的恒等闸门不过：两个 seed 都是 NC / DAC / DDC / TLC / TTC / LK / HC 全等，EP 在同样 2 个 token 上不等（存档 0.61 / 0.95，score-poses 1.0）。
按规则停下，只读了恒等行（外加只打恒等 key 的两次 24 token 复查）。原因：bench navtest 走 devkit 默认的 `traffic_agents: non_reactive`（log replay），
`score-poses` 写死 reactive IDM；EP 用 PDM 参考轨迹归一化，参考轨迹在 reactive 交通下撞了就不归一，所以两条路径本来不是同一个 metric。
存档的 89.55 / 91.71 与分桶表都是 non-reactive。

改动：`jevdrive/bench/poses.py` 加 `--traffic {reactive, non_reactive}`（默认 reactive，旧 run 目录的 identity 不变；non_reactive 进 identity），
本线全程用 `--traffic non_reactive`，与存档同一口径。其余（族、闸门容差、削减阶梯、判线 +4.0、分支规则）一字不改，stage 0 从 24 token 重新走。
reactive 那次 24 token 的 CSV 里非恒等行没有读过，也不会用。成本估计作废（IDM 占 reactive 成本的 97%，non-reactive 会便宜很多），以 300 token 实测为准。
