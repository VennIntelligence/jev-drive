# CORR0 预登记：「指令出口的车道 corridor」是不是修转弯失败的正确监督标签（2026-10-10，lane CORR0）

只测量：存档 plan（`SH30-F-s0/s1`）、存档逐 token 分数、nuPlan 地图、日志。不训练、不用 GPU、不做闭环、不做新的 WA-JEPA 推理。
**地图、日志未来轨迹、日志实际走的车道序列都是特权输入，只用于分析，不是方法输入。** 本文在任何一个读数之前提交并 push；偏离以带日期的补记写在受影响的读数之前。
接第 207 条（swap 机制、`score-poses --traffic non_reactive`、no-EC EPDMS × 100、按 log 的 cluster bootstrap B 10 000、seed 均值）、204、166、234、93。
token：navtest 12 146；转弯桶 = |日志 4 s heading 变化| > 20°（3 154）与 > 45°（1 517）。失败类别（切内角 / 转不过去）用第 153 / 207 条的定义与第 207 条已有的 replay。

## 共用几何（一次建好、缓存）
- **driven sequence D**：日志 t0 起至多 20 帧（10 s）的全局位姿，在 lane graph（LANE + LANE_CONNECTOR）上做 Viterbi：每帧候选 = 含该点且 heading 差 < 60° 的对象（没有则取 3 m 内最近的）；转移只允许同一对象、后继（至多跳过 2 个）、同 roadblock 相邻车道（变道）；代价 = heading 差 + 到 centreline 的横向距离。
  **匹配失败** = t0 无候选，或 0–4 s 的 9 个位姿没有连通的序列。转弯 token 失败率 > 10% 则停下只报这一点。
- **corridor centreline R**：D 的 baseline 串接，从 ego 在其上的投影起；D 用完之后沿 heading 变化最小的后继延长。变道 token 取变道后的车道链，单独列出。
- **主 re-target 路径 CP**（主定义）：R 加上 ego 在 t0 的横向偏移 d0，d0 在前 8 m 弧长内按 smoothstep 衰减到 0（变道 token：衰减长度取日志完成变道的弧长，不短于 8 m）；yaw 取路径切向。
  次定义 KP：日志曲线投到 corridor 内部，即日志相对 R 的横向偏移截到 ±(半车道宽 − 半车宽 − 0.2 m)，不足时取 0。

## 四个数
1. **误差分解**（每个 token-seed）。plan 的 9 个位姿用同一 Viterbi 匹配。
   (a) **走错出口 / 车道**：plan 末端匹配到的对象不在 D 的链（含上下游延长）上；细分 a1 不同出口（在分叉处进了另一条 connector，出口 roadblock 不同）、a2 同出口的另一条车道。plan 末端匹配不上的单列为 unmatched。
   其余按曲线拟合分：等弧长、DS 0.05 m、弧长取两条曲线的公共长度；
   (b) **along-track**：模型 A = 日志曲线沿自身提前 / 推迟 δ（|δ| ≤ 6 m；推迟 = 先直行 δ 再接日志曲线，提前 = 日志曲线从 |δ| 处起、以该处位姿重新对到原点）；
   (c) **cross-track**：模型 C = 日志曲线的平行偏移 c（|c| ≤ 3 m）。
   残差 RMS 小的那个模型定类；同时报 r0 = RMS|P − L| < 0.3 m 的「小误差」份额与两个模型都解释不到一半（min r > 0.5 r0）的「未解释」份额。
   读数：各类的 token 份额、DAC 失败份额、失分份额（失分 = LL − PP 的 no-EC EPDMS 之和），分切内角 / 转不过去，> 20° 与 > 45°。
2. **re-target swap**。plan 的逐 token 曲线误差 e(s) = 等弧长下 plan 曲线 − 日志曲线（第 207 条的曲线与 ext-arc 规则），在日志曲线的局部系里取（切向、法向、yaw 差），加到 CP 的同弧长点的局部系上，按 plan 自己的弧长-时间走：臂 **CE**（主读数）。
   参考臂：**CP** 按 plan timing、不加误差（上参考）；PP（存档 plan，恒等闸门：8 个子分与存档逐 token 相同）；LP 用第 207 条已存的分数，不重打。次定义同样两臂 KE / KP。
3. **地图 + 指令够不够给出 4 s heading**。R 的切向 heading（相对 t0 的 ego heading，不加 d0 混合），取在 plan 自己的 4 s 弧长处与日志的 4 s 弧长处，对日志 4 s heading 的误差，RMS（另报稳健 σ 与均值绝对值）；
   主集合是 navtest 全部 12 146 token（第 204 条的 7.4° / 3.7° 是这个口径），同时列 > 20° / > 45° 及 SH30 自己在同一集合上的 RMS。
   **泄漏的内容**：t0 在地图上的精确定位与所在车道；视野内每个分叉处日志选的后继（出口选择）；变道 token 的目标车道；「日志弧长」一栏还泄漏日志的 4 s 行程。不泄漏：日志在车道内的横向位置与路径形状。
   便宜的话加 exit pose 变体：D 中前方第一条 connector 结束处（出口车道起点）的 heading 对日志 4 s heading。
4. **反事实供给**（navtrain 103 288 token，及其中 |日志 4 s heading| > 20° 的 28 323 个）。从 ego 匹配到的对象沿 lane graph 向下游枚举 40 m 弧长内的全部路径（按末端对象去重）；
   **exit heading** = 该路径 centreline 在 40 m 处（或路径末端）的 heading。日志走的那条用同一 Viterbi 定。
   计数：路径数 ≥ 2 的 token，按备选条数（1 / 2 / 3+）；备选与日志所走路径的 exit heading 差 ≥ 20° / ≥ 45° 的 token 数与备选行数；同样按 log 计（有 ≥ 1 行的 log 数、每 log 行数的分位）。
   只算 ego 自己车道的后继（主）；同 roadblock 任一车道的后继为次读数。日志路径定不出的 token 单列。

## 读线（读数，不是 go / no-go）
- N1：> 45° 的 DAC 失败多数落在哪一类。走错出口 → 反事实指令是主杠杆；along-track → 定位精度，encoder 的工作前移；cross-track → target 的摆放 / hinge。
- N2：CE 对 PP 在 > 20° 上的 no-EC EPDMS 增益与去掉的 DAC 失败份额（gross / net），带 CI；「接近零」= > 20° 桶的 CI 上界 < +1.0。并列 LP（第 207 条 +7.39）。
- N3：地图 heading 的 RMS 误差 ≤ 3.7° 过，> 7.4° 不过，之间为部分。
- N4：有 ≥ 20° 备选的反事实行：报数；< 5 000 token 或 < 200 log 则标出。

## 步骤
先在约 10 个代表 token（切内角、转不过去、通过；左、右）上跑通四项并看 BEV 图核对车道序列与 corridor 几何，再全量。N4 与 navtest 无关，并行跑。bootstrap 单元 = log。

## 补记 A（2026-10-10，10 个 token 的匹配核对之后、任何四项读数之前）
只看了 10 个代表 token 的匹配状态与车道序列，没有任何分数或误差读数。
1. **匹配容许缺帧**。10 个里 1 个「转不过去」token 的日志在 connector 内有 4 帧没有候选（偏离 connector centreline 的 heading 差 > 60° 或距离 > 3 m）。改为：没有候选的帧跳过；失败 = t0 无候选，或 0–4 s 的 9 个位姿里有候选的不足 5 个，或有候选的帧之间没有连通序列。缺帧数（0–4 s）逐 token 记录并在结果里报：它本身是「日志路径不在 corridor 内」的读数。
2. **plan 末端无候选**。10 个里 1 个切内角 token 的两个 seed 的 plan 末端都没有候选。登记的 unmatched 行不变；另加一行 fallback：用最后一个有候选的 plan 位姿定类，并报它离末端几个位姿。
3. **重叠 polygon 的歧义**。connector polygon 在路口内互相重叠。plan 的 Viterbi 末端对象不在 D 的链上、但该位姿同时也是链上某对象的候选时，记为 same（细分 same_amb 单列），不计入 (a)：(a) 只在 plan 末端位姿不可能属于日志所走的链时成立，是偏保守的口径。a2 = 末端对象与链上某对象同 lane group / 同 roadblock connector，否则 a1。
4. **KP（次定义）的起点**。t0 时 ego 可能已在 clamp 限之外（10 个里 d0 最大 1.83 m）。clamp 的横移量乘以 smoothstep(弧长 / 衰减长度)，与 CP 的 d0 衰减同长度，使 KP 仍从 ego 出发。
5. 变道转移加固定代价 0.5（防止 polygon 重叠处的假变道），属实现细节。

## 补记 B（2026-10-10，第一次全量读数之后：实现错误，第一次读数作废）
第一次全量跑完（恒等闸门过，3 083 个匹配上的转弯 token）后读了四项的表。第 2 项里 CP（不加误差的 corridor 路径）比存档 plan 低 6.28、DAC 失败多一倍，与「车道 centreline 在路内」不符，于是画了 10 个 CP 新增失败 token 的 BEV，查出两处**几何实现错误**（不是定义改动）：
1. `centreline()` 只把 ego 投影到序列的第一个对象上。t0 时 ego 已过该对象末端（在路口里、只匹配到 3 m 内的上游车道）时，R 从 ego 身后数米开始；
2. 变道 token 取的是 10 s 窗口内**最后一次**变道之后的车道链，变道发生在 4 s 之后时 R 在十几米之外（d0 到 16.7 m）。
修正：ego 投影到整条链（前 80 m）上；corridor 取日志在 4 s 时所在的那一段车道 run（4 s 之后的变道不计，之后沿最直后继延长），变道 token = 0–4 s 内换过 run 的 token，衰减长度取到进入该 run 的日志弧长。新增 sanity 列 `start_x`（R 起点在 ego 系的纵向坐标，应为 0）。
**第一次读数全部作废**，受影响的是第 1 项（plan 末端的链）、第 2 项（CP / CE / KP / KE）、第 3 项（R 的 heading）；第 4 项不经过这两处，不重跑。作废的数（> 20°）：CE −11.62 [−13.64, −9.56]，CP −6.28 [−8.71, −3.79]，KE −6.88，KP −4.30；地图 heading RMS 全体 5.52° / 4.57°（plan 弧长 / 日志弧长）。读线、类别定义、主 re-target 路径的定义不变。
