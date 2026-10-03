# 预登记：navhard 可行驶区（DAC）失败归因（2026-10-04，读各类拆分之前写）

问题：第 103 条里 navhard 最大的一块是 DAC（当前最好 driver 剩 13.0 分）。第 104 / 108 条已把相机几何排除为区分原因。
把 DAC 失败拆成原因，每类有计数、占比、CI、一个检查，以及只替换该类 token 的 oracle 可回收分。

写这份文件之前已经看过的：第 103 条的总数（DAC 全族 15.1 / 13.0、三类 d88 拆分）、第 88 条的 d88 类别计数。没有看过下面任何一个新类别的计数或分数。

## 数据
- 臂：原生 Cinque（`gimm-cinque__base`，33.33）与 it_dw3 + 选择器（`gimm-cinque_Oit_dw3-s0_al-sel-rot0-r0.6__base`，35.76）。
- 每 token 官方分行：`runs/leaderboard_audit/loss_budget/prep_navhard.pkl`（in-process devkit 打分，复现 33.33 / 35.76），参考 = PDM-Closed（同一 pkl）。
- 几何：metric cache（评分多边形 = ROADBLOCK ∪ INTERSECTION ∪ DRIVABLE_AREA ∪ CARPARK_AREA 的缓存层），nuPlan 原始地图 gpkg（generic_drivable_areas、carpark_areas、walkways）。
- 「DAC 失败」= 该臂该 token 官方 `drivable_area_compliance` < 1。stage 1、stage 2 分开计。

## 每个 DAC 失败 token 的检查（CPU，评分器自己的 LQR + 自行车回放）
- **S0 起点已在外**：t = 0 的车框角点已在多边形外。
- **M 地图窄**：整条回放里所有越出多边形的角点都落在（多边形 ∪ 原始 generic_drivable_areas ∪ carpark_areas）里，容差 0.1 m。walkways 单独记标志，不算可行驶。
- **W 方向反**：第 88 条规则（计划终点与参考终点 |y| > 1 m、异号、差 > 2 m）。
- **L 跟踪器滞后**：把计划本身（8 个位姿线性插值成 41 个状态，按计划航向画车框）当作理想跟踪，41 个状态全在多边形内，而 LQR 回放越界。
- **E 起点被挪偏的早越界**：stage 2，首次越界 ≤ 1.5 s，起点到路线中心线 |偏移| ≥ 0.5 m 或 |朝向差| ≥ 0.1 rad（第 88 条 early clip 规则）。
- **U 该转没转够**：参考需要转（|Δψ_ref(4 s)| ≥ 10° 或 |Δψ_ref(2 s)| ≥ 5°，后者即第 110 条集合），首次越界的角在弯的外侧，计划 4 s 朝向变化（与参考同号方向的投影）< 0.75 × |Δψ_ref(4 s)|。
- **O 转过头 / 直路上乱转**：参考需要转且越界角在弯内侧、计划转角 > 1.25 × 参考；或参考 |Δψ_ref(4 s)| < 5° 而计划 |Δψ(4 s)| ≥ 10°。
- **F 太快进弯**：计划 4 s 路程 ≥ 1.15 × 参考，且「计划路径按参考的路程–时间曲线重新取时」（路径不变，只换速度，回放同一 LQR）不越界。
- **C 变道 / 汇入**：参考 4 s 横向 |y| ≥ 2 m 且 |Δψ_ref(4 s)| < 10°。
- **R 其余**。
- 互斥主类按上面顺序取第一个命中的（S0 > M > W > L > E > U > O > F > C > R）：先排除评分器 / 起点造成的，再排计划方向，再排计划幅度与速度。所有标志同时以重叠形式报告。

重叠标志（不参与主类）：
- **参考也越界**（PDM 参考自己 DAC 失败）：oracle 替换对它们 DAC 无效，单独报。
- **第 110 条集合**（|Δψ_ref(2 s)| ≥ 5°）。
- **路口 / 弯 / 直路**：越界处 3 m 内有 INTERSECTION 或在 LANE_CONNECTOR 上 = 路口；否则参考 |Δψ_ref(4 s)| ≥ 10° = 弯；其余直路。
- **缺指令可修**：主类 U 或 W，且驾驶指令是左 / 右且与参考转向同号（原生 desire 关闭，第 88 条）。
- **合成历史**：stage 2 且原生 rot0（去历史转动，`gimm-cinque_al-rot0__base`）在同一 token 上回放不越界。
- **浅越界**：最大越出 < 0.3 m。

## 可回收分（oracle 上限，不是可得增益）
- 主读数：只把该主类的 DAC 失败 token 的轨迹项（NC、DAC、DDC、TLC、EP、TTC、LK）换成 PDM 参考，clip（不比自己差），comfort 保留，用 devkit 两阶段聚合重算（同第 103 条）。
- 次读数：只把这些 token 的 DAC 设为 1（其余项不变），回答「只修越界本身值多少」。
- CI：按两阶段映射组（`reactive_all_mapping` 的每个 (orig, prev) 对）做 2 000 次 bootstrap，给计数占比与可回收分的 95% 区间。
- 先核对：快速聚合实现与 devkit 的 `calculate_individual_mapping_scores` 对 native / best 相等到 1e-6；回放的几何 DAC 与官方 DAC 的一致率报出来（< 97% 则先查原因再分类）。

## 读法（事先定）
- 一个原因「是主要的」：主类占比的 CI 下界 ≥ 20%，或 DAC 设 1 的可回收分 CI 下界 ≥ 2 分。
- 「评分器类」（S0 + M + L）合计占比 ≥ 1/3 → 结论写成「DAC 一项相当部分不是驾驶错误」；< 15% → 写成「主要是计划本身」。
- 「幅度类」U 对 O：U / (U + O) ≥ 0.7 读作「欠转为主」（与第 88 条第 6 点、第 104 条第 7 点一致），≤ 0.3 读作「过转为主」，之间读作混合。
- 选择器 / 适配在每类上拿走了多少：两臂同一类计数差，按 token 配对（McNemar 式，计数与 CI）。
- 典型例子：每个主类挑 2–3 个（按越出量中位附近、stage 2 优先），画 BEV + openpilot 输入视角，不看分数挑。
