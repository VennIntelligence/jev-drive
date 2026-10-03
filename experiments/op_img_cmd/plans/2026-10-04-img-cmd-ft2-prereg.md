# 图像通道路线指令 Q3：不漂移的微调 + 闭环 smoke（lane IMG2，2026-10-04）：预登记

写于任何新训练之前。只看过 Q2 已有模型（ft2-s0 / ft3-s0）的漂移分解（下面「诊断」），没有新模型的任何读数。
来源：[2026-10-04-img-cmd-ft-prereg.md](2026-10-04-img-cmd-ft-prereg.md)、[../results/ft-q2.md](../results/ft-q2.md)、第 99 条。

## 诊断（已有数据，决定了设计）

ft2-s0 的 `none` 漂移按池拆开（0–5 s 平均 L2 的中位，|dx| / |dy| 为同口径的纵 / 横分量中位）：

| 池 | 路口 drift | 路口 \|dx\| / \|dy\| | 直行 drift |
|:--|:--|:--|:--|
| 训练帧（D 行本身） | 0.18 | 0.16 / 0.03 | 0.07 |
| dev（同分布、留出 log） | 0.28 | 0.26 / 0.04 | 0.20 |
| 评测 | 0.29 | 0.22 / 0.07 | 0.18 |

ft3（蒸馏 ×4）：训练帧 0.095 / 0.04，评测 0.195 / 0.14。两点：(1) 漂移主要是**纵向**（速度剖面），横向只有 3–7 cm；(2) 训练帧与留出帧差一倍，D 池太小（1 634 帧被看 ~30 遍）。
纵向漂移的来源：I 行的目标用日志速度，原模型 4 s 平均比日志短 3 m，覆盖物于是学成了「走」，并漏到无覆盖输入上。

## 改动（三个配置共有）

1. **I 行目标的时间剖面用原模型自己的**：支路 c 的中心线（含进口，t0 横向偏移按原规则三次衰减），弧长–时间取原模型同一帧 `none` plan 的弧长（不再用日志速度）。
   所有类别（含实际支路）一律用中心线，类别之间对称。覆盖物只该改横向路线，不该改速度。
2. **大蒸馏池**（D 行，无覆盖，教师 = 原模型）：L3 的 `op_adapt_H/samples` nav（lb_h1train，去掉含评测 token 的 log）/ WOD / CARLA（p6，路线不在 bench2drive220）的 train 切分，约 6 900 帧；加上 Q2 训练池的无覆盖帧和 CARLA 路口训练帧的无覆盖帧。
3. **CARLA 路口帧进训练**：179 条重录路口按路线分 train / dev / test。test 先放入闭环 smoke 要用的路线和第 95 条 19 条集合里出现在 179 中的 13 条（27043 15102 24944 27870 27297 9196 28147 16390 15612 15483 17280 16529 16508），再随机补到 60 条；dev 18 条；其余 train。train 中与 test 共用 (map, 路口节点) 的样本删掉。训练族仍是 band + barrier。
4. **负样本**（N 行，plan 一致到原模型同一帧的 `none` plan）：直行帧上沿本车道画 band（Q2 的 C 行）；路口帧上 `band_all`（每条支路都涂，不带路线信息；nav 训练池和 CARLA train）。

每批 56 行：I 24（nav 16、CARLA 8）；D 22（Q2 池路口 4、直行 2、CARLA 路口 2、L3 nav 6、WOD 4、CARLA p6 4）；N 10（直行 band 4、nav band_all 4、CARLA band_all 2）。
其余沿用 ft2：stage 4 + plan 通路可训，lr 1e-4，warmup 100，cosine，3 000 步，AdamW wd 0.01，seed 0。

## 三个配置（同时训，只按 dev 选）

| 配置 | lam_d（D 行蒸馏全部输出头） | lam_c（D + N 行 plan 一致） |
|:--|:--|:--|
| A | 10 | 1 |
| B | 30 | 3 |
| C | 60 | 6 |

## 读数

- dev：Q2 dev（136 路口 / 30 直行，nav）、CARLA dev 路线、L3 三个域的 dev 切分。量 band / barrier uptake（τ = 4 s，固定集合，同 img_report）、`none` 漂移中位（对 port-O）。
- 考题（每个模型只读一次）：Q1 评测集（385 路口 / 293 直行，navtrain）和 CARLA test 路线；指标与 Q2 相同（Δ、uptake、fixed-set correct、各留出族、直行帧横向误差、`none` 向实际支路的移动），配对 cluster bootstrap（nav 按 log，CARLA 按路线）。
- 另报：覆盖物让 plan 纵向变化多少（dx，对本模型自己的 `none`），检查「走」的信号是否消失。

## 判读线（事先定）

闭环只能画 band（只有官方路线，没有其他支路的几何；不调 CARLA 地图），所以 **band 是门槛族**，barrier 只报告。

- **漂移护栏**：`none` 漂移中位 ≤ 0.10 m，分别在 nav 路口、nav 直行、CARLA 路口三处都过。
- **uptake**：band uptake ≥ 0.4（点估计）在 nav 评测和 CARLA test 都过。
- **车道保持不变**：直行帧 ft `none` 的 3 s 横向误差 − O ≤ +0.10 m，且各覆盖族在直行帧上引起的横向误差变化 ≤ +0.10 m。
- **不猜支路**：无覆盖时向实际支路的移动 ft − O 的 CI 含 0 或 |均值| ≤ 0.20 m。

### 选择与迭代

- dev 选：三个配置里 dev 漂移（nav 路口、nav 直行、CARLA 路口）都 ≤ 0.10 m、且 band dev uptake 在 nav 和 CARLA 都 ≥ 0.45 的，取 band dev uptake（两域较小者）最高的一个。
- 没有配置过 dev 线：不再加第四个配置，停下报告原因（三个配置给出漂移–uptake 的取舍曲线）。三个配置仍在考题上各读一次，供报告。
- 过 dev 线：该配置再训 seed 1、2。**通过** = 3 个 seed 每个都过漂移护栏和车道保持，且 band uptake 3-seed 均值 ≥ 0.4、每个 seed 点估计 ≥ 0.35，nav 和 CARLA 都算。通过才做闭环 smoke。

## 闭环 smoke（只在上面通过时做）

- 模型：选中配置的 seed 0，op_l_onnx.py 导出 ONNX，经 op_arb server 服务。
- 覆盖物：agent / server 每帧用官方路线（`set_global_plan` 的稠密路线，RoadOption）画 band：进入路线指令段（第 95 条的路口窗口：LEFT / RIGHT / STRAIGHT 段起点前后）时，沿路线折线画 band，坐标由 ego 位姿换到当前帧；不调 CARLA 地图。
- 路线：turn_agree 失败的 4 条 27297、27043、9196、24944（drive 的 turn_agree 0.36 / 0.43 / 0.46 / 0.40，lmain 0.32 / 0.37 / 0.38 / 0.48），2 个 seed；对照 = 原生 drive 臂（同路线同 seed 新跑）。
- 读数：turn_agree（op_l_b2d_report 口径，TURN_WIN 15 m、AGREE 1.0 m）、DS、路线偏离。预期线：turn_agree 比 drive 高 ≥ 0.10（op_adapt_l B3 的同一条线），DS 不低于 drive 超过 5 分（只描述，n 太小不做显著性）。

## 已知限制

覆盖物不被真实车辆遮挡；地面平面假设（CARLA 有 44 个样本 |dz30| > 0.5 m）；nav 评测样本多为停止线附近低速帧；时间剖面用原模型的，若原模型在某帧本来不动，该帧的 uptake 天然小。

## 追加记录

（迭代与偏离按时间追加于此）
