# 图像通道路线指令 Q2：轻量微调（lane IMG，2026-10-04）：预登记

写于任何训练之前（只跑过 6 个样本的 port smoke：port + 5 Hz 10 帧格点 vs TensorRT 31 步，4 s 点相差约 1–5 cm）。
来源：本目录 [2026-10-04-img-cmd-prereg.md](2026-10-04-img-cmd-prereg.md)（Q1 zero-shot = 部分吸收：barrier uptake 0.12、fill_grass 0.11，
涂色路线 ~0.01）。机器：L3 的 port 和训练配方（`experiments/op_adapt_h`、`op_adapt_l`），只 import 不改。

## 问题

Q2：轻量微调（stage 1–3 冻结，stage 4 + off-policy plan pathway 可训练，L3 量级）能不能让 Cinque 跟着画在图像里的路线走？
学到的是「读路线」还是只记住训练用的两种画法？

## 数据（全部 navtrain，与 Q1 的评测按 log 不相交）

| 池 | 内容 | 数量 |
|:--|:--|:--|
| 训练 | 第 93 条 pair-eligible 路口帧（≥ 2 出口类别、实际支路已知），log 不含任何 Q1 评测 token（385 路口 + 293 直行），token 不在 op_lb `lb_navtrain` / L3 `lb_h1train`；每个进口段（log, 分叉车道）≤ 3 帧；车道匹配 ≤ 1.5 m。外加同 log 的直行车道保持帧 | 1 500 路口 + 300 直行，375 log；按 log 分 train 1 364 + 270 / dev 136 + 30（`navsim/img-ft-{train,dev}`） |
| 评测 | Q1 的同一批：385 路口帧（247 log）+ 293 直行帧 | 不变 |

帧协议：op_lb 的 4 关键帧 + 6 GIMM 帧（新 data 名 `lb_imgtrain`），port 用 L3 的 5 Hz 10 帧格点（h_prep.NavSrc）。覆盖物在每个有效帧上用该帧源时刻的
ego 位姿画（与 img_run 相同的 img_overlay.draw）。

## 训练（一个臂 `ft`，seed 0）

- 训练族：**band**（涂色 = 路线）+ **barrier**（挡其他支路）。留出族：lines、arrow_road、sign、cones、wall、fill_grey、fill_grass（从未见过）。
- 每批 48 行：
  - I 24：路口帧 + 训练族覆盖物（指令 c 取该帧每个出口类别）；目标 = 支路 c 的路线：c = 实际支路时为日志未来；否则沿 c 的中心线（进口 + 支路），
    按日志未来的弧长–时间走（保留日志速度剖面），t0 时相对中心线的横向偏移在 max(8 m, 2.5 s × 速度) 内三次衰减到 0。
  - D 16（路口 10 + 直行 6）：无覆盖物，蒸馏到原模型（plan 一致 + 全部输出头）→ 无指令时不许学会猜支路。
  - C 8：直行帧上沿本车道画 band，plan 一致到原模型的 `none` plan（与车道一致的涂色不该改变什么）。
- 损失尺度沿用 op_adapt_l / L3（16 点后轴网格 Huber，lam_i 1、lam_d 10、lam_c 1），lr 3e-5，warmup 100，cosine，2 000 步，AdamW wd 0.01。
- 阶段 1–3 的输出预先算成 trunk bank（每个 (帧, 族, 指令) 一份）；训练和评测读同一 bank。

## 读数（评测集，port 上 O 与 ft 同一代码；指标与 img_report.py 相同，τ = 4 s；CI = 按 log 的配对 cluster bootstrap）

- 每族：Δ、uptake（满切换 = 1）、fixed-set correct（none = 机会水平）、toward L / S / R、dx（相对本模型自己的 `none`）；ft − O 的配对差。
- 直行帧：覆盖物引起的 3 s 横向误差变化（相对本模型 `none`）；ft `none` 相对 O `none` 的 3 s 横向误差。
- 护栏：`none` plan 对 O 的漂移中位（op_adapt 的 plan_drift，0–5 s 平均 L2）；`none` 时 4 s 点向实际支路的移动（ft − O）。
- 另报：port-O 对 TensorRT（Q1 的原始结果）在 `none` 上的一致性（4 s 点距离中位）。

## 判读线（事先定）

- **训练族学会**（band、barrier 各自判）：ft 的 uptake ≥ 0.5 **且** fixed-set correct ≥ 0.75，**且** ft − O 的 uptake 差 CI 下界 > 0。
  只有 CI > 0 但没到线 = **部分**；CI 含 0 = **没学会**。
- **泛化到留出族**：某留出族 ft − O 的 uptake 差 CI 下界 > 0 = 该族泛化。按「涂色 = 路线」组（lines、arrow_road、sign）与「挡 / 填其他支路」组
  （cones、wall、fill_grey、fill_grass）分别数；两组都有族泛化 → 学到的是「读路线」；只有与训练族同类的泛化 → 记住的是画法类别。
- **护栏**（任一不过 = 结论附「有代价」，不改判读）：
  - `none` 漂移中位 ≤ 0.10 m（路口、直行分别）；
  - 直行帧：ft `none` 的 3 s 横向误差 − O 的 ≤ +0.10 m（点估计），且各覆盖族在 ft 上引起的横向误差变化 ≤ +0.10 m；
  - 无覆盖物的路口帧不系统性选支路：`none` 向实际支路的移动 ft − O 的 CI 含 0，或 |均值| ≤ 0.20 m。

## 迭代规则（只看 dev，评测集每个模型只读一次）

- 训练完先在 dev（136 路口帧，band / barrier 两族）上算 uptake。若两族 dev uptake **都 < 0.30**：做一次迭代 `ft2`（lr 1e-4、3 000 步，其余不变），
  在本文件末尾追加一行再跑；评测报告 `ft` 和 `ft2` 两个，`ft2` 为主。否则不迭代，`ft` 为主。
- 评测集上的任何数不用来选模型或调参。

## 已知限制

覆盖物不被真实车辆遮挡；地面平面假设；评测样本多为停止线附近低速帧（距分叉点中位 3.3 m）；只有 1 个 seed；nav 一个域（WOD / CARLA 未训也未测）。

## 追加记录

（迭代与偏离按时间追加于此）
