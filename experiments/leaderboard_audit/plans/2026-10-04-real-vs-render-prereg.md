# 预登记：同位姿真实帧 vs 渲染帧，openpilot 输出差（2026-10-04，跑 openpilot 之前写）

问题（第 108 条的下一步）：openpilot 在 WOD / navtest（真实画面）正常、在 navhard 第二阶段 / HUGSIM / B2D（合成画面）不正常；几何已排除。
渲染域本身是否让 openpilot 的输出偏离？简单的图像端修补能不能补上？

## 数据与配对
- **HUGSIM nuScenes 19 个场景**（`datasets/hugsim/scenes/nuscenes`，scene-0010 … scene-0930；scene-0383 有官方训练原图）。
  每个场景 180 帧 × 三个前相机（FRONT / FRONT_LEFT / FRONT_RIGHT），名义 12 Hz、15 s。
  - 真实帧：nuScenes 原 JPEG 1600×900 → 800×450（INTER_AREA）。帧对齐规则：第 i 帧 = 时间上最接近「该相机首个关键帧 + meta timestamp_i」的 sweep；
    在 scene-0383 上与 HUGSIM 自带训练原图逐帧比对，三相机 PSNR 42–44 dB（即 JPEG 重编码误差），规则成立。
  - 渲染帧：同一 3DGS 在 meta_data 记录的相机位姿 / 内参 / 动态物体位姿下渲染（训练视角，渲染器的最好情况）。
    闭环里 HUGSIM 相机还被压低 0.3 m 且离开日志轨迹，所以这里测到的差是**下界**。
  - 适配器：两边一律用闭环同一个 `hugsim_zs.OpenpilotFrames(calibs(nuscenes_camera.yaml, cam_rect))`，Cinque（ORT-TRT），desire 0，traffic (1,0)。
- **navhard 第二阶段**：合成帧与日志帧共享时间戳，但合成自车位姿离日志位姿多数 > 1 m；距离 < 0.5 m 且偏航 < 1° 的只有 146 帧（< 0.2 m 27 帧）。
  只在这些帧上做单帧（同一帧重复 1.5 s）配对，车道线 / 路沿按位姿差做刚体补偿；结论只作旁证。
- CARLA：无真实对应，不做配对。

## 协议（HUGSIM）
1. **连续流**：每个场景、每个臂从零状态 20 Hz 流过 15 s（每步取最新的 12 Hz 帧），在每个新帧到达的那一步读输出；只用 ≥ 3 s（帧 36 之后）。
2. **历史偏航探针**（第 92 / 100 条口径）：从帧 18 起每 12 帧一个探针帧 k；零状态、1.5 s 窗口（31 步）。
   注入 ±1°/s 与 ±10°/s 假偏航（过去帧用偏航虚拟相机重采样），G_w = (左 − 右)/2 的 3 s 计划朝向（度，左正）。
   起步增益 L：帧 k 静止 5 s，然后 1 s 内偏航从 0 线性到 ±1°（只转不移），L = (左 − 右)/2 的 3 s 朝向 / 1°。
   按探针帧自车速度分档：停（< 0.5 m/s）、低（0.5–3）、中（≥ 3）。

## 读数
每个帧对的绝对差：本车左右车道线 y@10 m、@20 m（两边 p > 0.5）；路沿 y@10 m；车道线概率；前车概率、前车距离（两边 p > 0.5）；
计划横向 y@2 s、@4 s；计划速度 v@0 与 x@4 s；计划朝向 @3 s；另记模型路面 z 与车道宽（尺度）。
**主读数（4 个）**：计划横向 @4 s、路沿 y@10 m、车道线 y@10 m、计划速度 v@0。其余为次要。

## 线（事先定）
- **底噪**：真实 vs 真实相邻帧 |o_real(k) − o_real(k+1)|（83 ms）。
- **渲染差有意义**：RS = |o_real(k) − o_render(k)|；比值 median(RS) / median(RR)，按场景 cluster bootstrap 95% CI，**下界 > 1** 判有意义；
  4 个主读数中 ≥ 2 个有意义 → 判「渲染域让 openpilot 偏离」= yes；0 个 → no；1 个 → partly。
- **偏航敏感度**：G_1、G_10、L 的配对比 render / real（场景 bootstrap），CI 不含 1 且偏离 ≥ 20% 判「渲染改变偏航敏感度」。
- **图像端修补**（在 5 个标定场景 scene[::4] 上拟合参数，在其余 14 个上评）：
  F1 锐化（unsharp，强度 / 半径按高频能量占比对齐真实帧）；F2 颜色（每通道均值 / 方差对齐）+ 噪声（按 Immerkær 噪声估计补到真实帧水平）；F3 = F2 + F1。
  「有效」= 主读数上 closure = 1 − median|real − fix(render)| / median|real − render| ≥ 0.5，且「不伤真实帧」：
  median|o_real − o_fix(real)| ≤ 底噪 median(RR)。修补若被采用，是 HUGSIM 专用 trick，单独标注。
- 判读：渲染差 yes 且有修补有效 → 渲染域是原因之一且可廉价修；渲染差 no → 渲染帧本身（训练视角）不是原因，问题在新视角 / 闭环 / 相机压低。
