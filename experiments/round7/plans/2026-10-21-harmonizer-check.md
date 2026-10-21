# Harmonizer 开环检查：登记线（2026-10-21，写于任何渲染 / 编码 / 计划差计算之前）

来源：[2026-10-21-round7-framework-draft.md](2026-10-21-round7-framework-draft.md)「Harmonizer 风险」。问题：同一批 PAI 状态，NuRec 渲染器开 /
关 DiffusionHarmonizer，冻结编码器 token 与我们的计划移动多少？结果写 `experiments/round7/results/harmonizer_openloop.md`。

## 做法

- 状态：`on1_PAIH`（人类状态行，d410）的训练场景（scenes.tsv `held = False`），按行的人类未来轨迹分 5 层，每层 60 个状态（每场景每层至多 6 个、
  相邻至少隔 5 个决策），选点只用 10-20 已建的 H 行文件（`$DATA_DIR/runs/pai_rows/rows/H`），在任何新渲染之前定好：
  - 起步 launch：v0 < 0.3 m/s，人 4 s 内走 > 2 m
  - 停车接近 stop：v0 > 3 m/s，人在 3.5–4 s 段的速度 < 1 m/s，4 s 航向变化 < 15°
  - 路口转弯 turn：4 s 航向变化 > 35°，v0 > 1 m/s
  - 弯道 curve：4 s 航向变化 6°–20°，v0 > 5 m/s，末段速度 > 0.7 v0
  - 直路 straight：4 s 航向变化 < 1.7°，4 s 横向 < 0.5 m，v0 > 5 m/s，末段速度 > 0.7 v0
  - 优先级按上面顺序；只取 usable、非冷启动、非事件后、log_fut 8 点齐全的行
- 渲染：所选场景用 d410 的人类状态路径（`onp_log_driver.py`，A0 serve，NTV0-L18-s0 在环、自车跟日志）跑三遍，分块完全相同：
  OFF-A、OFF-B（重渲噪声底）、ON（`HARMONIZER=1`，`--enable-harmonizer` 保留、权重来自 `$DATA_DIR/datasets/nurec/harmonizer`）。
  每遍 `onp_rows` → `onp_cache build`（普通缓存同序编码，d410 的教训）。状态按 (场景, k) 对齐。
- 自身状态行（MV 系自己的 rollout）：需要一个按已录轨迹回放的 driver 才能让三遍渲染同一组位姿，暂不做（若人类状态结果已经很大或很小，结论不依赖它）。
- 比较：
  - token：每个状态最新一对帧的 token（32 × 512），ON 对 OFF-A 与 OFF-B 对 OFF-A 的余弦距离（1 − cos）与相对 L2（‖Δ‖ / ‖OFF-A‖）；只描述，不设线。
  - 计划：NTV0-L18-s0、R5LC-L18-s0（P0 出厂底作参照，不进判定），各遍用各自的 token 与 ego 前向（onp_cache gid 的出口：后轴系 x, y），
    0.5 / 2 / 4 s 处 |Δ横向|（y）与 |Δ纵向|（x）分开，报中位与 p90，按层与合并。

## 判定线（合并 300 个状态，NTV0 与 R5LC 各自判）

- **「小」**：ON 对 OFF-A 的 |Δ| 中位在 0.5 / 2 / 4 s、横向与纵向的 6 个格子里都 ≤ 2 × 噪声底（OFF-B 对 OFF-A 的同一格中位），
  **且** 4 s 横向 |Δ| 中位 < 0.2 m。两个模型都「小」→ 结论「小」，闭环配对照计划排在第 6 轮之后、不加急，不做增强。
- 否则 **「需闭环配对」**：闭环配对读数（PAI-FIX TARGET 60 + LAUNCH-37，开 / 关各一次）提前，增强（部分训练行用 Harmonizer 渲染）进入第 7 轮排期。
- 按层：任何一层 4 s 横向中位 ≥ 0.2 m，或该层 4 s 纵向中位 > 2 × 该层噪声底且 > 0.3 m，单独标出该层（即使合并判「小」）。
- **噪声底退化**：若渲染器确定（OFF-B 与 OFF-A 的帧逐字节相同或某格噪声底中位 < 0.01 m），「2 × 噪声底」那一格改用绝对线：
  该格 ON 对 OFF-A 中位 < 0.05 m（0.5 s）/ 0.15 m（2 s）/ 0.3 m（4 s），纵横同线；4 s 横向 0.2 m 线不变。
- 代价：ON 与 OFF 每场景墙钟、卡时一并报告（10-09 Tokyo 3090 全相机为 7.1 倍）。

## 不做

- 不改 decisions.md（主会话写）；不碰第 6 轮各道；作业 PRIO 16。
