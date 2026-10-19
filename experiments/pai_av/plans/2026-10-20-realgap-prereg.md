# REAL-GAP 预注册（2026-10-20，结果出来之前写）

**问题。** 在花几天下载 PAI-AV 相机（计划约 1 TB）用真实画面训练之前：同一 clip、同一时刻，真实画面和 NuRec 渲染给我们的 PAI 模型
（NTV0-L18-s0）和冻结的 Cinque 编码器带来多大差别？驱动里的 f-theta -> openpilot 转换在真实画面上是否正确？

**样本。** `results/real_sample.csv`（`scripts/pav_real.py pick`）：test281 里在 PAI-AV 中、盒子上有 usdz 的 186 个 clip 中分层抽 50
（split `physical_ai_av/realgap-exam50`），PAI-AV train 划分、不在 curated-val（441）里的 clip 抽 50（`realgap-nonexam50`）；分层 =
国家组（US / DE / 其他）x 白天 / 夜间（hour_of_day 6-18）。相机按 zip 成员逐个 range 下载，约 1.3 GB，超 3 GB 中止。
**test281 的真实画面只做测量，永不进入训练行或训练缓存。**

**渲染侧。** 50 个考试场景沿人类日志回放（`onp_log_driver.py`，NTV0-L18-s0 喂入并记录，PAI_DUMP 每秒存一次 8 个槽帧 + ego 行 +
源 JPEG）。**真实侧**：同一时刻（rollout 时钟 = NuRec 绝对时钟，减去 clip 起点 = PAI-AV clip 时钟）从视频取最近帧，经驱动同一个
`pai_core.Maps`，分别用 PAI-AV 标定和会话的 NuRec CameraSpec。ego 行、路线记忆、命令全用渲染那一步的，只换图像。

**读数。**
- A 转换：两套标定的旋转 / 平移 / 主点 / f-theta 半径差；时钟偏差；图像滞后扫描（-3..+3 帧）；渲染 vs 真实 road 帧相位相关平移（度）。
- B 计划：ADE / FDE / 4 s 速度偏差 vs 人类日志，真实 - 渲染配对差（场景聚类 bootstrap）；真实 vs 渲染计划距离，对照尺 = 同一渲染输入上
  NTV0 vs RSG-L18-s0 的计划距离。
- C 特征：t0 槽 token 余弦（对照：真实 t vs t+1 s；不同 clip 的空对照）、线性 CKA、域分类准确率。
- D 探针：岭回归（速度、3 s 速度变化、3 s 航向变化），在真实非考试 clip 上训练 -> 考试真实 vs 渲染；考试内按场景 5 折交叉域。

**判读规则（事先定）。**
1. 转换正确 = 旋转差中位数 < 0.2 度、road 帧平移中位数 < 0.3 度、最佳滞后为 0 帧。否则先修转换，不谈下载。
2. 差距"小" = 同时满足：(a) |真实 - 渲染 ADE| 的 CI 落在渲染 ADE 的 ±10 % 内；(b) 真实 vs 渲染计划距离 <= NTV0 vs RSG 计划距离；
   (c) 渲染-真实 token 余弦中位数 >= 真实 t vs t+1 s 的中位数；(d) 真实非考试训练的探针迁移到渲染，速度与速度变化的 R2 损失 <= 0.10。
3. 下载规模：差距小 -> 真实行与渲染行对模型几乎等价，真实数据的价值只在"更多驾驶情形"，按 2-4k clip 试点规模下载并先做一个联合训练臂，
   不直接全量；差距大且探针不迁移 -> 真实行教的东西到不了渲染考场，全量 1 TB 不支持，至多试点；差距大但探针迁移 -> 试点规模，
   视觉差主要在像素层面。任何情况下全量下载都要等试点在 PAI 板上给出正向读数。
