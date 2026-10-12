# FAMILY-Q 预注册：重建质量是不是 openpilot 家族独掉的原因（2026-10-12，读任何回归之前写）

## 背景

第 263 条：标定失配量级够、但 APY 排序反了、找不到来源；第 265 条（补正）：长度不是原因。新假设：private 的 MTGS 重建质量更差，
而我们的家族对重建质量比 LTF / DD / garage 敏感。注意：`results/asset_grade_public.csv` 是**路面坡度**（日志 ego 俯仰、高度变化），
不是重建质量；这里另算质量量。

## 质量量（每个 asset 一个值，CPU，全部 1491 个；`scripts/asset_quality.py`）

- Q1 位姿修正量：asset 的 `ego2global`（重建后的位姿）对 `ego2global_original`（原始日志位姿），以 scene 起点帧（frame 18）对齐后，
  frame 18..68 的平移残差中位数（m）和旋转残差中位数（deg）。大 = 重建对日志位姿改得多（位姿图不一致）。
- Q2 内参异常：|fy/fx − 1|（colmap 精化后的 CAM_F0；fy/fx 0.53 的那种会被渲染成压扁的图）。
- Q3 畸变偏离：|k1_colmap − k1_raw|。
- Q4 背景模型大小：background ckpt 文件字节数（高斯数的代理）。
- Q5（只做验证，96 个 scene）：k=0 渲染帧对同刻真实 nuPlan 帧的光度误差（灰度、均值方差归一化后的 L1）和边缘 NCC；渲染帧来自对照 arm
  重跑一遍时存下的 JPEG（`SH30_DUMP=96`，同一闭环，与 A0 逐 scene 相同）。用来检验 Q1–Q4 里哪个与渲染质量相关；Q5 本身样本太少，不作主读。
- 协变量（非质量）：路面坡度 `pitch_mean`、`dz_end`。

## 读数

驱动：YR10m10-F-s0、APY10m10-AB-s0、WA-JEPA（aud1 存储）、LTF、DD、garage 0014（proxy anchors）；1479 个有指标的 scene。
结果量：scene score、GT distance（截断后每 clip 最大值）、k=10 离轨距离（有 per-decision 表的驱动）。
每个驱动、每个质量量（标准化成 z 分数）做单变量 OLS 斜率，CI 用 44 个 log 的聚类 bootstrap（4 000 次）；
**家族差** = mean(YR, APY) 的斜率 − mean(LTF, DD, garage) 的斜率，同一套 bootstrap 抽样。

判据：某个质量量上家族差对 score 为负（我们更陡）且 95 % CI 不含 0，且对 GT distance 同向（我们 GT 涨得更快），记为「家族更敏感」。
4 个质量量 × 2 个结果量，共 8 个检验，按 Bonferroni 报 99.4 % CI 作为严格版本。
尾部：每个质量量最差 10 % 的 scene（以及最差 5 %）上，各驱动的平均分 / GT、对其余 90 % 的差（bootstrap CI）。
外推：用斜率预测质量量在 public 第 95 百分位处的分数（带 CI）；private 的质量分布未知，只给出「如果 private 整体落在 public 的尾部」时的数。

## 不做

不改任何驱动或第三方代码；GPU 只用于那一个 96 scene 的存图 job（已在跑，约 7 min）。
