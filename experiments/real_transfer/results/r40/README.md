# R40：第 40 条「Lebowski `cls_late` 够到原生」按原判据 3 seed 重判

零拟合。代码原样：`jevdrive.drive_backbones.heads_train`（seed 只改 K=1024 词表的 k-means 种子与 cls λ 的内层 sequence 划分，
`ridge_late` 无随机性）+ `heads_readout`（`cls_late` − 原生的 frame-mean 配对差、sequence bootstrap CI；G = (cls_late − ridge_late) / (native − ridge_late)，
同一 bootstrap 比值 CI）。判据与第 40 条原文一字不改：CI 跨零 = 够到。

## 复现检查（先于任何 3 seed 判定）

第 40 条的数字来自 run `drive_backbones/heads_train/20260925-110819`（train 训协议，无 `--seed` 参数，等价 seed 0）：
`cls_late − 原生` 逐位重算得 Cinque **−0.2886 [−0.4722, −0.1001]**、Lebowski **−0.1012 [−0.2922, +0.0936]**，
与第 40 条写的 −0.29 [−0.47, −0.10] / −0.10 [−0.29, +0.09] 一致（四舍五入到两位小数完全相同）——**这一格逐位复现**。

用 `--seed 0` 显式重跑一次（`heads_train-seed0/20260926-011909`，[SEEDS] 队列 01:31 已记录的复现探针）不能逐位复现：
`cls_late` 的 RFS cluster mean 与上面差 0.015（Cinque）/ 0.051（Lebowski），在夜间日志已定性的「WOD 分类头 GPU k-means/L-BFGS 非确定性、
汇总量差 0.00–0.05」范围内，不是新问题，是已知噪声。按此结果**不停**，继续用 seed 0（decision-40 源 run）+ seed 1 + seed 2 三点判定；
`--seed 0` 重跑单独列在下表底部作噪声参照，不计入三点。

## 3 seed × 2 模型

| 模型 | seed | run | `cls_late` − 原生 [95% CI] | 够到原生？ | G [95% CI] |
|:--|:--|:--|:--|:--|:--|
| Cinque | 0（第 40 条源 run） | `heads_train/20260925-110819` | −0.289 [−0.472, −0.100] | 否 | 0.403 [−0.021, 0.753] |
| Cinque | 1 | `heads_train-seed1/20260926-020802` | −0.460 [−0.661, −0.256] | 否 | 0.048 [−0.515, 0.426] |
| Cinque | 2 | `heads_train-seed2/20260926-021025` | −0.419 [−0.602, −0.227] | 否 | 0.132 [−0.376, 0.477] |
| Cinque | 3-seed 预测均值 | `heads_train-seed-mean/20260926-mean` | −0.262 [−0.429, −0.069] | 部分（对 ridge_late 的 CI 不跨零） | 0.459 [0.085, 0.800] |
| Lebowski | 0（第 40 条源 run） | `heads_train/20260925-110819` | −0.101 [−0.292, +0.094] | **是** | 0.696 [0.167, 1.418] |
| Lebowski | 1 | `heads_train-seed1/20260926-020802` | −0.352 [−0.555, −0.165] | 否 | −0.058 [−1.224, 0.469] |
| Lebowski | 2 | `heads_train-seed2/20260926-021025` | −0.230 [−0.419, −0.042] | 否 | 0.310 [−0.487, 0.860] |
| Lebowski | 3-seed 预测均值 | `heads_train-seed-mean/20260926-mean` | −0.106 [−0.286, +0.075] | **是** | 0.682 [0.164, 1.348] |
| Cinque | 0-rerun（`--seed 0`，噪声参照，不计入判定） | `heads_train-seed0/20260926-011909` | −0.289 [−0.476, −0.095] | 否 | 0.402 [−0.014, 0.750] |
| Lebowski | 0-rerun（`--seed 0`，噪声参照，不计入判定） | `heads_train-seed0/20260926-011909` | −0.131 [−0.320, +0.059] | 是 | 0.606 [0.032, 1.260] |

「3-seed 预测均值」是把三个 seed 的逐帧预测轨迹（`cls_late`、`cls ego` 两个 arm；`ridge_late`、原生、`ridge ego` 三个 arm 无随机性，
逐位相同）按元素平均，再走同一套 RFS / 配对 / bootstrap 代码；是一个 3-seed ensemble 的读数，不是「哪个 seed」的读数，单独列出供参考。

## 判定

- **Cinque**：3 个 seed 都判「没补上」（CI 不含 0，全部为负）。**不随 seed 变。**
- **Lebowski**：3 个 seed 中只有 seed 0（第 40 条原用的那次 run）判「够到原生」；seed 1、seed 2 都判「没补上」。
  **是否够到原生随词表 seed 变。** G 在 3 个 seed 上是 0.696 / −0.058 / 0.310，均值 **G = 0.316**，极差 0.754（大于原 CI 半宽 0.63）。
- 两模型合起来：G 在 2 模型 × 3 seed 共 6 个点上跨 **−0.06 到 0.70**，均值 Cinque 0.19、Lebowski 0.32。原来「0.4–0.7」只是 seed 0 一次的读数，
  不是词表 seed 下的稳定范围。

小表：[r40_seeds.csv](r40_seeds.csv)。
