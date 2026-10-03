# navhard DAC 补充：门控跟踪器补偿 + M 类目测（预登记，写在任何门控臂的分数之前）

接第 112 条与第 97 条。此前已读过的数字：第 97 条的无门控臂（full / path 的 alpha 1、path alpha 0.25）在三个榜上的分数，以及第 112 条的类别计数。门控臂的分数一个都没读。

## 1. 门是否能不看地图
- 类 L（计划在区内、LQR 回放越界）的定义用了评分多边形，所以「按 L 门控」是评分器侧 oracle，不是可部署的 trick：只报上限（navhard，原生与 best 各自的 L 集合）。
- 不看地图的代理门（只用计划 + ego_status，经 devkit 自己的 simulator，与补偿本身的输入相同；不用参考、地图、stage 标签、评分器输出）：
  - gE：预测跟踪误差 = 未补偿计划的 LQR 回放到计划折线的平均距离（`path_d_mean`，已在 trk diag 里）。
  - gK：计划终点朝向 |h(4 s)|（曲率代理）。
- 没有试 openpilot 自己的路沿：预测文件里没存，补存要重跑模型。留作未试。
- 诊断（不用来选门）：navhard 上两个门对 L 类 token 的 AUC，以及 stage 1 / stage 2 被门住的比例。

## 2. 网格与选择（只在 navtrain 3 000 token 上）
- 模式 {full, path} x alpha {0.25, 0.5, 0.75, 1} x 门 {gE, gK} x 被补偿的 token 比例 {5, 10, 20, 35, 50, 100}%（100% = 第 97 条无门控）。阈值 = navtrain 上该门分数的分位数，测试榜沿用同一绝对阈值。
- 门控臂的 token 分数 = 门内取该 alpha 补偿臂的官方分，门外取未补偿的官方分（补偿是逐 token 独立的，所以与直接提交混合位姿等价）。
- 选择：navtrain PDMS 差最大者（并列取较小 alpha，再取较小比例）；差 <= 0 则「无」，报告为否。只选一个配置，原生与 best（it_dw3 + 选择器）共用，best 不重新拟合（best 没有未见过的 navtrain 预测）。

## 3. 测试榜读数
- navhard 两阶段 EPDMS（in-process devkit 聚合，对官方 CSV 核对）与 navtest PDMS（官方 v1 CSV），原生与 best；对未补偿臂配对差，navhard 按 225 个映射组 bootstrap（5 000 次），navtest 按 token（5 000 次）。
- 判据同第 97 条：navhard 差 > 0 且 CI 下界 > 0，且 navtest 差 >= -0.30，才算过；差 > 0 但 CI 含 0 = 不确定；其余 = 否。同一 alpha 的无门控臂并列报，看门有没有帮上忙。
- 不做多重校正；测试榜臂数：选中配置 x 2 模型，加 oracle L（原生 / best，full alpha 1、path alpha 1、选中 alpha）。

## 4. M 类目测
- 原生臂的主类 M 共 345 个 token，种子 0 无放回抽 40 个，做审阅表：CAM_F0 t0 带评分多边形、BEV（评分多边形、nuPlan generic drivable area、计划、首次越界框）。
- 按眼睛分三类：真实铺装（车能开）/ 不可行驶（草地、路沿外、人行道、建筑、停车位占用等）/ 不清楚。报占比与 Wilson 95% CI；「真实铺装」占（真实 + 不可行驶）的比例另报。先验：第 104 条 40 例约 38% 真实铺装在多边形外。
