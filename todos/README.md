# todos/

实验规划与执行记录。简单计划使用 `YYYY-MM-DD-<slug>.md`；需要协作、日志与图表的实验使用同名文件夹。
新建时复制 `TEMPLATE.md`。

用中文写。临时、随手的东西放 `tmp/`（已 gitignore），不要放这里。

- [2026-09-22 B2D controller](2026-09-22-b2d-controller.md)：已完成轨迹控制器实施、反馈迭代与Dev10验收；无合格替代默认，全部结果见目录。

- [2026-09-23 真实TCP与转弯控制](2026-09-23-tcp-controller/README.md)：真实模型纵向对照、静止速度反馈修复，以及按转弯窗口单独优化横向。

- [2026-09-23-lateral-followup/](2026-09-23-lateral-followup/README.md)：逐弯控制诊断、Hermite负面闭环、固定后轴传播补偿及全部中间图表。

- [2026-09-23 横向 v2](2026-09-23-controller-next/lateral-v2-report.md)：后轴速度系 pursuit + Ackermann 反解的预注册闭环（[协议](2026-09-23-controller-next/lateral-v2-protocol.md)）；CTE 主目标与 held-out 全部改善，动作保护量失败，候选不通过。

- [2026-09-23 W2：TFv6 + 我们的控制器](2026-09-23-tfv6-controller/report.md)：TFv6 waypoint 交给 production 控制器与作者 waypoint PID 的四臂预注册闭环（[协议](2026-09-23-tfv6-controller/protocol.md)）；route+speed 表示比 waypoint 高约 14 DS；W2 的 C/D 被静止时的坐标变换缺陷污染；修正后重跑（[W2b](2026-09-23-tfv6-controller/report-w2b.md)），C−B 合并 +1.0 [−12, +15]，仍未检出。
- [2026-09-23 P4 CARLA 特征差距](2026-09-23-p4-carla-feature-gap.md)：Waymo 训的 head 在 CARLA 帧上能不能用（domain AUC、词表覆盖、head 迁移），预登记判据决定 P5 的前提。
