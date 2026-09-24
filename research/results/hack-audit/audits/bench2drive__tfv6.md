# Bench2Drive · TFv6（LEAD）

- 固定源码：[kesai-labs/lead `730bc1a2f44d5f28312dd55f0ca958e94a24c038`](https://github.com/kesai-labs/lead/tree/730bc1a2f44d5f28312dd55f0ca958e94a24c038)，`cvpr2026` 分支；本地 `repos/bench2drive__tfv6/`。
- 论文：[TFv6](../../papers/tfv6.pdf)；榜单为 Bench2Drive 0.0.3，README §1.3 报 95.28 DS。

## 已读与未读

已读 `README.md` §1.3 与闭环评测说明、`lead/inference/config_closed_loop.py`、`lead/inference/closed_loop_inference.py`、`lead/inference/sensor_agent.py`、`lead/common/pid_controller.py` 的控制相关段落，以及论文 §3–4、表 1–5。以代码搜索核对了评测脚本中 `LEAD_CLOSED_LOOP_CONFIG` 和推理后处理开关。

未逐行读训练网络各层、专家采集实现、全部 SLURM 脚本及数据文件；未运行 agent、CARLA、模型或评测。不能从静态材料单独确定每个后处理在 220 路线的触发频率。

## 整体印象

论文的核心研究是专家与学生状态、意图不对称；其数据与结构消融提供了较好的可迁移性证据。闭环 95.28 DS 还依赖选定的控制输出接口和 README 复现命令中的启发式。配置类默认关闭 Kalman、creeping 和停车牌规则，但 README 要求开启它们复现表中分数；不能把默认参数当作报告配置。论文称每个模型评测一次、结果取 3 个独立训练 seed 均值。

## 发现

- `B2D-TFV6-001` `control_interface_selection`：默认以 route 转向、target speed 控油门和刹车；同一推理代码还计算 waypoint 控制。证据见 [配置 L29–35](https://github.com/kesai-labs/lead/blob/730bc1a2f44d5f28312dd55f0ca958e94a24c038/lead/inference/config_closed_loop.py#L29-L35) 和 `closed_loop_inference.py` L215–252。论文 §3.1 披露 target speed 头，但未给同 checkpoint 两控制接口的 Bench2Drive 消融；影响数值 `none`。部署部分成立，控制器可迁移但参数和相对优势需验证；置信度中。
- `B2D-TFV6-002` `manual_control_override`：长时间静止后、LiDAR 安全框为空时强制至少 0.4 油门并清刹车。证据见 [后处理 L854–861](https://github.com/kesai-labs/lead/blob/730bc1a2f44d5f28312dd55f0ca958e94a24c038/lead/inference/sensor_agent.py#L854-L861)；README §1.3 明确要求开启。部署部分成立；README 给三项启发式联合开关约 95→94 DS，未单列 creeping 数值；置信度中。
- `B2D-TFV6-003` `manual_control_override`：停车牌检测框触发网络控制后的强制停车。证据见 [后处理 L753–756](https://github.com/kesai-labs/lead/blob/730bc1a2f44d5f28312dd55f0ca958e94a24c038/lead/inference/sensor_agent.py#L753-L756)；README §1.3 披露。部署部分成立；只有同上联合消融，单项影响 `none`；置信度中。

论文/README 原文定位：论文 §3.1 “an additional learned query predicts target speed”；README §1.3 “To reproduce these results, enable the Kalman filter, stop-sign, and creeping heuristics”，其紧随表格的 Bench2Drive 数字为 95→94（联合关闭后）。

## 局限与候选实测

控制接口与各启发式单项对 DS 的增益无法由论文隔离。候选实测：固定同一 checkpoint、同一 220 条路线与 seed，分别比较 `route+target_speed` 对 `waypoint`，再逐项关闭 creeping、停车牌规则、Kalman；同时记录 blocked、stop infraction 与行人/cut-in 反应。此处未运行。
