# CARLA Leaderboard 2.0 · TF++

- 固定源码：[autonomousvision/carla_garage `f22bc491b3094792aef475149e09a94dcbb526f9`](https://github.com/autonomousvision/carla_garage/tree/f22bc491b3094792aef475149e09a94dcbb526f9)，`leaderboard_2` 分支；本地 `repos/carla_lb2__tfpp/`。
- 论文：[TF++ Leaderboard 2.0 技术报告](../../papers/tfpp.pdf)。SimLingo 论文 Table 1 将 TF++ 官方 5.56 DS 列在 **MAP** 区、5.18 DS 列在 **SENSORS** 区；两者不能混排。公开仓库 `leaderboard/run_leaderboard.sh` 默认 `SENSORS`，agent 根据 `CHALLENGE_TRACK_CODENAME` 可切换为 MAP；5.56 对应的确切提交配置未见公开脚本。

## 已读与未读

已读 `README.md` 的模型、评测、Bench2Drive 与长路线说明，`team_code/sensor_agent.py` 的 setup、传感器、推理控制和后处理，`team_code/config.py` 的蠕行参数，`leaderboard/run_leaderboard.sh`、`leaderboard/test_run.sh`，以及技术报告 §2–6、表 2–5。用 SimLingo 论文 Table 1 交叉核对 MAP/SENSORS 分区。

未逐行读训练网络、完整专家数据采集及所有 scenario_runner 文件；未下载权重，未运行 agent 或 CARLA。官方 MAP 提交的确切环境变量、模型目录和 seed 数不能由公开默认脚本完全复原。

## 整体印象

论文主动指出 CARLA LB2 长路线 DS 的数学缺陷，并明确使用提前终止，同时提出 normalized DS。这是披露充分、影响显著的榜单计分机制发现。公开 agent 还包含默认停车牌控制器和卡住蠕行；两者是实际推理后处理，但论文未单独披露或消融。论文 §4 称最终模型结合 Big、Pre、Ens，README 也允许多模型集成；集成可在部署系统使用，故仅列为配置事实，未直接判为 hack。

## 发现

- `LB2-TFPP-001` `metric_early_termination`：[agent L673–676](https://github.com/autonomousvision/carla_garage/blob/f22bc491b3094792aef475149e09a94dcbb526f9/team_code/sensor_agent.py#L673-L676) 在行驶超过 `STOP_AFTER_METER` 后主动停车。论文 §5.2 “Early termination” 称实际使用 1.5 km，并解释何以减少 RC 却提高 DS；表 5 Town13 相同模型约 0.96→5.10 DS，但不是官方 LB2 对照。代码默认 `STOP_AFTER_METER=-1`，官方参数未随公开默认脚本给出。部署不成立，披露 yes，置信度高。
- `LB2-TFPP-002` `manual_control_override`：[agent L668–671](https://github.com/autonomousvision/carla_garage/blob/f22bc491b3094792aef475149e09a94dcbb526f9/team_code/sensor_agent.py#L668-L671) 在预测控制后强制停车牌刹车；setup L126 默认为启用，L697–737 是检测框缓存和几何相交逻辑。论文 §2–6 未说明该后处理；影响 `none`。部署部分成立，置信度中。
- `LB2-TFPP-003` `manual_control_override`：[agent L657–661](https://github.com/autonomousvision/carla_garage/blob/f22bc491b3094792aef475149e09a94dcbb526f9/team_code/sensor_agent.py#L657-L661) 在静止帧阈值后、LiDAR 安全框为空时强制油门并取消刹车。论文 §2–6 和 README 未说明；影响 `none`。部署部分成立，置信度中。

论文原文定位：技术报告 §5.2 “Early termination” 中写 “we set target speed to 0 after d = 1.5km in practice”；表 5 另给 normalized DS 对照。SimLingo 论文 Table 1 的 MAP/SENSORS 分区用于检查 5.56 数字所属轨道。

## 局限与候选实测

提前终止的量化例子来自 Town13 validation，并非官方 LB2 test 的同配置配对消融。候选实测：固定模型和路线，分别设置 `STOP_AFTER_METER=-1` 与论文 1500 m，报告 RC、IS、DS 与 normalized DS；再单独关闭停车牌、蠕行规则。需要先确认 MAP/SENSORS 轨道及所用 checkpoint；此处未运行。
