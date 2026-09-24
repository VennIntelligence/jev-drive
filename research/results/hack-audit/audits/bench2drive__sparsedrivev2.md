# Bench2Drive · SparseDriveV2

- 原备料 main 固定版本 `696ef77924eb9e0a4b4047d013a50e9854bfa026` 仅含 NAVSIM；其 README L14 明示 Bench2Drive 代码另在 `bench2drive` 分支。
- 本单元实际审计：[swc-17/SparseDriveV2 `e42ea59dd4946238dc65097495a9aa0708121fae`](https://github.com/swc-17/SparseDriveV2/tree/e42ea59dd4946238dc65097495a9aa0708121fae)，本地 `repos/bench2drive__sparsedrivev2_b2d/`。阶段 0 清单应记录此 commit。
- 论文：[SparseDriveV2](../../papers/sparsedrivev2.pdf)；表 4 报 Bench2Drive DS 89.15、SR 70.00。

## 已读与未读

已读 main README、Bench2Drive 分支 `leaderboard/team_code/sparsedrive_b2d_agent.py`、`leaderboard/team_code/pid_controller.py`、`projects/configs/sparsedrive_stage2.py` 的推理管线、`projects/mmdet3d_plugin/datasets/pipelines/transform.py` 对全局变换的处理，以及论文 §3–4、表 4–6、Appendix A。

未逐行读大型检测网络、完整训练与数据生成代码、全部仿真器附带文件；未运行 agent 或评测。不能由静态代码确定官方提交与公开分支逐字一致，也没有精确位姿替换为 GPS 的消融。

## 整体印象

规划网络、路径与速度分解的研究内容有完整论文说明。Bench2Drive 的公开 agent 声明 SENSORS，却不使用已读取的 GPS 为主定位：它直接查询 CARLA world 中 LiDAR actor 的真值 transform，以此构造世界/车辆变换、当前坐标和路线目标。该信息在真实部署和严格传感器隔离下不可获得。分支中还有 50 m 高俯视 RGB `bev` 传感器，但 `test_pipeline` 的模型 `Collect` 键不含 `bev_img`，因此没有把俯视图误记为模型输入。

## 发现

- `B2D-SPARSE-001` `simulator_pose_access`：[agent L360–370](https://github.com/swc-17/SparseDriveV2/blob/e42ea59dd4946238dc65097495a9aa0708121fae/leaderboard/team_code/sparsedrive_b2d_agent.py#L360-L370) 从仿真器 world 取 LiDAR 真值位姿，并让路线规划器使用据此得到的 `pos`；`run_step` L430–476 将该变换送入模型元数据与目标点转换。论文 Appendix A 只说明六相机、词汇和控制器，未披露此直接查询；影响 `none`，部署不成立，置信度高。审计对象声称 SENSORS，见同文件 L66–69。

论文原文定位：§3 的 “sensor observations”；Appendix A 的 “Images from all six cameras are used as input”。两处均未交代 CARLA actor transform 作为在线定位来源。

## 局限与候选实测

未证实官方评测是否允许 agent 直接访问 `CarlaDataProvider`；代码无条件调用，所以若官方严格隔离，应进一步检查提交是否与公开代码一致。候选实测：固定模型，把 actor 真值定位替换为 GPS/IMU 估计，在相同路线比较 DS、路线偏差和控制稳定性；此处未运行。
