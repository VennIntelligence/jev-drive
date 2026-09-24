# 待决问题与已知限制

- CARLA 官方网页当前表格为空，抽样依赖 SOTA2 对官方榜单的转录；SENSORS 和 MAP 成绩不可直接比较。
- NAVSIM 官方 Hugging Face 提交数据返回 401，排名用公开论文对比表及 SOTA2；navtest 与 navhard 分开处理。
- nuScenes 开环规划论文存在不同 L2、碰撞协议，横向名次只用来抽样，不把数值差当能力差。
- TOAD 的 NAVSIM-v1 单元已单独固定有完整路径的 `nav1` 分支提交；README 当前称 v1 94.9，论文 Table 1 为 94.7。抽样按论文 94.7，两个设置的成绩不互相代替。
- 独立复核补出 TOAD nav1 README 训练命令的 `long_trajectory_additional_poses=2`，但该命令引用的 `drivoR_speed_up` Hydra 配置不在固定仓库；公开文档关联 checkpoint 与长目标，不能逐步重建该权重训练史，故发现保留中置信度。
- DriveVLA-M0 论文 navtest 94.1 属 Scale（10K memory），固定提交的公开评测入口只部署 Base，Retrieve 为独立验证入口，未找到完整 memory+TTT 推理链。按完整论文方法衡量公开程度可能从 A 降为 B，现有发现仅能审公开 Base 机制。
- GTRS 抽样 45.4 是修复官方指标后 navhard 双阶段合并 EPDMS（DrivoR 后续论文 Table 3），固定 GTRS 2025 论文及 README 仍列旧协议 V2-99 GTRS-Aug 42.1；没有本固定提交对 45.4 的直接重放配置/日志。其旧协议 ablation 不用于分解 45.4。
- SparseDriveV2 Bench2Drive 分支 `e42ea59...` 的 SENSORS agent 从 `CarlaDataProvider.get_world()` 查询 LiDAR actor 真值 pose。公开代码可证，但需官方提交包/规则核实该接口是否在正式评测可用；纯静态材料没有去除真值 pose 的 DS 消融。
- CarLLaVA/SimLingo-BASE 论文以 2100 m 提前终止取得官方 6.87 DS（SENSORS），但固定公开 main 未找到对应 Base 闭环 agent/早停实现；两项已降 C、排除。若能获得 2024 官方提交包，应重新固定 commit 并审计。
- TF++ 论文与 SimLingo Table 1 的 5.56 DS 属 MAP，公开 `run_leaderboard.sh` 默认 SENSORS；公开代码允许环境变量切 MAP，但未提供官方 5.56 的确切脚本、`STOP_AFTER_METER` 值和 checkpoint 配置。
- HUGSIM WA-JEPA 0.4462 缺逐场景推理失败/制动回退日志，无法判断 aggregate_results.py 所述潜在路径是否进入论文主分数。
- HUGSIM UniAD_SIM 固定客户端配置六相机，而 WA-JEPA Appendix A 称同表除 LTF 外用四相机；重测的精确客户端配置未公开于此 fork。
- HUGSIM LTF 客户端将导航命令固定为索引 1，未使用 info[command]；无配对分数，不能判定增减。
- WOD RAP 的通用缓存构建函数会在 `future_states` 非空时用未来位置计算历史朝向；Waymo 官方测试协议扣留未来驾驶记录，矩阵按官方测试记 `future_label_conditioning=no`。本次未下载真实 test TFRecord/预制缓存，不能排除非标准缓存另有行为；最终独立复核判 `NA`，分歧见 `out/review.md`。
