# 阶段 2 矩阵重编码说明

`matrix.csv` 是 20 个入选审计单元 × 19 个最终类别的完整长表，`board` 加 `repo` 唯一确定审计单元。`yes` 只表示固定代码有该机制且对应至少一条最终发现；`no` 表示机制适用、相关入口已读、未见；`NA` 表示协议不适用或代码不足以判定。33 个 yes 来自 35 条发现，TFv6 和 TF++ 各有两条手工车控覆盖，故合为单个单元格。

## NA 判定

- 闭环控制类（`control_interface_selection`、`manual_control_override`、`simulator_pose_access`、`metric_early_termination`、`simulator_protocol_dependency`、`inference_failure_brake_fallback`）对 NAVSIM、nuScenes、WOD 开环榜单记 NA。
- nuScenes 三秒 L2 运动捷径类（`ego_only_open_loop_planning`、`ego_state_fusion`）仅在 nuScenes 记 yes/no，其他榜单记 NA；NAVSIM/WOD 常规使用 ego 状态不自动等于 nuScenes 的短时 L2 捷径。`future_label_conditioning` 对闭环 Bench2Drive、CARLA-LB2、HUGSIM 记 NA。
- `benchmark_prompt_specification` 只对已读到文本提示/语言规划路径的 DriveVLA-M0、SparseOccVLA、DriveMA、AutoVLA、BLUE 判 yes/no，其余方法记 NA。
- B 档 BLUE、AutoVLA、UniAD、LTF 无该榜完整训练链，训练目标/奖励/划分适配类缺证据时记 NA。AutoVLA 缺 Waymo 专用提交路径，测试时择轨、权重及未来标签类缺证据时也记 NA。
- SparseOccVLA 测试代码读取 `gt_planning_command`，但固定仓库缺该字段的生成链；`future_label_conditioning=NA`，不因字段名直接判 yes，也不能确认 no。
- TOAD v1/v2 的初始 DrivoR 候选 argmax 和 CEM 失败回退属于同一测试时搜索路径，按最终 codebook 只记 `metric_proxy_test_search=yes`，`metric_proxy_candidate_selection=no`；手设评分权重、基座训练目标等独立机制仍可另行编码。
- WOD RAP 通用缓存构建函数可读取 `future_states` 算输入朝向，但官方 Waymo E2E 数据说明测试集未来驾驶记录被扣留，提交脚本读取 `test` 缓存，故正式协议下 `future_label_conditioning=no`。最终独立复核因缺实际 test TFRecord/缓存判 `NA`；保留此分歧与非标准缓存风险。

其他适用的缺席机制记 `no`。本表由 [build_outputs.py](sources/build_outputs.py) 从固定 `repos.csv`、阶段 1 发现和最终 codebook 静态生成；没有运行仓库代码或模型。类别合并、保留和排除的裁决见 [recode_decisions.md](sources/recode_decisions.md)。
