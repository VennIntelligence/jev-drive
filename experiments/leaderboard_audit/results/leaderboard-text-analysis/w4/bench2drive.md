# Bench2Drive 0.0.3：DS 与 SR

代码版本：基准作者 [Thinklab-SJTU/Bench2Drive，`0.0.3` 分支，`2645714eb1f3a100217928dd113093cae0779f36`](https://github.com/Thinklab-SJTU/Bench2Drive/tree/2645714eb1f3a100217928dd113093cae0779f36)。`autonomousvision/Bench2Drive-Leaderboard` 的 `c335475fcdfe1ec5f12c41e4625f0d5a308e1a6a` 是排名页面，不含这里审读的 evaluator。以下公式对应 0.0.3；附带 `scenario_runner` 是同一仓库的代码版本。只静态审读。

## 指标公式

- 单路 `RC=score_route`，来自 `ROUTE_COMPLETION` 事件的 `route_completed` 百分数；`IP=∏` 固定违规因子与按百分比计算的路外因子；`DS_route=max(RC×IP,0)`。固定因子：行人碰撞 0.5、车辆 0.6、静态物 0.65、红灯 0.7、停车牌 0.8、场景超时 0.7、未礼让应急车 0.7；路外比例因子为 `1−p/100`。**0.0.3 的最低速度事件标成 `unused`，只记录而不扣 DS**。[因子表 `statistics_manager.py:21–38`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L21-L38)、[事件乘法和 DS `:352–413`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L352-L413)。
- 汇总 DS 为 220 条路线 `score_composed` 的算术平均；SR 计数 `status` 为 `Completed/Perfect`、且除 `min_speed_infractions` 外无违规的路线，再除以 220。每路的 `Completed/Perfect` 由到达终点及违规总数决定。[`merge_route_json.py:9–40`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/tools/merge_route_json.py#L9-L40)、[`statistics_manager.py:415–423`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L415-L423)。
- 路程完成率按路线 waypoint 的累计距离更新，满足近终点阈值才设为 100。碰撞判据由 CARLA collision sensor 事件触发，静止 ego 的碰撞在本版本被过滤；车辆卡住超过 60 秒可提前结束路线。[`atomic_criteria.py:1550–1597`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py#L1550-L1597)、[碰撞过滤 `:374–405`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py#L374-L405)、[卡住阈值 `route_scenario.py:429–434`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/scenarios/route_scenario.py#L429-L434)。

## 攻击面与第一轮对照

| 攻击面 | 代码所证机制与边界 | 第一轮 |
|---|---|---|
| 加不依赖网络决策的停车牌规则层 | 停车牌违规每次乘 0.8；只改评测 agent 的后处理即可改变该扣分项，不等于改进网络规划。规则对真实驾驶是否有益、实际 DS 增量须另证。[`statistics_manager.py:21–27`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L21-L27) | `B2D-TFV6-003` |
| 用定时蠕行覆盖模型刹车，避开卡住终止 | `VEHICLE_BLOCKED` 作为失败状态保留当前 RC；60 秒低速阈值会终止路线，短时强制前进可延后此状态。但是否增加 DS、是否更安全均未经此代码证明。[`route_scenario.py:429–434`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/scenarios/route_scenario.py#L429-L434)、[`statistics_manager.py:397–413`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L397-L413) | `B2D-BLUE-001`; `B2D-TFV6-002` |
| 以低速换安全项而不付最低速度直接扣分 | `MIN_SPEED_INFRACTION` 在 0.0.3 为 `unused`，SR 汇总也排除此项；若速度仍满足卡住和超时限制，则相同路线完成率下，放慢驾驶可免掉最低速度违规而保持 DS/SR。是否实测提高分数未观察。[`statistics_manager.py:31–38`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L31-L38)、[`merge_route_json.py:20–27`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/tools/merge_route_json.py#L20-L27) | 未观察到 |

## 盲区与证据界限

DS 是完成率与列出的离散事件罚分，未把舒适度、接管次数或耗时直接放进公式；耗时只通过路线超时影响终止。静止车辆受撞的 collision sensor 事件在本版本被忽略；这不是普遍的无碰撞证明。[`statistics_manager.py:21–38,410–413`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py#L21-L38)、[碰撞过滤 `atomic_criteria.py:382–384`](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py#L382-L384)。本页只把代码支持的可优化机制列为攻击面；不从机制推出第一轮方法的实际分数收益。
