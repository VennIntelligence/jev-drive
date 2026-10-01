# CARLA Leaderboard 2.0：DS

代码版本：官方 `carla-simulator/leaderboard` 的 [`leaderboard-2.0` 分支，`a87a3419e9d2e0d36deb25f1f26c17edee2d1420`](https://github.com/carla-simulator/leaderboard/tree/a87a3419e9d2e0d36deb25f1f26c17edee2d1420)；与其配合审读的官方 `scenario_runner` 固定在 [`94ff3b8af752bad2b9d464ad5105868906aa34c0`](https://github.com/carla-simulator/scenario_runner/tree/94ff3b8af752bad2b9d464ad5105868906aa34c0)。未运行 CARLA，也未假定两个仓库由提交锁定为同一发布包。

## 指标公式

- 单路线 `RC` 取 `ROUTE_COMPLETION` 百分数，`IP` 从 1 开始，每次指定违规乘固定因子：行人碰撞 0.5、车辆 0.6、静态物 0.65、红灯 0.7、停车牌 0.8、场景超时及未礼让应急车 0.7；路外比例按 `1−p/100`，最低速度按 `1−0.3(1−p/100)`。`DS_route=max(RC×IP,0)`。[`statistics_manager.py:21–37`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L21-L37)、[`statistics_manager.py:336–395`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L336-L395)。
- 榜单总 DS 是各路 `score_composed` 的等权算术平均；完成状态和 DS 是不同字段，未到终点路线仍可有正 DS。[`statistics_manager.py:392–405`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L392-L405)、[全局平均 `:430–437`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L430-L437)。
- 评测树持续监测碰撞、红灯、停车牌和最低速度；偏航与低于 0.1 m/s 达 180 秒会提前结束路线。[`route_scenario.py:409–436`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/scenarios/route_scenario.py#L409-L436)。

## 攻击面与第一轮对照

| 攻击面 | 代码所证机制与边界 | 第一轮 |
|---|---|---|
| 行驶一定距离后主动停车，截断后续风险 | 代码把已完成的 RC 乘已发生的 IP；未到终点仍保留正 DS。因此若预期后半程罚分的乘积损失超过 RC 增量，停车可提高 DS；这是**条件推论**，不意味着普遍有效。超过 180 秒低速将以 blocked 结束，但当前分仍按 RC×IP 记录。[`statistics_manager.py:379–405`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L379-L405)、[`route_scenario.py:429–434`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/scenarios/route_scenario.py#L429-L434) | `LB2-TFPP-001` |
| 把停车牌后处理接在模型控制之后 | 每次停车牌违规乘 0.8，评测 agent 的独立规则可消掉此类扣分而网络不变；是否实际加分需日志或消融。[`statistics_manager.py:21–29`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L21-L29) | `LB2-TFPP-002` |
| 用强制蠕行避免 blocked 状态 | 180 秒卡住会终止场景并停止 RC 增长；额外油门规则有机会绕过该终止条件，不能由此推断安全或净分收益。[`route_scenario.py:429–434`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/scenarios/route_scenario.py#L429-L434)、[`statistics_manager.py:379–395`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L379-L395) | `LB2-TFPP-003` |

## 盲区与证据界限

公式未单独评分乘坐舒适度、控制平顺性和接管需求；超时及最小速度通过特定事件处理。路线 DS 可与失败状态并存，因此仅 DS 不能表达是否完成任务。[`statistics_manager.py:21–37,392–405`](https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/statistics_manager.py#L21-L37)。第一轮 TF++ 论文给出提前停车消融，但本页只据官方计分代码确认机制；实际官方提交配置仍需独立确认。
