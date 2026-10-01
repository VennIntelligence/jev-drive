# HUGSIM：HD-Score

代码版本：作者仓库 [`hyzhou404/HUGSIM` 提交 `62c690d39fd90020e68a196bd8bcc1c4d4191f2e`](https://github.com/hyzhou404/HUGSIM/tree/62c690d39fd90020e68a196bd8bcc1c4d4191f2e)，静态审读 `sim/utils/score_calculator.py`。此处 HD-Score 是该实现的计分函数，不把注释里的 PDMS 当成 NAVSIM 原版 PDMS。未运行仿真或 GPU 代码。

## 指标公式

- 对每个 `is_key_frame` 且规划轨迹至少 2 点的时刻，算 `frame_score = NC×DAC×(5×TTC+2×C)/7`。`NC` 为给定规划轨迹对背景点云和 **`obj_names == 'car'`** 的目标框无碰撞时 1、否则 0；`DAC` 用车体 2×2 网格内是否含路面点得 0/0.5/1；`TTC` 将每个规划点按当前轨迹速度外推 0.5 和 1 秒后再做碰撞查询；`C` 检查轨迹的加速度、jerk、yaw 等边界。[筛选与子项 `score_calculator.py:475–548`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L475-L548)、[碰撞 `:373–405`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L373-L405)、[DAC `:216–239`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L216-L239)、[TTC `:432–457`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L432-L457)、[舒适边界 `:20–28,299–326`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L20-L28)。
- `HD-Score = mean(frame_score over counted keyframes) × min(max(frame.rc),1)`。虽然 `score_weight` 字典有 `'ep':5` 且存在 `_calculate_progress` 函数，**实际循环中 EP 调用已注释，不进 `frame_score`**；进度仅作为最后的路线完成率乘子。[权重字典 `score_calculator.py:30–34`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L30-L34)、[实际循环 `:542–566`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L542-L566)。

## 攻击面与第一轮对照

| 攻击面 | 代码所证机制与边界 | 第一轮 |
|---|---|---|
| 规划失败时输出静止或制动轨迹保住短窗子分数 | 单帧公式没有 EP；静止/平滑轨迹可能保住 TTC 与 C，若此前已有正路线完成率，总分仍可为正。最终 RC 乘子会限制收益，不能说“原地不动可得满分”。[公式 `score_calculator.py:542–566`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L542-L566)、[TTC `:432–457`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L432-L457) | `HUG-WA-002`（第一轮观察到制动 fallback，未量化得分作用） |
| 非关键帧或少于两点的规划不进均值 | 分数循环直接跳过非 `is_key_frame` 和规划点数 `<2` 的帧；若提交接口容许控制规划点数，差帧有可能被排除出均值。实际接口是否允许此操作需另验；不能从此函数推断任何方法已这样做。[`score_calculator.py:475–488,550–558`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L475-L488) | 未观察到 |
| 对未列为 `car` 的动态目标缺少目标框碰撞处罚 | 循环只把 `obj_names == 'car'` 的目标框送入障碍检查；其他目标若未包含在静态场景点云中，预测轨迹与它们相交也不影响 NC/TTC。后半句依赖场景点云内容，是条件推论。[`score_calculator.py:503–509,524–536`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L503-L509)、[碰撞 `:373–405`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L373-L405) | 未观察到 |

## 盲区与证据界限

HD-Score 不直接把交通规则、行人/骑行者动态框、单帧路线进度或真实乘客评价纳入单帧式子；背景点云可覆盖某些静态结构，所以不能把“目标框只选 car”误说成完全不检测其他物体。[`score_calculator.py:503–566`](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/sim/utils/score_calculator.py#L503-L566)。第一轮 `HUG-WA-001` 是控制器协议依赖，属于运行接口而非此计分函数能独立证实的指标攻击面。
