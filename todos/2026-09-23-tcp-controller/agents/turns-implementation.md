# 隔离转弯开发：一个前视时间参数

2026-09-23。仅在 `/data/worktrees/jev-drive-controller-turns` 实现；真实TCP主工作区运行时未修改。当前交付实现与合成G1，尚未运行本轮CARLA G2。

## 已实现、已验证

`Controller` 新增末尾可选参数 `max_lookahead_time_s=.5`，只用于 `lookahead='max'` 的 `max(3., max_lookahead_time_s*speed)`。参数必须可转换成有限正数；3m下限、additive/fixed4公式、实测车辆几何、转向限幅、PI与adapter均保持原值。候选仅将时间系数改成 `.375`：8m/s前视4→3m；实际速度≤6m/s两组均为3m。S控制实际速度会稍超6m/s，不能声称整次闭环或所有S帧精确相同。

- `test_b2d_controller_turns.py` 四项通过：冻结旧源码的432tick黄金输出，默认省略与显式.5均逐值相等；参数拒绝；8m/s几何改变/低速floor与其他规则不变；8m/s左右镜像和0/.3s延迟圆弧。
- 黄金数据 `scripts/testdata/b2d_controller_turn_default_golden.json` 含旧源码SHA与确定性生成协议；覆盖3preset×2纵向模式×3前视规则×24tick。无需读取绝对旧源码路径即可运行测试。
- `PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python -m unittest discover -s scripts -p 'test_b2d_controller*.py'`：**112项通过**。测试日志中的worker错误是已有mock失败路径，不是实际CARLA启动。
- `b2d_controller_turn_selftest.py` 候选G1 **18/18通过**：12圆弧（6/8m/s、左右、0/.3s延迟、8m/s±偏置/航向）、2S、1停车、3速度阶跃/执行延迟。
- 最大圆弧稳态CTE RMS .02618m，最大S RMS .009755m；停车误差−.02833m，减速速度RMS .11222m/s，5s保持零漂移；速度阶跃最大稳态RMS .007258m/s。

[G1摘要](turns-candidate-g1-summary.json) 与 [SHA索引](turns-candidate-g1-hashes.json) 保留在notes；完整逐帧轨迹和归档源码位于 `/data/runs/b2d/controller/turns-candidate-g1-v1`。这是分析路径与合成自行车/踏板模型，不是CARLA轮胎/转向动力学测试；没有神经模型、交通交互或新默认资格。直接复现到新目录：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python scripts/b2d_controller_turn_selftest.py --out /data/runs/b2d/controller/turns-candidate-g1-replay
```

## 符号复核：假设是外侧残差，不是已证明切内角

26966历史G2-v4 max的左正CTE，在core后半段37.5–46m均值+.8075m、exit46–51m均值+.6467m、之后51–61m均值+.3091m；峰值+.8565m对应frame8361。原校验器使用 `tangent_y*residual_x−tangent_x*residual_y`，在CARLA坐标中明确**左正**。这条路是右弯，所以正残差在弯道外侧，不能叫“corner cutting”。先前协作消息曾误读正号，已在本轮G2前立即向根代理和分析代理更正。

因此保留.375候选的合理假设是：较短前视增强近处路径纠偏，降低右急弯外侧跟踪与出弯残差。是否确有时延、转向模型偏差、adapter rejoin影响，需要控制/真值时序进一步区分；不能从上述数字宣称已识别唯一根因。该历史窗口无steer限幅；改变steer_rate不是当前目标。S的方向盘变化率已到2/s，本轮保持该限值，专门检查代价。

## 建议在首次G2之前冻结的六例登记

执行集固定为 **24240@8m/s、26966@8m/s、17563@6m/s × baseline/candidate**，均scenario清空、同原XML路线/天气；新建两份配置，唯一差异是 `max_lookahead_time_s:.5→.375`。均pursuit/max、PI Kp=.5/Ki=.25、near窗、steer_rate=2、max_steer=.8、同实测车辆及v2adapter。

使用既有 `b2d_controller_validate.py --variants <JSON> --route-cruises <JSON>`，每条路线相邻运行两配置，同server跨case复用、每例重建actor/agent；输出原G2全部gates和完整raw，不改指标实现。配置映射每项 `{"controller_config":"/absolute/path.json","presets":["pursuit"]}`；cruise映射 `{"default":8,"17563":6}`。源码/路线/配置SHA在启动前由根代理固定，本文未启动或授权额外CARLA进程。

固定窗口与后处理使用分析代理 `lateral-evidence-v2` 方法，并保存其版本。26966 core29–46m/pad24–51m，24240core23–59/pad18–64，17563两窗core32.5–43.5及78.5–90/pad27.5–48.5及73.5–95。另报core之后10m恢复；24240该core仅约49°，整路约90°，不能用核心代替整弯覆盖。各case参考几何必须一致后才能共享station边界。

建议验收规则，需由根代理在首例前纳入正式protocol：

1. 两组均通过原G2所有gates，完整到达全部窗口及终点；所有碰撞、blocked、低速、异常和未覆盖窗口保留。低速不能被moving过滤成成功。
2. 主目标26966固定窗口CTE RMS降低至少15%，P95不升；出弯后10m RMS不升。不能仅以全程均值下降作决定。
3. 每个左弯/S窗口P95增加≤.05m、最大CTE增加≤.10m；左右弯entry/core/exit分别展示。S两个窗口分别列，避免平均掩盖单边回摆。路径航向跟踪误差与定位航向误差分开。
4. 同窗口速度误差RMS增加≤.10m/s、平均速度降低≤.20m/s；报告方向盘变化率/限幅比例和横向加速度，不通过降低车速或放宽转向限制取巧。建议舒适性约束为横向加速度P95增加≤10%、方向盘变化率P95增加≤20%，不足样本标证据不足，不当通过。这是开发取舍而非通用安全标准。
5. 如有完整窗口任一未覆盖或必要gates失败，不调整阈值、不扩增益网格；回到机制分析。若全部满足，只列开发候选；同三条B/A反序确认另6例后再讨论交互/泛化，不直接升级默认。

以上是下一轮对照建议，不覆写此前正式G4失败结论；根代理若基于签名证据决定不继续，也应保留已完成的负/正证据。

## 真实TCP迁移仍未验证

本轮oracle输入是20×2、.25s未来点，真实TCP是4×2、.5s/2s，预测物理原点尚未确认。当前真实TCP六例纵向对照保持原生横向；本文不把它们作为pursuit横向收益。后续必须显式支持原生短时域、证实原点、定义native target arbitration；不补5s尾点，不用dense真值路线或oracle rejoin改写模型预测。
