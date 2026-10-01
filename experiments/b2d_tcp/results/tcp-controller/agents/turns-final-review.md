# 横向六例最终独立审查

2026-09-23，只读审查；未重跑CARLA、模型或已有测试，未修改分析代理结果。结论：**候选未通过的判定成立；六例原G2均通过不覆盖四项转弯必要条件失败。**

审查来源：`/data/runs/b2d/controller/turns-v1`、`/data/runs/b2d/controller/turns-v1-analysis-v1`、冻结的 `turns/protocol.md`、主工作区 `docs/b2d-controller.md`。126项检查中122pass、4fail，无insufficient；逐项阈值与冻结协议一致。

| 失败项 | baseline | candidate | 预登记限值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 26966窗口CTE RMS m | .5586937053 | .5164470635 | ≤.4748896495（下降≥15%） | 仅下降7.5617% |
| 26966方向盘变化率P95 /s | .8001055870 | 1.0797112163 | ≤.9601267043（+20%） | +34.9461% |
| 17563第一个S横向加速度P95 m/s² | 5.8278060058 | 6.6136953909 | ≤6.4105866064（+10%） | +13.4852% |
| 24240方向盘变化率P95 /s | .2183799831 | .3779408281 | ≤.2620559797（+20%） | +73.0657% |

我直接读取各例原始 `control.jsonl`、`validation_trace.json`、`route_reference.json`，另写只读数值计算，复算八个case-window的nearest-segment左正CTE RMS、实际sim dt下的steer变化率P95、真值acceleration·right_vector的横向加速度P95，以及实际平均速度；与分析CSV误差均小于1e-9。没有调用原分析器的计算函数或重跑其测试。

八个主要窗口的全部帧数依次为：26966 baseline/candidate 70/71，24240 117/117，17563第一个S 70/70、第二个S 71/71。每例完整控制帧与独立truth trace帧逐值相同且连续；所有主要窗口原始字段无缺失，最小实际速度5.6757m/s，全部窗口all与moving≥2m/s样本数相同。因此四项结论没有靠低速剔除、删帧或缩小分母取优。全程停车和trajectory_behind等安全控制仍保留；其合法安全制动与非法输出的区分已写在首例前冻结的协议说明里。

运行时归档配置唯一差异是 `max_lookahead_time_s:.5→.375`；PI、3m下限、车辆标定和转向限制一致。冻结controller SHA `701c929a6c86477d9c5d1847181b94b8ae828f9ab75951a84dbfb723a289f87b` 与G1及当前turns源码一致。默认等价引用此前已经完成的112测试、18种组合/432tick历史golden和18/18合成G1，本次未重复运行，也不扩张为对所有可能输入的形式证明。

协议SHA `08bec8b1081bacaf8e6b548504b99687338ff90a5c5fff36fe43a478eabc04e4` 与本轮启动前归档完全相同；分析脚本SHA `8e0f3238bdb9af5d2969a7562a88f4701860cf602a4ad7b5d552efeda3f0ef64` 与启动前归档相同。分析manifest全部输入SHA均复核通过。`required-conditions.json` SHA `d39977d86e1fb42eb107b3d0b64dddc41940b9eaf75832f59417afb38682354f`。

主文档新增转弯结果段落的7.56%、34.95%、73.07%、13.49%、4/126和“未反序确认/未升级默认”表述正确。发现一处无关但已过时的旧句：“Real TCP controller comparisons remain a separate planned experiment.” 实际paired-v2已经闭合，应改为独立完成的真实TCP纵向对照并链接其结果；已通知根代理处理。本文不修改主文档。

有效结论限于固定无交互oracle转弯开发集；短前视改善部分外侧残差，但未达到主收益门槛且增加部分控制变化/横向加速度。没有据此确认真实TCP横向迁移、全路线收益或新默认资格。
