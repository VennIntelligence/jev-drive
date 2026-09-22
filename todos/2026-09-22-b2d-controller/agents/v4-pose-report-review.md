# 定位修复后报告验收复核

2026-09-23 JST。本文仅记录报告边界；没有修改运行代码或脚本。

定位修复验证日志已归档到[verification-v4-pose](../results/verification-v4-pose/README.md)：11项专项与126项全套通过；motion-capture的首次失败与修正后10项通过同时保留。article-notes已补评分边界与v2/v4证据，其所有本地Markdown链接在交接前均有效。随后该文件交由根代理合并primary较完整的v1文章，报告代理停止编辑文章。

## G4现有helper与原协议

`scripts/b2d_controller_g4.py`固定CARLA共享PI(.5,.25)+additive参考、TCP vendor对照、pursuit max候选，并验证实际归档配置字节。要求两个seed各三组完整最终记录，官方TickRuntime保留分母，人为cap/缺项不成为成功；完整重试记录保留。

严格subset SR的记录条件为Completed/Perfect且除min_speed_infractions之外没有任何非空违规列表，分母为所需子集路线数。驾驶完成是completion≥100。DS使用官方已序列化score_composed；舒适性独立，不从速度RMS推断。

原计划“均值改善但失败增多时不选默认”需要额外检查：helper当前输出每组driving_completed_count，但未将每seed未完成路线数不增加列为机器条件。正式最终审计将逐seed额外列出reference/candidate失败route IDs及数量，并合并为验收必要条件；不能只引用helper的listed_g4_conditions_satisfied状态。

此外helper刻意不能自动完成：G1–G3完整资格、与复跑波动的比较、控制导致blocked/deviation的证据归因、正式确认/泛化判断。非零失败事件由负责失败审计的agent提供controller/external/unknown与原始证据，无法判明继续unknown。不根据某张低CTE图或碰撞事件单独推定控制器责任。

两seed的completion分别对照；主横向/速度为两seed等权route均值，每seed数值另报。全程/首次碰撞前分开；轨迹导数reference-speed与独立巡航真值、官方舒适性不混称。

正式60例启动manifest记录runtime commit `2cca3a9660d989de0d09a7cb1d4e51610c814417`。本文件不包含尚未完成的正式G4成绩，不选择部分结果赢家。

正式运行补充审计边界：原始`vehicle_blocked=0`/`route_dev=0`不能证明没有真实停滞。官方TickRuntime、持续低进展与驾驶未完成也必须单列，并由事件前后轨迹判断controller/external/unknown；事故后倒车或invalid_motion不自动成为初次碰撞的原因。若TickRuntime路线没有official blocked事件，不能仅凭计数0就宣称“控制导致blocked不增加”已完全证明。最终人工资格审查覆盖这些未标记的stall，保留未知归因。
