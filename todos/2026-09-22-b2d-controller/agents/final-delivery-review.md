# 最终交付只读审阅

2026-09-23，正式 v4 campaign 尚在执行时审阅。只读源码、计划、文档与已完成证据；未重跑测试、模型、GPU 或 CARLA，未执行 git。此文只列交付动作，不修改已冻结控制参数，也不把当前候选提前宣布为新默认。

## T0–T8 / G1–G5 对照

| 项 | 当前证据与结论 | 最终交付动作 |
|---|---|---|
| T0 基线/契约 | 原始失败、源码/输入快照、实测常数及 research 反证已存在 | 更新过期契约和研究头部；清楚列各阶段源码/config，而非只列最新 HEAD |
| T1 标定 | 轴距/后轴/最大转角、km/h曲线与gyro符号均有实测 | 已足够；保留轮胎/动态转向非精确自行车的限制，不需要重新标定 |
| T2 / G1 | NumPy控制器、vendor黄金回归、PI状态/复位与数学测试已实现；v4 PI G1为14/14 | **现有14例中的圆弧是additive，不是冻结max组合**，见下述P0；CARLA旧圆弧失败不得被“所有控制器G1通过”覆盖 |
| T3 / G2 | v4六条件×三配置18/18驾驶完成、17/18gate通过；PI CARLA与max各6/6，additive26966 p95失败 | 六条件、失败、原selection分支完整保留；最终配置坡道hold待campaign完成。原v1坡道不能冒充新配置验证 |
| T4 | 双频率、源时戳历史、v2重接路径、有限compass预测和超时制动/复位已有实现及测试 | 更新contract字段；不把重接路径改善归因于控制器独自贡献 |
| T5 | CLI/每preset归档config、报告、全attempt/失败/重试日志已存在 | 最终逐route×preset×seed长表、source index与配置身份对齐；源文件快照与测试执行结果是不同证据 |
| T6 / G3 | 恢复smoke官方2390完成100/DS100；213帧、1次NaN预测、20项独立审计PASS | 两次86.11% Agent crashed仍属于证据；新smoke不是无NaN输入，也不是停车保持测试 |
| T7 / G4 | 正式60例、三组两TM seed继续运行；参考为CARLA lateral+PI .5/.25、TCP vendor，候选max+PI | 完整分母、首碰撞前/全程、未知归因、逐seed强参考比较后才决定；G2资格不等于默认替换 |
| T8 / G5 | v1、v2、v3/v4、舒适性和smoke图/CSV/source索引基本齐全 | 冻结最终结果版本、成本、研究/使用说明、交付commit与尚未完成项；真实TCP仍是后续方案 |

## P0：正式结果之外需要收口的证据/协议点

1. **精确候选的G1配置标识。** `results/v4-pi-g1/summary.json` 的circle entries写 `lookahead="additive"`；`scripts/b2d_controller_pi_selftest.py` 和 `test_b2d_controller_pi_gains.py` 没有传max。因此“PI .5/.25 14/14”正确，“最终max+PI完整14项均已验证”不成立。最小动作是给冻结max组合增加明确配置的小型离线验证并归档，或明确说明目前证据为组件测试+精确配置6/6真实G2，不能重命名原artifact。此审阅没有重跑测试。
2. **原开发复跑条款。** `agents/v2-analysis.md` 在选择规则后还写“两列各5条seed=1复跑”。目前看到v4 G2单次18例，以及后续Dev10两TM seed；未找到同五条无交互开发路线的seed1记录。两者并不是同一项试验。根代理应明确是补齐还是记录偏离/覆盖理由，不要静默把正式TM seed当原开发复跑已经完成。原文也明确“只有max通过可列候选”，该分支已经在v4选择报告正确保留，不应重新收紧成必须双方通过。
3. **新配置坡道hold** 属于G2必要旁证，campaign末尾现已计划；必须记录实际坡度、signed speed和5s位移，不用v1通过替代，也不称坡道接近/减速验证。若无合格坡度则显式untested。
4. **G4资格与兼容默认分开。** CLI目前仍是carla/vendor。它是兼容默认，不意味着它通过了所有新增G1/G2场景；新候选若不满足G4，不更改CLI默认，也不能因此抹掉v4速度/jerk的真实改善。若G4通过，后续已见过的六条holdout只能称确认集，不是blind holdout。本文不预判最终结果。

## P1：已有文档具体要修的地方

- **复现命令不匹配当前正式实验。** `docs/b2d-controller.md` campaign示例给所有preset同一个旧 `results/controller_config.json`；它不会复现PI-CARLA/TCP-vendor/PI-max。应补上冻结 `results/v4-freeze/preset-configs.json` 的精确 `--preset-configs` 命令，注明旧命令只是vendor兼容示例。不要用旧 `campaign-v2` 名称暗示当前formal-v4。
- **contract.md过期。** 仍说标定默认稍后提供，constructor没有 `longitudinal_mode/pi_kp/pi_ki`；没有 `motion.jsonl` 非有限字符串编码、`pose_status` 的degraded/age、compass<=.2s预测、超时制动、复位后强制新轨迹，以及故障tick日志边界。与实现/使用说明同步，不要让后续planner按旧契约接入。
- **研究/索引头部状态落后于正文。** `research/trajectory-to-control.md` 首行仍称首轮Dev10完成、第二seed/holdout运行中，正文已追加v4；共享README仍笼统“独立v2开发继续”。`docs/b2d-controller.md`末尾Last verified仍为2026-09-22。最终按阶段给“v1完成、v4正式完成/不合格/确认待做”等状态，历史进展段落则保留时间上下文。
- **阶段版本表。** 参数/控制核心d140e30、集成修复1ee2eb2、正式执行归档2cca3a9等各有含义；列源码hash和config hash对应实验，避免把文档后续commit当已跑的运行时代码。
- **目录交付。** 原计划G5写 `research/results/b2d/controller/`，当前实际材料在已接受的 `todos/.../results/`。该旧目录不存在；建议把计划/研究链接指向共享目录并明确路径调整，不为字面一致复制大份CSV/图。审阅时两个工作树的controller/adapter源码字节相同，`docs/b2d-controller.md`不同；最终确认用户工作树能看到最新文档和准确交付路径。
- `iteration-v2.md`的现有v4、舒适性、NaN、真实TCP说明总体准确。完成正式阶段后更新“正在运行”，补上最终结论链接；不需要重写已保留的历史失败。

## P1：最终G5材料仍需一起封存

1. Campaign退出并停止server之后再做最终逐文件inventory；把基础设施首失败、重试、取消、partial文件都计入，不覆盖中间版审计。正式配置是每preset不同，表中身份必须可追溯到各自归档JSON。
2. 各seed/preset全10条与最终采用attempt长表；tick加权和route等权、全程与固定首次碰撞前、已完成/失败tick分布都列。Official status、completion、DS、Success Rate口径不混淆；低进展和scene/controller/unknown归因不能删正式失败。
3. 新阶段成本追加到 `docs/bench2drive-cost.md` / `docs/tokyo-box.md`，勿覆写旧seed0观测为新实验。保留setup/loop/cleanup、server启动/重试、失败smoke与capture、模型/无模型及GPU身份的边界；如果只有总wall和loop，余量写“未分离开销”，不杜撰setup/cleanup拆分。
4. 把已有最终测试执行记录与源码快照链接清楚；不必为文档更改重跑全部测试。源码被归档不等于曾执行过测试，测试数量也须跟当时版本对应。最终commit/push状态由根代理核实；没有push证据就不宣称已push。
5. 用户要驾驶行为：主结论同时列速度波动、stop/恢复、控制饱和、碰撞/进度、纵横向加速度/jerk。v3→v4纵向jerk降低25.9%、横向加速度增加2.0%的成对诊断已有证据，不能被单一DS结论掩盖，也不能包装成全面舒适性或官方Smoothness分数改善。

## 真实TCP范围：明确后续，不扩成当前未完成实现

`agents/tcp-controller-integration-plan.md`已经完成官方路径/checkpoint/4×0.5s时域/坐标/混合与末端规则审计。旧tcp-smoke/tcp-fast跑过真实模型但没有新旧控制器对照；当前formal-v4仍是policy=none。

最小后续建议是固定网络/原生desired speed与转向，仅比较native纵向和PI，并共同处理执行限制，按三路线两臂6例再确认范围。完整pursuit需要原生短时域API及标签物理原点证据，不能补成5s或把oracle重接路径替代模型输出。用户目前强调驾驶表现，不自动授权更大的9例/完整模型接入实现或新增调参网格；最终回报应把“已有行为诊断收益”与“真实TCP尚待配对验证”分别说清。
