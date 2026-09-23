# 横向后续：轨迹更新跳变与动作代价

2026-09-23。用户已授权继续解决横向问题。本目录保存独立协议审查、历史[草案](protocol-draft.md)及[最终冻结协议](protocol.md)。**Hermite六例闭环已完成：147必要条件143通过、4失败，候选拒绝；固定系数定位传播六例也已完成：四窗CTE均下降，但156项条件仍有1项航向失败**。候选为pursuit目标点linear→rotation-equivariant local chord-length Hermite插值，max .5/speed_at/adapter/PI/限幅不动；可执行输入与逐窗验收已经根代理审定。

上一轮`.5→.375`短前视候选没有通过：26966窗口CTE RMS仅下降7.56%，未达原15%目标，且有转向变化率/横向加速度代价超限，共4项必要检查失败。这一结论保持不变。新轮基线仍为原`max(3,.5*speed)`，不能换成失败的`.375`让新候选更容易获胜。

本轮拟检验不同机制：固定轨迹更新时刻的参考变化是否放大控制跳变，能否在保留跟踪、速度和恢复表现的前提下降低动作代价。同状态单次更新干预已经提供瞬时机制证据；这些结果不能追认旧短前视15%目标通过。四个固定窗口、三条完整路线与原G2/逐窗guardrails继续保留，真实TCP四点/2s与本轮oracle20点/5s证据仍分开。

冻结条件：左右两个目标窗emitted steer-rate P95各降≥20%且physical lateral jerk各下降；两个S的rate/jerk代价上限；逐窗CTE/速度/恢复回归余量及有限试验/反序确认安排。无噪声圆轨迹扫描已提供插值机制线索，但不替代闭环验收。每项均已在首例前冻结；不根据候选结果增加参数网格或挑有利窗口。即使本轮开发筛选通过，也不升级默认或推断榜单提升。

外部专家目前不是启动分析的前置条件：帧同步、几何/符号、轨迹更新相位、目标点跳变、限幅和API命令交付都可在本地继续核验。若排除这些后仍存在无法区分的执行器时滞、车辆物理或参考构造机制，再提交带原始轨迹/控制/真值、冻结源码、逐窗图表和具体问题的证据包。草案末尾列了触发条件与最小交付；控制诊断没有联系外部人员；随后为恢复驱动从NVIDIA官方源下载匹配库。

依据：[旧冻结协议](../2026-09-23-tcp-controller/turns/protocol.md)、[独立失败复核](../2026-09-23-tcp-controller/agents/turns-final-review.md)、[更新同期诊断](../2026-09-23-tcp-controller/results/turns-v1/replan-diagnostic/manifest.json)。本次所读来源SHA见[review-sources.json](review-sources.json)，旧结果原样保留。

[验收/图表helper与验证](analysis/README.md)已经完成；[首例前负面shadow证据](negative-evidence-before-live.md)完整保留，不修改门槛或预报收益。


## 当前阶段结果与继续任务

Hermite仅插值这一改动没有通过：急右弯CTE RMS/P95回归超限、左右窗转向变化率未达到下降20%，共4项必要失败。六例G2通过、左右physical jerk下降也不能覆盖。首轮启动失败和成功重试分别保留于[启动失败](results/aim-g2-v1-startup-failure/README.md)与`/data/runs/b2d/controller/lateral-followup/aim-g2-v2`；[驱动恢复证据](diagnostics/driver-recovery/README.md)记录用户态隔离、实际加载库与GPU UUID。

[后轴传播审查](agents/pose-propagation-audit.md)、[单系数留出检验](agents/pose-lateral-regression.md)和[完整定位重放](agents/pose-compensation-replay.md)形成新的机制假设。右急弯定位改善但左弯退化，必须进入实际闭环再判断。[文章诊断图](pose-diagnostic-figures.md)保留逐点CSV和全部图版。

根代理已批准[固定k后续协议](pose-followup/protocol.md)：只增加一个显式后轴侧向传播项，6例开发，全部条件通过才最多6例反序确认。参数、旧默认、四窗和主要目标已经冻结；实现、125项测试、15项验收测试、独立重放及六例闭环均已完成。急右弯CTE RMS降低23.21%，但航向P95增加1.178°超1°余量，仍判不通过，不触发反序确认，不自动扫参。

[外部审阅问题包](expert-brief.md)已准备但未发送。当前本地仍可区分具体机制并检验，没有必须外部接手的阻塞。

[本轮最终结论、全部工作和下一步](final-report.md)统一收拢两个候选、方向诊断及恢复/归档证据。
