# 转弯开发阶段二：冻结六例协议

登记日期：2026-09-23；本协议在新一轮CARLA转弯matrix开始之前编写。真实TCP纵向六例使用主工作区冻结版本，本轮使用独立 `/data/worktrees/jev-drive-controller-turns`。这是已见开发路线上的新假设检验，不是盲测、真实TCP横向结果或默认控制器资格证明。

## 假设与唯一变化

26966右急弯的历史pursuit/max在弯后半段与出弯存在**外侧跟踪残差**。较短前视可能增强局部纠偏、减小该残差；其对动态滞后、噪声和回摆的影响由本次闭环决定。**不将它表述为已经证明的corner cutting。** 先前协作消息误读过CTE符号，已在本轮运行前更正。

唯一配置差异：`max_lookahead_time_s`，baseline显式`.5`，candidate显式`.375`。前视公式`max(3, coefficient*speed)`，3m下限、steer_rate=2/s、max_steer=.8、实测车辆参数保持不变；两组均pursuit/max、PI Kp=.5/Ki=.25、near速度窗、同v2adapter/定位过滤器。8m/s前视4→3m；实际速度≤6m/s两组均3m，S的少量超6m/s帧仍可能不同。不得在运行中调整参数、提高转向限制、降低巡航速度或修复路线几何。

## 路线、顺序与输入

| 顺序 | route | cruise | 预定义核心区间m | 主要评价窗口m |
| --- | --- | ---: | --- | --- |
| 1 | 24240 左弯 | 8m/s | 23–59 | 18–64 |
| 2 | 26966 右急弯 | 8m/s | 29–46 | 24–51 |
| 3 | 17563 第一个S | 6m/s | 32.5–43.5 | 27.5–48.5 |
| 3 | 17563 第二个S | 6m/s | 78.5–90 | 73.5–95 |

每条路线按baseline→candidate相邻执行，总计6例；17563的一次运行包含两个分别评价的窗口。路线取旧development-v4对应XML的三条原始元素，其scenarios原本为空，保持waypoints/weather；整条跑完并检查终点5s停车。24240窗口是约49°高曲率核心，整条路线约90°，因此同时展示整段轨迹和窗口外误差。不得把低曲率部分删除后声称已经评价完整90°过程。

根代理创建并冻结 [routes.xml](routes.xml)、[baseline配置](configs/baseline-max.json)、[candidate配置](configs/short-max.json)、[variants](configs/variants.json)、[cruise映射](configs/route-cruises.json)。矩阵用已有 `scripts/b2d_controller_validate.py`，默认1800tick上限保持不变；同一server跨case复用、地图改变用load_world，每例重建actor/agent及控制器状态。所有首次尝试、异常、超时和失败保留，不用重试替换原始结果。仅基础设施失败可另行记录重试，路线行为失败不通过重试取优。

```bash
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python scripts/b2d_controller_validate.py \
  --routes todos/2026-09-23-tcp-controller/turns/routes.xml \
  --variants todos/2026-09-23-tcp-controller/turns/configs/variants.json \
  --route-cruises todos/2026-09-23-tcp-controller/turns/configs/route-cruises.json \
  --out /data/runs/b2d/controller/turns-v1 --server-index <root-selected-free-index>
```

根代理启动前归档源码、实际配置/路线bytes与SHA、GPU UUID、server端口/版本。上述命令是复现模板，本文编写者没有启动CARLA。矩阵 `inputs/matrix.json` 的配置和执行顺序是最终机器可读索引；若与本协议不一致，先纠正输入而非先跑。

## 独立指标与符号

原G2全部gates原样保留：completed、no_collision、全程CTE RMS≤.5m/P95≤1m、固定5s启动宽限后的巡航误差RMS≤.5m/s、定位P90≤.5m/航向定位P90≤1°、终点≤1m、保持≥5s/位移≤.1m/速度<.1m/s、遥测帧完整。原校验器的gate_pass不包含下述转弯门槛，必须另列。

主要转弯报告用分析代理的新 `analyze_turns.py`；nearest-segment真值投影与5m居中弦航向定义沿用旧 `agents/lateral-evidence-v2/analyze.py`，使用已冻结的上述station窗口，但CTE统一为与validator一致的左正。新helper在首例前冻结/归档，后续不得根据结果改变定义。各case先比对参考线与旧窗口参考一致，再使用同一窗口；投影不会读取控制器估计progress。未到窗口、缺失真值、断帧、指标无有效样本均显式失败或证据不足，不以finite过滤后的剩余样本暗示完整通过。

CTE符号必须写在图轴：本轮新分析器使用`ty*residual_x−tx*residual_y`，在CARLA世界中**左正**，与validator一致。旧封存v2分析器使用相反的右正，绝对值/RMS不受影响，但历史签名曲线不能直接拼接。26966右弯外侧在本轮/validator为正、在旧v2离线分析器为负。航向跟踪误差用truth yaw减参考弦航向、wrap至±π后转度；不要与pose_heading_error混淆。

每个窗口输出全部帧以及speed≥2m/s辅助子集，主结论使用全部帧。记录样本数、原始frame范围、是否完整进入/退出、参考/实际速度、CTE RMS/P95/峰值与峰值frame、航向RMS/P95、raw/emitted steer、变化率、限幅比例及rejoin concern。原始指标不做事后低通以改善分数；如展示滤波曲线须另标，并保留原始值。

entry诊断为pad起点之前5m；core由表中区间；exit为core末尾至pad末尾；post为pad末尾之后10m，各自报告。恢复诊断从第一次core exit起观察到pad末尾+10m：首次连续≥.5s满足|CTE|≤.25m且|heading|≤5°，记录该连续段起点相对core exit的时间。未观察到满足条件记censored，不记恢复0s或通过。每个S分别计算，不能只汇总平均。

控制变化率用连续frame实际sim dt，方向为CARLA steer右正；raw与emitted差异只证明发生输出限制，另根据前帧和2/s阈值区分slew与幅值。`validation_trace.applied_control`是当前新命令施加前读到的旧控制，不能与同frame新命令的差值当作执行器误差。CARLA angular_velocity字段为deg/s，横向加速度用真值acceleration与right_vector点积，不读取或反馈到控制。

## 候选的必要通过条件（首例前固定）

1. 六例均满足原G2门槛；两组覆盖全部四个固定窗口和终点。任何碰撞、blocked、invalid控制或未覆盖窗口保留为失败，不能靠moving过滤剔除。
2. **主目标26966：** candidate主要窗口CTE RMS相对本轮baseline降低≥15%，P95不升；post10m CTE RMS不升。全程均值只作补充。
3. **逐窗回归限制：** 每个窗口CTE P95增量≤.05m、最大绝对CTE增量≤.10m；24240及两个S各自RMS增量≤.03m。每窗航向跟踪P95增量≤1°；分别展示entry/core/exit，防止只改善一个局部却在出弯留下更差残差。
4. **速度限制：** 每窗参考速度误差RMS增量≤.10m/s、平均实际速度下降≤.20m/s。仍保留低速/停车帧；通过减速获取更低CTE不构成纯横向改善。
5. **舒适性代价：** 每窗实际横向加速度绝对值P95增加≤10%，emitted steer变化率绝对值P95增加≤20%。这些是此次开发取舍阈值，不是通用车辆安全标准；近零基线/样本不足时标证据不足并交根代理审阅，不能自动计通过。jerk及恢复时间完整报告，但不额外发明运行后的隐藏阈值。

阈值逐窗口应用，不从时间池化平均取优。任何必要条件失败则本次候选不通过，不改阈值或自动扩参数网格。若全部满足，只列开发候选；根代理可按同三路线B/A顺序另6例确认重复性后再决定扩大范围。同seed/同图/同天气不保证传感器噪声或actor世界完全相同，不宣称一轮已证明统计显著提升。

## 已有前置证据及边界

[实现与审查记录](../agents/turns-implementation.md)、[候选合成G1摘要](../agents/turns-candidate-g1-summary.json)、[原始证据SHA索引](../agents/turns-candidate-g1-hashes.json)：112 controller测试通过，18/18合成G1通过；默认与旧源码432tick/18种组合逐值等价，镜像/延迟/低速floor验证通过。完整raw/source在 `/data/runs/b2d/controller/turns-candidate-g1-v1`。这是本轮G2前的新开发证据，不重新解释此前正式G4结论。

本轮输入是oracle的20点/5s分析路线；真实TCP只有4点/2s且物理原点尚未确认。禁止补点伪造5s、用真值dense route或oracle rejoin修复模型预测。此次结果仅能说明当前plant/adapter下固定转弯路径的跟踪差异，不能直接宣称真实模型、交互驾驶或跑榜提升。

## 首例前的指标实现说明

“invalid控制”指非有限/越界/踏板冲突的输出，以及明确invalid输入、pose或stale等异常；trajectory_behind返回的是已定义safe brake，单列原因、station和终点上下文，不自动等同非法控制，也不删去这些帧。每个主要窗口少于20个有效样本标证据不足；舒适性相对比例的基线绝对P95≤.01（对应指标单位）也标证据不足，不能计通过。上述实现解释在本轮首例前固定，所有原始原因和数值保留。
