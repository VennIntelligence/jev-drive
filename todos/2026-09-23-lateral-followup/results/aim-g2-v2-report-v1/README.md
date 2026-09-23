# Hermite 六例冻结验收：未通过

2026-09-23。唯一变化是 pursuit aim 的 linear→local chord-length Hermite 插值，max 前视 .5、PI .5/.25、速度、适配器和限幅不变。实际成功运行是 `aim-g2-v2`，六例原 G2 均通过、无碰撞；**147 项必要条件中 143 通过、4 失败、0 证据不足，总体 FAIL**。本轮不将 Hermite 升级为合格候选或默认控制器，不触发“筛选全通过后”的确认。

此前 short .375 的旧 15% CTE 主目标失败仍保持原判。本轮是插值纹波的新机制试验，不能以基础 G2 或部分 jerk 下降追认旧目标。这里是无背景交通的导航 oracle 闭环，不是真实 TCP 模型、交互驾驶或 220 路榜单结果。

## 必要条件失败

| 条件 | baseline | Hermite | 冻结上限 |
|---|---|---|---|
| turn/26966/1/cte_p95_delta | 0.849224 | 0.903508 | 0.899224 |
| turn/26966/1/cte_rms_delta | 0.558694 | 0.589129 | 0.588694 |
| turn/26966/1/primary_emitted_rate_20percent | 0.800106 | 0.940687 | 0.640084 |
| turn/24240/1/primary_emitted_rate_20percent | 0.218380 | 0.232566 | 0.174704 |

右急弯 CTE RMS 超余量约 0.000435m、P95 超余量约 0.004284m；即使差距较小，也按原精度判失败，不四舍五入放行。两个目标窗 emitted-rate P95 均上升，未达到各降至少 20% 的主要求。

## 四窗口完整指标

以下全部为预登记 pad 窗口、全帧统计，左右正号和物理量定义保持冻结协议。`B→H` 表示本轮 linear/.5 基线到 Hermite；不使用窗口平均抵消失败。

### CTE RMS m

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 0.558694 | 0.589129 | 0.030435 |
| 24240 左弯 | 0.107807 | 0.115557 | 0.007750 |
| 17563 S1 | 0.235236 | 0.236510 | 0.001275 |
| 17563 S2 | 0.273198 | 0.283736 | 0.010537 |

### CTE abs P95 m

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 0.849224 | 0.903508 | 0.054284 |
| 24240 左弯 | 0.179908 | 0.192478 | 0.012570 |
| 17563 S1 | 0.504773 | 0.485905 | -0.018868 |
| 17563 S2 | 0.483505 | 0.513068 | 0.029563 |

### CTE abs peak m

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 0.856458 | 0.911731 | 0.055273 |
| 24240 左弯 | 0.182633 | 0.195621 | 0.012988 |
| 17563 S1 | 0.623970 | 0.601892 | -0.022078 |
| 17563 S2 | 0.563005 | 0.601529 | 0.038524 |

### emitted-rate abs P95 /s

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 0.800106 | 0.940687 | 0.140581 |
| 24240 左弯 | 0.218380 | 0.232566 | 0.014186 |
| 17563 S1 | 2.000000 | 2.000000 | 0.000000 |
| 17563 S2 | 2.000000 | 2.000000 | 0.000000 |

### lat jerk abs P95 m/s³

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 33.811203 | 31.664052 | -2.147150 |
| 24240 左弯 | 11.891843 | 8.737700 | -3.154143 |
| 17563 S1 | 44.689580 | 42.892512 | -1.797069 |
| 17563 S2 | 50.214285 | 45.594195 | -4.620090 |

### lat accel abs P95 m/s²

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 8.598159 | 8.816088 | 0.217929 |
| 24240 左弯 | 1.953200 | 1.913198 | -0.040002 |
| 17563 S1 | 5.772651 | 6.032548 | 0.259897 |
| 17563 S2 | 6.909302 | 6.640622 | -0.268680 |

### heading abs P95 °

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 4.271547 | 4.672212 | 0.400666 |
| 24240 左弯 | 1.701105 | 1.726401 | 0.025296 |
| 17563 S1 | 4.298981 | 4.248088 | -0.050893 |
| 17563 S2 | 5.744524 | 6.160853 | 0.416328 |

### speed error RMS m/s

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 0.149878 | 0.149376 | -0.000502 |
| 24240 左弯 | 0.153223 | 0.155207 | 0.001984 |
| 17563 S1 | 0.116021 | 0.125392 | 0.009372 |
| 17563 S2 | 0.148391 | 0.140369 | -0.008022 |

### mean speed m/s

| 窗口 | B | H | H−B |
|---|---|---|---|
| 26966 右急弯 | 7.885128 | 7.886454 | 0.001326 |
| 24240 左弯 | 7.882531 | 7.881544 | -0.000988 |
| 17563 S1 | 5.965469 | 5.966490 | 0.001022 |
| 17563 S2 | 6.001075 | 6.001451 | 0.000376 |

四窗物理横向 jerk P95 均下降，但这个分位数不能代表完整舒适性、安全或所有瞬态。物理 jerk 是 world acceleration 先按实际时间求导、再投当前 body；它与归一化 steer-rate 是不同量，不可互相替代。四窗 CTE RMS 均上升；S1/S2 emitted-rate P95 仍为 2/s，不能描述为完全消除动作跳变。

## 覆盖、模式、故障与恢复

| case | 全帧/truth | requested | used | fallback |
|---|---|---|---|---|
| 26966/baseline-linear | 387/387 | {'linear': 387} | {'linear': 285, 'None': 102} | {'None': 387} |
| 26966/candidate-hermite | 388/388 | {'hermite': 388} | {'linear': 2, 'hermite': 284, 'None': 102} | {'two_or_fewer_distinct_knots': 2, 'None': 386} |
| 17563/baseline-linear | 559/559 | {'linear': 559} | {'linear': 457, 'None': 102} | {'None': 559} |
| 17563/candidate-hermite | 559/559 | {'hermite': 559} | {'hermite': 457, 'None': 102} | {'None': 559} |
| 24240/baseline-linear | 434/434 | {'linear': 434} | {'linear': 336, 'None': 98} | {'None': 434} |
| 24240/candidate-hermite | 433/433 | {'hermite': 433} | {'hermite': 332, 'None': 101} | {'None': 433} |

六例共 2760 个控制/truth 帧，全部连续对齐；0 invalid_control、0 unknown reason、0 requested-mode 缺失/错误或 used/fallback 契约错误。Hermite 共 1380 帧：1073 帧使用 Hermite、2 帧有理由地回退 linear、305 帧没有求 aim；基线 1380 帧：1078 帧 linear、302 帧没有求 aim。无 aim 的 null 不当作 Hermite 成功。两次回退均为 `two_or_fewer_distinct_knots`；逐帧位置保留在原 frames.csv。

窗口样本为右急弯 70/71、左弯 117/117、S1 70/70、S2 71/71，共 657 帧；各窗全部保留，均超过 20 帧门槛，没有低速排除。合法停车原因也不删除：trajectory_behind 共 11 帧，均在参考终点附近；stationary_trajectory 共 596 帧；stop_hold 共 3 帧。详情见 case-summary.csv 与 control-reasons.json。

右急弯两臂恢复均 censored：不能称 recovery pass，也不能证明恢复不恶化。固定 post10m CTE RMS 0.317045→0.302212m，单独的残差保护条件通过。另三窗恢复时间基线/Hermite 相同：左弯 0s、S1 0.1s、S2 0.7s；没有新增截尾，满足原时间余量。完整路线图保留未恢复区域，未为候选延长观察窗。

所有 147 项原始判定见 [all-conditions.csv](all-conditions.csv) 与 [required-conditions.json](required-conditions.json)。[update-phase-supplement.csv](update-phase-supplement.csv) 额外列出 update/non-update/unknown 与全窗 raw/emitted rate、jerk 分布；这是按已登记字段的补充展示，不增加或替换验收条件，同期差不能当作更新的独立因果效果。

## 完整路线与图

| case | G2 | ticks | 全程CTE RMS m | 全程CTE P95 m | 巡航speed RMS m/s | case wall s |
|---|---|---|---|---|---|---|
| 26966/baseline-linear | PASS | 387 | 0.257783 | 0.773178 | 0.123041 | 1.209121 |
| 26966/candidate-hermite | PASS | 388 | 0.269674 | 0.810441 | 0.123356 | 1.174517 |
| 17563/baseline-linear | PASS | 559 | 0.145055 | 0.356844 | 0.098362 | 7.336606 |
| 17563/candidate-hermite | PASS | 559 | 0.147576 | 0.386262 | 0.098376 | 6.399616 |
| 24240/baseline-linear | PASS | 434 | 0.081230 | 0.158033 | 0.086944 | 1.579704 |
| 24240/candidate-hermite | PASS | 433 | 0.086224 | 0.170676 | 0.085326 | 1.423969 |

基础 G2 全程含停车阶段，而逐窗使用固定 station 窗；两个范围不能混用。下面为首例前冻结 plot_aim 原样输出，未为结果改绘图脚本或截取有利片段。每图同列两臂，共 7 PNG + 7 PDF，3 份全路线逐帧绘图 CSV。

[route-26966-window-1 PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-26966-window-1.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-26966-window-1.pdf)

[route-24240-window-1 PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-24240-window-1.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-24240-window-1.pdf)

[route-17563-window-1 PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-window-1.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-window-1.pdf)

[route-17563-window-2 PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-window-2.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-window-2.pdf)

[route-26966-full PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-26966-full.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-26966-full.pdf)

[route-24240-full PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-24240-full.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-24240-full.pdf)

[route-17563-full PNG](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-full.png) · [PDF](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1/route-17563-full.pdf)

窗口图包括等比例 world XY、CTE、raw/emitted steer、实际/参考速度、横向加速度/jerk、steer-rate、heading 及更新/插值事件；full 图覆盖起步、转弯、停车、恢复观察区。原始脉冲和终点安全制动均保留，没有平滑。

## 运行成本与来源

成功六例 campaign start→end 为 **49.767866s**；六例 validation.wall_s 合计 **19.123534s**。前者包括 CARLA/map 启动开销，但 end 在最终 server.stop 之前；后者是每例 actor setup 至 cleanup/指标写入的计时，并不是只计 simulation step。

此前 aim-g2-v1 因 CARLA 端口启动失败，**0 例**，另记 182.182474s 基础设施成本；不拼入六例控制表现或剔除成本。两次已记录 start→end 合计 231.950340s，不包含驱动恢复准备及本次离线分析时间，也不作 harness 加速因果结论。

原始闭合运行：[/data/runs/b2d/controller/lateral-followup/aim-g2-v2](/data/runs/b2d/controller/lateral-followup/aim-g2-v2)。冻结分析：[/data/runs/b2d/controller/lateral-followup/aim-g2-v2-analysis-root-v1](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-analysis-root-v1)。图/全帧 CSV：[/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1](/data/runs/b2d/controller/lateral-followup/aim-g2-v2-figures-v1)。运行源码 18d6790；协议 SHA 624a72c286282efec63538e1ddaa26c05438f98cc9aa665110fd165ea30e6143，验收器 c940883c4753a1a8409649272bd28be31da61f9313df6e58949ec0479745fef3，plotter ac4add41911b8199d03cfda8fc77cdc781e7c188810e3fb33a6d1d2a236fdd42。原始和图文版本完整保留，manifest 列出输入与输出 SHA256。

理想圆轨迹插值诊断与历史真实 state shadow 的相反现象均保持原样。本轮闭环结果只否定该候选满足预登记收益/保护条件，不能据此证明所有几何插值无用，也不能证明侧向运动模型已是唯一主因。后续其他机制若获授权，必须保持独立协议和结果，不能在本轮调阈值补判。
