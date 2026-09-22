# v2：26966弯道误差诊断与有限lookahead对照

2026-09-22；只读旧原始数据，未运行CARLA，未改controller/adapter或旧结果。分析对象为`/data/runs/b2d/controller/development3/26966/{carla,pursuit}`。本文件位于独立v2 worktree。

## 结论与可检验机制

26966的pursuit**通过了已定G2门槛**，但比CARLA差，且p95距离1m门槛只有约1.7cm余量。主要问题是**弯后半段的向外偏移与出弯恢复**，不是起步，也不是这段的硬转角限幅。固定约7m的additive lookahead会同时看到半径约7m的急弯和之后更平的出口；纯几何转向没有CARLA角度PID同样强的前期动作。原始数据支持“目标点预瞄与曲率变化造成误差形状差异”这一待验证机制，但不能仅凭两次闭环数据归因于lookahead，更不能断言换max一定改善。

同一观测状态、同一条冻结轨迹上，把lookahead从additive换max，确实会在弯中/出口请求更强右转；同时它在入弯前会出现更弱甚至短暂反向请求。后一现象与短目标点、估计横向偏移、原点连接段共同有关，强调了与adapter-origin-bridge v2修改的解耦必要性。

## 原始证据：独立真值而非controller内部误差

用`validation_trace.json`的独立后轴投影作为主误差；以frame连接`control.jsonl`，按独立`progress_m`对齐两次运行，不能直接拿不同preset的同一tick当同一地点。

| 指标 | CARLA | pursuit |
|---|---:|---:|
| 全程ticks | 392 | 393 |
| 全程CTE RMS m | 0.201480 | 0.342246 |
| 全程abs CTE p95 m | 0.441466 | 0.982723 |
| 全程abs CTE max m | 0.830195 | 1.211479 |
| 去前2s RMS m（既有固定定义） | 0.212329 | 0.360947 |
| 仅speed≥0.5m/s RMS m（补充诊断，不替代gate） | 0.241641 | 0.409309 |
| G2独立恒速段speed RMS m/s | 0.220584 | 0.266575 |
| fused pose全程p90 m | 0.365661 | 0.392294 |
| 碰撞 | 0 | 0 |
| 既有G2全部gate | pass | pass |

去掉起步反而略增RMS；长时间停车保持也降低全程误差，因此不能用全程RMS孤立解释运动时精度。本轮不更改gate，只并列运动段。

下面空间区间是**看过26966后用于解释的事后诊断**，不是额外验收集或新的选择门槛：

| 路线进度m | CARLA RMS m | pursuit RMS m | pursuit平方误差占全程比例 |
|---|---:|---:|---:|
| 0–20 | 0.0512 | 0.0646 | 0.8% |
| 20–28 | 0.0588 | 0.0610 | 0.2% |
| 28–34 | 0.5506 | 0.2749 | 2.5% |
| 34–40 | 0.6613 | 0.4053 | 6.1% |
| 40–48 | 0.2279 | 1.0577 | 48.6% |
| 48–60 | 0.2776 | 0.7612 | 37.8% |
| 60–81 | 0.0976 | 0.0969 | 4.1% |

pursuit在入弯28–40m其实更贴近参考，但40–60m贡献约86.4%平方误差。CARLA峰值在进度34.44m、t=6.15s，signed CTE=-0.830m；pursuit峰值在46.97m、t=7.85s，signed CTE=+1.211m。该路是CARLA world yaw由约-90.51°变至0.066°的右弯；validator定义left-positive，因此CARLA早期负误差是弯内，pursuit后期正误差是弯外。不是“两者同样的位置误差只差一个增益”。

## 几何、动作与定位的分解

参考`world_xy`总长77.4065m，validator另有3m终点延伸。原始polyline有两对完全重复点（原index18/19与36/37）；**曲率诊断必须忽略零长度段**，否则数值差分会造出±数rad/m的伪曲率。这里只为读图清除零段，没有改输入或重算验收指标。

去重后弯入口约30.4m，31–36m的离散曲率约0.142–0.144m⁻¹（半径约7m），38–44m约0.077m⁻¹（半径约13m），45m后变直。8m/s在最急段对应理想`v²κ≈9.2m/s²`，远高于离线鲁棒网格预先用的5m/s²可行界线；这不证明CARLA不能完成，但说明不能用轻缓圆弧的小误差承诺这条路线同样表现。**本轮不改26966巡航速度来换更漂亮结果**。

两种方法的现有lookahead都为additive，即8m/s时约7m。pursuit在进度28–48m共52tick中，`steer_limited`为0，最大命令约0.319；其全程23次limited主要发生终点低速（60m后21次，其中6次命令0.8）。所以全程最大steer=0.8不能解释弯道最大CTE。CARLA在相同进度区间最大命令约0.400，动作与pursuit不同，不能把两种preset的差异直接归于lookahead。

| 同一路线进度附近m | CARLA CTE m | pursuit CTE m | CARLA steer | pursuit steer |
|---|---:|---:|---:|---:|
| 28 | -0.131 | -0.098 | 0.167 | 0.109 |
| 32 | -0.708 | -0.377 | 0.301 | 0.269 |
| 36 | -0.746 | +0.010 | 0.270 | 0.318 |
| 40 | -0.394 | +0.737 | 0.212 | 0.258 |
| 44 | -0.143 | +1.089 | 0.088 | 0.154 |
| 48 | +0.194 | +1.178 | 0.035 | 0.044 |

这些不是同状态control-only对照：车辆位置、heading与估计误差均已不同。CARLA较强入弯动作对应早期内切，然后较早恢复；pursuit较缓入弯后在曲率降低处留下外偏。可检验假设是纯几何长lookahead对曲率变化平滑过多，而非加一整个bearing PID。

定位并非完全可忽略：pursuit峰值处fused pose误差0.411m，estimated-route CTE约+0.863m，独立真值+1.211m；估计低报约0.35m外偏。进度34–40m两种preset的pose p90均约0.49–0.51m，CARLA在40–48m甚至达到0.568m。全程pose通过并不保证弯中没有偏差。不过两组全程fused pose RMS都约0.209m，heading p90弯中CARLA0.381°/pursuit0.309°，不支持“pursuit比CARLA差主要因为其定位整体坏了”的强断言。

用实际yaw rate和speed反推理想自行车等效steer，仅作响应诊断：进度28–48m，pursuit的等效steer与当前command RMS差约0.0282，对前1/2tick command约0.0226/0.0207；CARLA为0.0529、0.0397、0.0410。数据与一至数tick的响应/采样滞后相容，但**不能据此标定100ms执行器延迟**：采样边界、左右轮Ackermann、轮胎侧偏和反馈相关性未分离，日志没有实测轮角。仅凭这次比较也不能排除plant响应对急弯误差的贡献。

## 同状态几何反事实：只用于决定比较是否值得做

取pursuit每次更新的原始local trajectory，保留现有原点桥接，按记录速度分别算两种aim和pure-pursuit原始steer；additive结果复现记录raw_steer。没有推进另一个plant或生成新闭环成绩。

| 原pursuit状态的进度m | additive原始steer | max反事实原始steer |
|---|---:|---:|
| 28.04 | +0.109 | -0.088 |
| 29.60 | +0.180 | -0.040 |
| 31.24 | +0.246 | +0.048 |
| 36.03 | +0.318 | +0.363 |
| 40.30 | +0.250 | +0.370 |
| 41.90 | +0.212 | +0.364 |
| 43.38 | +0.181 | +0.378 |
| 48.19 | +0.044 | +0.193 |

max在8m/s时约4m（6m/s时3m），能增加弯中纠正，但牺牲更早的入弯预瞄，还可能更敏感于GNSS/原点桥接。因此建议只跑已列出的additive/max两种，不做3/4/5/6/7m细搜，不加PID，不同时改steering gain或纵向gain。

## 最小开发矩阵与冻结顺序

**先完成独立的adapter-origin-bridge v2修正及其回归，再冻结adapter/定位/速度窗口/source hashes，之后开始lookahead比较。** 新旧adapter性能差异单独记录；下面两列必须使用同一份冻结v2 adapter，不能拿v1 additive对v2 max。若v2修正后不再存在原问题，应报告问题已缓解，不为了优化而换默认。

| 无交互开发几何 | 地图 | cruise m/s | pursuit additive | pursuit max |
|---|---|---:|---|---|
| 1773 | Town12 | 8 | 1次 | 1次 |
| 24240 | Town10HD | 8 | 1次 | 1次 |
| 26966 | Town05 | 8 | 1次 | 1次 |
| 25854 | Town03 | 8 | 1次 | 1次 |
| 新真实S弯17563 | Town12 | 6 | 1次 | 1次 |

核心矩阵**10个case**，固定seed=0、同车辆physics、20Hz控制/5Hz轨迹、near速度窗口、相同stop参数与输出限幅，唯一改动为`lookahead=additive|max`。17563是新增开发路线，不是保留集；既有6条holdout不用于调参。两列均跑满所有5条，不因26966单条改善提前停止。每条路线两列相邻运行以减少环境漂移，但重新spawn/reset车辆/agent状态，复用CARLA进程也必须清理场景；不能让第二列继承第一列PID/定位状态。

若需要宣称v2 pursuit优于CARLA，必须有**同一v2 adapter/config下的CARLA固定基线5个case**。如果adapter子任务已经产生这5条，就直接引用；否则增加5条，总计15条。旧development3 CARLA仅作历史诊断，不替代v2正式参考。这一基线不再搜索CARLA gain/lookahead。

控制器common gains、几何参数、传感器频率、noise配置、GNSS融合、route_end_extension、停车尾段、truth logger均冻结；保留原始`trajectories/control/validation_trace`及resolved config hashes。raw imu/yaw response可以追加记录，但不反馈改控制。

### 预先写定的判定与停止条件

- 主门槛仍为每条完整G2：完成、零碰撞、CTE RMS≤0.5m、abs p95≤1m、独立恒速RMS≤0.5m/s、pose/heading/停车保持/日志完整等原gate。失败都保留，不用速度或warmup特判救单条。
- 同时列全程、原固定2s之后、moving≥0.5m/s三组CTE，以及已有独立真实cruise指标；只有全程/原gate决定资格，moving用于暴露停车稀释。controller trajectory derivative speed与独立cruise速度分开。
- 两列全部5条都达到门槛后，按5条路线等权CTE RMS均值比较；建议只有max相对additive降低≥20%、且没有任何路线新增gate failure或abs p95增大超过0.1m，才列为替换候选。速度RMS任一路线劣化>0.1m/s则不选。该20%沿用主计划的明显改善标准，不在看结果后移动门槛。
- 若只有max通过5条、additive有真实控制失败，可将max列为候选并解释tradeoff；若两者均不通过，不继续无限搜索。先检查新S/26966的物理需求、输入轨迹可行性、时间/定位和执行响应，再决定是否触发已有计划的升级条件。
- 初轮有候选后仅追加**两列各5条seed=1复跑（10case）**，验证优势/失败不由单seed决定。近似持平或优势小于复跑波动就保留additive。没有候选则不自动增加运行预算。
- 每组按原计划不超过1h；超时保留全部attempt/缺失并查基础设施，不能缩减分母。

以上只是开发选择；不根据Dev10或holdout结果回调lookahead。最终G3/Dev10/保留集是否重跑由根代理在配置冻结后按计划决定。

## 复算路径与限制

只读分析helper：`/tmp/b2d_v2_26966_analysis.py`；输出`/tmp/b2d_v2_26966_analysis.{json,txt}`。helper按frame连接、独立进度分箱，可由根代理按需归档到v2 evidence目录；本次未覆盖任何raw文件。弯道曲率诊断去掉零长度段，不修改正式投影。所有空间分段都是事后解释；闭环因果必须由上面只改lookahead的矩阵检验。

输入SHA256（root为`/data/runs/b2d/controller/development3/26966`）：

| preset | raw file | SHA256 |
|---|---|---|
| carla | control.jsonl | `5ab7c7f76127cd3f61e5b6a518f45ee238d133d44c114684ce12b86b96275d75` |
| carla | validation_trace.json | `ad2960d69e7d37e7dcdc61b1a081fb1181fe450a24f05aa512f21677406b23ab` |
| carla | trajectories.jsonl | `115e1d163c7cd217852a8d677049fcd72d0814c45e971df4fb963a837c1090a1` |
| carla | route_reference.json | `3f4b77b0987bfa7689ac5e1d4752e15909e1d64cb53b09be7a4c202f12ab9979` |
| pursuit | control.jsonl | `b8c691de1082ca8716c0f00d78c36281c63b3f31bc10bcb08c68c676ff36ee6b` |
| pursuit | validation_trace.json | `228c7ad304c82599a4c763f597fde3a46072307be52cd78e3590c0b8cdbabdf9` |
| pursuit | trajectories.jsonl | `82d8b4d846aede5d41b71eac37265ce9dcc67ef1f34824918cf2d8db829be772` |
| pursuit | route_reference.json | `3f4b77b0987bfa7689ac5e1d4752e15909e1d64cb53b09be7a4c202f12ab9979` |
