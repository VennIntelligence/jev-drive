# 真实TCP六例最终报告：公共限幅下的纵向执行对照

2026-09-23。paired-v2六例全部闭合，数据契约完整；PI满足冻结的**“值得继续验证的行为信号”数值条件**。这不等于安全验收通过：两臂均只完成2/3条路线，1773均碰撞后停滞至4000tick上限。PI在1773更早碰撞、停滞更久，碰撞前速度误差更高；不能把较低的全程jerk与速度误差直接解释为更安全或更好的交互。没有新默认、显著性或全220榜单提升结论。

## 固定范围与数据身份

使用真实TCP checkpoint，三路线24211/1711/1773，seed0，20Hz，only_traj四点/.5s时域，PI Kp=.5、Ki=.25；横向沿用原生仲裁/PID。两臂相同throttle[0,.75]、brake[0,1]与互斥，均移除原官方低速限油门和尾部二值化，因此比较的是公共执行限幅下B/C，不是未经修改的官方最终策略。相对旧官方更快不能归为PI或harness收益。

初版paired-v1已发现微负静止速度误guard，四完整+第五中断全保留；本报告只用修复后[protocol-v2](protocol-v2.md)重新完整执行的六例，绝不拼接。finite |raw speed|<.01归零只用于公共有效速度/反向判断，原生模型/PID仍读raw。主运行源码commit `c09c9688d491859a0d8d9e97c5409a65670add12`；checkpoint SHA256 `e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`。

## 全程主结果

每个受控tick均保留，包含首帧neutral、模型停车意图、故障、碰撞及停滞。速度误差为truth世界速度投影当前车头forward减模型desired。desired由四预测点间三段距离/.5得到，不加入原点桥。主jerk先对世界加速度按实际dt求差分，再投影当前body；无滤波或warmup删除。这些是本实验行为指标，不是自动复现的官方Driving Smoothness分数。

| 路线 | 臂 | 完成% | DS | 帧数 | 速度RMS | 纵向abs(a)p95 | 纵向abs(j)p95 | reverse guard帧 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 24211 | 原生B | 100.00 | 100.000 | 393 | 2.1675 | 5.6111 | 67.7002 | 5 |
| 24211 | PI C | 100.00 | 100.000 | 401 | 2.1168 | 5.6676 | 48.2215 | 0 |
| 1711 | 原生B | 100.00 | 100.000 | 520 | 1.6847 | 9.8639 | 147.2035 | 0 |
| 1711 | PI C | 100.00 | 100.000 | 499 | 1.3867 | 4.6208 | 72.9077 | 0 |
| 1773 | 原生B | 34.88 | 20.928 | 4000 | 4.2258 | 2.6921 | 38.5040 | 68 |
| 1773 | PI C | 34.88 | 20.928 | 4000 | 3.8725 | 0.6643 | 14.0355 | 47 |

前四例均Completed/DS100、无碰撞；两个1773均Failed - TickRuntime，完成34.88%、DS20.928、1车辆碰撞。两臂路线等权平均完成均78.2933%、平均DS均73.6427、driving-completed均2/3；若按完成且无有效扣分事件定义本子集成功，也是2/3。分母是3，不套官方全220脚本常量。MinimumSpeed仅保留原始事件（六例合计86条），当前版本不作为罚分解释；没有把它们当0事件。

| 三路线等权全程指标 | B | C | C−B | 相对变化 |
|---|---:|---:|---:|---:|
| 速度误差RMS (m/s) | 2.692664 | 2.458681 | -0.233983 | -8.69% |
| 纵向abs(accel)p95 (m/s²) | 6.055712 | 3.650913 | -2.404799 | -39.71% |
| 纵向abs(jerk)p95 (m/s³) | 84.469272 | 45.054934 | -39.414338 | -46.66% |

每条路线速度RMS与纵向jerk p95均下降，但24211纵向加速度p95上升0.056445m/s²（5.611118→5.667564），作为明确tradeoff保留。均值下降及安全事件/未完成计数未增加，符合冻结的有限继续条件；未完成本身没有变通过。核对结果见[acceptance.json](results/paired-v2-final-audit/acceptance.json)。只有三条近直导航路线、一个seed、每臂一次，不能推出更广泛的转弯效果。

## 1773：相同规则分段后才看得见的限制

相对各自首受控tick的首次碰撞时间，B为36.20s（frame3674，原timestamp36.25），C为21.05s（frame7546，原timestamp21.10）。官方记录均为Mercedes coupe，actor id分别11757/15545，接触位置接近；相同seed与车辆类型不证明完全相同世界实现或接触责任。责任仍未知，不能把早接触解释为PI使相同场景必然变差或变好。

| 1773分段 | B | C |
|---|---:|---:|
| 严格碰撞前帧数 | 724 | 421 |
| 碰撞前速度RMS (m/s) | 1.4401 | 1.7726 |
| 碰撞前纵向abs(jerk)p95 (m/s³) | 335.2868 | 47.8003 |
| 最长连续低速实际跨度 (s) | 163.70 | 178.80 |
| 碰撞及之后实际平均速度 (m/s) | 0.00651 | 0.00575 |
| 碰撞及之后模型desired均值 (m/s) | 4.4630 | 3.9524 |
| 碰撞及之后drive命令帧/总帧 | 3208/3276 | 3532/3579 |
| 碰撞及之后模型desired<.4帧 | 0 | 0 |

碰撞后的停滞更像物理阻挡/接触交互相关：模型持续要求前进，selected throttle>0且brake0占绝大多数；两臂applied_control都与上一命令精确吻合。这仅验证API交付，不证明驱动力或可通行空间。B/C的reverse guard68/47帧全部在接触之后，接触前均0；旧静止误guard不能解释这一整段停滞。

B在碰撞前另有7.75s低速段，其中84/156帧desired<.4，体现模型停/慢意图；C没有同样的>=5s碰撞前低速段。网络目标在不同闭环轨迹上变化，不能当作同输入控制器回放。前两对的模型desired变化率|p95|约30.5–38.1m/s²，此量是重规划变化，不等于可执行期望加速度。

全程jerk p95被長停滞稀释，1773 B/C全程为38.50/14.04，却对应更高的碰撞前335.29/47.80；C的全程速度RMS更低也掩盖了碰撞前更高的误差。prefix时长和世界状态不同，只能补充描述，不能替代主指标或当因果配对。

原生导航target仲裁B3854/3999、C3939/3999可用帧；按归档vendor规则重算与angle_final全部一致。平均绝对角减少13.67°/17.84°，说明网络轨迹方向可能被导航目标替代，但尚不能证明该仲裁导致碰撞。模型物理原点未确认，图中actor/−1.4m GNSS两个原点仅作可视化假设。后续横向阶段必须分别定位预测轨迹、仲裁和物理执行，采用预定义turn windows及完整路线回归，不用本报告挑有利片段。

分段采用全部固定5s箱、严格首次碰撞前/后、全部|truth speed|<.5且实际跨度>=5s低速区间；无挑段。见[8000逐帧及89分段](results/paired-v2-stall-final/README.md)。

## 覆盖、成本与复核

六例共9813受控帧；6帧合法初始化neutral，9807次模型forward及9807次原生PID调用，非有限模型/target/speed故障0。GPS/IMU/SPEED/三RGB和truth均同帧，0解析错误、0时间gap、0读取时源变化。无独立网络source-frame字段，只能报告同run_step的传感器帧、输入RGB hash与一次forward/capture链路证据；异步bev不参与模型同步判定。

raw负速度1225帧，其中(-.01,0)为1105帧，reverse guard120帧；原始符号、原因与命令完整保留。两臂normal tick都计算原生和PI分支。selected与native steer最大差<3e-8，selected与选中分支差均<=1e-7；这是CARLA float32存储容差，不是横向算法变化。selected与official tail的油门/刹车差另列，是公共envelope可能造成的有意变化，不混同舍入。

6个attempt均保留，0基础设施失败/重试；两个TickRuntime是驾驶失败。attempt墙钟合计1267.6s，runner分组合计1273.145s；这是实际运行成本，不以提前失败或停滞较轻计算harness加速。17项接入测试由根代理预运行归档；本分析9项测试通过并保存[当次源码与日志](results/analysis-tests-v4/manifest.json)。分段覆盖、来源/输出hash与六例条数另有[最终验证](results/paired-v2-final-audit/verification.json)。

## 可复查交付

- [六例comparison JSON](results/paired-v2-final/comparison.json)、[每例汇总CSV](results/paired-v2-final/cases.csv)，同目录六份完整逐帧CSV与当次分析脚本。
- [最终PNG/PDF与抽样预测CSV](results/paired-v2-figures-final/README.md)：同route行为指标共享纵轴；world等比例共享tight limits，轴交换有标签；首次碰撞红线；无平滑。原始全帧预测不抽样删除。
- [1773双臂分段](results/paired-v2-stall-final/README.md)及vendor源字节。
- [报告/分析/图表/原始来源索引](results/paired-v2-final-audit/manifest.json)，包括固定协议、运行源commit、权重hash和原始attempt路径；raw根`/data/runs/b2d/tcp-controller/paired-v2`。
- [v1发现过程](results/v1-discovery.md)及所有先前analysis/figure editions原样保留。本报告不合并另一个正在进行的转弯阶段结果。
