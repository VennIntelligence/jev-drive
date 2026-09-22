# TCP 横向控制审计与转弯开发方案

日期：2026-09-23。本次只读审计未启动 CARLA、GPU 或 git，也未修改运行源码。

**现有三条 TCP 路线不能说明横向优化价值不大。** 它们的官方中心线几乎都是直线。原生 TCP 还包含 waypoint 与导航目标的仲裁，oracle 的 `tcp` preset 没有完整复现。已有 oracle 转弯案例暴露了全路线均值容易掩盖的误差，支持继续做小范围横向实验，但不能据此宣称真实 TCP 模型已经获益。

## 固定三路线覆盖了什么

直接根据本地固定版本 `bench2drive220.xml` 的位置计算：

| 路线 | 场景 | 长度，m | 到首尾连线的最大距离，m | 平滑后的航向变化范围 |
| --- | --- | ---: | ---: | ---: |
| 24211 | DynamicObjectCrossing | 134.000 | 0 | 0° |
| 1711 | ParkingCutIn | 134.012 | 0.0746 | 1.15° |
| 1773 | ParkedObstacle | 134.005 | 0.0657 | 1.15° |

后两条的微小航向变化受 XML 坐标取整影响。它们能覆盖交互、可能的绕障和转向修正，但没有持续左右转或中心线 S 弯。场景的 crossing angle 描述横穿对象，不能当作 ego 道路转角。仅凭中心线也不能证明模型没有横向绕障，需要保留模型预测和实际运动才能判断。

[旧集成方案](../../2026-09-22-b2d-controller/agents/tcp-controller-integration-plan.md) 保留原生横向，是为了隔离纵向干预，并非横向优化无价值的证据。

## 原生横向不只是 waypoint PID

本地来源：`Bench2DriveZoo/TCP/model.py:286`、`TCP/config.py:35`、`team_code/tcp_b2d_agent.py:204`；完整路径与文件哈希见 [lateral-source-hashes.json](lateral-source-hashes.json)。

- 原始预测是四个 `[前, 右]` 点，时刻为 +0.5、+1.0、+1.5、+2.0 秒。原生 `control_pid` 将坐标交换为 `[右, 前]`；其 metadata 已交换。若转为共同的前/左坐标，应取原始 `[x, -y]`。调用前必须复制原始输出，CPU NumPy 视图可能原地改变输入的坐标顺序。
- 它寻找中点模长最接近 4 m 的相邻点对，然后取该点对的**第一个点**作为 aim，不是取中点。
- 导航目标的绝对角更小时，或者它与最后一段 waypoint 方向的角差超过 `0.3` 且目标前向坐标小于 10 m 时，改用导航目标。角度按 90°归一化，`0.3` 是 27°，不是 0.3 rad；后一条件没有显式要求目标在前方。
- 即使 steer 使用导航目标，`metadata['aim']` 仍保存轨迹选中的点。只画这个 aim 会误判控制意图。
- 原生转向 PID 为 P=.75、I=.75、D=.3、40 样本窗口。I 是窗口均值，D 是相邻样本差，均不是 SI 时间积分/导数。原生输出限制为 ±1，不能与 oracle 控制器的 ±.8 幅值和 2/s 变化率限制混同。
- vendor 最后的处理会在 `abs(steer) > .07` 时改变油门/速度包络。因此真实 TCP 横向改动即使不改纵向代码，也可能改变纵向行为；对照中必须明确保留并记录这一交互。

## 坐标轴可确认，训练标注的物理原点仍不确定

`tools/gen_tcp_data.py:173,192` 读取 annotation 的 x/y，并从 10 Hz 标注每隔 5 帧取四个未来位置。`TCP/data.py:161–187` 减去当前 annotation x/y，再按 compass−π/2 旋转。这能确认相对坐标轴和时间间隔，**不能确认 annotation x/y 对应车身哪个物理点**。

在线导航目标减去 GPS 推导位置，原生 GNSS 安装位置为 x=-1.4 m。可视化 wrapper 的 `prediction_origin=[-1.4,0]` 是绘图假设，不能独立证明训练标签原点。

本地采集代码的证据链不完整：`Bench2Drive/tools/data_collect.py:793` 从 `tick_data['pos']` 写 annotation x/y；第 699 行 `_get_position` 会变换 GPS。但本地 `tick` 只返回 GPS，没有 pos；已搜索的本地 Python 树也未找到生成 pos 或将该 helper 接到标注的调用。`docs/anno.md` 只写世界坐标位置，并将来源说明留为 TODO。仓库 `docs/carla.md:293` 说明训练数据集未下载。当前没有与该 checkpoint 对应的数据/采集调用链，能排除 actor origin、GNSS 或其他锚点。

因此保持物理原点**未确认**，不能依据 preview 原点给真实模型点新增后轴平移。后续需要真实训练 annotation 及 ego transform/传感器外参，或者完整采集链和数据版本证据。下面的 oracle 转弯实验不需要引入这个假设。

## 预先固定的转弯窗口与日志帧

[lateral_windows.py](lateral_windows.py) 只读取已保存的原始路线几何和 telemetry。[v1 manifest](lateral-evidence-v1/manifest.json) 保存输入哈希和协议，不用跟踪误差选择片段：

1. 原始 world XY 按弧长 0.5 m 重采样，用中心 5 m 弦方向计算展开航向，端点截断；曲率为航向对弧长的导数。
2. 以绝对曲率 ≥.02/m 为种子，合并间隔 ≤3 m 的组，保留累计绝对转角 ≥15° 的组，前后各增加 5 m。
3. 正负累计转角分别 ≥15° 标为 S，否则按符号标左右；净转角 ≥60° 或峰值曲率 ≥.08/m 标为急弯。CARLA world XY 航向正变化对应右转。
4. 独立投影记录的 truth 后轴位置，保留窗口内全部帧，包括反向、停滞；`speed >= 2 m/s` 只作为辅助统计。逐段记录连续帧访问，避免把缺帧误写成连续区间。

这些是较高曲率的核心窗口，不是所有道路转角的完整分割。例如 24240 全路线航向范围 89.9°，核心只累计 49.2°；Dev10 26405 有 46.2° 的宽缓弯，却没有触发核心阈值。没有核心窗口**不等于直线**。

| 数据/路线 | 核心弧长，m | 加边界后的窗口，m | 类型 | 对应日志帧示例 |
| --- | --- | --- | --- | --- |
| G2 26966 | 29–46 | 24–51 | 右急弯，+88.4° | max-PP 8315–8384 |
| G2 17563 | 32.5–43.5 | 27.5–48.5 | S，累计绝对转角 68.0° | max-PP 5399–5468 |
| G2 17563 | 78.5–90 | 73.5–95 | S，累计绝对转角 68.8° | max-PP 5552–5622 |
| G2 24240 | 23–59 | 18–64 | 左弯核心，−49.2° | max-PP 6947–7063 |
| G2 25854 | 3–34.5 | 0–39.5 | 右弯核心，+48.6° | max-PP 9808–9947 |
| Dev10 17569 | 37.5–48.5；83.5–95 | 32.5–53.5；78.5–100 | 两个 S 窗口 | pursuit seed0 14566–14618；14681–14734 |
| Dev10 2091 | 24–46 | 19–51 | 左急弯 | pursuit seed0 9818–13705 |
| Dev10 27494 | 31.5–49.5 | 26.5–54.5 | 左急弯 | pursuit seed0 13915–14188 |
| Dev10 28198 | 28–54.5 | 23–59.5 | 左急弯 | pursuit seed0 14918–15012 |

帧号只属于对应运行，不能跨运行直接复用。[帧 CSV](lateral-evidence-v1/frame-windows.csv) 包含 G2 三配置及准确路径。Dev10 v1 仅索引 pursuit seed0 attempt1，不是结果筛选对照；v2 才按各组第一个 harness-finished attempt 比较，并保留全部尝试索引。

## 转弯统计揭示的问题

[v2 逐窗口结果](lateral-evidence-v2/results-v1/per-window.csv) 包含 36 cases、45 windows、90 条全帧/移动辅助指标。[恢复表](lateral-evidence-v2/results-v1/recovery.csv) 和[口径](lateral-evidence-v2/README.md) 保留进出窗口行为以及缺失/删失观测。

G2 使用同一 oracle、adapter、PI 纵向配置，无交互交通；**不是实际 TCP 模型测试**：

| 窗口 | CARLA-PI CTE RMS / p95，m | Additive-PP RMS / p95 | Max-PP RMS / p95 |
| --- | ---: | ---: | ---: |
| 26966 右弯，8 m/s | .520 / .913 | .727 / 1.212 | .559 / .849 |
| 17563 S #1，6 m/s | .377 / .712 | .368 / .822 | .222 / .461 |
| 17563 S #2，6 m/s | .346 / .665 | .375 / .848 | .270 / .473 |
| 24240 左弯，8 m/s | .213 / .277 | .139 / .211 | .108 / .180 |

Max-PP 比 additive PP 的右弯峰值更低，但该弯 CARLA-PI 的 RMS 仍更低。三个配置在这些窗口都未触发 ±.8 幅值饱和。26966 两种 PP 都未触发变化率限制，因此提高限制不是有证据支持的修复。Max-PP 的 S 弯变化率 p95 达到 2/s，存在另一项精度/动作变化权衡。26966 出窗口后 10 m 的 CTE RMS，CARLA/additive/max 分别为 .274/.601/.317 m；两种 PP 在该观测范围内都未满足诊断用持续恢复条件。

正式对照保留停滞：pursuit seed0 的 2091 窗口共 3,888 帧，仅 87 帧速度 ≥2 m/s；全窗口 CTE RMS 为 1.327 m，移动辅助为 .712 m。不能单独拿其中一个数描述转弯成功。正式各配置的纵向模式和交互 actor realization 不同，结果只能描述运行，不能隔离横向因果。正式 `tcp` 组也是 oracle preset，没有原生模型完整目标仲裁。

## 下一轮最小可执行对照

真实 TCP 的六例纵向 B/C 单独进行。横向开发采用 **26966 右急弯 8 m/s、17563 两个 S 窗口 6 m/s、24240 左弯 8 m/s**，不放 NPC；共三路线、四窗口、两个配置、六 case。固定原始几何、初始位姿偏差、adapter、PI、时间调度、停止行为和车辆标定，同时保留全路线回归结果。

根代理实际选择的候选已收窄为：

- 基线：`Ld = max(3 m, .5 s × speed)`。
- 候选：`Ld = max(3 m, .375 s × speed)`。

**只改时间系数，不改 3 m 下限，不提高 steer_rate。** 先前独立审查提出的 `max(2.5 m, .4 s × speed)` 是被收窄的设计建议，不是执行参数。实际比较在 8 m/s 时把 lookahead 从 4 m 缩到 3 m；6 m/s 时两者都为 3 m，S 弯主要检验不应被改变的行为及微小速度偏移带来的影响。若瞬时速度超过 6 m/s，原基线可能略高于 3 m，不能宣称两组全时刻数值完全一致。

候选是关于弯道跟踪的有限假设，不是已证明的收益。关注 26966 出弯误差与 S 弯变化率的共同表现，不因某个均值更低就接受其他窗口退化。目前证据也没有定位出应优先修改的某个 CARLA PID 增益。

开跑前固定参数和四个原窗口，比较 CTE RMS/p95/max、航向误差、进出窗口、转向峰值/变化率/限幅占比，以及已有采样支持的横向加速度和 jerk；保留全路线完成、停滞、终点保持和直线行为。峰值改善但恢复变差仍是权衡。本审计不新增数值验收门槛。

方向口径需明确：validator 的 CTE 为左正；26966 右弯的正残差对应弯外侧，不支持“内侧切弯”的解释。候选检验的是弯外残差和出弯跟踪滞后。旧 v2 独立计算的叉积符号为右正，但主表只有 RMS/绝对 p95/max，不受符号影响；封存口径不回写。新工具显式采用左正 CTE，同时保留有符号均值、峰值与帧号。

新分析工具 [analyze_turns.py](analyze_turns.py) 接受独立 `--run-root`，读取 `<route>/<baseline-max|short-max>/pursuit`，默认使用已封存的四个几何窗口，绝不按新运行误差重新挑片段。预定 rawroot 为 `/data/runs/b2d/controller/turns-v1`。

## 真实 TCP 横向替换前应记录的诊断

记录来源 frame/time、推理时间与年龄、不可变 raw 4×2 点及坐标轴/时间/原点状态、原生交换后的点、导航目标和 command；记录选中点对/中点模长、轨迹 aim、两个仲裁条件、选择来源和**最终生效 aim**。四种角度同时标清 90°归一化值与度数；记录 preclip steer、P/I/D 分量与窗口状态、clip 后 steer、最终 applied steer、包络分支，区分 `only_traj`、融合、直接控制模式。

统计幅值饱和和持续时间、实际 dt 转向变化率、目标仲裁切换、航向滞后/符号变化，以及完整入弯/弯中/出弯轨迹。预测帧局部跟踪与世界路线跟踪分开，truth 只进 logger；保留模型曲率、距离、速度、有符号误差和失败。模型主动绕开中心线时，更低的中心线误差不能证明绕障更好。

复算须用全新目录，已有输出目录会被拒绝：

```sh
python3 todos/2026-09-23-tcp-controller/agents/lateral_windows.py --out /fresh/path/lateral-windows
python3 todos/2026-09-23-tcp-controller/agents/lateral-evidence-v2/analyze.py --out /fresh/path/lateral-metrics
python3 todos/2026-09-23-tcp-controller/agents/analyze_turns.py --run-root /data/runs/b2d/controller/turns-v1 --out /fresh/path/turns-metrics
```

唯一非标准依赖是 NumPy。直线和镜像 90° 几何检查通过，已有原始/输出哈希已保存并复核；大型源日志留在 `/data/runs`。
