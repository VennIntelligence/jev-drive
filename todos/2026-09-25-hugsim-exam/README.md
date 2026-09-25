# HUGSIM zero-shot 考试：Alpamayo 1.5 与 openpilot（Cinque v3 主、Lebowski 次）

状态: 预注册（2026-09-25，写于任何计分运行之前）；rate study 与 openpilot 适配清单排队中，Alpamayo 部分等 GPU 2
接口: [docs/hugsim.md](../../docs/hugsim.md)；清单: [docs/zeroshot-adapters.md](../../docs/zeroshot-adapters.md)
教训来源: [openpilot 迁移](../2026-09-24-zeroshot-exam/openpilot-migration.md)、[Alpamayo 闭环诊断](../2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md)
代码: `jevdrive/hugsim_zs.py`（几何）、`scripts/hugsim/zs_agent.py`（每场景的 agent 进程）、`scripts/hugsim_zs_server.py`（常驻模型）、
`scripts/hugsim/zs_run.py`（批量跑官方 `closed_loop.py`）、`scripts/hugsim/zs_exam.sh`（各阶段）、`scripts/hugsim/zs_rate_{op,alp}.py`、
`scripts/hugsim/zs_collect.py`、`scripts/hugsim/zs_figs.py`

## 要回答什么

两个不是为 HUGSIM 训练的模型，接上 HUGSIM（3DGS 重建的闭环仿真器：4 Hz、每步 0.25 s，最多 400 步，6 路 nuScenes 式 800×450
相机，plan 是前相机原点、x 右 y 前、0.5 s 间隔的 waypoint，经 iLQR 跟踪 + kinematic bicycle 执行）之后，HD-Score（每步
PDMS 的均值 × RC，按**计划轨迹**打分）是多少，和官方基线 UniAD / VAD / LTF 比如何，失败在哪。前提是先证明分数不是适配 bug 造成的：
B2D 考试里两个模型的闭环失败都是适配 bug（plan 原点、倒车 “stop” plan、帧时序），所以计分之前必须过清单。

## 1. 4 Hz 帧率的代价与帧合成（预注册，写于看 rate study 结果之前）

HUGSIM 只在 0.25 s 的步上渲染；openpilot 要 20 Hz 帧（5 Hz context），Alpamayo 要 4 帧 @10 Hz。仿真器之外不能补帧，
只能决定怎么把 4 Hz 帧喂给模型。离线在真实数据上量代价，同一帧时刻、只改时间轴：

| 模型 | 数据 | 变体 | 含义 |
|---|---|---|---|
| openpilot | comma1M 8 段（rig study 的 native model frame） | `native` | 真车：每 20 Hz 帧都喂，读 5k 帧的 plan |
| | | `ctx5-hold` | 5 Hz context 帧各保持 4 步（检查：应≈native） |
| | | `h4-dilate` | **当前适配**：0.25 s 一帧当作一个 0.2 s context 步喂，模型时钟快 1.25×，plan 时刻 τ 读成真实 1.25τ |
| | | `h4-hold` | 20 Hz 时钟照常，4 Hz 帧保持 5 步（只对 Cinque 的 20 Hz 队列有意义） |
| Alpamayo | nuScenes 400 个样本（nuScenes 考试的输入） | `native` | 4 帧 @10 Hz、20 Hz 位姿的 egomotion |
| | | `h4-hold` | **当前适配**：每个 10 Hz 槽取最近的 4 Hz 帧（→ t0−0.25, −0.25, 0, 0），4 Hz 位姿线性插值、只有 yaw |
| | | `h4-spread` | 最近 4 张 4 Hz 帧当作 10 Hz 喂 |

指标：openpilot 在 1/2/4 s 真实时刻的横向 / 纵向误差（对 localizer 的未来轨迹），按 100 帧的 block 做 cluster bootstrap；
Alpamayo 的 L2@1/2/3 s，按 scene cluster bootstrap，所有变体配对。

**选择规则（先定死）：**
- openpilot：默认 `h4-dilate`（opt `op_clock: dilate`）。只有 Cinque 的 `h4-hold` 在 2 s 横向**和**纵向误差上都比 `h4-dilate`
  低、且两者的配对差 95% CI 都不含 0 时，Cinque 改用 `h4-hold`（`op_clock: hold`，已实现）。Lebowski 只能 dilate（它的
  context 步就是 0.2 s）。
- Alpamayo：默认 `h4-hold`。只有 `h4-spread` 的 L2@3s 比它低且配对差 CI 不含 0 时改用 `h4-spread`（需要实现，届时记偏离）。
- 不管选哪个，4 Hz 相对 native 的代价（相对误差 + CI）作为结果报告，它是 HUGSIM 分数的已知 handicap。

## 2. 适配层（运行前定死，除非清单查出 bug）

| 项 | Alpamayo 1.5 | openpilot |
|---|---|---|
| 相机 | 4 个 f-theta 虚拟视图（576×320），旋转重投影自 6 路；Waymo / KITTI-360 的后 3 路是全零图，不用 | road / wide model frame（512×256，calib = 前相机水平正前），来自 FRONT / FRONT_LEFT / FRONT_RIGHT，BT.601 limited YUV |
| 时间 | 见第 1 节 | 见第 1 节；第一步先用首帧预热 5 s 模型时间 |
| 历史 | 16 × 0.1 s 后轴 egomotion，4 Hz 位姿线性插值；episode 开始前按起始速度与朝向倒推 | 模型自己的隐状态 |
| 路线信息 | `command`（0 右 1 左 2 直）→ nav 文本 "Turn right / Turn left / Continue straight" | → desire turnRight 2 / turnLeft 1 / none 0，持续给 |
| 交通规则 | — | nuScenes 新加坡场景 left-hand `traffic=[0,1]`，其余 right-hand |
| 输出 → plan | 后轴轨迹 → 前相机刚体变换 cam(t) = rear(t) + R(ψ_t)(d,0) − (d,0)，d = 1.73 m；插值到 0.5…3.0 s | calib frame 已在相机；按 1.25 的时钟读 0.5…3.0 s |
| 倒车 | **forward-only**（`jevdrive.hugsim_zs.forward_only`）：从原点沿 plan 走，向后的段置零再累加。“停车” 的倒车 plan 变成零长度 plan，iLQR 不会去倒车追它 | 同左 |
| 规划节奏 | 每步（4 Hz）重规划 | 每步 |
| 随机性 | 每步 seed = crc32(scenario) + step | 确定性 |

## 3. 适配验收清单（HUGSIM 版，计分前必须过）

[docs/zeroshot-adapters.md](../../docs/zeroshot-adapters.md) 是 CARLA 版；HUGSIM 没有红绿灯、没有停车再起步，第 8/9 项换成 “前车静止时停车”。
场景：`checklist.txt` 的 8 个（nuScenes 0071 直行、0383 左转、0920 右转、0062-medium 前方 30 m 静止车、0383-hard 两个 actor；
Waymo 右转、KITTI-360 左转、PandaSet 直行）+ 两个派生场景（0071 从静止起步；0071 起点右移 0.8 m、右偏 6°）。这些 scene
都不进计分样本。每个模型 × 两个控制器（official、fixed = PR #57 heading fix），另加 **shadow mode**（route follower 以 ≤ 5 m/s
开车，模型每步照常规划、只记录）、去掉路线信息（desire / nav 关）的转弯对照、以及起步场景上的 engage-while-rolling。

| # | 检查 | 怎么查 | 通过 |
|---|---|---|---|
| 1 | 坐标与航向 | shadow mode：模型 plan 对 oracle 实际未来轨迹的 1/2/3 s 横纵误差；plan 投影到前相机图 | 左右、前后符号一致；误差量级与模型在真实数据上的开环误差相当（Cinque comma1M 2 s 横向约 0.2 m；nuScenes 上 Alpamayo、Cinque 的 L2@2s 约 0.9 m），不是系统性偏一侧；plan 落在路面上 |
| 2 | 参考点与倒车 | 所有 run：plan 的 t=0 就是 ego；统计 forward-only 改动了多少 plan；起步场景 | 模型说停时 ego 不倒车（`ego_velo` 不小于 −0.1 m/s）、不被拖着走 |
| 3 | 传感器时序 | 6 路同一步渲染（仿真器保证）；上下文间隔按第 1 节 | 日志里 Alpamayo 槽时刻、openpilot reps 与预注册一致 |
| 4 | 预热 | 前 1 s 的输出 | openpilot 首步 plan 不是冷启动的 80 m 长 plan；Alpamayo 首步 plan 速度与 1 m/s 起始速度相符 |
| 5 | 静止起步 | 派生的 start_velo = 0 场景 | 10 s 内离开起点；否则按下面的规则用 engaged |
| 6 | 车道保持 | 0071、PandaSet 021 的直行段（fixed 控制器） | 到路线终点或 10 s 以上无 “Far from preset trajectory”、|横向偏移| 中位数 < 1 m |
| 7 | 转弯左右 | 0383 / KITTI-360 左转，0920 / Waymo 右转；有、无路线信息各一次 | 至少在有路线信息时，转弯方向与 command 一致（进弯后 2 s 内 ego 航向变化的符号）；左右映射错会表现为 “无路线信息更好” |
| 8 | 前车静止 | 0062-medium（30 m 外静止车） | 停住或绕开不撞；撞了要能归因到模型输出（plan 本身穿过前车），而不是控制器 |
| 9 | 偏置起点 | 派生 offset 场景（右 0.8 m、右偏 6°） | plan 往左修正（符号检查） |
| 10 | 控制器 | `zs_run.py` 每个 job 前核对两棵树的补丁状态 | official = 上游 + 我们的非行为补丁；fixed = + PR #57 |

**官方控制器的已知缺陷**（heading 转置，docs/hugsim.md）会让直行 plan 向右漂；debug 跑里 openpilot 在 official 下原地打转
（转向 0.6 rad），这是控制器 × 模型的相互作用，不是适配 bug 的证据；清单的 1、6、7、9 项按 fixed 控制器判，official 只记录。

**由清单决定、但规则先定死的两件事：**
- engage-while-rolling：所有计分场景都以 1 m/s 起步。若 fixed 控制器下，模型在 0071 / PandaSet 021 两个直行场景上起步 10 s
  后车速都 < 2 m/s（模型自己不走），计分用 `engage_s = 5`（前 5 s 由 route follower 以 ≤ 3 m/s 开），分数标 **engaged**；
  否则不用。两个模型分别判。
- 某项不过且查出是适配 bug：修复、记偏离、重跑整个清单，再计分。查不出 bug、失败来自模型输出：写明，照常计分。

## 4. 计分运行（预注册）

**场景**：`scored.txt`，64 个 = 4 数据集 × 4 难度（easy / medium / hard / extreme）× 每格 4 个，seed 0 分层随机
（`scripts/hugsim/zs_sample.py --k 4`）；排除需要 HD map 的 16 个 nuScenes 场景（trajdata map cache 没配，记偏离）和清单用过的 scene。
64 个场景来自 45 个 scene（同一 scene 不同难度），所以 CI 按 scene 做 cluster bootstrap。若清单阶段量出的单 job 时长让计分超过
约 10 h，就按抽样顺序降到每格 3 个（48），只看时长，不看分数。

**被测**（每个都在 official 与 fixed 两个控制器下各跑一遍，配对）：Alpamayo 1.5、openpilot Cinque v3（主）、Lebowski（次）；
基线在同 64 个场景上测：官方 LTF client、constant velocity（cv，直行 ≥1 m/s）。
**主结果 = official 控制器**（与已发表数字可比）；fixed 是配对的次结果（DrivoR 报告 fix 后 +8.5 HD-Score）。

**外部参照**（不同样本，只作量级参照）：我们的 LTF smoke（4 个 nuScenes 场景）0.500 official / 0.557 fixed；
HUGSIM 论文 Table 13（全部场景、旧控制器）按 16 格等权平均：UniAD 0.299、LTF 0.264、VAD 0.132。我们的 64 个场景每格 4 个，
场景均值就是 16 格等权平均，口径一致。

**指标**：HD-Score（headline）、NC、DAC、TTC、C、RC、PDMS；按数据集 × 难度拆分；95% CI 为按 scene 的 cluster bootstrap
（10 000 次）；模型对 LTF、fixed 对 official 的差用配对 cluster bootstrap。结束原因（背景碰撞、前景碰撞、偏离路线 > 10 m、
完成、400 步）逐场景统计。**失败分析**：每个模型的 fixed 控制器 run 里，按结束原因分组看 ego 轨迹对路线的横向偏移、
速度、plan 与 command，挑典型场景出图。
**基础设施**：一个 job 崩溃（非模型结束）重跑一次；每个 tag 崩溃 ≤ 3 个时剩下的按缺失报、不补分；超过就停下查。

## 5. 资源与排程

GPU 1 现在 ≤ 15 GB（与 SimLingo、openpilot 闭环 agent 共用），GPU 2 在 `alp-b2d-full.done` 之后 ≤ 60 GB；≤ 8 核。
openpilot server（Cinque TRT session 约 2.3 GB / 连接，Lebowski 约 1.5 GB）+ 每个仿真器 2.4–6 GB。Alpamayo（~25–30 GB）只能等 GPU 2。

| slot | GPU | 内容 |
|---|---|---|
| `hugsim-rate-op` | 1 | openpilot rate study（~30 min） |
| `hugsim-check-op` | 1 | openpilot 清单 + route / cv 参照（2 条 lane） |
| `hugsim-rate-alp`, `hugsim-check-alp` | 2 | 等 `alp-b2d-full` |
| `hugsim-scored-*` | 1 / 2 | 清单通过、本文件提交之后 |

## 结果

### R1. openpilot 的 4 Hz 代价（2026-09-25 11:10，`rate_op.jsonl`）

comma1M 8 段，每段 5k 帧（k ≥ 20）上的 plan，对 localizer 真值；误差是绝对值均值，括号是对 native 的配对差按 100 帧 block
bootstrap 的 95% CI。

| 模型 | 变体 | n | 横向 1 / 2 / 4 s (m) | 纵向 1 / 2 / 4 s (m) | 横向 2 s 相对 native | 纵向 2 s 相对 native |
|---|---|---:|---|---|---|---|
| Cinque | native | 1760 | 0.052 / 0.163 / 0.711 | 0.32 / 0.85 / 2.55 | — | — |
| Cinque | ctx5-hold | 440 | 0.052 / 0.159 / 0.701 | 0.32 / 0.85 / 2.54 | 0（逐位相同） | 0 |
| Cinque | **h4-dilate** | 1760 | 0.064 / 0.203 / 0.880 | 0.38 / 0.97 / 2.80 | **+25%** [+0.019, +0.063] m | **+14%** [+0.05, +0.19] m |
| Cinque | h4-hold | 1760 | 0.353 / 1.189 / 4.134 | 9.8 / 19.3 / 36.8 | +630% | +2160% |
| Lebowski | native | 1760 | 0.053 / 0.163 / 0.752 | 0.33 / 0.86 / 2.62 | — | — |
| Lebowski | **h4-dilate** | 1760 | 0.072 / 0.225 / 0.935 | 0.38 / 0.96 / 2.79 | **+38%** [+0.033, +0.094] m | **+11%** [+0.00, +0.19] m |
| Lebowski | h4-hold | 1760 | 0.309 / 1.047 / 3.789 | 11.8 / 23.2 / 44.1 | +541% | +2604% |

读法：`ctx5-hold` 与 native 相同，说明 5 Hz context 相位的处理是对的。把 4 Hz 帧当 0.2 s context 步喂（dilate）只多出
横向 25–38%、纵向 11–14% 的误差，而 “20 Hz 时钟照常、帧保持” 让模型把 0.25 s 的运动当成静止 + 跳变，纵向误差到 19–23 m，
不可用。按第 1 节规则（hold 须两项都更好）**保持 `h4-dilate`**。这个 +25–38% 横向误差是 openpilot 在 HUGSIM 上的已知
handicap。

### R2. openpilot 适配清单（2026-09-25 13:30；`check-op` + 修复后重跑的 `check-op2`）

逐 run 表：[checklist_runs.csv](../../research/results/hugsim-exam/checklist_runs.csv)；shadow mode 误差：
[checklist_shadow.csv](../../research/results/hugsim-exam/checklist_shadow.csv)。

**查出并修掉的适配 bug（偏离 1，写于任何计分之前）：停车 plan 让 iLQR 倒车打转。** 第一轮里 Lebowski 在 PandaSet 021 停住后
（plan 3 s 终点 < 0.1 m，forward-only 已经起作用），`ego_velo` 掉到 −0.91 m/s、方向盘从 0.08 打到 0.88 rad：iLQR 用相邻 waypoint
的差算参考航向，毫米级的横向噪声让航向在 ±180° 乱跳，于是原地倒车、转向。修复 `jevdrive.hugsim_zs.straight_stop`：3 s 终点
离 ego < 1 m 的 plan 视为停车，横向分量置零、纵向进度保留。纯语义修复（停车就是停在原地），不看分数；修复后所有模型 run
的最低车速 ≥ 0（原来 −0.91）。按预注册的规则重跑了全部模型驱动的清单 run（`check-op2`），shadow 与参照 run 不受影响、没重跑。

| # | 检查 | 结果 | 判定 |
|---|---|---|---|
| 1 | 坐标与航向（shadow，oracle ≤ 5 m/s 开，n = 514–594 次规划） | 横向误差 1 / 2 / 3 s：Cinque 0.11 / 0.25 / 0.46 m，Lebowski 0.11 / 0.24 / 0.48 m（直行外推 0.13 / 0.60 / 1.40 m）；横向偏差 −0.02 ~ −0.07 m；真值横移 > 1 m 时 3 s 横向符号一致 98%；纵向 0.79 / 0.94 m @2s，偏短 0.15–0.18 m | 过：误差与 comma1M 上的开环误差同量级（2 s 横向 0.20 m，4 Hz 代价之后），没有系统性偏置 |
| 2 | 参考点与倒车 | 修复前 1 例倒车 −0.91 m/s；修复后 0 例 | 修复后过 |
| 3 | 时序 | reps：Cinque 首步 100、之后 4；Lebowski 首步 25、之后 1；与第 1 节一致 | 过 |
| 4 | 预热 | 首步 plan 3 s 终点 0.03–0.07 m（预热 5 s 喂的是同一帧，模型认为车停着），第二步起正常（4–7 m）；起步时车速从 1 m/s 先掉到约 0.7 m/s 再加速 | 过（记为已知现象：静止帧预热与 1 m/s 起始速度不一致） |
| 5 | 静止起步（fixed） | 两个模型都**一直不动**（400 步，车速 < 0.01 m/s）；engage 5 s 后 Cinque 能开，但在无车道线的停车场里向左打转 | 模型行为：与 B2D 一致，openpilot 从全停不自己起步 |
| 6 | 车道保持（fixed） | 0071（无车道线的货场）：Cinque 13 s 后右拐冲向停着的卡车（plan 投影在图上是一条连贯的右转路径），Lebowski 开头绕了 3 圈之后沿路线开完；PandaSet 021：两者都在 3–4 s 后自己把 plan 缩到 0 并停住，之后不再起步（lead_prob < 0.1，前方无车） | 失败来自模型输出（停车、选错方向），不是适配；闭环打转见下 |
| 7 | 转弯（fixed，有 / 无路线信息） | 左转（0383、KITTI-360）：两模型航向变化都向左（Cinque −45° / −49°，Lebowski −83° / −56°）；右转（0920、Waymo）：Lebowski +115° / +50°，Cinque +20°（冲出路线）/ 停住。desire 的作用方向对：Cinque 在 0383 无 desire 只转 −6°，有 desire −45°；comma1M 离线 probe 里 turnRight 让 4 s 横向 +1.1 m（Cinque）/ +0.7 m（Lebowski） | 映射正确；Cinque 的右转执行不了是模型行为 |
| 8 | 前车静止（0062） | Lebowski 停在车后不撞（400 步）；Cinque 向左绕开后偏离路线 | 过（没有穿过前车的 plan 被执行成碰撞） |
| 9 | 偏置起点（右 0.8 m、右偏 6°） | 两模型前 2 s 的 plan 都向左修正（3 s 横向 −0.20 / −0.21 m）；Cinque 开完 HD 0.96 | 过 |
| 10 | 控制器 | 每个 job 前核对补丁状态，全部一致 | 过 |

**official 控制器**：10 个清单场景上两个 openpilot 模型 20/20 都在 9–50 步内背景碰撞（HD 0.00–0.12，只有 Lebowski 在
PandaSet 021 开完），轨迹是原地打转：heading 转置让直行 plan 向右拐，openpilot 从画面里看到自己在转，下一次 plan 跟着转。
同一批场景上 route follower 在 official 下 7/10 完成，所以这是 “控制器缺陷 × 会外推自身 yaw 的模型” 的相互作用，按预注册
official 仍是 headline，fixed 是配对次结果。

**fixed 控制器下仍有的闭环打转**（Lebowski 0071 开头 3 圈、Cinque engage 后 14 圈）只出现在没有车道线的货场里：模型把刚执行出的
yaw 外推成更大的弯。推测（未验证）：HUGSIM 的 bicycle 把前相机当后轴积分，转弯时相机没有真车前相机那 ω·1.7 m 的横向速度，
模型看到的运动与训练分布不同。验证办法是在仿真器里把渲染位姿前移 1.7 m 做对照，这超出适配层，不做。

**决定（按第 3 节预注册规则）**：没有待修的适配 bug，计分照常进行。engage 规则的触发条件（0071 与 021 起步 10 s 后都 < 2 m/s）
不成立（0071 上 5.5 / 3.9 m/s），**计分不用 engage**；两个模型的静止起步失败与 021 的自停作为模型行为报告。
