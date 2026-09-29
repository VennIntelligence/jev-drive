# CARLA 就地回退（rewind）分叉：能不能代替从头重跑，省多少（2026-09-29 登记）

状态: done 2026-09-29（判格见执行记录最后一条：回退不等价，WL-2 从头 + zygote）
主题: [research/carla-rewind-branching.md](../research/carla-rewind-branching.md)（外部提案 + POC）；WL 分叉 [todos/2026-09-28-wm-loop.md](2026-09-28-wm-loop.md)
调度: `runs/sched/table.tsv` 的 `carla-rewind` 行（GPU 5，≤ 3 个 CARLA，index 440–442，核 184–207，与 wm-loop 训练共卡）；预算 ≤ 6 GPU·h、约 1 天。

## 要回答的问题

提案说：到分叉 tick 在 Python 内存里记下 ego 与附近 actor 的 transform + 速度，跑完一个 3 s 分支后 `set_transform` + `set_target_velocity` 退回去，
在同一个世界里接着跑下一个分支，每次回退 61 ms，WL 吞吐 6.7×（170.5 → 25.4 worker·h）。POC 在 Town10 上只有 3 辆 NPC、没有 scenario。
我们的分叉在 Bench2Drive scenario 里面：hazard 行人由 scenario_runner 的行为树驱动（trigger 条件看 ego、`KeepVelocity` 累计走过的距离、走完 `ActorDestroy`），
背景车由 BackgroundActivity + Traffic Manager（TM：CARLA 的背景车控制器，内部有路径缓冲和随机数）开，还有红绿灯计时、车辆内部状态（轮速、档位、悬挂）。
只恢复 transform / 速度，分支可能从一个与「从头重跑」不同的世界出发。这里用 WL 已有的从头生成分支当真值，量回退分支与它差多少，以及真实省多少。

名词（本文第一次出现处）：
- **回退（rewind）**：同一个 CARLA 世界、同一次 route 里，把状态退回分叉 tick 再跑下一个分支，不重新装图、不重跑前缀。
- **死 tick（dead tick）**：执行回退的那一个 tick：在 agent 调用里把世界、行为树、GameTime 设回分叉前一 tick 的状态，返回当时 expert 的控制；物理从回退后的状态推进一步，下一 tick 起就是分支的第 0 tick。
- **floor**：从头重跑同一个分支两次之间的差（CARLA 与渲染本身的不确定），所有误差都拿它当尺子。

## 已有的成本事实（不是保真度数字，登记前从全量 WL 的 done 记录读出，box `runs/rewind/wl_cost.csv`）

提案第 1、4 节的「每个 run 192 s 里 160 s 是前缀」是估算，与实测不符。全量 WL 2 814 个分支 run（BA 2 254、P6 560）：

| 量（中位 / 均值） | BA | P6 |
|:--|--:|--:|
| 单 run 墙钟 wall_s | 180 / 181 s | 116 / 115 s |
| tick 数（前缀 + 3 s 分支 + 20 s 续跑） | 497 / 473 | 348 / 390 |
| 分叉 tick（前缀长度） | 97 / 110（约 5 s 模拟时间） | 113 / 108 |
| 前缀占 tick 的比例 | 20% / 24% | 29% / 30% |
| 每 tick 墙钟（server + agent + 树） | 171 / 214 ms | 136 / 145 ms |
| 非 tick 部分（起 client、装图、建 scenario、传感器、收尾）= wall − ticks × tick | 94 / 84 s | 41 / 58 s |

前缀全程渲染（`NORENDER=0`，wm-loop 第 1 级 no_rendering 不过），但前缀只有约 5 s 模拟时间、约 20–25 s 墙钟；大头是装载（约 84 s）和分支后的 20 s 续跑（约 400 tick、约 70 s）。
续跑是 WL 的 D2 训练窗口（expert + 随机动作），它跟在每个分支之后、依赖该分支的结果，回退省不掉它。所以回退能省的上限是「装载 + 前缀」的 6/7，不是 83%。
按上表粗算（BA）：从头 7 × 181 ≈ 1 270 s；回退（保留续跑，只省装载与前缀）≈ 7 × 181 − 6 × (84 + 22) ≈ 630 s，约 2.0×；
不要续跑时从头 7 × (84 + 22 + 13) ≈ 830 s 对回退 84 + 22 + 7 × 13 ≈ 200 s，约 4×（这些是登记前的预估，下面按实测改）。另：170.5 worker·h 里含失败重试（rc139 装图崩溃约 7%），回退少装图也少崩。

## 回退方法（同一批分叉点上都跑，比较）

共同部分（全部方法）：快照点 = 分叉前一 tick（k−1）的 agent 调用末尾；回退在分支结束后下一个 agent 调用里做（死 tick），返回快照里的 expert 控制 c_{k−1}，下一 tick 即虚拟 tick k，开下一个动作的窗口。
P7 控制器（纯 Python）在快照时 deepcopy、回退时换回；ego `set_transform` + `set_target_velocity` + `set_target_angular_velocity`。分支只跑 3 s（61 tick），**不跑续跑**（续跑依赖分支，没法共享）。

- **a `poc`**：提案原样——ego + 50 m 内的车辆 transform / 线速度 / 角速度；不管行人、行为树、时间、红绿灯。
- **a+ `teleport`**：所有车辆和行人（不限距离）transform / 速度，行人再 `apply_control` 回快照时的 WalkerControl；红绿灯 `set_state` 回快照状态（计时无法经 API 设回，记为已知偏差）；不管行为树与时间。
- **d `tree`**：`teleport` + 行为树与 scenario 侧的 Python 状态：scenario 树每个节点的 `__dict__`（status、`current_index` / `current_child`、`KeepVelocity` 的 `_distance` / `_location` / `_start_time`、trigger 状态、BackgroundActivity 的内部表，含其中引用的 srunner / leaderboard / agents 对象，逐层拷贝；carla actor / world 引用保持原对象）、
  py_trees blackboard、`CarlaDataProvider` 的位置 / 速度 / transform 缓存与 `_rng` 状态、`GameTime._current_game_time`（退回 t_{k−1}，frame 号照常单调）。这是「hazard 行人的脚本行为正确续上」需要的最少集合。
- **b `respawn`**：`tree`，但背景车辆（非 hero、非 scenario actor）销毁后按快照的 blueprint / 颜色 / transform / 速度重生，重新交给 TM autopilot，并把行为树、CarlaDataProvider 里对旧 actor 对象的引用换成新对象（`ignore_lights` 等逐车 TM 参数不恢复）。做不通（崩溃、BackgroundActivity 报错）就记为不可行并写原因。
- **c `replay`**：前缀里开 CARLA recorder，回退时 `replay_file` 定位全部 actor（含红绿灯计时，这是 Python API 做不到的）→ `stop_replayer(keep_actors=True)` → 注入快照速度 + `tree` 的 Python 状态。
  在 leaderboard 的 tick 循环里要多占 tick、要求 replayer 复用同 id 的 actor；限时 2 h 做不通就记为不可行并写原因。

## 分叉点（选择规则写在读任何回退数字之前；来源 box `runs/rewind/candidates.csv`，已去掉位姿剔除组与成对读数退出组）

规则：每类里在 7 个分支的 unsafe 标签不全相同（1–6 个 unsafe）的分叉点中，按分叉时 50 m 内 actor 数降序、fork_id 升序取；同一 base 路线不重复。

| 类 | 取法 | fork_id |
|:--|:--|:--|
| P5 行人 x⁺，trigger 在窗口里且依赖 ego（hazard 行人分叉时静止，至少一个分支 3 s 内 > 1.5 m/s，至少一个分支一直 < 0.3 m/s） | 前 2 | 78（DynamicObjectCrossing k2）、96（DynamicObjectCrossing k2） |
| P5 行人 x⁺，分叉时已在走（hazard 速度 ≥ 1 m/s） | 前 1 | 60（DynamicObjectCrossing k1） |
| P5 行人 x⁺，其余 3 个 family 各 1 | 前 1 | 42（PedestrianCrossing k2）、212（ParkingCrossingPedestrian k1）、310（VehicleTurningRoutePedestrian k1） |
| P5 行人 x⁻（hazard 藏在地下） | 前 1 | 131（VehicleTurningRoutePedestrian k3） |
| cut-in x⁺ | 前 2 | 226（StaticCutIn k1）、148（StaticCutIn k3，cut-in 车分叉时已在动） |
| P6 x₁₀ | 前 2 | 328（ConstructionObstacleTwoWays k2，50 m 内 31 个 actor）、378（HazardAtSideLane k3，hazard 两轮车在动） |

共 11 个分叉点。每个回退 run 跑全部 7 个动作；动作顺序按分叉点在上表的次序 j 轮转（从 `ACTIONS[j mod 7]` 开始循环），
所以每个动作都有当第 0 个分支（没有回退，等于一次从头重跑）的时候，也能看误差是否随分支序号累积。

## 真值与 floor

- **真值**：WL 全量里同一分叉点、同一动作的从头分支（`runs/wl/gen/<set>/attempts/<route>/<attempt>`）。
- **floor**：(1) 每个回退 run 的第 0 个分支（11 个分叉点 × 方法数，都是从头重跑）对真值；(2) 分叉点 78 与 148 用 WL 原配置（`scripts/wl_fork_agent.py`，含 20 s 续跑）从头重跑全部 7 个动作各 1 次（14 个 run），对真值。
  floor 的每个量取这两部分合起来的 p95。

## 读数（每个 方法 × 分叉点 × 动作 的回退分支，对同一动作的真值）

在分支内 t = 0（虚拟 tick k）、0.5、1、2、3 s：
- ego：位置差（m）、航向差（°）、速度差（m/s）；
- hazard（`hidden.json` 里的 actor；x⁻ 不读）：位置差、速度差；行人另读「开始走」的时刻（速度首次 > 0.5 m/s 的 tick）是否一致；
- 其余 actor（按类型与分叉 tick 的最近位置对上，`jevdrive.wl._match_actors`）：50 m 内每个 actor 的位置差，报中位 / p95 / 最大，**只描述**；
- 标签（`jevdrive.wl.outcome` 原样）：collision、collision_road、unsafe，`gap_min_m`（d_front）与 TTC 的差；
- openpilot Cinque `temporal`：回退分支的前相机帧按 WL 的流构造（来源 run 的前缀帧 + 本 run 分叉前的帧 + 该分支的帧）跑一遍，对真值同一虚拟 tick 的 `temporal` 取余弦，每个分支取 3 s 内 15 帧的最小值与均值；
- 墙钟：每个回退 run 的总墙钟与分解（装载、前缀、每个分支、每次回退），对同一分叉点 7 个真值 run 的 wall_s 之和，以及 floor 重跑的 14 个 run 在本卡上的墙钟。

## 判据（写在任何回退数字之前）

用户 09-29 的要求：必须忠实恢复的是 hazard 行人（位置、朝向、速度，以及脚本行为正确续上）；其余可以务实处理，允许一些误差。
主判据只看 hazard、ego 动力学和标签；背景 actor 的差只描述，只按「是否改变了标签」计入。判据在「回退分支」（第 1–6 个分支）上算，每个方法单独判。

| # | 判据 | 过线 |
|:--|:--|:--|
| R1 | hazard 行人（x⁺ 行人 6 个分叉点）3 s 内各时刻的位置差 p95 | ≤ max(0.25 m, 3 × floor p95) |
| R1b | hazard 行人「开始走」的时刻：真值与回退里有一边开始走的分支中，两边都走且时刻相差 ≤ 2 tick 的比例 | ≥ 95% |
| R2 | ego 位置差（3 s）p95；速度差（各时刻）p95；航向差 p95 | ≤ max(0.3 m, 3 × floor)；≤ max(0.3 m/s, 3 × floor)；≤ max(2°, 3 × floor) |
| R3 | 标签一致率：unsafe、collision 两项，全部 11 个分叉点的回退分支 | 各 ≥ 98%，且不低于 floor 的一致率 − 2 pp |
| R4 | openpilot `temporal` 余弦：回退分支逐分支 3 s 内最小余弦的中位 | ≥ floor 中位 − 0.01（WL 的 z 要用它；R1–R3 过而 R4 不过 = 标签可用、z 不可用） |
| R5 | cut-in 车（2 个分叉点）与 P6 hazard（2 个分叉点）的位置差 p95 | 与 R1 同线，分类报；不过则该类不在「可回退」范围内 |

**读法**：
- 某方法 R1–R3 全过 → 该方法对 WL 的**标签**与轨迹可用；R4 也过 → z 也可用。按 R5 分类写清适用范围（行人 / cut-in / 障碍）。
- 只有 `tree` / `respawn` / `replay` 过而 `poc` / `teleport` 不过 → 行为树状态是必要的，提案的做法不够（对提案第 2 节的更正）。
- 都不过 → 回退只能用在「hazard 不依赖 ego」的子集或根本不用；写清是哪一项（行人 trigger、TM、红绿灯、车辆内部状态）先坏。
- 过了才做 WL harness 的正式选项（默认关）和文档；成本按实测写回，更正提案第 4 节。

## 追加臂：跨 run 复用地图（2026-09-29 13:10 用户经 main 追加，写于任何复用数字之前）

目的：装载（约 84 s / run）是最大的成本，Bench2Drive 只有十来张图。下一条 route 与上一条同图时，不调 `load_world`，而是在原世界里就地复位，再照常建 route 与 scenario；run 按图分组排程。
smoke 的 route.log 显示「装载」其实是两段：client 进程起来 + import（含 agent 模块、LEAD / torch）约 50 s，`load_world` + `RouteScenario` 构建约 57 s（新 server 第一条 route）。复用地图只省后一段里的 `load_world`；import 另算，一并实测报出。

**做法（`B2D_REUSE_MAP=1`，默认关）**：`LeaderboardEvaluator._load_and_wait_for_world` 在 server 当前地图 == 本 route 的 town 时不 `load_world`：销毁残留的车、行人、walker controller、传感器、scenario 放的 static prop，tick 一次，
其余照原函数（设置 Large Map 流送距离、`reset_all_traffic_lights`、CarlaDataProvider、TM 种子、tick、核对地图）。TM、GameTime、CarlaDataProvider 本来就随每条 route 的新进程重建。天气由 scenario 设。
另开 `B2D_PHASES=1`（默认关）：每个 run 写 `phases.json`（进程启动 → 进 `_load_and_wait_for_world`、装载或复位、`RouteScenario` 构建）。

**跑什么**：
- U（只复用地图）：floor 分叉点 78、148（都是 Town12）的 14 个从头分支，加 96（Town12，雨）与 60（Town12，夜）各 7 个，共 28 个从头 run，WL 原配置（含 20 s 续跑），`B2D_REUSE_MAP=1`，同一张图连续跑（每个 server 只有第一条 route 真装图）。
- UR（复用 + 回退）：11 个分叉点的 `tree` 回退 run 按图分组连续跑（BA 的 Town12 七个接连跑），`B2D_REUSE_MAP=1`。
- 对照 = 上面的 floor（照常每条 route `load_world`）。

**判据（写在任何复用数字之前）**，都对 WL 的真值 run 比：

| # | 判据 | 过线 |
|:--|:--|:--|
| U0 | 分叉前一致：ego 位姿差 ≤ 0.01 m（WL 的组剔除门）的 run 比例；分叉前按类型与位置对上的 actor 位置差最大值 | 比例不低于 floor；actor 最大差 ≤ max(0.05 m, 3 × floor p95) |
| U1 | 分支的 unsafe、collision 与真值一致 | 不一致的 run 数 ≤ floor 的不一致数 + 1 |
| U2 | 渲染：逐 run，前相机亮度与真值同 tick 差 > 10 的帧占一半以上，或过曝跳变（`render_diag.blowout`），记为渲染故障；另报逐帧 \|Δ亮度\| 的中位与 p95 | 故障 run 数 ≤ floor + 1；\|Δ亮度\| p95 的中位不超过 floor 的 2 倍 |
| U3 | openpilot `temporal`：分叉前逐 run 最小余弦（WL 的逐 run 门 0.95）；分支 3 s 内最小余弦的中位 | 全部 ≥ 0.95；中位 ≥ floor 中位 − 0.01 |
| UR | 复用 + 回退的 `tree` run：R1–R4 照上面的判据 | 同 R1–R4 |

读法：U0–U3 全过 → 地图复用对 WL 数据是等价的，可以按图分组排程；UR 也过 → 两者叠加可用，报合起来的实测加速。U2 不过（例如夜里的灯、同图前驱的泛光，见 decisions 第 60 条）→ 只在白天切片上用，或者复用前另做灯的复位。

## 追加臂：预热的 route 进程（zygote）（2026-09-29 14:20 main 追加，写于任何 zygote 数字之前）

setup 里约一半是 route 进程自己的 Python import（torch、LEAD、leaderboard）。`b2d_run.py --zygote`（默认关）：每个 worker 在当前 route 跑的时候就起好下一个 route 进程，
它先 import 这些重模块（不含我们自己会打补丁的 b2d_hooks、agent、p4，补丁顺序与平时一致），然后在 stdin 上等 route 参数，拿到后照常跑。仍然是一 route 一进程，崩溃隔离不变。
- Z（只开 zygote）：floor 分叉点 78、148 的 14 个从头分支，`ZYGOTE=1`；判据同 U0–U3（对真值；这里 floor 是逐位相同，所以要求逐位相同：U0 位姿差 0、U1 标签全同）。
- 组合臂：U（地图复用）与 UR（复用 + 回退）都开 `ZYGOTE=1`，报三者叠加的实测墙钟；单独的收益从 `phases.json` 的 zygote 预载时间与 Z 对 floor 的 setup 差读。

## 步骤

1. 写 `scripts/carla_rewind.py`（快照 / 回退，按方法开关）并接进 `scripts/wl_fork_agent.py`（`wl_rewind` 配置键，job 里的 `actions` 列表；默认关，原路径逐字不变）。
2. 单点 smoke：分叉点 78，方法 `tree`，看 `rewind.jsonl`、每个分支的 ego / hazard 轨迹是否像话、有无报错。
3. 全部方法 × 11 个分叉点 + floor 14 个 run，GPU 5 上 ≤ 3 个 server。
4. 读数脚本 `scripts/rewind_eval.py`（把回退 run 按分支切成虚拟的 attempt 目录，复用 `jevdrive.wl.outcome` / `_match_actors`）；openpilot 流用 `scripts/p5_openpilot.py`（P5_SET=rewind）。
5. 结果进本 todo、`research/decisions.md`、`research/carla-rewind-branching.md` 末尾的更正节；过了再做 harness 选项和文档。

## 执行记录
- [E] 2026-09-29 14:45 CST 中间状态（未判格，交接 [tmp/2026-09-29-carla-rewind-state.md](../tmp/2026-09-29-carla-rewind-state.md)）。
  floor 为 0（同配置从头重跑与 WL 真值逐位相同）。提案的 `poc` 与无 warm-up 的 `teleport` / `tree` / `respawn` 在 R1–R3 上全不过：交接时 ego 速度差中位 0.5 m/s（传动系状态没恢复），
  行人开始走的时刻一致率 0–16%，unsafe 一致 70–92%。加 warm-up（回退前重放最后 N 个前缀 tick）后 ego 与 cut-in 进线（`tree+w20`：ego 3 s p95 0.33 m、cut-in 车 p95 0.30 m、标签 24/24），
  行人还不过（开始走的一致率约 53%，状态跟着上一个分支走），正在查。成本：前缀只占 WL 单 run 墙钟约 23 / 181 s，装载 60 s、import 18 s、续跑约 60 s，提案的「160 s 是前缀」不成立。
  地图复用、zygote、openpilot 余弦都还没跑。
- [E] 2026-09-29 18:40 CST **判格**（小表 [research/results/carla-rewind/](../research/results/carla-rewind/)：summary.csv、lines.json、cost.csv、opcos.csv；box `runs/rewind/eval/`）。

  **行人修复**。旧的 `tree+w<N>` 有两处坏：(1) warm-up 期间每 tick 把行为树恢复到分叉态再 tick 一次，KeepVelocity 等行为对行人反复 `apply_control`；(2) teleport 不清行人移动组件的速度与上一次控制。
  新变体 `+w<N>f`（`scripts/carla_rewind.py`）：warm-up 期间 scenario 树的 `tick_once` 是 no-op（最终 thaw 自动删掉）；warm-up 开始时销毁并按快照重生所有地上的行人（藏在地下、physics 关的只 teleport），warm-up 期间用录下的 WalkerControl 驱动，所以分叉时正在走的行人（fork 60）也能被带到速度。
  pilot（4 个分叉点）上开始走的时刻 24/24 相差 ≤ 1 tick 后上全部 11 个分叉点。fork 310 剩下的 2 例不一致是 ego 差 0.1–0.3 m 让 `InTriggerDistanceToLocation` 翻转，不是行人状态。

  **R1–R5**（回退分支 = 第 1–6 个分支；floor 的 p95 全为 0，所以线都是固定下限）：

  | 方法 | n | R1 行人 p95（≤ 0.25 m） | R1b 开始走（≥ 95%） | R2 ego 位置 / 速度 / 航向 p95（0.3 m / 0.3 m/s / 2°） | R3 unsafe / collision（≥ 98%） | R5 cut-in / P6（≤ 0.25 m） | R4 最小余弦中位（≥ 0.99） |
  |:--|--:|:--|:--|:--|:--|:--|:--|
  | `poc`（提案） | 61 | 7.57 ✗ | 21% ✗ | 3.79 / 4.14 / 8.1 ✗ | 78.7 / 77.0 ✗ | 12.1 / 0.0 ✗ | 0.524 ✗ |
  | `teleport` / `tree` / `respawn`（无 warm-up，4–5 个分叉点） | 24–30 | 2.8–4.0 ✗ | 5–16% ✗ | 3.6–8.9 ✗ | 67–96 ✗ | 3.0–14.7 ✗ | — |
  | `tree+w20`（旧行人处理，7 个分叉点） | 42 | 4.05 ✗ | 52% ✗ | 0.30 / 0.49 / 1.7 ✗ | 97.6 / 95.2 ✗ | 0.30 / — ✗ | — |
  | `tree+w20f` | 66 | 0.10 ✓ | 91.7% ✗ | 0.79 / 1.08 / 2.1 ✗ | 97.0 / 97.0 ✗ | 0.73 / 0.19 ✗ | 0.885 ✗ |
  | **`tree+w40f`** | 66 | **0.036 ✓** | **97.1% ✓** | 0.295 ✓ / **0.305 ✗** / 1.68 ✓ | 98.5 ✓ / **97.0 ✗**（2/66） | **0.39 ✗** / 0.07 ✓ | 0.869 ✗（42 个分支） |
  | `tree+w80f`（8 个分叉点跑完） | 48 | 0.60 ✗ | 82% ✗ | 1.89 / 2.73 / 2.9 ✗ | 100 / 97.9 ✗ | 0.74 / 0.23 ✗ | 0.790 ✗ |

  读法：没有一个方法 R1–R3 全过。最好的 `tree+w40f` 过了用户最在意的行人（R1、R1b），ego 位置、航向和 unsafe 标签也在线内；差在 ego 速度 p95（0.305 对 0.3）、collision 一致（2/66：fork 60 shift_L 真值有碰撞、回退没有；fork 131 op_stop 反过来，都是低速擦碰的临界）和 cut-in 车（0.39 m）。
  R4 远不过：分支内 `temporal` 最小余弦中位 0.87（均值中位 0.975），分叉前的前缀一致（≥ 0.99）。warm-up 不是越长越好：w10 → w20 → w40 改善，w80 反而变差（原因没查）。
  方法与 warm-up 长度是在这 11 个分叉点上选的，w40f 的数有选择偏差；「过线差一点」不改判格。
  按登记的读法：回退不是 WL 标签的等价替代。**WL-2 用从头生成**（加 zygote）；回退（`tree+w40f`）只作为默认关的选项，给「只要 unsafe 标签、能接受约 1.5% 标签差」的场合用。

  **地图复用 U / UR**（`B2D_REUSE_MAP=1`，同时开 zygote；4 个分叉点 28 个从头 run + 9 个分叉点的 UR）：
  U0 不过：分叉前 ego ≤ 0.01 m 的 run 只有 32%（floor 100%），ego 最大差 0.10 m，背景车最大差 54 m（前缀的背景交通变了）。U1 过（unsafe / collision 28 / 28 一致）。
  U2 没有渲染故障（亮度坏帧 > 一半的 0 个、blowout 0 个），但 |Δ亮度| p95 的中位 1.07，floor 为 0.055，超过 2 倍线 ✗。U3（UR 的前缀余弦 ≥ 0.95）只有 86% ✗。UR 的 R1b 91.7%、R2 0.96 m ✗。
  结论：复用地图后的世界不是新装的世界，WL 前缀逐 tick 对上来源 run 的前提不成立，**地图复用不用于 WL**。

  **zygote Z**（78、148 两个分叉点的 14 个从头分支）：分叉前 ego / actor 差 0（14 / 14），标签 14 / 14 一致，亮度 |Δ| p95 中位 0.008，前缀余弦 ≥ 0.9999。
  严格「逐位相同」的要求：分支 3 s 位置 13 / 14 是 0，148 shift_R 差 0.04 m；但 floor 本身也不是逐位 0（14 个里有 78 shift_L 0.013 m、148 hold 0.014 m），所以按 floor 的口径判等价，**zygote 可以默认开**。

  **成本（实测，BA，cost.csv；box 负载在这几个小时里变化大，所以按 setup + tick 数 × 同一 tick 时长拆开比）**：
  | 每个 run | 从头 | zygote | 地图复用 + zygote | `tree+w40f` 回退 run（7 分支） |
  |:--|--:|--:|--:|--:|
  | setup = 墙钟 − tick 数 × tick（s） | 85 | 73 | 49 | 107 |
  | 其中 `load_world` / scenario 构建（s） | 52 / 14 | 44 / 14 | 9 / 18 | 53 / 20 |
  | tick 数 | 489（含 20 s 续跑） | 489 | 481 | 812 |
  | 一次回退（恢复 + 死 tick，ms） | — | — | — | 24（快照 74） |
  每个分叉点 7 个分支、不带续跑，按 0.217 s / tick：从头 7 × (85 + 171 × 0.217) ≈ 855 s；zygote 从头 ≈ 770 s；`tree+w40f` 实测 283 s（**3.0×**，对 zygote 从头 2.7×）。
  WL-2 的 11 个分支 / 分叉点：从头 zygote ≈ 11 × 110 ≈ 1 210 s，回退 ≈ 107 + (110 + 11 × 61 + 10 × 41) × 0.217 ≈ 365 s（约 3.3×）。地图复用再省约 36 s / run，但不可用。
  每 tick 墙钟（中位）：server `world.tick` 约 27 ms，三路 1088 × 1560 相机的传感器等待 + 传输约 100 ms，存图折合约 12 ms，expert 11 ms，scenario 树 8 ms；剩下的瓶颈仍是渲染和传图。
  GPU 用量约 3 卡·h（GPU 6，14:50–18:40，3–6 个 CARLA）。调度行 `carla-rewind` 已 finish。
