# HUGSIM 控制器验收：把场景自己的 logged 轨迹当 plan 喂给控制器

状态: 预注册（2026-09-25 15:40 CST，写于任何验收场景运行之前；只跑过一个集合外场景的 plumbing smoke，见下）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)；姊妹篇: [B2D 控制器验收](b2d-controllers.md)
接口: [docs/hugsim.md](../../docs/hugsim.md)
代码: `jevdrive/hugsim_preset.py`（logged 轨迹 → plan）、`scripts/hugsim/preset_agent.py`（agent）、
`patches/hugsim/optional/ideal-tracker.patch`（参考控制器）、`scripts/hugsim/zs_run.py`（`--agent preset`、`--controller ideal`）、
`scripts/hugsim/preset_accept.sh`（驱动）、`scripts/hugsim/preset_eval.py`（指标与判定）

## 问题

HUGSIM（3DGS 重建的闭环仿真器，4 Hz，每步 0.25 s；plan 是前相机原点、x 右 y 前、0.5 s 间隔的 waypoint）的闭环分数里，有多少是
控制器造成的？官方控制器（`traj2control` → nuPlan iLQR（iterative Linear Quadratic Regulator，把 plan 转成加速度与转向角速度）
→ kinematic bicycle）有 heading 转置缺陷，openpilot 在它下面 20/20 原地打转；修了 heading 的 fixed 控制器（上游 PR #57）
是否就够好，也没人量过。两条控制器路径都要验收：

| 路径 | 控制器树 | 适配层 |
|---|---|---|
| (a) official | `HUGSIM-zs/official`：上游 + 我们的非行为补丁 | `zs_agent.Agent` 的同一条 plan 路径：forward_only、straight_stop |
| (b) fixed | `HUGSIM-zs/fixed`：(a) + PR #57 heading fix | 同上 |
| 参考 ideal | `HUGSIM-zs/ideal`：(a) + `ideal-tracker.patch` | 同上 |

办法与 B2D 那篇相同：喂一份**已知是好的 plan**。控制器拿到它还开不好，就是控制器（或适配层）的问题，与模型无关。

## 设计

**Plan 来源：场景自己的 logged ego 轨迹（preset trajectory）。** 每个 HUGSIM 场景的 `ground_param.pkl` 存着录制时前相机的位姿
序列，仿真器拿它当路线（RC、“Far from preset trajectory”、`command` 都从它算）；`meta_data.json` 里同一批位姿带时间戳
（nuScenes 12 Hz、PandaSet 10 Hz 是秒；Waymo、KITTI-360 存的是帧号，按 10 Hz 换算）。`LoggedPlan` 每一步：

1. 把 ego（前相机）投影到 logged 路径上，得到弧长 s0（搜索窗 [s_last − 5, s_last + 30] m，防止路径自交处跳变）；
2. 从 ego 当前车速出发，以 +2.0 / −4.0 m/s² 的上限朝 logged 车速（该弧长处，0.5 s 窗口的平均速度）逼近，积分 3 s；
3. 在 t = 0.5 k s（k = 1..6）处取 logged 路径上对应弧长的点（过了终点就沿最后一段直线外推），换到 ego 的 plan 坐标系。

这就是一个知道路线和录制车速的完美 planner 在这一刻会给的 plan：从车现在的位置出发，回到录制路径上。为什么不直接按
录制时刻表回放：所有场景都以 1 m/s 起步，而录制车速是 4–15 m/s，原样回放的第一个 waypoint 要求 0.5 s 内加速到录制车速，
超出 iLQR 的 |a| ≤ 3 m/s² 与 HUGSIM comfort 的 2.40 m/s² 上限，测的就是饱和而不是跟踪；+2.0 m/s² 的 ramp 在两者之内。
为什么不用已有的 route follower（`agent_client.RoutePolicy`，清单里 official 下 7/10 完成）：它按固定 6 m/s 上限开，不带
录制车速，也不走 `zs_agent` 的 plan 后处理；这里要的是和模型 plan 一模一样的路径。

**同一条适配路径。** `preset_agent.PresetAgent` 是 `zs_agent.Agent` 的子类，只把“调模型”那一格换成 `LoggedPlan`，
其余（逐步循环、forward_only、straight_stop、`zs_steps.jsonl` 记录）是模型 run 的同一段代码。录制轨迹本身不倒车，
forward_only 预期不改动任何 plan；录制里有停车时 straight_stop 会接管，这正好也验一下它。两者改动了多少 plan 逐 run 统计。

**参考：ideal 控制器。** HD-Score（每步 PDMS 的均值 × RC，PDMS 按**计划轨迹**打分）的“参考值”不能拿 1.0：录制轨迹在
3DGS 重建里未必处处满足 DAC（drivable area compliance，footprint 下要有地面点）/NC（no collision）/TTC，
在插入了 actor 的场景里更可能撞上。所以参考是同一个 plan 来源在**完美执行**下的结果：`ideal-tracker.patch` 让 `closed_loop.py`
不调 `traj2control`，每步让 ego 严格沿 plan 走 0.25 s（过原点与前两个 waypoint 的二次曲线，终点取曲线的切向与速度），
碰撞、路线、actor、渲染、打分全是原样的 env 代码（实现上是预置 bicycle 一步的航向、速度、转角，使零控制的一步正好落到目标上；
env 类本身不改）。于是 “控制器 vs ideal” 的差就是控制器造成的全部差别。

**场景（14 个，选择只看数据集、难度、有无 actor，不看任何结果）。**

| 组 | 场景 | 说明 |
|---|---|---|
| static（9，无插入 actor） | nuScenes 0071-easy（直行）、0383-easy（左转）、0920-easy（右转）；Waymo 130854534658-easy、113792265837-easy；KITTI-360 1290_1490-easy、2800_3000-easy；PandaSet 021-easy、034-easy | 清单的 6 个 easy 场景，再给 Waymo / KITTI-360 / PandaSet 各补一个（`scored.txt` 里该数据集 easy 的第一个），每个数据集至少 2 个 |
| actor（5） | nuScenes 0062-medium-00（前方 30 m 静止车）、0383-hard-00（2 辆车）；Waymo 130854534658-hard-00；KITTI-360 1290_1490-hard-00；PandaSet 021-hard-01（AttackPlanner，冲向 ego 的车） | 清单的 2 个 actor 场景 + 每个其余数据集一个与 static 同 scene 的 hard 场景 |

列表：[hugsim-static.txt](hugsim-static.txt)、[hugsim-actor.txt](hugsim-actor.txt)。3 个控制器 × 14 = 42 个 run，每条 lane 一个
控制器、一个仿真器，GPU 0。

**actor 场景的参考怎么定。** 录制轨迹不知道后插进来的 actor，可能直接撞上。这里不把 “不碰撞” 当参考，而是**预注册：actor 场景
的参考 = ideal 控制器在同一场景上的结果**（包括它的结束原因与步数）。ideal 也撞了，那次碰撞就不算控制器的错；控制器要做的是
“和 ideal 一样”：结束原因相同、HD-Score 相差不大。

## 指标（跑之前定死）

跟踪误差逐步算，都用 `zs_steps.jsonl` 里仿真器给的 ego 位姿（前相机，地面投影）：

- **对所喂 plan（主）**，在第 k 步 plan 的坐标系里：第一个 waypoint（t = 0.5 s）对 k+2 步的实际位置，拆成 lateral（x，右为正）
  与 longitudinal（y，前为正）；heading 误差 = k+2 步航向 − plan 在该 waypoint 处的切向（过原点与前两点的二次曲线在 0.5 s 的切向，
  即第二个 waypoint 的方向）。另报一步（0.25 s）的版本：同一条二次曲线在 0.25 s 处对 k+1 步，ideal 在这上面按构造是 0，用来
  验证指标代码本身。
- **对 logged 路径**：每步 ego 到录制路径的有符号 cross-track 距离（右为正）、ego 航向 − 路径切向。
- **结束原因**：background collision / foreground collision / off-route（离路线 > 10 m）/ complete / 400 步。
- **分数**：HD-Score、RC、NC、DAC、TTC、C、PDMS（`eval.json`），逐场景对 ideal 配对相减。
- 统计范围：tracking 的主判定只用 static 场景的全部步（pooled，报 median 与 p95）；actor 场景的跟踪误差照报，不进判定
  （AttackPlanner 等可能提前结束 episode）。

## 通过标准（每条控制器路径分别判；四项都过才算 pass）

| # | 项 | 通过 |
|---|---|---|
| P1 | 跟踪所喂 plan（static，0.5 s） | median \|lateral\| ≤ 0.10 m 且 p95 ≤ 0.30 m；median \|longitudinal\| ≤ 0.20 m 且 p95 ≤ 0.50 m；median \|heading\| ≤ 1.5° 且 p95 ≤ 5° |
| P2 | 跟踪 logged 路径（static） | 每个 run 的 median \|cross-track\| ≤ 0.30 m；max \|cross-track\| ≤ 1.0 m 的 run ≥ 8/9 |
| P3 | 结束原因（static） | complete ≥ 8/9，且与 ideal 相同的 ≥ 8/9 |
| P4 | HD-Score 对 ideal | static：均值差 ≥ −0.05，且每个场景差 ≥ −0.15；actor：结束原因与 ideal 相同 ≥ 4/5，\|HD 差\| ≤ 0.15 的 ≥ 4/5 |

阈值的来由：0.5 s 内 iLQR 对一条可行、光滑的 plan 应做到厘米级；10 cm / 1.5° 留了 4 Hz 离散与 iLQR 0.5 s 离散不一致
（上游 issue #75）的余量。cross-track 0.3 m 是车道宽度（约 3.5 m）的一小部分，1 m 是开始可能压线的量级。HD 的 0.05 是我们
在 HUGSIM 上关心的模型差距（LTF official→fixed +0.057）的量级：控制器自身造成的差若大于它，就会淹没模型之间的比较。

**参考自身的 sanity（不是控制器判定，但先看）：** S1 ideal 的一步（0.25 s）误差 < 1 cm（否则指标代码或 ideal 实现有错，先修）；
S2 ideal 在 static 场景上 complete ≥ 8/9、HD 均值 ≥ 0.80。S2 不过说明录制轨迹在 3DGS 场景里本身不 “好”（地面点缺失、
重建偏差），照样拿 ideal 当参考做配对比较，但要在结论里写明参考本身的分数。

**预先写下的偏离规则：** 基础设施崩溃（不是 episode 结束）的 run 重跑一次；plumbing 问题修复后整组重跑，记偏离。
阈值、场景、plan 来源在看结果之后不改。

## Plumbing smoke（集合外，只验代码）

nuScenes scene-0051-easy-00（不在上面 14 个里），ideal 与 official 各一次，只用来确认 agent、ideal 树、`preset_eval.py`
能跑通、S1 成立。第一次 smoke 暴露一个 plumbing 问题并已修复：HUGSIM 的 env 类（`hugsim_env`）是 editable install，
**不论从哪棵树运行都从 `$DATA_DIR/third_party/HUGSIM` 导入**；`traj2control` 则从运行目录的树导入（已核对：official 树里
是未修 heading 的版本，fixed 树里是 PR #57 的版本，所以考试里 official / fixed 的区分是真的）。因此 ideal 补丁只改
`closed_loop.py`，不改 env 类。修复后的 smoke：ideal 的一步误差 median 0.3 mm、p95 0.9 mm（`zs_steps.jsonl` 的位姿只保留到 mm，S1 成立），每个 run 约 25 s。official 在这个集合外场景上的数字也看到了；上面的阈值在 smoke 之前已随 `preset_eval.py` 提交（commit c54f88a 的 `CRIT`），没有改动。

## 结果

（跑完再填。）
