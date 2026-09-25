# Alpamayo 1.5 在 Bench2Drive 全量里“停住不动”的诊断

状态: 诊断完成（2026-09-25 10:30）；旧全量停在 119/220 并作废；main 批准 F1 + 两臂 re-smoke（第 6 节，写于运行之前），slot `alp-b2d-smoke2` → `alp-b2d-full`（从头 220 条）已排上；smoke2 选了 `f1`，f1 全量被用户暂停在 17/220，“开得慢”的诊断见第 7 节（2026-09-25 14:40）
主题: ../../research/openpilot-and-open-driving-models.md、../../research/trajectory-to-control.md
相关: [B2D 考试](bench2drive.md)（预注册与“控制器换成 Zoo PID”的偏离）、[openpilot 迁移](openpilot-migration.md)（同类适配 bug 的先例）、
[控制器 API](../../docs/b2d-controller.md)

## 结论先行

- **停住的主因是适配 bug，不是模型要停。** Zoo PID（Bench2DriveZoo 的 UniAD/VAD 官方 trajectory→control PID）把
  waypoint 之间的**距离绝对值**当作目标速度（`desired_speed = Σ‖wp[i+1]−wp[i]‖·2/5`），不看方向。Alpamayo 在静止时有
  48% 的 plan 是“向后退”（x < −0.3 m）：它的 unicycle 积分器没有 v ≥ 0 的约束，v0 ≈ 0 时一个刹车 plan 积分出来就是倒车，
  语义上是“停着”。旧的固定控制器把这种 plan 执行成刹车（84% 的 tick 刹车），Zoo PID 把它读成 0.6 m/s 的前进速度，
  **68% 的 tick 踩 0.75 油门往前冲**。
- **连锁**：静止 → 倒车 plan 或 0.8 m/s 的蠕行 plan → Zoo PID 满油门保持 0.5 s（2 Hz 规划、控制量保持到下一次规划）
  → 1 s 后车速约 2.0 m/s，是 plan 的 2.4 倍 → 模型从历史里看到自己在加速，plan 拉长 → 追尾前车或撞上施工牌 →
  车贴着障碍物，模型说“停/后退”，Zoo PID 又把后退读成前进，油门、刹车交替，车一直顶在障碍物上。
  **80 段 ≥ 10 s 的停车里 71 段（停车时长的 91%）开始于一次碰撞前后 3 s 内**；v < 3 m/s 的 54 次碰撞里 48 次在碰撞前
  2 s 内有一个“静止时拿到油门”的 plan，其中 30 次是倒车 plan。
- 117 条完成路线 82% 的仿真时间 v < 0.5 m/s；35 条跑满 4000 tick（TickRuntime），23 条被判 blocked。起步没有问题
  （第一次 v ≥ 0.5 m/s 的中位数 1.4 s），坐标系、时间基、egomotion 历史都查过是对的。
- **处置**：全量已暂停（干净地停掉了 slot 的进程，没写 `.done/.failed`，依赖它的 `simlingo-exp-extra`、`hugsim-exam`
  会一直等，需要 main 重新安排）。建议的修复 F1（forward-only plan，速度截在 0 以上）是纯语义修复，离线重放在已记录的
  9021 个静止倒车 plan 上让 Zoo PID 的刹车比例从 39.9% 升到 99.4%，对前进 plan 零影响。控制节奏 F2 是否一起改要 main 定。
  119 条旧结果应作废。

## 数据与方法

- 全量 run：`$DATA_DIR/runs/zeroshot-exam/b2d/full220-alpamayo-zoopid/`，暂停时 119 条完成，其中 117 条日志完整进入分析
  （`attempts/<id>/<n>/{plans,ticks,results}.jsonl`：2 Hz 的每次规划含 20 点 rear-axle path、Zoo PID 的 metadata 和
  CoC 文本；20 Hz 的每 tick 控制量与仿真真值位姿；官方 `results.json`）。
- 对照：`smoke-alpamayo-zoopid`（02:09，同一套 Zoo PID，5 条预注册路线，DS 68.4）和 `smoke-alpamayo`（预注册的固定控制器，
  DS 60.8）。
- 定义：**stall** = 车速 v < 0.5 m/s；一段 stall episode = 连续 ≥ 10 s。静止时的 plan（v < 0.3 m/s）按 3 s 处前进距离分三类：
  **stay**（x@3s < 1.2 m，即平均 < 0.4 m/s，正好是 Zoo PID 的刹车阈值）、**reverse**（前 3 s 内 x < −0.3 m）、**go**（其余）。
  碰撞时刻：官方 infraction 里的位置是 ego 车辆中心，取真值轨迹上离它最近的 tick。
- 代码：`scripts/zeroshot_b2d_alp_stall.py`（逐路线 / 逐规划表）和 `scripts/zeroshot_b2d_alp_stall_probes/`（episode、
  碰撞、静止响应、离线重放、坐标检查），在 box 上用 `envs/carla` 跑，只读日志，不占 GPU。结果文件在
  [research/results/zeroshot-b2d/alp-stall/](../../research/results/zeroshot-b2d/alp-stall/)。
- 缺口：这一轮的 agent 还没有写 `route.json`（路线命令），所以“在不在路口”只能用 nav 文本是否出现（前方 60 m 内有左/右转）
  和 CoC 关键词近似。

## 1. stall 的量

| run | 路线数 | DS | RC | Completed / TickRuntime / blocked | v < 0.5 的时间占比 | ≥ 10 s episode | 其中始于碰撞 |
|---|---:|---:|---:|---|---:|---:|---:|
| 全量 Zoo PID | 117 | 44.0 | 76.7 | 59 / 35 / 23 | **82%**（9603 / 11649 s） | 80（8438 s） | **71（7700 s，91%）** |
| smoke Zoo PID | 5 | 68.4 | 89.6 | 4 / 0 / 1 | 65% | 3 | 2 |
| smoke 固定控制器 | 5 | 60.8 | 70.0 | 3 / 0 / 0（2 条 route deviation） | 44% | 0（最长 6.1 s） | — |

DS（Driving Score，route completion 乘违规系数）和 RC（Route Completion）是官方值。读法：全量里一半路线（58/117）不是开完的，
而是停到 4000 tick 上限或被判 blocked；这 58 条的 DS 平均 22–27，开完的 59 条平均 63.4。55 条路线以一段 stall 结束，
其中 48 条在那段 stall 开始时刚发生过碰撞。smoke 的 5 条里 2373、2390 已经出现同样的“撞前车后顶住”，只是 n = 5 时
被 DS 的均值盖住了，闸门没有拦下来。

![stall overview](../../research/figs/alp-b2d-stall-overview.png)

图 1：(a) 117 条路线的仿真时间拆成行驶（蓝）、没有碰撞的停车（橙）、碰撞后 ≥ 10 s 的停车（红），按红色时长排序；
右侧三分之一的路线几乎全程是红色，都是撞上之后顶在障碍物上直到 200 s 上限。(b) 碰撞发生前的静止时刻，三类 plan 下
控制器踩油门的比例：固定控制器对倒车 plan 基本刹车（16% 油门），Zoo PID 反而 68% 踩油门；离线加上 forward-only 截断后
倒车 plan 的油门比例降到 0.6%，go plan 不变。

**在哪里停。** 没有“起步停”：第一次 v ≥ 0.5 m/s 的中位数 1.4 s、最大 7.1 s，静止的历史没有让模型不肯出发。
71 段碰撞后的停车里，碰撞对象是车辆（前车、停着的事故车、cut-in 车）和施工牌、交通灯杆、护栏；全量共 144 次碰撞
（车辆 95、静物 48、行人 1），每条 1.23 次，固定控制器的 smoke 是 5 条 1 次。9 段没有碰撞的长停车里，CoC 多是
“Yield to the cross-traffic…”、“Keep distance to the lead vehicle…”，nav 文本在其中 3 段里出现（路口转弯前），
这些才更像模型自己的“等”。红灯相关的 CoC 只占停车 plan 的 1%。

## 2. 停车时模型在说什么、控制器在做什么

| 停车段 | 静止 plan 数 | stay | reverse | go |
|---|---:|---:|---:|---:|
| 碰撞后顶住（≥ 10 s） | 15233 | 35% | 44% | 21% |
| 无碰撞的长停车（≥ 10 s） | 1472 | 34% | 45% | 21% |
| 短停车（< 10 s） | 2202 | 26% | 48% | 25% |

模型在停车时并不是一直说“走”：约 80% 的 plan 是 stay 或 reverse，CoC 的首位是“Keep distance to the lead vehicle since it is
stopped ahead”“Stop for the construction barricade blocking the lane ahead”。它说的是对的——车就顶在前车或施工牌上。
所以顶住之后的长停车里，“模型要停”和“车动不了”是同时成立的，问题出在**怎么走到顶住这一步**。

碰撞之前（去掉每条路线第一次碰撞前 2 s 之后的所有 plan）的静止时刻，控制器的响应：

| plan 类别 | n | 固定控制器 smoke：油门 tick 占比 | Zoo PID 全量：油门 tick 占比 | Zoo PID：0.5 s 后车速 | Zoo PID + forward-only（离线重放）：油门占比 |
|---|---:|---:|---:|---:|---:|
| stay | 370 | 68%（跟蠕行） | 4% | 0.03 m/s | 1% |
| reverse | 667 | **16%** | **68%** | 0.58 m/s | **0.6%** |
| go（plan 平均 0.81 m/s） | 352 | 91% | 98% | 0.66 m/s；1.0 s 后 **1.96 m/s** | 98% |

固定控制器的 n 是 smoke 的 41/66/47 个 plan。两件事：倒车 plan 被 Zoo PID 执行成前进，这是 bug；go plan 在 Zoo PID 下
1 s 后车速是 plan 平均速度的 2.4 倍。**更正（同日，写 re-smoke 验收标准时重看）**：初稿把这 2.4 倍整个算在“2 Hz 保持”上，
不对。第一个 0.5 s 里满油门只把车带到 0.66 m/s，还低于 plan 的 0.81 m/s；冲到 1.96 m/s 发生在下一次规划之后——车动起来，
模型看到自己在走，plan 拉长，目标速度随之升高。保持本身的份额是：30% 的起步在 0.5 s 窗口里已经超过 1.1 倍目标速度
却仍在踩油门（per-tick 调用时这些 tick 会刹车）。所以 2.4 倍主要是模型对自身运动的反应加上 Zoo PID 追“0.5–3 s 平均速度”
的定义，节奏只放大了一部分。

![route 1833](../../research/figs/alp-b2d-stall-route1833.png)

图 2：route 1833（ConstructionObstacleTwoWays）12–40 s。(a) 车速；(b) 每次规划 3 s 处的前进距离，按 stay / reverse / go
着色；(c) 油门（上）和刹车（下）。看 12–23 s：每一次 go 都换来 0.5 s 满油门、车冲到 2 m/s，然后 1 s 满刹车，停-冲-停
三个循环；第四次冲到 1.8 m/s 时在 24.2 s 撞上施工警示牌。之后模型一直给 reverse / stay（“Stop for the barricade…”），
Zoo PID 把 reverse 读成前进，油门和刹车交替，车顶着警示牌 175 s 直到 4000 tick。

## 3. 适配层逐项审计

| 项 | 怎么查的 | 结果 | 判定 |
|---|---|---|---|
| plan 原点与坐标系 | 预注册：PhysicalAI-AV 的 rig 原点在后轴中心地面，x 前 y 左，egomotion 与输出轨迹同一系；plan 第一个点紧接原点（例如 −0.10 m @0.25 s） | 与控制器要的后轴坐标一致，没有 openpilot 那种相机原点偏移 | 正确 |
| 时间基与单位 | 速度 ≥ 3 m/s 的 1096 次规划：plan x@1s / 车速 | 中位数 1.048（真值 1 s 后实际前进 / 车速 = 0.969），64 点 @10 Hz 重采样到 0.25 s 没有错位 | 正确 |
| 左右与航向符号 | 同上，plan y@1s、航向对真值 1 s 后的横向位移、航向变化 | 横向符号一致 83%（r = 0.44），航向 76%（r = 0.31）；Zoo PID 常改用 route target 转向，所以一致性不高，但远高于反号时的 < 50% | 正确 |
| egomotion 历史 | 代码：16 × 0.1 s 后轴位姿（GNSS + 罗盘 + 里程计），t0 系、yaw 逆时针；server 转成 xyz + 旋转矩阵 | plan 隐含的 v0 与车速一致（上一行），说明历史的尺度和帧率对；起步中位数 1.4 s，静止历史没有造成“起步停” | 正确 |
| 相机时序与 f-theta | 预注册 smoke 已核（4 帧 @10 Hz、组内 ±1 tick 抖动、f-theta 重采样的图与真车一致）；这次没改 | 与固定控制器 smoke 相同，不能解释控制器换了之后的差别 | 不变 |
| nav 文本时机 | 每次规划按路线进度给“Turn left/right in d m”（≤ 60 m） | 未改；9 段无碰撞长停车里 3 段有 nav，在路口前等横向车流 | 不变 |
| **倒车 plan 的语义** | Alpamayo `action_to_traj`：v = v0 + cumsum(a·dt)，没有 v ≥ 0 截断；Zoo PID 的 `desired_speed` 用距离绝对值 | 静止 plan 的 48% 是倒车，Zoo PID 68% 踩油门往前；9021 个倒车 plan 里只有 39.9% 触发刹车 | **bug** |
| 2 Hz 规划 + 控制保持 | Zoo PID 在每次规划调用一次、保持 0.5 s（照 AD-MLP agent 的节奏）；速度环 K_P = 5、delta 截在 0.25，差 0.15 m/s 以上就饱和到 0.75 | 从静止起步 0.5 s 满油门；30% 的起步在这 0.5 s 里超过 1.1 倍目标速度仍踩油门（1 s 后的 2.4 倍主要来自下一次 plan 拉长，见第 2 节的更正）；刹车同样保持 0.5 s，形成停-冲-停循环。UniAD/VAD 每 tick 调用，超速 1.1 倍时下一 tick 就刹 | 节奏选择放大了问题，要 main 定 |
| aim point / target | aim 取中点离原点最接近 4 m 的那段；plan 很短时 aim 是第一个点 | 倒车 plan 的 aim 角 ≈ ±2（180°），但 `|angle_target| < |angle|` 让它改用 route target 转向，没有出现满舵 | 无害 |
| 去掉的低速限油门 | AD-MLP 在车速 > 3 m/s（转弯 2.5）时把油门压到 0.05 | 只在 3 m/s 以上生效，不影响起步冲撞；它影响的是 v ≥ 3 m/s 的 90 次碰撞里有多少能避免，不是 stall | 不是 stall 的原因 |
| 预热 / stop-hold | Alpamayo 没有预热，第一组图就规划；Zoo PID 没有 stop-hold，每次规划重新判刹车 | 起步正常；缺 stop-hold 本身不是问题，问题是倒车 plan 让“停”判不出来 | — |

**控制器从固定控制器换成 Zoo PID，改变了什么。** 固定控制器 20 Hz 跟踪、按里程计把 plan 重投影到当前位姿，倒车或过短
的 plan 进 stop-hold 刹车，起步速度跟着 plan 走（1 s 后 1.51 m/s，对 plan 1.37 m/s）。Zoo PID 丢了这三样：倒车当前进、
0.5 s 内没有速度反馈、没有停车保持。模型、rig、nav、推理配置两次完全相同，所以同样 5 条 smoke 路线上 stall 从最长 6 s
变成 61–90 s、碰撞从 1 次变成 3 次，差别来自执行层（全量的 DS 44.0 是另一组路线，不直接和 smoke 的 60.8 比）。

## 4. 处置：暂停全量（2026-09-25 09:23）

- 先 SIGKILL 了 `slot_run.sh`（pid 633596），让它不写 `alp-b2d-full.done/.failed`：否则 b2d_run 被打断后，launcher 按
  “没跑完 ≤ 15 条”的规则会判成功、写 `.done`，把依赖它的 slot 放出去。
- 再给 `b2d_run.py`（pid 289616）发 SIGINT：它按设计取消 4 个 worker、停掉自己的 CARLA（server index 410–413）；launcher
  的 trap 停掉 Alpamayo policy server。之后核对：没有残留的 route 进程、CARLA 端口和 policy server，GPU 1 释放约 34 GB。
  只按 pid 杀，没有用任何 `pkill -f`。
- 119 条完成、4 条 `cancelled_by_user`，可以续跑，但结果带 bug，**不建议续跑，修复后从头跑**。
- `$DATA_DIR/runs/zeroshot-exam/gpu-plan.md` 已追加记录。`simlingo-exp-extra`、`hugsim-exam` 以 `alp-b2d-full` 为依赖，
  现在会一直等，要 main 重新安排。tmux 窗口 `jev:alp-b2d-full` 留着显示 `runner exit 130`，main 看过可以关。

## 5. 建议的修复与 re-smoke（待 main 决定；写成偏离，先于任何新分数冻结）

**F1（bug 修复，建议必做）：forward-only plan。** 在 `scripts/b2d_zoo_pid_wrap.py` 取 6 个 waypoint 之前，把 plan 的每一段
前进分量为负的位移置零（等价于 unicycle 的速度截在 v ≥ 0：刹车刹到 0 就停住，不会倒车），再累加成 path。理由是语义
而不是分数：真车刹车不会倒车，Alpamayo 的倒车只是 v0 ≈ 0 时积分器没有截断；固定控制器本来就是这样执行的。
Zoo PID 本身一字不改。做成 agent 配置开关（默认关，复现当前 run）。离线重放（box 上 vendored `control_pid`，所有 run 的
静止 plan）：

| run | plan 类别 | n | 刹车比例：现在 | 刹车比例：F1 | 目标速度：现在 → F1 (m/s) |
|---|---|---:|---:|---:|---|
| 全量 | reverse | 9021 | 39.9% | **99.4%** | 0.61 → 0.01 |
| 全量 | stay | 6768 | 98.3% | 98.9% | 0.18 → 0.16 |
| 全量 | go | 4351 | 2.0% | 2.0% | 0.80 → 0.80 |
| smoke Zoo PID | reverse | 206 | 31.1% | 99.5% | 0.78 → 0.01 |

**F2（控制节奏，需要 main 选）。** (a) 保持现在的 AD-MLP 节奏（每次规划算一次、保持 0.5 s），只修 F1；
(b) 每 tick 调用一次 `control_pid`，用最新 plan 按其“年龄”平移时间（取 age + 0.5 … age + 3.0 s 的点，按里程计重投影到当前
位姿），这就是 UniAD/VAD 在 20 Hz 下的闭环语义，超速 1.1 倍时下一 tick 就刹车，消掉保持窗口里的超速油门（2.4 倍的起步比例
主要不是它造成的，见第 2 节的更正，所以不拿它当 F2 的判据）。
我倾向 (b)：Zoo PID 的增益（K_P 5、delta 截 0.25）是按 20 Hz 反馈设计的，2 Hz 保持把它变成了 0.5 s 的 bang-bang。
但 AD-MLP 确实这样发布，(a) 也站得住，所以这是决策，不是 bug。

**re-smoke（约 1.5 h，GPU 1 一个 server + 1–2 个 worker）。** 路线：5 条预注册 smoke 路线 + 全量里 6 条“撞后顶住”的
诊断路线（1833、1852、1956、2668、4183、11381），TM seed 0。两个 arm：F1、F1 + F2(b)。
验收只看适配指标，不看 DS：碰撞前静止倒车 plan 的油门比例 ≤ 5%；v < 3 m/s 的碰撞里“碰前 2 s 静止拿油门”的次数；
go plan 起步 1 s 后车速 / plan 速度的中位数；碰撞后 ≥ 10 s 的停车段数。F2 选 (a) 还是 (b) 的规则也事先写死：
（初稿写的“(b) 只在起步过冲从 2.4 降到 ≤ 1.3 时采用”作废，理由同上；main 批准后的最终规则见第 6 节。）

## 复现

```bash
# box, repo root: per-route / per-plan tables (routes.csv, plans.csv, cot.json)
D=$DATA_DIR/runs/zeroshot-exam/b2d
$DATA_DIR/envs/carla/bin/python scripts/zeroshot_b2d_alp_stall.py full=$D/full220-alpamayo-zoopid \
    smokezoo=$D/smoke-alpamayo-zoopid smokefixed=$D/smoke-alpamayo --out $D/diag-alp-stall
# probes (stdout JSON): episodes, collisions, standstill response (2nd arg = before the first collision), replay, frames
for r in full220-alpamayo-zoopid smoke-alpamayo-zoopid smoke-alpamayo; do
  $DATA_DIR/envs/carla/bin/python scripts/zeroshot_b2d_alp_stall_probes/standstill.py $D/$r free; done
$DATA_DIR/envs/carla/bin/python scripts/zeroshot_b2d_alp_stall_probes/clamp_replay.py $D/full220-alpamayo-zoopid \
    $D/smoke-alpamayo-zoopid $D/smoke-alpamayo
# Mac: figures (1833 ticks/plans.jsonl pulled into <dir>)
.venv/bin/python scripts/zeroshot_b2d_alp_stall_figs.py --route-dir <dir>
```

`episodes.csv` / `collisions.csv` 是 episodes.py / collisions.py 的输出再加上“episode 起点前 3 s 到结束之间有没有碰撞”
（`ncol_in`）和“碰前 2 s 有没有静止拿油门的 plan”（`lunge`、`revthr`）两列，合并在 Mac 上做。

## 6. 修复与 re-smoke：预注册（2026-09-25，main 批准，写于 smoke2 运行之前）

**偏离记录（B2D 考试 Alpamayo 部分，第二条，第一条是控制器换成 Zoo PID）。** 执行层加两个适配开关，都在
`scripts/b2d_zoo_pid_wrap.py`（`ZooPID(forward_only, cadence)`），agent 配置键 `plan_forward_only`、`zoo_cadence`；
**默认值保持现有行为**（`false` / `"plan"`），openpilot 的配置不受影响（main 通知 openpilot agent）。vendored 的
`b2d_zoo_pid.py` 一字未改。

- **F1 `plan_forward_only: true`**：plan 前面补原点 (0, 0)@0 s，前进分量为负的每一段位移置零再累加，即速度截在 ≥ 0。
  理由见第 5 节，是语义修复，不看分数。
- **F2b `zoo_cadence: "tick"`**：每次规划只保存 plan（连同规划时刻的后轴世界位姿，由位姿历史插值到相机组时刻），
  每个 20 Hz tick 用当前位姿把它重投影到当前后轴系、时间减去 plan 的年龄，再调用一次 `control_pid`（UniAD / VAD 的调用方式，
  PID 窗口按每 tick 推进）。tick 日志多一列 `zoo_desired`。
- 单元测试 `scripts/test_b2d_zoo_pid_wrap.py`（box `envs/carla` 上 4/4 通过）：默认路径与改前逐位相同；倒车刹停 plan 在 F1 下变成
  刹车、不加 F1 时是油门；前进 plan 不变；年龄平移与位姿重投影（直行 5 m/s、0.3 s 后 0.5 s 处的点在前方 2.5 m，左右不翻）。

**smoke2（slot `alp-b2d-smoke2`，GPU 1，一个 Alpamayo server，两臂并行各 2 个 CARLA worker，server index 420–421 / 430–431，
`--no-reap`）。** 路线 11 条：5 条预注册 smoke 路线（2390、24211、1711、2373、3564）+ 6 条旧全量里“撞后顶住”的路线
（1833、1852、1956、2668、4183、11381），TM seed 0，`--max-attempts 2`。两臂：`f1`（F1 + AD-MLP 保持，按原预注册的节奏）、
`f1f2b`（F1 + F2b）。输出 `$DATA_DIR/runs/zeroshot-exam/b2d/smoke2-alpamayo-{f1,f1f2b}/`。

**验收（只看适配指标，DS 不参与；`scripts/zeroshot_b2d_alp_smoke2_check.py` 自动算，写 `smoke2-alpamayo-choice.json`）：**

| 判据 | 定义 | 门槛 |
|---|---|---|
| A0 基础设施 | 11 条都 `finished`、没有 CARLA 重启、每条都有 plans / ticks / 官方结果 | 必须 |
| A1 倒车不变油门 | 自由静止（v < 0.3 m/s、在该路线第一次碰撞 2 s 之前）的倒车 plan 之后 0.5 s 内油门 tick 占比 | ≤ 5%（旧全量 68%，旧 smoke 78%） |
| A2 没有倒车引起的碰撞 | v < 3 m/s 的碰撞，碰前 2 s 内有“静止倒车 plan 后踩了油门” | 0 次 |
| 只报告 | 保持窗口里超速 1.1 倍仍踩油门的 tick 占比；起步 v(+1 s) / plan 平均速度；所有“静止拿油门后低速碰撞”；碰撞后 ≥ 10 s 停车段数；DS / RC | — |

**选择规则**：`f1f2b` 过 A0–A2 就用它（与 UniAD / VAD 基线调用控制器的方式一致，main 定的默认）；否则 `f1` 过就用 `f1`；
都不过则 smoke2 slot 失败、全量不启动，回报 main。用旧 smoke（`smoke-alpamayo-zoopid`）试跑检查脚本：A1 = 78%、A2 = 2 次，
按预期不通过。

**全量（slot `alp-b2d-full`，同名重新 arm，`--after alp-b2d-smoke2`，GPU 1，4 worker，server index 440–443）。**
`scripts/zeroshot_b2d_alp.sh full 1` 读 choice 文件，220 条从头跑到 `full220-alpamayo-zoopid-<arm>/`；旧的
`full220-alpamayo-zoopid/`（119 条）作废，不并入。报告 DS 均值与按路线 bootstrap 95% CI、SR（Bench2Drive
`merge_route_json.py` 定义：Completed 且除 min-speed 外无违规）与 Wilson 95% CI、没跑完的路线数。

## 7. f1 全量为什么“开得慢”（2026-09-25 14:40，全量被用户暂停在 17/220）

f1 全量（`full220-alpamayo-zoopid-f1`，F1 forward-only + AD-MLP 节奏的 Zoo PID，未加 Zoo 的低速限油门）暂停时 17 条：
DS 44.4、SR 41%、RC 67.7，17/17 条有 min-speed 记录，8 条以 `TickRuntime` 结束。只读日志诊断，没有动这个 run，也没有跑 CARLA
（离线证据已经能下结论）。

### 结论先行

- **`TickRuntime` 不是“开得慢”，是“撞上之后顶住不动”。** 8 条全部在 6–32 s 之间发生第一次碰撞，之后车速 < 0.5 m/s
  直到 4000 tick（200 s）上限：8 × 200 s 里 **1395 s（87%）是碰撞后顶住**。按各自碰撞前的平均速度推算，7/8 条本来在
  48–84 s 就能跑完（1956 在 6.3 s 就撞了，推算 204 s）。所以 TickRuntime 的亏空约 100% 来自碰撞 + 顶住，0% 来自延迟，
  慢速巡航本身不会超时。
- **顶住是执行层的问题（c）。** 顶住时 Zoo PID 每 0.5 s 给一次 0.75 油门，0.5 s 后车速中位数 0.03 m/s、1 s 位移中位数 ≤ 0.08 m
  ——车头顶着障碍物，只会往前推；Zoo PID 不会倒车，转向 |steer| ≈ 0.02。同一模型、同一批路线用预注册的固定控制器
  （`full220-alpamayo`）跑，这 8 条里的 6 条 **Completed**（其中 5 条也撞了，但 1–7 s 内就重新开起来）；9 条共享路线上
  DS **58.2 vs 25.2**、RC **93.1 vs 51.8**。
- **撞上的一个主要原因也是执行层：Zoo PID 基本不执行 plan 的横向。** 车速 ≥ 2 m/s、plan 在 2 s 处横向偏 ≥ 1 m 时，
  2 s 后实际横向位移只有 plan 的 **4%**（固定控制器 **64%**）。Zoo PID 的 aim point 取中点离车最接近 4 m 的那段，
  在 0.5 s 间隔的 waypoint 上，车速 ≥ 4 m/s 时就是 0.5 s 那个点（横向还很小）；route target 的角度更小时又改用 target。
  1825 / 1833（ConstructionObstacleTwoWays）撞前 2 s 的 plan 在 3 s 处向左 2.7–8.3 m（CoT “Nudge to the left to clear
  the construction trailer”），转向 −0.02–0.05，5 m/s 正面撞上施工警示牌。
- **（a）模型确实开得慢，但它解释的是 pace，不是超时。** 碰撞前，plan 自己要求的平均速度（每个 0.5 s 窗口 plan 走的距离 /
  时间）只有 2.40 m/s，实际 1.99 m/s，PDM-Lite 专家在同 17 条上平均 5.22 m/s：差距的约 87% 在 plan 本身，13% 在执行。
  32% 的规划窗口是停车 plan，CoT 里一半是“Keep distance to the lead vehicle”，23% 是 stop sign。
- **（b）延迟不存在。** CARLA 同步模式下推理墙钟（往返中位数 3.2 s）不推进仿真时间；plan 用的相机帧与施加控制的
  tick 相差 0（最大 0.05 s），仿真时间里每 0.5 s 一次规划，控制在规划时刻就用新 plan。
- **min-speed 记录不是信号。** Bench2Drive 的 `statistics_manager.py` 把 `MIN_SPEED_INFRACTION` 设成 `'unused'`，
  每个 checkpoint 都记一条“Average speed is X% of the surrounding traffic's one”，不管 X 是多少：f1 的记录从 11% 到 1479%，
  很多 > 100%。17/17 条有记录是计分器的记账方式，不影响 DS，也不说明慢。
- **建议修复（适配层，需要 main 决定）**：P1 让横向跟 plan 走（Zoo 纵向 + 固定控制器的 20 Hz 横向，或 Zoo 的 aim 改成按时间
  取约 1.5 s 的点、去掉 route target 替换）。预期：把固定控制器跑通的 6 条 TickRuntime 路线换成它的结果粗算，17 条 DS 44 → 约 58–62、RC 68 → 约 89
  （单 seed、n = 9 条的对照，是估计不是承诺）。F2b（每 tick 调 `control_pid`）单独做只消掉“刹车保持刹停”，pace 只 +4%，
  不值得单独换。

### 7.1 仿真时间花在哪里

| f1 全量 17 条 | 时间 (s) | 占比 |
|---|---:|---:|
| 碰撞后顶住（第一次碰撞之后 v < 0.5 m/s） | 1545.8 | 74.4% |
| 碰撞前静止：路线起步 | 41.2 | 2.0% |
| 碰撞前静止：由“超速刹车保持”刹停（plan 期望 ≥ 0.4 m/s） | 104.8 | 5.0% |
| 碰撞前静止：由停车 plan 刹停（期望 < 0.4 m/s） | 82.1 | 4.0% |
| 碰撞前行驶 | 263.4 | 12.7% |
| 碰撞后行驶 | 40.1 | 1.9% |
| 合计 | 2077.5 | 100% |

8 条 TickRuntime 单看：顶住 1395.0 s（87.2%）、碰撞前静止 101.5 s（6.3%）、碰撞前行驶 90.3 s（5.6%）、碰撞后行驶 13.3 s。
“超速刹车保持”指 Zoo PID 在 speed > 1.1 × 期望速度时刹车、AD-MLP 节奏把这一刹保持 0.5 s：碰撞前行驶中 23% 的规划窗口是它，
0.5 s 里车速中位数掉 3.1 m/s（从 5.2 到期望 2.8 的 0.17 倍），127 次里 63 次直接刹停；刹停之后 60% 的时间模型接着给停车 plan
（例如 1711 在 40.1 s：plan 要从 4.7 减到 1.9 m/s，0.5 s 满刹车把车刹到 0，模型随后 3.5 s 一直 “Keep distance to the lead
vehicle”）。它是真实的执行层失真，但量不大，见 7.4。

### 7.2 八条 TickRuntime 和两条 blocked

| route | 场景 | RC | 专家用时 (s) | 第一次碰撞 t / v | 对象 | 碰前 pace (m/s) | 按 pace 推算完成 (s) | 顶住 (s) | 顶住时油门 0.5 s 后车速 | 固定控制器同路线 |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 1825 | ConstructionObstacleTwoWays | 33.4 | 31.5 | 25.3 s / 5.3 | 施工警示牌 | 1.74 | 76 | 174.2 | 0.025 | TickRuntime |
| 1833 | ConstructionObstacleTwoWays | 37.6 | 31.5 | 31.0 s / 5.0 | 施工警示牌 | 1.59 | 84 | 168.5 | 0.028 | Completed 47 s |
| 1852 | AccidentTwoWays | 39.1 | 40.9 | 24.5 s / 0.5 | 警车（事故车） | 2.15 | 62 | 175.5 | 0.027 | Completed 45 s |
| 1956 | ParkingExit | 3.1 | 22.9 | 6.3 s / 1.2 | 停着的车 | 0.66 | 204 | 193.7 | 0.024 | Completed 55 s |
| 2084 | NonSignalizedJunctionLeftTurn | 42.6 | 16.8 | 20.5 s / 6.1 | 横穿车 | 1.64 | 49 | 177.0 | 0.000 | Completed 34 s |
| 2086 | NonSignalizedJunctionLeftTurn | 61.8 | 15.6 | 27.8 s / 5.2 | 横穿车 | 1.26 | 63 | 169.9 | 0.023 | 旧 run 崩溃 |
| 2091 | NonSignalizedJunctionLeftTurn | 53.2 | 14.0 | 32.1 s / 2.2 | 横穿车 | 0.97 | 79 | 163.0 | 0.069 | Completed 60 s |
| 2115 | NonSignalizedJunctionRightTurn | 57.0 | 14.0 | 24.8 s / 0.0 | 横穿车 | 1.35 | 53 | 173.2 | 0.028 | Completed 68 s |
| 2127 | OppositeVehicleTakingPriority | 60.8（blocked） | 15.9 | 42.6 s / 3.3 | 植被 | 1.14 | 67 | 60.2 | 0.024 | 旧 run 崩溃 |
| 2143 | OppositeVehicleTakingPriority | 61.7（blocked） | 15.8 | 21.6 s / 2.7 | 消防车 | 1.72 | 46 | 70.2 | 0.047 | 旧 run 崩溃 |

专家用时是 PDM-Lite 公开结果（`research/results/b2d-family/public/pdm_lite/merged.json`）同一路线的 `duration_game`。
“固定控制器同路线”是预注册控制器的旧全量 `full220-alpamayo`（13 条，同模型、同推理配置、TM seed 0；“崩溃”是那一轮的
基础设施失败，没有可比结果）。

碰撞分三类，都以顶住收尾：

| 类型 | 路线 | 撞前发生了什么 | 归因 |
|---|---|---|---|
| 横向避让没被执行 | 1825、1833 | 5 m/s 接近施工区，plan 3 s 处向左 2.7–8.3 m，Zoo PID 的 steer −0.02–0.05（aim = 0.5 s 点，或被 route target 替换） | 执行层（Zoo PID 横向规则） |
| 静止起步冲撞 | 1852、1956 | 静止时模型给 0.5–1.0 m/s 的蠕行 plan，Zoo PID 0.75 油门保持 0.5 s，0.5–1.2 m/s 撞上停着的车 | 执行层（油门 bang-bang + 保持）为主 |
| 路口决策 | 2084、2086、2091、2115 | CoT “Turn left since cross-traffic has cleared”，0–6 m/s 与横穿车相撞；2086、2091 1 s 内重新开起来，后来又撞标志牌 / 护栏顶住 | 模型决策；固定控制器下 2084、2091、2115 也撞了，但 1.7–6.8 s 内恢复 |

固定控制器在 1833 上执行了“向左绕”（碰撞前后 plan 3 s 处向左中位数 3.8 m），只在 8.1 m/s 时擦到一个锥桶，1 s 后回到行驶，
47 s 完成；Zoo PID 下同一条路线正面顶在警示牌上 169 s。

### 7.3 执行器对照

| 指标（碰撞前） | f1 全量（17） | smoke2 f1（11） | smoke2 f1+F2b（11） | 固定控制器（9） | 旧 Zoo，无 F1（119） |
|---|---:|---:|---:|---:|---:|
| plan 要求的 pace (m/s) | 2.40 | 2.28 | 2.33 | 2.16 | 3.37 |
| 实际 pace (m/s) | 1.99 | 1.83 | 1.92 | 1.69 | 2.76 |
| 行驶中 实际 / plan 距离 | 0.85 | 0.83 | 0.85 | 0.84 | 0.85 |
| 横向：2 s 实际 / plan（v ≥ 2、\|y@2s\| ≥ 1 m） | **0.04**（n 47） | 0.05（27） | 0.17（39） | **0.64**（37） | 0.13（392） |
| 超速刹车保持刹停次数 | 63 | 51 | 2 | 0 | 495 |
| 碰撞后顶住占总仿真时间 | 74% | 74% | 72% | 40% | 76% |
| TickRuntime 条数 | 8 / 17 | 4 / 11 | 3 / 11 | 1 / 9 | 36 / 119 |

pace 定义：每条路线第一次碰撞之前的规划窗口，plan 在窗口时长内走的弧长（或真值实际走的距离）之和除以总时长。
读法：plan 要求的 pace 在四种执行器下都是 2.2–2.4 m/s（旧 Zoo 的 3.37 是倒车 plan 被读成前进的假象），所以碰撞前的慢
主要是模型自己的节奏；执行层再丢 15%。F2b 把刹停从 51 次降到 2 次，但 pace 只从 1.83 到 1.92、顶住占比几乎不变，
它不是瓶颈。真正把执行器区分开的是横向执行（0.04 vs 0.64）和撞后是否顶住（74% vs 40%）。固定控制器那一列的
“停车 plan 占比”没有列，因为它的日志是未截断的原始 plan（倒车会被算成 go）。

### 7.4 逐项回答（a）（b）（c）

| 候选 | 证据 | 判定 |
|---|---|---|
| (a) 模型规划保守 | plan 要求的 pace 2.40 m/s vs 专家 5.22 m/s；32% 窗口是停车 plan（CoT：前车 52%、stop sign 23%、让行 14%、红灯 6%）；行驶中期望速度中位数 4.7 m/s、p90 8.4 m/s；行驶时车速是期望的 0.74 倍，模型一直在要求加速 | 解释 pace 差距的约 87%；不解释 TickRuntime |
| (b) 延迟、陈旧 plan、索引 | 同步仿真，plan 帧到施加控制 0 tick（最大 1 tick）；每 0.5 s 仿真时间一次规划；AD-MLP 节奏在规划时刻用新 plan 的 0.5–3 s 点，不存在按墙钟错位的索引 | 不是原因 |
| (c1) Zoo PID 横向（aim 取 0.5 s 点 + route target 替换） | 横向执行 4% vs 固定控制器 64%；1825 / 1833 正面撞施工牌 | **TickRuntime 的主因之一** |
| (c2) 撞后顶住、不会倒车 | 顶住时油门 0.75 → 0.03 m/s；固定控制器下同样的碰撞 1–7 s 恢复 | **TickRuntime 的直接原因（87% 时间）** |
| (c3) 油门 bang-bang + 0.5 s 保持的起步冲撞 | 1852、1956 以 0.5–1.2 m/s 撞停着的车；smoke2 里 f1 5 次、F2b 7 次（F2b 没有改善） | 次要原因 |
| (c4) 超速刹车保持 0.5 s | 23% 的行驶窗口，63 次刹停，104.8 s 静止；F2b 消掉它，pace 只 +4% | 真实但量小 |
| (c5) F1 截断 | 只作用于倒车段；行驶中的 plan 不变（前进分量全为正）；静止时把倒车读成停车，是第 5 节的语义修复 | 不是原因 |
| (c6) nav 文本 / egomotion 让模型以为自己停着 | 第 3 节已核对坐标、时间基、历史；起步静止 41 s / 17 条（每条 2.4 s）；刹停后 60% 时间模型继续给停车 plan，但 F2b 下刹停少了、停车 plan 占比没降（34% → 36%） | 不是主要原因 |
| min-speed 记录 | B2D `MIN_SPEED_INFRACTION: [0.7, 'unused']`，每个 checkpoint 都记，11%–1479% | 记账方式，不是信号 |

### 7.5 建议（待 main 决定；全量不续跑、不改预注册）

1. **P1 横向跟 plan（推荐）**：纵向保持 Zoo PID（官方 PID 的可比性），横向改成固定控制器的 20 Hz 路径跟踪（按里程计重投影
   plan）；或者更小的改动：Zoo 的 aim 改成按时间取 plan 上约 1.5 s 的点、关掉 route target 替换。两种都是新的偏离，要先写
   预注册再跑。预期：横向执行从 4% 回到约 60%，1825 / 1833 这类绕行场景、ParkingExit 可以通过；按固定控制器在 9 条共享路线的
   结果替换 6 条 TickRuntime，17 条 DS 44 → 约 58–62、RC 68 → 约 89（单 seed、小样本的粗估）。
2. **P1 的 re-smoke**：smoke2 的 11 条 + 本节 8 条 TickRuntime（去重后 16 条），验收看横向执行比例 ≥ 0.5、碰撞后 ≥ 10 s
   顶住段数、TickRuntime 条数，DS 只报告。
3. F2b 可以一起开（消掉刹停，和 UniAD / VAD 的调用方式一致），单独开不值得。
4. 撞后恢复（倒车脱困）属于驾驶启发式，不建议加进适配层；P1 之后先看顶住还剩多少。
5. 模型本身的慢（pace 2.4 vs 专家 5.2 m/s）是 zero-shot 的真实表现，适配层不该去改。

### 7.6 复现

```bash
# box, repo root (NumPy only; the PDM-Lite reference is research/results/b2d-family/public/pdm_lite/merged.json)
D=$DATA_DIR/runs/zeroshot-exam/b2d
$DATA_DIR/envs/carla/bin/python scripts/zeroshot_b2d_alp_speed.py f1=$D/full220-alpamayo-zoopid-f1 \
    s2f1=$D/smoke2-alpamayo-f1 s2f1f2b=$D/smoke2-alpamayo-f1f2b zoo=$D/full220-alpamayo-zoopid fixed=$D/full220-alpamayo \
    --ref research/results/b2d-family/public/pdm_lite/merged.json --out $D/diag-alp-speed
```

输出 `routes.csv`、`summary.json` 已放在 [research/results/zeroshot-b2d/alp-speed/](../../research/results/zeroshot-b2d/alp-speed/)
（`windows.csv` 4.8 MB，留在 box）。只统计有 `done/<id>.json` 的路线（暂停时被取消的 4 条不算）。
