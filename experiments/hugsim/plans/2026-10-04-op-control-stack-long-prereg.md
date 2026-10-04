# openpilot 纵向控制路径接在 d118 横向路径之上：HUGSIM（预登记）

2026-10-04 写，在本臂的任何闭环运行之前（诊断部分是 CPU 回放，已做，见第 2 节）。接第 118 条（openpilot 自己的横向路径：打转 10 → 0，非打转 HD −0.009 [−0.058, +0.042] 线不过，代价来自更多卡在停车计划的 run；纵向仍是 iLQR 跟踪计划位置）。前一个预登记：[2026-10-05-op-control-stack-prereg.md](2026-10-05-op-control-stack-prereg.md)。

## 1. 源码路径（openpilot master ec95db3f，opendbc 35f7e081；本地 tmp/opsrc，不入库；逐字移植的函数标 VERBATIM）

| 环节 | 源码 | 常数 / 行为 |
|---|---|---|
| modeld | `selfdrive/modeld/modeld.py` `get_action_from_model` | 有 `action` 头：desired_accel = action[1]；stop = `should_stop(v_ego, desired_accel)` = v < 0.3 且 a < 0.1，用**未平滑**的原值；desiredAcceleration = smooth_value(raw, prev, `LONG_SMOOTH_SECONDS` = 0.3)，20 Hz（DT_MDL 0.05） |
| long_action_t | 同上 | longitudinalActuatorDelay + 0.3 + 0.05 + 0.025；非混动 Toyota 的 longitudinalActuatorDelay = 0.15（interfaces.py 默认；混动 0.05）→ 0.525，与考试喂的 (0.275, 0.525) 一致 |
| 纵向规划 | `longitudinal_planner.py` | 实验模式（e2e 模型的跑法）：a_target = min(mpc, cruise, e2e)；无前车雷达、设定速度高于车速时 mpc 与 cruise 为正，e2e 胜出；output_a_target = clip(e2e, ACCEL_MIN, ACCEL_MAX)，output_should_stop = 任一候选的 should_stop（只有 e2e 的会触发）。**限定：模拟器里没有雷达，MPC 的前车保护不在环内** |
| LongControl | `selfdrive/controls/lib/longcontrol.py` | 状态机 off / stopping / pid（`long_control_state_trans` VERBATIM，不含 brake_pressed / cruise_standstill）；Toyota TSS2 的 kpV = 0、kiV = 0（`interfaces.py` 默认，`toyota/interface.py` 不改）→ pid 态输出就是前馈 a_target；stopping 态：输出 = 上一次输出，若 > stopAccel（−2.0）则 min(·, 0) − 1.0 m/s² 每秒的斜坡；pid → stopping 当 should_stop，stopping → pid 当 not should_stop。无单独的起步逻辑 |
| 加速度限制 | `toyota/values.py` CarControllerParams | ACCEL_MIN −3.5，TSS2 的 RAISED_ACCEL_LIMIT → ACCEL_MAX 2.0 |
| 车 | `toyota/carcontroller.py` | 请求先按 ±4.0 m/s² 每 3 个控制帧（jerk 4 m/s³）限速（ACCEL_WINDUP / WINDDOWN_LIMIT），再进 PCM + long_pid 环（kiBP [2, 5] → kiV [0.5, 0.25]），使 a_ego 跟上请求 |

**车辆响应模型**（与 d118 的横向同一原则：用 openpilot 自己对闭环的模型，不加没有数据的动力学）：实现加速度 = 限速后的请求延迟 longitudinalActuatorDelay（0.15 模型秒，100 Hz tick）。PCM / long_pid 环的内部动力学不建模。

**单位**。模型时钟比仿真器快 1.25 倍（速度喂 1.25 v，0.25 s 一步 = 4 个模型步 = 0.2 模型秒）。控制器在模型坐标里跑（v_m = 1.25 v，a_m = 1.25² a_sim，tick 0.01 模型秒，20 tick 一步），输出换回仿真单位：a_sim = 一步内平均实现 a_m / 1.5625，即 HUGSIM `velo += acc * dt` 要的速度增量；再夹 a ≥ −v / dt（车不倒车）。Toyota 的 −3.5 / 2.0 限制按 openpilot 在模型坐标里施加，换算后仿真单位 −2.24 / 1.28。20 Hz 的平滑：服务器每个仿真步只回最后一个模型步的 action，所以一步内按常数输入迭代 4 次 smooth_value（等效每仿真步 α = 0.486）。实现：`lib/op_ctrl.py` `OpLongitudinal` + `hugsim_acc`。

## 2. 诊断（CPU，已做；复现 `experiments/hugsim/scripts/opctrl_long_diag.sh`、`opctrl_long_diag.py`、`opctrl_long_diag_acc.py`）

数据：d118 的 `cinque-opctrl`（24 个 max_steps）对同日基线 `cinque-fixed-base3`（15 个 max_steps）的 `zs_steps.jsonl`；CPU 重放（`opctrl_replay.py` 加 `acc_act`）前 100 步，把 action 头加速度按考试同样的喂法重算。

1. **新增的 9 个卡死 run 里只有 2 个是「基线完成、本臂卡死」**（3000_3200-medium：基线 51 步完成；0418-hard：基线停在同一位置 24 步后在第 51 步自己重新起步，本臂不起步）。另 7 个（0528、152217047339、8440_8640、102751446607、0041、2510_2710-hard、034-hard-01）基线是 17–119 步内崩溃（bg / fg），本臂不崩，于是走到了和基线那 15 个卡死 run 同一种状态：车在第 13–65 步停下，模型在停车计划。这 7 个里 6 个 HD 上升（如 034-hard-01 0.14 → 0.79、8440_8640 0.47 → 0.78），1 个下降（2510_2710-hard 0.087 → 0.003）；卡死在这里多数是「活下来了」，不是退步。
2. **卡死的原因是模型：停车计划，不是 iLQR 跟踪近零计划。** 24 个卡死 run 里车在 ≥ 0.3 m/s 之下的步占 84–97%，`straight_stop` 标志占同样比例；停下后计划 1.2 s 的速度中位 0.00–0.02 m/s，6 s 处 0.05–1.6 m/s（多数 < 0.7）。action 头加速度（模型单位）：停车阶段中位 −0.05–0.05，最大 0.08–0.24，≥ 0.1 的步只占 0–18%（中位约 7%）。也就是 action 头和计划都说「别动」，不是计划说走而 iLQR 不跟。
3. **停车位置差 1–2 m 与脚本化攻击者**：0418 两臂停在几乎同一处（base 10.1 m，opctrl 12.1 m；航向 0.7 vs 0.5 度），模型都要停。基线之所以起步，是第 50 步前后速度从 0.02 → 0.03 → 0.04 m/s、前车速度读数同时从 −0.2 升到 +0.6 m/s，模型的 1.2 s 速度计划 0.04 → 0.26 → 0.96 → 2.4，自己放大（车速本身是模型的输入，从 0 附近的微小爬行开始正反馈）；本臂车速精确在 0.01 m/s、前车读数一直 ≤ 0，没有这个种子。3000_3200：本臂在第 21–24 步前车 lead_prob 0.47 → 0.95（前车 lx 20 → 11 m、lv 2.3 → 0），车减速停在 21.6 m，之后 lead_prob 一直 0.9–1.0；基线同一时刻前车在 23–30 m 以 5–6 m/s 开走（lead_prob 0.1–0.3）。（未进一步分离：本车到达前车刹停那一段的时刻与场景里前车的时间脚本不同，是场景侧的差别还是本臂车速轨迹的差别，这里不下结论。）
4. **openpilot 真栈在这里会怎样**（开环：把重放的 action 加速度和记录的车速喂给 `OpLongitudinal`）：停车阶段状态机在 stopping，只有 action 加速度 ≥ 0.1 的步（上面的 0–18%）进 pid，此时平滑后的 a_target 只有 0.05–0.1，实现加速度最大 0.07–0.14 m/s²（仿真单位），即每步 ≤ 0.03 m/s 的速度增量，没有起步逻辑把它放大。这个量级和基线 0418 起步前的爬行（0.02–0.04 m/s）同级：所以 openpilot 纵向路径**有可能**提供那颗「种子」让模型重新起步，也可能不够；这就是本臂要回答的问题。预测（事先写）：0418 和 3000_3200（基线完成）这类 run 里，卡死减少；基线自己卡死的 15 个不变。

结论：d118 的「卡死代价」主要是（a）7 个被基线崩溃遮住的、模型本来就要停的场景，（b）2 个丢分场景是前车脚本 / 微爬行的偶然；不是欠转，也不是 iLQR 跟踪的人为产物。纵向换成 openpilot 路径能改的只有「微爬行种子」和起步 / 刹车的加速度形状。

## 3. 臂 `opctrl_long`（唯一主臂，参数事先定，不调）

- 横向与 d118 完全相同（`OpLateral`，delay 0.25 仿真秒）。纵向：agent 把模型的原始 action 加速度（服务器 `accel`）作为倒数第二行 `[accel, 2e9]` 随计划发给仿真器，tree `opctrl_long`（= opctrl + `patches/hugsim/optional/op-ctrl-long.patch`，env `OP_CTRL_LONG`）剥掉它，经 `OpLongitudinal` 给 `acc`，iLQR 的加速度被丢弃（iLQR 仍算，不用）。agent 选项 `op_long`。计分用的 planned_traj 仍是原计划（`straight_stop` 等对计划的处理照旧，不进控制）。
- 模型调用、输入、action_t 与考试完全相同。
- 参数不调：RULE `{}`、LRULE `{}`（默认值即第 1 节的源码值）。

## 4. 运行（HUGSIM，64 场景，Cinque，单次，一张卡，租约 `opctrl-long`）

1. 烟测 2 个场景（`lowsel_smoke.txt`：scene-0013-medium-00 基线打转、scene-3000_3200-medium-00 d118 丢分最多），不计分：检查 sim.log 有 `op_long` 行、加速度行被剥掉、不崩、起步能走。
2. 全部 64 个。
3. 同日基线重跑 `cinque-fixed-base4`（tree fixed，同一卡同一 server），与 d118 里的 base3 一起作为「同日重跑」。

## 5. 线（预先定，不看结果后改）

- (i) 64 个里打转 ≤ 2（打转、起步事件、HD 配对差的定义与 `opctrl_report.py` 完全相同；脚本 `opctrl_long_report.py`）。
- (ii) 考试基线不打转的 54 个场景，HD 配对差（场景 bootstrap 95% CI）下界 > −0.02，对考试基线；同时报对 base3、base4 的配对差，对 d118 `opctrl` 的配对差。
- (iii) 卡死数（end = max_steps）：报 opctrl_long 对考试基线 15、同日基线、d118 opctrl 24 的数；不设通过线，判读见下。另报「基线完成而本臂卡死」的场景（d118 里是 3000_3200、0418）和 7 个「基线崩溃、本臂卡死」的场景是否仍卡。
- 另报：RC、fg/bg/off_route、起步事件打转、闭环 c（同一读数）、`op_long` 日志里 stopping 态的占比。

## 6. 判读（事先写）

- (i)(ii) 都过：openpilot 横纵向整套换进两个闭环榜的默认方向，下一步 B2D（另一车道在跑 B2D，本条不跑）。
- (i) 过、(ii) 不过：看卡死数和丢分场景。卡死数不降（≈ 24）：纵向路径也不解卡死，卡死是模型 + 场景属性（诊断第 2 节），不是 iLQR 跟踪的人为产物；报告 d118 的「待定」不变，换新问题（停车计划本身）。卡死数降但 HD 仍不过：看是否来自 Toyota 限速（1.28 m/s² 仿真单位）导致的起步变慢 / 追尾——报告，不调。
- (i) 不过：纵向路径让起步更猛，报机制。
- 不调参数、不追加臂；要加先在本文件追加并提交。

## 7. 已知偏离（写在前面）

模拟器没有雷达，MPC 前车保护不在环内（实车实验模式有）；cruise_standstill / brake_pressed 不建模；PCM 内部环用纯延迟代替；加速度限制在模型坐标里施加（仿真单位 1.28 / −2.24，比 iLQR 的 ±3 小）；一步内 4 个模型步只用最后一个 action。
