# 预登记：PAI 上的 openpilot 式纵向（ACC，serve 侧，不训练），配对 box g10，2026-10-16

用户 10-16 批准的设计。本文件在任何读数（冒烟之外）之前写好并推送，之后只追加补记。参照与路径照
[2026-10-16-pai-gains10-prereg.md](2026-10-16-pai-gains10-prereg.md)（g10 = RSG-L18-s0、转向权重 10、默认 serve）。

## 0. 背景

- [d315](../../../research/decisions/315.md)：模型的纵向是自身运动的趋势跟随者（数值历史增益 γ 0.17，交接时 ≥ 16 m/s plan / 喂入 0.867，不动点约 3 m/s），
  「慢」来自交接即短再被回路放大；「人停了车没停」= 车不读「该停」（14 个非评估器 route_end 零分里 13 个停车点前 3 s 无减速意图）。
- [d293](../../../research/decisions/293.md)：过弯速度是人的 1.33 倍；人类转弯横向加速度中位 2.38 m/s²，入口速度中位 4.6 m/s，减速起点约入口前 26 m。
- [d285](../../../research/decisions/285.md) / [d309](../../../research/decisions/309.md)：交接时车以日志（人的）速度到达，模型 plan 慢 7–12 m/s，
  VCONT 1 s 抹平要求大减速，控制器以转向消耗距离（交接 spin）；v8 gate 已把它压到 0–2 个。
- 思路：横向留给模型（路径几何不变），纵向按 openpilot 的分工交给规则：巡航设定速度 = 交接时人的速度，弯道限速，前车用已有的 lead MPC，
  只有模型要停 / 要刹时才听模型。
- **风险（d315）**：此前每个加速修法都多出零分，因为模型不读「该停」；ACC 让车更快地到达灯口 / 停止线，而停车守卫只在模型自己要停时生效。
  **反证**：[d296](../../../research/decisions/296.md)（减速剖面几乎不改变转弯完成）、[d252](../../../research/decisions/252.md)
  （navtest 上一刀切速度缩放没有净正档，快 5 % 就新增 NC 失败）。本臂可能只是把慢的零分换成快的零分。

## 1. 臂（开关 `JEV_ACC=1`，`experiments/alpasim/lib/serve_fix.py` 第 (d) 项 + `pai_driver.py` 的钩子）

serve：`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`，`JEV_TRACK` 不设（v8 gate 默认），`controller.gains.rel_front_steering_angle_weight=10`，checkpoint RSG-L18-s0。
路径 = 模型 plan 的路径（几何不变）；速度剖面在 TG 网格（0.05 s，4 s）上重定时：

- **v_set**：会话第一个 `t_s ≥ 1.6 s` 的决策（force-GT 1.7 s 结束前最后一步，此时车速是人的）锁定的车速，作巡航设定速度保持整个会话；
  每决策取 max(v_hand, 当前模型 plan 自身 2–4 s 段速度)（交接时静止的场景否则设定速度为 0，永远不走；高速交接时模型更慢，不起作用）。
  （冒烟第一版用「交接后模型 2–4 s 段速度的滑动最大值」，在 a174faed 把 2.3 m/s 的交接推到 19.1 m/s：模型跟随自己更快的历史，最大值只升不降，是向上的棘轮；
  当前值没有棘轮，d315 的 g < 1。改动在冒烟之后、任何读数之前，见补记 1。）
- **v_curve**：AlpaSim route 消息（每条 10 点、4.2 m 间距、车前 42–80 m）按 d287 的方式作记忆合并（`rmem.Tracker.polyline`），所以弯一进入 80 m 就一直被覆盖；
  1 m 重采样，曲率 = 前后各 6 m 弦的航向差 / 6 m（上限 0.2 m⁻¹），`v² |κ| ≤ 2.4 m/s²`（d293 人类中位），向后递推以 **1.5 m/s²** 舒适减速，
  使车在弯处正好降到限速（15 m/s → 4.9 m/s 约在弯前 67 m 开始减；人从约 7 m/s 开始，26 m）。route 未覆盖处（会话前约 40 m）不限；路线离车 > 8 m 不用。
  模型自身路径上的弯仍有 v8 gate 的 VCURV（4 m/s²）兜底。
- **剖面**：从实测车速起，上升不超过 openpilot 巡航加速表 A_CRUISE（0 / 10 / 25 / 40 m/s 处 1.6 / 1.2 / 0.8 / 0.6 m/s²），下降不超过 2.5 m/s²
  （openpilot COMFORT_BRAKE），目标 min(v_set, v_curve(s))。
- **v_lead**：已有 `JEV_LEAD` 的 openpilot lead MPC（`LeadPlanner`，不改），e2e 候选取 ACC 剖面，结果取 min。
- **停车守卫**：模型**上一决策**的 plan（serve 前）2 s 弧长 < 2 m，或段速 (s1 − s4) / 1.5 s ≥ 1 m/s²（`vhist.py` holdlead 的两个释放）时，
  剖面 = min(ACC, 模型自己按 VCONT 服务的剖面)；模型重新走起来守卫即撤，回到 ACC（也提起慢起步）。
- 剖面超出 plan 末端时沿 plan 末端航向直线延长（300 m）。
- 每决策日志 `fix.acc`：v_set、v_hand、v_curve（1.5 s 处到达弧长的限速）、v_curve_min（前方 route 最小限速）、v_lead、v_model、v15、
  胜出项 won（set / curve / model / lead；离该项 > 0.3 m/s 时记 ramp，near 给最近项）、guard（stop / brake / 空）、route 覆盖米数。
- 关时逐位相同：`experiments/alpasim/scripts/acc_check.py`（4 组开关 × 3 000 个随机决策，新旧 serve_fix 输出与日志 0 差异；驱动只在 `FX.ACC` 时走新分支）。

## 2. 冒烟（Tokyo，不计入判据）

6 个场景（90ab8c24 交接 26.7 m/s、49bcab6f 18.5 m/s、054b5901 L-STOP 考题、fc9fde87 排队、2f227f2c 与 a174faed 转弯），gains 10，
卡 0 ACC、卡 1 同表 ACC 关。查：不崩、每个项都在预期处胜出（高速直路 set、弯前 curve、跟车 lead、停车 model）、guard 有开有关。

## 3. 读数（box，与 g10 同一路径）

`pai_box.sh submit $DATA_DIR/runs/route_follow/pai_g10/acc RSG-L18-s0`，`CHUNKS_FROM=$DATA_DIR/runs/alpasim/pai_box/tokyo/r281/rsg/chunks`（与 g10 同 8 个 chunk 表），
`WIZ=controller.gains.rel_front_steering_angle_weight=10`，`SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1"`，PRIO 13，排在 gains-10 的 job 后。
报告 job（CPU，`--after` g10 与 acc 的 16 个 job）：
`rf_pai281.py --rows g10=$O/g10 acc=$O/acc --lists exam=hard_exam.txt control=control.txt stop17=route_end_scenes.txt --vs g10 --ref g10 --out $O/acc.md`
（`O=$DATA_DIR/runs/route_follow/pai_g10`；stop17 = d315 的 17 个 `cor_route_end` 场景）。只比两边都有分的场景。

## 4. 判据（读数之前写死；对 g10，按场景配对 bootstrap 10 000 次）

ACC 进入提交配置当且仅当全部成立：
1. 281 全集配对均分 CI 下界 > −0.01 且点估计 > 0；
2. 考题 72 点估计 ≥ −0.01；
3. 零分 ≤ g10 + 2；
4. 交接 spin ≤ g10 + 1。

**主风险单独报告**：停车类零分（「人停了车没停」）——ACC 相对 g10 新增的每个零分逐个列出（类、场景、是否在 stop17、是否考题），
并看日志判是否「人在此停车、车没停」；stop17 子表的均分与零分。即使 1–4 全过，新增停车类零分 ≥ 3 也写明，提交与否由用户定。
其余描述：各项胜出比例、守卫开启比例、被追尾数、plan 2 s 欠量、转弯完成。

## 5. 跨主机一致性（描述性，不作闸）

免费 CPU 检查，供以后混用 Tokyo / box 读数参考：box `recovery_envelope/pai/RSG-L18-s0`（gains 20、v8 gate、native、code b7bcd3d3）对
Tokyo `v2pai/RSG-L18-s0-gate`（gains 20、v8 gate、docker、10-15，chunk 表与 r281/rsg 逐个相同；r281/rsg 本身是旧的 `JEV_TRACK 0` 路径，不可比），
`rf_pai281.py` 逐场景：配对均分差与 CI、逐场景分数相同 / 零分一致的比例。写作补记，不影响本臂判据（本臂只与 box g10 配对）。

## 6. 成本

box：8 chunk × 约 52 分钟 ≈ 7 job 小时，排在 gains-10 之后（预计 17:00–17:30 CST 开始，约 1–1.5 小时墙钟）；Tokyo 冒烟约 15 分钟两卡。
证据等级上限：1 seed、281 场景、按场景配对。

## 补记 1（冒烟与跨主机一致性，读数之前，2026-10-16 16:15 JST）

- 先修了一个与本臂无关的既有故障：TTS 提交（00580b53）里 `TS = S.TS` 让 Tokyo 镜像（旧 sh30_driver）的 PAI driver 在 import 时崩溃，
  改为 `getattr(S, "TS", None)`（81d39064，box 上逐位不变）。
- 冒烟 v1（fc245afc，v_set 取滑动最大值）：不崩；a174faed 交接 2.3 m/s 被推到 v_set 19.1 m/s（向上棘轮）→ 改成当前值（bed9cb8e，见第 1 节）。
- 冒烟 v2（bed9cb8e，Tokyo 卡 0，gains 10，6 场景，486 s；同表 ACC 关在卡 1 跑的是 v1 时刻的同一代码路径，关时与新代码逐位相同）：

| 场景 | 类型 | ACC 分 | 关 分 | 胜出项（决策数） | guard |
|:--|:--|--:|--:|:--|:--|
| 90ab8c24 | 交接 26.2 m/s | 1.000 | 0.526（被追尾，车速掉到 9.6） | set 199 | 无 |
| 49bcab6f | 交接 20.0 m/s | 1.000 | 1.000 | curve 68、set 58、ramp 72（v_curve 最低 13.3） | brake 1 |
| a174faed | 转弯 | 0.507（被追尾） | 0（走廊） | set 127、lead 57（v_curve 最低 4.2） | brake 7 |
| 2f227f2c | 转弯、交接 1.3 m/s | 0（走廊） | 0（走廊） | set 109、model 44、lead 8 | stop 54 |
| fc9fde87 | 排队 | 0（走廊） | 0（走廊） | set 117、model 46、curve 9 | brake 52 |
| 054b5901 | L-STOP 考题 | 0（走廊） | 0（走廊） | set 181、model 12 | brake 12 |

  每一项都在预期处胜出。**主风险已经能看到**：054b5901 关时车在线前停下（末速 0.1），ACC 以 14.7 m/s 开过去（模型只有 12 个决策刹车，守卫不够），两者都是 0 分。
  6 个场景不是读数，不判。
- 跨主机一致性（第 5 节，描述性）：`experiments/route_follow/results/pai_acc/parity.md`。box 对 Tokyo 的配对均分差 +0.005 [−0.008, +0.019]（box 0.5806 / 72 个零分，
  Tokyo 0.5860 / 71），考题 −0.001 [−0.023, +0.021]，对照 +0.001；281 个场景里 162 个分数逐位相同、265 个差 < 0.05，零分一致 276 / 281（只 box 零分 3 个：05855d57、
  1bfad87c、e1c6ae9f；只 Tokyo 2 个：229791a5、94c1d722），相关 0.961。读法：两台主机的均分可以混用（差在 ±0.02 以内），逐场景比较不行（约 2 % 的零分会翻转）。

## 补记 2（读数之后，2026-10-16 晚；主风险报告，第 4 节要求）

结果全文 [results/pai_acc/read.md](../results/pai_acc/read.md)，报告 `results/pai_acc/report.md`（登记的报告 job）与 `report_logwindow.md`（加日志窗转弯代理）。

- **判据 1–4 全过**（对 g10 配对，279 个两边都有分的场景）：均分 0.6785 对 0.5733，+0.105 [+0.066, +0.145]；考题 71 +0.045 [−0.045, +0.131]；
  零分 76 对 76（281 直接计数；共同 279 上 74 对 74）；交接 spin 0 对 0。两边各有 1 个 RPC 错误零分（g10 13d646af、acc 054b5901，
  "Expected all tensors to be on the same device"），所以共同集 279、考题 71。
- **报告 bug 核查**：首表 ACC 与 g10 的零分分类 74 / 9 / 16 / 49 相同，直接计数确认无误，是巧合（新增 15 个 = 碰撞 2 / 出路 4 / 走廊 9，
  修好 15 个 = 3 / 3 / 9，另有 8 个两边都零但类别不同），脚本不改。另发现转弯代理把车开过日志终点之后（评分已不看）的转向也算进去，
  见下条；修在 b9fbd62f（加日志窗列，旧列保留）。
- **停车类新增零分 3 个（达到登记的 ≥ 3，写明）**：444fca69（对照）、47db2eb6（对照）、9b6c68b4（考题）。三个都是人在日志终点停下
  （47db2eb6、9b6c68b4 在 STOP 线，444fca69 在转弯入口），ACC 以 6–9 m/s 经过，起作用的是弯道限速或巡航设定速度，停车守卫没有触发
  （只有 47db2eb6 晚到的 brake 守卫）。g10 在这三处也不停，只是太慢，20 s 内差 6–24 m 没开到停车点，所以有分。与 d315 的
  `cor_route_end` 同一机制；都不在 stop17（gains 20 的 RSG 同样没开到）。抵消项：stop17 里的 cbd87234 被修好（弯道限速把终点处车速压到 4.8 m/s，g10 14.2）。
  对照列表的 2 个零分就是这两个停车类。
- 其余 12 个新增零分：3 个高速横向漂出（32–38 m/s 保持人的车速，13–15 s 出走廊，g10 慢或被追尾没开到）；3 个弯里更快
  （curve 胜出但仍快于 g10、慢于人）；4 个与 g10 同站点车速差 < 1 m/s 的近同状态翻转；2 个转弯（下条）。
  修好的 15 个：7 个慢致零分（以交接车速巡航修好）；5 个 g10 比人快（弯道 / 终点限速修好）；3 个 ACC 先被追尾、没开到 g10 的失败点（不算真修好）。
- **转弯代理下降的原因**：整段轨迹口径下 ACC 左 5 / 9、右 1 / 3（g10 9 / 9、3 / 3），其中 4 个是代理假象（3ffad395、4e8c8f9e、51400458、9d9052bd，
  日志只转 1–27°，g10 的 67–81° 是开过日志终点之后转的，两边都 1.0 分）。日志窗口径 8 个转弯场景：**ACC 左 4 / 5、右 2 / 3，g10 5 / 5、3 / 3**。
  真损失两个：abd45a30（左转 92°）ACC 更早更快到弯（124 m 处 6.2 对 g10 3.9、人 3.5 m/s），路径向右走宽（−5.3 m），弯前出走廊；弯道限速确实生效（v_curve_min 5.3、route 覆盖），
  但 2.4 m/s² 的目标高于人在此处的车速；同站点模型 plan 4 s 转角 49° 对 58°，即 d310 的速度相关 plan 曲率。e0f6ebe1（右转）弯本身完成（−87°，弯中比 g10 慢），
  出弯向设定速度加速时向左漂 +4.2 m 出走廊（g10 同处 +2.9 m），边际。没有一个是「场景在弯前结束」。
- 描述：55 495 个决策里胜出 set 0.47、lead 0.31、ramp 0.09、model 0.07、curve 0.06；守卫开 0.19（brake 0.10、stop 0.09）；被追尾 40 对 75；
  plan 2 s 欠量考题 −0.11 对 −0.72 m；stop17 均分 0.0625 对 0、零分 15 对 16。
- ACC 与 holdlead 没有一起读过。提交与否由用户定。
