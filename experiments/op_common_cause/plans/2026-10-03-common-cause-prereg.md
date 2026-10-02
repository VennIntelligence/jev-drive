# 共性假设检验（lane B，2026-10-03 夜）：预登记

写于跑任何新数之前（2026-10-03 02:55 JST）。来源：tmp/2026-10-03-overall-plan.md「第 3 层」与任务 B；背景 decisions 66、88、90。
不训练。模型：openpilot Cinque 原生（TensorRT，`scripts/op_lb.py` 的 backend 与 action_t），只改输入。

## 假设

H：openpilot 的计划在 (低速) × (历史短或不正常) × (没有指令) 时不可靠，且这个效应在真实数据和 CARLA 上同号、量级相近。
若成立，第 3 层的「同画面不同历史 → 同计划」「同画面不同指令 → 不同计划」两种配对可以在两个域上一起调；否则分开调。

## 三个域（每个 domain 一个样本表，预先抽样、固定 seed 0）

| domain | 帧 | 历史来源 | 专家未来 | cluster |
|:--|:--|:--|:--|:--|
| `nav`：navtrain（`runs/op_lb/lb_navtrain` 的 3 000 token，每 command 1 000） | CAM_F0 渲染 | 榜单协议：4 张 2 Hz 关键帧 + GIMM 补到 5 Hz（op_lb 的 `_steps` 原样） | 人类 8 × 0.5 s | log |
| `wod`：WOD-E2E val 的 rater + extra（1 437 帧） | 前三相机渲染 | 真实 10 Hz 流取偶数帧（5 Hz） | 人类 20 × 0.25 s | sequence |
| `carla`：P4 Bench2Drive 路线（`processed/carla_p5`，source = p4，151 条路线） | P4 的 Waymo 式三相机 | 真实 5 Hz 流 | BehaviorAgent 20 × 0.25 s | route |

抽样：速度档按 t0 速度 `stop` [0, 0.5)、`low` [0.5, 3)、`mid` [3, 8)、`high` ≥ 8 m/s。nav：stop / low / high 全取，mid 随机 600；
carla：每档最多 stop 400、low 600、mid 600、high 全取，且要求流里 t0 前至少 1.6 s 的帧（不足的不进样本；`long` 另要求 4.8 s，不足则该样本没有 `long`）；
wod：全取（`long` 需要 4.8 s 历史）。command：nav 用 driving command，wod / carla 用 intent（左 / 直 / 右）。

## 统一协议

20 Hz 步、从零状态开始，t0 读最后一步的输出。正常历史 = t0 前 1.5 s 共 31 步，每步用时间 ≤ 该步的最新 5 Hz 帧（nav 原样复用 op_lb 的 schedule；
未扰动的 nav 重跑必须与缓存的 `plans/gimm@cinque.npz` 一致到 < 1 cm，否则整组作废）。traffic convention 按地图（nav）/ (1, 0)。

## 因子

历史 H（对同一组帧做输入层的改动）：

- `normal`：上面的 1.5 s。
- `long`：同样的流取 4.8 s（97 步，填满 Cinque 的 feature queue）。只 wod / carla。
- `repeat`：31 步全是 t0 帧（静止历史，等于「刚起步、之前一直停着」的画面）。
- `single`：只有 t0 帧 4 步（历史被丢掉）。
- `rotL` / `rotR`：正常历史的每一帧在时间 t（< 0）处用 `op_interp.warp_frame` 绕后轴转 ψ(t) = ±r·t 渲染（r = 10°/s，左为正；
  即自车看起来在 1.5 s 里向左 / 右转了 15°，而 t0 帧不变）。

指令 D（只在 command 为左 / 右的帧上，方向取 command）：`off`；`pulse` = turn desire 从 −1.0 s 起保持（OPModel 的单个上升沿）；
`sustained` = 同上但每步都重新给上升沿；`lc` = laneChange 从 −1.0 s 起（第 66 条选中的臂）；`wrong` = 反方向 turn 从 −1.0 s 起（操纵检查）。
直行帧只跑 `off`。

交叉：直行帧 H 全部 × `off`；转弯帧 H 全部 × {off, pulse, sustained}，另加 `normal` × {lc, wrong}。

## 输出与度量

每次运行存 openpilot plan（33 点位置、yaw）。换到后轴系（`wod_zeroshot.openpilot_to_wod` 的 lever 换算，x 前 y 左），插到 0.25 s 网格。
记 y(t)、ψ(t)、x(t) 为计划在 t 秒的横向、朝向（plan_yaw）、纵向；y*(t)、x*(t) 为专家未来。主时刻 t = 3 s，副 t = 2 s。

- 偏差：`lat3` = |y(3) − y*(3)|（m）。
- 自洽：某变体对 `normal` 的 |Δy(3)|、|Δψ(3)|。
- E1 历史跟随增益：G = (ψ_rotL(3) − ψ_rotR(3)) / 2（度）；副量 G_y = (y_rotL(3) − y_rotR(3)) / 2（m）。G > 0 = 计划跟着假历史转。
- E2 静止历史：Δlat3(repeat − normal)；同时报 Δx(3)（静止历史是否让计划不动）。
- E3 丢历史：Δlat3(single − normal)。
- E4 历史长度：Δlat3(normal − long)（> 0 = 1.5 s 比 4.8 s 差）。只 wod / carla。
- E5 指令：转弯帧上 Δlat3(pulse − off)、(sustained − off)、(lc − off)（< 0 = 指令有用）；操纵检查 S = sgn(cmd)·(y_pulse(3) − y_wrong(3))（> 0 = desire 能把计划推向指令侧）。
- E6 交互：转弯帧上 G 在 `off` 与 `sustained` 之间的差（指令能否压住假历史）；静止历史下 Δlat3(sustained − off)。
- E0 基线：`normal` × `off` 的 lat3 按速度档。

每个效应按域、按速度档报均值与 95% cluster bootstrap CI（`jevdrive.stats`，B = 10 000，cluster 见上表）；
速度依赖 = `low` 档减 `mid` 档的差（两档独立样本，各自 cluster bootstrap 后差的 CI，用同一组 bootstrap 种子分别重采后相减）。

## 跨域一致的判据（预先定好）

对每个效应 E 与每对域（主对比 nav ↔ carla；干净对比 wod ↔ carla，两者都是真实连续流、协议完全相同）：

- **一致**：两域的 CI 都不含 0、同号，且点估计之比在 [0.5, 2]。
- **同号不同量**：两域都显著、同号，比值在 [0.5, 2] 之外。
- **只在一个域**：一个显著，另一个 CI 含 0。
- **相反**：两域都显著、异号。
- **都无效**：两个 CI 都含 0。

主效应三个：E1（G，合并 v ≥ 0.5 m/s 的帧）、E2（repeat 的 Δlat3，合并 v ≥ 0.5）、E5（pulse 的 Δlat3，转弯帧，合并 v ≥ 0.5）。
速度依赖两个：E1 与 E2 的 low − mid 差。

结论规则：

- 「同画面不同历史 → 同计划」这种配对可以两域一起调：E1 与 E2 在 nav ↔ carla 都是「一致」或「同号不同量」，且 wod ↔ carla 不出现「相反」。
- 「同画面不同指令 → 不同计划」可以一起调：E5 或 S 在两对比较里都是「一致」或「同号不同量」。
- 共性假设（三个条件同一个病）成立：两域都有 G > 0 且 E1 或 E2 的 low − mid 差显著为正（低速更差），并且 E5 < 0（有指令时变好）。
  只成立一部分时按实际成立的写，不扩大。

## 操纵是否到达模型（看到某个因子无效时必须先查）

- 每种历史变体：输入张量与 `normal` 的平均绝对像素差 > 0（逐变体抽 20 个样本记录），以及 t0 处 `hidden_state` 与 `normal` 的余弦距离。
- desire：抽样记录送进 ONNX 的 desire pulse 张量非零的步数（pulse 应为 1 步，sustained 为 21 步）。
- `rotL` / `rotR` 的符号：在 carla 的真实转弯流上，用同一个 warp 把 t−1 s 的帧按 ψ 网格转过去，与 t0 帧的像素差最小的 ψ 应与真实偏航变化同号（单元检查，先跑）。

## 不做

不训练；不跑新的 CARLA；不在 navhard 合成场景上取样（那是考题）。HUGSIM 帧不在本次（没有现成的连续流 + 专家未来）。
