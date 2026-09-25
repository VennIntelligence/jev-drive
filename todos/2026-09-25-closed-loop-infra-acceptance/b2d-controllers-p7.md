# B2D 控制器复验：P6、P7 过验收（与 P5 复验同一协议）

状态: 预注册（2026-09-26，写于任何 P6 / P7 臂运行之前）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)；同协议的上一批：[b2d-controllers-p5.md](b2d-controllers-p5.md)
**在 GPU box 上跑的只有本文的验收臂**（专家 replay、20 条路线）；P7 登记的 L1 与 TFv6 / TCP 闭环评测由 Tokyo 跑，不在这里。

## 问题

P5 在验收里横向合格、纵向不过：每次起步比专家晚约 0.5 s，落后 2.6–3.3 m。之后 tfv6-controller 线又出了两版
（[protocol-v2](../2026-09-23-tfv6-controller/controller-eval-protocol-v2.md) 末段）：

- **P6** = P5 + 终点逼近只在实测 plan 周期 ≥ 0.2 s 时启用（`terminal_approach_min_period_s 0.2`）。本验收的 plan 是 1 / 2 / 5 Hz，
  周期 1 / 0.5 / 0.2 s，都 ≥ 0.2 s，所以按设计 **P6 在这里应与 P5 行为相同**（5 Hz 正好在边界上，实测周期的浮点误差可能让它
  偶尔不启用）。P6 仍然跑，一是它是 P7 的直接基线，二是顺便核对 P5 结果在同一台 box 上的可复现程度。
- **P7** = P6 + `accel_request_max_mps2 = 2.0`：正向加速度命令上限 2 m/s²，刹车不限（为 TFv6 起步时的不可行 plan 设计）。
  **预期**：专家 PDM-Lite 起步和再起步时的加速度可能超过 2 m/s²，P7 的上限可能让 P5 诊断出来的起步滞后变大。
  这正是这个测试要量出来的东西，不据此调参。

配置原样使用：`todos/2026-09-23-tfv6-controller/controller-eval/P6.json`、`P7.json`（commit 3331941）。
接线与 P5 相同（`controller_preset: pursuit`，pose-adapter 键进 `PoseFilter`）；`scripts/test_infra_ctl_p5.py` 已扩展到 P5 / P6 / P7，
三者经考试 agent 构造与 `pursuit_from_config` 逐位相同。

## 协议与判据（与 P5 复验完全相同）

时间索引、平滑追上的 replay（`replay_plan: time`），同 20 条路线、TM seed 0，专家 E-a / E-b 复用第一轮，A1–A5 门槛不变
（A4 = 90.5）。每臂分别判，一个控制器只在它通过的节奏下被验收。

| 臂 | 控制器 | plan 节奏 |
|---|---|---|
| p6x2 / p6x5 / p6x1 | P6 | 2 / 5 / 1 Hz（plan_every 5 / 2 / 10） |
| p7x2 / p7x5 / p7x1 | P7 | 2 / 5 / 1 Hz |

只报告、不作判据：与 P5 复验同一组指标，另加配对差（逐路线均值 + route bootstrap 95% CI）：P6 对同节奏 P5，P7 对同节奏
P6 和 P5，P6 / P7 对 f2t；纵向时间滞后（`scripts/infra_ctl_lag.py`：起步延迟、时间滞后中位数）。P5 与 f2t 用上一批已有的结果
（同一台 box，GPU 2），不重跑。
不通过时只诊断（横向 / 纵向、哪类场景），不调参。

## 运行与成本

`ACCEPT_OUT=$DATA_DIR/runs/infra-accept/b2d-ctl-v2`（与 P5 批同目录，新臂新子目录），GPU 1 与 GPU 2 各 6 个 CARLA server
（每臂 2 个 runner，index 700–733），`--client-threads 8`，约 30 核。GPU 4 此刻有 FASTPERC 的延迟测量（对争用敏感），不放 CARLA。
估计：P5 批一臂 20 条约 85–100 worker-min，6 臂约 550 worker-min，12 个 runner 约 50–60 min。
