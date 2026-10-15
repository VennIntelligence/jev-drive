# 计划：LON-INVENTORY，纵向包络行的存量盘点（不训练、不仿真、只 CPU），2026-10-16

为什么：下一轮 on-policy 训练要把纵向和横向一起教。设想是策略**自己状态**上的非对称空间速度包络：有理由（红灯 / 停止线 / 弯 / 前车）的地方给上界
v ≤ v_ref(s) + m，空路上给下界 v ≥ v_ref(s) − m；s = 沿路线的站位，v_ref = 规则专家在该站位的速度（rule_expert.Scene.profile），没有专家包时退回人类日志同站位速度。
造行之前先数有多少材料。依据：决定 315 / research/lon-diag.md、research/next-round/traffic-light-probe.md、experiments/stopline_probe、LONG-PROBE（lc_ctx 的 ln_elig）、on_lon.py。
本文件只写数什么、从哪数，不设判词线。

## 库

| 库 | 内容 | 路径（盒） |
|:--|:--|:--|
| 策略自身 | RSG-L18-s0 在 onpol-train 上的 18 决策 rollout（每场景 1 次），池 `on1_L18RSG` | `runs/onpol_nuplan/rows/L18RSG/{meta,all,lon}.npz`、`runs/op_parity/cache/on1_L18RSG{,@warp}`、rollout `runs/onpol_long/roll/RSG-L18-s0/c0..c5`（run 元数据 `runs/onpol_long/labels-L18RSG`） |
| 场景上下文 | 1491 个公开场景的 LONG-PROBE 上下文（路线 1 m 网格、停止线类型 / 弧长 / 灯态 24 关键帧 / ln_tstop / ln_elig、人类按弧长速度 vlog） | `runs/onpol_nuplan/ctx/ctx.npz` |
| 专家包 | rf_expert.py pack 的场景包（train64 + held 460） | `runs/route_follow/rewind_train64/pack`、`runs/route_follow/rule_expert/pack_held` |
| 人类对照 | navtrain 12 个分片的 token（LONG-PROBE ctx + body1 agent 标签 + tab 速度） | `runs/long_contact/ctx/ctx.npz`、body1 labels |
| PAI 高速直路 | `on1_PAITV`（SLOW 的 S2 / S3 行） | `runs/op_parity/cache/on1_PAITV`、`experiments/route_follow/results/slow/` |

只数池里的行（有帧）；all.npz 里被排除的决策按原因单列（dump 在，没有 token）。行在路线上：|l0| ≤ 3.5 m、pack ok。

## 数什么（每类：决策 / 场景 / log / 有帧）

1. **红灯接近**：路线上的信号灯停止线在车头前 0–40 m（ln_d − sf0），行的日志时刻（最近关键帧，≤ 0.3 s）该线连接器灯态红，策略速度 ≥ 6 m/s。
   子计数：served plan 在线前没降到 ≤ 1 m/s（字面）；其中 4 s 内越线；4 s 末状态以 ≤ 3 m/s² 也停不住（运动学上不停）；人类在该线停过（ln_tstop 有限，LONG-PROBE 口径：后轴在 [d − 15, d − 车头 + 1.5] 内 < 1 m/s，越线前）；4 s 窗内一直红。
   同样数 **stop sign 线**（类型 1，无灯态条件）与 **LONG-PROBE 合格线**（ln_elig；灯线要求此刻红，stop sign 合格即人在线前停）。
2. **空路偏慢**：rf_lonol free 格子（40 m 内无 A-eligible 前车、60 m 内无任何停止线、60 m 内路线航向变化 < 30°、速度 ≥ 2 m/s），策略速度 < 0.85 × 人类同站位速度（ctx vlog），
   按人类速度箱 < 8 / 8–12 / 12–16 / ≥ 16；有专家包的场景同样对专家 v_ref；另对全部场景用「ctx 路线上的同一 profile 函数」做代理 v_ref，并在 64 个有包场景上报代理与真包的一致度。
3. **弯前过快**：严格弯（rf_follow.find_turn：30 m 窗净航向 ≥ 60°，入口 = 10 % 处）入口在后轴前 0–40 m，策略速度 > 专家同站位速度 + 1 m/s（真包 / 代理），并报 > 人类 + 1。
4. 每类是否能直接做成当前格式的 on-policy 行：池行有 front.npy / tab / agent / road / route；缺的是纵向包络标签包（v_ref(s) 与理由掩码，按 on_lon.npz 的样子做）。
5. 专家包覆盖：1021 个有 rollout 的场景里多少有包；补包的 CPU 代价按 xp-pack / rwn-pack 两个池 job 的日志计时外推。
6. `on1_PAITV` 单列：行、场景、速度分布、SLOW 判词（S1 / S2 不过，S3 过护栏、held 不过）。

人类对照：navtrain 103 288 个 token 上同样的 1–3 类计数（人自己的速度、自己的停车），说明人类行里这些状态有多少、多快。

产出：`experiments/route_follow/results/lon_inventory/{inventory.md, inventory.json, *.csv}`；脚本 `experiments/route_follow/scripts/rf_loninv.py`（盒，op-train python，CPU 池 job）。
