# op_route_ft 预登记（2026-10-05 夜，训练前写定）

问题：给 openpilot 一个选路信号（小输入 adapter），轻度微调后能不能在 B2D 路口自己转过去。方案见 `tmp/2026-10-05-merged-finetune-plan.md` 第 7 节（已定：通道 a，先 T1–T3）与 `tmp/2026-10-05-night-plan.md`。代码 `scripts/rft.py`（训练）、`scripts/rft_eval.py`（开环读数）、`lib/route_adapter.py`（adapter，闭环服务端也用）。

## 臂（单 seed 0，从 shipped Cinque 出发）

| 臂 | 指令输入 | 数据 / 监督 | 地位 |
|:--|:--|:--|:--|
| rc-bear | `bear`：[present, has, 到下一个机动的距离 / 50, sin b, cos b − 1, b / 90°]，由加噪折线算出，再加距离 N(0, 2 + 0.1 d) 与方位 N(0, 6°) | T1 + T2 + T3 + 蒸馏 | 主臂 |
| rc-poly | `poly`：加噪 16 点折线（x, y / 50 m + mask） | 同上 | 指令形式对照 |
| rc-ctl | adapter 存在但输入恒为 0（= 无指令，输出严格为原模型通路） | 同样的行、同样的目标 | 分离「微调本身」与「指令」 |
| rc-all | 同 rc-bear | rc-bear + T4（op_adapt_L start + stop 两个切片 + stay 对照，dw 3）+ T5（layer-3 H / O 历史一致性配对，it_dw3 配方，不含 ln 起步配对）+ T6（navtrain 行上 nuPlan 地图可行驶区 hinge，权重 0.3，余量 0.4 m，只在日志路径本身在区内的帧上） | **探索性**，优先级最低，合并版本预览 |

可训练：stage 4 + plan pathway（off-policy，节点 479–665）+ **on-policy action pathway（节点 665–830）** + adapter；stage 1–3 冻结（trunk bank）。action pathway 是相对既有配方的唯一增项，理由：闭环横向的来源是 action 头（第 118 条），既有线只放开 plan pathway，action 只能经 stage 4 间接变；不放开它，开环 plan 学会转弯也传不到闭环。action[0] 的目标 = 目标轨迹 1 s 点的 pure-pursuit 曲率 × max(1, v0)²（v0 < 1 m/s 不监督）。

超参（沿用，不扫）：layer-3 it_dw3 的 lam_i 1 / lam_d 10 / lam_c 1 / dw 3，base lr 3e-5，adapter lr 3e-4（op_adapt_L 的 lr_new），AdamW wd 0.01，warmup 100，cosine，clip 1，batch 48，4000 步（op_adapt_L 的步数；layer-3 用 2500）。pilot = 400 步（1/10）。

## 数据（每批 48 行）

- P（正样本，带指令）：real nav 7 + wod 5（op_adapt_H 池，route.npz 事后折线，目标 = 日志未来）；CARLA 出口配对 16（**重渲的对齐视角 1.86 m 集合** `carla_pairs_s10000ol`；目标 = 出口折线平滑后按原模型自己的弧长计时，横向加速度上限 3 m/s²）。只有代码 smoke 用旧 1.22 m 集合。
- T2：P 行按转角 / 半径采样加权：直行 1，25–60° 1.5，≥ 60° 2，≥ 60° 且 R_min < 15 m 4。
- N（负样本，6/48 = 12.5%）：CARLA N1（进近道路没有的出口类，按 CARLA 拓扑是确定不存在的）3；real N3（逆向侧 4–7 m）/ N4（无路口直路上掉头）3，`lib/route_neg.make` 在线生成。目标 = 原模型 plan + 全部输出头。另一位执行者的自动筛选负样本集（navtrain token）若与图像池有交集再加入；没有就只用以上两类（在报告里写明）。
- D（无指令）：nav 5 / wod 4 / layer-3 CARLA 2 / CARLA 配对位姿 3，plan 一致性 + 全部输出头蒸馏回原模型。
- 切分：op_adapt_H 的 train / dev；CARLA 配对按路口 hash 的 train / dev（`b2d/route-carla-*`，与 B2D 172 个路口不相交）。

## 读数与线（训练前写定）

| 读数 | 线 | 备注 |
|:--|:--|:--|
| **B2D 25 个路口转弯（第 127 条那组，seed 2，`spec`，zones 关，指令 = 稠密路线转成的导航折线，无噪声）走对出口** | **主读数**：rc-bear 显著高于 shipped（1/25）与 rc-ctl（配对、按转弯 bootstrap，CI 下界 > 0）；目标 ≥ 50%（≥ 13/25） | 失败时按 forced / choice、R_min、出口可见性（重渲的可见性表）拆 |
| 同上，出车道率、碰撞数 | 不高于 shipped zones 关（第 127 条：出车道 choice 100% / forced 78%） | |
| CARLA 留出路口（dev junction），每个（位姿，出口）行：4 s 时 plan 的朝向类（> +30° 左，< −30° 右）= 指令出口 | 行正确率 ≥ 0.8；同时报三出口位姿「三条都对」比例与 action 符号一致率 | 只取原速度下 4 s 越过路口口 15 m 的位姿 |
| 开环：无指令漂移 | 4 s 横向 |Δy| 中位 ≤ 0.10 m（路口帧 / 直路帧分开） | 相对 shipped |
| 开环：负样本偏移 | 均值 ≤ shipped + 0.3 m（shipped 不读指令，即 ≤ 0.3 m） | CARLA N1、real N3 / N4 分开报 |
| 开环：real 转弯帧指令吸收 | 报告，不设线 | 4 s 横向误差（对日志）有指令 vs shipped |
| 护栏子集（navtest、navhard、WOD、HUGSIM 11） | merged plan 第 4 节的线 | 由 op_guard 工具跑，视时间；rc-all 另报 navhard DAC、navtest DAC / EP（看 T6 是否拿 EP 换 DAC，第 104.9 条那种模式） |

pilot 放行条件（rc-bear，400 步）：(1) imit 与 act 损失比第 25 步下降；(2) CARLA dev 行正确率比 shipped 高 ≥ 0.10；(3) 无指令漂移中位 ≤ 0.15 m（pilot 宽一点）。任一不满足先查原因再放全量。

## 判读

- rc-bear 过主读数且 rc-ctl 不过：选择信号教会了转弯。
- rc-bear 与 rc-ctl 都涨：涨幅来自视角 / 急弯幅度的微调本身，指令贡献按两者差算。
- 都不涨而开环 CARLA 正确率高：开环到闭环的传递问题（action 头、反馈、可见性），按拆分回答「不选 / 选了转不够 / 看不见」。
