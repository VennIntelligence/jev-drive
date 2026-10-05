# op_route_ft 预登记（2026-10-05 夜，训练前写定）

问题：给 openpilot 一个选路信号（小输入 adapter），轻度微调后能不能在 B2D 路口自己转过去。方案见 `tmp/2026-10-05-merged-finetune-plan.md` 第 7 节（已定：通道 a，先 T1–T3）与 `tmp/2026-10-05-night-plan.md`。代码 `scripts/rft.py`（训练）、`scripts/rft_eval.py`（开环读数）、`lib/route_adapter.py`（adapter，闭环服务端也用）。

## 臂（单 seed 0，从 shipped Cinque 出发）

| 臂 | 指令输入 | 数据 / 监督 | 地位 |
|:--|:--|:--|:--|
| rc-bear | `bear`：[present, has, 到下一个机动的距离 / 50, sin b, cos b − 1, b / 90°]，由加噪折线算出，再加距离 N(0, 2 + 0.1 d) 与方位 N(0, 6°) | T1 + T2 + T3 + 蒸馏 | 主臂 |
| rc-poly | `poly`：加噪 16 点折线（x, y / 50 m + mask） | 同上 | 指令形式对照 |
| rc-ctl | adapter 存在但输入恒为 0（= 无指令，输出严格为原模型通路） | 同样的行、同样的目标 | 分离「微调本身」与「指令」 |
| rc-all | 同 rc-bear | rc-bear + T4（op_adapt_L start + stop 两个切片 + stay 对照，dw 3）+ T5（layer-3 H / O 历史一致性配对，it_dw3 配方，不含 ln 起步配对）+ T6（navtrain 行上 nuPlan 地图可行驶区 hinge，权重 0.3，余量 0.4 m，只在日志路径本身在区内的帧上） | **探索性**，优先级最低，合并版本预览 |

可训练：stage 4 + plan pathway（off-policy，节点 479–665）+ **on-policy action pathway（节点 665–830）** + adapter；stage 1–3 冻结（trunk bank）。action pathway 是相对既有配方的唯一增项，理由：闭环横向的来源是 action 头（第 118 条），既有线只放开 plan pathway，action 只能经 stage 4 间接变；不放开它，开环 plan 学会转弯也传不到闭环。action[0] 的目标 = −0.45 × 目标轨迹 1 s 点的 pure-pursuit 曲率（左正）× max(1, v0)²（v0 < 1 m/s 不监督）；−0.45 是 shipped 头自己的关系（op_adapt_H 教师在 v0 > 3 m/s 上对日志 1 s 曲率的斜率 nav −0.51、wod −0.39，r −0.92），保持它在直路与宽弯上的增益不变。

超参（沿用，不扫）：layer-3 it_dw3 的 lam_i 1 / lam_d 10 / lam_c 1 / dw 3，base lr 3e-5，adapter lr 3e-4（op_adapt_L 的 lr_new），AdamW wd 0.01，warmup 100，cosine，clip 1，batch 48，4000 步（op_adapt_L 的步数；layer-3 用 2500）。pilot = 400 步（1/10）。

## 数据（每批 48 行）

- P（正样本，带指令）：real nav 7 + wod 5（op_adapt_H 池，route.npz 事后折线，目标 = 日志未来）；CARLA 出口配对 16（**重渲的对齐视角 1.86 m 集合** `carla_pairs_s10000ol`；目标 = 出口折线平滑后按原模型自己的弧长计时（creep / cruise 位姿至少保持自身速度：smoke 中原模型在空城路口 4 s 只走 13 m），横向加速度上限 3 m/s²）。只有代码 smoke 用旧 1.22 m 集合。
- T2：P 行按转角 / 半径采样加权：直行 1，25–60° 1.5，≥ 60° 2，≥ 60° 且 R_min < 15 m 4。
- N（负样本，6/48 = 12.5%）：CARLA N1（进近道路没有的出口类，按 CARLA 拓扑确定不存在）3；real 3，按类均分：navtrain 地图筛过的 `route_neg.npz`（main 规则：N1 只取 tier A 且 clear_m > 6，N2 / N4 clear_m > 8，N3 照筛选结果，去掉 trivial straight，不用 aux）落在 nav 图像池上的 N1 / N2 / N3 / N4 各一份，加 WOD 在线 N3 / N4（无地图，未筛）一份。目标 = 原模型 plan（权重 lam_n 1，与 P 行模仿同量级）+ 全部输出头。pilot（旧数据、训练前）里 N 行用 dw·lam_c = 3 的权重时指令吸收几乎为零，改为 1 后再放全量（见下）。
- D（无指令）：nav 5 / wod 4 / layer-3 CARLA 2 / CARLA 配对位姿 3，plan 一致性 + 全部输出头蒸馏回原模型。
- 切分：op_adapt_H 的 train / dev；CARLA 配对按路口 hash 的 train / dev（`b2d/route-carla-*`，与 B2D 172 个路口不相交）。

## 训练前在旧 1.22 m 数据上的代码迭代（只为排错，不进结论）

旧集合 1500 步 rc-bear：(1) 只用 4 s 模仿网格时 CARLA 行正确率 0.20（shipped 0.195），原因是原模型在空城路口 4 s 只走中位 13 m（d 中位 20 m），目标在 4 s 内到不了路口，指令信息落在监督视野之外；(2) 改成「路径形状」损失后 plan 变得更短（静止的 plan 天然贴路径），作废；(3) 定稿：CARLA 正样本在 plan 自身的时间点上监督到 8 s（sigma 随时间增长，与 L 网格同式），运动位姿的目标速度下限 4 m/s（过路口），正确率 0.53、到达路口的行 0.77、带符号横向 3.45 m（shipped 0.23），无指令漂移 ≤ 0.07 m，CARLA N1 负样本偏移 0.83 m（超 0.3 线，记下待看）。CARLA 行的 action 目标改为目标路径上前视 clip(v0·1 s, 4, 15) m 的 pure pursuit（与计时无关）。

## 读数与线（训练前写定）

| 读数 | 线 | 备注 |
|:--|:--|:--|
| **B2D 25 个路口转弯（第 127 条那组，seed 2，`spec`，zones 关，指令 = 稠密路线转成的导航折线，无噪声）走对出口** | **主读数**：rc-bear 显著高于 shipped（1/25）与 rc-ctl（配对、按转弯 bootstrap，CI 下界 > 0）；目标 ≥ 50%（≥ 13/25） | 失败时按 forced / choice、R_min、出口可见性（重渲的可见性表）拆 |
| 同上，出车道率、碰撞数 | 不高于 shipped zones 关（第 127 条：出车道 choice 100% / forced 78%） | |
| CARLA 留出路口（dev junction），每个（位姿，出口）行：plan 路径在弧长 d + 12 m 处的朝向类（> +30° 左，< −30° 右，否则直行；10 s plan 到不了记为 short，算错）= 指令出口 | 行正确率 ≥ 0.8；同时报三出口位姿「三条都对」比例与 action 符号一致率 | 只取运动中的位姿（profile ≠ stopped）。原写「原速度下 4 s 越过路口口 15 m」：smoke（旧 1.22 m 数据、训练前）发现 shipped 在 CARLA 路口 4 s 只走中位 13 m，790 个 dev 位姿只剩 2 个，改为按 plan 路径弧长判类 |
| 开环：无指令漂移 | 4 s 横向 |Δy| 中位 ≤ 0.10 m（路口帧 / 直路帧分开） | 相对 shipped |
| 开环：负样本偏移 | 均值 ≤ shipped + 0.3 m（shipped 不读指令，即 ≤ 0.3 m） | CARLA N1、real N3 / N4 分开报 |
| 开环：real 转弯帧指令吸收 | 报告，不设线 | 4 s 横向误差（对日志）有指令 vs shipped |
| 护栏子集（navtest、navhard、WOD、HUGSIM 11） | merged plan 第 4 节的线 | 由 op_guard 工具跑，视时间；rc-all 另报 navhard DAC、navtest DAC / EP（看 T6 是否拿 EP 换 DAC，第 104.9 条那种模式） |

pilot 放行条件（rc-bear，400 步）：(1) imit 与 act 损失比第 25 步下降；(2) CARLA dev 行正确率比 shipped 高 ≥ 0.10；(3) 无指令漂移中位 ≤ 0.15 m（pilot 宽一点）。任一不满足先查原因再放全量。

## 判读

- rc-bear 过主读数且 rc-ctl 不过：选择信号教会了转弯。
- rc-bear 与 rc-ctl 都涨：涨幅来自视角 / 急弯幅度的微调本身，指令贡献按两者差算。
- 都不涨而开环 CARLA 正确率高：开环到闭环的传递问题（action 头、反馈、可见性），按拆分回答「不选 / 选了转不够 / 看不见」。

## pilot（对齐视角集合，rc-bear 400 步，2026-10-05 01:47 JST）

损失下降（imit 2.64→1.79，long 2.75→1.57）；CARLA dev 行正确率 0.445 [0.42, 0.47] 对 shipped 0.156（+0.29，线 +0.10 过）；无指令漂移中位 ≤ 0.03 m（线 0.15 过）。三条都过，放全量：rc-bear / rc-poly 并行，rc-ctl、rc-all 随后。记下：左转指令只 0.165（右 0.275、直 0.894），CARLA N1 负样本偏移 0.63 m（超 0.3 线）。

## 2026-10-06 追加：接近路口的提前转向监督（rc-bear-pre / rc-ctl-pre，训练前写定）

问题（第 128 条嫌疑 a）：微调后的 action 头转得够但太晚，可能因为接近路口的位姿上 action 监督几乎为零（旧目标 = 1 s 点 / 前视 clip(v0, 4, 15) m 的 pure pursuit，入弯前是直行）。在接近位姿上监督「提前打方向」，指令能不能传到闭环转向？

**唯一改动**（`rft.py act_target_pre`，`RCfg.pre`）：P 行（real 与 CARLA 出口配对）中的接近位姿，action[0] 目标换成提前转向目标；其余所有行、损失、数据、步数（4000）、超参、seed 与 rc-bear / rc-ctl 完全相同。
- action 的定义（openpilot）：action[0] = t + lateralDelay（0.2 s）时刻的期望横向加速度，闭环曲率 = action / max(1, v)²。照着 plan 跟踪的车，期望曲率就是「人开出来的路径」在车所在处的曲率；人在入弯前就开始打方向，所以目标取指令路径的曲率在前视窗口里的平均。
- 接近位姿：目标路径（指令出口折线平滑后 1 m 间隔；real 行是日志事后折线）从车到 s_d = v0·0.2 s + 2 m 是直的（|κ| ≤ 0.02），且下一个机动（|κ| > 0.02，R < 50 m）在提前转向距离 W = v0·1.5 s + R/2 之内开始（R = 机动最小半径，上限 30 m；R/2 = 长约 R 的回旋线入口以切点为中心，提前 R/2 开始；1.5 s = 驾驶员预瞄，含 lateralDelay）。
- 目标曲率 = 指令路径曲率在 [s_d, s_d + W] 上的平均（转弯起点距离从 s_d + W 到 s_d 时，目标从 0 线性升到弧的曲率，与指令出口成正比、同号）；action 目标 = −0.45·κ·max(1, v0)²（与其余行同一换算）；v0 < 1 m/s 仍不监督（不改）。
- 训练前统计（4000 个 CARLA train 行，`pre_stats`）：brake d 10 / 20、cruise d 20、creep d 10 的转弯行 80–99% 成为接近位姿，|目标| 0.1–0.8 m/s²（旧目标 0.04–0.31，且与出口无关）；直行指令行 ≤ 2%；d 30 与 creep d 20 / 30 基本不变；real 接近帧 nav 467 / 1957、wod 359 / 1938，新旧目标相关 0.78 / 0.48（real 旧目标是日志 1 s 点，已含真人的提前转向，这里按同一规则替换）。
- 同一 CARLA 位姿、不同出口 → 不同目标：只有读指令才能拟合。rc-ctl-pre 没有指令，只能学到各出口的平均（对照：提前转向来自指令还是场景线索）。

**臂**：rc-bear-pre（= rc-bear + 改动），rc-ctl-pre（= rc-ctl + 改动）。比较对象：rc-ctl-pre（指令贡献），rc-bear / rc-ctl（改动本身的效果，同 seed）。

**desire**：另一条线（`results/desire_off.md`）在跑闭环关 desire 的 rc-bear / rc-ctl / shipped。出结果后按它显示的正确设置评估；评估开始时还没结果，则 desire 关、开各跑一遍，主读数取 desire 关（指令只走 adapter，desire 不再带侧向信息）。

**pilot**（rc-bear-pre，400 步，放行条件）：(1) imit 与 act 损失比第 25 步低；(2) 开环提前转向（`rft_eval.turnin`，CARLA dev 运动位姿的转弯行）：接近位姿上朝指令侧的期望曲率均值 ≥ 目标均值的 0.3，且左右指令差（同一位姿左指令 κ − 右指令 κ）的 CI 下界 > rc-bear-s0 的均值；(3) CARLA dev 出口正确率 ≥ shipped + 0.10；(4) 无指令漂移中位 ≤ 0.15 m。任一不过先查原因。

**读数与线**

| 读数 | 线 |
|:--|:--|
| **主**：B2D 25 转弯（guard `b2d_turns` 那组，seed 2，zones 关）走对出口 | rc-bear-pre ≥ 13/25，且对 rc-ctl-pre 配对差 > 0（按路线 bootstrap，CI 下界 > 0） |
| 闭环转向时机（`rft_split.py`）：进入的转弯里打到 0.5 / R_min 的弧长位置（相对转弯起点）中位；「太晚」个数 | rc-bear-pre 中位 ≤ 0 m（转弯起点前），且「太晚」少于 rc-bear（6） |
| 开环提前转向：左右指令差、接近位姿 κ / 目标 | 左右差 > rc-bear-s0（CI 下界之上）；报告按 profile × d |
| CARLA dev 出口走对（按行） | ≥ 0.8（原线）；且不低于 rc-bear 0.583 − 0.05 |
| 无指令漂移 | ≤ 0.10 m |
| 负样本偏移 | CARLA N1 ≤ 0.3 m，navtrain 筛过 ≤ 0.3 m（原线） |
| 护栏子集（guard.py subset） | 工具的线（rc-bear 当时 navhard −0.43 与早转子集 −0.30 不过）|

判读：rc-bear-pre 过主线且转得更早 → 嫌疑 a 成立，监督缺口是主因；rc-bear-pre 与 rc-ctl-pre 都转得更早、走对数相近 → 提前转向来自场景线索 / desire，不来自指令；开环左右差变大但闭环时机不变 → 传递问题在闭环（速度、窗口、desire），不在监督。

### pilot（rc-bear-pre 400 步，2026-10-05 08:06 box 时间）

(1) imit 2.65 → 1.80、act 0.140 → 0.114（第 100 → 400 步）过；(2) 开环提前转向：接近位姿朝指令侧期望曲率 0.0035 1/m 对目标 0.0107（0.33，线 0.3 过）；左右指令差 0.005 [0.005, 0.006] 对 rc-bear-s0 0.002（过）；action 符号对指令侧 0.81（rc-bear-s0 0.53，shipped 0.50）；(3) CARLA dev 出口走对 0.447 对 shipped 0.156（过）；(4) 无指令漂移中位 0.027 m（过）。四条都过，放全量。记下：d 30 的 creep 位姿（目标约 0）也朝指令侧转 0.0036 1/m，头把提前转向泛化到了目标为 0 的远处；CARLA N1 负样本 0.63 m（rc-bear-s0 0.47）。

## 2026-10-06 near：近距离、低速的 CARLA 入弯位姿（rc-bear-near / rc-ctl-near，训练前写定，2026-10-05 box 时间 10:40）

问题（第 129 条）：rc-*-pre 失败的直接原因是真实数据接近段的日志目标（真人提前转向 0.27–0.34 m/s²）被更小的合成斜坡覆盖，真实接近 / 弯中帧的 action 增益从约 0.96 降到 0.22–0.50；另外 CARLA 位姿全部 ≥ 10 m、≥ 4 m/s 目标，而闭环在 1–3 m/s 进弯。只改训练状态分布、不碰真实目标时，入弯能不能提前 / 打够？

**唯一改动（相对 rc-bear / rc-ctl）**：每批 16 个 CARLA P 行里 6 个换成新的近距离集合（`scripts/near_plan.py`，渲染器 `op_route_cmd/scripts/carla_pairs_render_ol.py` 原样、对齐相机、留出 172 个 B2D 路口、dev 用同一 sha1 规则）。真实行的目标、其余行、损失、4000 步、超参、seed 0 全部不变；**不含** act_target_pre。
- 位姿（4000 个，7941 行，914 个路口，dev 356 个位姿）：接近 2681 个（后轴在连接段起点前 0–10 m，均匀；行 = 该车道所有合法出口，3 出口车道 1018 个位姿，不做变道混合）；弯中 1319 个（后轴在一个合法转弯出口的连接段上 0.5–6 m；行 = 该出口 + 仍未分叉（0.5 m、6° 内）的其他合法出口）。速度：stopped 10%、halted（t0 停住，之前以 0.5–1.5 m/s² 减速）10%、rolling 80%（v0 ~ U(0.3, 3)，1.8 s 历史内匀加速度 ~ U(−1, 1)，速度夹在 [0, 5]）；历史帧按该运动学沿车道中心线放置。
- 目标：用每行的稠密车道中心线（1 m 间隔，与折线同样的起点收敛）而不是 10 m 顶点折线（顶点切掉每个直角弯的角，近处差别大）。action[0] = rft 原规则 `act_target_path`（纯追踪，前视 clip(v0·1 s, 4, 15) m，−0.45·κ·max(1, v0)²，在圆弧上即 1/R），监督下限改为 0.3 m/s（openpilot latActive；曲率 = action / max(1, v)² 在 0.3–1 m/s 同样成立；旧集合与真实行仍是 1 m/s）。plan 目标：rft 原 `path_target`，速度下限从 v0 以 1.5 m/s² 升到 4 m/s（原规则是第一步就跳到 4 m/s；v0 < 0.3 用原模型自己的计时，同 stopped）。
- 训练前目标统计（`near_eval.py stats`，带符号均值，1/m）：弯中 左 0.102 / 右 −0.147（路径最大曲率中位 0.098 / 0.148，即目标 ≈ 1/R）；接近 d 0–3 m 左 0.037 / 右 −0.062；d 3–10 m ≈ 0.001–0.006（纯追踪在 4 m 前视下只在最后约 3 m 开始转，对精确跟车道中心线的车这是对的）；直行 ≤ 0.002。监督占比 0.76–0.82（其余 v0 < 0.3）。对照：旧集合 creep d10 的旧目标约 0.002。

**臂**：rc-bear-near（主），rc-ctl-near（无指令对照，卡空闲时与主臂同卡并行）。比较对象：rc-bear-s0 / rc-ctl-s0（同 seed，近集合的效果），rc-ctl-near（指令贡献）。

**闭环设置**：desire 关（第 128 条第 8 点：desire 不带侧向信息；这样指令只走 adapter）；zones 关；guard `b2d_turns` 那组 25 个转弯，seed 2，`spec` 相机。横向执行器按 plan-tracking 线（rft-pt，p7）的结论；评估开始时它还没结论，则 curv（action 曲率，spec 默认）与 p7 各跑一遍，主读数取 curv（本次改动针对 action 头），p7 一并报告。

**pilot**（rc-bear-near 400 步，对 pilot-bear = rc-bear 400 步，同 seed），放行条件：
1. imit 与 act 损失比第 100 步低；
2. **真实行 action 增益**（`pre_diag.py`，真实 dev P 行，v0 ≥ 1，对日志 1 s 纯追踪目标的过原点斜率）：nav 接近、wod 接近两类各 ≥ pilot-bear − 0.05（wod 弯中 n = 7 只报告）；
3. 近集合 dev 弯中行（`near_eval.py eval`）增益 > pilot-bear 的，且符号对指令侧 ≥ 0.9；
4. CARLA ol dev 出口走对 ≥ shipped + 0.10，无指令漂移中位 ≤ 0.15 m。
任一不过先查原因，不放全量。

**读数与线**

| 读数 | 线 |
|:--|:--|
| **主**：B2D 25 转弯走对出口（desire 关、zones 关、curv） | rc-bear-near ≥ 13/25；且 rc-bear-near − rc-bear（同设置，desire 关 6/25）配对、按转弯 bootstrap CI 下界 > 0 |
| 指令贡献（训了 rc-ctl-near 时） | rc-bear-near − rc-ctl-near CI 下界 > 0（报告，不作为放行） |
| 真实行 action 增益（pre_diag，全量） | nav 接近、wod 接近、wod 弯中都 ≥ 0.90（rc-bear 0.96 / 0.97 / 0.98），即不出现 rc-*-pre 那种塌陷 |
| 近集合 dev 增益（near_eval） | 弯中 ≥ 0.5 且符号 ≥ 0.9；接近 d 0–3 报告；接近左右指令差 > rc-bear-s0 |
| 闭环转向时机（rft_split） | 进入的转弯里到达 0.5 / R_min 的位置中位早于 rc-bear（desire 关），「太晚」少于 rc-bear |
| CARLA ol dev 出口走对 | ≥ 0.583 − 0.05 |
| 无指令漂移 | ≤ 0.10 m |
| 负样本 | CARLA N1 ≤ 0.3 m（rc-bear 0.47 已不过，只要求不更差 > 0.05）、navtrain 筛过 ≤ 0.3 m |
| 护栏子集（guard.py subset，desire 开，工具的线） | 工具的线；rc-bear 的 navhard −0.43、早转 −0.30 当参考 |

判读：主线过且真实增益保持 → 训练状态分布是瓶颈；闭环不变而近集合开环增益高 → 瓶颈不在 action 头的监督，在闭环（速度控制 / 停车 / 执行器）；真实增益掉 → 近集合目标与真实目标冲突（与第 129 条同类），看是哪类位姿。

### 更正（2026-10-06，训练前）：action 目标的换算错了，rc-* 全部臂都带着它

第 130 条（`experiments/op_closed_loop/results/action_scale.md`）：`ACT_ALPHA = 0.45 × 1 s pure pursuit × max(1, v0)²` 不是 shipped 头的约定。modeld 的约定是 action[0] = κ_inst(t + 0.275 s) · max(1, v_model)²，增益 1，右正；0.45 来自 1 s pure pursuit 当比较目标、回归方向，以及开环榜相机高度让模型低估自身速度（v_model / v：WOD 0.92、nav 0.81、B2D spec 0.87）。**第 128、129 条的 rc-bear / rc-poly / rc-ctl / rc-all / rc-*-pre 都用了旧目标**（相对 shipped 自己的尺度约一半）。本节以上关于「0.45」「纯追踪前视 clip(v0, 4, 15) m」的写法作废，改为（`rft.py` e57a3bb5 / eb749f1a，单测 `tests/test_rft_act_target.py`）：
- 真实行：`act_target(hum, v0, v_model)`，κ_inst = 日志未来 0–1.5 s 三次拟合在 t = 0.275 s 的曲率，v_model = 原模型 plan 在 t0 的速度（`vmodel(tea mu)`，相机尺度不对时用模型自估速度），|κ| 截到 0.2（controlsd 上限），v0 < 1 不监督（不变）。
- CARLA 行与近距离行：`act_target_path`，κ_inst = 稠密目标路径在弧长 v0 · 0.275 处的曲率（3 m 窗口的航向差，路径起点处窗口前移），v_model 同上；近距离行从 0.3 m/s 起监督。
- `act_target_pre` 保留旧尺度，仅用于复现 rc-*-pre。
- 目标统计（新约定，带符号均值 1/m）：近集合弯中 左 0.067 / 右 −0.096（瞬时曲率 = 车所在处的曲率；连接段入口段曲率仍在上升，故低于弧的最大曲率 0.098 / 0.148）；接近 d 0–3 左 0.036 / 右 −0.064；d 3–10 ≤ 0.011；直行 0.000。真实 train 行（v0 ≥ 1）新目标对旧目标斜率 nav 1.24、wod 1.12（相关 0.94 / 0.95；含 v_model² 的 0.66–0.84 缩小）。

**因此设计改一处**：rc-bear-near 与 rc-bear 之间现在有两处差别（目标约定 + 近集合），配对不再是单变量。加对照臂 **rc-bear-fix** = rc-bear 配方 + 新目标、无近集合行。主比较改为 **rc-bear-near − rc-bear-fix**；rc-bear-fix − rc-bear 读「目标更正本身」。rc-ctl-near 降为可选（卡有余时）。
- pilot：rc-bear-near 与 rc-bear-fix 各 400 步并行；放行条件 2 改为「pilot-bear-near 的真实 nav / wod 接近增益（对新目标）≥ pilot-bear-fix − 0.05」；条件 3 改为对 pilot-bear-fix 比；条件 1、4 不变。
- 主读数：rc-bear-near ≥ 13/25，且 rc-bear-near − rc-bear-fix 配对 CI 下界 > 0；同时报 rc-bear-fix − rc-bear（desire 关，6/25）。
- 真实行增益线：对新目标，rc-bear-near 在 nav 接近、wod 接近、wod 弯中都 ≥ rc-bear-fix − 0.05（旧线「≥ 0.90，rc-bear 0.96」是对旧目标量的，作废；rc-bear 对新目标的增益在全量 pre_diag 里一并报）。
- 其余读数与线不变。

### 范围变更（2026-10-05，用户；pilot 读数之前写定）

B 只做 400 步 pilot（rc-bear-fix、rc-bear-near）+ B2D 小集（8 条路线、25 个转弯中的 9 个：choice 迟入 10255 / 5423 / 15102 / 34183 / 28008#1，forced 25051 迟入、28180 急弯低速、27994 与 28008#0 已走对；desire 关、zones 关、curv、seed 2）。不训全量、不跑 25 转弯与护栏。产出是打包好的近距离数据与更正后的目标代码，交给合并训练。
结论只回答「合并训练里要不要放近距离 / 低速目标」：
- **要**：pilot-bear-near 小集走对 ≥ pilot-bear-fix + 2，或进入的转弯里入弯位置中位早 ≥ 2 m；且真实接近行增益 ≥ pilot-bear-fix − 0.05，近集合 dev 弯中增益高于 pilot-bear-fix。
- **不要**：小集走对差 ≤ 1 且入弯时机不变，或真实接近行增益掉 > 0.05。
- 介于两者：报「不确定」，给出数字。参照 rc-bear-s0（desire 关）在这 9 个上走对 2 个。
