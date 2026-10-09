# op_parity innovation-gain 预登记：SH30 缺的是不是「由场景引起的变化」（lane IG1，2026-10-09，任何模型 plan 的 innovation 统计之前写定并 push）

纯测量 lane：只用存档的 plan、逐 token 分数和 token 表，CPU；唯一的 GPU 作业是第 8 节的一次图像消融推理。没有任何内容作为方法提出，
日志未来只作读数的标尺。判定线（第 9 节）在第一个数之前写死，之后不改。

## 1. 假设与读数（用户提出）

SH30（冻结 openpilot encoder + adapter + hinge，第 170 条；AP2 是 AlpaSim 输入口径的变体）在未来延续过去时开得好，未来应当不同于过去时，
不从画面决定怎么变；它的 plan 约等于运动学外推 + 域先验 + 指令先验。

- **innovation**（新息）= 日志 4 s 未来 − 最近 1.5 s ego 历史的恒曲率、恒加速度外推，拆成横向与纵向；plan 做同样的减法。
- **innovation gain** = plan innovation 对日志 innovation 回归的斜率（1 = 完全跟随场景，0 = 纯外推）。
- P1：SH30 的 gain 明显低于 WA-JEPA。P2：按日志 innovation 分位数分层，对 WA-JEPA 的分差集中在高分位、低分位持平。
  P3：控制 innovation 后，转角桶效应消失。
- 证伪：gain 与 WA-JEPA 接近，或分差在各分位均匀；那时缺的是精度而不是这项能力。第 178 条说没有全局欠转，横向一半可能被证伪。

## 2. 已有条目与本线的差别（不重测）

- 第 92 / 96 / 98 条：shipped Cinque 外推历史**画面**里的偏航（注入假偏航的探针 G）。那是对输入扰动的响应，本线量的是未扰动 token 上
  plan 对日志「变化」的跟随程度，对象是 SH30 与阶梯上的其他模型。含义：ego 运动也经画面进入模型，第 8 节的图像消融因此要配运动学匹配的对照。
- 第 156 条：历史只经 t0 加速度进 P2H，匀速历史只 +0.13 m。第 162 / 167 / 169 条：adapter 常数项是域内起步先验。这些是「域先验」一侧的证据，不给 gain。
- 第 173 条：WOD 上薄 head 从 F20 候选里选，视觉流不加分。对象是 479 帧上的选择器，不是 SH30 的 plan，也不在 navtest：**不回答**第 8 节的问题。
- 第 178 条：> 20° token 上没有可用的固定变换（无全局欠转）。它不排除「大 innovation 处响应不足、小 innovation 处无偏」这种依赖幅度的缺口。
- 第 196 条：NC 失败 53% 是前方静止 / 慢车，73% 的 plan 比日志长 1.1 倍。本线把它重表达为纵向 innovation 分位上的份额，不重做分类。
- 第 202 条：AlpaSim 零分主因是直行段上 plan 自己横移 1–2 m（日志不动）。即「虚构的变化」，第 6 节单列。
- 第 183 条：WA-JEPA 在 WOD 上的图像消融（图像 +2.78）。对象是 WA-JEPA，不是 SH30。
- plan.md 第 6 节第 6 点（MemoryDrivoR 不看当前相机也有 91.1）：换掉相机的对照臂还没做过。

写本稿前看过的数：plan.md 第 1 节的全榜与转角桶分差表、第 170 / 178 / 196 / 202 条的结论；为确认数组格式打印过 `tab.npz` 两行（历史位姿、未来、ego）、
SH30 一行导出位姿、WA-JEPA 轨迹字典的键与形状。没有算过任何模型 plan 的 innovation、任何 gain、任何按 innovation 的分层。

## 3. 定义（全部写死）

**数据**：`runs/op_parity/cache/lb_navtest/tab.npz`：`pose`（t0 后轴系下 t = −1.5 / −1.0 / −0.5 / 0 s 四个历史位姿）、`speed`（t0 速度）、
`fut`（日志未来，0.5–4 s 八个位姿）、`cmd`、`log`。plan 为 bench 导出位姿（后轴，0.5–4 s，(n, 8, 3)）。split `navsim/navtest`。

**外推 E-cc（主口径，恒曲率恒加速度）**：
- 历史三段弦长 ℓ₁ ℓ₂ ℓ₃，L_h = Σℓ；段速 u_k = ℓ_k / 0.5 s。v₀ = `speed`。a = (u₃ − u₁) / 1.0 s，截到 [−4, +3] m/s²。
- 曲率在弧长域取：κ = (ψ(0) − ψ(−1.5 s)) / L_h，截到 |κ| ≤ 0.2 m⁻¹。**L_h < 1.5 m（1.5 s 内均速 < 1 m/s）时 κ = 0**：
  4 个 2 Hz 位姿在低速下的航向差除以弧长是病态的，此时外推为沿当前航向的直线。
- v(t) = max(v₀ + a t, 0)（不倒车，减到 0 就停住），s_E(t) = ∫v；位置在曲率 κ 的圆弧上，航向 κ s_E。
- 备选：**E-cv**（κ = 0，a = 0，匀速直线）与 **E-ccv**（κ 同上，a = 0）。E-cv 用来说明结论不依赖运动学模型（第 7 节），E-ccv 只进表。

**坐标（Frenet / 弧长）**：以外推路径（同曲率向前无限延伸的圆或直线）为参考线。任一轨迹（日志或 plan）在视野 h 的
- **横向坐标** d(h)：其 h 时刻位置到参考线的带符号垂距，左为正（外推自身恒为 0）；
- **纵向坐标** L(h)：轨迹自身从原点到 h 的折线弧长（只含速度，不含路径形状）。
- **横向 innovation** = d(h)；**纵向 innovation** = L(h) − s_E(h)。日志的记 x_lat、x_lon，plan 的记 y_lat、y_lon。单位 m。
- 主视野 h = 4 s；h = 2 s 只进表。次要横向读数（只进表）：航向 innovation = ψ(h) − κ s_E(h)（度）；
  **等弧长横向**：日志路径与 plan 路径在同一弧长 s* = min(L_log, L_plan) 处的 d（s* < 2 m 的 token 不计），用来排除「同一路径只是开得慢，
  d 就小」的串扰。

**token 集合**：gain 在全部阶梯模型都有 plan 的 token 上算（WA-JEPA 轨迹存档 12 027 个，预计即此集合）；分数份额在全部有分数的 12 146 个上算。
**分位**：按 |x| 的秩分 10 等份（并列按 token 顺序），每个轴各自分；「高三档」= 最高的 3 个 decile。联合：两轴秩的较大者再分 10 档，另给 3 × 3 tercile 表。

**gain 的估计**（每个模型、每个轴）：OLS y = α + g x。报 g、α、R²、符号一致率（|x| ≥ 0.5 m 横向 / 1.0 m 纵向的 token 上 sign y = sign x 的比例）、
**top-decile gain**（|x| ≥ P90(|x|) 的 token 上带截距的 OLS 斜率；过原点的 Σxy / Σx² 并列报）。
CI：按 log 的 cluster bootstrap，B = 10 000，`default_rng(0)`，与 `jevdrive.stats` 同一重采样约定（斜率不是均值，用每 log 的充分统计量重采样）；
模型间差用同一组重采样（配对）。SH30 = 两个 seed 的 gain 取平均，逐 seed 并列。

## 4. 衰减对照：gain 阶梯（修正 1）

回归训练的 planner 输出条件均值，斜率必然 < 1；x 里的测量噪声对所有模型是同一个，不影响阶梯内的比较。所以绝对斜率不读，只读同一批 token 上的阶梯：

| 档 | 来源 | 说明 |
|---|---|---|
| EXT | 外推本身 | y ≡ 0，gain 0（自检） |
| LOG | 日志 | y ≡ x，gain 1（自检） |
| CONST | navtrain 日志未来的均值轨迹当 plan | 不看任何输入；量出「plan 与日志共用同一个被减项」带来的机械斜率 |
| BLIND-E / BLIND-EC | 梯度提升回归（sklearn HistGradientBoosting），只吃 ego 历史（位姿、速度、加速度、v₀、a、κ）/ 再加指令 one-hot，在 navtrain 的 `tab.npz` 上拟合 x_lat、x_lon，在 navtest 上预测 | 「运动学外推 + 域先验 + 指令先验」的显式版本，不看画面的回归能到的 gain；navtrain 与 navtest 按 split 不相交 |
| cinque（shipped，G 帧） | `cinque-gimm__base.npz` | 出厂 openpilot plan |
| P2-F-s0、RH0 / SHP（pilot）、GH0-F-s0/s1 | bench 存档 | 无 hinge、pilot 规模的对照 |
| **SH30-F-s0 / s1** | bench 存档 | 被检对象 |
| OT30-F-s0 / s1 | bench 存档 | 昨夜离轨行臂（第 198 条） |
| GEB / GEW / GEX / GEP-F-s0/s1 | bench 存档 | 第 200 条的 pilot 臂；GEP = JP，泄漏日志路径的 oracle 臂（seed 1 学会了读，seed 0 没有），作上参照，逐 seed 报 |
| AP2-AB-s0 | 若 navtest 上有存档 plan（`runs/alpasim/ap2/offline`，m = 4） | 没有就记「未存」，不补跑 |
| **WA-JEPA** | `trajectory_cache/done_union.pkl` 存档轨迹与逐 token 分数 | 只作对照列；不做任何新的 WA-JEPA 推理 |

存档里没有的档写「未存」并说明，不补推理。**「明显低于」的阈值（先于任何数）：WA-JEPA 减 SH30 的 gain 差 ≥ 0.10（绝对值，全体斜率与 top-decile 斜率同一阈值），
且配对 CI 不含 0。** 依据：0.10 的 gain 差在 top decile（横向 innovation 预计数米量级）对应 4 s 处约 0.5 m 以上的少跟，与 hinge margin 0.5 m 同量级；
更小的差在分数上分辨不出。另报对机械项稳健的**偏 gain**：y 对 [x, s_E(h), v₀, a, κ] 回归时 x 的系数。

## 5. 分数挂钩（修正 3，决定这个想法值多少）

- 逐 token EPDMS 差 Δ = SH30（两 seed 均值）− WA-JEPA（bench `units.csv` 的 `score`，全榜 −2.16）。每个 decile 的**份额** = Σ_decile Δ / Σ_all Δ，
  横向、纵向、联合各一张，带 log bootstrap CI；并列：该档 token 占比（均匀时 10%）、SH30 与 WA-JEPA 各自失分（100 − score）的份额。
  **任何「innovation」方法的上界 = 高分位档里的份额 × 2.16。**
- 第 196 条的 NC 失败（NC < 1）与 TTC-only（TTC < 1 且 NC = 1）：两 seed 合并，按同样的 decile 给份额；另给 SH30 特有（WA-JEPA 同 token 通过）的份额。
- DAC 失败按类型：切内角 / 转不过去 / 其他，分类沿用 `turn_oracle.py` 报告里的规则与存档重放（`replay_sh.parquet`），不重写、不重放。

## 6. 虚构的变化（修正 2）

日志 innovation 近零处 plan 自己的 innovation，逐模型、逐轴：
- 集合：该轴 |x| 最低的 2 个 decile；另给绝对带 |x_lat| < 0.25 m、|x_lon| < 0.5 m，以及「日志延续」集合（两轴都在最低 3 个 decile 且 v₀ ≥ 2 m/s）。
- 读数：|y| 的均值、中位数、P90，带符号均值；横向 |y| ≥ 0.5 m / ≥ 1.0 m 的比例，纵向 |y| ≥ 1 m / ≥ 2 m 的比例；SH30 − WA-JEPA 的 |y| 均值差带配对 CI。
- 一个模型可以同时响应不足（gain 低）又虚构变化（此处 |y| 大），两者分开报，不合成一个数。

## 7. P3 与定义稳健性（修正 4、5）

**P3**：逐 token Δ 的三个线性模型（哑变量 OLS）：M_T 只有转角桶（< 5 / 5–20 / 20–45 / > 45°，bench strata）；M_I 只有 innovation（横向 10 档 + 纵向 10 档，加性）；M_TI 两者。
- 报转角桶对比（相对 < 5°）在 M_T 与 M_TI 里的系数与 log bootstrap CI。**P3 成立** = M_TI 里 20–45° 与 > 45° 两个对比的 CI 都含 0 且幅度都比 M_T 缩小 ≥ 70%。
- 反向：innovation 对比（最高档相对最低 5 档合并，两轴）在 M_I 与 M_TI 里的系数；按 log 分 5 折的 out-of-fold R²：M_T、M_I、M_TI 及两两之差的 CI。
  **innovation 有增量** = M_TI 的 out-of-fold R² 高于 M_T 且差的 CI 不含 0；**转角桶有增量**同理对 M_I。
- 共线性如实报：|dyaw| 与 |x_lat| 的 Spearman 相关、转角桶 × 横向 5 分位的 token 计数与格内 Δ 均值。重叠格（如 > 45° 而横向低分位）不足 50 个 token 时，
  该对比记「不可辨」，不硬判。

**定义稳健性**：(a) 全部 token 与去掉 v₀ < 2 m/s 的 token 各算一遍（分位在子集内重排），gain 阶梯的 SH30 / WA-JEPA 行与份额表都给；
(b) E-cv 外推下重算同两张表；(c) h = 2 s、等弧长横向、航向 innovation、偏 gain 作为补充行。判定只用主口径（E-cc、4 s、全部 token）；
若主口径与 (a)(b) 的判定格不同，结论里写明并降一档可信度。

## 8. 图像消融（修正 7）

第 173 条不回答这件事（见第 2 节），其余条目里也没有对 SH30 的图像消融，所以做。一个 GPU 作业，经 pool 提交，VRAM 按 pool 历史申报。
- SH30-F-s0 / s1，navtest 全部 token（冻结 encoder 的 token cache 上推理约 1 分钟，取子集不省事）；ego、历史、指令输入原样。
- **KIN**：前视 token 换成另一个 log 里运动学最接近的 token（标准化的 v₀、a、v₀κ 最近邻）；**SHUF**：换成另一个 log 的随机 token。
  ID：不换，必须复现存档 plan（导出位姿最大差 ≤ 1e-3 m，否则作业判错）。不做全黑帧：token cache 在 encoder 之后，且 ego 运动也经画面进入模型（第 92 条），
  黑帧同时拿掉场景与视觉里程，KIN 才是只换场景的对照。
- 读数：三个版本的 gain 与 top-decile gain（x 仍是该 token 的真实日志 innovation），保留率 = 消融后 / 原样。
  登记读法：KIN 的 top-decile 保留率 ≥ 0.8 → 该轴上「视觉对变化几乎没有贡献」；≤ 0.5 → 变化主要由视觉带来；之间为部分。并与 BLIND 档对照。
- 不给消融后的 plan 打分（要占整台机器的 CPU，不在本 lane 范围）。

## 9. 判定线（主会话给定，逐轴；写死）

记 D = WA-JEPA 的 top-decile gain − SH30 的（两 seed 均值），S = 该轴高三档的 EPDMS 分差份额。
- **支持**（某轴）：D ≥ 0.10 且 CI 不含 0，**且** S ≥ 60%。
- **证伪**（某轴）：D 的 CI 含 0（或 D ≤ 0），**或** S < 40%。
- 其余为**部分**，逐轴报。
- 总判定：至少一个轴「支持」则假设**支持**（联合分位的份额并列报；若只有联合份额 ≥ 60% 而单轴份额都不到，记部分并写明）；两个轴都「证伪」则**证伪**；其余**部分**。
- P1 用第 4 节阈值单独报，P2 = 份额表，P3 = 第 7 节规则。虚构变化与图像消融不进判定，单独下结论。

## 10. 榜（修正 6）

- **navtest**：主榜，以上全部。
- **navhard stage 1**（450 个真实 token，G 帧，`lb_navhard/tab.npz` 里有日志未来的行）：只给 gain 阶梯（SH30、OT30、P0 / shipped、WA-JEPA 的存档位姿），不进判定。
  stage 2 是合成的离轨后续，没有日志未来，不做。
- **WOD val**：跳过。RFS 榜上对 WA-JEPA 没有可拆的分差（WA-JEPA 零样本反而落后，第 183 条），plan 存在另一套管线和格式里，不属于「代价很小」。
- **HUGSIM**：跳过。闭环，存档 plan 出自偏离日志的状态，没有与之对应的日志未来，innovation 无定义。

## 11. 执行

代码 `experiments/op_parity/scripts/innov.py`（`jevdrive.run.Run`，split 取自 `jevdrive.data.splits`，核数取 `jevdrive.common.n_cpus()` 且自限）；
结果 `results/innovation_gain.md` 与 `results/innovation_gain/`，图 `figs/innovation_gain/`（gain 阶梯、份额按分位、虚构变化）。box 上只写小表（< 几十 MB）。
CPU 步骤都是分钟级，不属于长跑。结论记第 206 条。
