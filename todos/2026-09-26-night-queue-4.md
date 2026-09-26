# 队列 4（2026-09-27）：背题的签名（ghost test + 扰动）、材料包阶梯、世界模型配对测试

状态: registered（2026-09-26 18:00 CST，写于本队列的任何数字之前）。CARLA 部分接在 night-queue-3 的 lane B 之后（约 09-27 09:00–10:00），
不占 CARLA 的准备与 W 今晚就开，占用见「时间表」。

## 为什么有这一队列

文章的形状定了（讨论记录 2026-09-26 17:30）：**诊断**榜单高分由什么构成 → **材料包阶梯**证明「分数可以只靠配方往上推、能力不动」→
**配对与闭环考卷**证明榜单模型在需要判断的地方崩、配对差分激发的模型不崩。night-queue-3 给出了第三层与闭环的第一批数，但缺三样东西：

1. 「背题」没有可证伪的签名。现在的证据只是「看到因素后不反应」（第 46 条 T3：BridgeDrive 控车用的 target speed 通道翻转 0.2%），这也可以只是反应弱。
   而 B2D 上突发 hazard 近乎饱和（第 38 条），「开环里有没有行人开得一样」和「闭环里能过行人题」同时成立，最可能的解释之一是位置 / 路线记忆，
   另一种是开环回放的输入偏离了它自己的分布。**ghost test** 能分开这两者。
2. 材料包只有开环半边（第 40、53 条、N3 的阶梯零散在各处），闭环半边没有；也没有一个「我们自己背题」的阳性对照来证明 ghost test 真能测出背题。
3. 第 38 条（BLUE 在突发 hazard 上更好）在开环配对上是反向证据（BLUE − SimLingo −8.1 pp），它的闭环优势是否来自评测器改动没测；night-queue-3 的 CL10 在截止线上，今晚未必跑到。
   本队列的 G 把 BLUE / SimLingo 的原版路线 3 seed 一起跑了，CL10 的复核由这里保底。

## 通用规则

照抄 night-queue-3 的规则 1–10（判据先于数字、估时 + 2 倍停、3 seed、口径、适配器等价检查、**P7 执行层与归因**、闭环读数）与它的「一次性脚本与轮询」一节（每条 lane 一个链式脚本，`DONE` / `READY` / `ERROR` / `STATUS.md`，执行员 1–2 h 轮询一次，另挂一个只等 `ERROR` / `DONE` 的等待）。另加：

11. **世界变体的生成**复用已有的 hook：删 scenario 用 P5 x⁻ 与 P6 x₀₀ 的机制（actor 藏到地下**并删掉 PDM-Lite 读的 `active_scenarios` 登记**，`scripts/b2d_hooks.py`）；
    新写的 hook（触发点平移、actor 换类）先 smoke 10 个世界，PDM-Lite 在变体里的行为按 P6 的 expert 统计口径核一遍，再给考生用。
12. **位置窗口**：每条路线的「触发窗口」= 原版里 scenario 触发点沿路线 [−30 m, +10 m]；「对照窗口」= 同一路线上离任何 scenario 触发点 ≥ 80 m、路形同类（直行 / 路口，按路线折线曲率分）的 40 m 段，每条路线最多 2 个。

## G. ghost test 与扰动崩塌（闭环，CARLA）

**路线**：B2D 220 里 trigger 是生成的 actor 的路线——P5 的 10 个可见反应 family（每类 5 条）+ P6 的 8 类可用障碍（每类 5 条），约 90 条。
**世界变体**：
- `orig`：原样；
- `ghost`：scenario 删掉（规则 11），路线、TM seed、天气不变；
- `shift`：触发点沿路线前后平移 15 m（两个方向各一半路线，按路线号奇偶定）；
- `swap`：同类内换 actor（Accident ↔ ConstructionObstacle ↔ ParkedObstacle，行人 ↔ 自行车，cut-in 车型换一类），不能换的路线不进 swap。

**考生**：PDM-Lite（参照：删了登记后应当没有 ghost 反应）、TFv6、BridgeDrive、SimLingo、BLUE（作者执行层）、openpilot Cinque 原生、M-C 与 Q2 绕行 head（P7，都用 K 节的两折交叉拟合、每条路线由没见过它的读出开；绕行 head 只上障碍类路线）、阳性对照 = K3 的 seen 版（见过这条路线录像的读出）。
**seed**：`orig` 与 `ghost` 3 个 TM seed；`shift` / `swap` 1 个 seed，读数有信号再补。`orig` 里 night-queue-3 已跑过的（同考生、同路线、同 seed）直接复用，不重跑。

**读数与判据（写在数字之前）**：
1. **幽灵反应**（主读数）：`ghost` 世界里触发窗口内出现「减速 ≥ 3 m/s（窗口入口速度 − 窗口内最低速度）或横向偏离路线中心 ≥ 1.0 m」，且窗口内本车道前方 30 m 没有交通 actor（从 recorder 的 actor 表判，排除跟车）。
   幽灵率按路线整组 bootstrap（seed 先平均）。判格「位置记忆」= 幽灵率 CI 下界 > max(该考生同路线对照窗口的同一比率, PDM-Lite 的 ghost 幽灵率) + 10 pp。
2. **可见前反应**（副读数，只在 `orig`）：hazard 首次可见（recorder 的可见性相机）之前 ≥ 1 s 就开始减速（加速度 < −1 m/s² 持续 0.5 s）的 episode 比例；PDM-Lite 靠登记会提前，不作参照，只在考生之间比。
3. **扰动崩塌**：`shift` / `swap` 相对 `orig`，同路线同 seed 的 scenario 通过率（该 scenario 无碰撞且路线未 blocked）配对差；判格「崩塌」= 下降的 CI 下界 > 10 pp。1 seed 时只写方向。
4. **读法**：榜单族「位置记忆」且「崩塌」、我们的读出都不 → 文章第 3 部分的主张成立；榜单族不「位置记忆」但「崩塌」→ 写「对具体实例过拟合」而不是背位置；
   都不 → 「背题」这个说法撤掉，第 46 条 T3 的 0.2% 归为开环回放的分布偏离；K3 seen 版（阳性对照）不「位置记忆」→ ghost test 灵敏度不够，主读数不下结论。
5. **第 38 条复核**：BLUE − SimLingo 在 `orig` 突发 hazard family 上的通过率配对差（3 seed），CI > 0 支持第 38 条的闭环优势来自模型、≤ 0 则来自评测器 / 其他改动（BLUE 作者报的数用的评测设置与官方的差别由执行员先只读核对并写日志）。
   同时报每个考生在非 hazard 路段（对照窗口）的平均巡航速度与路线完成时间：hazard 通过率高而巡航速度显著低于 SimLingo 的，写「保守驾驶」而不是「反应更好」（2026-09-26 18:30 补，写于任何 G 数字之前）。

**规模**：ghost 90 × 3 × 9 考生 ≈ 2 400 次路线运行，shift / swap ≈ 1 400，orig 补跑约 700（作者执行层 seed 1、2 与阳性对照）；B2D 短路线按 4–8 worker·min 估，约 330–600 worker·h。
链式脚本先跑前 20 条路线 profiling，按实测重估；超 600 worker·h 按优先级砍（先砍 `swap`，再砍 `shift` 的考生到 TFv6 / BridgeDrive / BLUE / 我们的 M-C）。

- [F] 2026-09-26 17:25 CST 开工与分步估时（执行员 F；写于任何 G / X 数字之前）。代码：`scripts/nq4_hooks.py`（ghost / shift / swap 的世界 hook 与 20 Hz 自车 / actor 轨迹、可见性记录，
  由 `scripts/b2d_route.py` 按路线 XML 的 `nq4_world` 属性与 `B2D_NQ4_TRACE=1` 挂上，缺省时对别的 lane 没有任何作用）、`jevdrive/nq4_g.py`（路线与窗口清单、变体 XML、读数、小表）、
  `scripts/nq4_x_agent.py` + `jevdrive/nq4_x.py`（X 的几何路径与交叉拟合 head 导出）、链式脚本 `scripts/nq4_gk.sh`（tmux `jev:nq4-gk`），run 在 `$DATA_DIR/runs/nq4/gk/`。
  资源（批量前）：写代码不占资源；smoke 借 1 张卡 ≤ 2 个 CARLA server（server index 480–489，不与 lane B 的 300–479 重叠），核 ≤ 4，借哪张卡写在下一条。
  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | F0 | 代码：hook、读数、X agent、链式脚本 | 17:25–23:00 | Mac |
  | F1 | G-prep smoke：新 hook 10 个世界（ghost / shift / swap），PDM-Lite 按 P6 expert 统计口径核对 | 1 h | 借 1 卡 2 server |
  | F2 | X：交叉拟合 Q2 head（等 lane C 的 `q2/closed_loop_head/READY` 与 K 的 `route_split.json`，没到位先用占位接口）、3 条路线逐 tick 等价检查 | 1.5 h | 借 1 卡 2 server + GPU 6 小量 |
  | F3 | 各考生 2–3 条路线的 profiling（每路线 worker·h、每 tick CPU / GPU 分解），定每卡 server 数与 worker 数 | 1.5 h | 借 1 卡 2 server |
  | F4 | `nq4_gk.sh` 空跑（假臂验证队列、熔断、STATUS、出表），挂进 tmux 等门 | 0.5 h | CPU |
  | 批量 | 等 `runs/nq3/b/DONE` 且 lane B 的 server 全部退出；前 20 条路线 profiling 后重估 | 约 12–18 h（按 profiling 重估） | GPU 0–5、每卡 ≤ 6 server、90 核 |

## K. 材料包阶梯：分数动、能力不动

同一个 openpilot Cinque `temporal` 读出，逐项加榜单配方，每一级同时报榜单分和能力读数。

**开环半边（NAVSIM；大部分格子已有数，由 night-queue-3 Q6 的主表出）**：`ridge_late` → `cls_late` → Hydra 打分头（PDM 子分数，N3：84.2）→ + NAVSIM 2 Hz 协议 → + ego status 输入。
能力读数：I3 车辆翻转与选择性、WOD 21 个 nudge 帧、P6 bypass（Q1 / Q2 的口径）；P5 只在 night-queue-3 Q4b 兼容检查过线后才读（第 53 条）。缺的格子补上，3 seed。

**闭环半边（B2D 220，P7；新）**：

**先说一个我们自己的问题**：P5 / P6 的路线都取自 `bench2drive220.xml`，我们在 P5 v1 上训的 head 在 220 上闭环时，有一部分路线训练时见过录像——正是我们要在榜单模型上查的背题
（night-queue-3 规则 10 已补 seen / unseen 分组报）。所以 K 的每一级都用**两折交叉拟合**：把录过的路线按 scenario 类分层、按路线号奇偶分成 R₁ / R₂，
每一级训两个读出（只用 R₁ 的录像 / 只用 R₂ 的录像），闭环时每条路线由**没见过它**的那个读出来开（从未录过的路线两个都没见过，用 R₁ 读出）。
这样每一级的主读数都是干净的 unseen 分数；「见过」的分数只在 K0 与 K3 上额外跑（每条录过的路线再由见过它的读出开一次），得出背题溢价，同时充当 G 的阳性对照。

| 级 | 加什么 | 来源 |
|:--|:--|:--|
| K0 | `ridge_late` 轨迹 → P7 | 同 CL3 的配方，改成交叉拟合 |
| K1 | + TFv6 接口：route + target speed 分类头替代 waypoint（第 31 条：TFv6 高分来自这个表示） | 同一特征、同一训练数据重训 |
| K2 | + creep 规则与停车标志规则（照抄 TFv6 / SimLingo 作者 agent 的规则与阈值） | 规则 |
| K3 | + lead 门 × 常数刹车（G1c，navtest +0.53、直行帧激活 0%） | 规则 |

- K0–K3 的读出训练在今晚 GPU 6 空档做完（不占 CARLA），每级两份权重写 `runs/nq4/k/<级>/{R1,R2}/`，齐了写 `READY`；路线分折写 `runs/nq4/k/route_split.json`，先于任何训练。
- unseen 主读数每级 3 seed 跑 220；K0、K3 另跑 seen（录过的路线 × 3 seed）。能力读数：每级在 P5 v1 BA（开环，按同一交叉拟合出预测）、P6 bypass、I3 上各出一次。
- **判据（写在数字之前）**：
  1. 「材料包成立」= unseen 上 K0 → K3 的累计 DS 配对差 CI 下界 > 0，**且**每一级的 P5 行人翻转、I3 车辆翻转与 K0 的配对差都在 ±10 pp 内（CI 落在区间内才算「不动」，跨出区间写「动了」）。
  2. 「背题溢价」= 同一条录过的路线上 seen − unseen 的 DS 配对差 CI 下界 > 0（K0、K3 各判，按路线 bootstrap，seed 平均）；
     seen 版在 G 的 ghost test 上应当「位置记忆」（阳性对照，见 G 读法）。
  3. 背题溢价的 CI 覆盖 0：写「这套路线的录像不带可背的位置信息（对线性读出而言）」，阳性对照失效，G 的主读数只写描述；
     night-queue-3 里我们 head 的合并分数与 unseen 分数差不多也由此得到说明。
- 读法：1 成立 → 文章第 2 部分的因果证据（分数由配方推高、能力不动）；1 里 DS 涨而能力也涨 → 那一级配方不是纯 hack，单独写。

- [K] 2026-09-26 17:30 CST 开工与分步估时（执行员 K，K-prep；写于任何 K 数字之前）。代码：`jevdrive/nq4_k.py`（分折、标签、拟合、导出、numpy apply、规则、离线等价检查），
  闭环接线是对 `scripts/nq3_cl_server.py`（新 arm `k0`–`k3`）与 `scripts/b2d_zeroshot_agent.py`（按路线选折、K2 / K3 的规则后处理）的增量改动，lane B 的既有 arm 行为不变。
  产物在 `$DATA_DIR/runs/nq4/k/`，齐了写 `READY`。资源：GPU 6 空档（显存 ≤ 16 GB），核 `taskset -c 146-149`（4 核；lane D 的 Q4a 收工后可扩到 146-153），
  BLAS / OMP 线程 = 4；闭环等价检查借 1 张卡 ≤ 2 个 CARLA server（server index 490–499，不与 lane B 的 300–479、F 的 480–489 重叠），借哪张卡开跑前在这里写一行。
  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | K0 | 代码（上面几个文件） | 17:30–21:30 | Mac |
  | K1 | `route_split.json`（下一条的规则），先于任何训练，commit 后 F 的 X 也用它 | 18:15 前 | CPU |
  | K2 | Cinque 的 lead 输出：P5 v1 BA 全部 858 条流、P6 v0 全部 605 条流重跑一次（`scripts/p5_openpilot.py --arrays temporal lead lead_prob`），`temporal` 对已存 `op_streams_vis` 逐位相同才用 | 1–1.5 h | GPU 6 ≤ 4 GB，2 进程 |
  | K3 | K1 的 route 标签（每帧的 B2D 稠密路线）；K0 / K1 / M-C（K3 的常数）× {R₁, R₂, 全量} 拟合；GPU 与 CPU `eigh` 对照（预测差 ≤ 1 mm）；全量 K0 对 lane B `heads.npz` 的复现 | 0.5 h | GPU 6 ≤ 8 GB |
  | K4 | 开环导出：P5 v1 BA（交叉拟合）、I3、P6 v0 考卷帧上每级的 (n, 20, 2) 预测 | 0.3 h | GPU 6 |
  | K5 | 闭环等价检查（规则 8）：4 级 × 3 条路线，每个请求 dump，离线重算逐位比；规则层按逐 tick 日志离线重放逐位比 | 2 h | 1 卡 2 server |
  | K6 | 写 `READY`、交接 | 0.2 h | |
  目标 02:00 前写齐 `READY`（todo 的截止 03:00）；任何一步超估计 2 倍就停，写 `runs/nq4/k/ERROR`。
- [K] 2026-09-26 17:40 CST **操作性选择**（写于任何 K 拟合与任何 K 数字之前；此前只读过 night-queue-3 CL 节、第 31 / 33 条、G1c 代码、LEAD `730bc1a` 与 SimLingo 作者 agent 的规则代码，以及 P5 v1 BA 各路线的帧数）。
  1. **分折（`route_split.json`）**。录过的路线 = P5 v1 BA 里出现的全部 216 条 base 路线（151 条来自 P4、65 条只来自 P5；其中 170 条在 `bench2drive220.xml` 里，46 条不在 220 里，是 P5 v1 从别的路线集补的）。
     分层的类 = 路线在 220 里的 scenario 类型（220 每条恰好一个）；不在 220 里的 46 条用 P5 的 family（`pairs.csv`，含 `Light`）。字面的「路线号奇偶」在 8 个类里把全部录过的路线分到同一边
     （例如 OppositeVehicleTakingPriority 5 / 0、ConstructionObstacle 2 / 0），那样另一折的读出连这个 scenario 类都没见过，unseen 就混进了「类没见过」，所以改成**类内按路线号排序后的奇偶位**：
     每个类里先排录过的路线（按路线号升序）、再排 220 里没录过的路线（同样升序），沿这个顺序交替分给 R₁ / R₂；起始折在「路线数为奇数的类」之间按类名顺序轮换，使两折总数差 ≤ 1。
     这样 220 的每条路线都有折标签（F 的 X 用 P6 路线的折标签做它自己的交叉拟合），而 K 的读出只在录过的路线上有 seen / unseen 之分：录过的路线由另一折的读出开（unseen），
     从未录过的 50 条路线由 R₁ 的读出开（todo 原文）；seen 版 = 录过的路线由自己那一折的读出开。
  2. **seed**。四级的读出全是确定性拟合（闭式 ridge、从零初始化的凸 logistic 回归、闭式 M-C），同一折重拟合逐位相同，所以每级每折只有一份权重（Q6 的口径标「确定性」）；
     3 seed 是闭环的 TM seed，由 F 的链跑。不另造「内层 CV 置换」的伪 seed。
  3. **K0**：`jevdrive.nq3_cl.export` 的 prior 部分原样（`ridge ego` + Cinque `temporal`（`op_streams_vis`）上的 `ridge_late`，λ 由 `ridge_cv` 的 4 折按路线分组内层 CV 选），
     训练行 = role == train 且 base 路线在 R_k 里的行。全量版（全部 train 行）必须复现 lane B 的 `runs/nq3/b/heads/heads.npz`（We、Wp、统计量，预测差 ≤ 1e-3 m），不过就停。
  4. **K1（TFv6 接口）**。同一输入（96 维 ego 历史 + 4 维 intent，Cinque `temporal`）、同一训练行，两头都是 late fusion（先 ego、再 `temporal` 修残差 / 加 logit 偏置，与 `ridge_late` / `cls_late` 同构）：
     - **route**：TFv6 的 10 个 checkpoint（LEAD `num_route_points_prediction` 10，第一个点距原点 2.5 m、之后每 1 m 一个，`smooth_path`）。标签取该帧录制时的 B2D 稠密路线（recorder 的 `route.json`，1 m 点列），
       从最近点起变到该帧后轴系（x 前 y 左），第一个点 = 沿路线第一个离原点 ≥ 2.5 m 的点，之后按弧长每 1 m（1 m 点列上弧长与 LEAD 的圆弧截点差 < 1 cm）；回归头 = `ridge ego` → `ridge_late`，与 K0 同一 `ridge_cv`。
     - **target speed**：TFv6 的 8 档 [0, 4, 8, 10, 13.89, 16, 17.78, 20] m/s，two-hot 软标签（LEAD `encode_two_hot` 的线性插值，> 20 记 20），解码 = 期望（`decode_two_hot`）。
       标签速度 = expert 未来轨迹 0.75–1.25 s 段的平均速度（P5 BA 没有 expert 的目标速度指令，取约 1 s 后的实际速度作代理：TFv6 的目标速度是 expert 当下的指令，实际速度落后约 1 s）。
       分类头 = `planner.ce_solve`（软标签交叉熵 + L2，L-BFGS），先 ego 后 `temporal`（偏置 = ego 头的 logit），λ 在 `planner.LAM_CLS` 上按 80 / 20 路线分组留出（`GroupShuffleSplit`，random_state 0）的交叉熵选，再在全部训练行上重拟合。
     - **交给 P7 的轨迹**：折线 [原点, 10 个 checkpoint] 按弧长 s(t) = v̂·t 取 0.25 … 5 s 的 20 点，超出最后一个 checkpoint 沿末段方向直线外推。TFv6 作者 PID 的 brake 条件（v̂ < 0.01 或 v / v̂ > 1.1）
       不照搬：执行层按规则 9 一律是 P7（纵向由 P7 的 accel 模式跟这条等速轨迹），这是表示的替换，不是控制器的替换。开环读数用同一条 (20, 2) 轨迹。
  5. **K2（规则，照抄 LEAD `730bc1a` `sensor_agent.py` 的 README 95 分配置）**，作用在 P7 输出的油门 / 刹车上，顺序同作者（先 creeping、再 stop sign）：
     - **creeping**（`ForceMovePostProcessor`）：速度 < 0.1 m/s 连续 > 1100 tick 后强制 20 tick 油门 ≥ 0.4、刹车 0；安全盒（车体系 x ∈ [2.45, 4.95] m、|y| < 0.85 m、z ∈ [0.5, 1.5] m）里有东西就改为刹停并把 20 tick 重置。
       作者用 LiDAR 点判安全盒；我们的 agent 没有 LiDAR，改用 CARLA 里 vehicle / walker / static prop 的包围盒与安全盒相交（特权替身，只在 creeping 进行中查询；漏掉的是建筑、护栏等非 actor 几何）。
       SimLingo 作者 agent 的 creep（800 tick / 15 tick / 0.4，无安全盒）不用：K1 用的是 TFv6 的接口，规则跟着 TFv6。
     - **stop sign**（`StopSignPostProcessor`）：阈值 1.0 m、清除冷却 120 tick、减速计数 40 tick、减速时油门上限 0.1，逻辑逐行照抄。作者的停车牌框来自网络检测，训练标签是「影响本车且未清除的停车牌」的 trigger volume 中心；
       我们没有停车牌检测，用特权替身：`traffic.stop` actor 的 trigger volume 中心（车体中心系）落在 TFv6 的 BEV（x ∈ (−32, 64)、y ∈ (−40, 40) m）内、且 trigger volume 覆盖本车前方的稠密路线点、且未被本规则清除过的，当作检测到。
       这比作者网络的检测更准（上界），写作时照写。
     - Kalman（作者的第三项）只滤 GPS 给网络的 target point，我们没有这个输入，不适用。
  6. **K3 = K2 + G1c**：plan 加 g3 · c，g3 = `real_g1.g3`（Cinque 同一步的 lead / lead_prob 输出，v_ego = ego 输入里 t0 的速度模长，TTC 2–6 s 的映射原样），
     c = `real_g2._const` 的定义：Σ g3·Δ / Σ g3，Δ = M-C `pair`（Qwen `L18_last` ⊕ `temporal`，`reactivity_mc.fit_fold` 的闭式解、λ 网格与内层 CV 原样），只取纵向分量；
     交叉拟合版的 Δ 与 Σ 都只用 R_k 的数据（R_k 的配对行拟合、R_k 的全部行求加权平均）。表示是 route + speed，所以「纵向」按沿轨迹弧长加：s′(t) = s(t) + g3·c_x(t)，截到 ≥ 0 并取累计最大（不倒车）；直路上与 G1c 的 x 向加法相同。
     BA 与 P6 没有存 lead 输出，按 K2 步重跑一次；I3 用已存的 `hugsim_pairs/op_streams_lead`。
  7. **开环导出**（供 F 的链出表，本 lane 不判格）：P5 v1 BA 的每一行由没见过它路线的读出预测（交叉拟合）；I3 与 P6 v0 考卷帧当作不进拟合、不进标准化的附加行（`elicit_i3` / `nq3_q1` 的做法），
     I3 与未录路线用 R₁ 读出（另存 R₂ 版），P6 帧按它的路线走第 1 条的规则；K2 的开环预测 = K1（规则只在闭环里起作用）。全量 K0 另存一份作代码路径对照。
  8. **等价检查（规则 8）**：3 条路线 = 一条停车牌路线（VanillaNonSignalizedTurnEncounterStopsign）、一条前车急刹（HardBreakRoute，g3 会开）、一条行人（DynamicObjectCrossing），各取 220 里录过的、路线号最小的一条，
     4 级各跑一遍（K0 / K1 unseen，K2 / K3 unseen），每个请求 dump；离线用同一份 JPEG、新 Cinque session 同顺序步进、`nq4_k` 的 apply 重算，轨迹、v̂、g3 逐位相同；规则层把逐 tick 的输入（速度、P7 的油门 / 刹车、停车牌替身、安全盒）
     记进 `ticks.jsonl`，离线重放规则类得到的油门 / 刹车逐位相同。不过就停。

## O. B2D 训练数据与 220 评测路线的重叠（只读，CPU）

Bench2Drive 官方训练集（base / full）与 220 条评测路线逐条比：同 town、同 scenario 类、触发点距离 < 30 m 的训练 clip 数；按榜单族实际用的训练集（TFv6 / BridgeDrive 用 LEAD 数据、SimLingo / BLUE 用 SimLingo 数据集，执行员先查清各自的数据来源）分别报。
判（描述性，不设门）：每条评测路线在各训练集里有多少个「近邻 clip」，与 G 的幽灵率做路线级相关（Spearman，路线整组 bootstrap）。正相关 → 背题的机制证据。

- [CX] 2026-09-26 17:15 CST 开工口径：评测单位是 `bench2drive220.xml` 的一个 `route` 及其原始 scenario 触发点。对官方 B2D base / full、TFv6 / BridgeDrive 实际使用的 LEAD、SimLingo / BLUE 实际使用的 SimLingo 数据集，仅以有论文、README 或代码 manifest 行级出处的成员关系为准；同一训练 clip 只要有一个元数据触发点与评测触发点的 CARLA town 标识和原始 scenario 类字符串完全相同，且 CARLA 世界平面坐标的欧氏距离严格小于 30 m，就对该路线计一个近邻；多触发点取 clip ID 并集，不重复计数。缺 town / 类别 / 坐标的样本排除并报数，不推断；族数据集查不到就写「未查清」。只取元数据，预计下载量超过 5 GB 即停。估时 2 h（出处核查 45 min、元数据盘点与取得 45 min、匹配 QA 与报表 30 min），超过 4 h 即停并报问题。

## W. 世界模型配对测试（GPU 探索，不训 policy）

问的是：在 latent 里预测未来的世界模型，能不能分清有没有 hazard、能不能推演不同动作的后果。这是「JEPA 世界模型 + openpilot 低成本训策略」的前置检查，过了才考虑下一轮的 latent MPC。

- latent：z = openpilot Cinque `temporal` ⊕ V-JEPA 2 `mean`（第 48 条同款抽取），5 Hz；数据：P5 v1 BA / PDM + P6 v0（night-queue-3 的 v1 到了再补），按路线分 train / test（5 折）。
- 动作：expert 下一步的（纵向加速度、横摆角速度），1 s 内 5 步。模型：8 步历史 → 预测未来 10 步 z 的 action-conditioned predictor（小 transformer，≤ 50 M 参数），3 seed。
- 读数用冻结在真实 z 上训的 probe（N2 的 a / b / c 与第 48 条的行人在走廊 probe），作用在**预测的** z 上：
  1. **配对分离**：给 x⁺ 与 x⁻ 喂同一段 expert 动作，预测 1 s / 2 s 后的 hazard probe 读数，判「能分清」= x⁺ 对 x⁻ 的读数区分 AUC ≥ 0.70（1 s 与 2 s 都要），且高于用天气 null 对算的同一 AUC + 0.10。
  2. **动作敏感**：在 x⁺ 的 hazard 帧上把动作换成「保持速度」对「刹停」、「保持车道」对「向 expert 绕行方向横移」，预测的碰撞 / 本车道障碍距离 probe 朝正确方向变化的比例 ≥ 70%。
  3. 两条都过 → 下一轮登记 latent MPC 选轨（对 Hydra 的词表做 rollout 打分），与 Hydra 在 P6 和闭环上比；1 不过 → 这套 latent 的世界模型推演不出 hazard，JEPA + openpilot 训策略这条路先搁置，写进 decisions。
- 预算：约 4–6 GPU·h。
- [E] 2026-09-26 17:30 CST 开工与分步估时（写于任何 W 数字之前）：代码 `jevdrive/nq4_w.py`，链式脚本 `scripts/nq4_w.sh`，运行目录 `$DATA_DIR/runs/nq4/w/`。
  分步：(1) 缺的 V-JEPA 2 `mean` 补抽（P5 v1 PDM 的 obs 行 17 340、P6 未抽的约 18 500 行，每行 3 相机，约 10.8 万个 4 帧 clip，按 nq3 实测 ~100 clip/s 估 20–30 min，先在 BA 已存行上做抽取等价检查）；
  (2) 标签与动作（CPU，10 min）；(3) 子集 profile + 一次短训练 smoke（40 min）；(4) 3 seed × 5 折训练与 probe 读数（链式，估 1.5–2.5 GPU·h）；(5) 出表（10 min）。合计约 3–4 h 墙钟、≤ 3 GPU·h；超 2 倍（8 h 墙钟或 6 GPU·h）停。
  GPU 6、显存 ≤ 16 GB、核 200–207（`taskset`，线程数 8）。按 2026-09-26 17:25 用户新规，smoke 过后链式脚本交 Codex 看护（`tmp/2026-09-26-codex-handoff.md`），P3 可行性留给 Opus 执行员。
- [E] 2026-09-26 17:40 CST 操作性选择（写于任何 W 数字之前；smoke 只读训练折内的 loss，不读 probe / 判据）：
  - **数据宇宙**：`carla_p5v1_ba` 与 `carla_p5v1_pdm` 的 `role == obs` 行（5 Hz，含 plus / minus / null 世界），`carla_p6` 全部行（5 Hz，七种世界）。P5 的 train 行是 2.5 Hz，不进。
    z = Cinque `temporal`（`op_cinque_vis`，512 维）⊕ V-JEPA 2 ViT-L `mean`（4 帧 clip、三相机拼接，3 072 维；BA 与 P6 已存的直接用，缺的用同一模型与 transform 补抽）。
    每个 run 按帧号切成间隔恰为 4 tick（0.2 s）的连续段，窗口 = 18 个连续帧（8 帧历史，锚点 t₀ 是第 8 帧，10 帧未来），步长 1。
  - **动作**：锚点前后每帧的 (a, ω) = ((v_{t+1} − v_t)/0.2 s, wrap(ψ_{t+1} − ψ_t)/0.2 s)，v、ψ 取 `pose.jsonl` 在该帧与 +4 tick 的平面速度与 yaw（rad）。历史 token 带上一步动作与 v_t，未来 query token 带该步动作。
  - **分折**：按 base 路线（三个集合合在一起，同一路线不跨折）5 折，seed s 决定路线排列、模型初始化与 batch 顺序；每折再按 seed 留 10% 训练路线作 inner-val，只用于选 checkpoint 与 ridge 的 λ。
  - **模型**（约 23 M 参数）：输入 z 按训练折统计量逐维标准化；线性投到 d = 512；8 个历史 token + 10 个未来 query token（动作 → MLP 嵌入 + 可学位置编码），6 层 pre-LN Transformer encoder（8 头，FFN 2 048，GELU，dropout 0.1，全注意力），
    query 位置输出 512 → 3 584 的 Δz，预测 ẑ_{t+h} = z_t + Δz_h（h = 1…10，一次并行出，不自回归）。loss = 两个块（Cinque 512 维、V-JEPA 3 072 维）各自的逐维 MSE 取平均后再等权平均。
    AdamW（lr 3e-4、wd 0.05、β 0.9 / 0.95），batch 256，500 步 warmup + cosine，bf16 autocast，梯度裁剪 1.0；总步数在 smoke 里按训练折 loss 定一次（写在下一条 [E]），之后 15 次训练一律不变；每 1 000 步算 inner-val loss，取最低的 checkpoint。
  - **probe**（每折只在训练折路线的**真实** z 上训，同一套标准化）：N2 的 a / b / c（`night2_labels.parquet`，P6 同名文件）；新增三个 ego 坐标系的 GT 标签（`actors.npz`，可见 = 与 ego 高差 ≤ 5 m，同 N2）：
    `ped` = 有行人在 ego 前方 0 < x ≤ 30 m、|y| ≤ 4 m（第 48 条的「行人在走廊」，走廊宽度照 P1）；`occ` = 有车或行人在 0 < x ≤ 30 m、|y| ≤ 1.75 m；`d_front` = 0 < x ≤ 40 m、|y| ≤ 1.75 m 内最近 actor 的 x（没有记 40 m）。
    分类 probe 用 N2 的 `logreg_auc` 同款（标准化 L2 logistic，C = 1，L-BFGS）；`d_front` 用 ridge，λ ∈ {1e1, 1e2, 1e3, 1e4, 1e5} 按 inner-val 选。
  - **判据 1 配对分离**：测试折里 x⁺ / x⁻ 对 = P5 两个集合的 `obs.parquet`（按 k 对齐）与 P6 的 x10 / x00（同 base、seed、k），锚点 k 两边都有完整 18 帧窗口；P6 只取 k ≥ t_vis。
    两边都喂 **x⁺ 的** expert 动作（历史与未来）与 x⁺ 的 v。hazard probe 按类：行人 4 个 family → `ped`；cut-in 3 个 family → `occ`；P6 障碍 → `a`。
    HardBreakRoute（x⁻ 里前车还在）、OppositeVehicleRunningRedLight 与 Light 没有对应 probe，只描述不判。天气 null 对 = P5 的 `null.parquet`（x⁺ 对 null）与 P6 的 x10 / wnull（seed 0），同一流程；
    AUC = 测试折合并后 x⁺ 读数对 x⁻ 读数的 ROC AUC（每个 seed 五折 OOF 合并），null 的 AUC 取 max(AUC, 1 − AUC)（天气可能把读数往任一方向推）。
    每类判「能分清」= 3 seed 平均 AUC 在 1 s 与 2 s 都 ≥ 0.70，且都 ≥ 同类 null AUC + 0.10；**判据 1 过 = 行人、cut-in、障碍三类都过**，只过一部分按类写。
    同时报（描述、不进判格）：persistence（probe 读真实 z_{t₀}）与 oracle（probe 读真实 z_{t₀+h}）的同一 AUC、成对胜率、按路线 bootstrap 的 CI、seed 极差。
  - **判据 2 动作敏感**：hazard 帧 = 测试折 x⁺ 世界（P5 plus 与 P6 x10）里锚点真实 `occ = 1` 且 v ≥ 3 m/s 的窗口。
    纵向：「保持速度」= 未来 10 步 a = 0、ω 取 expert 的；「刹停」= a = −5 m/s² 直到按动作积分的速度到 0（之后 0）、ω 同上；读 `d_front` 预测，正确 = 刹停 > 保持。
    横向（只 P6 x10、expert 模式 bypass_L / R / wait_then_bypass、锚点 k < t_div_lat 即 expert 还没横移）：「保持车道」= expert 的 a、ω = 0；「横移」= expert 的 a、ω 先 +s·ω₀ 1 s 再 −s·ω₀ 1 s，ω₀ = min(3.5 m / v, 0.6 rad/s)，s 是 expert 绕行方向的 yaw 符号
    （CARLA 左手系，左绕 = ω < 0；在读任何 probe 之前先用 expert 在绕行段的实测 ω 核一次符号）；读 `d_front` 预测，正确 = 横移 > 保持。
    判「动作敏感」= 纵向与横向的 2 s 正确比例（3 seed 平均）都 ≥ 70%；1 s 与 `occ` 读数只描述。
  - 读法照 W 节第 3 条；另：判据 1 在 persistence 上就已过而 oracle 更高时，写明「分清」有多少来自历史里已经看见的 hazard（描述，不改判格）。
- [E] 2026-09-26 17:32 CST 借卡：GPU 6 只剩约 5 GB（92.5 / 97.9 GB、100% 利用），放不下 W；借空闲的 GPU 5（26 MiB、0%），已写 `gpu-plan.md`。链式脚本每一步开始前重新选卡：GPU 6 空出 ≥ 18 GB 就回 6，否则只用 3–5 里 < 5 GB 且没有 CarlaUE4 进程的卡，都没有就等；CARLA 起来后下一步自动让出（单步 ≤ 40 min）。
  V-JEPA 抽取等价检查（BA 已存 48 行重抽）：cos ≥ 0.9999、最大相对差 0.7%（bf16 batch 噪声），过。

## X. 判断与执行拆开：模式头 → 几何路径 → P7（闭环，CARLA；2026-09-26 18:30 补，专家回复第 6 问）

P7 的横向过了验收（cross-track p95 0.11 m、lateral ratio 0.9），没过的是纵向，所以只要读出给出绕行轨迹，执行器已经有了。这里加一个把判断和执行完全拆开的臂：
Q2 的模式头（交叉拟合的 unseen 版）判 bypass-L / R 时，按 PDM-Lite 的几何生成路径（障碍前 50 m 内把路径平移到相邻车道中心，车道取 CARLA map），交给 P7；判 stop / wait 时路径不变、目标速度降到 0。
触发来自我们的感知，不读 `active_scenarios` 登记。路线 = G 的障碍类约 40 条 × 3 seed，与 CL5（学出来的横向轨迹）、CL5d（desire）、PDM-Lite 同路线配对。
读法（写在数字之前）：X 的障碍类 SR 不低于 CL5（配对差 CI 下界 > −10 pp）→ 第三层的瓶颈是判断，执行用几何就够，文章里执行层写成可替换模块；X 明显低于 CL5 → 学出来的轨迹带了几何路径没有的东西（例如与对向车的时机），单独分析。
在 G + K 闭环链里排在 K1、K2 之前；约 120 次路线运行。

## P. 行人的真实数据（2026-09-26 18:30 补，专家回复第 1 问）

- **P1 数事件（CPU，今晚可开）**：OpenScene / nuPlan（box 上已有的部分，执行员先盘点）按 GT 框数「走廊 ±4 m、30 m 内有行人」的独立事件（同一行人连续出现只算一次），按城市与 ego 速度档分开报。
  判：独立事件 ≥ 2 000 → 开 P2；< 2 000 → 写「真实 log 里行人正例不够」，这一条只剩 P3。
- **P2 匹配对训练（等 P1）**：有行人帧与「同 ego 速度档、同路形、同地点附近、走廊干净」的帧做倾向匹配，匹配对的 human future 之差当 Δ 目标，与 night-queue-3 Q4a 的真实帧零约束一起训 M-C 式的 Δ。
  判据照第 53 条的能力包：Hydra + Δ 的 navtest PDMS 掉 ≤ 1.0，且在 P3 的真实外观行人考卷（有的话）或 P5 行人上翻转 ≥ M-C 的 80%。
- **P3 真实外观的行人考卷（GPU，可行性先行）**：drivestudio（OmniRe）在 10 个带走廊行人的 WOD 场景上重建，删掉真实存在的行人得到 x⁻，原 log 重渲染为 x⁺，未改动的重渲染对作 null。
  可行性门：null 对上 openpilot `ridge_late` 的误翻率与 I3 的 null 同一水平（≤ 7%），且被删行人的区域没有肉眼可见的残影（每场景存对比图）。过门后扩到 ≥ 60 个场景，成为 I3-ped，所有考生都上。
- 两条都不通：论文里写「真实数据上行人通道 = 外接检测（metric depth 放置，N5）+ 规则」，作为可部署的下限与限定。

- [CX] 2026-09-26 17:15 CST 开工口径：扫描 box 上 `$DATA_DIR/datasets/navsim/navsim_logs/<split>/` 实际存在的 OpenScene / nuPlan log pickle，先把 split 与 log 盘点写入运行日志。对每个 GT 行人框中心（不含骑行者），投影到从当前 rear-axle pose 开始、按 logged future 归迹弧长截到 30 m 的前向折线；要求投影弧长在 [0, 30] m 且绝对横向距离 ≤ 4 m。同一 track token 在连续采样帧中持续合格只算一次；离开走廊或不合格后再进入则是新事件。城市优先取 log / scenario 的 map/location 元数据；速度取事件起始帧 logged velocity 的模，速度档为 [0, 2)、[2, 5)、[5, 10)、[10, ∞) m/s。每个事件在起始时向 CSV 写一行，并做手工样本和汇总一致性核对。估时 90 min，超过 180 min 即停并报问题；只用 CPU 2 核并限制在 196–199 号核中。

## E. 专家汇总的补充

`tmp/2026-09-26-round-expert-brief.md` 的第 4 问（第 38 条）在 G 的第 5 条读数出来后补一段；G 的主读数、K 的判格出来后，各补一条到「主要结论」，标日期。

## 时间表与资源（box 时钟 CST）

| lane | 内容 | 卡 | 核 | 何时 |
|:--|:--|:--|--:|:--|
| W | 世界模型配对测试 | 6（≤ 16 GB，与 night-queue-3 C / D 共卡） | 8（200–207，T2 收工后空出） | 今晚开，约 6 h |
| O | 重叠只读核查 | — | 2 | 今晚开，约 2 h |
| P1 | 行人事件盘点 | — | 2 | 今晚开，约 1 h；P3 可行性等 W 收工后用 GPU 6 |
| K-prep | K0–K3 两折读出的训练与导出（numpy apply，与 CL 的 head 同格式）、K2 / K3 规则的 agent 接线与 3 条路线等价检查 | 6 空档 | 4 | 今晚 22:00 后（lane D 的 Q4a 收工后），目标 03:00 前写齐 `READY` |
| G-prep | 世界变体 hook（shift、swap）、路线与窗口清单、读数模块；smoke 等 CARLA 空档（lane A 在 v1 结束的 03:00 后有 2 个 server 的空当） | 03:00 后借 1 张卡 ≤ 2 server | 4 | 今晚写代码，03:00 后 smoke |
| G + K 闭环 | 一条链：G 的 `ghost`（先 TFv6 / BridgeDrive / BLUE / SimLingo / M-C / 阳性对照）→ K0、K3 的 unseen 与 seen → G 的其余考生 → K1、K2 → `shift` / `swap` | night-queue-3 lane B 写 `runs/nq3/b/DONE` 后接管 0–5（每卡 ≤ 6 server） | 90 | 09-27 约 09:00 起，按 profiling 重估，约 12–18 h |

- 今晚新增的核：W 8 + O 2 + K-prep 4 + G-prep 4 = 18，加上 night-queue-3 的 160，到 178 / 175，所以 K-prep 与 G-prep 等 lane D 的 Q4a（约 22:00）或 lane A 的 v1（约 03:00）收工再开；W、O 现在开。
- 线程：G + K 闭环最多 36 个 server，规则同 night-queue-3（起新 server 前读 `pids.current`，> 17 000 就等）。
- 盘：闭环只存 per-route json 与 20 Hz 自车 / actor 轨迹（幽灵反应要用），不存图，估 < 20 GB。

## 分派

| 执行员 | lane | 先做 |
|:--|:--|:--|
| 新执行员 E | W | 写 `nq4_w.sh`，今晚开 |
| 新执行员 E（W 之后） | P1 → P3 可行性 → P2 | P1 今晚与 W 并行 |
| 新执行员 F | O → G-prep → G + K 闭环链 | O 今晚开；G-prep 写代码；接管 lane B 的卡后写 `nq4_gk.sh` 的优先级队列（含 profiling 重估与砍单规则） |
| night-queue-3 的执行员 D（Q4a 收工后）或 C（`READY` 交付后） | K-prep | 谁先空谁领，领的时候在本文件写一行 |

- 2026-09-26 17:30 CST [main] 实际分派：P1 与 O 交给本机 Codex 执行员 CX（`tmp/2026-09-26-codex-brief.md`，只计数、不改判据，结论由 main 复核）；
  W → P3 可行性由 Opus 执行员 E；G-prep + X 的 agent 与 `nq4_gk.sh` 链由 Opus 执行员 F；K-prep 由 Opus 执行员 K（不等 night-queue-3 的 C / D 空出来）。
  准备好的批量链（G + K 闭环）在 lane B 交卡后启动，批量期的看护可交给 Codex。

每节结果写回下面「结果」，结论回填 decisions（新条或就地修正，标**待定**），E 节按上面补专家汇总。

## 结果

（待写）

### CX 产出，待 main 复核（2026-09-26 17:29 CST）

| 项 | 本轮核查结果 | 判格 | 待处理问题 |
|:--|:--|:--|:--|
| P1 | 复用既有盘点：test 147 个 log，trainval 1 310 个 log；草稿尚未运行计数 | 未出判格 | logged future 无非零线段、尾部不足 30 m 时的处理未规定；草稿端点截断会计入身后与 30 m 端点外的点，先停下确认 |
| O | 已核查训练数据出处及已有元数据；[来源与阻塞表](../research/results/nq4/o/summary.md) | 描述性任务，无门槛；近邻数尚未计算 | B2D clip 到触发点的映射未核实；BLUE gate 数据尚未公开；不能以路线模板替代实际训练 clip 或把未知写为零 |

按任务书第 5 条暂停，问题详见 `tmp/2026-09-26-codex-status.md`。不启动 P2，不做与 G 的相关，不修改 decisions。
