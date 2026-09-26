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

## O. B2D 训练数据与 220 评测路线的重叠（只读，CPU）

Bench2Drive 官方训练集（base / full）与 220 条评测路线逐条比：同 town、同 scenario 类、触发点距离 < 30 m 的训练 clip 数；按榜单族实际用的训练集（TFv6 / BridgeDrive 用 LEAD 数据、SimLingo / BLUE 用 SimLingo 数据集，执行员先查清各自的数据来源）分别报。
判（描述性，不设门）：每条评测路线在各训练集里有多少个「近邻 clip」，与 G 的幽灵率做路线级相关（Spearman，路线整组 bootstrap）。正相关 → 背题的机制证据。

## W. 世界模型配对测试（GPU 探索，不训 policy）

问的是：在 latent 里预测未来的世界模型，能不能分清有没有 hazard、能不能推演不同动作的后果。这是「JEPA 世界模型 + openpilot 低成本训策略」的前置检查，过了才考虑下一轮的 latent MPC。

- latent：z = openpilot Cinque `temporal` ⊕ V-JEPA 2 `mean`（第 48 条同款抽取），5 Hz；数据：P5 v1 BA / PDM + P6 v0（night-queue-3 的 v1 到了再补），按路线分 train / test（5 折）。
- 动作：expert 下一步的（纵向加速度、横摆角速度），1 s 内 5 步。模型：8 步历史 → 预测未来 10 步 z 的 action-conditioned predictor（小 transformer，≤ 50 M 参数），3 seed。
- 读数用冻结在真实 z 上训的 probe（N2 的 a / b / c 与第 48 条的行人在走廊 probe），作用在**预测的** z 上：
  1. **配对分离**：给 x⁺ 与 x⁻ 喂同一段 expert 动作，预测 1 s / 2 s 后的 hazard probe 读数，判「能分清」= x⁺ 对 x⁻ 的读数区分 AUC ≥ 0.70（1 s 与 2 s 都要），且高于用天气 null 对算的同一 AUC + 0.10。
  2. **动作敏感**：在 x⁺ 的 hazard 帧上把动作换成「保持速度」对「刹停」、「保持车道」对「向 expert 绕行方向横移」，预测的碰撞 / 本车道障碍距离 probe 朝正确方向变化的比例 ≥ 70%。
  3. 两条都过 → 下一轮登记 latent MPC 选轨（对 Hydra 的词表做 rollout 打分），与 Hydra 在 P6 和闭环上比；1 不过 → 这套 latent 的世界模型推演不出 hazard，JEPA + openpilot 训策略这条路先搁置，写进 decisions。
- 预算：约 4–6 GPU·h。

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
