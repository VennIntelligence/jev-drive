# op-adapt L 证据链审阅与下一步（2026-10-01）

状态：独立审阅，只读。没有启动实验，没有改 decisions，没有碰 box 上的进程与调度表。
范围：decisions 第 77–79 条，[log expert audit](../todos/2026-10-01-log-expert-audit.md)、[op-adapt L 预登记](../todos/2026-10-01-op-adapt-L-prereg.md)、[后续检查](../todos/2026-10-01-op-adapt-L-followup.md)、[RFS 诊断](../todos/2026-10-01-op-adapt-L-rfs-diagnosis.md)、[B2D 预登记](../todos/2026-10-01-op-adapt-L-b2d-prereg.md)，以及 `research/results/{log-expert-audit,op-adapt-L}/` 下的 CSV。
B2D 闭环批次的结果按要求当作未知处理（box 上已有 `results/summary.md`，我没读）。

名词（每个只在这里解释一次）：**capture rate**（捕获率）指 plan 复现人类机动「至少一半」的帧占比；**RFS**（Rater Feedback Score）是 WOD-E2E 榜单分，plan 落进某条 rater 轨迹在 3 s / 5 s 的 **trust region**（纵向 ±4 m、横向 ±1 m 的信任区，5 s 处放大）就拿该 rater 的分；**×1.06** 指 plan 的 x 乘 1.06 的纵向校准，7.92 的 test 提交用的就是它；**dw** 是蒸馏权重；**stay** 是「停着且人类接下来 4 s 也不动」的帧，stay 上的 **false start**（假起步）是 plan 4 s 位移 ≥ 3 m；**O** 指原模型 Cinque；`main` 是 op-adapt L 选出的配置（stage 4 解冻 + intent adapter + dw 3）；**pace-matched control**（配速对照）是把无感知路线 base 开慢到被测臂同样的均速；**DiD** 是「被测臂对自己配速对照的差」减去「drive 对自己配速对照的差」。

## 0. 结论

1. 77–79 的核心数字都能从 CSV 复现，计算层面没有错。问题出在**解读层面**，有三处明显说过了头：
   (a) RFS 的「空间」是拿 val 的原始 RFS 去比 test 榜首，算 headroom 时也没有扣掉 ×1.06；扣掉以后，就算把 logged future（日志里人类实际开出的未来轨迹）原样当 plan 交上去，在 479 个 val rater 帧上对 O×1.06 也只多 +0.11 [−0.12, +0.32]（帧均值）。
   (b) 说「stop 的 log 未来不是 RFS 好目标」，依据只有 16 帧。
   (c) 说「只有提高 dw 能压进假起步线」，依据只是一条一维扫描，加一个看过数之后才加的消融。
2. op-adapt L 的开环增益是真的，不是阈值造出来的：换成不依赖阈值的 ADE@4s，start、stop、turn onset 也分别降了 0.22、0.27、0.38 m，CI 都不含 0。intent 置换的结论站得住。stop 增益「不是跟车」也站得住，但「学到了停车时机」证据不足：other 类里有多少是红灯 / 停止线，一直没拆。
3. 对用户的原问题：r2 那种配对样本监督，作为 openpilot 的训练信号应该停掉。log 里的人类未来是**行为先验**的好来源（start 尤其好），但它**不是**能把 WOD RFS 推过榜首的目标，因为它本身的 RFS 只和 O×1.06 差不多。少量手工轨迹不够训 stage 4，但足够拿来裁决「哪种标签源在哪类帧上可信」。真正还缺的是一条数据量的 learning curve，它花不了多少钱，又直接回答「多少条专家轨迹才够」。
4. 下一步排序：先补 B2D 判定（批次已经停了，见 §6）；接着做纯离线的 **start gate**，去攻 stay 假起步；然后跑 **learning curve**；然后把标注页做成**标签源裁决**用；巡航切片先做零训练的校准基线，做完再决定要不要训练；仿真数据只先做一个 recovery 可行性探针。WOD RFS 不要再当主攻目标。
5. 流程上最要紧的一件事：B2D 全批 **09:08 CST 已经 ERROR 停了**（探索臂 `ld-dtz-s0` 连挂两次），chain 09:21 退出，三张卡 09:37 时仍是 0% 空转，judge 没跑，调度表还写着 running。

## 1. 数字核对（CSV 直接读出）

| # | 条目里的说法 | CSV 读数 | 结论 |
|:--|:--|:--|:--|
| 1 | 77：O 的 capture，WOD train / navtrain GIMM / nuScenes：start 0.58 / 0.73 / 0.64，stop 0.29 / 0.87 / 0.38，turn onset 0.74 / 0.72 / 0.61 | `errors.csv`：0.583 / 0.733 / 0.644；0.289 / 0.869 / 0.377；0.736 / 0.721 / 0.613 | 成立 |
| 2 | 77：独立事件，WOD train / navtrain：start 1 096 / 2 994，stop 727 / 1 225，turn onset 912 / 3 665 | `counts.csv` 一致 | 成立（独立性见 §2 W6） |
| 3 | 78：`main` 的 Δcapture 分别为 +0.101 [0.082]、+0.307 [0.250]、+0.117 [0.097]；假起步 +2.7 pp（上界 3.9）；漂移 0.059 / 0.337 m；RFS +0.009 / +0.019；ADE −3.0%；navtest +0.27 / +0.33 / +0.30 | `arms.csv`、`lines_by_model.csv` 逐项一致；RFS CI [−0.067, +0.078] | 成立 |
| 4 | 78：dw 0.3 / 1 / 3 / 10 下 start +0.318 / +0.199 / +0.101 / +0.039，假起步 +25.4 / +8.1 / +2.7 / +0.9 pp；`stayheavy` +0.077、+1.7 pp（2.7） | 一致 | 成立 |
| 5 | 78：`main` 对 `noint`，turn onset +6.8 pp [5.6, 8.1]、start +1.3 pp | `compare.csv`：+0.0684 [0.056, 0.081]、+0.0129 [0.004, 0.023] | 成立 |
| 6 | 78 后续：flip 后 sign agreement −7.5 pp [−9.3, −5.9]；b1 / b2 / c / d 分别 +0.064 / +0.016 / +0.006 / +0.011 | `perm_contrasts.csv`、todo 表一致 | 成立 |
| 7 | 78 后续：strict other（YOLO 和 O 的 lead 头都没看到前车 / 行人的 stop 帧）1 117 帧 / 164 段，+0.330 [0.271, 0.394]；lead 98 / 24，+0.133 [0.050, 0.253] | `stop_by_cause.csv` 一致 | 数字成立，解读见 W4 |
| 8 | 79：S（start ∪ stop ∪ turn onset）覆盖 102 / 479 帧，占 headroom 37.7%，other 占 56.0%；logged − O：start +0.74 [0.18, 1.29]，stop −2.20 [−3.90, −0.67]；gap ≥ 2 且不在 S 内的 110 帧上 5.70 → 8.04 | `q1_coverage.csv`、`q3_logged_rfs.csv`、`q5_big_gap_frames.csv` 一致 | 数字成立，解读见 W1、W2 |

我另外读了阈值无关的连续量（`metrics_all.csv`，`main` 三个 seed 平均，WOD val）：

| 切片 | ADE@4s O → `main` | Δ [CI] | 读法 |
|:--|:--|:--|:--|
| start | 1.548 → 1.331 m | −0.217 [−0.247, −0.190] | 不靠「一半」阈值也成立 |
| stop | 2.032 → 1.762 | −0.270 [−0.313, −0.224] | 同上 |
| turn onset | 1.683 → 1.305 | −0.379 [−0.434, −0.326] | 同上 |
| stay | 0.170 → 0.288 | +0.117 [0.098, 0.139] | 假起步的代价在连续量上同样看得到（相对 +69%） |
| other | 1.476 → 1.474 | −0.002 | 非模仿帧基本没变，巡航落后的问题没被碰到 |

## 2. 薄弱环节

按影响排序。「改写」一列是我建议的措辞，供用户就地改 decisions 时参考，我自己没有动它。

| # | 位置 | 问题 | 严重度 | 建议改写 |
|:--|:--|:--|:--|:--|
| W1 | 79 第 1、5 点与「对方向的含义」 | **headroom 用错了基准**。(i)「到榜首的差 0.166」是 test 榜首 8.17 减去 **val** 上的原始 RFS 8.004。O 在 test 上是 7.92（×1.06），在 val 上 ×1.06 是 8.119，val 和 test 不可比。(ii) 巡航帧的「logged 8.04 对 O 5.70」是按 O 的低分挑出来的帧，回归均值的效应很大，镜像组「gap < 2」的 303 帧上 logged 反而比 O 低 0.72。(iii) 不挑帧、再扣掉 ×1.06 以后，other 帧上 logged − O 是 +0.137 [−0.145, +0.416]，全部 479 帧是 +0.111 [−0.119, +0.322]。也就是说，**完美模仿日志未来，在 val RFS 上相对 O×1.06 的上限大约是 +0.1，CI 含 0** | 高：直接决定巡航切片该不该做 | 「把 logged future 原样当 plan 交，在 479 个 val rater 帧上对 O×1.06 是 +0.11 [−0.12, +0.32]（帧均值），其中巡航（other）帧 +0.14 [−0.15, +0.42]。log 模仿在 RFS 上的总上限与 CI 宽度同量级；110 帧的 +2.35 是按 O 低分选帧的结果，不作为可得增益」 |
| W2 | 79 第 4 点、「stop 不能再用 log 人类未来当 RFS 的目标」 | rater 帧里的 stop 只有 16 帧 / 16 段，有一半的 logged future 落在 floor 上。rater 帧本身是长尾挑选出来的帧，WOD val 全量的 stop（1 268 帧）上 `main` 涨了 +0.31，但这个增益一帧都没落到 rater 帧上（1/16 → 1/16）。把 16 帧的读数推到「stop 这类」整体不成立 | 中 | 「在 16 个 rater stop 帧上，rater 偏好的轨迹不停车，logged future 的 RFS 比 O 低 2.2 [−3.9, −0.7]；rater 帧不代表全量 stop 切片，这只说明 stop 模仿对 RFS 没用，不说明 stop 的 log 标签错」 |
| W3 | 78 标题与第 4 点「只有提高蒸馏权重能压进线」 | (i) 结论只来自 dw 的一维扫描，再加一个看过 wave 1 val 之后才加的 `stayheavy`（D4）。gate、按 stay 加权的 hinge、两阶段都没试过。(ii) L2 的线写的是「CI 上界 ≤ +2 pp」，stay 只有 144 段，CI 半宽约 1.2 pp，所以点估计实际要 ≤ 约 +0.8 pp 才过，比字面严得多。(iii) 停着的车 4 s 内会不会起步，有一部分本来就取决于前视 1.6 s 里看不到的东西（灯什么时候变、前车什么时候走），这个取舍可能有信息上限。把 dw 扫描画成前沿，每多 1 pp 假起步换来的 start 增益分别是 0.013 / 0.025 / 0.037 / 0.043（dw 0.3 → 10），`stayheavy` 是 0.045 | 中高：决定 (b) 怎么设计 | 「在测过的一维 dw 扫描里，只有 dw 10 压进了 L2，代价是 start 增益只剩 +0.039；没有试过 gate 或 stay 加权损失，所以不能说只有 dw 能压」 |
| W4 | 78 后续（2）「学到了停车时机」 | (i) other 是剩余类，红灯、停止线、让行、漏检全在里面。YOLO 的 COCO 类别里本来就有 `traffic light` 和 `stop sign`，这次没用，拆不开。(ii) 平地抬升后车辆的 gx 最小只有 6.7 m，更近的车被压到约 7 m，lead 类的召回不明。(iii) **没有速度匹配的假停线**：control 要求 v0 ≥ 5 m/s 且匀速，而 stop 切片的 v0 中位数是 4.6 m/s，两者速度几乎不重叠。所以 control 上假停 ≤ 0.3 pp 并不排除「低速时更爱停」这种先验。other 帧上的 slow 率 +0.8 pp 只能部分兜住 | 中 | 「stop 增益主要出现在没检测到前车 / 行人的帧上，不是跟车。原因没有分出红灯 / 停止线；也没有对同速度、人类没停的帧量过假停」 |
| W5 | 77 标题「原生 plan 在这三类上各漏 3–7 成」 | start 漏 27–42%；stop 在 WOD 漏 71%、navtrain 只漏 13%；turn onset 漏 26–39%，而这时模型**没有导航输入**，漏的有相当一部分是缺条件，不是缺能力。另外，navtrain 的 stop capture 0.87 很可能有一部分来自 GIMM plan 整体偏慢（全部帧 bias_lon3 −2.5 m，control 上 −4.2 m），不全是两个数据集的行为不同。「WOD val 0.07（n = 41）」是另一组帧（考试帧，与 rater 帧重叠），不是协议导致的下降：同一批 rater 帧在 5 Hz 协议下也是 1/16 | 中 | 「原生 plan 漏掉 start 的 27–42%、WOD stop 的 71%（navtrain 只漏 13%，可能与该 plan 整体偏慢有关）、无导航输入时 turn onset 的 26–39%」 |
| W6 | 77 第 2 点「合计各约 2 千到 4.6 千个独立事件」 | (i) navtrain 的 token 本身是过滤后的子集，帧 / 事件比只有 2.4（2 Hz），WOD 是 8.9（5 Hz），navtrain 的事件很可能被采样空洞切碎了，数目偏高。(ii) op-adapt L **没有用 navtrain**（teacher 是 sample-and-hold，坏了）。实际训练的是 WOD 的 889 / 579 / 705 段，「上千」这个量级只有 start 在 WOD 上勉强够到 | 低中 | 「WOD train 实际可用、也实际用于训练的是 start 889、stop 579、turn onset 705 段；navtrain 的事件数可能被 token 过滤切碎，而且它的 teacher 不可用」 |
| W7 | 78 全部「留出」读数 | WOD val 是唯一的留出集，目前已经被约 45 个模型读过，外加 follow-up、RFS 诊断，B2D 臂的选择（`ldw10`）也看了 val。它已经不能再当下一轮的干净选择集。dev 上的 stay 只有约 1 200 帧，假起步以 0.083 pp 为一格跳，选择本身也很粗 | 中：影响下一轮 | 不改措辞；下一轮要另开一个干净的留出集（§5 统一约束） |
| W8 | 78 「intent 主要买起转」 | 置换检验站得住。但 turn onset 帧有 98% 带转向 intent，intent 在这类帧上几乎就是答案，这条增益衡量的是「会不会读导航」，不是对场景的理解。部署时没问题，写进论文时要说清楚 | 低 | 加一句：「turn onset 帧的 intent 几乎决定了方向，这条增益是导航条件化的效果」 |
| W9 | 79 第 2 点「上限很低（合计 0.017 到 0.03）」 | 0.017 是按**现在的翻转数**算出来的兑现量，不是上限。以 logged future 为目标，start 的上限是 +0.74 × 61 / 479 ≈ +0.09（帧均值；×1.06 下 +0.085），`main` 只拿到约五分之一（rater start 帧上 5 s 处仍比 rater 轨迹落后 2.4 m） | 中 | 「按当前翻转数兑现只有 +0.017；start 若完全贴住 logged future，上限约 +0.09（帧均值），`main` 拿到约 1/5」 |

以下几条我检查过，不构成问题：
- **5 Hz、1.6 s 前视协议 vs 官方协议**：O 的 RFS 8.004 对官方 8.005，RFS 口径一致。capture 在两种协议下的差别来自帧集合不同（见 W5）。
- **`main` 由回退规则选出**：四个 dw 3 候选在 val 上彼此相差 ≤ 0.02，选哪个都不影响结论。
- **seed CI 不含 seed 方差**：三个 seed 之间相差 < 1 pp，不影响结论。
- **nuScenes 上的迁移**：start、stop 迁移（下界 0.014 / 0.104），turn onset 不迁移（那边没有 intent），说法与数字一致。
- **记录不一致（小事）**：D4 写 `stayheavy` 是 23:47 CST 看过 wave 1 val 之后加的，但 commit c4f9ae8（「added while wave 1 runs」）是 23:35 CST。当时有部分冻结 stage 4 的 run 已经跑完读数，「看过多少」说不准。

## 3. 对用户原问题的回答

**配对样本 lane（r2）还值得做吗？** 作为 openpilot 的训练信号，不值得，建议正式停掉而不只是搁置。证据有四条：
- r2 stage 1 把模型整体推保守了（漂移 0.40 m，null 帧 slow 率 +11.7 pp）。
- CARLA 配对激发出来的 reaction 一挪到真实数据上就有害（第 44 条：WOD RFS −1.0 到 −1.5）。
- openpilot 特征在 CARLA 里读不出行人（第 62 条，AUC 0.51–0.53）。
- WOD 里真实的走廊内行人事件只有约 230 个，216 个编辑对上 ridge 什么也没学到（第 44 条）。

op-adapt L 在同样的轻度解冻下做到了不漂移，差别在监督：真实域里成对的「该动 / 不该动」。CARLA 配对剩下的价值是**考卷**（配对行为考试）和世界模型的分叉数据，不是 openpilot 的监督。

**log 里的人类未来是不是对的「专家轨迹」？** 要分用途看：
- **作行为先验，是**：start、stop、turn onset 都能学到，留出集上 capture 和 ADE 双涨，没有漂移，navtest 不掉（+0.3），nuScenes 上 start、stop 能迁移。
- **作 RFS 的目标，基本不是**：logged future 在 val 上对 O×1.06 只多 +0.11，CI 含 0（W1）。其中 start 是好目标，16 帧的 stop 是坏目标，turn onset 中性。WOD 榜首大约就在「日志质量」附近，模仿日志最多追平，很难超过。
- **单条人类轨迹天然有噪声**：停车时机取决于看不到的原因（灯的相位），模型学到的可能是「这类场景平均停在哪」。假起步的取舍很可能就是这个噪声的体现（W3）。

**少量手工专家轨迹能教会模型吗？** 按这里的实验，**数量级不够训练，但够做裁决**：
- op-adapt L 每类用了 580–890 段（8 400 / 3 867 / 14 448 帧），过了 4–5 遍。没有做过数据量的消融，所以「最少要多少」目前**不知道**。
- 只训 adapter 的 `tr_ad`（3 × 32 × 512 ≈ 4.9 万参数）拿到了约一半的增益。这说明小容量的干预承载了相当大一部分效果，小容量正好是少量标签有机会够用的那一侧。
- 216 对 WOD 编辑对（标签是「log 未来 − CTRA」）什么也没学到，但那次的标签本身很弱，不能直接用来说明量不够。
- NAVSIM 那条 6 万行饱和的曲线是**打分头**（在 22 个候选里挑一个，每行带稠密的 metric 标签），不能拿来推算轨迹模仿需要多少数据。
- 人手画 5 s 轨迹的误差是米级，已经大于 3 s 处横向 ±1 m 的信任区。手工标注更适合**在候选里选**（与 rater 同一种格式），不适合画线。

结论：几百条以内的手工标签，用来（i）裁决标签源，（ii）定 gate 的阈值，（iii）当考卷。要不要用它们**训练**，等 §5 的 R3 learning curve 出来再定，那个实验正好回答这个问题。

**下一步该造哪些数据，按什么顺序？**
1. 先不造新数据。用现有的 WOD 切片做 learning curve 和 start gate（R2、R3），它们决定后面需要哪种数据、要多少。
2. 再造少量**裁决标签**（标注页，R4），对象是标签源存疑的帧：stop、巡航落后、`main` 的假起步帧，每类约 100 帧。
3. 巡航切片不需要新数据，用 train 上的无标签定义就够。但它的价值要先过「零训练校准基线」这道门（R5）。
4. 仿真数据只针对日志里没有的东西（recovery、静态障碍 nudge），并且先做可行性探针（R6）。行人反应类在 CARLA 里读不出，先不造。

## 4. B2D 闭环：几种结局各意味着什么

先说统计功效。第 74 条里 drive 对配速对照的 CI 宽约 17 DS（[−16.9, +0.2]）。L-S2、L-S1 都是**点估计**线，10 条路线上一个 +2 DS 的「过」就是噪声。我会把强读法定为：L-DiD 的 CI 下界 > 0，或者三个训练 seed（`lmain`、`lmain1`、`lmain2`）的 ΔDS(X − slow_X) 同号且都 > 0。B1–B3 这些行为读数是逐步或逐事件计数的，比 DS 灵敏，更值得看。stage 2 里 `lmain` 的均速是 `drive` 的 1.25 倍（2.56 / 2.05 m/s），配速对照就是为了扣这一项。

| 结局 | 读法 | 下一步 |
|:--|:--|:--|
| A：L-S2 ∧ L-S1 ∧ L-Safe 过，B1 过（timer resume 占比降 ≥ 0.15），三个 seed 同号 | log 模仿第一次在闭环里变成了行为和分数 | 按登记跑一次 held-out 19 条（约 2.5 h）；再上 HUGSIM 做 no-harm；log 模仿升为 openpilot 适配的主线，R2（gate）优先，重点盯闭环里起步后的闯红灯 |
| B：B1 / B3 过，DS 对配速对照不过 | 行为接上了，但 DS 被别的失败形态占满（路口冲突碰撞、红灯、施工卡死，第 74 条）。B2D 的 DS 量不出这几类行为 | 不再为 B2D 加 log 模仿；论文里用行为读数报告闭环；DS 的瓶颈去红灯 / 横向冲突那边找（第 74 条 R3a 的放行通路），那不是模型的事 |
| C：DS 过，B 都不过 | 分数来自别的机制（ghost stop 变少？开得快？） | 先按 standstill 情境和违规类型拆清楚再说，不写成「log 模仿有效」 |
| D：DS 与 B 都不过（`lmain` ≈ `drive` ≈ slow） | 开环 capture 带不到 CARLA，与第 55 条「真实数据监督完全带不到 CARLA」一致；第 57 条的静止先验在 CARLA 画面上没被改掉 | B2D 不再作 openpilot 适配的判定尺，改用 HUGSIM（真实感渲染）。如果仍要 CARLA，就需要 CARLA 域的起步数据（R6 的同一套机制），而且要带真实域 no-harm 线 |
| E：L-Safe 不过（横向占比、off-lane、blocked 变差） | stage 4 解冻在域外不稳 | 用只训 policy 或 adapter 的 checkpoint（`sel_polia_dw3`、`tr_ad_dw3`）重跑同一个 set |

不管哪种结局，`lmain − lnoint`（adapter 在闭环里的贡献）和 B3（转弯跟随）都是独立的读数。它们回答的是「导航输入接法」，与 DS 无关。

## 5. 下一步（排序）

统一约束：
- **WOD val 不再用于选择**。从 WOD train 里按段重新划出一个从未读过的 10% 作为新的留出集（op-adapt L 的 dev 已经用于选择，不能算），val 和 rater 帧只在最后读一次。
- 一次只跑 1–2 个 job。
- 下面的成本按三张卡计：stage 4 解冻的 run 每卡并发 2 个，约 21 min / run；冻结 stage 4 的 run 约 5 min。

| 排名 | 选项 | 回答的问题 | 成本 | 主要风险 | 什么结果毙掉它 |
|:--|:--|:--|:--|:--|:--|
| 0 | 补 B2D 判定（必做，算不上选项） | 闭环有没有接上 | CPU 几分钟；held-out 若符合条件约 2.5 h | 探索臂缺失会被误读 | — |
| 1 | **R2：start gate（离线）** | 假起步是设计问题还是信息上限 | < 1 GPU·h，约 2 h 墙钟 | gate 学到的是 stay / start 的选择偏差，不是真信号 | dev 上 AUC < 0.75，或 gated 前沿不比 dw 前沿好 |
| 2 | **R3：数据量 learning curve** | 每类要多少条专家轨迹 | 约 2.5 卡·h，约 1 h 墙钟 | 小子集过拟合，和步数混在一起 | 不会被毙（这是测量），只决定 R4 怎么用 |
| 3 | **R4：标注页做标签源裁决** | 哪种标签在哪类帧上可信；手工标签有没有用 | 无 GPU；建页约半天 agent，标注 3–5 人·h | 标注员看到未来会偏向 logged；两人一致率低 | κ < 0.4 |
| 4 | **R5：巡航切片（先校准、后训练）** | RFS 和闭环里的「开得慢」能不能在不加误触发的前提下修 | 阶段 0 CPU 几分钟；阶段 1 约 9 run，约 1.5 h | 学成「整体开快」 | 阶段 0 后残余上限 < 0.07，或阶段 1 漂移 / fast 超线 |
| 5 | **R6：仿真 recovery 探针** | 日志没有的行为是否真缺 | 1 卡约 2 h | openpilot 本来就会 recovery，数据白造 | O 在偏移起点上 90% 以上已经回正 |

**R2：start gate，攻 stay 假起步（对应要求 b）。** 做法：
- 在 WOD train 的 stay ∪ start 帧（13 403 + 8 400）上，从 O 的 stage 4 输出缓存 `t/H`（已经有）训一个小分类头 g，预测「人类 4 s 内会不会起步」。
- 推理时只在静止帧（vmax1 ≤ 0.5）上起作用：g > τ 用 `main` 的 plan，否则用 O 的；其余帧照用 `main`。
- val 上 O 和 `main` 的 plan 都已经存在 readout 里，所以**整个实验不用重训 `main`**。
- τ 在新留出集上定，使假起步 Δ ≤ +1 pp。

这个实验同时测一件更根本的事：g 的 AUC 就是「前视 1.6 s 里有多少起步信息」。如果 AUC 只有 0.7 左右，假起步的取舍基本是信息上限，L2 线就该改成按前沿报告，而不是硬线。

预登记读数与过线：
- val 上 start Δcapture ≥ +0.07，且 CI 下界 > +0.05；
- stay 假起步 Δ 的 CI 上界 ≤ +2 pp；
- RFS「无害」（下界 ≥ −0.10）；
- 前沿条件：同一假起步水平下，start 增益高于 dw 扫描插值出来的前沿（在 +1 pp 假起步处 dw 前沿约 +0.045）。

g 不够好时的第二臂（要训练，约 1 h）：stay 帧上加一个 hinge 损失，把 plan 4 s 位移压在 0.5 m 内，配合 `stayheavy` 的 3 : 1 : 1 比例。

**R3：数据量 learning curve（回答「少量专家轨迹」）。**
- 设置：`main` 配置，按段对三类各自子采样 1/32、1/8、1/3、1（约 25–30、110、300、全部 580–890 段），每档 2 个 seed，对照帧和蒸馏不变。
- 另加一档「1/8 数据 × 同步数」，用来区分数据量和训练遍数；再加一档冻结 stage 4、只训 adapter 的同样曲线，看小容量是不是更省数据。
- 读数：新留出集上三类的 Δcapture、ADE@4s 与假起步。
- 登记的决策规则：每类 ≤ 110 段就拿到全量增益的 ≥ 60%，说明手工 / 半自动标注 10² 量级可以训练小容量干预，nudge 这种日志里只有 200–700 个事件的行为可以直接用日志加少量人工清洗；≤ 30%，手工标签只作裁决和考卷。

**R4：标注页（要求 d）：裁决标签源，不训练。**
- 抽样：从 WOD **train** 抽，不用 val。只给过去 2 s 的视频和 BEV 上画出来的候选轨迹：logged future、O、O×1.06、`main`，加一条 R5 的校准版，按随机顺序、匿名显示。不给未来视频，避免看出哪条是人类。
- 每帧要标：哪条最好（可以并列）、每条能不能接受，以及原因码（灯、停止线、前车、行人、让行、看不出）。
- 帧集合，每类约 100 帧：stop 切片、巡航落后帧（R5 的定义）、`main` 的假起步帧、turn onset 里 O 没捕获的帧。另取 50 帧双人重标，算一致性。
- 读数：每类里人对 logged 和 O 的偏好率，以及 Wilson CI；n = 100 时半宽约 ±10 pp。
- 过线：logged 被偏好 ≥ 60% 且 CI 下界 > 50%，才把该类的 log 标签记为「可信」；原因码里「灯 / 停止线」的比例，正好补上 W4 拆不开的那部分。
- 毙掉条件：双人 κ < 0.4。
- 这个页面做的事，就是用户原来「暴露开环考题、人工标难例」那个想法里**可以兑现的部分**：用几百个标签判断哪种标签源能用，而不是直接拿来训练。

**R5：巡航 / 纵向切片（要求 a），分两段。**
- 阶段 0（纯 CPU，无训练）：在 WOD train 的非切片帧上，拟合一个只依赖 v0 的分段纵向缩放 s(v0)，替代固定的 ×1.06。在新留出集和 val rater 帧上读三样：巡航帧 lonADE、RFS 对 O×1.06、logged-future 上限对校准后的 O。
  过线：如果 logged 对校准 O 的残余上限（巡航帧，帧均值）< 0.07，那么训练能拿到的 RFS 一定落在 ±0.07 的 CI 里，阶段 1 对 RFS **不做**，只在闭环需要更快巡航时再做。现在已知对 ×1.06 的残余是 +0.14 [−0.15, +0.42]，我预计会落在边缘。
- 阶段 1（满足条件才做）：在 train 上定义巡航帧：v0 ≥ 3、不在任何切片内、|O 的 lon@4s − 人类的 lon@4s| > max(1.5 m, 15%)，按落后、超前两个方向分开。对这些帧做模仿，其余帧蒸馏（dw 3），沿用 op-adapt L 的 L2 / L3 线，另加速度匹配的假停线（补 W4 的漏洞）。
  - 基线：阶段 0 的校准 O，不是原始 O。
  - 过线：新留出集巡航 lonADE 相对降 ≥ 10% 且 CI 不含 0；other 帧 slow / fast 的差 ≤ +2 pp，漂移中位 ≤ 0.10 m；val rater RFS 对校准 O 的点估计 ≥ 0，CI 下界 ≥ −0.05。
  - 成本：3 个配置 × 3 seed，约 1.5 h。
  - 毙掉条件：fast 率超线（学成了「整体开快」），或 RFS 对校准 O 的点估计 < 0。

**R6：仿真数据（要求 e），先探针。**
- 日志里没有的三类，按可行性排：
  - **recovery**：横向偏移或航向偏差下回到车道。只依赖车道几何，openpilot 在 CARLA 里看得到车道线。
  - **静态障碍 nudge**：FOD、施工这种大物体，大概能读出来。
  - **行人反应**：第 62 条已经证明 CARLA 特征里读不出行人，先不做。
- 探针：在 op-drive 的 dev 路线上用特权 expert 制造 0.5–1.5 m、5–15° 的偏移起点，离线读 O 的 plan 是否朝车道中心回正。openpilot 训练时很可能已经在它自己的学习型模拟器里做过 on-policy 的偏移增强，所以 recovery 有可能本来就会。
- 过线：回正率 < 80% 才造数据。造的话按 op-adapt L 的格式（模仿 + WOD / nuScenes 蒸馏 + 真实域 no-harm 线）进训练。
- 成本：探针约 2 h × 1 卡；全量生成约一天三卡。

**我认为比上面这些都更重要的两件事：**
1. **把 WOD RFS 从主攻目标里拿掉。** O×1.06 在 val 上是 8.12，log 模仿的上限大约 +0.1，而且 val 和 test 有约 0.2 的系统差。这块地已经榨干了。「trick or trade」的论点要在闭环和跨榜 no-harm 上证（navtest +0.3、nuScenes 迁移、B2D 行为读数），不要在 RFS 的第二位小数上证。
2. **r2 正式关掉**（见 §3），CARLA 配对只留作考卷和世界模型的数据。

## 6. 流程卫生（只列事实，没有清理）

1. **B2D 全批已停**：`$DATA_DIR/runs/op_l_b2d/ERROR` 写于 09:08:18 CST，内容是「unit ld-dtz-s0 failed twice (rc 0)」，该 unit 10 条路线只完成 8 条。daemon 09:08:27 报 FATAL，chain 09:21:14 以 exit 1 退出。没有 `DONE`，judge 没跑。
   - 已完成的 unit 日志（不含 stage 1）：判定臂和配速链看起来都齐了（`drive`、`dbase`、`lmain`、`lnoint`、`dnod`、`lkd`、`ldw10`、`lmain1`、`lmain2` 各 2 个 seed，以及各自的 dbaseslow / dbaseslow2）。
   - 缺 `ltz` 两个 seed；`l2-dbaseslow2-s0` 没有，可能是配速已经提前终止，没核对。
   - 09:37 CST 时 GPU 0/1/2 利用率 0%、显存 0 MiB。brief 里「约 2.5 h 后完成」的预计已经过时。
2. **调度表** `$DATA_DIR/runs/sched/table.tsv`：`op-l-b2d` 一行仍写着「full batch running」，GO 文件还在。`wl-tokens` 一行（done）列的是 GPU 4,5，box 现在只有 0–2。
3. **box 仓库**：HEAD 9a0a91f，与 Mac 一致。
   - 有一处被改过的 tracked 文件：`research/results/wl/sanity_full.json`（+5 行，未提交）。
   - untracked：`scripts/op_l_b2d_daemon.py`、`scripts/.zeroshot_b2d_op_test.sh`、`scripts/p3/p3_autoscale.py`、`scripts/resources/`、两个 `*.box-stale`、`research/results/nq3/*`、`research/results/wl/*pilot1*`、`research/results/night2/N1/vocab_k1024_seed0.npy`。
4. **git 之外的拷贝**：`scripts/op_l_b2d_daemon.py` 在 Mac 和 box 上都是 untracked，md5 相同（4695cfd8…），文件时间相同（Mac 09:24 JST = box 08:24 CST），是绕开 GitHub 直接拷过去的。Mac 上另有 untracked 的 `scripts/op_l_b2d_wait.sh`。正在跑的批次依赖的守护脚本不在 git 里。
5. **误提交的历史**：
   - bf87089 把另一个会话的五个脚本移出了跟踪，它们是被 63f4e6f 的宽 `git add` 带进来的；随后 dc38d2e（「op-adapt L follow-up: fix eval path helper, add stop chain」）又把**同样五个**加了回去（`make_article_closed_loop_figs.py`、`make_article_critical_moment_figs.py`、`make_article_figs.py`、`make_top_decile_sheet.py`、`poc_carla_rewind.py`），现在仍然被跟踪。
   - 4d41f14（「rfs diagnosis: both/neither flip split…」）里混入了 B2D lane 的 ONNX 服务改动（`jevdrive/openpilot/model.py`、`scripts/op_arb_server.py`、`scripts/zeroshot_policy_server.py`）。实际做这件事的 5f0b5dc 只含 `op_l_onnx.py`。两个并行 lane 的提交串了。
6. **tmux 残留**：`jev` 里还开着已结束 lane 的窗口 `r2-chain`、`r2-side`、`wl2-report`、`opL-fu-stop`、`opL-fu-perm`、`opL-fu-perm2`，以及已经退出的 `opl-full`。另有约 9 个 defunct 的 python 僵尸进程，无害。
7. **文档过时**：CLAUDE.md 和 memory「gpu-box-single-card」写的是 7 卡 / 175 核 / 644 GiB；op-adapt L 预登记记录的 box 是 3 卡 / cgroup 75 核 / 296 GiB。
8. **已知坏掉的缓存还在原位**：r2 的 `t/teacher/nav`（2 Hz sample-and-hold，有速度的帧上 plan 系统偏长约 20 m）。第 77 条已经记了，但目录里没有标记，后面的 lane 有可能误用。

## 7. 没能核实的

- B2D 的任何 DS / 行为读数：按要求没读，box 上 `results/` 已有汇总。
- `l2-dbaseslow2-s0` 缺失的原因（配速提前终止还是漏跑），以及 `ld-dtz-s0` 失败的根因（rc 0 却判失败，可能是路线数不足的判定规则）。
- navtrain 事件数被采样空洞切碎的程度：我只是从帧 / 事件比推断的，没有回到 token 时间戳去核。
- O 是否已经会 recovery（R6 的前提）：这是推断，没有测过。
- stop other 类里红灯、停止线的占比：没有标注，YOLO 的 `traffic light` / `stop sign` 类别也没跑。
- 第 77 条第 47 条规则 13 / 15 的差 2：仍然没查。
