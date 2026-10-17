# 草案：第 335 / 336 条之后的候选 pilot——(iii) 保留 P0 的「到达时停」（RETAIN，a 补丁 / b 重建底座）与 (iv) 打断自身运动趋势（ANTI-COPY）（2026-10-17 夜）

**状态：正式预登记（2026-10-17 04:00 CST，主会话依用户当晚「遇异常读数可改道、卡要排满」的授权定稿）**。三条 pilot 都起：(iii-a) RTA03P / RTA1P、(iii-b) RTB1 / RTB3、(iv) AC 两档，各自按下文的线与 gate 判；pilot 都是开环判词，(iv) 过 gate 时自动交 held 460（链里自带），(iii-a) 过 gate 时由主会话补交 held 460 对 WTCP-L18-s0，(iii-b) 过 gate 不自动往下（重放 RB0 / L18 等用户）。任何正式臂（2 seed 全长）都等用户早上定。下面的线写在任何一个 pilot 臂的输出之前，未改。
（去掉「草案」、补上选择理由），push 后再提交。下面的线写在任何一个 pilot 臂的输出之前。

代码：`experiments/route_mem/scripts/rm_train.py`（`--kd-lam / --kd-mode / --kd-k / --kd-pool / --kd-groups / --kd-nav / --kd-nav-k`，`--ego-scale-speed / --ego-scale-max`，
全关时不加任何操作）、`lib/retain_train.py`、`experiments/wait_cap/scripts/nx.sh`（链）、`nx.py`（逐位比较、接近行、到达重放、历史缩放读数、各组线）。盒上输出 `runs/wait_cap/next/`，结果 `results/next/`。

## 1. 来源

第 326 / 331 / 332 / 335 条：出厂 Cinque（P0）在人等的帧上 0.978 等（D 0.64）、平历史下停在静止前车后 S_L 0.845、让行 Y_P 0.218 / Y_V 0.335、
变道 desire 4 s 响应 1.04 m；**第一个 adapter 阶段（P2-F，RB0-F 同样）把这一族洗掉**（W 0.008–0.012，S_L 0.39–0.41，desire −69 %）。
第 335 条：在 RSG-L18 上补等待行，开环 W 到 0.86，但教出的是无条件的低速变慢（dev ADE +0.079，4 s 弧长 −0.49 m），闭环 held 零分 +12、出路翻倍、距离 −11 %，
闭环里看不到等。第 315 条：纵向通道是自身运动的趋势跟随者，开环 4 s 短 2.8 % 在闭环成了距离短 11 %。早上的两个选项：

- **(iii)** 重建底座时保留 P0 的有条件纵向通路（攻原因：通路在 P2-F 被洗掉）；
- **(iv)** anti-copycat：削弱 plan 对自身历史速度的跟随（攻放大器：任何纵向训练都经它放大）。

**第 336 条改了 (iii) 的靶子**：P0 在闭环里会停（静止 scene 0.48 m，人 0.50 m；stop17 零分 14 / 17，其后各段 16 / 17），但 P0 是很差的司机（held 460 零分 213、出路 98、
46 个 scene 原地不动）；WAIT 行已经补回「停着等」，缺的是 **到达时停**：交接后第一个决策（0.44 m/s 仍在滚、前车静止 1.6 m）P0 plan 4 s 0.43 m，J 2.0–2.3 m，随后蠕行、越来越长、撞上。
所以保留项**只学 P0 的纵向（弧长 / 速度剖面），不碰横向路径，只用在滚动接近与静止等待的帧上**，不做整条 plan 的 LwF（会把 P0 的横向弱点和无条件的等一起搬进来）。

## 2. (iii) RETAIN：只保留 P0 的纵向、只在到达帧上

**保留项**（`rm_train --kd-lam L`，`lib/retain_train.py`；全关时不加任何操作，第 5 节逐位检查）：每步额外一次前向，行为下面两类，主 batch 的行、抽样、损失都不变（独立 rng 流 [seed, 0, 151]）：

- **池行 K = 16**（WAIT-TRAIN 的 on1_WAIT，108 个 trainval log，对 held 460、公开 1491、PAI 281、waitcap_test、capledger_test 泄漏检查全 0），按组抽（`--kd-groups AR:2,W:1`）：
  - **AR 滚动到达（2 / 3）**：0.2 ≤ v0 ≤ 6 m/s 且属于 W_L（人在 1 m 内停在静止前车后）、S_L（停在静止车后）、S_N（无前车的停下：停止线）、Y_P / Y_V（让行）；
    S / Y 行喂匀速历史（历史里还没有刹车，第 331 条 F 读的处理），约 580 行（W_L 112、S_L 166、S_N 198、Y 103）；
  - **W 静止等待（1 / 3）**：W_L / W_N 且 v0 < 0.2（P0 的保持，0.978）；不抽 G（人起步的静止帧：P0 在那里 1 / 3 会等，是它无条件的那一部分）。
- **navtrain 接近行 K_nav = 8**（`--kd-nav`，`nx.py approach` 选行：训练 split、0.2 ≤ v0 ≤ 6 m/s，t0 的 agent 标签里车道走廊内 15 m 有静止前车（< 0.5 m/s），
  或前车接近且碰撞时间 < 4 s（wt_devdecomp 的 low-TTC 规则）），**改喂匀速历史**（vy = ax = ay = 0，过去位姿按 v0 沿自身折线重排，parity_adapter.cv_history），
  所以停车只能来自画面。这补上了 W（v0 < 0.3）与 S（v0 ≥ 2）之间 0.3–2 m/s 的空档，正是第 336 条那次到达的速度。停止线行只来自池的 S_N（navtrain 没有现成的停止线标签；未加）。
- **损失（`--kd-mode lon`）**：plan 自己 8 个后轴位姿的累计弧长对 P0 的累计弧长（wc_arc 的 `arc`：对旋转不变，梯度只沿 plan 自身切向，**横向路径不动**），smooth L1（β 0.5）
  8 个位姿求和（wc_arc 的尺度），行平均 × λ。目标是 P0 自己的 plan（池的 teacher.npz / store 的 t_plan = 出厂 Cinque 同一 warp token），不是人的。
  λ = 1 时每步的质量约为 WAIT j2 弧长项（13 个池行里期望 2 个新行，× 10 × 3 / 模仿行数）的 1.7 倍，但集中在到达帧上。
- **D2 desire 增量蒸馏**（rm_train 已有，`--d2-lam`）：P0 有 / 无 desire 的差，P0 自己的横向偏差相减抵消；(iii-b) 与保留项同一 λ，(iii-a) 固定为 1（修订 C 4 偏好的 D2 权重；保留项的 λ 因冒烟改小，D2 不跟着改）。

**两个变体**（同一保留项，同一组线）：

| | (iii-a) 补丁 | (iii-b) 重建底座 |
|:--|:--|:--|
| 配方 | RSG-L18 原命令（`--init RB0-F-s0`，池 on1_L18RB0，`--lr-scale 0.1`），**1 000 步**（WAIT pilot 同形） | RB0-F 原命令（从出厂权重起，无池），**10 000 步**（阶段全长） |
| 臂 | `RTA03P-L18-s0`、`RTA1P-L18-s0`（λ 0.3 / 1；见第 5 节冒烟） | `RTB1-F-s0`、`RTB3-F-s0`（λ 1 / 3） |
| 对照（关掉保留项的同一命令，逐位检查成立） | 存档 WTCP-L18-s0 | 存档 RB0-F-s0 |
| 到 pilot 判词 | **~0.6 卡时**，墙钟 ~25 min | **~1.6 卡时**，墙钟 ~50 min |
| 到一个可上闭环、可与现役比的模型 | 就是它（held 460 @18 每臂 ~1.1 卡时，对 WTCP） | 还要在新底座上重做 L18：滚 1 020 个 train scene（~0.75–1.5 卡时）+ tokens / teacher / 池（~0.3）+ RSG 续训 2 500 步（~0.15）+ held（~1.1）：**合计 ~4–4.5 卡时**、墙钟 2–3 h；若偷懒用旧池 on1_L18RB0（对新底座是 off-policy）约 ~3 卡时 |

**我的建议**：先 (iii-a)。它便宜 3–7 倍，第 335 条的对照、读数、闭环参照（WTCP）全是现成的；它回答的问题（「到达时停」能不能用针对性的 P0 纵向项在现役配方上装回来、代价多少）
正是决定要不要花 (iii-b) 的那个问题。(iii-a) 过线而闭环上能看到停，再考虑 (iii-b) 是否值得（底座阶段的洗掉是原因，补丁在 RSG 之后才装回，可能更脆）。

**线（开环 + 一个重放；每臂全部成立才算过）**，C = 该变体的对照，P0 = 出厂 Cinque：

| # | 线 | 读数 | 参照 |
|--:|:--|:--|:--|
| 1 | **到达重放**：交接后第一个决策（k = 1，0.5 s，0.44 m/s）plan 4 s 弧长 ≤ 1.0 m | `nx.py arrival`：在 P0 自己那次静止 scene 闭环的输入上重放（wt_xe xreplay 的同一调用，ego 行为实测值；路线记忆统一取 RSG-L18-s0 那次静止 run 同一决策的记录） | P0 0.43、J2P 2.04、WTCP 3.06、RB0 3.30（第 336 条） |
| 2 | S_L、Y_P、Y_V（F 读）**各自**至少回到一半：J ≥ C + 0.5 (P0 − C) | `led.py plans --conds F` | P0 0.845 / 0.218 / 0.335 |
| 3 | 等 W ≥ 0.60 且起步 G ≥ 0.85（保持而不锁死） | `wc.py plans`（waitcap_test） | P0 0.978 / 0.666 |
| 4 | desire 4 s 响应 ≥ 0.8 × P0 | `dc.py plans` → `dt.desire_read` | P0 1.04 m |
| 5 | dev ADE ≤ C + 0.01 m | DONE `dev_ade` | WTCP 0.5290 / RB0 0.5276 |
| 6 | **自由路 4 s 弧长偏差不变**：dev 行（自己的历史）里自由路的行（wt_devdecomp 的上下文：20 m 内无前车、无行人），偏差（plan − 人）J − C 在 ±0.10 m 内 | `nx.py hist` 的 s = 1 | 第 335 条 J2 的毛病就在这里 |
| 7 | 横向不变：2 s 处 plan 横向对 C 的逐行中位 \|差\| ≤ 0.05 m | `wt.py devplans` | |
| 8 | 仅 (iii-b)：路线记忆依赖读数 `read` 仍为真 | 训练结束的 `rely.json` | RB0-F-s0 真 |

只报：D、撞进前车、W 的配对差、到达重放 k = 0..9 的整条序列（看是不是停住后又起步）。
**选择**：只一个 λ 过 → 用它；两个都过 → 小的 λ；都不过 → 停，报告栽在哪条线，不做第三个变体。
(iii-a) 过线后的下一步（不在本草案）：held 460 @18 对 WTCP + 公开静止 scene + PAI stop17（第 335 条补记 D 的同一套闭环读数）。

**一条命令**（盒上 repo 根目录；依赖 `nx-approach` 与对应的逐位检查成功）：
- (iii-a)：`bash experiments/wait_cap/scripts/nx.sh rta-pilot`
- (iii-b)：`bash experiments/wait_cap/scripts/nx.sh rtb-pilot`

## 3. (iv) ANTI-COPY：历史速度噪声（第 336 条后降为次要）

第 336 条：把喂给 adapter 的速度历史置零不让任何段在闭环停下（到达时 plan 反而更长），所以 (iv) **不是**「等」的修法；它针对的只是第 335 条的开环 → 闭环纵向偏差放大（第 315 条），这一点没被测过。代码与检查已备好，排在 (iii-a) 之后。

**选 (a) 自身历史瓶颈，用「缩放噪声」而不是置零**。理由：

- (a) 的「置零」已经做过两次：RSG-L18 上速度历史 drop（RSGallV，第 286 条日间补记）去掉 75 % 短缺但公开零分 +8.5；SLOW S1（空路行置零，第 320 条）held 零分 +5、关闭。
  再做一遍置零没有新信息。置零把历史整个拿走（服务时永远不会出现的输入）；缩放噪声保持输入在分布内（同一条路径、按 s 倍速度走过），只打断「历史速度 → 未来速度」的对应：
  目标仍是人的日志，历史却快 / 慢了 s 倍，plan 的速度只能从画面与场景来。这正是 ChauffeurNet 的 past-motion 扰动、Wen 2020 copycat 线的做法（research/lit/2026-09-22-round4-intervention-pairs.md）。
- (b) 逆动力学头：仓库里有 select_4 的 tap（HEAD1b），但「由相邻帧预测记录的动作」在这里要么只能预测过去的动作（上下文 9 帧之间的自身运动，ego 行里本来就有），
  要么需要未来帧（store 里没有）；它让表征更会读自身运动，并不削弱趋势跟随（第 315 条：趋势同时来自历史与画面）。实现也要新参数组、另存头，
  不能做到「零初始化 / 关时逐位相同」之外还有清楚的因果故事。所以选 (a)。

**做法**（`--ego-scale-speed P --ego-scale-max M`）：每步对非锚、非池的行（模仿行：普通行 + yaw-rate 行）按份额 P 抽中，抽 s ~ U[1 − M, 1 + M]：vx、vy × s，ax、ay × s²，
4 个历史位姿的 x、y × s（yaw、命令、路线记忆不动）；独立 rng 流 [seed, 0, 149]；池行（S 的 self 行，没有人的目标）与锚行不动。只在训练时，服务不变。

**配方** = RSG-L18 原命令（`wt.sh train off`，即 WTCP-L18-s0 的命令：`--init RB0-F-s0 --lr-scale 0.1 --warmup 1 --on on1_L18RB0 --on-k 13 --on-mode self …`），
1 000 步、seed 0，与 WAIT pilot 同形。臂：P = 0.5，M ∈ {0.2, 0.4}：`ACS2P-L18-s0`、`ACS4P-L18-s0`。**对照 = 存档 WTCP-L18-s0**（同命令、同步数、同 seed；
池抽样不受新 rng 流影响，on_hash 不变）。

**线（开环；每臂全部成立才算过）**。行 = gate 4 的 1 789 个 dev 行里 0.5 ≤ v0 < 8 m/s 的行（补记 D 里 J 的短缺所在），新读数 `nx.py hist`：
同一行、同一帧，喂入的速度历史 × s（s ∈ {0.8, 1.0, 1.2}，列与幂同训练），s = 1 必须复现 DONE 的 dev ADE（差 < 1e−3，否则读数作废）。

| # | 线 | 说明 |
|--:|:--|:--|
| 1 | dev ADE ≤ C + 0.01 m | C = WTCP-L18-s0 的 0.5290 |
| 2 | **慢历史下 4 s 弧长偏差更小**：s = 0.8 时 \|plan − 人\| 的 4 s 弧长偏差，J − C 的配对差（log 聚类 bootstrap）CI 上界 < 0 | 这是闭环棘轮的开环代理：车比场景慢时，plan 跟不跟着慢 |
| 3 | 自己历史下不引入新偏差：s = 1 时 4 s 弧长偏差 \|J − C\| ≤ 0.05 m | 防「整体变快 / 变慢」冒充线 2 |
| 4 | 横向不变：2 s 处 plan 横向对 C 的逐行中位 \|差\| ≤ 0.05 m | `wt.py devplans`（行进中 ≥ 2 m/s 的 dev 行），与 WAIT 闸 5 同一读数 |

只报：历史增益 g = d arc4 / d s（s 0.8 → 1.2 的中心差，m / 单位 s）臂对 C 及其比值、s = 1.2 的偏差、等 W（waitcap_test）。
**选择**：过线的臂都进 held；两个都过时报告两个，闭环先读 M 小的。

**闭环（过线后由 gate job 自动提交，`nx.sh ac-held`）**：held 460 @18（rm_chain.sh read，6 chunk / tag）对存档 WTCP-L18-s0 同读数，rf_follow 数转弯；
描述线（单 seed，只是方向）：零分 J ≤ C + 3 且转弯完成 J ≥ C − 2，另报均分配对 CI（会话聚类）、progress_rel、出路、at-fault。之后是否上公开 1491 / 正式由用户定。

**一条命令**（依赖 `nx-bitcmp-ac` 成功）：`bash experiments/wait_cap/scripts/nx.sh ac-pilot`


## 4. 成本（卡时，估计）

| 项 | 卡时 | 依据 |
|:--|--:|:--|
| (iii-a) 2 臂 × 1 000 步 | ~0.3 | 冒烟 3.7 it/s → 1 000 步约 4.5 min + 加载 / 评估 / rely 约 6 min |
| (iii-a) 读数（wc、led F、dc、devplans、hist、arrival）+ gate | ~0.3 | |
| **(iii-a) 到 pilot 判词** | **~0.6** | 墙钟 ~25 min |
| (iii-b) 2 臂 × 10 000 步 | ~1.0 | 冒烟 3.9 it/s（含启动期；RB0-F-s0 稳态 7.6 it/s）→ 估 25–45 min / 臂 |
| (iii-b) 读数 + gate | ~0.3 | |
| **(iii-b) 到 pilot 判词** | **~1.3–1.6** | 墙钟 ~50 min；到可上闭环的模型另 ~2.5–3 |
| (iv) 到开环判词 | ~0.4 | 2 × 1 000 步 + 读数 |
| 每个过线臂的 held 460 @18 + rf_follow | ~1.1 | WAIT 预登记的单价 |

## 5. 补记 0：恒等与冒烟（读数之前；不读任何能力数）

- **代码恒等**（`nx.sh codecheck 936ac891 NXGG2`，936ac891 = 加这些开关之前的代码，对 8a19229e 全关）：RB0-F 配方 60 步 29 个标量、全部权重逐位相同、dev ADE 1.46977 两边相同
  → `results/next/rt_identity.json` identical；RSG-L18 配方 60 步 89 个标量、全部权重逐位相同、dev ADE 0.529079、on_hash_60 两边 = RSG-L18-s0 的 ea8cacc3e451c577
  → `ac_identity.json` identical。所以存档的 RB0-F-s0 与 WTCP-L18-s0 就是两个变体关掉保留项的对照。
- **接近行**（`nx.py approach`）：navtrain 103 288 行里 0.2–6 m/s 的 63 638 行，选中 6 706 行（静止前车 5 974、TTC < 4 s 1 128，v0 中位 1.94 m/s），训练 split 里 6 617 行。
- **到达重放的仪器**（`nx.py arrival`，路线记忆统一取 RSG-L18-s0 的记录）：k = 1 的 4 s 弧长 P0 0.43、WTCP 3.06、RB0 3.30、RSG 2.90、J2P 2.04，与第 336 条逐个相同；
  gate 代码用 RSG-L18-s0 当替身臂、`--dry` 跑通（不是读数）。
- **冒烟**（40 步，λ 3，接近行开）：RTSMOKEA-L18-s0（RSG 配方）3.71 it/s、22.8 GB，第 40 步 kd 2.47 对 imit 0.87，**dev ADE 0.528 → 0.694**；
  RTSMOKEB-F-s0（RB0 配方，从出厂权重起，P0 即初值）3.88 it/s、21.0 GB，kd 0.004。由此 **(iii-a) 的剂量改为 λ ∈ {0.3, 1}**（λ 3 在 40 步里已把 dev ADE 推高 0.17，
  多半过不了线 5）；(iii-b) 起点就是 P0，保留项从 0 起步，λ 仍为 {1, 3}。另：第一版设计的冒烟 RTSMOKE-F-s0（整条 plan、W / G / S 均分）与 (iv) 的 ACSMOKE-L18-s0（5.69 it/s、18.2 GB）都跑通。

## 6. 疑点（写给早上）

1. **(iii) 的接近行标签粗**：navtrain 的「静止前车」来自 t0 与 0.5 s 两帧的标注框（wt.lead_ctx），减速中的前车只能靠 TTC 规则；停止线行只有池里的 S_N（无 navtrain 停止线行）。
2. **匀速历史的接近行与人的模仿目标在同一 token 上并存**：同一帧以真实历史出现时目标是人，以匀速历史出现时目标是 P0 的弧长。adapter 可能学会「平历史 = P0 模式」这一捷径
   （服务时交接后的历史正是平的或在蠕行，所以这个捷径也许恰好是想要的，也许在巡航时误触发）。线 6 的自由路偏差和第 335 条的闭环读数是检查它的地方。
3. **到达重放只是一个 scene**（第 336 条唯一的人整窗不动的公开 scene），线 1 很窄；它是必要条件，不是充分条件。闭环（held、stop17）才是判断。
   路线记忆统一取 RSG-L18-s0 那次 run 的记录（第 336 条用的是各段自己 run 的记录）：对照重新在同一条件下读，比较只在同一条件内做。
4. **P0 的弧长目标在 AR 行上不是「零」**：P0 在 S 格中位开过人 2.9 m（第 326 条），所以保留项要的是 P0 的程度，不是更好的停车。λ 太大可能连 P0 的晚刹一起学。
5. **(iii-a) 是在 RSG 之后补回来的**：洗掉发生在底座阶段，RSG 的 on-policy 行又在其上；补丁能否在闭环保持，pilot 判不了。
6. **(iv)**：只削弱数值历史一路；第 336 条显示这一路不是「等」的杠杆。
7. 第一版设计（整条 plan 的 LwF、W / G / S 均分、`--kd-own`）的 40 步冒烟 RTSMOKE-F-s0 已跑过（代码路径通），之后按第 336 条改掉；`--kd-mode full` 保留为选项，pilot 不用。

## 补记 1：用户决定把 RTB1 往下走（2026-10-17 ~08:00 CST；写在任何续训、闭环与 dev ADE 分解的输出之前）

**用户决定**：(iii-b) 的 pilot 闸判 STOP（`rtb_pilot_gate.md`）：RTB1-F-s0（λ 1）除线 5 外全过（到达 0.61 m；S_L / Y_P / Y_V 回到 P0 水平；W 0.967、G 0.879；desire 0.96；
自由路偏差 −0.02；横向 0.024；路线依赖仍在），只栽 dev ADE 0.5556 对 RB0-F-s0 0.5276（+0.028，线 +0.01）。用户推翻这个 STOP，**作探索性续跑**，同时分解 dev ADE 的上升；
分解显示真问题就停链。第 2 节的线与判词不改：RTB1 在登记意义上仍是 STOP，下面的一切都是探索，不产生登记判词。

**跑什么**（一条池链，`nx.sh rtb-cont`，两个变体并行；训练只变一个量：底座 RB0-F-s0 → RTB1-F-s0；不加等待行、不加 anti-copy 噪声、不加保留项）：
- **(A) 快**：`RTB1O-L18-s0` = RSG-L18 原命令（= WTC-L18-s0 的命令：`--init` 底座、`--steps 2500 --warmup 1 --lr-scale 0.1`、池 `on1_L18RB0`、`--on-k 13 --on-mode self`、seed 0），
  只把 `--init RB0-F-s0` 换成 `--init RTB1-F-s0`。池是旧底座的 on-policy 行（对新底座 off-policy），不滚新 rollout。
- **(B) 忠实**：按 on1_L18RB0 的建法重建池：RTB1-F-s0 在 onpol-train 上 18 决策滚一次（`ol_chain.sh submit-roll` / `roll1` 原样，6 chunk）→ 行 `L18RTB1`
  （`ol_chain.sh geo` / `tok`（interpreted control smoke10eY，同 rf_round2）/ `strip`）→ `rm_build.py pool --name L18RTB1` → 池闸（`runs/route_follow/poolgate.py`：
  行 ≥ 10 000、route ok ≥ 0.95、route0_share_005 ≥ 0.99；tok 的 R1 / R2 闸不过则 (B) 停在那里）→ `RTB1N-L18-s0` = 同一命令，`--on on1_L18RTB1`。
- **每臂读数**：held 460 @18（`rm_chain.sh read`，6 chunk）、公开静止 scene @10（readlist，SH30_DUMP）、PAI stop17（PAI 提交配置，`pai_box.sh`，与第 335 / 336 条同一仪器）、
  rf_follow（两臂 + RSG-L18-s0 + WTC-L18-s0 的 held 读数，24 个 held 转弯）；顺带（描述）：到达重放（`nx.py arrival`）与 DONE 的 dev ADE。
- **成本（估）**：(A) 训练 ~0.15 + 读数 ~1.3 卡时，墙钟 ~45 min；(B) rollout ~1–1.5 + 行 / tokens / 池 ~0.3 + 训练 ~0.15 + 读数 ~1.3，合计 ~3–3.5 卡时，墙钟 ~2.5–3 h。

**dev ADE 分解**（与链并行；CPU；第 335 条探索 1 / explore_d 的同一读法 `wt_devdecomp.py`，对照换成 RB0-F-s0）：行 = 闸 4 的 1 789 个 dev 行（32 log），
plan 取 `nx.py hist` 已存的 s = 1 plan（读数时已对 DONE 的 dev ADE 复现到 1e−3），P0 取 explore_d 的 devall（同一行序）。读：沿轨迹 / 横向分解；
4 s 带符号弧长偏差按 v0 区间、行为类、上下文（自由路 / 前车近 / 低 TTC / 行人）及 v0 × 上下文；top 5 % 行。

**停链规则**（看分解之前定）：**停** = 上升主要是自由路或低速的变慢，同第 335 条：自由路行承担 ≥ 50 % 的上升，**且** v0 3–8 m/s 的自由路行 4 s 弧长偏差 J − C ≤ −0.10 m
（第 335 条 J2：自由路 0.64、整体 −0.49 m）。停时按 job id `cl cancel` 掉链上所有未完成作业并报告。**不停** = 上升落在接近 / 类等待行（前车近、低 TTC、行人），
或主要是横向且小，或沿轨迹但不是变慢（自由路 v0 3–8 m/s 偏差 > −0.10 m）。

**闭环描述线**（单 seed，只是方向；与第 335 条补记 D 同一套）：held 零分 J ≤ C + 3、24 个 held 转弯完成 J ≥ C − 2、均分配对 CI（会话聚类），C 分别取 RSG-L18-s0 与
2 500 步对照 WTC-L18-s0（与 RSG-L18-s0 训练逐位相同，闭环各自一次读数）；另报 progress_rel、出路、at-fault。**静止 scene**：P0 停在 0.48 m（人 0.50），之后每一段都开 2.3–2.7 m 撞上；
**PAI stop17**：P0 零分 14 / 17，之后各段 16 / 17。「闭环看得到到达时停」= 静止 scene 不接触且距离 ≤ 1.0 m，或 stop17 零分 ≤ 15。
之后是否做 2 seed 正式由用户定。
- 补记 1 追加（2026-10-17 ~08:40，用户批准、在其输出之前）：再加两臂、同一读数，参照另加 RSG-L18-s1 / WTC-L18-s1：(1) `RTB1O-L18-s1` = (A) 的命令，续训 seed 1，`--init RTB1-F-s0`；(2) 第二个底座 seed：`RTB1-F-s1`（RB0-F 配方 10 000 步 + 保留项 λ 1，seed 1，同 RTB1-F-s0 的命令）→ `RTB1OB1-L18-s1`（(A) 的命令，`--init RTB1-F-s1`，seed 1）。实际上是正式的第二个 seed；判读仍是探索性的。另从 (A) 已存的 run 诊断 9 对 4 的 at-fault 前向碰撞（迟到的到达停车、起步冲进空档，还是 veh-52 那一簇）。
- 补记 1 追加 2（~09:05，输出之前）：用空卡给 RTB1O-L18-s0 补大样本读数（优先级低于 b / c / d 的读数）：公开 1491 @10（10 chunk，对存档 RSG-L18-s0 的 guard/read10 配对）与 PAI 281（提交配置 g10 + VCONT + LEAD + ACC，ALPASIM_FAST=1，对存档 pai_g10/acc 的 RSG-L18-s0 配对）；报均分配对 CI、零分、at-fault、出路、前向 / 侧向碰撞、stop17。仍是探索，单 seed。

## 补记 2：边跑边拆 RTB1 的代价，两个针对性变体并行（2026-10-17 ~10:35 CST；写在两个变体与诊断的任何输出之前）

**目的**：第 337 条。RTB1 在闭环里买回了到达时停（四臂静止 scene 0.53–0.63 m 不接触，stop17 15 / 17），代价是 held 零分每 seed 约 +5、at-fault +5、
公开 1491 均分 −0.011 [−0.023, −0.001]、被追尾 24 对 18（PAI 持平）。7 个新零分四臂共有。用户：卡要排满，边分解边跑，出问题就停。仍是探索，不产生登记判词；
第 2 节和补记 1 的线与判词不改。

**两个变体**（一条池链 `nx.sh rtb-v`，两条并行；对 RTB1-F-s0 只改一个量，其余命令逐字相同）：
- **V1 `RTB05-F-s0`**：保留项 `--kd-lam 1 → 0.5`，保留得更弱。行、组份额 `AR:2,W:1`、K 16、接近行 K_nav 8 都不变；D2 仍是 λ 1（RTB1-F-s0 的值），不跟着减，这样只变保留项这一个量。
- **V2 `RTB1AR-F-s0`**：λ 1，加 `--kd-groups AR:1`：池行只从滚动到达组抽，去掉 W 静止等待那 1 / 3；K 16、接近行（本来就是 0.2–6 m/s 的滚动行）、D2 不变。
  **假设**：W 行教的是静止时无条件的保持。在 pilot 里它表现为 v0 < 0.5 起步帧的起步延迟（4 s 弧长 −0.44 m），在闭环里表现为时机不对的再起步。
- 两个都只是现有开关（`--kd-lam`、`--kd-groups`），`lib/retain_train.py` 与 `rm_train.py` 不改，训练代码路径不动，所以不重跑 60 步逐位检查（第 5 节的恒等仍覆盖关时的配方）。
- 每个底座都按补记 1 (A) 的命令续训：RSG-L18 配方 2 500 步，`--init` 换成底座，旧池 on1_L18RB0，seed 0。得到 **`RTB05O-L18-s0`**、**`RTB1ARO-L18-s0`**。
- **底座读数**（开环，描述）：到达重放（`nx.py arrival`）、DONE dev ADE、`wc.py plans`（W / G）、`led.py plans --conds F`（S_L / Y_P / Y_V）、`nx.py hist`（s = 1）→ `nx.py devdecomp`（对 RB0-F-s0，同列 RTB1-F-s0）。
- **续训读数**：held 460 @18、公开静止 scene @10、PAI stop17（提交配置）、到达重放、24 个 held 转弯（`nx.sh cont-reads`）；公开 1491 @10 对存档 RSG-L18-s0
  的 guard/read10（`nx.sh rtb-wide`，只读公开，不跑 PAI 281）。

**停规则**（看任何输出前定）：每个变体在底座训练与到达重放之后有一个 gate 作业（`nx.py vgate`）。到达重放 k = 1 的 4 s 弧长 > 2.0 m（到达停车没了），
**或** DONE dev ADE > 0.5556（比 RTB1-F-s0 差），该变体即 STOP：gate 非零退出，续训与其后全部读数经 `--after` 一起失败、不开跑（另一变体照常）。
底座的其他开环读数照跑，只作描述。

**线**（描述，单 seed；对 a = RTB1O-L18-s0 与 RSG-L18-s0）：
- 「代价修好」= held 零分 ≤ RSG + 3（≤ 37）**且** at-fault ≤ RSG + 2（≤ 6），**同时**「闭环看得到到达时停」仍成立（静止 scene 不接触且 ≤ 1.0 m，或 stop17 零分 ≤ 15）。
- 公开 1491：对 RSG 的均分配对 CI 含 0，被追尾 ≤ RSG + 2（≤ 20）。
- 只报：均分配对 CI（对 a、对 RSG）、7 个共有零分里还剩几个、出路、转弯、底座的 W / G / S_L、v0 < 0.5 起步帧的 4 s 弧长偏差。
- 读法：V2 修好、V1 没修 → 代价来自 W 行（支持假设）；V1 修好但停不住 → 只是拿剂量换代价，没有免费点；都没修 → 代价不在保留项的组和剂量上。

**诊断**（CPU，与链并行，只读已有 run，不交 GPU 作业）：对象是 7 个共有零分（53d3、de4a、6d2b、9d1d、9a67、ddc1、1f54；a 对 RSG-L18-s0，b / c / d 核对），
以及 a 在公开 1491 上比 RSG 多出来的被追尾。每个 scene 对人（日志）和 RSG 读：ego 何时停（v < 0.3 m/s）、停多久、何时再起步、起步后 2 s 的加速度。
分类规则（看数之前定；按把 scene 打成零分的第一个事件）：
(i) **晚起步**：停着时同一时刻人已在走（v > 1 m/s），零分来自进度不足或停着被后车撞；
(ii) **起步撞车**：再起步后 4 s 内接触前方或横穿的车；
(iii) **人没停的地方停了**：ego 停或急刹的位置 / 时刻上人 v > 2 m/s，因此被追尾；
(iv) **横向 / 对向车道**：出路、进对向车道、切角；
(v) 其他。
(iii) 占多数即支持 V2 的假设。结果写 `results/next/explore_f.md`，附几条速度轨迹。

**成本（估）**：每变体底座 ~0.8 卡时（RTB1-F-s0 实测 46 min）、续训 ~0.15、held / 静止 / stop17 ~1.3、公开 10 chunk ~1，合计约 3.2 卡时 × 2；墙钟 ~2.5 h。

## 补记 3：V2 的第二个底座 seed 与 PAI 281（2026-10-17 ~14:00 CST；写在下列作业的任何输出之前）

**目的**：第 339 条把 V2（`RTB1AR`：保留项 λ 1，池行只从滚动到达组抽）立为保留线的配方，但**还不是提交候选**：只有 1 个底座 seed，公开 1491 只是与 RSG 不可区分，
PAI 281 没读（RTB1ARO-L18-s0 的 `rtb-wide` 当时 `WIDE_PAI=0`，只有公开读数）。第 339 条要求谈提交之前先补这两项读数。它们是读数，不是新训练、不是新变体。
仍是探索，不产生登记判词；补记 2 的线与判词不改。

**跑什么**（一条池链 `bash experiments/wait_cap/scripts/nx.sh rtb-v2s1`，一次提交，`--after` 串联，默认优先级 9，不抢占其他车道的作业）：
1. 底座 `RTB1AR-F-s1` = `nx.sh train rtb1ar 1 10000 RTB1AR-F-s1`：RTB1AR-F-s0 的命令逐字不变，只把 seed 0 换成 1
   （RB0-F 配方 + `--kd-lam 1 --d2-lam 1 --kd-groups AR:1` + 接近行）。
2. 到达重放（`nx.py arrival`）与停规则（`nx.py vgate`），和 `rtb-v` 里一样，规则原样：k = 1 的 4 s 弧长 > 2.0 m，**或** DONE dev ADE > 0.5556（RTB1-F-s0 的值）即 STOP；
   STOP 时续训与其后全部读数经 `--after` 一起失败、不开跑，直接报告，不改配方。底座的其余开环读数（wc / led / hist / devdecomp）这次不跑。
3. 续训 `RTB1AROB1-L18-s1` = RSG-L18 配方 2 500 步，`--init RTB1AR-F-s1 --allow-init`，旧池 on1_L18RB0，seed 1
   （与 d 臂 RTB1OB1-L18-s1 的命令逐字相同，只换 `--init`；`nx.sh train rtbo=RTB1AR-F-s1 1 2500 RTB1AROB1-L18-s1`）。
4. `cont-reads v2s1`（与 a ~ d、V1、V2 同一套）：held 460 @18（6 chunk）、公开静止 scene @10、PAI stop17（提交配置）、到达重放、24 个 held 转弯（rf_follow）
   → `results/next/rtb_cont_v2s1.{json,md}`。
5. `rtb-wide RTB1AROB1-L18-s1` 开 PAI：公开 1491 @10（10 chunk，对存档 `guard/read10/RSG-L18-s0` 配对）+ PAI 281 提交配置
   （`PAISERVE` = `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`，`PAIWIZ` = `controller.gains.rel_front_steering_angle_weight=10`，ALPASIM_FAST=1，沿用 RTB1O-L18-s0 那次 PAI 读数的同一仪器，
   对存档 `route_follow/pai_g10/acc` = RSG-L18-s0 配对）→ `rtb_wide_RTB1AROB1-L18-s1`。
6. **RTB1ARO-L18-s0 的 PAI 281**：同一条命令里再调 `nx.sh rtb-wide RTB1ARO-L18-s0`（公开读数已存在，复用原作业 id，不重读；只新交 PAI 8 chunk 与汇总），
   同配置、同参照；覆盖写 `rtb_wide_RTB1ARO-L18-s0`（公开部分与旧文件同源）。
7. 汇总：读数到齐后用 `nx.py v2seeds`（CPU）写 `results/next/rtb_v2_seeds.md`。行 V2-s0、V2-s1、RSG-L18-s0（RSG-L18-s1 有读数的列也列）。
   held 460：零分、均分配对（对 RSG-L18-s0，另列对种子配对的 RSG-L18-s1）、at-fault、出路、转弯 /24、静止 scene、stop17；公开 1491：零分、均分配对、at-fault、出路、前 / 侧 / 后；
   PAI 281：均分配对、零分、at-fault、stop17。

**线与判词（补记 2 原样，不改）**：
- 「代价修好」= held 零分 ≤ RSG + 3（≤ 37）**且** at-fault ≤ RSG + 2（≤ 6），**同时**「闭环看得到到达时停」仍成立（静止 scene 不接触且 ≤ 1.0 m，或 stop17 零分 ≤ 15）。
- 公开 1491：对 RSG 的均分配对 CI 含 0，被追尾 ≤ RSG + 2（≤ 20）。
- 数字按 RSG-L18-s0 的那一组（34 / 4 / 18），V2-s1 也对这一组判；对 RSG-L18-s1 的配对只作描述。只报：均分配对 CI、出路、转弯、被追尾。
- 这两项新读数没有登记线：PAI 281 只报对 RSG-L18-s0 的均分配对 CI、零分、at-fault、出路、前 / 侧 / 后碰撞、stop17。

**会推翻第 339 条的证据（原文）及其读法（读数前定）**：
- **(R1)** V2 第二个底座 seed（RTB1AROB1-L18-s1）的公开 1491 被追尾 **≥ 22**。计数口径与第 339 条的表相同（`collision_rear > 0` 的场景数；RSG 18、a 24、V2-s0 17）。
- **(R2)** V2 的 PAI 281 at-fault / 零分**明显高于** RSG-L18-s0。第 339 条没给数；这里在读数前定成三个描述标记，任一 V2 seed（RTB1ARO-L18-s0、RTB1AROB1-L18-s1）越过任一个即记为 R2 成立：
  零分 ≥ RSG + 5；at-fault ≥ RSG + 4；对 RSG 的均分配对 CI 上界 < 0（RSG = `nx.py wide` 输出里配对场景上的 C 值，RTB1O-L18-s0 那次是零分 76、at-fault 9，即 ≥ 81 / ≥ 13）。
  参照点：RTB1O-L18-s0（a）的 PAI 读数是零分 76 对 76、at-fault 12 对 9、均分 +0.003 [−0.034, 0.038]，三个标记都不越过，第 337 条就是这么读的（「PAI 持平」）。
  未越过标记的差值（含 a 的 at-fault + 3）只作描述，不记为 R2 成立。
- 结果文件按上面的口径逐条报 R1、R2 是否成立，并附原始数，由主会话据此改第 339 条；本车道不改建议、不入 decisions。

**成本（估）**：底座 ~0.8 卡时（RTB1-F-s0 实测 46 min）、到达重放 + 停规则 ~0.1、续训 ~0.15、held / 静止 / stop17 ~1.3、公开 1491 ~5 作业时、PAI 281 ~7 作业时 / 标签（两个标签）；
合计 ~21 作业时，墙钟几小时（与 cl-compile 的编译验证、cl-pack-ch 的调度验证共用池，同优先级，按提交顺序排队）。

**范围收缩（用户；取消于 2026-10-17 ~14:52 CST，本行补记于读数之后）**：seed 1 由用户取消，seeds 表现相近。上面第 1 ~ 5 项（RTB1AR-F-s1 底座、到达重放 + vgate、RTB1AROB1-L18-s1、cont-reads v2s1、其 rtb-wide）
已用 `cl cancel` 按作业号取消，没有任何读数（取消落下时底座刚训完，ckpt 留在盒上；到达重放与停规则没跑）。只保留第 6 项 RTB1ARO-L18-s0 的 PAI 281：它在取消之前已读完（14:33 汇总），
结果在 `results/next/rtb_v2_pai.md`，按上面读数前定的标记逐条核对。R1 不再评估；线、判词与 R2 的标记都不变，本行只记范围变化。
