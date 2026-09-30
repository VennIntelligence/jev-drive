# op-adapt L 后续两项检查：intent 置换、WOD stop 按原因拆分（预登记）

状态: 预登记（写于任何读数之前；此前只读了 decisions 第 78 条、op-adapt L 的预登记与代码、盘点了 box 上有哪些检测缓存，没有跑任何新的前向或检测）。
主题: [decisions 第 78 条](../research/decisions.md) 末尾「会推翻或推进本条的证据」里的两项；母文档 [op-adapt L 预登记](2026-10-01-op-adapt-L-prereg.md)。
约束: 只用**现有** checkpoint（`$DATA_DIR/runs/op_adapt_L/runs/{main-s0,main-s1,main-s2,noint-s0}/ckpt-final.pt`），不训练；一张卡（GPU 0，调度表登记为 `op-adapt-L-fu`），核 8–74；预算约 2 h 墙钟、远低于 1 GPU 小时。原模型读数直接用母 lane 已存的 `readout/O/eval/wodval.npz`（同一批行、同一个前向协议）。r2 的 run 目录不动。

名词（首次使用处各给一句）：**capture rate**（捕获率）= 母预登记 §5 的定义：start = plan 4 s 位移 ≥ 人类位移的一半；stop = plan 最后 0.5 s 的段速 ≤ 1 m/s；turn onset = plan 4 s 横向偏移与人类同号且 ≥ 人类的一半。**sign agreement**（符号一致率）= turn onset 切片上 plan 4 s 横向偏移与人类未来 4 s 横向偏移同号的比例（不要求幅度）。**false turn** = 母 lane 的定义，`|y(4 s)| ≥ FALSE_TURN_M` 的帧占比，读在 control 与 straight_int 切片上（这些帧的人类都没有转）。**intent** = WOD 给每帧的路线意图（直行 / 左 / 右 / 未知），`main` 经 adapter 读它。所有区间按**整段（segment）聚类 bootstrap**，2 000 次，seed 0，同一次重抽用于被比较的两侧（与母 lane 的 `paired_delta` 同一函数）。

## 1. intent 置换

### 1.1 问题

`main` 对 `noint` 的配对差是 turn onset +6.8 pp（decisions 第 78 条）。这个差可以是 adapter 读到了**路线意图**（应当如此），也可以是 adapter 的 embedding 碰巧成了某种**场景线索**的开关（比如 intent 与场景相关：WOD 里左右转意图和路口外观高度相关，adapter 训练时见到的 intent 与场景是同步的，所以「读意图」和「读场景」在训练分布内不可分）。置换 intent 打破这个同步：如果增益真来自意图，换掉意图增益就该消失或反向；如果增益在意图被打乱后依然在，增益来自图像里的场景线索。

### 1.2 条件（WOD val 留出，全部 479 段的偶数帧，与母 lane 读数同一批行，Data.rows("wodval","val",need_future=True)）

模型：`main` seed 0、1、2（汇报 seed 平均，逐 seed 也存），`noint` seed 0（没有 adapter，intent 对它恒无效，作参照；置换后读数必须与 (a) 逐位相同，这是管线自检）。对每个模型在同一份 stage-4 隐状态上重算 policy，只改喂入的 intent 数组：

| 条件 | 喂入的 intent | 精确定义 |
|:--|:--|:--|
| (a) given | 原 intent | 与母 lane 读数相同（自检：必须复现 `main` 的 start / stop / turn onset 捕获率到 1e-6 内） |
| (b1) shuffled within slice | 同层内随机置换 | 每帧按优先级打一个**互斥层标签**：turn_onset > start > stop > stay > control > straight_int > other（任何帧只属于其中第一个成立的层）。`rng = default_rng(20261001)`，对每层内部的 intent 数组做一次 `rng.permutation`，层按上面的顺序处理；层内只有 1 帧时该帧不动。这个条件保持每层 intent 的边缘分布（turn onset 层的左 / 右 / 直行比例不变），只破坏「这一帧的 intent 与这一帧的场景」的配对 |
| (b2) shuffled globally | 全局随机置换 | `default_rng(20261002).permutation` 作用在全部 val 帧的 intent 数组上（turn onset 帧将大多拿到「直行」，因为整体 84% 直行）。不保持层内边缘分布 |
| (c) forced straight | 全部 = 直行（1） | 常数 |
| (d) flipped | 左 ↔ 右 | 2 ↔ 3，直行（1）与未知（0）不变 |
| (e) none | 全部 = 未知（0） | adapter 不加任何东西。描述性：`main` 在拿掉 intent 后还剩多少增益（stage-4 解冻本身带来的部分）；这不是登记的判定条件 |

### 1.3 读数

每个 (模型, 条件) 给：三个切片的 capture rate 与对原模型 O 的配对差；turn onset 的 sign agreement 与对 O 的配对差；turn onset 里按人类转向分左、右的 sign agreement；control 与 straight_int 上的 false turn 与对 O 的差；stay 上的 false start（附加）。`main` 三个 seed 的汇报值 = 每行先对三个 seed 的指示量取平均再做一次 bootstrap（与母 lane 的 seed 汇总同一习惯，CI 不含 seed 方差，seed 间散布另表列出）。

### 1.4 登记的读法（先写死，再看数）

记 G_c = `main`（三 seed 平均）在条件 c 下 turn onset capture 对 O 的配对差；G_noint = `noint-s0` 的同一量（条件 (a)）；G_a = `main` 的条件 (a)。G_a − G_noint 是「intent adapter 的专属增益」（decisions 第 78 条报 +6.8 pp）；G_noint 是不依赖 intent 的增益（stage-4 解冻与场景线索，预期 +4–5 pp）。

- **读路线意图**：(b1)、(b2)、(c)、(d) 四个条件里**每一个**都满足 G_c ≤ G_noint + 2 pp（专属增益被打掉，只剩不依赖 intent 的那部分）。而且 (d) 下 turn onset 的 sign agreement 比 (a) 下**低至少 5 pp**，CI 不含 0（被翻转的意图把 plan 推向反方向，说明意图被因果地使用）。
- **读场景线索**：某一个条件 c ∈ {(b1), (b2), (c), (d)} 满足 G_c ≥ G_a − 2 pp（增益在意图被破坏后保持）。只要有一个条件满足就记为「增益（部分）可由场景解释」，并逐条件写明。对 decisions 第 78 条来说，这条会**推翻**「intent 主要买起转」的说法，把 `main` 对 `noint` 的 6.8 pp 改读成「adapter 是另一个有用的场景适配容量」。
- **部分**：以上两者都不满足（G_c 落在 G_noint + 2 pp 与 G_a − 2 pp 之间）。按比例报「保留的专属增益」= (G_c − G_noint) / (G_a − G_noint)，逐条件写出，不强行归类。
- **false turn 的附带读法**（描述，不判定）：(b2)、(d) 下 control / straight_int 的 false turn 比 (a) 高出 > 2 pp 时，记为「adapter 的转向输出被 intent 直接驱动，没有经过场景核对」；这对部署意味着导航输入错了模型会照走。
- **管线自检不过（任何一项）就停**：`noint-s0` 在 (a)–(e) 之间不逐位相同；`main` (a) 不复现母 lane 的捕获率；(e) 下 `main` 与 O 在 intent 未知帧上的 plan 差 > 1e-3 m（母 lane 已验证零 intent 路径 = 原模型的恒等，(e) 只在 `main` 已训练后比较，所以此项只是期望「接近」，不是判据，不过时只记录）。

## 2. WOD stop 切片按原因拆分

### 2.1 问题

`main` 对原模型在 stop 的 capture +0.307（0.252 → 0.559）。stop 在 WOD 里有几类原因：前车在前面跟停（ego 只是在减速贴车）、行人 / 骑车人在前方、其他（红灯、停止线、让行、没有检测到的原因）。如果增益集中在前车跟停，它是「学会跟车」或标定 / 协议问题，不是学到停车时机；如果在没有检测到任何物体原因的那一类里也有增益，才是对停车时机本身的学习（仍不能说是红灯，因为「其他」是个混合类）。

### 2.2 有什么数据（盘点结果，2026-10-01）

- WOD-E2E 公开格式没有 3D box / lidar 标注（只有图像与 ego 状态）；WOD v2 的 lidar box 在 perception 数据集里，box 上没有这些 e2e 段（`waymo_ds` 只有 perception training 段，不含 e2e val 的段）。**没有标注可用**。
- 现成检测缓存：`processed/fusion_diag/sam/wod/`（SAM 3.1，20 237 张前视，含 479 rater 帧 + p2p3 subset）**只覆盖本切片 1 350 帧里的 266 帧（20%）**，覆盖不够，不用作主标注；`fastperc/dets/*` 只有 P5 与 nuScenes。
- 所以本检查**新跑一遍推断**（不是训练）：`fastperc detect`（`jevdrive/fastperc.py`，YOLO26x-seg，imgsz 1280，半精度，score ≥ 0.5，前视 FRONT 相机图，1 268 张）+ `fusion_q4.lift`（平地抬升，用 `processed/wod_zeroshot/op_calib.json` 的每段 FRONT 标定）。这是对 brief「用现有的」的偏离，记入文末「偏离」：现有缓存覆盖不足，且新检测只是对已缓存的 1 268 张图做一次 inference，不碰任何训练。
- 辅助标注器（独立于 YOLO，同样不需要训练）：原模型 O 自己的 lead 头（`readout/O/eval/wodval.npz` 里的 `lead`、`lead_prob`，`nq4_k.lead_decode`）：`P(lead) > 0.5` 且 lead 的纵向距离落在下面的走廊长度内。它是模型自己的输出，和被比较的 capture 相关，所以只作一致性核对与「严格 other」的第二道筛，不作主标注。

### 2.3 标注规则（先写死）

行集合：`Data.rows("wodval","val","stop",need_future=True)`（母 lane 的 stop 读数行）。设人类未来（0.5 s 格点、后轴自车系）`fut[0..7]`，`S = ‖fut[7]‖`（4 s 位移，m）。

- **走廊**（人类将要走过的路）：折线 (0,0) → fut[0] → … → fut[7] → fut[7] + 8 m·û，û = fut[7]/‖fut[7]‖（`S < 1 m` 时 û = +x）。走廊长度阈值 `Dc = max(15 m, S + 8 m)`（即「停车点之后 8 m 以内」的东西都算在前面，相当于约 2 s 以上车头时距的下限；S 最大约 2 v0）。
- 一个检测的位置 = 它的地面接触点经平地抬升后的自车系 (gx, gy)（必须 lift_ok，gx > 0，gx ≤ Dc）；`d` = 该点到走廊折线的最近距离。
- **vehicle in corridor**：class vehicle（car / motorcycle / bus / truck，`fastperc.COCO_MAP`）且 `d ≤ 1.75 m`。
- **VRU in corridor**：class pedestrian 或 cyclist 且 `d ≤ 3.0 m`（人站在路边、人行横道上也算）。
- **原因标签**（互斥，按优先级）：`lead` = 有 vehicle in corridor；否则 `vru` = 有 VRU in corridor；否则 `other`。另报两个标志都有的帧数（优先级落在 lead）。
- **严格 other** = `other` 且 O 的 lead 头也不报 lead（`P(lead) > 0.5` 且 lead_x ∈ (0, Dc]）。判定（2.5）里的「其他」类用**严格 other**；`other`（仅 YOLO）与 lead 头标注器的各自结果也列出，作敏感性。
- 已知偏差，先写下：前视 + 平地抬升在远处与遮挡下漏检、错位，漏掉的前车会落进 `other`，所以 `other` 里混着未检出的前车跟停（让「其他」类的增益偏高，是 `other` 增益的**上界**）；严格 other 用第二个标注器收一道，但不能完全消除。VRU 类事件数预计很少。

### 2.4 读数

每个原因类：事件数（帧数与段数）、O 与 `main`（三 seed 汇总）的 stop capture rate、配对差与 95% CI（按段聚类，同一次重抽）。另附各类的 `main` 逐 seed 差、类内人类 v0 与 S 的中位数（看这几类的运动学是否可比）。事件少于 **20 个段**的类只报数字，不下判断。

### 2.5 登记的读法

记 Δ_all = 整个 stop 切片的配对差（预期 +0.307）、Δ_lead、Δ_vru、Δ_so = 严格 other 的配对差。

- **学到了停车时机（不止是跟车）**：严格 other 段数 ≥ 20、Δ_so 的 CI 下界 > 0，且 Δ_so ≥ 0.5 · Δ_all（≥ 0.15）。注意这仍不说明是哪种原因（红灯、停止线、未检出的前车），只说「没有检测到前车 / 行人的 stop 上也涨了」。
- **基本是前车跟停**：严格 other 段数 ≥ 20 且 Δ_so 的 CI 包含 0 或 Δ_so < 0.10，同时 Δ_lead ≥ 0.25 且下界 > 0。对 decisions 第 78 条，这条把 stop 的 +0.307 改读成「标定 / 协议问题或跟车」，不再算学到的停车时机。
- **混合**：介于两者之间，按类原样报。
- 严格 other 段数 < 20 时本项判定为「不可判」，这是一个可能的结果（WOD val 里大多数 stop 有前车）。

## 3. 产物

- 脚本 `scripts/op_adapt_l_followup.py`（子命令 `perm`、`stopcause-list`、`stopcause`，同一脚本里的 `figure`）；小表 `research/results/op-adapt-L/followup/*.csv`；图 `research/figs/op-adapt-L-followup.png`（若有）。
- 跑在 box 的 tmux 窗口 `jev:opL-fu` 里（`scripts/tmux_run.sh`），GPU 0，核 8–74，`tqdm`，run 目录 `$DATA_DIR/runs/op_adapt_L/followup/`（`log.txt`、`events.jsonl`）。

## 偏离

（读数前为空；所有后续偏离追加在此，逐条说明原因与是否看过数字。）

## 结果

（待填。）
