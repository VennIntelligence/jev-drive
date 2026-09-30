# op-adapt L：用真实 log 的人类未来做模仿的轻度适配（预登记）

状态: 预登记（写于任何训练结果之前；此前只做了数据准备：WOD val 全量 trunk 缓存、原模型在它上面的输出、切片表，以及一次只查数值恒等的 selftest，没有任何训练步、没有任何适配模型的读数）。2026-10-01 夜间跑，三张卡。之后的偏离全部记在文末「偏离」一节。
主题: [research/decisions.md](../research/decisions.md) 第 77 条（log expert audit）、第 55 / 57 / 66 条（stage 4 解冻的代价、turn desire 在路口前不改 plan、desire 时机）、第 34 / 36 条（WOD 协议与纵向 ×1.06 校准）；[log expert audit](2026-10-01-log-expert-audit.md)；[op-adapt r2](2026-09-29-op-adapt-r2-prereg.md) 与它的 [交接](../tmp/2026-09-30-op-adapt-r2-state3.md)（基础设施只读复用）。
协调: r2 的 run 目录与 chain 文件一律不动（只读它的 `t/samples`、`t/teacher/{wod,nus}`、`teacher/tstd.npy`、rater 缓存）；新产物全在 `$DATA_DIR/runs/op_adapt_L/`。三张卡 0、1、2（box 现在只有这三张，cgroup 75 核、296 GiB 内存），核 8–74，调度表登记为 `op-adapt-L`。

## 用户决定（2026-10-01，原样记录）

1. 用 WOD train 训练是接受的；本 lane 的结果一律描述为「在 WOD train 上轻度适配（lightly adapted on WOD train）」，**不是 zero-shot**。
2. op-adapt r2 的 full 搁置，让位给本 lane；r2 的 run dir 与 chain 文件不删不改，缓存只读复用。

## 1. 要回答的问题

第 77 条量到：真实 log 里「从停止起步（start）」「巡航到停（stop）」「起转前（turn onset）」各有上千个独立事件（WOD train 每类 700–1 100 个事件，起步与起转在 NAVSIM navtrain 里近 3 000 个），而原生 Cinque 的 plan 只复现人类机动的 58–73%（start）、29%（stop，WOD）、61–74%（turn onset，且没有导航输入）。
问题：把人类 log 未来当作这三类切片上的专家轨迹、把 WOD 的路线意图（routing intent，WOD-E2E 给每帧的「直行 / 左转 / 右转」指令）当作导航条件、其余帧蒸馏（distillation，让改后模型的输出贴住原模型）回原模型，做一次**轻度**适配，能不能在留出数据上提高这三类的捕获率（capture rate，见 §5），**同时不变得更保守也不变得更急**。
天然的对照对都在 log 里：起步对「停着且继续停着」（stay）、停车对稳态直行（control）、起转对「路线意图是直行的直行」。**对照帧上的误触发率与捕获增益同等重要**：一个把 plan 整体拉慢或拉快的模型会在捕获上「涨」，在对照上「坏」。

## 2. 数据与切片

### 2.1 划分（全部按整段 / 整场景切，seed 写在脚本里）

| 集合 | 用途 | 内容 |
|:--|:--|:--|
| WOD train 中的 train | 训练 | 2 037 段里去掉 dev 后的 1 833 段（偶数帧，5 Hz，前视，9 个 5 Hz context = 1.6 s，r2 的缓存） |
| WOD train 中的 dev | 选择与 checklist，不进训练 | 204 段（按段，`default_rng(20261001)` 的排列取前 10%）；r2 自己的 val 部分（60 条 stream）不用 |
| WOD val（479 段） | **留出读数**，与训练、dev、选择完全无关 | 全部 479 段的偶数帧（5 Hz）的 trunk 缓存是本 lane 新建的（`scripts/op_adapt_cache.py wodval`，同一个渲染与 trunk 路径，1.7 min），每帧 9 个 context |
| nuScenes | 蒸馏（train scene）；dev（r2 的 50 个 dev scene）做漂移；**val 是域外检查**（不训练，不给 intent） | r2 的 `nus` 域，5 Hz 格点 |
| NAVSIM navtrain | **不用** | r2 的 `t/teacher/nav` 是 2 Hz sample-and-hold 喂法，在有速度的帧上系统偏长约 20 m（第 77 条第 6 点、audit 偏离 2），不能蒸馏；不修（修 = 重跑 GIMM 补帧的 trunk，超出今晚），所以**排除**。navtest 只作 no-harm 读数（走 r2 已有的 GIMM 补帧 harness） |
| WOD test | **不碰**；不向任何榜单提交 | |

### 2.2 切片（沿用 audit 的 BASE 阈值，一个数都不改；只用人类 4 s 未来，后轴自车系，0.5 s 网格）

- **start**：`vmax1 ≤ 0.5`（t0、−0.5 s、−1 s 三个速度的最大）且 4 s 位移 `|p₈| ≥ 3.0 m` 且最大段速 ≥ 1.5 m/s。**stay**：`vmax1 ≤ 0.5` 且 `|p₈| ≤ 0.5`。
- **stop**：`v0 ≥ 3.0` 且最后两段速度都 ≤ 0.5 m/s（3.5 s 内停住并保持到 4 s）。**control**：`v0 ≥ 5`、各段速度与 v0 相差 ≤ 1.5、航向变化 ≤ 5°、横向偏移 ≤ 0.75 m、过去不在转、intent 直行或未知。
- **turn onset**：过去不在转（`dpsi_past < 10°`），未来最大航向 ≥ 30°，第一个 |ψ| ≥ 10° 的时刻在 [0.5, 3.0] s。
- **straight_int**（本 lane 新加，用作起转的对照）：WOD intent = 直行，最大航向 < 10°，`v0 ≥ 2`，且不是 stop。它包含 control 的大部分。
- 其余帧记为 other。in_turn、nudge、lane_change 只作描述行，不训练不判。

WOD train 里的量（全 9 个 context 的帧 / 涉及的段）：train：start 8 400 / 889，stop 3 867 / 579，turn onset 14 448 / 705，stay 13 403 / 526，control 33 095 / 1 304；dev：start 927 / 100，stop 418 / 63，turn onset 1 566 / 76；val：start 2 048 / 241，stop 1 268 / 181，turn onset 3 825 / 179，stay 4 116 / 144，control 7 826 / 315；nuScenes val：start 284 / 34 个 scene，stop 92 / 18，turn onset 540 / 40。**独立事件与聚类**：帧内相邻高度相关，所以所有区间都按**整段（WOD）/ 整场景（nuScenes）聚类 bootstrap**，样本单位是「段」。

## 3. 训练设计

一个 batch = 64 条序列（每条 = 一个 5 Hz 帧 + 它的 9 帧 context），组成：

| 部分 | 条数 | 来源 | 损失 |
|:--|--:|:--|:--|
| 模仿 | 30（三类各 10；只训单类的 arm 全给那一类） | WOD train 的 start / stop / turn onset 帧 | L_imit；除 plan 以外的输出仍蒸馏 |
| 对照 | 18（stay、control、straight_int 各 6；「无对照」arm 里换成 other 帧） | WOD train | 蒸馏（权重 ×2）+ L_cons |
| 其他 WOD | 8 | WOD train 里不属于以上任何切片的帧 | 蒸馏 + L_cons |
| nuScenes | 8 | nuScenes train 帧 | 蒸馏 + L_cons |

三个损失，全部在后轴自车系、0.25 … 4.0 s 的 16 个点上（plan 的 33 点在这些时刻线性插值，相机到后轴的偏移 WOD 1.519 m、nuScenes 1.70 m 按 audit 的同一公式换算）：

- **L_imit**（模仿人类未来）：`mean_k Σ_{c∈{x,y,v}} Huber₁((ĉ_k − c_k)/σ_c(t_k))`，`σ_x = 0.3 + 0.2t`、`σ_y = 0.1 + 0.1t`（m）、`σ_v = 1 m/s`；人类的 v 是 0.25 s 网格上位置的 0.5 s 中心差分。x、y 的尺度让 1 m 纵向和 0.3 m 横向都落在 Huber 的线性区之前。
- **L_cons**（plan 贴回原模型的 plan）：同一个距离，目标换成原模型（teacher）的 plan，作用在所有非模仿帧上。
- **L_distill**（r2 的蒸馏）：改后模型的全部输出（去掉 hidden 与 MDN 标准差）对原模型输出、按 r2 的 `tstd` 归一的 MSE；模仿帧上把 plan 那一块关掉（因为它在被模仿），其余输出照蒸馏。

总损失 `L = λ_i L_imit + dw · (λ_c L_cons + λ_d L_distill)`，`λ_i = 1`、`λ_c = 1`、`λ_d = 10`（r2 dev 选出的）、`dw = 1`（蒸馏权重 ablation 用 0.3 / 3）。

优化：AdamW（weight decay 0.01），fp16 计算 + GradScaler + fp32 master 权重，梯度裁剪 1.0；可训练权重 lr 3e-5（stage 4 与 policy 权重），adapter 3e-4；warm-up 100 步，之后 cosine 到 0；**4 000 步**（25.6 万条序列；模仿帧约 12 万次抽取，对三类共 2.67 万个 train 帧是 4–5 遍）；dev eval 在第 0、2 000、4 000 步；不做早停、不选 checkpoint（末步即读数）。

### 3.1 可训练的部分与 intent 怎么进模型

policy 是冻结的，而且已知它的 desire 输入在路口前不改 plan（第 57、66 条），所以 intent 的进入方式是开放的设计选择。四个候选（selection wave，§4）：

| 名称 | 可训练 | intent 怎么进 |
|:--|:--|:--|
| `sel_s4ia` | stage 4（1.1 亿参数，r2 用的那一组） + adapter | **`ia`**：每个 intent（直行 / 左 / 右）一个可学的 32×512 embedding，加到 9 个 context 帧的 hidden token 上，intent 未知时什么都不加（零初始化，所以第 0 步与原模型逐位相同） |
| `sel_polia` | off-policy plan 通路（temporal summarizer 的 4 个 transformer block + plan 的 hydra 头，1 630 万参数；stage 4 冻结）+ adapter | 同上 |
| `sel_s4polia` | 以上全部 | 同上 |
| `sel_polid` | off-policy plan 通路（stage 4 冻结） | **`id`**：原生 desire 输入承载 intent：左转 = turnLeft、右转 = turnRight，在 t0 前 1.0 s 打一个脉冲（第 66 条的时机），直行 / 未知 = 无 desire；没有新模块，靠改 policy 权重去响应 |

另一个不在候选里、只作 trainable-set 消融的：`tr_ad`（只训 adapter，stage 4 与 policy 全冻）。

### 3.2 与 r2 的区别（写在前面，读结果时要带着）

r2 用行人配对与规则打分的监督，训出 dev 漂移 0.40 m、null 减速率 +11.7 pp（整体更保守）。本 lane 的监督是人类未来，目标域是真实 WOD，且蒸馏 / L_cons 直接作用在 WOD 的对照帧与 other 帧上（r2 的蒸馏主要在 nuScenes / navtrain / sim 的 x⁻ 上，WOD 上的 plan 没被钉住，漂移主要出在 WOD 上 0.71 m）。

## 4. Arms、selection wave 与队列

**selection wave（登记的小集合，seed 0，4 000 步，全部读 dev）**：`sel_s4ia`、`sel_polia`、`sel_s4polia`、`sel_polid`。

**选择规则（写死，只看这 4 个 run 的末步 dev）**：一个配置「合格」= dev 上 (i) other 帧（WOD dev ∪ nuScenes dev）对原模型的 plan 漂移中位数 ≤ 0.10 m、p95 ≤ 0.50 m；(ii) 每个误触发率对原模型的差 ≤ +2 pp（stay 上的「假起步」、control 上的「假停」与「假转」、straight_int 上的「假转」）；(iii) other 帧的 slow 率与 fast 率对原模型的差 ≤ +2 pp。合格者里取三个切片捕获增益（配对差，dev）的均值最高者为 `main`；没有一个合格，就取「各条线的 值 / 线 之比的最大值」最小的那个；并列取捕获增益高者。选出的名字写进 `selection.json`，之后的 ablation 都以它为底。

**队列（选择之后，优先级顺序）**：

1. `main` seed 1、2（连同选择 run 里的 seed 0 = 三个 seed）；
2. `noint`（main 去掉 intent）seed 0；
3. 单类：`only_start`、`only_stop`、`only_turn`（各只模仿一类，对照帧与蒸馏不变）；
4. 蒸馏权重：`dw03`（×0.3）、`dw3`（×3）；`nocontrast`（不放 stay / control / straight_int 对照帧，它们的名额换成 other 帧）；
5. `tr_ad`（只训 adapter）；
6. `noint` seed 1、2；单类与蒸馏权重 ablation 的 seed 1（有余量才跑）；
7. 若时间还有：`main` 8 000 步（看轻度适配是否随步数继续变化）。

trainable-set 的归因：`sel_s4ia`、`sel_polia`、`sel_s4polia`、`tr_ad` 四个同长度 run 就是「stage 4 only（+ adapter）/ adapter only / + late policy」的比较；intent 的归因：`main` 对 `noint`；每类切片的归因：`main` 对 `only_*`；蒸馏与对照帧：`dw*`、`nocontrast`。

## 5. 读数（全部在留出集合上，改后模型与原模型同一批行、同一次 bootstrap）

行集合：WOD val（全部有 9 帧 context 与人类未来的偶数帧，5 Hz）、nuScenes val（5 Hz 格点，无 intent）、479 个 WOD val rater 帧。原模型在**同一批行上重新前向**（同样的 5 Hz 前视 1.6 s context 协议），不用 audit 里官方协议（3 相机、10 s 历史）的 preds；audit 的原模型捕获率只作参照，两者的差在结果里如实列出。

**捕获率**（audit 的定义原样）：start = plan 4 s 位移 ≥ 人类位移的一半；stop = plan 在最后 0.5 s 的段速 ≤ 1 m/s；turn onset = plan 4 s 横向偏移与人类同号且 ≥ 一半。报每个切片的捕获率、对原模型的配对差、和 lon / lat ADE（对人类未来）。
**误触发**：stay 上 plan 4 s 位移 ≥ 3 m（假起步）；control 上 plan 在 4 s 停住（假停）、plan 4 s 横向偏移 ≥ 2 m（假转）；straight_int 上假转。
**漂移与保守 / 急**：other 帧（不在三个模仿切片里的全部帧）上对原模型 plan 的位置漂移（0–5 s 平均 L2 距离）的中位与 p95；plan 2 s 速度低于当前速度 − max(1 m/s, 20%) 的比例（slow）与高于当前速度 + max(1 m/s, 20%) 的比例（fast），各对原模型的差。
**RFS**：479 个 val rater 帧，官方 `rater_feedback_score`，按场景类别取均值再平均（榜单口径），原始 plan 与纵向 ×1.06 校准（waypoint 的 x 乘 1.06，7.92 提交用的约定）各一份，总体与逐类别，对原模型的配对差；CI 对 rater 帧（RFS）/ 帧（类别）bootstrap。
**navtest PDMS**：NAVSIM navtest 12 146 个场景，r2 的 decision-36 harness（GIMM 补帧输入），改后模型的 native plan 与 port 原模型同一次跑；只对 main 的三个 seed 与 `noint` seed 0 跑（每个约 10 min 的 CPU 打分，放在空闲核上）。
**CI**：WOD 按段、nuScenes 按 scene 聚类 bootstrap，2 000 次，seed 0，取 2.5 / 97.5 分位；同一次重抽同时用于改后模型与原模型（配对）。

## 6. 登记线：什么叫「有效」，以及每种结果怎么读

**主 arm = `main`，三个 seed；一条线「过」= 三个 seed 各自都过。** 所有线都用 WOD val（除非另写）、原始 plan（×1.0）；×1.06 校准版并列报告，不作判据。

| 编号 | 读数 | 线 |
|:--|:--|:--|
| L1 捕获增益 | start / stop / turn onset 各自的捕获率配对差 | **每个切片**：CI 下界 > 0 |
| L2 误触发 | stay 假起步、control 假停、control 假转、straight_int 假转，各对原模型的差 | 每一项 CI **上界** ≤ +2 pp |
| L3 漂移与保守 / 急 | other 帧漂移；slow、fast 对原模型的差 | 中位 ≤ 0.10 m 且 p95 ≤ 0.50 m（r2 的 N-drift 线）；slow、fast 的差各 ≤ +2 pp（点估计）；同一批线在 nuScenes val 上再读一遍 |
| L4 RFS 不低于原模型 | 479 帧 RFS 的配对差，原始与 ×1.06 各一份 | 「无害」：CI 下界 ≥ −0.10（r2 的 N-rfs 线）；「不低于」：点估计 ≥ 0；两条分开报，都要在两个约定下成立才算过 |
| L5 ADE 无害 | WOD val 全部帧对人类未来的 ADE（4 s）相对变化 | CI 上界 ≤ +2% |
| L6 域外迁移（nuScenes val，不训练、无 intent） | 同样三个切片的捕获率配对差 | CI 下界 > 0 算「迁移」，逐切片报；同时 L2 / L3 在 nuScenes 上不违反 |
| L7 navtest | PDMS 对同次 port 原模型 | ≥ O − 1（r2 的 N-nav 线） |

**结果怎么读（登记）**：

| 结果 | 条件 | 读法 |
|:--|:--|:--|
| **有效** | L1 三个切片 + L2 + L3 + L4「无害」+ L5 都过 | 「在 WOD train 上轻度适配」的模型在留出段上捕获了更多起步 / 停车 / 起转，且没有变保守或变急；L6、L7 决定它能否说「不限于 WOD」 |
| **部分有效** | L1 只过其中几个切片，其余线（L2–L5）全过 | 只报过的切片；没过的切片按 ablation 看是数据（切片只有几百帧）、条件（intent 没进去）还是模仿本身 |
| **靠代价换的捕获** | L1 过，但 L2 或 L3 不过 | 捕获增益是整体拉慢 / 拉快 / 误触发买来的，不算有效；读 `dw*`、`nocontrast` 看蒸馏权重能不能救 |
| **无捕获增益** | L1 不过 | 模仿这三类的路径在这个可训练集合与步数下不成立；读 selection wave 与 `tr_ad` 看是不是可训练部分太少 |
| **RFS 变差** | L4「无害」不过 | 即使捕获过了也不能作为改进 |

主 arm 在实质上失败（如所有登记设置下漂移或误触发都超线）时，登记的 ablation 队列**仍然跑完**，让今晚出归因；结果如实写成失败。**不放宽任何登记线去让 checklist 过**：实现 bug 要修，线不动。

对 ablation 的比较（描述，同一批行的配对 bootstrap）：`main` 对 `noint` → intent 的贡献；`main` 对 `only_x` → 三类互相帮忙还是互相抢；`main` 对 `dw*` / `nocontrast` → 蒸馏权重与对照帧的作用；trainable-set 四个 run 互比；三个 seed 的散布。

## 7. 分级启动与 sanity checklist（写死）

**1 个单位（登记）**：`sel_s4ia`，seed 0，**800 步**，dev eval 在第 0、400、800 步，看：
1. loss 全部有限，L_imit 与总损失后 20% 步的均值低于前 20%；
2. 第 0 步 dev：漂移 ≈ 0（中位 ≤ 1e-3 m）、各捕获与误触发的差 = 0（adapter 零初始化的恒等性）；
3. 吞吐与显存：记实测，与 r2 的 288 序列 / s、31.8 GB（batch 64）相比在 ±30% 以内，且单个 run 显存 ≤ 40 GB（每卡放两个）；
4. dev 上三个切片的捕获增益均值 > 0，且至少两个切片的方向为正；
5. dev 漂移中位 ≤ 0.15 m、每个误触发差 ≤ +5 pp（短训练，线比终线宽，只用来拦崩坏）；
6. dev eval 一次不超过 5 min。

**约 10 个单位 = selection wave**（4 个候选全长）：每个 run 的 dev 上看：完成率 100%、无 NaN / OOM；末步 dev 漂移中位 ≤ 0.15 m；plan 2 s 速度分布对原模型的 KS 统计量 < 0.1（非退化）；至少一个候选合格（§4 的规则）。全都不合格不停批（登记的 ablation 队列照跑），但要在状态里如实标出。

**全量**：队列由一个自推进的链脚本跑，每个 run 结束后自动读数（`eval`、`read`，main 与 noint 另加 `navtest`）。任一 checklist 不过：诊断、修真 bug，不动线。

## 8. 成本（实测后填，估计写在这里）

r2 实测：stage 4 训练、batch 64 单卡约 288 标注序列 / s，峰值约 32 GB；冻结 stage 4 的 run 便宜得多（只前向 stage 4）。估计每个 stage 4 run（4 000 步）训练约 15 min + 3 次 dev eval（各约 2–3 min），读数（WOD val 4.9 万行、nuScenes val、rater）约 5 min。三张卡满载约 8 h 可排 20–30 个 run。实测吞吐、显存与每 run 时长会写进执行日志。

## 9. 不做的事与限制

- 不碰 WOD test，不提交任何榜单；不改 r2 的任何文件；不用 navtrain；不改任何登记线。
- 人类未来是单条轨迹，「捕获一半」的阈值是任意的（只用来跨切片比较）；多模态下原生 plan 与它不同不等于错。
- WOD val 上没有独立于 WOD 的检查（同一数据集、同一相机 rig、同一 intent 定义）；nuScenes val（无 intent、另一个相机与国家）是唯一的域外读数，样本小（每切片 18–40 个 scene）。
- intent 是数据集给的指令，不是模型自己推的；deployment 需要导航输入。turn onset 的「原生只有 61–74%」里有多少是缺 intent，由 `main` 对 `noint` 回答。
- 原模型的 stop 在 WOD val 官方协议下只有 7%（n = 41，audit），5 Hz 前视协议下另算，两者的差异会在结果里列出。

## 执行日志

（按时间追加：每次看 dev / 读数都记一行，写明看了什么。）

- 2026-09-30 22:39–23:0x（准备，无训练）：box 现状 = 3 张 RTX 6000D（0、1、2，空闲）、cgroup 75 核、296 GiB 内存；tmux 里没有 r2 进程（r2 chain 已在 stage 1 checklist 后 ERROR 退出），调度表 `op-adapt-r2` 行按用户决定 finish、登记 `op-adapt-L`（GPU 0,1,2，核 8–74）。WOD val 全量 trunk 缓存（482 个 stream、53 204 个 slot、13 GB）1.7 min 建好，原模型在它上面的输出（teacher，5 min）建好，切片表建好（含 dev 划分）。selftest：原模型的 LModel 前向与 r2 teacher 的 plan 逐位相同（max abs 0.0）；`sel_s4ia`、`sel_polia` 第 0 步与原模型逐位相同；`sel_polid` 第 0 步不同（turn desire 在冻结 policy 上本来就改 plan，selftest 里 max 101 m 是用「保持 8 个 block」的旧写法量的，之后改成登记里写的 t0 前 1.0 s 单脉冲）；batch 组成 16 / 30 / 18（其他 / 模仿 / 对照）；损失与梯度到得了 adapter 与 stage 4。

## 偏离

（无；此节在预登记提交之后才可能出现内容。）
