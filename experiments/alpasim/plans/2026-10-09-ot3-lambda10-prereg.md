# OT3 预登记：λ 10 基底上的比赛 driver 与 yaw-rate rows（2026-10-09，任何新 checkpoint 的闭环分数读出之前写定并 push）

lane OT3，decision 212（λ 10 基底与输入标准，第 1–3 项）、213（yaw-rate rows，第 4 项）、214（只在第 5 项跑了才写）。接 d205（M1：强 hinge 触发、plan 续接合成 slot 里的 yaw rate、MPC 一比一执行）、d201 / d198（离轨行）、d189（AP2 输入标准）、d170（强 hinge 的开环收益）、d211（同配方 seed 差 0.0077，plan 平均无用，仿真只对同一 scene 列表与分块确定）。只有本地 run；不向 AlpaSim 注册、warm-up 或提交。

写本文件时已经知道的东西：M1 的全部读数（P2H10-F-s0 在 700 scene 上 0.9484，s1 0.9496，分块与本 lane 不同）、OT2 的 SH30-F-s1 / AP2-AB-s1 / APO 的闭环正在跑（分数没有读）。本 lane 新训的 checkpoint 此刻还没有训完，闭环分数一个都没有。

## 问题

1. λ 10 / 0.3 m 换到 AP2 的输入标准（训练进去的 cold-start keyframe mix）之后，是否保住 P2H10 的低漂移，同时拿到 AP2 的「偏慢 scene 减半」。
2. λ 10 加离轨行（±0.5 m / ±2° 与 ±1.5 m / ±5°）在闭环里值多少，在哪一类零分上。
3. 把「合成 slot 里呈现的自转动一定预示未来」这条在 log 上永远成立、在闭环里对自己的误差也成立的关系打断（yaw-rate rows），闭环航向误差的增长是否变小，转弯是否不受伤。
4. 这些成分叠起来，哪一个 checkpoint 该拿去提交，它在 navtest / navhard 上付出多少。

## 与旧条目的差别（不重测已定的机制）

- d132 / d143 / d146（重投影 DAgger、精确自车引擎 on-policy、纵向也闭环）：那里的状态分布来自 rollout，副作用是起步停滞。这里没有 rollout、没有 on-policy、不动纵向、不在 AlpaSim 的 rollout 上训练（公开 scene 是 navtest 的 log）；扰动是加在 navtrain 真实行上的脚本化 profile，只取 v > 3 m/s 的 token。d141 定下的 plane 引擎原样使用。
- d161（replay hinge，plan 预补偿跟踪器）：loss 里不放跟踪器。
- d189 排除的选项（置零 slot 的冷启动、route waypoint 进 adapter）不重测。AP2 的输入标准原样使用，只把 hinge 从 30 / 0.5 m 换成 10 / 0.3 m。
- d198 / d201（离轨行）：那里是强 hinge 基底。这里是 λ 10 基底，并加 ±1.5 m 一档。
- d205 的输入侧阻尼（W0 / W0G，把呈现的自转动置零）不重测：置零伤转弯。yaw-rate rows 是加性的，呈现的转动 = log 的转动 + 注入的转动，什么都不置零。
- d211：不做 ensemble。
- d198 的离轨行与本文件 yaw-rate rows 的差别（看 `op_adapt_h.drift` 才清楚）：离轨行的航向误差在 1.6 s 内线性爬到 dψ，所以注入的 yaw rate 恒等于 dψ / 1.6 s，与航向偏置是同一个数；模型只看图像里的车道方向（偏置）就能解这些行，不需要改变「续接 yaw rate」。d205 记的「OT30 在后段拉回、注入量不减」与此一致。yaw-rate rows 把两者解耦。

## 臂

全部：冻结 Cinque、P2 adapter、navtrain 12 shard、split `navsim/op-parity-full`、10 000 步 × 128、hinge λ 10 / margin 0.3 m、seed 0 / 1。行的占比均为每个 batch 的份额。

| 臂 | 训练代码 | 输入标准 | 附加行 |
|:--|:--|:--|:--|
| `P2H10-F-s{0,1}`（已有，基线） | pp_train | NAVSIM | 无 |
| `P2-F-s{0,1}`（已有，无 hinge 参照） | pp_train | NAVSIM | 无 |
| `AP2H10-AB-s{0,1}` | `ap2_train.py --cold backwarp` | AlpaSim（m = 1..4 mix 0.1 / 0.1 / 0.1 / 0.7） | 无 |
| `OT10a05-F-s{0,1}` | `ot_rows.py train --ot ot1 --ot-mass 0.1` | NAVSIM | d198 的 `ot1` 缓存（±0.5 m、±2°），13 / 128 |
| `OT10a15-F-s{0,1}` | `ot_rows.py train --ot ot2 --ot-mass 0.1` | NAVSIM | OT2 的 `ot2` 缓存（±1.5 m、±5°；12 个 prep 作业的 DONE 文件齐了才用），13 / 128 |
| `YR10m10-F-s{0,1}`、`YR10m25-F-s{0,1}` | `ot3_rows.py train --std navsim --ot-mass 0.10 / 0.25` | NAVSIM | 新缓存 `yr1`（下节），13 / 128 与 32 / 128 |
| 组合臂（有条件，见「顺序」） | `ot3_rows.py train --std alpasim` 或 `ap2_ot.py train` | AlpaSim | 第 1–3 项与第 4 项里最好的一族行 |
| 第 5 项 `YR30-F-s{0,1}`（有条件） | `ot3_rows.py train --hinge-lam 30 --hinge-margin 0.5` | NAVSIM | `yr1`，占比同选中的那档 |

离轨行与 yaw-rate rows 放在 NAVSIM 标准上训，是因为 `P2H10-F` 是现任最好的 driver，这样每一族行都只改一个因素；AP2 标准是另一个因素，单独一臂；两者的叠加留给组合臂。

## yaw-rate rows 的构造（`experiments/alpasim/scripts/ot3_rows.py`，缓存前缀 `yr1`）

- 相对 log 位姿的航向误差 ψ(t) 分段线性，节点在 keyframe：t ≤ −1.5 s 为 0，三段 [−1.5, −1]、[−1, −0.5]、[−0.5, 0] s 的增量 a1、a2、a3 独立，N(0, 1°) 截断在 ±2.5°。40% 的行是 `recent`：a1 = a2 = 0，即模型上一步的 plan 把车多转了 a3 之后的那个状态（有 yaw rate，几乎还没有偏置）。其余 60% 是三段 random walk：最近 0.5 s 的注入 yaw（a3）与总航向偏置（a1 + a2 + a3）相关系数约 0.58，可以分开回归。
- 横向误差 y(t)：y(0) ~ U(±0.3 m)（recent）/ U(±1.0 m)（random walk），dy/dt = v ψ（车沿自己的车头方向走）；历史中 |y| > 2 m 的重抽。
- 图像：10 帧历史（4 个 key + 6 个 lattice）每帧都是最近真实 key 的一次 plane warp，目标位姿 = log 位姿(t) ∘ (0, y(t), ψ(t))（`ot_rows._job` 原样）。于是合成 slot 里呈现的自转动 = log 的转动 + 注入的转动，与闭环里 driver 的 `lattice` 看到的是同一种东西（真实 key 处在偏了的位姿上，中间帧沿 ego 轨迹 warp）。
- ego 特征：历史位姿重表达到扰动后的 t0 系；速度 / 加速度 / command 用 log 的（AlpaSim 标准的臂用它自己的定义，route 重表达到扰动系）。
- 目标：log 未来在扰动位姿下的重表达（不续接注入的转动，回到车道方向）；不做平滑的恢复路径（同 d198）。hinge 用 `ot_rows.off_hinge`。
- 这些行从不作 anchor 行，AlpaSim 标准下只作 m = 4 的决策。决策时不用任何特权输入；没有 WA-JEPA 的权重或特征。

幅度的依据：d205 的 replay 里 plan 在 0.5 s 的 yaw sd 是 1.49°（SH30），log 直行 token 上过去 0.5 s 的转动 sd 0.34°；N(0, 1°) 覆盖强 hinge 级别的自注入，λ 10 的注入更小，落在分布中部。

## 闭环前的探针与闸门（不含闭环分数，可以先读）

`ot3_rows.py probe`，留出行（`navsim/op-parity-full-dev` 的 yaw-rate rows 与同一批 token 的 log 位姿行，按 log 整簇 bootstrap）：

- α：plan 在 0.5 s 的 yaw（扰动行减同 token 的 log 位姿行）对 a3 的系数，回归里同时放总航向偏置与横向偏置。α = 0.93 是 d205 的放大器，0 是目标。
- β：对总航向偏置的系数（目标 −1）；recent 行上的净响应（α + β 的直接估计）；由此算 e[k+1] = (1 + α + β) e[k] − α e[k−1] 的谱半径（跟踪器一比一执行 plan yaw 时航向误差的递推；只报告，不设线）。
- legit slope：未扰动留出 token 上 plan 0.5 s yaw 对 log 过去 0.5 s 转过的 yaw 的斜率，分全部行进 token 与转弯 token（|4 s 航向变化| ≥ 20°）；旁边给 log 自己的斜率（navtest 上 0.97）。
- **行构造闸门**（shard 2，先于其余 11 个 shard）：只在 log 上训过的 checkpoint（P2H10-F-s0 / s1、P2-F-s0、SH30-F-s0）在这些行上的 α 必须 ≥ 0.5。不到 0.5 说明注入的转动没有以闭环里的方式呈现给模型，行的构造不对，停下来查，不训。零偏置对拍（64 行）必须复现 W 缓存。
- **占比的选择**（只用探针，不看闭环）：取满足「α ≤ 0.5 × P2H10 的 α，且两个 seed 的转弯 legit slope 都 ≥ 0.87」的最小占比；都不满足 α 条件时取转弯 legit slope ≥ 0.87 者中 α 较小的；都不满足转弯条件时取转弯 legit slope 较大的，并记「续接被伤」。选中的那档先跑闭环；另一档只在预算有余时跑。

## 闭环读数

- scene：d201 的固定 700 scene（`c0b/lists/all.txt`，27 个 log），**所有 driver 用同一组三个分块** `chunk0 / 1 / 2`（d211：仿真只对同一 scene 列表确定）。M1 的 P2H10 / P2 是按 400 + 300 的列表跑的，所以四个基线在这三个分块上重跑；M1 的旧分数只用来量「列表组合噪声」（同一 checkpoint 两种分块的逐 scene 差），不进判线。
- chain：`ot2_loop.py`（`OT_LANE=ot3`；c0b 的看门狗 20 min、缺 scene 补跑、只留零分 rollout 的 `.asl`），同时在池中的仿真栈 ≤ 3。driver 代码不改：`run.sh <dir> sh30`（NAVSIM 标准的 checkpoint）与 `ap2`（AlpaSim 标准）。每个 checkpoint 跑一次。
- 每个 driver：mean scene score、零分按类（at-fault 碰撞 / offroad / 出 corridor）、偏慢 scene（0 < score < 1）数、at-fault 事件数（`offroad_or_collision_at_fault` 之和）、航向误差 sd。
- 航向误差：M1 那张图的集合与算法原样（诊断 400 scene 里 log 直行、起步速度 > 2 m/s 的 scene；`m1_decisions.py` + `m1_analysis.growth_table`；log 轨迹只作标尺），每个 checkpoint 一条决策 2–9 的 sd 曲线与均值曲线，图进结果文档。

## 判定线（写定，不事后改）

**主比较**：每个配置两个 seed 的逐 scene 平均，减 `P2H10-F` 两个 seed 的逐 scene 平均，700 scene 配对。报告两种 95% CI：按 scene 重采样、按 log 整簇重采样（27 个 log），各 10 000 次、seed 0。**起判定作用的是按 log 整簇的 CI**（M1 的 held-out 结果按 scene 过、按 log 不过，所以事先说死）。

一个配置成为**提交候选**，当且仅当同时满足：

1. 差 ≥ **+0.010**，且按 log 整簇的 CI 下界 > 0；
2. at-fault 事件数（两 seed 平均）不高于 P2H10 的（两 seed 平均）；
3. 偏慢 scene 数（两 seed 平均）≤ P2H10 的 + 10。

各配置各自对线，不做多重比较校正，全部报告。多个过线取两 seed 平均分最高者；都不过线时照实写，记「未过线的最佳」，**建议提交的仍是 `P2H10-F`**。要交单个 checkpoint 时取 seed 0（事先定死，不按 700 scene 的分数挑 seed：同配方 seed 差是噪声）。

功效（写在前面）：700 scene / 27 个 log 上配对差的整簇 CI 半宽约 0.02（M1：[+0.0140, +0.0549]）。P2H10 在 700 scene 上总共只丢 0.052（25 个零分约 0.036，偏慢约 0.016），+0.010 且下界 > 0 实际要求拿回剩余损失的四成左右。过不了线不等于没有效应；点估计与两种 CI 都报告。

**第 4 项「修法成立」的判据（与分数无关，三条都要）**：

- (i) 留出行上的 α（两 seed 平均）≤ 0.5 × `P2H10-F` 两 seed 平均的 α；
- (ii) 转弯 token 上的 legit slope 两个 seed 都 ≥ 0.87；
- (iii) 闭环决策 9 的航向误差 sd（M1 集合，两 seed 平均）≤ 0.75 × `P2H10-F` 两 seed 平均（M1 读到 2.7°，本 lane 在自己的分块上重量）。

读法事先写明：(i)(ii) 过而 (iii) 不过 = 行在离线上学会了，但 λ 10 下闭环的航向增长不是这些行覆盖的那种（渲染域，或增长主要不来自 yaw-rate 续接）；(iii) 过而分数线不过 = 漂移已经不是 P2H10 剩余零分的主因（M1 的 held-out 上 P2H10 的 16 个零分里 M 类只有 2 个，T / R 类 14 个）；(ii) 不过 = 续接被伤，按 d205 的 W0 读法预计转弯零分上升，照实报告。对离轨行的臂报告同样三个读数，不设线。

**基线读数（第 1 项，无线）**：`P2H10-F` s0 − s1、`P2-F` s0 − s1 的配对差与两种 CI，零分 / 非零分不一致的 scene 数；`P2-F` 对 `P2H10-F`；同一 checkpoint 在 M1 分块与本 lane 分块下的逐 scene 差。

**AP2 标准的问题（第 2 项，除主线外的两个读数）**：`AP2H10-AB` 的偏慢 scene 数对 `P2H10-F`（d201：AP2 53 对 SH30 95），决策 9 的航向误差 sd 对 `P2H10-F`（「保住低漂移」= 不超过 P2H10 的 1.25 倍）。

## 顺序、条件臂与停止规则

1. 立即：四个基线的闭环；`AP2H10-AB`、`OT10a05-F`、`OT10a15-F` 的训练（各先 20 步 smoke 读峰值）与闭环；同时做 yaw-rate rows 的 prep（零偏置对拍 → shard 2 → 行构造闸门 → 其余 11 个 shard）。
2. yaw-rate rows 过闸门后训 `YR10m10-F`、`YR10m25-F`（各 2 seed），探针选占比，选中的那档跑闭环。
3. 组合臂（至多一个配置，2 seed），条件：`AP2H10-AB` 对 `P2H10-F` 的点估计 ≥ +0.005，或偏慢 scene 少 ≥ 10 且 at-fault 事件不高。满足时在 AlpaSim 标准上叠加 {OT10a05, OT10a15, 选中的 YR} 里两 seed 平均分最高且点估计高于 P2H10 的那一族行；没有哪一族高于 P2H10 时不训组合臂。
4. 第 5 项（至多一臂）：只在 yaw-rate rows 的 (i)(ii)(iii) 全过、且已用卡时 ≤ 26 时训 `YR30-F`（λ 30 / 0.5 m + `yr1`），对线同上，另报它对 `SH30-F` 的航向误差 sd。
5. 候选（或未过线的最佳）的旁读与护栏做完后停。超预算时的裁剪顺序：第 5 项 → 另一档 YR 占比的闭环 → `P2-F-s1` → 非候选配置的 navhard。

## 旁读与护栏

- 每个最终 checkpoint：navtest 与 navhard（`python -m jevdrive.bench`，NAVSIM 标准读；AlpaSim 标准的 checkpoint 这样读不是它的训练标准，只看开环代价），对 `P2H10-F` 与 `SH30-F` 两 seed 的配对差（按 log 整簇）；AlpaSim 标准的离线读数（`ap2_offline.py`，navtest，m = 1 / 4，2 035 token 子集的 EPDMS，各配置 seed 0 对 `P2H10-F-s0`）。
- HUGSIM 64（`spec_plan_smooth`，每场景一次）只对最好的那个配置的两个 seed：起步停滞数与 HD 对 `P2H10-F`，护栏 = 起步停滞不增加且 HD 点估计 ≥ −0.03（d198 的口径）。

## 后落盘 scene 上的确认

700 scene 已经多次用来在设置之间做选择：其中 400 个在 M1 里选了 P2H10（对 SH30），全部 700 个被 c0b / OT2 / 本 lane 用来比较配置。最终候选（或未过线的最佳）与 `P2H10-F` 两个 seed 还要在 DL1 之后落盘的 shard 上读一次：scene 列表取写结果时 `.done` 标记齐全、不在 700 个里的全部 scene，三个交错分块，四个 driver 同一组分块；报告配对差与两种 CI、零分分类。这是确认读数，不改 700 scene 上的判定；结果文档里写明哪些 scene 参与过选择。新 shard 在本 lane 收尾时还没落盘的话，写明没有做，留给下一步。

## 成本与分阶段（docs/long-runs.md）

实测参照（2026-10-09 的 pool 记录与本 lane 的 smoke）：AlpaSim 标准训练 37.1 GB VRAM、6.2 it/s 约 27 min（申报 42 GB）；`ot_rows.py` 训练 25.8 GB（申报 31 GB）；非可回收 RAM 约 40 GB（申报 45）；离轨 prep 每 shard 5–7 min、29–42 GB（申报 44）、19 核、RAM 29 GB（申报 40）；闭环每个分块作业实测 VRAM 12–32 GB（申报 32）、RAM 28–41 GB（申报 42）、8 核，CPU 争用下每块 35–40 min。卡时按「申报的 VRAM 占一张卡 84 GB 的份额 × 小时」计（M1 的口径：400 scene 0.35 卡时）。

| 项 | 数量 | 卡时 |
|:--|:--|--:|
| 闭环 700 scene | 基线 4 + 新 6 + YR 2（另一档 +2）+ 组合 2 + 第 5 项 2 = 14–18 个 driver × 约 0.7 | 10–12.6 |
| 训练 | 6 + 4 + 2 + 2 = 14 次 × 约 0.25 | 3.5 |
| yaw-rate rows prep | 12 shard + 对拍 | 1.0 |
| 探针 | 3 次 | 0.5 |
| navtest / navhard | 约 14 个 checkpoint × 约 0.2 | 2.8 |
| AlpaSim 标准离线读数 | 3 次 | 1.0 |
| HUGSIM 64 | 2 个 checkpoint | 2.0 |
| smoke、补跑余量 | | 2.0 |
| 合计 | | **约 23–26，上限 35** |

分阶段：训练 20 步 smoke 读峰值 → 全量；prep 64 行零偏置对拍 → shard 2 → 闸门 → 其余；闭环沿用 d201 / d211 验证过的 chain，不再做 pilot（driver 代码未改，checkpoint 格式与 OT30 / AP2 相同），但每个新 checkpoint 的第一个分块作业结束时查 driver 健康（inference / input error 为 0）。磁盘剩余 < 150 GB 立即停。

## 限定（写在前面）

- 每个配方 2 个 seed；27 个 log；本地渲染，未与官方环境对分；公开 700 scene 不是 private 评测集，且已参与多次选择。
- plane 引擎对路面以上物体的畸变随偏置变大（d141、d198 的同一限定），yaw-rate rows 的转动幅度小于 ±1.5 m 一档，但畸变仍与「状态本身」没有分开。
- 目标是 log 未来的重表达，0.5 s 内就要求回正，物理上达不到；回归学到的是折中，α、β 的目标值（0、−1）是方向不是可达值。
- 航向误差的读数只在诊断 400 scene 的 log 直行子集上（log 轨迹只在那批 scene 的记录里），不是 700 个。
- 行只在 m = 4；冷启动三步里已经偏了的状态没有对应行。
- 探针量的是开环响应；闭环里渲染域、MPC 与多步累积不在探针里。
