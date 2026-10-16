# WAIT-TRAIN 预登记：同域「人等 / 人起步 / 人停下」行 + 只沿 plan 自身路径的弧长损失，能不能把「等」训回来而不伤形状（2026-10-16，任何模型在这些行上的输出之前）

本文件在任何训练臂的读数之前 push；之后只追加补记，不改上文。已知的只有：行数与格子统计（日志本身的）、泄漏检查、
池的逐位检查、60 步恒等运行的逐位比较（这些都不读任何能力数）。方案即用户批准的
[2026-10-16-wait-train-design.md](2026-10-16-wait-train-design.md)（下称「设计稿」），偏离逐条写在第 9 节。

## 1. 问题与来源

第 326 条：出厂 Cinque（P0）在人等的帧上 0.978 等、人走的帧上 0.334 等（D = 0.64）；第一个 adapter 阶段 P2-F 就降到 0.012，之后所有阶段（含 RSG-L18）
0.005–0.008；没有任何阶段横向绕行。病是「静止即直着起步」。来源是数据（navtrain_full 静止 8 421 行里人 4 s 不动只有 82 行）。
问题：在 RSG-L18-s0 的原配方上，只加同域（nuPlan / NAVSIM trainval）等待 / 起步 / 停下行与一个只管纵向的弧长损失，
**能否在 waitcap_test（从不进训练）上把「人等时等」拉到 ≥ 0.60，同时「人走时走」保持 ≥ 0.85、形状不动**；若能，闭环守不守得住。

## 2. 行（训练用，trainval 新下载的 17 个相机归档）

- 来源：`results/trainval_archives.tsv` 的 17 个归档（111 个 log，`results/trainval_logs.txt`），盒上 `datasets/navsim/_archives/state/<name>.done` 全在。
- **去掉 3 个含 navtrain dev token 的 log**（navsim/op-parity-full-dev；否则 dev ADE 闸会被同 log 的训练行抬高），剩 108 个。
- `wt.py select`：每帧 `wc.log_rows` + `wc.cell_of`（格子定义与 WAIT-CAP 完全相同：W_L、W_N 等；G_L、G_N 起步；S_L、S_N 行进中停下；F_L 不取），
  另要求 4 个历史帧的 CAM_F0 在盘上；同 log 同格隔 ≥ 2 s（4 帧），每格随机 ≤ 5 000（种子 20261016）。split `navsim/waitcap_train`。
- 行数：见补记 0（select 跑完后填，只是日志统计）。
- **泄漏检查**（`results/wait_train/select.json`）：111 个 log 对 held 460（AlpaSim nuPlan held，含 452 配对）、公开 1491、PAI 281（clipgt，另一个数据集）、
  waitcap_test 的 130 个 log：按 nuPlan db 名精确比较、并按驾驶会话（date_veh）比较，**全部 0 重叠**（select 时断言，有重叠即停）。
- 链（`wt.sh rows`，一次提交）：select → index（navsim2，trainval SceneLoader）→ `pp_prep.py --data idx_waitcap_train --frames warp --no-side`
  → `rm_build.py nav`（路线记忆，与 waitcap_test 同一做法）→ teacher（出厂 Cinque port fp16 在同一 W token 上，pp_prep / on_rows 的 make_teacher 同一调用；
  同调用在 on1_L18RB0 的 64 行上复现其存储 teacher 作为协议检查）→ pack。

## 3. 池

- `on1_WAIT`：名字 `on:<token>`；tab = 缓存行的 ego / pose / vel / acc / cmd / fut（人的 0.5–4 s 后轴位姿）/ cam / lht / speed + `cell`、`s_h`（(n, 8) 人的弧长，
  从后轴原点沿人的 8 个位姿累加）；agent / road / route 标签全部 `ok = False`（**不进任何形状 hinge**）；front = 缓存的 W token，teacher 如上，路线特征 = rm_build 的。
- **S 行（S_L、S_N）喂匀速历史**：`rf_lonol.flat_rows` 同一规则（vx 保留，vy = ax = ay = 0，3 个历史位姿按 v0 沿自身折线重排，`vhist.respace`）；
  标签仍是人减速的弧长。W / G 行的历史本来静止，不改。原始 ego 另存 `ego_measured`（不读）。
- **合并池 `on1_L18RBW` = on1_L18RB0 全部行在前（前缀 s）⊕ on1_WAIT（前缀 w）**（`rf_pait.py merge`）；pack 时检查：前 17 429 行的 ego / fut / teacher /
  路线特征与 on1_L18RB0 逐位相同、front 抽 64 行逐位相同、新行 agent / road ok 全 False。`wc_arc.npz`（tokens、s_h、kind）放在合并池目录：
  merge 只保留字符串标签，所以 s_h 单独成文件（第 9 节 b）。

## 4. 损失（只有纵向）

`lib/wc_arc.py`，作为 rm_train 的 `--on-lon` 模块（on_train 的池项钩子，`--on-mode self` 下作用于每个池行）：
s_p = plan 自己 8 个后轴位姿从原点起的累计弧长（段长 sqrt(dx² + dy² + 1e-6) − 1e-3，站立 plan 梯度有限），d = s_p − s_h；
W / G 行（kind 1）smooth L1（beta 0.5）双向；S 行（kind 2）只罚超出人：smooth L1(relu(d − 1 m))；8 个位姿求和；L18RB0 行（kind 0）恰为 0。
弧长对旋转不变，梯度只沿 plan 自身切向，没有形状项。与其他池项同一归一：λ 10（`--on-lon-lam` 默认）× ho_w 3 × 行和 / 模仿行数。
新行的非 plan 头照常蒸馏到出厂 teacher（Losses43 的 hinge-only 行路径，family `on`），与每个池行相同。已过 CPU 已知答案检查（站立 plan / 人走：双向罚、停下格不罚；
移动 plan / 人等：双向罚、停下格只罚超出 1 m 的部分；梯度横向分量为 0）。

## 5. 抽样与臂

- 配方：RSG-L18-s0 的原命令不变（`--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape
  --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-k 13 --on-mode self`，两 seed 都从 RB0-F-s0 起，与 RSG-L18 一样），池换成 on1_L18RBW，加 `--on-lon wc_arc`。
- 抽样（rm_train `--on-weight`，字符串键 `cell`）：L18RB0 行权重 1；新行总质量 = q · 17 429 / (1 − q)，q = 剂量 / 13（13 个池行里新行的期望个数）；
  新行内部 W（W_L ∪ W_N）: G（G_L ∪ G_N）: S（S_L ∪ S_N）= 1 : 1 : 1（即静止行等 : 走 = 1 : 1，S 占新行的 1 / 3），格内按自然比例。具体权重 `results/wait_train/pack.json`。
- **J**（联合）：上面的抽样；**C**（对照）：同一命令、`--on-weight pool=w:0`（新行权重 0，在池尾）：rm_train 的 weight-0 性质给出与 RSG-L18 逐位相同的抽样。
- **恒等证明**（读数之前完成，补记 0）：60 步 WTGG-off（RSG-L18 原配方、池 on1_L18RB0、无 `--on-lon`）对 WTGG-w0（C 的命令：合并池、新行权重 0、`--on-lon wc_arc`），
  标量与权重逐位相同、on_hash_60 都等于 RSG-L18-s0 存档的 ea8cacc3e451c577（`wt.py bitcmp`，rf_jdagger 的做法）。不相同即停，不跑 pilot。

## 6. 剂量 pilot 与闸

- seed 0、1 000 步：**WTJ2P**（q = 2 / 13）、**WTJ5P**（q = 5 / 13）、**WTCP**（C，同 1 000 步，ADE / 横向对比用）。
- 读：`wc.py plans` 在 waitcap_test（第 326 条的 4 674 行，不进训练；人的历史，与第 326 条同一输入）；`wt.py devplans` 在 navtrain s0 的 dev 行里行进中（≥ 2 m/s）的行；
  dev ADE = 训练结束时 rm_train 记录的全 dev 行 ADE（DONE 里的 `dev_ade`）。分类与第 326 条完全相同（等 = plan 4 s 弧长 < 1 m；撞进前车 = 车头点进前车框）。
- **闸（每个剂量各自判，全部成立才算过）**：
  1. W 上等 ≥ 0.60（RSG 0.005，P0 0.978）；
  2. G 上走（= 1 − 等）≥ 0.85；
  3. W_L 撞进前车 ≤ 0.05（RSG 0.31）；
  4. navtrain dev ADE ≤ C + 0.01 m；
  5. 行进中 dev 行 2 s 处 plan 横向（后轴 y）对 C 的逐行 |差| 中位数 ≤ 0.05 m（形状不动）。
- **登记的失败形状（launch-lock，第 167 / 174 条）**：W 上等对 C 上升 ΔW ≥ 0.20 而 D（= 等(W) − 等(G)）的上升 < 0.5 ΔW（W 涨、G 也跟着不走、D 不涨）
  → 该剂量判失败，**停，不做任何变体去救**。两个剂量都不过 → 停，不做全量。
- **选剂量**：只一个过 → 用它；两个都过 → D 大的那个；两者 D 差 < 0.03 → 小剂量（j2）。
- 只报不判：S_N / S_L 停住比例与 S_N 超出人中位（人的减速历史，与第 326 条同口径）；配对 CI（log 聚类 bootstrap）；绕行（W_L ∪ S_L）。

## 7. 全量与闭环（按成本顺序，每段由前一段的线放行；gate job 自己提交下一段）

- 全量：J（选中的剂量）与 C 各 2 seed × 2 500 步（WTJ{2|5}-L18-s{0,1}、WTC-L18-s{0,1}）；同样读 waitcap_test 与 dev 行（主报，不再判闸）。
- **(1) held 460 @18 决策**（rm_chain.sh read，6 chunk / tag，4 个 tag）：共同场景上零分数（2 seed 平均）**J ≤ C + 3** → 放行 (2)；否则停。
- **(2) 逐场景看**：公开 1491 里唯一的人整窗不动 scene（`2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790`，前车 1.6 m，第 325 / 326 条全撞）4 个 tag @10 决策；
  PAI 的 17 个 `cor_route_end`（第 315 条，人停了车没停）J-s0 在 PAI 提交配置下（一个 17 场景 chunk）对存档参照 RSG-L18-s0 同配置（第 328 条 `pai_g10/acc`）。
  放行 (3) 的线：J 的 2 个 seed 里至少 1 个在静止 scene 上得分 > 0，**或** J-s0 的 stop17 零分数 < 参照的；两者都没有 =「闭环里看不到能力」→ 停。
- **(3) 公开 1491 @10 护栏**（ol_chain.sh readlist，10 chunk / tag，4 个 tag）：每 scene 2 seed 平均分，J − C 配对均值的 95 % CI 下界（按驾驶会话聚类）**> −0.01**，
  且零分数（2 seed 平均）**J ≤ C + 5** → 放行 (4)；否则停。
- **(4) PAI 281 全集**：J-s0 在 PAI 提交配置（`controller.gains.rel_front_steering_angle_weight=10`，`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`，第 327 / 328 条；
  参照的同 8 个 chunk 表 `pai_box/tokyo/r281/rsg/chunks`）对存档 RSG-L18-s0 同配置（`runs/route_follow/pai_g10/acc`），按场景配对。报均分配对 CI、零分、stop17 零分。最后一段，只报。
- **主报告 = 三项能力**：人等时等的比例（W）、人走时走的比例（G）、PAI stop17；分数是护栏不是目标。

## 8. 成本（卡时，估计）

| 项 | 卡时 | 依据 |
|:--|--:|:--|
| 行（pp_prep W 约 8–10 k 行、teacher、路线记忆 CPU） | ~0.3 | 第 326 条 4 674 行 pp_prep 2 min |
| 恒等 2 × 60 步 | ~0.05 | |
| pilot 3 × 1 000 步 + 读 | ~0.4 | RSG-L18 2 500 步 382 s |
| 全量 4 × 2 500 步 + 读 | ~0.6 | |
| held 4 tag × 6 chunk | ~4.5 | chunk ~11 min |
| (2) 4 个单场景 + PAI 17 场景 | ~0.6 | |
| 公开 4 tag × 10 chunk | ~7.5 | chunk ~11 min |
| PAI 281 J-s0 | ~7 | 8 chunk × ~52 min |
| **合计** | **~21** | pilot 不过只花 ~0.8 |

全部经 GPU 池，`--after` 串接；gate job 在过线时提交下一段（rf_jdagger / rf_recov_post 的做法）。

## 9. 相对设计稿的偏离（理由）

a. pilot 闸按用户批准的计划：去掉设计稿的「S_N 超出人中位 ≤ C」（改为只报），加「W_L 撞进前车 ≤ 0.05」与登记的 launch-lock 形状。
b. `s_h` / `kind` 写在合并池的 `wc_arc.npz`（`rf_pait.py merge` 只保留字符串标签与共有键）；on1_WAIT 的 tab 里也存着 `s_h`、`cell`。没有 `mode = "lon"`：
   用 `--on-mode self`（RSG 的模式），新行靠 ok = False 不进 hinge、只进 `--on-lon` 项，C 因此与 RSG-L18 逐位相同。
c. J 的池抽样是 rm_train `--on-weight` 的加权有放回抽样（L18RB0 行在 J 里也就不再是 RSG 的无放回均匀抽样）；q 是 13 个池行里新行的期望个数，不是固定个数。
d. 去掉 3 个含 navtrain dev token 的 log（第 2 节）。
e. 闭环第 (2) 段的放行线是本文件定义的（设计稿只写了「逐场景看」）；PAI 对照用存档的 RSG-L18-s0 同配置读数（用户计划：J 对 RSG-L18、同一提交配置），不另跑 C 的 PAI。
f. held 的线用 2 seed 平均零分；公开护栏用每 scene 2 seed 平均分的配对 CI（会话聚类）与 2 seed 平均零分。


## 修订 A：加让行行 Y_P / Y_V（用户决定，理由 = 第 331 条；写在任何训练臂读数之前，替换上文与之冲突处）

第 331 条（CAP-LEDGER）：P2-F 丢掉的是纵向「看画面减速」一族，除了等（W）与平历史下停在静止前车后（S_L），还有**平历史下让行人 / 横穿车（Y_P / Y_V）**。
用户决定把让行行加进同一个联合臂。逐项：

1. **格子**：第 331 条 `experiments/cap_ledger/scripts/led.py` 的 `log_cells` / `cells_of` 定义不改，用在同样的 108 个 trainval log 上（select 改在 navsim2 环境跑，
   因为 led 要读地图）。**一帧只进一个格**：先按 wc.cell_of（W_L、W_N、G_L、G_N、S_L、S_N），不在其中再看 Y_P，再看 Y_V。所以已在 S_L / S_N 里的让行帧只算一次、留在 S 格
   （S 与 Y 的处理完全相同，只影响抽样分组）。同 log 同格隔 ≥ 2 s、每格 ≤ 5 000，与其他格相同。led 的 `log_cells` 只在停止线上下文缺失时丢帧，W / G / S 行因此可能比纯 wc 少几行。
   人工普查（不读任何模型输出，`results/wait_train/ycount.md`，当时只有部分相机在盘）：108 个 log 上 Y_P 240 帧 / 隔 2 s 后 91 行 / 31 log（其中 45 帧已在 S_L / S_N），
   Y_V 111 帧 / 47 行 / 25 log（19 帧已在 S_N）。
2. **Y 行的处理与 S 行相同**：喂匀速历史（`flat_rows`），标签 = 人的弧长，损失 = 只罚超出人（smooth L1(relu(d − 1 m))，kind 2），无形状项、agent / road 标签 ok = False。
3. **新行内部抽样**：静止 2 / 3（W 1 / 3、G 1 / 3，等 : 走 = 1 : 1）；S + Y 合计 1 / 3，按两者行数比例分；若 Y 不足新行抽样的 1 / 12，提到 1 / 12（S 相应为 1 / 4），
   并在 `pack.json` 里记 `y_raised`。剂量不变（13 个池行里新行期望 2 个 / 5 个），一个联合臂 + weight-0 对照，1 000 步 pilot 后按登记的链走。
4. **pilot 闸不变**（第 6 节五条 + launch-lock）；**新增登记的次要读数（不是闸）**：第 331 条账本格 S_L、Y_P、Y_V 在 F 读（匀速历史，`capledger_test` 行，
   `led.py plans --conds F`，`led.measures` / `led.success` 不改）上 J 对 C 的成功比例与配对差（log 聚类），列在 pilot 报告里；全量臂同样读。
5. **池**：pack 尚未构建过，合并池仍叫 `on1_L18RBW`（含 Y 行），60 步 weight-0 恒等运行 + bitcmp 在这个池上做（第 5 节不变）。

## 修订 B：分批建行、对照先跑（用户要求盒子满载；写在任何读数之前）

1. **分批建行**：不等 17 个归档全到。批 A = 当时已有 `.done` 的 9 个归档里的可用 log，批 B = 其余归档到齐后的 log（`wt.sh rows-batch <B>`；
   归档 → log 表 `results/trainval_archive_logs.json`）。每批各自 select（只做 2 s 稀疏，不截断）→ index → pp_prep W → 路线记忆 + teacher，缓存目录 `idx_waitcap_tr<B>`，
   split `navsim/waitcap_tr<B>`。**每格 5 000 的截断在 pack 时对全部批的并集做**（种子 20261016，按 log / 帧排序后随机取），与一次性 select 的结果同分布；
   pack 断言各批合起来正好覆盖 108 个可用 log；打包后的 token 登记为 `navsim/waitcap_train`。泄漏检查每批各做一次。
2. **对照臂先跑**：C 的三个 checkpoint（WTCP-L18-s0 1 000 步、WTC-L18-s0 / s1 2 500 步）用 **RSG-L18 原配方（池 on1_L18RB0、无 `--on-lon`）** 训练，
   不等合并池。理由：第 5 节的恒等证明（WTGG-off 对 WTGG-w0 逐位相同）正是「C 的命令 ≡ RSG-L18 原配方」，所以这三个 checkpoint 就是 C。
   **bitcmp 不逐位相同 → C 作废**，按合并池 weight 0 重训 C，pilot 与之后各段都等它（gate job 依赖 bitcmp 成功）。
   C 的读数（waitcap / dev / 账本 F 的 plan、held 460 @18）随训练一起提交，但只在登记的 gate job 里读。J 臂仍在 bitcmp 之后才训练。

## 补记 0（读数之前）

（select / pack / 恒等运行完成后填：行数与格子（含 Y）、泄漏检查、pack 检查、抽样份额、bitcmp 结果。）

## 修订 C：WAIT 与 desire 训练（D1 + D2）合成一个正式臂、一条闭环链（用户决定 2026-10-17，方案 A；写在 WAIT pilot 任何输出之前）

来源：第 332 条（出厂 Cinque 的变道 desire 通路是部分、分级的；P2-F 起被洗掉 69 %、左边 97 %）与 [desire_cap 方案草图](../../desire_cap/plans/2026-10-16-desire-route-design.md) 第 2 节（D1 / D2 / D3）。
用户决定：开 D1 + D2，与 WAIT-TRAIN 合成**一个**联合正式臂，由**一条**闭环链读；另在 held 段加转弯完成数的线。本修订替换上文与之冲突处；写时
WAIT 的 pilot 尚未开始（链在等最后 5 个 trainval 归档），对照臂 C 的任何能力数都没有读过。

1. **链的改动（已推 54020417，先于本文）**：pilot 照第 6 节跑、照登记的闸与选剂量规则判（不变）；但 `wt-gate-pilot` **不再提交 WAIT-only 全量**，只把选中的剂量
   （j2 / j5 / STOP）写进 `runs/wait_cap/train/pilot_pick.txt`。`wt.sh full` 只在第 5 条的退路下手动跑（`WT_FALLBACK=1`），否则拒绝。
2. **正式臂（a）**：**WD** = WAIT（pilot 选中的剂量，池 on1_L18RBW + `--on-lon wc_arc`，抽样与第 5 节相同）+ D1 + D2，一个臂，2 seed × 2 500 步，
   RSG-L18 配方从 RB0-F-s0 起（与第 5 节相同）；tag `WD{J2|J5}-L18-s{0,1}`。对照 **C 不变** = WTC-L18-s{0,1}（pilot 用 WTCP-L18-s0）。
   **C 为什么仍有效**：D1 / D2 全关（不喂 desire = 今天的全 0 队列、不抽 D1 行、无 D2 项）时，trainer 必须与 WAIT 配方逐操作相同：D1 与 D2 各用自己的 rng 流
   （[seed, 0, 131] / [seed, 0, 127]），关时不建、不抽、不多一次前向。证明：60 步 **WDGG-w0**（合成命令：WAIT 新行权重 0、D1 每 batch 0 行、D2 权重 0）对 WTGG-off，
   标量与权重逐位相同、on_hash_60 = RSG-L18-s0 的 ea8cacc3e451c577（`wt.py bitcmp` 同一比较）。逐位相同 → C 的三个 checkpoint 就是合成命令在权重 0 下的对照（与修订 B 同一逻辑）；
   不相同 → C 作废，用合成命令在权重 0 下重训 C（WDC*），合成 pilot 与之后各段都等它。
3. **D1 / D2 的做法（实现细节在合成 pilot 之前补进补记 C1）**：
   - 前向：策略的 desire 队列（`_to_copy`，(B, 1, 33, 8)，块 32 = 最近 0.2 s）作可选输入；None = 今天的 0，调用不变（逐位）。脉冲格式与 dc.py 完全相同。
   - **D1（变道行带 desire，模仿人）**：navtrain 训练 split 的 `lc` 行（rf_cfpool geom：4 s 内换车道且 +4 s 离 t0 车道 ≥ 2 m）中，用行自己的路线记忆跑
     `lib/route_desire.py` 的规则（lc_intent 与速度门，D0 线所有，**只 import 不复制**）触发、且触发方向与人 4 s 横移方向一致的行；
     每 batch 抽 4 行（自己的 rng 流）替换最后 4 个普通行（同 `--lc-cf` 的位置），喂规则给出的变道 desire，目标 = 人的真实轨迹（普通模仿项，不加新项）。
     合并池（on1_L18RBW，含 WAIT 行）里规则在其喂入路线记忆上触发的行：抽到时同样喂该 desire，损失照该行自己的配方（S 行 = hinge + 蒸馏；WAIT 行 = 弧长项），
     **不加模仿目标**（RSG 的 `self` 行没有人的目标，`fut` 是恢复路径、配方从不读）。脉冲年龄：缓存里 navtrain 行是孤立帧、读不到规则在该 log 上的首次触发时刻，
     所以每次抽到时从 {0, 0.4, 1, 2} s 均匀抽（dc.py 的年龄集）。所用规则版本（git hash）记在行表与 run summary；合成 pilot 之后不再换版本。
   - **D2（desire 增量蒸馏，保通路）**：每 batch 的锚行（adapter 关）与非锚普通行中按 1 / 8 抽的行（自己的 rng 流），随机侧（LCL / LCR）与年龄 ∈ {0, 0.4, 1, 2} s，
     对这些行再前向一次带 desire；损失 = 行平均的 8 个后轴位姿 (x, y) 上 ‖(plan_θ(x, d) − sg[plan_θ(x, 0)]) − (plan_P0(x, d) − plan_P0(x, 0))‖ 的均值（m），× λ_D2；
     plan_θ(x, 0) 取同一 batch 主前向并停梯度（none 的 plan 不受这一项影响）。教师 = 出厂 Cinque 同一输入（dc.py 的同一脉冲写法），离线存 `teacher_desire.npz`。
   - **D3（守卫）**：即第 2 条的逐位证明；dev ADE 线见下。
4. **合成 pilot（b）**：seed 0、1 000 步，WAIT 用 pilot 选中的剂量，D2 两个剂量 **λ_D2 ∈ {1, 3}**（方案草图「D2 权重 1 / 3」按本仓库写法读作两个剂量；两臂只差 λ_D2），
   tag `WD{J2|J5}P{1|3}-L18-s0`；在 WDGG 逐位证明之后训练。每臂**全部成立才算过**：
   - WAIT 的五条 pilot 闸 + launch-lock 形状（第 6 节原文、原读数、对 WTCP-L18-s0）；其中闸 4 即 dev ADE ≤ C + 0.01 m；
   - desire 离线闸（dc.py plans 协议，capledger_test 的 C_fast 行，年龄 0，LCL / LCR 合并、带符号朝 desire 一侧）：4 s 横移均值 ≥ 0.9 × P0（P0 = 同一读数的存档 P0 plan，1.04 m）、
     LCL 与 LCR 各 ≥ 0.8 m、方向对（4 s 响应 > 0.1 m 的比例）≥ 0.95；
   - 只报：PAI on-policy 行 P_lca（路线要求变道处）朝路线一侧 4 s 响应（对 1.5 m 报，不判），turn desire 行与 RSG-L18 同口径（第 332 条线 2）。
   - 选：只一个过 → 用它；两个都过 → λ_D2 = 1（干预小的）。都不过 → 第 5 条退路，不做第三个变体。
5. **退路（c）**：截止 = WAIT pilot 判词（`pilot_gate.json` 写出）后 **24 h**。截止前合成 pilot 没有一个臂过（两臂都不过、D0 规则未落地、或 D1 / D2 未就绪）
   → 按原登记跑 WAIT-only 全量（`WT_FALLBACK=1 wt.sh full <dose>`，第 7 节不变），WAIT 不丢。WAIT pilot 本身判 STOP（两剂量都不过或 launch-lock）→ 不做任何全量，合成的也不做。
6. **held 段新增转弯线（d）**：第 (1) 段除「零分 J ≤ C + 3」外，加 **held 转弯完成数**：rf_follow.py 在同一 held 460 @18 读数上（6 chunk / tag）数 `turn` 场景的
   `completed`；转弯场景由 log 自己在 18 个决策内转完的弯定义，所有 checkpoint 同一集合（第 303 / 307 条的 24 个）。线：**2 seed 平均 J ≥ C − 2**（容差，不要求严格相等），
   每 seed 也报，并报共同转弯场景上的配对差（log 聚类）。两条都过才放行 (2)。这一条抓「变慢 → 转弯变差」。
7. **其余不变（e）**：闭环链 held → 逐场景 → 公开护栏 → PAI 281 的线、读法、对照（C 与 RSG-L18-s0 存档 PAI 参照）全部照第 7 节，只是 J 换成 WD。
   serve 配置也照登记（driver 不喂 desire，队列全 0）：这条链量的是 WAIT 能力与「desire 训练关着 desire 时不伤闭环」；desire 在闭环里的作用是 D0 线（driver 侧喂 desire）的读数，不在本链。
8. **成本增量**：教师 desire 预算 ~0.7 卡时、WDGG 60 步 ~0.05、合成 pilot 2 臂 + 读 ~0.5、dc 读 ~0.2；正式 + 闭环链与原登记相同（~20 卡时，WAIT-only 全量不再跑）。
