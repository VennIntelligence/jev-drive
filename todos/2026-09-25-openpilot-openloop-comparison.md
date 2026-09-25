# openpilot 开环地位：WOD-E2E 与 NAVSIM 上的并排对比，以及补齐公平比较缺的几格

状态: 进行中（预登记写于任何新分数之前，2026-09-25 15:40）
上游: 第 34 / 37 / 39 / 40 条；[wod-e2e.md](2026-09-24-zeroshot-exam/wod-e2e.md)、[navsim.md](2026-09-24-zeroshot-exam/navsim.md)、
[nuscenes-physicalai.md](2026-09-24-zeroshot-exam/nuscenes-physicalai.md)、[driving-backbones](2026-09-24-driving-backbones/README.md)、
[p5route](2026-09-25-openpilot-temporal-p5-and-route.md)
结果文档: [research/openpilot-openloop-standing.md](../research/openpilot-openloop-standing.md)（写完后）

## 为什么要做

openpilot 已定为主线 backbone（第 40 条）。它的开环读数散在五份文档里，口径不一：WOD 上有原生 plan、冻结 `temporal` + 薄 head、
Alpamayo、我们自己的 ego head，但没有一张配对的总表；NAVSIM 上只有原生 plan 的 zero-shot 行，**没有我们自己的 head**，
也没有文献里最该对照的 blind baseline（Ego Status MLP）；而第 36 / 37 条已经指出 NAVSIM 的 2 Hz、1.5 s 输入让 openpilot 的原生 plan
几乎不代表模型本身，却没有量过这一项在 WOD 上值多少。本 todo 把这些缺口列出来，便宜且不依赖闭环 infra 的就跑掉。
不训练、不微调 openpilot；head 是项目既有的冻结特征 + 线性读出。

## 缺口清单

| # | 缺什么 | 做不做 | 理由 |
|---|---|---|---|
| G0 | WOD 总表：Cinque / Lebowski / small 原生、`temporal` + `ridge_late` / `cls_late`、2b 的 desire 变体、Alpamayo、我们的 ego head、cv、log，全部在同 479 个 rater 帧上配对 | 做（CPU） | 逐帧预测都在盘上，只缺一次统一的配对 |
| G1 | 同一批 WOD rater 帧上，把 openpilot 的输入换成 NAVSIM 的时间轴（2 Hz、1.5 s、sample-and-hold、零状态），量时间轴单独扣多少 | 做（GPU 分钟级） | 把两个 benchmark 接起来：NAVSIM 上的差距有多少是输入协议造成的 |
| G2 | NAVSIM 上我们自己的 ego-only head（`ridge ego`、`cls ego K1024`），navtrain 拟合、navtest / navhard 官方 devkit 打分 | 做（CPU + 少量 GPU） | NAVSIM 上没有任何我们的 head；文献的 Ego Status MLP（PDMS 65.6）比两个 zero-shot 模型都高，必须并排 |
| G3 | NAVSIM 上冻结 openpilot `temporal` + 薄 head（第 40 条协议搬到 NAVSIM） | 做（GPU 约 1 h） | 主线用法就是「冻结 backbone + head」，只报原生 plan 等于只报了一个没适配的读出 |
| G4 | WOD-E2E test split：Cinque / Lebowski 原生 plan 的提交包 | 准备好、**不提交** | 下载在续（剩 74 个 shard），提交配额每 30 天 6 次，要用户决定 |
| G5 | NAVSIM 原始 10 Hz 相机（nuPlan sensor blobs） | 不做，列为 open | TB 级下载，不便宜；G1 给出同一问题在 WOD 上的量级 |
| G6 | navhard 上 openpilot 的 cmd 变体、WOD 上的 cmd / desire 变体 | 已有（navhard cmd 已跑）/ 由 p5route 2b 给出 | 不重跑 |

## 预登记

所有选择在看任何新分数之前写死；之后的改动记到「偏离记录」。

### G0：WOD 总表

- 帧集：val 479 个 rater 帧（第 34 条同一批）。RFS 榜单口径 cluster mean 为主，frame mean 并列，ADE@5s vs rater_best。
- 行：log、cv、`ridge ego` / `cls ego K1024`（train 训）、Alpamayo nav 期望 / medoid-of-6、openpilot small / Cinque / Lebowski 原生（exam）、
  Cinque / Lebowski `temporal` 的 `ridge_late`、`cls_late`（第 40 条 (iii) 的 `heads_train/20260925-110819`）、
  p5route 2b 的 desire 原生 plan 与 desire head（出来后并入）、公开榜 test 行（不可配对，只作量级）。
- 配对：每一行对 cv、对 `cls ego K1024`、对 Cinque 原生，frame-mean 按 sequence bootstrap（每个 sequence 一帧，等价于按帧），10 000 次，95% percentile。
  cluster mean 的 CI 用 cluster 内分层重抽（第 34 条同一实现）。

### G1：WOD 上的 NAVSIM 时间轴

- 模型：small、Cinque、Lebowski；渲染与第 34 条考试相同（FRONT / FRONT_LEFT / FRONT_RIGHT → road / wide，最近邻），desire 0。
- 变体（都从零状态开始，只在 t0 取 plan）：
  - `base`：考试原样（10 s、10 Hz，每帧喂两次；Lebowski context-rate）。已有预测，不重跑。
  - `ctx1.5`：只保留 t0 前 1.5 s 的 10 Hz 帧（f−15 … f）。20 Hz 时钟 t = −1.5 … 0 共 31 步，每步喂时刻 ≤ t 的最近一帧；Lebowski 走 t = −1.4 … 0 的 8 个 context 步。
  - `nav2hz`：只保留 f−15、f−10、f−5、f 四帧（NAVSIM 的 −1.5 / −1.0 / −0.5 / 0 s），同样的 31 步 / 8 步 sample-and-hold。
    这是 `scripts/navsim_zs_openpilot.py` 的 `schedule()` 原样搬过来。
- 读数：RFS cluster mean；`variant − base` 的配对 frame-mean Δ 与 CI；5 s 终点纵向偏差（对 log）。
- 判据（写死）：记 S = (base − nav2hz) / (base − cv)，即 NAVSIM 时间轴吃掉了 openpilot 相对 cv 的多少优势（frame mean）。
  S ≥ 0.5 → NAVSIM 的 openpilot zero-shot 行主要是输入协议的读数，不能当模型能力；S ≤ 0.25 → 时间轴不是主因，NAVSIM 上的差距是真实的
  ODD / 技能差；之间 → 两者都有。`ctx1.5 − base` 与 `nav2hz − ctx1.5` 分开报：前者是 context 长度，后者是帧率。
- 预期（跑之前写下）：nav2hz 让三个模型都掉到 cv 附近或以下（S ≥ 0.5），纵向偏差显著变正（第 37 条的「速度被放大约 2.5 倍」）；ctx1.5 掉得少。

### G2 / G3：NAVSIM 上的薄 head

- 拟合集：navtrain（官方 split 的全部 stage-one token）；评测：navtest 全部 12 146 token（PDMS，devkit v1.1；EPDMS，devkit main @ 0a380a9）
  和 navhard two-stage（EPDMS）。打分与第 37 条完全相同：离线算 8 个位姿，回放 agent 交给官方 devkit。
- ego 特征（只用 NAVSIM agent 合法可见的量，`scripts/navsim_zs_index.py` 冻结的 AgentInput）：4 个历史位姿（x, y, yaw，t0 后轴系）、
  4 个速度、4 个加速度、当前 driving command one-hot，共 32 维，在 navtrain 上标准化。
- 目标：logged future 8 个位姿 (x, y, yaw)，0.5 … 4.0 s。
- head（第 9 / 10 条和 P0 的 recipe，不改结构）：
  - `ridge ego`：ridge，24 维输出，λ 由 navtrain 上按 log 分组的 5 折 CV（最小 ADE）选。
  - `ridge_late <tap>`：在 `ridge ego` 的按 log 分组 out-of-fold 预测之上，用标准化后的 `temporal` 拟合残差，λ 同样按组 CV 选。
  - `cls ego K1024`：navtrain future 的 (x, y) 16 维 k-means，K = 1024（seed 0），线性 softmax（L-BFGS，`planner.ce_solve`），
    λ 在按 log 分组的 inner split 上按 top-1 ADE 选；输出得分最高的 anchor，heading 取该 anchor 成员 yaw 的圆均值。
  - `cls_late <tap>`：同一 vocabulary，`temporal` 的线性 softmax 加在 `cls ego` 冻结的 out-of-fold logits 上（late fusion）。
- `temporal`：Cinque、Lebowski 的 512 维 `temporal` tap，输入与第 37 条考试逐位相同（CAM_F0 → road / wide，2 Hz sample-and-hold、1.5 s、零状态、
  desire none、交通方向按城市），t0 那一步读出。等价性检查（先于批量）：新抽取器在 navtest 前 64 个 token 上给出的原生 plan 位姿与考试已存的
  `openpilot/navtest/<model>_none.npz` 最大差 < 1 cm。
- 读数：每个 head 的 PDMS / EPDMS（token bootstrap 95% CI）及 sub-score；配对 Δ（逐 token 分数，token bootstrap 10 000 次）：
  `ridge_late` − `ridge ego`、`cls_late` − `cls ego`、每个 head − 同模型原生 plan、每个 head − constant velocity。
- 判据（写死）：`temporal` 在 NAVSIM 上「有用」= 同族 late − ego 的 EPDMS 配对 Δ 的 CI 整体 > 0（两个模型分别判）；
  「达到 blind 文献线」= PDMS 点估计 ≥ Ego Status MLP 的 65.6；「达到 specialist 线」= PDMS ≥ TransFuser 84.0（文献不可配对，只比点估计）。
- 预期：ego head PDMS 在 60–70（与文献 Ego Status MLP 同量级）；`temporal` head 比同族 ego head 高几分（CI > 0）但低于 TransFuser；
  head 都远高于原生 plan（第 40 条 nuScenes 上重拟合的读出吸收了分布差）。

### G4：WOD test 提交包

- 按 `$DATA_DIR/runs/zeroshot-exam/wod-test/chain.sh`：下载完成 → 索引 → test 帧集（1505 帧）→ Cinque、Lebowski 按考试协议跑 → 写 + 校验提交包。
  GPU 3。只生成文件，**不上传**；提交需要用户批准。

### 追加预登记（2026-09-25 16:25，G1 出来之后、任何 G1b / I2 数字之前）

**G1b：有没有一个 NAVSIM 合法的输入适配能救回 openpilot。** G1 说明掉分来自 sample-and-hold 下的帧率错配；HUGSIM 考试（R1）里把 4 Hz 帧
当作 0.2 s context 步「压缩」地喂（`h4-dilate`），比 hold 好一个数量级。NAVSIM 只有 4 帧 2 Hz，对应的变体是
`nav2hz-dilate`：同样四帧（f−15、f−10、f−5、f），但当作 0.2 s 间隔喂（small / Cinque 在 20 Hz 时钟上 t = −0.6 … 0 共 13 步，每帧保持 4 步；
Lebowski 4 个 context 步）。时间被压缩 2.5 倍，预期速度被高估、纵向冲出，但不会有 hold 的「静止 + 跳变」。
判据：在 WOD 上 `nav2hz-dilate − nav2hz` 的配对 Δ 的 CI 整体 > 0（三个模型分别判），**且**至少回到 cv 之上，才在 NAVSIM navtest 上用它重跑三个 openpilot
（v1 / v2 两个 devkit），作为「WOD 上选出的适配」的次行，主表不替换。选择只看 WOD，不看任何 NAVSIM 分数。

**I2：continuation share 表**（reactivity 计划的 I2 行，并入本 todo）。同一组不看图像的基线在五个 benchmark 上占「顶分」的比例：
- 基线：cv（匀速直行）、ctrv（当前速度 + 当前角速度的圆弧，WOD 用 `waymo.baselines`，NAVSIM 由 AgentInput 最后两个位姿的 yaw 差 / 0.5 s 得角速度）、
  ego-only 学习 head（WOD：`ridge ego` / `cls ego K1024`；NAVSIM：G2 的两个 head；nuScenes：文献 Ego-MLP / AD-MLP 的 BEV-Planner 重算值；
  其余有就报，没有标「—」）。
- 顶分：该 benchmark 标准 split 上的最好公开条目（WOD：test 榜 RAP 8.04；NAVSIM：PDMS / EPDMS 取研究笔记表里已核实的最高行；
  nuScenes：L2 取 BEV-Planner 统一实现下最好的一行；HUGSIM：论文 Table 13 最好的 UniAD 0.299；B2D 开环：Bench2Drive 论文的开环 L2 最好行），
  另报 human / log 行作第二个分母。
- share：越高越好的指标 = 基线 / 顶分；越低越好的指标（L2）= 顶分 / 基线。一个数不代表能力，只回答「这个 benchmark 的分数里有多少不用看路就能拿到」。
- 只用 CPU；缺的 NAVSIM 基线（ctrv）用 devkit 跑，缺的 B2D 开环、HUGSIM 数字只收已有的，不新跑闭环。

## 预算

| 步骤 | 估计 | 依据 |
|---|---|---|
| G0 | CPU 分钟级 | 逐帧预测都在 |
| G1 | 479 × 3 模型 × 2 变体，每目标 ≤ 16 帧，GPU 约 5 min | 考试 0.46 s / 目标（101 帧） |
| G2 | index navtrain（CPU，约 10–20 min）+ ridge / cls 拟合（GPU 分钟级）+ 打分 3 arm × 3 考卷（CPU，每次 10–15 min） | 第 37 条实测 |
| G3 | navtrain 约 10 万 token × 2 模型的 `temporal` 抽取；Cinque 31 步 / token ≈ 0.15 s，6 进程分到 GPU 2 / 3 约 45 min | 第 37 条实测 0.15 s / token |
| G4 | 下载约 2.2 h（16 MB/s）+ openpilot 约 20 min | 下载器实测 |

G3 预计刚过 1 h：先在 1 000 个 token 上 profile（渲染 vs GPU），瓶颈在我们的代码里就先改，再批量。

## 结果

代码：`scripts/wod_openpilot_timeline.py`（G1 抽取）、`jevdrive/openloop_standing.py`（G0 / G1 / NAVSIM 读数）、
`scripts/navsim_zs_openpilot.py feat` 与 `jevdrive/navsim_heads.py`（G2 / G3）。小表在
[research/results/openpilot-openloop/](../research/results/openpilot-openloop/)。

### G1：NAVSIM 的时间轴在 WOD 上值多少（2026-09-25 15:33）

479 个 rater 帧，RFS cluster mean；Δ 是对同模型考试原样（`base`，10 s、10 Hz）的配对 frame-mean 差，frame bootstrap 10 000 次；
S = (base − nav2hz) / (base − cv)（frame mean），cv 为 7.10；「5 s 纵向偏差」是预测终点减 log 终点的 x 均值。

| 模型 | base | `ctx1.5`（10 Hz、1.5 s） | Δ [CI] | `nav2hz`（2 Hz、1.5 s） | Δ [CI] | S | 5 s 纵向偏差：base / ctx1.5 / nav2hz (m) |
|:--|--:|--:|:--|--:|:--|--:|:--|
| small | 7.640 | 7.528 | −0.07 [−0.15, +0.02] | 5.191 | **−2.57 [−2.80, −2.33]** | 3.9 | +0.7 / +0.9 / **+22.8** |
| Cinque | 8.005 | 7.878 | −0.11 [−0.20, −0.03] | 5.126 | **−2.88 [−3.11, −2.64]** | 3.2 | +0.9 / +0.2 / **+20.9** |
| Lebowski | 7.886 | 5.780 | **−2.00 [−2.26, −1.74]** | 4.942 | **−2.89 [−3.13, −2.65]** | 3.8 | +0.3 / −3.3 / **+19.4** |
| 参照：cv / 原地不动 | 7.103 / 5.383 | | | | | | |

**判定：S ≥ 0.5 那一格，而且远超**——NAVSIM 式的 2 Hz sample-and-hold 输入让三个模型在 WOD 上从 cv 之上 0.5–0.9 掉到 cv 之下 2 分，
和「原地不动」（5.38）同一水平；5 s 终点平均比 log 远 19–23 m。所以第 37 条 NAVSIM 表里 openpilot 的原生 plan 行主要是**输入协议的读数**，
不能当作模型的开环能力。预期里「nav2hz 掉到 cv 附近或以下、纵向偏差显著为正」成立，而且比预期更重。

两个时间轴因素分开看：small / Cinque 对 context 长度几乎不敏感（只看 1.5 s 的 10 Hz 帧只掉 0.07–0.11），**掉分几乎全来自帧率**
（nav2hz − ctx1.5 = −2.50 / −2.76）：sample-and-hold 下 t0 那一步模型看到的 t−0.2 s 帧其实是 0.5 s 前的，自车运动被放大 2.5 倍，
plan 随之冲出去。Lebowski 不同：它的 context 是 24 个 0.2 s 的 hidden state（4.8 s），只给 1.5 s 就已经掉 2.0 分（纵向反而偏短 3.3 m），
帧率再扣 0.9。也就是 Lebowski 比 Cinque 更依赖长时序记忆。
（1 个 rater 帧的 1.5 s 窗口里有一帧不在 slim shard 上，按「保持上一帧」处理，见偏离 1。）

## 偏离记录

1. **G1：slim shard 上缺帧时保持上一帧。** 预登记没写缺帧怎么办；考试的 runner 对缺帧是直接跳过。479 个目标里只有 1 个的 1.5 s 窗口缺帧，
   按「取时刻 ≤ 需要时刻的最近一帧」处理（sample-and-hold 本来就是这个语义）。对表的影响不超过 1/479。
2. **G3 等价性：Cinque 的 tapped 图与考试图不是逐位相同。** 预登记的判据是前 1000 个 navtest token 上原生位姿最大差 < 1 cm。
   Lebowski 逐位相同（0.0）；Cinque 的 ADE 差中位 1.1 cm、p99 4.1 cm、最大 6.6 cm（单点最大 18 cm）。原因是暴露 `temporal` 之后
   TensorRT 按 fp16 重新做了 fusion，和第 40 条 WOD 抽取时「中位数 ≤ 1 cm」是同一现象。处理：`temporal` 照用（它是 head 的输入，不是被比较的量）；
   NAVSIM 表里 openpilot 原生 plan 行一律用考试已存的位姿，不用 tapped 图的。
