# openpilot 开环地位：WOD-E2E 与 NAVSIM 上的并排对比，以及补齐公平比较缺的几格

状态: done（预登记写于任何新分数之前，2026-09-25 15:40；G3 结果 2026-09-26 00:20；G4 提交包等下载）
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

### G1b：dilate 救不回来（2026-09-25 16:25）

| 模型 | `nav2hz-dilate` RFS | 对 `nav2hz` 的 Δ [CI] | 对考试输入的 Δ | 5 s 纵向偏差 (m) |
|:--|--:|:--|:--|--:|
| small | 5.12 | −0.06 [−0.13, +0.02] | −2.62 | +24.1 |
| Cinque | 5.37 | **+0.23 [+0.11, +0.35]** | −2.64 | +22.3 |
| Lebowski | 5.28 | **+0.30 [+0.14, +0.45]** | −2.59 | +17.3 |

**判定：不采用。** 预登记要求两件事同时成立（对 `nav2hz` 的 CI > 0，并且回到 cv 7.10 之上）；Cinque、Lebowski 过了第一条，但三个模型都停在 5.1–5.4，
远在 cv 之下，所以不在 NAVSIM 上重跑。原因和 HUGSIM 的 4 Hz 不同：那里 dilate 只把 0.25 s 压成 0.2 s（1.25 倍），这里是 0.5 s 压成 0.2 s（2.5 倍），
速度被高估 2.5 倍，5 s 终点照样冲出 17–24 m。**只有 4 帧 2 Hz 的输入里，没有一种不改模型的喂法能让 openpilot 的原生 plan 读数有意义**；
NAVSIM 上能测 openpilot 的只剩「冻结特征 + 重拟合 head」（G3），head 可以学会把被放大的运动读回来。

### G2：NAVSIM 上我们的 ego head（2026-09-25 16:47）

navtrain 103 288 个 token（1192 个 log）上拟合，navtest 12 146 个 token 官方 devkit 打分；CI 为 token bootstrap。
`ridge ego` 的 λ 与 `cls ego` 的 λ 都落在网格下沿（1e-6 / 1e-7），即几乎不正则；32 维输入、10 万行，本来就不需要。
K = 1024 词表的 oracle ADE 0.29 m。

| 行 | PDMS [CI] | EPDMS [CI] | navhard EPDMS | NC / DAC / EP（EPDMS 口径） |
|:--|:--|:--|--:|:--|
| human（log） | 94.6 | 94.5 | — | 100 / 100 / 87.4 |
| **我们 `cls ego K1024`** | **68.4 [67.7, 69.1]** | **67.8 [67.1, 68.5]** | **13.6** | 93.0 / 80.2 / 86.5 |
| 我们 `ridge ego` | 62.7 [62.0, 63.5] | 64.2 [63.4, 65.0] | 13.3 | 89.7 / 79.0 / 85.6 |
| 文献 Ego Status MLP（arXiv 2406.15349） | 65.6 | — | — | |
| openpilot Cinque 原生（第 37 条） | 52.1 | 46.2 | 9.3 | 78.4 / 75.7 / 95.2 |
| Alpamayo 1.5 nav（第 37 条） | 44.3 | 43.2 | 10.8 | 76.8 / 70.6 / 84.8 |
| ctrv | 41.1 [40.4, 41.8] | 43.9 [43.1, 44.7] | 11.4 | 73.1 / 75.5 / 78.7 |
| constant velocity | 20.7 | 25.9 | 11.5 | 68.1 / 57.8 / 77.7 |

配对（EPDMS）：`cls ego` − cv +42.0 [41.1, 42.9]；`cls ego` − Alpamayo +24.7 [23.7, 25.6]；Cinque 原生 − cv +20.3 [19.4, 21.2]。

**读法**：两个 ego head 与文献 Ego Status MLP（65.6）同量级（`ridge` 62.7、`cls` 68.4），说明管线和口径对得上。
一个不看图像的线性分类头比所有 zero-shot 驾驶模型高 16–24 PDMS、21–25 EPDMS；在 navhard 上也是最高的一行（13.6，cv 11.5）。
NAVSIM 的分数里，「根据自车状态和 command 选一条合理的轨迹」这一部分就占了顶分的 70–75%（见 I2）。
预期「ego head 60–70」成立。

### I2：continuation share（2026-09-25 16:53）

`research/results/openpilot-openloop/continuation_share.csv`。share = 不看路的基线 ÷ 该 benchmark 已发表的顶分（L2 取倒数）。

| benchmark | 指标 | cv | ctrv | 最好的 ego-only 学习 head | 顶分（条目） | cv / ego head 的 share |
|:--|:--|--:|--:|--:|:--|:--|
| WOD-E2E val | RFS（下限 4） | 7.10 | 7.02 | 7.31（`cls ego`） | 8.04（RAP，test） | 0.88 / 0.91；扣掉下限 4 后 0.77 / 0.82 |
| NAVSIM navtest | PDMS | 20.7 | 41.1 | 68.4（`cls ego`） | 91.5（SimWAM） | 0.23 / 0.75 |
| NAVSIM navtest | EPDMS | 25.9 | 43.9 | 67.8 | 90.2（SimWAM） | 0.29 / 0.75 |
| nuScenes val | L2 均值 (m) | 0.83（GoStraight） | — | 0.35（Ego-MLP，文献） | 0.37（VAD-Base + ego） | 0.45 / **1.06** |
| Bench2Drive 开环 | L2 2 s (m) | — | — | 3.64（AD-MLP，文献） | 0.73（UniAD-Base） | — / 0.20 |
| Bench2Drive 闭环 | DS | — | — | 18.1（AD-MLP，文献） | 90.6（BLUE） | — / 0.20 |
| HUGSIM（64 场景） | HD-Score | 0.04（official 控制器）/ 0.29（fixed） | — | — | 0.299（UniAD，论文 Tab. 13） | 0.13 / —；fixed 控制器下 **0.98** |

读法：开环榜上，不看路能拿到的份额很高——nuScenes 上 ego-only 已经超过顶分（1.06），WOD 上 cv 拿到顶分的 77–88%，
NAVSIM 上学出来的 ego head 拿到 75%。闭环（Bench2Drive）只有 20%。HUGSIM 的 cv 份额取决于控制器：官方控制器下 0.13，
修正后的控制器下 cv 与论文最好的 UniAD 打平（同一批 64 个场景上 cv-fixed 0.29 对 LTF-fixed 0.28），这一格量的是控制器与场景长度，
不是驾驶能力（HUGSIM 考试另有判读）。B2D 开环 shadow 与 HUGSIM 的 ego head 没有数字，只收已有的（预登记）。

### G3：NAVSIM 上冻结 openpilot `temporal` + 薄 head（2026-09-26 00:20）

`temporal` 在 navtrain 103 288 + navtest 12 146 + navhard 5 912 个 token 上按考试输入抽取（Lebowski 位姿与考试逐位相同，Cinque 见偏离 2），
head 在 navtrain 上拟合（run：box 上 `navsim_zs/heads/20260925-232810`），官方 devkit 打分。Δ 是逐 token 配对差，10 000 次 token bootstrap。

| 行 | PDMS [CI] | EPDMS [CI] | navhard EPDMS | late − 同族 ego（EPDMS） | − 同模型原生（EPDMS） |
|:--|:--|:--|--:|:--|:--|
| **Cinque `temporal` + `cls_late`** | **77.9 [77.2, 78.5]** | **77.4 [76.8, 78.0]** | **19.8** | **+9.6 [+8.9, +10.2]** | **+31.2 [+30.4, +32.1]** |
| Lebowski `temporal` + `cls_late` | 77.3 [76.7, 77.9] | 76.7 [76.0, 77.3] | 17.5 | +8.8 [+8.2, +9.4] | +31.2 [+30.3, +32.0] |
| Cinque `temporal` + `ridge_late` | 73.5 [72.9, 74.2] | 73.9 [73.2, 74.5] | 16.8 | +9.7 [+8.9, +10.4] | +27.7 |
| Lebowski `temporal` + `ridge_late` | 72.4 [71.8, 73.1] | 72.9 [72.2, 73.6] | 17.0 | +8.7 [+7.9, +9.4] | +27.4 |
| 我们 `cls ego` / `ridge ego`（G2） | 68.4 / 62.7 | 67.8 / 64.2 | 13.6 / 13.3 | 0 | |
| openpilot 原生 Cinque / Lebowski | 52.1 / 50.9 | 46.2 / 45.5 | 9.3 / 10.2 | | 0 |
| *文献* Ego Status MLP / TransFuser / DiffusionDrive | 65.6 / 84.0 / 88.1 | — / 76.7 / 84.5 | — / 23.1 / 27.5 | | |

（navtest 上 (x, y) ADE 对 log：`ridge_late` 0.74 / 0.76 m、`cls_late` 0.87 m、`ridge ego` 1.04 m、原生 8.4 / 9.3 m。）

**判定（按预登记）**：
- `temporal` 在 NAVSIM 上**有用**：两个模型、两种 head 的 late − ego 配对 Δ 都是 +8.7 到 +9.7 EPDMS，CI 远离零。
- **达到 blind 文献线**（PDMS ≥ 65.6）：四个 head 全部达到。
- **没有达到 specialist 线**（PDMS ≥ 84.0）：最好的 77.9，差 6 分。EPDMS 上 77.4 与 TransFuser 的 76.7 同量级，但我们的 devkit 版本与文献不同
  （human 94.5 对 90.3），这一格只作量级。navhard 上 19.8，低于 TransFuser 23.1。
- 预期「head 高于同族 ego 几分、低于 TransFuser、远高于原生 plan」三条都成立，增益（+9–10）比预期的「几分」大。

读法：原生 plan 在 NAVSIM 上只比 cv 高 20 EPDMS，重拟合的线性读出比原生 plan 再高 31。G1 / G1b 已经说明原生 plan 差是 2 Hz 输入造成的；这里说明
同样被 2 Hz 输入扭曲的 `temporal` 里信息基本还在，只是要一个在该分布上拟合的读出。这是第 40 条 nuScenes 上「重拟合的读出吸收了分布差」在第三个数据集上的复现。

## 偏离记录

1. **G1：slim shard 上缺帧时保持上一帧。** 预登记没写缺帧怎么办；考试的 runner 对缺帧是直接跳过。479 个目标里只有 1 个的 1.5 s 窗口缺帧，
   按「取时刻 ≤ 需要时刻的最近一帧」处理（sample-and-hold 本来就是这个语义）。对表的影响不超过 1/479。
2. **G3 等价性：Cinque 的 tapped 图与考试图不是逐位相同。** 预登记的判据是前 1000 个 navtest token 上原生位姿最大差 < 1 cm。
   Lebowski 逐位相同（0.0）；Cinque 的 ADE 差中位 1.1 cm、p99 4.1 cm、最大 6.6 cm（单点最大 18 cm）。原因是暴露 `temporal` 之后
   TensorRT 按 fp16 重新做了 fusion，和第 40 条 WOD 抽取时「中位数 ≤ 1 cm」是同一现象。处理：`temporal` 照用（它是 head 的输入，不是被比较的量）；
   NAVSIM 表里 openpilot 原生 plan 行一律用考试已存的位姿，不用 tapped 图的。
3. **G3 抽取的调度偏离预算。** 第一次 8 个进程各自反序列化完整的 navtrain 索引（8 路相机 × 4 帧标定，每进程约 30 GB），加上 navsim_heads 里
   逐 token 读 npz 的写法，把 box 内存推近 cgroup 上限，被我停掉；改成只含 CAM_F0 与 ego 状态的 slim 索引（1.4 GB）后重跑。
   抽取按 7 个 shard 分到 GPU 2 / 3 / 4，墙钟约 65 min（预算约 45 min），瓶颈是每进程单核的 Python 步进 + TRT 调用（并发时 Cinque 的
   GPU ms/token 从单进程的 75 涨到 190），不是 GPU。结果不受影响（chunk 文件可续跑，合并时断言 token 完整）。抽取在 17:30 结束后我的 watcher
   没有触发，head 拟合与打分晚了约 6 h。
4. **G4 没做完。** WOD test 下载在共享网络上只有 2–5 MB/s，剩约 65 个 shard（ETA 约 15 h）；链式脚本（下载 → 索引 → openpilot → 提交包）挂在
   tmux `wodtest-chain`，用 GPU 3，只写文件，不上传。
