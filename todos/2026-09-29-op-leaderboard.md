# openpilot 上榜：NAVSIM 导航输入 + WOD-E2E 提交可行性（中期）

状态: running（预登记 2026-09-29，写于看任何新变体分数之前）
主题: ../research/openpilot-openloop-integration.md，decisions 第 37 条

## 目标

在固定输入契约（input contract：榜单给 agent 的输入，NAVSIM 是 4 张 2 Hz 关键帧 + ego status + driving command）下，
给**不改动的** openpilot（Cinque 为主，Lebowski 若便宜则附）拿到最强且诚实的榜单数字：
NAVSIM navtest PDMS（v1.1 devkit）、navhard two-stage EPDMS（v2 devkit，`extended_pdm_score_combined`），WOD-E2E val RFS 与 test 提交可行性；
并和公开榜单条目并列成表，统一标注「zero-shot，没在榜单 train split 上训练」。

## 为什么以前的导航读数不算数（设计依据，证据在 box `~/data/runs/op_lb/evidence/`）

1. **NAVSIM 的 `driving_command` 来自 route，不是未来轨迹**：OpenScene 沿 route centerline 取前方 20 m 的点，横向 ≥ 2 m 记 left / right
   （`DriveEngine/process_data/helpers/driving_command.py:40-100`）。t0 时的 left / right 基本是「正在转或 1–2 s 内开始转」：
   离转向起点（heading 变化 ≥ 10°）中位 1.5 s / 6 m，过去 1.5 s 里已转 ≥ 10° 的左转占 48–50%；96–99% 的转弯 command 车速低于 8.94 m/s。
   command 打开到真正转向中位领先 3.0 s / 14 m。
2. **openpilot 的 desire（意图输入，8 类 one-hot，只喂上升沿脉冲）**：已发布的 `desire_helper` 只发 laneChange（打灯 + 方向盘推 + 车速 ≥ 20 mph），
   从不发 turnLeft / turnRight；但模型训练过 turn 通道（comma1M 上 turnRight 让 4 s 横向 +1.1 m）。Cinque 的脉冲在 ONNX 里总共可见 6.6 s，Lebowski 5 s。
3. **六次读数里，和 openpilot desire 有关的三次（第 37 条 NAVSIM、第 40 条 WOD routing、第 57 条 CARLA）都不是在「补帧输入 + 脉冲贴着转向起点」下测的**：
   NAVSIM 那次是 hold 喂法（2 Hz 帧保持显示，后来证明契约本身坏了：52 对补帧 84 PDMS），约 90% 的脉冲落在零状态 rollout 的第一步（t = −1.5 s）；
   WOD 那次有 41–44% 的转弯 intent 帧是静止车；CARLA 那次脉冲在路口前约 20 m、3.5 s。脉冲大多还在模型的 desire 窗口里，所以「脉冲过期」解释不了失败。
   唯一的正例（HUGSIM 0383，第 57 条之外的预登记 R2）里脉冲是在车刚开始转、9 m/s 时打的，之后 1.5 s Cinque 有 desire 转 −45°、无 desire 只转 −6°。
   **猜测（本实验要检验的）**：模型学到的是「刚开始转时给脉冲、顺着弯开」，不是「提前几秒发起转弯」；所以脉冲时刻是关键变量。
4. openpilot 的 plan 头只有一个 Gaussian（mu + std，没有多假设），「按 command 在多条假设里挑」这条零训练路不存在，不登记。
5. 其余两次（Alpamayo nav 文本、`cls_late` 的 command）与 openpilot desire 无关，不影响这里的设计。

## Setup

- 输入：GIMM-VFI 补 t0 − 0.2k 的 6 帧（integration doc 第 7 节默认接法），不 retime，杠杆臂变换 + 线性重采样；Lebowski 另加 3.3 s warp 预热。
- 调参集：**navtrain** 按 seed 0 每个 command 类（left / straight / right）各抽 1000 个 token（token 表 `research/results/op-lb/`），v1.1 PDMS。
  **只在这里选臂；navtest / navhard 只跑一次选定臂和 `none`，不再改。**
- 臂（desire 在 t0 的 command 为 left / right 时才给，否则 none；desire 从时刻 T 起一直置位，所以上升沿脉冲落在 T）：
  | 臂 | 说明 |
  |:--|:--|
  | `none` | 当前默认（navtest 84.2 / navhard 33.3 的复现） |
  | `turn@−1.5`、`turn@−1.0`、`turn@−0.5`、`turn@0` | turnLeft / turnRight，脉冲在 t0 前 1.5 / 1.0 / 0.5 / 0 s；`turn@−1.5` 相当于第 37 条旧读数的时刻，但换成补帧输入 |
  | `turn@onset` | 脉冲在历史里第一个「按 ego 位姿插值的横摆角速度朝 command 方向 > 5°/s」的 20 Hz 步，没有则在 t0；只用契约内的过去位姿，模拟「开始转时打灯」 |
  | `lc@−1.5`、`lc@−1.0`、`lc@−0.5`、`lc@0` | laneChangeLeft / Right（真车软件唯一会发的 desire），同样四个时刻 |
- 模型：Cinque（主），Lebowski（附，同一套臂在 navtrain 上单独选）。

## 选臂与判读规则（写死）

1. 在 navtrain 子集上，按 PDMS 选出最好的非 `none` 臂 A*（每个模型各选一次）。
2. **A* 进入 navtest / navhard 的条件**：navtrain 上 A* − `none` 的配对 PDMS 差（token bootstrap 95% CI）下界 > 0；否则报告「导航通道在补帧输入下仍无增益」，
   navtest / navhard 只报 `none`，这就是 headline。
3. 若 A* 进入：navtest（12 146）与 navhard two-stage（5 912）各跑一次 A* 与 `none`，**无论结果如何都以 A* 为「加导航」行报告，`none` 行并列**；不在测试集上再挑臂。
4. navtest 上「导航有用」**成立**：left + right 合并的配对 ΔPDMS CI 下界 > 0，且 straight 的 Δ ≥ −0.5；左、右各自单报。
   否则「不成立」（包括转弯涨、直行掉超过 0.5 的情况，写成「换来的」）。
5. navhard：只报 EPDMS combined 与 stage 1 / 2，与 `none` 的差；navhard 的聚合分没有逐 token 配对 CI，只报点估计，不单独判「成立」。
6. 预期（写在看分数之前）：`turn@0` 或 `turn@onset` 最好；`turn@−1.5` 与第 37 条同号（≤ 0）；`lc@*` 在转弯上无益。

## 另两项

- **WOD-E2E**：查 test 提交是否仍开放（规则、截止、格式、配额），在 val 上按 test 契约给的历史量 RFS；可行且便宜就把 Cinque 的 test 提交文件准备好，**不提交**，由用户决定。
  笔记在 [tmp/2026-09-29-leaderboard-wod.md](../tmp/2026-09-29-leaderboard-wod.md)。
- **榜单对照表**：NAVSIM v1 navtest、v2 navhard、WOD-E2E test 的公开条目（带出处），我们的行标「zero-shot / 不在 benchmark train split 上训练 / 公开预训练权重 + GIMM-VFI 补帧」。
  navhard 的 devkit 版本与文献不同（human 94.5 对 90.3），只并列、不写「超过」。

## 资源

GPU 0–4（Cosmos 全量在跑）与 GPU 6，每卡 ≤ 12 GB、nice 19，卡上显存 > 78 GB 就退让；CPU 48–71，调度表行 `op-lb`。
估时：补帧 navtest ~7.4 GPU·h、navhard ~3.6、navtrain 子集 ~1.8，分到 6 张卡；openpilot 推理与 CPU 打分每臂每 split < 1 h。

## 步骤

- [x] 证据：command 语义、desire 语义、旧读数失败原因、MHP（2026-09-29）
- [x] 重建 GIMM 缓存（navtest / navhard / navtrain 子集），`none` 复现 84.2 / 33.3（2026-09-29，84.18 / 33.33）
- [x] navtrain 选臂（Cinque 与 Lebowski 各选一次）
- [x] navtest / navhard 跑 A* 与 `none`（只跑了 Cinque `lc@-1.0`；Lebowski 没有臂过线）
- [x] WOD-E2E 可行性与 val RFS（见笔记与下面的提交记录）
- [ ] 榜单对照表；写进 decisions 第 37 条与 midterm inventory

## 结果

run root：box `$DATA_DIR/runs/op_lb/`；小结果在 [research/results/op-lb/](../research/results/op-lb/)。结论见 [decisions 第 66 条](../research/decisions.md)。选臂规则原样执行，没有改动；navtest / navhard 各只跑了一次。

### 1. `none` 复现与缓存等价

| | 本次 | 参照（op_interp） |
|:--|--:|--:|
| navtest PDMS，Cinque `none`，12 146 token | 84.18 [83.77, 84.59] | 84.2 |
| navhard two-stage EPDMS combined，5 912 token | 33.33（stage 1 71.70 / stage 2 46.90） | 33.3 |

新缓存对 op_interp 旧缓存的等价检查（[equivalence_navtest.json](../research/results/op-lb/equivalence_navtest.json)、[equivalence_navhard.json](../research/results/op-lb/equivalence_navhard.json)）：
逐点位置差均值 0.019 / 0.020 m，token 内最大值均值 0.10 m，> 1 m 的 token 只有 1 / 0 个。这个量级就是 GIMM 在同卡上重跑两次的自身非确定性（0.16% 字节 ±1），不是 bug。

### 2. GIMM 补帧的前后数字与瓶颈

| 项 | 数 |
|:--|:--|
| 缓存格式 | 每 token 只存 6 张补帧（uint8）；navtest gimm 28.7 GB、navhard 14.0 GB、navtrain 子集 gimm 6.6 GB + warp 6.6 GB |
| 吞吐 | GPU 6 单独 0.65–1.0 s/token；被 Cosmos 占用的卡 1.2–2.0 s/token（GPU 上有别的任务时） |
| 瓶颈 | GIMM fp32 的 GPU 计算。TF32 / bf16 会改帧、破坏复现，所以没有不改数值的提速，只能多卡分片（6 张卡；chunk 32，峰值 11.8 GB） |
| navtrain 3000 token metric cache | 15.5 min；navtest / navhard 已有 cache 完整 |
| 打分 | 每个 navtrain 臂约 3 min；navtest 约 3 min，navhard 约 13 min（lane STATUS 时间戳） |

### 3. navtrain 选臂（3000 token，每 command 1000；gimm 行，PDMS）

Cinque `none` 82.12，Lebowski `none` 81.95。直行 token 不给 desire，所以直行分数与 `none` 逐位相同。Δ 为配对差，token bootstrap 95% CI（B = 2000）。

| 臂 | Cinque PDMS | Cinque Δ [CI] | Lebowski PDMS | Lebowski Δ [CI] |
|:--|--:|:--|--:|:--|
| `turn@−1.5` | 80.68 | −1.44 [−2.20, −0.67] | 81.25 | −0.70 [−1.28, −0.14] |
| `turn@−1.0` | 80.70 | −1.42 [−2.29, −0.62] | 81.60 | −0.35 [−0.95, +0.23] |
| `turn@−0.5` | 80.94 | −1.18 [−2.02, −0.39] | **82.02（A*）** | +0.07 [−0.51, +0.65] |
| `turn@0` | 81.45 | −0.67 [−1.51, +0.11] | 82.01 | +0.06 [−0.50, +0.62] |
| `turn@onset` | 80.82 | −1.30 [−2.08, −0.53] | 81.52 | −0.43 [−1.03, +0.15] |
| `lc@−1.5` | 82.82 | +0.70 [+0.07, +1.30] | 81.83 | −0.12 [−0.67, +0.43] |
| `lc@−1.0` | **82.95（A*）** | **+0.83 [+0.20, +1.45]** | 81.61 | −0.34 [−0.94, +0.27] |
| `lc@−0.5` | 82.55 | +0.43 [−0.27, +1.12] | 81.34 | −0.61 [−1.31, +0.04] |
| `lc@0` | 82.06 | −0.06 [−0.77, +0.64] | 81.11 | −0.84 [−1.53, −0.18] |

规则的判定：**Cinque** A* = `lc@−1.0`，CI 下界 +0.20 > 0，进测试集；**Lebowski** A* = `turn@−0.5`，CI 下界 −0.51，不进，headline 是 `none`，测试集不跑任何 Lebowski 臂。
Cinque A* 按 command 拆（navtrain）：左 −0.07 [−1.33, +1.26]，右 +2.57 [+1.20, +3.98]，增益全在右转。全表见 [navtrain_paired.csv](../research/results/op-lb/navtrain_paired.csv)。

### 4. navtest / navhard 测试集（Cinque `lc@−1.0` 对 `none`，各跑一次）

navtest（12 146 token，v1 PDMS）：

| 组 | n | `none` | `lc@−1.0` | 配对 Δ [95% CI] |
|:--|--:|--:|--:|:--|
| 全部 | 12 146 | 84.18 | **84.90** | +0.72 [+0.48, +0.95] |
| left + right | 4 076 | 76.51 | 78.65 | **+2.14 [+1.40, +2.86]** |
| straight | 8 070 | 88.05 | 88.05 | 0.00 |
| 左 | 2 501 | 77.80 | 80.69 | +2.88 [+2.00, +3.82] |
| 右 | 1 575 | 74.46 | 75.41 | +0.95 [−0.14, +2.04] |

按预登记判：left + right 合并的 Δ CI 下界 +1.40 > 0，straight Δ = 0.00 ≥ −0.5，**「导航有用」成立**（数据在 [navtest_paired.csv](../research/results/op-lb/navtest_paired.csv)）。

navhard two-stage（5 912 token，v2 EPDMS，只有点估计）：

| | `none` | `lc@−1.0` | 差 |
|:--|--:|--:|--:|
| combined | 33.33 | 33.05 | −0.28 |
| stage 1（真实） | 71.70 | 72.20 | +0.50 |
| stage 2（3DGS 合成） | 46.90 | 46.19 | −0.71 |

navhard 上没有增益（差在点估计的量级，不单独判）。

### 5. 对预期的读法与限定

登记时的预期是「`turn@0` 或 `turn@onset` 最好、`lc@*` 在转弯上无益」，结果**相反**：Cinque 上所有 `turn@*` 都不高于 `none`（`turn@0` −0.67 是最好的一个），有增益的是 `lc@−1.5 / −1.0`。`turn@−1.5` 与第 37 条同号（≤ 0）的预期成立。
限定：(1) 增益 +0.7 PDMS，量级小，navhard 不复现；(2) A* 在 9 个臂里按 navtrain 分数挑出，navtest 是独立的一次，所以 navtest 的 +0.72 不受选择偏差影响，但机制没查（为什么 laneChange desire 帮到转弯，而 turn desire 不帮）；(3) 「加导航」行 84.90 是补帧输入 + 提前 1.0 s 打 laneChange 脉冲，仍是 zero-shot，没有任何参数拟合；(4) Lebowski 没有测试集数字，headline 用 navtrain 上的 `none` 81.95。
运行事故：navtrain 打分里 `lc_0` 那次 Ray 打分空转了 1 h 52 min（所有 worker idle），按 PID 杀掉后单独重跑，分数正常（Cinque 82.06）。

## WOD-E2E test 提交记录

- 2026-09-29：用户手动上传了三次。前两次：原生 Cinque 的根目录 `.bin` 包，以及加了 val 两折均选出的纵向 ×1.06、按官方教程封装为 `MySubmission/part0` 的包，均报 `No shards found in submission_key`；Waymo 页面写明失败提交不扣配额。随后在 Waymo 工程师的 [#936 回复](https://github.com/waymo-research/waymo-open-dataset/issues/936)及已有成绩的 [OpenEMMA 公开代码](https://github.com/hansungkim98122/OpenEMMA-for-Waymo-E2E/blob/main/waymo_submission.ipynb)中确认，服务器实际要求分片名 `mysubmission.binproto-00000-of-00001`，与官方教程的 `part0` 不一致。第三次按该命名重包，1,505 条 test 轨迹及 protobuf 元数据与第二包相同；用户手动提交并成功出分：**RFS 7.921477、ADE@3s 1.2368475 m、ADE@5s 2.7405558 m**。[完整记录](../tmp/2026-09-29-wod-e2e-cinque-test-result.md)。读取时结果页仍显示 Publish 按钮，公开状态待确认。
- 名次：按 2026-09-29 榜快照（154 条提交、115 个方法），按方法取最好成绩排**第 19**（逐条第 23）；在 Poutine 7.986 之下、Poutine-Base 7.909 之上。可核实的 zero-shot 行此前最高是 LightEMMA 6.52。最弱场景 Spotlight 6.93，最强 Construction 8.49。
