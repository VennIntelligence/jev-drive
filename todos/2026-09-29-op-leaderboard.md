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
- [ ] 重建 GIMM 缓存（navtest / navhard / navtrain 子集），`none` 复现 84.2 / 33.3
- [ ] navtrain 选臂（Cinque；Lebowski 若便宜）
- [ ] navtest / navhard 跑 A* 与 `none`
- [ ] WOD-E2E 可行性与 val RFS
- [ ] 榜单对照表；写进 decisions 第 37 条与 midterm inventory

## 结果

跑完再填。run root：box `$DATA_DIR/runs/op_lb/`。
