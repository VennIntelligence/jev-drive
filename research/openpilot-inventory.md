# openpilot 盘点：适配版本与榜单接口（2026-10-05）

CPU-only 盘点，只读代码、结果和 decisions（第 34–119 条），没有跑任何东西，数字全部抄自已有的 decision / results，没有复算。接 [openpilot-cross-board-synthesis.md](openpilot-cross-board-synthesis.md) 的「怎么改」第 3 步（先盘点再合并适配）。

两部分：第 1 部分是每个我们训过或配置过的 openpilot 变体，结尾是目标冲突矩阵；第 2 部分是每个榜给 openpilot 和我们的 controller / rule 的信息，逐项标 (a) / (b) / (c)，并列出当前最好的臂里所有 (b)(c) 用法。

分级口径（第 2 部分用）：

| 级 | 含义 |
|:--|:--|
| (a) | 实车上有：相机帧、本车里程计 / 车速 / IMU、标定、雷达、openpilot 自己的 head 输出、导航给的「下一个机动 + 距离」（粗） |
| (b) | benchmark API 给的，但比实车精：稠密 leaderboard 路线、从日志未来 / 录制路线反推的离散指令、地图量出的常数 |
| (c) | 特权：地图车道几何、actor 列表和 box、真值红绿灯状态、评分器多边形 / PDM 标签 |

## 第 1 部分：适配与配置过的 openpilot 版本

约定：`$DATA` = box 上 `$DATA_DIR`（`/root/autodl-tmp/ujs`）；基座权重在 `~/data/models/openpilot/`；Cinque 的 PyTorch 精确 port 是 `jevdrive/op_adapt.py`（`LModel` 在 `experiments/op_adapt_l/lib/op_adapt_l.py`）。所有微调都是**同一个 port 上只放开 stage 4 + off-policy plan pathway**（stage 1–3 冻结在线跑），checkpoint 格式沿用 op_adapt_l，serving 时导出 ONNX。

### 1.1 总表

| 变体 | 位置（box checkpoint / 代码） | 训练目标与数据 | 改动的层 | 评过的榜（headline，决策） | 已知退步 |
|:--|:--|:--|:--|:--|:--|
| **shipped Cinque**（comma v3 382M） | `~/data/models/openpilot/cinque.ort.onnx`（TRT）；`jevdrive/openpilot/model.py` | 无（zero-shot） | 无 | WOD RFS 8.005 [7.79, 8.22]（d34）；navtest 84.18 [83.77, 84.59]、navhard 33.33（d66、d72，补帧输入）；HUGSIM 64 HD 0.278、打转 10（d90）；B2D `drive` DS 67.82（d107） | 打转 10/64；navhard 一半以上 DAC 失败在评分器侧（d112）；路口 action 幅度只有所需 0.43–0.55（d118）；B2D 借道会违规（用户实车经验，d103） |
| Small 30M / Lebowski 877M | 同目录；`scripts/op_lb.py` 里臂 | 无（zero-shot） | 无 | WOD Lebowski 7.89 [7.66, 8.10]（比 Cinque 低 0.14 [0.01, 0.26]）；HUGSIM Lebowski 打转 11/64（d90）；navtest `turn@−0.5` +0.07 [−0.51, +0.65]，没进测试集（d66） | Lebowski 4.8 s 记忆在 1.5 s 历史下掉 2.0 分（standing 文档）；只作对照，不做 driver |
| LegacyOPModel（sc0816 / sc094） | `jevdrive/openpilot/model.py:242`，只被 `scripts/p5_openpilot.py` 用 | 无 | 无 | 没有任何榜单读数；行人 dose-response 考试（P5）用，需要真 20 Hz 帧 | 不是 driver 候选 |
| op_adapt_r1（exact port） | `experiments/op_adapt_r1/lib`；`jevdrive/op_adapt.py` | 无训练，只是 fp32 精确 port | 无 | 行人 AUC 增益 +0.076 nuScenes、+0.009 CARLA（d55）；所有微调的基底 | – |
| op_adapt_r2（S_jev） | `experiments/op_adapt_r2`（superseded） | CARLA 行人配对 + 规则打分的 13 种扰动 | stage 1 | 无 | stage 1 在 2000 步失败：漂移 0.397 m（线 0.10）、slow +11.7 pp，没出完整 batch（d67）。方法没被证伪，被 L 取代 |
| **op_adapt_L `main`（lmain / lmain1 / lmain2，seed 0/1/2）** | `$DATA/runs/op_adapt_L/runs/<arm>-s<seed>/`；B2D ONNX `$DATA/runs/op_adapt_L/b2d/onnx/<model>.onnx`；`experiments/op_adapt_l` | 真实 log 人类未来模仿：WOD train 的 start / stop / turn-onset 三个切片，contrast 行（stay / control / straight-intent）+ 蒸馏回原模型；WOD 路线 intent 经 token-embedding adapter 进模型；dw 3；nuScenes train 行作其余 | stage 4 解冻 + intent adapter（`s4ia_dw3`） | WOD val 捕获率 start 0.531→0.632、stop 0.252→0.559、turn 0.726→0.843（d78）；navtest 84.44 / 84.50 / 84.47（原 84.169）；WOD RFS +0.009（CI 跨 0）；B2D 对配速对照 −4.1 / −7.8 / +9.8，对 `drive` −5.7 / −6.2 / +3.9（d81） | stay 假起步 +2.7 pp（线 +2）；B2D 更快（均速 +27–39%）、碰撞 8 / 10 对 6，另撞行人 2；路口 turn_agree 0.487 对 0.486 没动；错 intent 会把 plan 推偏，增益低于无 intent。没测 HUGSIM / navhard / WOD test |
| L 消融：`lnoint`、`ldw10`、`lkd`、`stayheavy`、`only_*`、`dw03/1/3`、`tr_ad*`、`long` | 同上 | 单项改：无 intent / dw 10 / 原生 desire + adapter 同开 / stay 加权 / 单切片 / 蒸馏权重扫 / 只训 adapter / 8000 步 | 同上，各自不同 | dw 是增益与假起步的单旋钮：dw 0.3/1/3/10 start 增益 +0.318/+0.199/+0.101/+0.039，假起步 +25.4/+8.1/+2.7/+0.9 pp；B2D `ldw10` −1.6 [−7.1, +2.7] 对配速；`lkd` turn_agree 掉到 0.383 | 没有同时过 L1（增益）与 L2（假起步）的臂 |
| **layer-3 `pilot`** | `$DATA/runs/op_adapt_H/runs/pilot-s0/`；`experiments/op_adapt_h`（`h_train.py` ARMS） | 两类配对：H = 假偏航 / 重复帧 / 丢历史，教师 = 日志未来（保留未扰动的真实转动帧）；O = 朝向 / 横向偏移重投影，教师 = 回到日志路径的修正段；U 行日志模仿、D 行蒸馏回原模型。数据 navtrain 2400 token + WOD train 2780 帧 + CARLA 2588 帧，2500 步 | stage 4 + plan pathway，intent none | navtest +0.82 [+0.47, +1.16]（ctl +1.43）；navhard 33.55（对照 ctl 30.12）；开环探针 G 0.5–3 m/s 18.4→4.1 / 12.9→3.7 / 8.9→2.1°（nav / WOD / CARLA）；HUGSIM 10 个打转场景 6/10（基线 8/10）；全 64：打转 15（基线 10），非打转 HD +0.075 [+0.007, +0.151]（d98） | 闭环打转不减反增（全 64 新增 9 个）；WOD turn 起始捕获 −0.019、静止帧假起步 +1.3 pp |
| `pilot_ctl` | 同上 | 同 pilot 的帧，只做日志模仿（无配对） | 同上 | navtest +1.43、navhard **30.12**（对原生 33.6 掉 3.5）、HUGSIM 7/10 | 纯 navtrain 模仿伤 navhard：配对才让 navhard 不掉；navtest 涨分主要是微调本身 |
| `it_half` / `it_lowrate` | 同目录 | half = 配对份额减半；lowrate = 只用 3–8°/s 小偏航 + 蒸馏 3 | 同上 | it_lowrate：HUGSIM 打转 9/10、navtest +0.40 [+0.15, +0.65]、navhard 34.8 | 打转不降 |
| **`it_dw3`（s0）** | `$DATA/runs/op_adapt_H/runs/it_dw3-s0/`；serving ONNX `$DATA/runs/op_adapt_H/onnx/it_dw3-s0.onnx` | pilot + D 行蒸馏权重 3（控漂移） | 同上 | navtest 84.68（+0.50 [+0.30, +0.71]）；navhard 35.17（+1.84 [−0.35, +3.98]，ONNX 路径）；HUGSIM 64：打转 13、非打转 HD +0.036 [−0.013, +0.086]；WOD val RFS 7.979（−0.026 [−0.118, +0.062]）；B2D `drive` 19 路线 × 2 seed DS 64.48（−3.34 [−7.74, +0.47]）（d98、d101、d107） | **B2D 掉分**，由 37969 seed 2 和 27297 seed 3 带动；增益只在我们渲染、相机几何有偏差的榜上，WOD / CARLA 闭环不迁移（d107）；HUGSIM 打转 13（比基线多） |
| **`it_dw3` + selector** | 同一 ONNX + 推理规则（见 1.3） | 无新训练 | 无（推理期双 rollout） | navtest 84.70（+0.52 [+0.31, +0.73]）；navhard 35.76（+2.43 [+0.17, +4.58]）；HUGSIM 打转 4/64、非打转 HD +0.056 [+0.001, +0.115]；WOD 选择器几乎不触发（0–1/479）；B2D −5.18 [−10.20, −0.46]（选择器相对 it_dw3 −1.84 [−4.77, −0.03]）（d101、d107） | B2D 显著掉分；单 seed，HUGSIM 每格单次；navhard 下界只有 +0.17；「三榜最好」只在 navtest / navhard / HUGSIM 成立，「跨榜不掉」已撤回 |
| `ln1` / `ln_heavy` / `ln3` / `ln4` / `ln4p10` / `ln5` / `ln6`（round 2，s0） | `$DATA/runs/op_adapt_H/runs/<arm>-s0/`；配方 `h_train.py` ARMS；预登记 `plans/2026-10-04-launch-pairs-prereg.md` | 在 it_dw3 上加：起步行 L（静止前缀后 1–3 帧运动 + 0.3–3° 假偏航，WOD 2229 / CARLA 398 个真实起步 + 合成起步）、小偏航率行 S（0.3–2°/s）、长起步行 M（ln3+，4–8 帧、2–12°）；ln4 / ln4p10 / ln5 加配对一致性项（假历史下的 plan 拉向真实历史下的 plan，权重 3 / 10）；ln5 / ln6 为修正起步假偏航符号后的重训（ln1–ln4p10 有 25–41% 起步行实际在教「跟随」） | 同上 | 起步大信号增益 0.59–0.66 of shipped（线 ≤ 0.5，7 个臂全否），局部第 1 步增益 5.25→0.46–1.03；小偏航率 G 只有 ln4p10 过线（0.45/0.36/0.28）；navtest −0.20..+0.52（ln_heavy +0.52 [+0.27, +0.79]、ln6 +0.44 [+0.16, +0.70]、ln4p10 −0.20 [−0.59, +0.18]）；HUGSIM 64：ln1 打转 7、ln3 8（基线 10）（d106） | **打转此消彼长**（ln1 消 6 新增 3，ln3 消 8 新增 6）；增益挪到第 3–6 步，学到的是「对任何转动给 1–3° 平台响应」；ln4p10 漂移 0.16–0.25 m 超护栏、WOD 假起步 +11.6 pp。**navhard 没跑**；nav / navtest 用旧 rig（d104），部分结果、因 rig 复查中止 |
| Q1/Q2 图像指令：band + barrier | `$DATA/runs/op_img_cmd/ft/`（`img_ft_train.py`，**tracking 名为 Q2**，非 q3） | 3000 步、1500 个与评测不相交的路口帧，只训路线带 / 挡板两种画法 | stage 4 + plan pathway | uptake band 0.01→0.54、barrier 0.12→0.64（navtrain，385 帧 / 247 log）；留出画法迁移 ≤ +0.16（d99） | 无覆盖时 plan 漂移 0.29 m（护栏 0.10）；覆盖物从「挡路」变成「走」的信号，学到的是两种固定画法；闭环没跑 |
| **`q3SA-s0`**（天空箭头，无负样本；SB / SC 为超参变体） | `$DATA/runs/op_img_cmd/ft/runs/q3SA-s0/ckpt-final.pt`；闭环 ONNX `$DATA/runs/op_img_cmd/cl/onnx/q3SA-s0.onnx`；`img2_train.py` | 洋红天空箭头（固定在地平线以上，大小随导航距离；只用指令 + 导航距离、不画路面、不用地图）；目标取残差形式（原 plan + 指令支路与原模型所选支路的中心线差），速度剖面沿用原模型；蒸馏池约 6600 帧（nav / WOD / CARLA）；3000 步，batch 56，lam_d 10 / lam_c 1（SB 30/3、SC lr 5e-5 2000 步） | stage 4 + plan pathway | uptake nav 0.29 [0.23, 0.34]、CARLA 0.39 [0.37, 0.42]；correct 0.71 / 0.78；车道保持不变；路口无覆盖漂移 nav 0.13、CARLA 0.27 m；闭环 smoke turn_agree +0.086（0.440→0.526，线 +0.10 否），DS 80.2 对 drive 60.1，n = 8（d102） | 部分学成颜色信号：无信息圆盘使 4 s 横向移动 3.0 m（原 0.28）、指向不存在出口的箭头 7.5 m（原 0.29）；漂移超 0.10 m 护栏 |
| **`q3NA-s0`**（SA + 负样本；NB 为 lam_d 20 / lam_c 2） | `$DATA/runs/op_img_cmd/ft/runs/q3NA-s0/ckpt-final.pt`（`img3_train.py`）；ONNX 同上目录 | SA 的权重 + 负样本（教师 = 原模型 plan）：无信息圆盘、指向不存在出口的箭头、直路直行箭头 | 同上 | uptake nav 0.18 [0.13, 0.24] / CARLA 0.30 [0.26, 0.33]；圆盘干扰修好（0.36 m，原 0.28）；过相对护栏（nav 0.12 / CARLA 0.25 m，线 0.17 / 0.24）；闭环 smoke turn_agree +0.047、DS 69.6（d102） | **仍执行指向不存在出口的箭头**（CARLA +5.7 m，原 −0.2 m），dev 上没过，按预登记停；「不存在」按本车道连接道定义，未核对其他车道；navtrain uptake 0.29→0.18 |
| 绿线臂（GA / GB / GC，感知放置） | 同 ft 目录 | 天空箭头 + 模型自己车道线中线 + 固定半径圆弧 | 同上 | correct 0.82 / 0.84、吸收 0.33 / 0.39 | 漂移更大（CARLA 0.37 m）；单独一臂，不是主线 |
| **N-series（N0–N4）** | `$DATA/runs/skill_pack/raise/`；`experiments/skill_pack/archive/navsim_raise.py`、`skill_pack_n0/n1.py` | **不动 openpilot 权重**：冻结 Cinque 的 `temporal` 特征（补帧输入）+ 原生 plan 放进独立候选槽 + 22 个原生族槽 + 第二输入视图，MLP（2×2048，5 seed 集成）打分头；训练标签是 navtrain 上 PDM scorer 的子分，行数 2 万→6 万（N3）→8.4 万（N4） | 头在特征之上，openpilot 权重 0 改动 | navtest PDMS 84.17→N0 84.94→N1 87.32→N2 90.60→**N3 91.59**→N4 91.43（−0.17 [−0.42, +0.09]）；navhard 官方分 N1 / N2 30.24 / 31.29、N3 33.28、**N4 36.07**（均匀权重对原生 +1.83 [+0.22, +3.51]，一次读数）（d64–d75） | 增益几乎全在 EP（超过 human，metric 对准而非更像人）；navhard stage 2 的 NC / DDC 仍比原生差；7 次看 navtest；**换成适配权重的特征要重训头**（d94 selector 也没对 N4 重拟合）；不跨榜 |
| E6 Hydra 式打分头 / `temporal` + `ridge_late` / `cls_late` | `experiments/driving_backbones`、skill_pack | 冻结 `temporal` + 薄 head | 无 | WOD Cinque `cls_late` 比原生低 0.24（d40）；E6 在 NAVSIM 候选集上 84.18，与补帧原生互补（oracle 91.2） | 没追上原生 plan |

注：`it_dw3-s0` 的 serving 路径比训练端读数低 0.23（navhard 35.17 对 35.40），配对按 35.17 算。op_adapt_h round 2 的 nav / pnav / navtest 帧与 HUGSIM / CARLA 帧所在 rig 都在 d104 之前的相机高度上，所有 ln 与 it 系读数在 rig 修正后可能变；这条没有重跑。

### 1.2 它们各自动了什么（统一视图）

| 轴 | op_adapt_L | layer-3（it / ln） | q3 天空箭头 | N-series |
|:--|:--|:--|:--|:--|
| 可训练 | stage 4（+ intent adapter） | stage 4 + plan pathway | stage 4 + plan pathway | 只有头 |
| 蒸馏防漂移 | 对照帧 + dw 3 | D 行 dw 3，每个输出头 | D 行 22/56 + 负样本 N 行 | 不适用 |
| 监督信号 | 人类日志未来（WOD） | 日志未来 + 配对（假历史 → 真实未来；偏移 → 回路径） | 指令支路中心线（残差） | PDM 子分 |
| 数据域 | WOD train + nuScenes | navtrain + WOD + CARLA 三域 | navtrain + CARLA 路口（nav / WOD / CARLA 蒸馏池） | navtrain |
| 评读主阵地 | WOD 开环 + B2D | navtest / navhard / HUGSIM | navtrain / CARLA 开环 + B2D smoke | navtest / navhard |

三条微调线（L、layer-3、sky-arrow）**都从 shipped Cinque 出发，互相没合并过**：q3 从 shipped 训，不是从 it_dw3；L 没进 HUGSIM / navhard；layer-3 没进 B2D 以外的指令读数。也没有共同的护栏评测集。

### 1.3 不改权重的适配：规则与 selector

这些与权重变体正交，会叠在上面，所以单列，因为冲突矩阵里要用。

| 规则 | 位置 | 做什么 | 读数（决策） | 判定 |
|:--|:--|:--|:--|:--|
| selector（sel-rot0-r0.6，d94） | NAVSIM serving；HUGSIM `derot_sel`（`zs_agent.py`）；B2D `OP_SEL`（`op_arb_server.py`） | 低速（< 3 m/s）时再跑一条「历史帧转到当前朝向」的 rollout，规则那条 0–4 s 横向 plan 标准差 < 0.6 × 原生时才换；约 1–5% 的 token / 步触发 | navhard +0.98 [+0.03, +1.97]、navtest −0.09 [−0.18, −0.01]（原生）；HUGSIM 打转 10→3、非打转 HD −0.005 [−0.015, +0.003]；B2D 相对 it_dw3 −1.84 | 待定；0.6 在原生 navtrain 上拟合，事后构造形式，推理算力翻倍 |
| derot3（d96） | `zs_agent.py` `derot_below` | 低速且历史有偏航时重置并把最近 25 帧转到当前朝向重放 | HUGSIM 打转 10→3，非打转 HD −0.011 [−0.030, +0.008]（线 −0.02 过） | 部分成立；全量 rot0（d94）navtest −8.04 被否 |
| 控制器低速低通 / 死区（d113、d114） | `lib/lowspeed_ctrl.py` | 低速均匀低通 / 2–4° 死区 | 打转 10→1，不打转 HD −0.028；死区 10→6；B2D 补跑 DS −4.11 | 否，不采用 |
| launch_long（d117） | `lib/launch_long.py` | 起步纵向重定时到真车剖面（1.2 m/s²，≤ 4 m/s） | 低于 3 m/s 时间 6.4→3.9 s，打转 11（基线 10），非打转 HD +0.011 [−0.029, +0.048] | 否 |
| op_ctrl 横向（d118） | `lib/op_ctrl.py`；HUGSIM patch `op-ctrl.patch` | 横向走 action 曲率（action[0] / max(1, v)² → clip_curvature → lateralDelay 0.25 s），不再跟踪 plan；有效 c 0.013（iLQR 0.127） | HUGSIM 打转 10→0；非打转 HD −0.009 [−0.058, +0.042]（线不过）；B2D −3.66 [−9.02, +0.34]（但 B2D 的 `drive` 本来就用 action 曲率，只多了停车保持 / clip / 延迟） | 待定 |
| op_ctrl 纵向（d119） | 同上 + `op-ctrl-long.patch` | 纵向也走 action 加速度 → LongControl | 打转 1；非打转 HD −0.077 [−0.171, +0.014]；20 个卡死从未起步 | 否 |
| launch_stab | `lib/launch_stab.py`；预登记 `experiments/hugsim/plans/2026-10-04-launch-stab-prereg.md`（未提交） | 起步窗口内第二个 session 在起步朝向下重渲染，用它的横向 plan | 代码在（inert），离线屏筛脚本没跑，没有读数 | 进行中 |
| 纵向 ×1.06（WOD trick） | WOD 评分 | 4 s 纵向拉长 6%（序列两折交叉拟合选 s） | shipped 8.005→8.119–8.120；it_dw3 8.080 | trick，WOD 单列，未对 it_dw3 重拟合 |
| laneChange desire@−1.0 s（NAVSIM，d66） | NAVSIM serving | 左 / 右指令时 t0 前 1.0 s 打 laneChange desire 脉冲 | navtest +0.72 [+0.48, +0.95]（左右转 +2.14），navhard −0.28 | 待定，一个小而稳的正读数 |

### 1.4 目标冲突矩阵

行 / 列是目标（objective），不是某个 checkpoint：O1 起步 / 低速偏航抑制（layer-3 pilot / it / ln\*，derot / selector / 低通规则），O2 起停转捕获（op_adapt_L），O3 图像指令吸收（q3 SA / NA），O4 metric 对准的打分头（N3 / N4），O5 借道抑制（**尚未训**，能力缺口 2 的目标，实车上 openpilot 违规借道），O6 B2D 借道（vmerge 系 bypass：路线横移 + 路线几何接管横向），O7 action 曲率控制（op_ctrl，d118）。

格内：**冲突**（已有读数显示目标互伤）/ **张力**（机制上相拉，读数不全）/ 兼容 / ？（没测）。只填上三角。

| | O2 起停转捕获 | O3 图像指令吸收 | O4 打分头 | O5 借道抑制 | O6 B2D 借道 | O7 action 曲率 |
|:--|:--|:--|:--|:--|:--|:--|
| **O1 起步偏航抑制** | **张力**：L 的 start 模仿教「该走就走」，ln4p10 抑制把静止帧假起步抬到 +11.6 pp，L 本身 +2.7 pp，同方向的代价；起步纵向（d117）更快不压打转。两者没在同一模型上训过 | **张力**：都改同一批参数（stage 4 + plan pathway），都靠蒸馏防漂移；q3 从 shipped 训而非 it_dw3；箭头漂移 0.13–0.27 m 与 L3 的 plan 漂移护栏 0.15 m 同量级，叠加没试过 | **张力（版本）**：头在 shipped 特征上训，it_dw3 / ln 换权重要重训；selector 没对 N4 重拟合（d94） | ？ | **？ / 板间冲突**：it_dw3 在 B2D −3.34、selector −5.18；B2D 55.5% 非预热步在 < 3 m/s，selector 一直开着；掉分在 37969 / 27297，未查是否与借道有关 | **冗余而非冲突**：op_ctrl 横向让打转 10→0，selector / derot 的职责消失而代价（算力翻倍、B2D 掉分）还在；ln 的小偏航率抑制也冗余。action 头本身对 1–2° 偏向不响应 |
| **O2 起停转捕获** | – | **张力**：L 的 intent adapter（token embedding）与 q3 的天空箭头是两条并行的「路线进模型」通路；L 的 adapter 读真路线（置换 / 翻转后增益降），但 B2D turn_agree 没动；两者同开没试过 | **张力**：打分头吃 EP（开得更快）；L 在 B2D 已「更快更多碰撞」，叠加更快 | ？ | **张力**：L 的 B2D 均速 +27–39%，碰撞 8 / 10 对 6，bypass 起步在密集车流里本来就是碰撞源（d95） | ？ |
| **O3 图像指令吸收** | – | – | ？（头的特征会随 q3 权重变，没测） | **直接冲突（预期）**：天空箭头 NA 仍照着箭头执行不可行出口；借道抑制要模型「看见可借的道也不借」，指令吸收要「指令说借就借」；负样本是目前唯一的解（教师 = 原 plan） | **张力**：bypass 路径是路线横移，由路线几何接管横向（`lat_src = route`），不经过模型；但感知 bypass 靠 openpilot 车道线 / 路沿 head，q3 / L3 的蒸馏保留了每个输出头，这条目前**兼容** | **互补而不闭合**：action 曲率路口只转出所需 0.43–0.55（d118），天空箭头吸收只有 0.3–0.4 of 全切换，两者之积仍不够；B2D 路口靠路线几何兜底 |
| **O4 打分头** | – | – | – | ？ | 不适用（头只在 NAVSIM） | 不适用 |
| **O5 借道抑制** | – | – | – | – | **直接冲突（预期）**：抑制模型自己借道 vs 规则要它在 bypass 时借道。当前缓解：bypass 时横向由路线接管，模型借道意愿不进执行；但 bypass 的起步 / 切出碰撞（d95、d105）说明模型在侧向让行判断上弱，这正是 O5 想训的 | ？ |
| **O6 B2D 借道** | – | – | – | – | – | **兼容**：`drive` 本来就用 action 曲率（`lat_exec=curv`），bypass 时 lat_src = route，action 不起作用 |

三条补充：

1. **launch-yaw 抑制 vs navhard 早转**：navhard 上 PDM 参考 2 s 内朝向变化 ≥ 5° 的 token，stage 1 占 51%、stage 2 占 66%（d110）；全量去历史转动在这些 token 上 stage 1 −0.148 [−0.206, −0.090]，navtest −8；selector 在这个集合上 +0.001 [−0.010, +0.014]；it_dw3（训练时保留 U 行真实转动，开环探针 10°/s 下 G 降到 0.32）navhard +1.84，说明 it_dw3 **没有**伤早转。ln 系（10°/s 下 G 降到 0.23–0.56）**没跑 navhard**，是否伤早转未知，这是合并前必须补的一格。
2. **sky-arrow 吸收 vs 漂移**：吸收（≥ 0.4）和路口无覆盖漂移（≤ 0.10 m）在 9 个配置、5 次设计变更里没同时过；相对护栏下 NA 在 CARLA test 刚过（0.25 对 0.24 的线已超），吸收 0.30 刚到线，navtrain 掉到 0.18。该 trade-off 是 SA→NA→NB 单调的（NB 吸收 0.07 / 0.15，漂移 0.10 / 0.21）。
3. **目标冲突在板与板之间**：同一组权重对 NAVSIM / HUGSIM 有益、对 B2D / WOD 无益（d107），原因是适配增益依赖我们渲染 / 相机几何有偏差的画面（d104：NAVSIM 相机高 1.87 m，openpilot 约 1.22 m），**相机几何修好之前，不再往 it_dw3 上叠东西**（d107 的结论，本盘点沿用）。

---

## 第 2 部分：各榜接口

先列执行方式，再逐榜列输入。**同一个 openpilot 在不同榜上的「执行层」不同，是 d111 / d118 解释打转差异的关键**。

| 榜 | 执行层 | 有效 controller c |
|:--|:--|:--|
| NAVSIM navtest / navhard | 开环：提交 plan（经 retime + 杠杆臂变换），评分器自己用 LQR + 自行车模型回放 | 评分器侧 LQR，跟踪滞后占 navhard DAC 失败 21%（d112） |
| WOD-E2E | 开环：提交轨迹，RFS 打分 | 无 |
| HUGSIM | 4 Hz；plan 交 iLQR（PR#57 修朝向）跟踪；opctrl 变体：横向 action 曲率 + 纵向仍 iLQR | iLQR c ≈ 0.127–0.19；opctrl 0.013（d111、d118） |
| B2D | `drive` 臂：车道保持段 openpilot 横向 = **action 曲率经自行车模型 20 Hz**（`lat_exec=curv`）；指令区与分歧兜底段横向 = 路线几何经 P7 `pursuit` 控制器（`ctl_every` 4）；纵向 = min(设定速度 8 m/s 与曲率限速、lead-head IDM、行驶中 plan) + `hold: intent` 停车锁存 + `coast_v` 2.5 | 车道保持段与真车同形（action）；路口走路线 |

### 2.1 NAVSIM navtest（开环，12146 token）

| 信息 | 去向（openpilot / 我们的规则） | 来源 | 级 |
|:--|:--|:--|:--|
| 4 张 2 Hz CAM_F0 关键帧（1.5 s 历史）→ GIMM-VFI 补成 t0−0.2k 的 6 帧 | openpilot | 榜单相机 + 我们的补帧（合成历史；实车有真 20 Hz 帧，d116 真 10 Hz 只高 +0.51 [−0.51, +1.54]） | (a)，补帧是契约适配 |
| CAM_F0 外参标定 | 图像重投影 | 榜单 calibration | (a) |
| 自车位姿（2 Hz）、车速、加速度 | retime（r = v_实测 / v_plan(0)）、ego-motion warp、selector 的历史转动（只用 4 个 2 Hz 位姿） | 榜单 ego status | (a)：实车 CAN / 里程计 |
| 左 / 右 / 直 driving command | 仅 `lc@−1.0` desire 臂（navtest +0.72）；headline `none` 行与 selector 不用；**N-series 头吃它**（ego 向量最后 4 维 one-hot，由 `navsim_raise.py` 的 `ego[:, -4:]` 推断，未核对内部） | 榜单；NAVSIM 里由日志路线 / 未来反推（未在本仓库核实其生成方式） | (b)：离散、无距离，但从日志反推 |
| 左 / 右手交通 | openpilot 的 traffic 标志 | 地图名（`LHT_MAPS = sg-one-north`） | (a)：实车知道所在国家 |
| PDM 子分（NC / DAC / EP / TTC / C） | **只在 N-series 训练**：navtrain 上拿 PDM-Closed 评分器的逐候选分数当标签 | NAVSIM 评分器（地图多边形 + actor box） | (c)，训练期；部署时头只看冻结特征 |
| 评分器多边形、actor 箱 | 只做评分 | NAVSIM | (c)，评估期 |
| 地图 / lead / radar | **不进** openpilot | – | – |

执行：plan 经 retime 与杠杆臂变换（lever）转成后轴位姿提交，评分器 LQR 跟踪。N-series 提交的是头选出的候选，不是控制器。

### 2.2 navhard 两阶段

与 navtest 相同的接口，另外：

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| stage 2 合成帧（3DGS 渲染）+ 合成历史的自车转动 | openpilot | 榜单合成场景 pickles | (a)（相机帧）；但合成帧的历史转动与位姿一致（相位相关 0.95），模型把它外推成反向 yaw（d88、d94） |
| stage 2 的挪偏起点 | 评分器 | 榜单（约 20% 的起点没有可行弧，d112） | 评分器侧，非信息 |
| selector 的双 rollout | openpilot 的第二 session | 只用 4 个 2 Hz 位姿 | (a) |

N4 是 N-series 唯一在 navhard 高于原生的一臂（一次读数）；selector 没对 N4 重拟合。

### 2.3 WOD-E2E（开环，479 个 rater val 帧）

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| FRONT / FRONT_LEFT / FRONT_RIGHT 10 Hz 帧（16 帧，−1.5…0 s，每帧喂两次） | 重渲染成 openpilot 的 road + wide rig（纯转动 warp、最近邻、10 s 预热） | 榜单（`wod_openpilot_rigs.py`） | (a)；相机高 1.81 m 对 openpilot 约 1.22 m，缩尺与 NAVSIM 一样（d108） |
| `past_states`（16 步 4 Hz 位姿 / 速度 / 加速度） | retime、L 的训练特征 | 榜单 ego | (a) |
| `intent`（GO_STRAIGHT / LEFT / RIGHT） | 只进 op_adapt_L 的 intent adapter（L 的 turn-onset 增益 +0.117，错置 intent 时 +0.016–0.064） | 榜单 | (b)：离散、无距离 |
| 无地图、无 lead、无 radar | – | – | – |
| rater 轨迹（RFS 的 trust region） | 评分 | 榜单 | (c)，评估期 |

执行：直接提交轨迹；纵向 ×1.06 是 WOD 单列 trick（拟合出来的标量，不是信息）。

### 2.4 HUGSIM（闭环，64 场景，4 Hz）

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| 800×450 六目 3DGS 渲染（我们用前视，重投影到 openpilot road + wide；4 Hz 帧经 dilate 时钟合成 20 Hz 上下文） | openpilot | 仿真器渲染 | (a)，但渲染 + 起步 5 s 同一帧预热（非真车起步）是 harness 选择；渲染画面本身排除出区分因素（d109） |
| `ego_velo`、`ego_pos` / `ego_rot`（yaw） | retime、selector / derot 的里程计去旋转 | 仿真器 info | (a) |
| `command`：0 右 / 1 左 / 2 直 | openpilot desire（`DESIRE`），desire 对打转 / 转弯无帮助（d92） | 仿真器**从录制路线在最近的录制位姿取** | (b)：离散，无距离；日志反推 |
| 初速 1.0 m/s + 5 s 同一帧预热 | 起步方式 | 我们 harness | 非信息，但它放大起步偏向（d100） |
| `obj_boxes`、`semantic`、`depth` | **不进** openpilot | 仿真器 | (c)，未使用；lead 只来自 openpilot 自己的 head |
| 特权路线跟随 `engage_s` / `oracle_vmax` | 只做诊断（d90 第 4 点） | 路线 | (c)，诊断不计分 |
| HD-Score 的 HD map | 评分 | 榜单 | (c)，评估期 |

后处理（我们的规则，不加信息）：`forward_only`、`straight_stop`（plan 终点在 eps 内当停车，横向不动）。

执行：iLQR 跟踪 plan（PR#57 修朝向后 HD 0.278），**这与真车不同**：真车跟随 action 头曲率。opctrl 变体（d118）横向改 action，打转 10→0。

### 2.5 B2D（闭环，19 路线 × seed 的诊断集）

B2D 里「榜给什么」：leaderboard `set_global_plan` 给稠密世界坐标路线 + 每点 road option（`_dense_plan`，`scripts/b2d_zeroshot_agent.py:337`）；传感器 GNSS / IMU / speedometer / 相机 / 最多 4 路 radar。`Route` 类由它得到：路线 polyline、弧长、命令（LEFT / RIGHT / STRAIGHT / CHANGE\_\*）、自车沿路线的进度。

**共同底座 `drive`（`lib/op_arb_agent.py`，`lib/vlm_arb_agent.py` VLM_ARM=drive）**

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| openpilot road + wide 相机帧（rig 渲染，每 tick，5 s 预热） | openpilot | CARLA 传感器 | (a) |
| 自车位姿（GNSS + IMU + 车速 + 后轴偏移，PoseFilter） | 全部 | agent 自己滤波 | (a) |
| 路口前 20 m 到后 5 m 的 turn desire；换道指令 | openpilot desire | `Route.desire()`，稠密路线的命令 | (b)：实车导航是「机动 + 距离」，误差米级；稠密路线给精确的命令起止 |
| **指令区内横向 = 路线几何**（`DRIVE_ZONES`：LEFT / RIGHT 前 15 / 后 5 m，STRAIGHT 5 / 5，换道 5 / 10；以及 openpilot 路径与路线在 15 m 处偏离 > 1 m 的兜底） | 控制器（P7 pursuit 跟踪 `bpath`，`RouteAdapter` 的 rejoin path） | **稠密路线 `_dense_plan`**（`lib/op_arb_agent.py:467` 的 `lat_why = zone / div`） | **(b) 半特权**：路口里车是沿稠密路线的米级精确几何转的；实车导航只有道路级折线，米级误差；d118：83% 的朝向变化在指令区内由路线几何完成，action 曲率只转出 0.43–0.55 |
| 车道保持段横向 = action 头曲率（`lat_exec=curv`） | 控制器 | openpilot | (a)，同真车 |
| 纵向：设定速度 8 m/s、曲率限速、lead-head IDM、行驶中 plan、停车锁存 | 控制器 | openpilot head + 我们的常数 | (a)；设定速度类似 ACC set speed |
| lead head / lane lines / road edges / trigger heads | 规则 | openpilot 输出 | (a) |
| `Privileged` 几何 / `_ctx`（真值灯、actor、接触） | **drive 臂只写日志**（`privileged.jsonl`、`plans.jsonl`），不进控制 | CARLA world | (c) 仅评估 |
| **`resume: nored`**：停车锁存超时后的 resume 在真值红 / 黄灯 40 m 内被扣住 | 控制器 | `_ctx()` 的 `tl` / `tl_dist`（CARLA actor 的灯状态） | **(c) 特权**。**vlm_arb 链的 `drive` 臂用默认 `resume: timer`（`op_arb.sh` 无 DRIVE_ARGS），不含它**；但 op_adapt_L 的 B2D lane（`op_l_b2d_chain.py` R1，`drive` / `lmain*` / `lnoint` / `ldw10`）与 op_img_cmd 的闭环 smoke（`img_cl_lane.py`，`drive` 与 `imgsky`）都用 `nored`，所以 **d81 与 d102 的闭环读数含真值灯信号**，需在报告中标注 |

**vmerge / vmerge2 / vmerge3（`lib/vlm_arb_agent.py`，当前 B2D 分数最高的一族：vmerge2 DS 76.7）**

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| 红绿灯读数（Qwen3-VL-4B zero-shot，一次前向，约 1153 token） | R2 停车 / cusum 放行 | 广角 + road 相机 | (a) |
| 提问窗口：路线命令起点前 40 m 到后 6 m；**停车线 = 命令起点 − 3.3 m** | R1 路口降速、R2 目标、R3 停牌 | 稠密路线命令 + 一个用地图量出的常数（`route_junction_error.md`：稠密路线中位 +3.26 m，13 个路口里 9 个在 1 m 内） | **(b)**；vmj 开关把 R2 目标放在路口入口。早期 vred2 / vred3 臂直接读 CARLA 地图的 `get_stop_waypoints()` 取停车线 = (c) |
| openpilot trigger heads（灯 / 牌检测） | 触发提问 | openpilot | (a) |
| **bypass（vmerge / vmerge2 / vm3priv，pbyp3）**：障碍存在 / 起止 = 世界快照里静止 ≥ 5 s 的 actor box；相邻车道 / 方向 = CARLA map `get_left_lane / get_right_lane`；gap = 目标车道所有车辆的真值速度 | 路线横移路径 + 方向 / gap | `Privileged.geometry / adjacent` | **(c) 特权**；这是 vmerge2 +10.8 DS 对 drive 的几乎全部来源（障碍路线 +44.8）。**当前最高分臂 vmerge2 含它** |
| **bypass（vmerge3，`VM3_BYP=perc`）**：障碍 = openpilot lead head（p > 0.5，x < 40 m，v < 1 m/s）+ 15 m 先验 + 前向雷达静止回波；侧别 / 偏移 = openpilot 车道线 / 路沿（偏移 ×1.2 在 debug 路线标定）；gap = 两个后角雷达 + 前雷达 | 同上 | openpilot head + 雷达 | (a)（雷达需要实车有 BSM / 前雷达）；**但 bypass 的路径仍是「稠密路线按偏移横移」**：路线几何 (b)，与 > 15 m 离路线路口的判据同源。感知侧别 / 偏移与地图差约 0.15 m |
| 放行检查（vmerge3，`VM3_REL`）：前向雷达 TTC | R2 / R3 放行等待 | 雷达 | (a)；无效，不保留（d105） |
| 常数（byp_stand 5 s、gap 3 + 3v m、cusum 5 …） | 规则 | debug 路线上调，不在 19 条评测路线内 | 非信息 |
| `pbyp` / `pbyp2` / `pred` / `vbyp` / `vall` / `dtl` / `oshadow` | 特权上限臂（真值灯、actor、map lane、real GT 速度治理） | CARLA world / map | **(c)，上限诊断，不是 openpilot 能力**（d82 B2D 特权上限：红灯 +20 DS） |

**IMG lanes（op_img_cmd，`img_cl_lane.py`：`drive` 对 `skyNA` / `skySA`）**

| 信息 | 去向 | 来源 | 级 |
|:--|:--|:--|:--|
| 天空洋红箭头：指令 + 到路口距离 | 画进 openpilot 每一帧（历史帧各自用自己的距离重画） | `lib/op_arb_agent.py img_cmd()`：稠密路线的下一个 LEFT / RIGHT / STRAIGHT 命令及距离（不用地图） | **(b)**：实车导航也给「机动 + 距离」，但稠密路线给的距离是米级精确；方向上是实车可得信息 |
| desire | 关（`DESIRE=false`） | – | – |
| 其余（路线几何接管指令区、`resume: nored`） | 同 `drive` | 同上 | 路线几何 (b)；**`nored` (c)** |

注意：smoke 没有「微调模型不画箭头」的对照，DS 增益可能来自微调本身而非箭头（d102 第 7 点）。

### 2.6 当前最好的臂里的半特权 / 特权项

| 臂（对应最好读数） | (b) 半特权 | (c) 特权 |
|:--|:--|:--|
| B2D `vmerge2`（DS 76.7，对 drive +10.8 [+0.3, +22.2]） | 稠密路线驱动指令区横向（所有 drive 系共有）；desire 的路线命令；停车线 = 命令起点 − 3.3 m 的地图量出的常数；bypass 路径 = 稠密路线横移 | **bypass 的障碍 box / map lane / 目标车道车辆真值速度（pbyp3）**——增益主体来自它 |
| B2D `vm3priv`（DS 75.4） | 同上 | 同上（bypass 仍特权） |
| B2D `vmerge3`（DS 68.3，非特权 bypass） | 稠密路线驱动指令区横向；停车线常数；bypass 路径 = 稠密路线横移 | 无（唯一非特权 bypass 臂，障碍路线 +12.9 [+3.6, +24.3]，线 +20.7 否） |
| B2D `drive`（67.82；vlm_arb 链默认 timer） | 稠密路线驱动指令区横向；desire 的路线命令 | 无（`Privileged` 只写日志）。op_adapt_L / op_img_cmd 闭环 lane 的 `drive` 含 `resume: nored` 真值灯 |
| B2D `vred`（DS +5.0 [+0.7, +10.6]）及 vred2 / vred3 | 停车线 / 路口入口（vred2 / vred3 读 CARLA 地图 `get_stop_waypoints`，**(c)**；`vred` 用路口入口，来自地图 `is_junction`，**(c)**；vmerge 版本换成路线命令 (b)） | 同左（vred 的停车目标来自地图） |
| HUGSIM `it_dw3 + selector`（navhard / HUGSIM 最好行） | `command`（从录制路线取）；harness 的 1.0 m/s 初速与 5 s 预热 | 无（`obj_boxes` / depth / semantic 不进 openpilot） |
| HUGSIM opctrl 横向（0 打转） | 同上；车被建成纯延迟 0.25 s（RAV4 TSS2），无 EPS 动力学 | 无 |
| NAVSIM `N3 / N4`（navtest 91.59 / navhard 36.07） | driving command（ego 向量）；左右手交通来自地图名（a） | **PDM-Closed 评分器的逐候选子分作训练标签**（训练期）；部署时无；navtest 看了 7 次 |
| NAVSIM `it_dw3 + selector`（84.70 / 35.76） | 无（headline 行不用 command） | 无 |
| NAVSIM `lc@−1.0` desire 臂（+0.72） | driving command → desire | 无 |
| WOD `L main`（捕获率 + RFS +0.009） | `intent` 经 adapter | 无 |
| WOD ×1.06 trick | – | 无（拟合的标量，在 val 上交叉拟合） |

### 2.7 已知口径问题（写作时要带上）

1. **B2D 的路口转弯是路线几何 (b) 完成的，不是 openpilot 完成的**：83% 的朝向变化在指令区内由路线几何转（d118）；真车导航是道路级折线，米级误差，所以 B2D 的 turn 成绩是半特权上限。要换成模型自己的转弯，走指令通道（q3 天空箭头，吸收 0.3–0.4，闭环 smoke 未过线）。
2. **`resume: nored` 的特权灯信号**存在于 d81 的 L lane 和 d102 的 IMG smoke；vlm_arb 链的 `drive` 与 vmerge 系（含 d107 的 od2 臂）用默认 `timer`。我只核对了 `op_arb.sh` 默认臂与链脚本；vlm_arb 链 `drive` 的 DRIVE_ARGS 为空这一点由读代码确认，没有读运行产物的 cfg 文件核对。
3. **vmerge 文档字符串写「no CARLA map input to any control decision」只对灯 / 牌成立**；vmerge2 的 bypass 走 `Privileged.geometry`，README「Next」也写着要把 map 输入（路口入口、停车线）换成官方路线命令，两者并存说明 vmerge（路线命令窗口）与 vred2 / vred3（地图）是两个不同版本。
4. **HUGSIM 与真车的执行层不同**（iLQR 跟踪 plan vs action 曲率）：HUGSIM 的非 opctrl 读数里，打转有一部分是接口造成的（d118）；NAVSIM 评分器的 LQR 又是第三种。三榜的横向 c 不同，所以「同一个 openpilot」在三榜上的闭环 / 评分增益不可直接相比。
5. **相机几何**：NAVSIM（1.87 m）与 WOD（1.81 m）高于 openpilot 训练相机（约 1.22 m），模型把世界读成约 0.7；适配增益只在偏差大的渲染板上出现（d104、d107、d108）。
6. 本文档对 NAVSIM driving command 生成方式、N-series 是否在 N3 / N4 里吃 command 的结论来自读代码，没有读 navsim 源码核对。
