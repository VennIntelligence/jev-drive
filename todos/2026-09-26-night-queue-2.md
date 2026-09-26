# 夜间队列 2（2026-09-26 晚）：第三层量具、榜单 head × 反应、快通道去 lift、backbone 补行

状态: registered（写于任何数字之前；执行员按节领活，每节结果写回本节末尾，结论回填 decisions）

上一轮的四条真实数据任务（G0 / G1 / G3 / R40，`2026-09-26-real-data-transfer.md`）由中央调度员派出，正在卡上跑；本队列只吃剩余容量，
每张卡 96 GB，现占 23–32 GB。**开工前 `nvidia-smi` 选最空的卡，run 目录里记下卡号；不碰别人的进程。**

依据的三份调研（都在 `tmp/`，2026-09-26）：`behavior-layer-survey.md`（第三层量具与 P6 草案）、`nohack-mechanisms.md`（no-hack 高分方法的机制）、
`ablation-matrix.md`（矩阵空格与口径不一致）。要点：

- 第三层（判断后的行为：bypass、negotiation、recovery）我们目前没有量具；公开开环榜都不按行为模式计分。P5 v1 的路线里横向避让类几乎没出题。
- PDM-Lite 会绕，但靠 `active_scenarios` 的特权登记触发（50 m 内平移路线，对向 gap 用真值速度）；BehaviorAgent 只停不绕，static prop 检测不到。
- Hydra 打分头（NAVSIM 84.2）从没上过 P5 / I3；pair-Δ head 上过 WOD / NAVSIM 且有害（第 44 条）。榜单最优 head 保不保反应，还没量。
- ego-only 在 P5 上的 6–8% 是 τ = 0 的标签伪影，不是反应；I3 上是 0。
- V-JEPA 2 权重在 box 上但没上过 P5；DINOv3 没权重（申请被拒）；SigLIP2、DINOv2、openpilot small 在 P5 上没数。
- 第 45 条：行人召回缺口在 20–40 m 的 flat-ground 放置（图像上检出 46–61%，BEV 只剩 12–22%）；E5 的检测 embedding 走的正是这条 lift。

## 通用规则

1. 判据写在数字之前；看了数再想改，只能新登记一行「事后」，不改原判格。
2. 不跑 CARLA 闭环。P6 是开环配对：录像 + 事后评测。
3. 估时后再跑；超过估计 2 倍先停下写日志。> 1 min 的活进 tmux `jev`，写 `log.txt` / `events.jsonl`。
4. 主表 3 seed（seed 定义沿用 `2026-09-26-overnight-queue.md` [SEEDS]）；单 seed 的数只能写「同一水平」，不能写「更好」。
5. 口径：RFS 报 cluster mean（frame mean 只在与旧表对照时并列）；P5 翻转率沿用 `p5_exam` 的 τ = 考生自身 null p95；I3 judge 不改。
6. 结果小文件到 `research/results/night2/<节>/`，图到 `research/figs/`，commit + push；结论回填 decisions（新条或就地修正，标**待定**）。

## N1. P6 v0：行为模式考试的数据与 expert 统计（GPU，约 4 h × 2 卡）

**目标**：造出第三层的第一版考卷，并先量 expert 自己在上面怎么做；本节不读任何考生。

**配对**（按 `behavior-layer-survey.md` §3.1，只用 220 集路线，expert = PDM-Lite，3 个 TM seed）：

| 世界 | 障碍 | 对向车流 | 预期 expert 动作 |
|:--|:--|:--|:--|
| x₀₀ | 藏 actor **并删掉 `active_scenarios` 里的登记** | 删 | keep |
| x₁₀ | 在 | 删 | bypass |
| x₁₁（只 TwoWays 类） | 在 | 在（对向车道确定车流，spawn 表写死，TM seed 固定） | 等 gap 再绕 |
| x₀₁（只 TwoWays 类） | 删 | 在 | keep（对向车本身不应引起反应） |

- scenario：1W = Accident、ConstructionObstacle、ParkedObstacle、HazardAtSideLane；2W = 四类的 TwoWays 版 + VehicleOpensDoorTwoWays；另加 InvadingTurn、YieldToEmergencyVehicle（救护车 {有, 无}）。
  unprotected turn 六类作第二期，本节不做。
- null：天气 null（同 P5）；**放置 null**（障碍移到路肩不占车道，expert 应 keep）；**镜像题**（相邻车道不可用：实线 + 护栏或对向车流不断，expert 应停）。
- 记录：P5 v1 的 recorder（`scripts/p5_pair_agent.py`，`driver: pdm_lite`，5 Hz Waymo 标定相机，`cams/<cam>/<frame>.jpg`），加 expert 的未来轨迹、GT actor、
  `active_scenarios` 状态、对向车 GT 速度；TFv6 考生要的传感器（LiDAR）若 P5 v1 已录则照录。
- 规模：约 580 个世界（1W 120、2W 300、InvadingTurn + Emergency 60、null 约 100）；P5 v1 单 run 中位 276 s、11 实例并行 → 约 4 h。两张卡各 6 个 CARLA server。

**开工前的 smoke（必过，10 个世界）**：x₀₀ 里 PDM-Lite 的横向偏移 |Δ_lat(3 s)| 相对 x₁₀ 同帧 < 0.3 m 的比例 ≥ 95%；不过就是删登记失败，先修再批量。
另一条 smoke：对向车流在 x₀₁ / x₁₁ 里确实出现在 expert 变道窗口内（每个 2W 世界至少 1 辆在 50 m 内）。

**expert 统计（本节的交付，写死）**：

| 量 | 读法 |
|:--|:--|
| 每类 scenario 的 expert 模式分布（keep / stop / bypass_L / bypass_R / wait-then-bypass），按 §2.1 的判定规则 | 有没有出题：bypass 类里 bypass 比例 ≥ 70% 才算这类可用 |
| 横向分叉时刻 t_div 相对障碍首次可见 t_vis（同 P5 的 t_vis 定义） | t_div ≥ t_vis 的对才计入主读数；PDM-Lite 靠登记，可能提前，提前的对单列 |
| x₁₁ − x₁₀ 的纵向 Δv 与横向起动延迟 | negotiation 有没有被造出来：x₁₁ 里 wait 比例 ≥ 50% |
| 放置 null 与镜像题上 expert 的模式 | 放置 null keep ≥ 90%；镜像题 stop ≥ 80%；不过就修生成器 |

**另两条 CPU 准备**（同一执行员）：(i) box 上数 `cls_late` K = 1024 词表里 bypass 形状的 anchor 数与 oracle 覆盖率（判定同 §2.1，先写阈值：3 s 处横向 ≥ 1.0 m
且 5 s 内回到 ±0.5 m）；(ii) 220 集 xml 里上述 scenario 各多少条路线。两个数进本节结果。

**判读考生**（登记在此，但**本节不执行**，等 N1 数据齐、expert 统计过门后另开一节）：回归类考生读 Δ_lat(2 s / 3 s) 的横向翻转率 + 4 × 4 模式混淆；
`cls_late` 类读词表模式质量 Δm_bypass(x₁₀ − x₀₀)、Δm_stop(x₁₁ − x₁₀)；null 地板 = 天气 null p95、放置 null 的 bypass 率、x₀₁ − x₀₀。
考生：openpilot `ridge_late` / `cls_late`（Cinque、Lebowski）、M-C、E5 student、TFv6 waypoint（若传感器已录）、Qwen `L18_last`、ego-only。

- 2026-09-26 09:52 CST [A] 开工，分步估时（写于任何 N1 / N2 数字之前）：
  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | 1 | 生成器（x₀₀ / x₁₀ / x₁₁ / x₀₁、放置 null、镜像题、天气 null 的 variant XML）+ hook（删登记、对向车流开关、障碍平移）+ recorder 补记（prop、登记状态）+ 分析模块 | 2 h（到约 12:00） | Mac + box CPU |
  | 2 | smoke 10 个世界 = profiling pass（逐 tick 分项、每 run 墙钟），两条 smoke 判据 | 0.5 h | 1 卡 |
  | 3 | 批量约 605 个世界（todo 的 580 + 镜像题 25），3 卡 × ≤ 6 server（受 pids.max 限制） | 约 3 h（P5 v1 PDM-Lite 实测每 run 均值 244 s；绕行类预计 5–6 min / run） | GPU 0–2，约 9 卡·h |
  | 4 | 建 index + expert 统计表 + 两条 CPU 准备（词表 bypass anchor、220 集路线数） | 0.5 h | CPU |
  | N2-a | probe a / b 在 P5 v1 帧上 + desire 执行器检查（与步 1–3 并行，子执行员） | 1.5–2 h | CPU + < 1 GPU·h（GPU 2 空档） |
  | N2-b | N1 数据到后重跑 probe a / b、跑 probe c、x₁₀ 帧补 desire | 1 h | CPU + 少量 GPU |
  合计墙钟约 7–8 h（预计 17:30–18:00 CST 收尾）。任何一步超估计 2 倍停下写日志。盘：box 剩 167 GB，P5 v1 每 run 约 70 MB → 本节约 45 GB。

- 2026-09-26 09:58 CST [A] N1 的操作性选择（写于任何 P6 世界生成之前；代码 `jevdrive/p6.py`、`scripts/b2d_hooks.py` 的 `p6_world`、`scripts/p6_gen.sh`，recorder 仍是 `scripts/p5_pair_agent.py`）：
  - **世界与规模**：220 集 55 条路线（1W 20、2W 25、InvadingTurn 5、Emergency 5，都不在崩溃表里），3 个 TM seed。每 case 有 x₁₀、x₀₀，2W 再加 x₁₁、x₀₁；seed 0 另加天气 null（x₁₀ 换天气）、
    放置 null（1W / 2W）和镜像题（只 2W）。合计 **605 个世界**（x₁₀ 165、x₀₀ 165、x₁₁ 75、x₀₁ 75、天气 null 55、放置 null 45、镜像 25），比 todo 的约 580 多出镜像题 25 个。variant id = base × 100 + 世界码 × 10 + seed。
  - **x₀₀（障碍删）**：scenario 照常构建和运行（对背景的命令不变），它 spawn 的**全部** actor（车、自行车、锥桶和警示牌等 prop、救护车）每 tick 压在地下 500 m（沿用 P5 的藏法），
    并从 `CarlaDataProvider.active_scenarios` 删掉它的登记；InvadingTurn 的侵入车流（`InvadingActorFlow`）不 spawn。
  - **对向车流**：x₁₀ / x₀₀ / 天气 null / 放置 null 里，2W scenario 在触发后起的 `OppositeActorFlow` 不 spawn；HazardAtSideLaneTwoWays 的对向流来自背景（`ChangeOppositeBehavior(spawn_dist)`），这几个世界里改成关掉对向 source。
    触发前的背景对向车在所有世界里相同（不删）。`OppositeActorFlow` 的间距与车型改从一条以 TM seed 播种的私有随机流抽（「spawn 表写死」），x₁₁ / x₀₁ / 镜像题的车流表相同。
  - **放置 null**：每个 actor 沿其所在驾驶车道（`get_waypoint(lane_type=Driving)`）的法向移到障碍一侧，内缘离车道边线 0.5 m（VehicleOpensDoorTwoWays 再加 1.2 m 的开门余量），已在车道外的（侧边警示牌）不动；
    HazardAtSideLane 的自行车改的是它们的行驶横向 offset；登记删掉。
  - **镜像题**：只做 2W（1W 的相邻车道同向、没有「一直不断的对向车流」可造；护栏类改动不做）。对向车流间距 10–14 m（约 1 s 车头时距），PDM-Lite 的 gap check 应一直不通过。
  - **recorder**：与 P5 v1 的 PDM-Lite 配置相同（TFv6 shadow 照录，触发后 40 s、静止 40 s、仿真 70 s），另记 static prop 的位置与可见像素（语义类 20–22）、每个相机帧 PDM-Lite 的登记状态（`frames.jsonl` 的 `reg`），
    并在 ego 越过全部曾在前方的 scenario actor 10 m 后再录 8 s 就停（省空路）。
  - **横向量 d(t)**：ego 位置到原路线（`route.json`，未平移）折线的有符号距离，左正。Δ_lat(τ) 在帧 k 处 = d(k + τ)。
  - **smoke 1**（删登记）：每个 (x₁₀, x₀₀) 对，取 x₁₀ 的 |d(k + 3 s)| ≥ 1.0 m 的相机帧 k（x₁₀ 在 3 s 处正在绕）；判据量 = 这些帧上 x₀₀ 的 |d(k + 3 s)| < 0.3 m 的比例，合并 ≥ 95% 过。
  - **smoke 2**（对向车流）：对向车 = 航向与 ego 相反 > 135°、离 ego ≤ 50 m 的车。窗口 = 同 case x₁₁ 的横向起动时刻前 10 s 到后 3 s（x₁₁ 不绕则取触发后 5–25 s）；每个 x₁₁ / x₀₁ 世界窗口内至少 1 辆才过。x₁₀ 同窗口的对向车数并列报（只报不判）。
  - **世界级模式**（expert 统计的主读数）：窗口 = t_vis（x₁₀ 里障碍首次 ≥ 20 px 可见的相机帧；没有障碍的世界用同 case x₁₀ 的 t_vis；从未可见用触发帧）到录制结束。
    横向起动 t_lat = |d| ≥ 0.5 m 且持续 ≥ 0.5 s 的第一个 tick；bypass = 窗口内 max|d| ≥ 1.0 m（§2.1 的 1 m 门槛），侧别取 max 处的符号；
    停 = 速度 < 0.5 m/s 持续 ≥ 1 s。模式：bypass 且 t_lat 之前停过 → wait-then-bypass；bypass → bypass_L / bypass_R；没 bypass 但停过 → stop；其余 keep。
    「bypass 比例」= bypass_L + bypass_R + wait-then-bypass。帧级另按 §2.1 在 Frenet 坐标（Δd、相对路线切向的航向）上分类，作为副读数和之后考生混淆矩阵的口径。
  - **t_div**：沿用 P5 的 ego 分叉（1 cm / 0.1°），(x₁₀, x₀₀) 对上算；另报横向分叉 t_div_lat（|d_x₁₀ − d_x₀₀| ≥ 0.3 m 的第一个 tick）。主读数只取 t_div ≥ t_vis 的对，提前的对单列。
  - **negotiation**：同 case 的 x₁₁ − x₁₀，报横向起动延迟 t_lat(x₁₁) − t_lat(x₁₀) 与 x₁₀ 起动时刻两侧的速度差 Δv；「wait」= x₁₁ 的世界级模式为 wait-then-bypass 或 stop，门槛 ≥ 50%。
  - **门**：每类 scenario 的 x₁₀ bypass 比例 ≥ 70% 才算可用（IT、Emergency 也按这条报）；放置 null keep ≥ 90%；镜像题 stop ≥ 80%。
  - **卡与 server**：GPU 0–2，每卡 ≤ 6 个 CARLA server（受 pids.max 实时限制），server index 800–949，每实例 3 核（开跑时取没有被 pin 的核）。
- 2026-09-26 10:01 CST [A] 补两条（写于看 smoke 的任何数字之前）：世界级模式里的「停」只算 ego 曾经快过 3 m/s 之后（出生点原地不动不算停）；帧级 §2.1 的航向用 3 点滑动平均后相对 t₀ 的变化。
- 2026-09-26 10:03 CST [A] CPU 准备 (i) 的口径（写于算之前）：词表 = `waymo_heads.vocabularies` 的 K = 1024 k-means（WOD train futures，seed 0，即 P0 / `cls_late` 的配方；
  box 上没有存盘的那一份，按同一配方重算）。bypass 形状：ego 系 y 在 3 s 处 |y| ≥ 1.0 m，且 3–5 s 之间某点回到 |y| ≤ 0.5 m（todo 写死的阈值），另按 §2.1 给每个 anchor 分类并列。
  oracle 覆盖率：在同一规则下为 bypass 形状的目标轨迹上，最近 anchor（20 点平均 L2）也是 bypass 形状的比例，以及 minADE 中位数；目标轨迹两套：WOD train futures、P6 x₁₀ 里 expert 的 5 s 未来（数据到后补）。

## N2. openpilot 里有没有绕行需要的信息 + desire 执行器检查（CPU + 少量 GPU，< 1 h）

激发的前提是冻结特征里有信息。行人那一轮 openpilot vision 层 AUC 0.51，只能外接。绕行需要三样，逐样 probe：

| probe（线性，CARLA GT 标签，5 fold，AUC） | 标签 | 数据 |
|:--|:--|:--|
| a. 本车道前方 ≤ 30 m 有静止障碍 | GT actor 在 ego 车道、v < 0.5 m/s | P5 v1 现有帧先跑（障碍类少，报 n）；N1 数据到后重跑 |
| b. 相邻车道 ±20 m 内有车 | GT | 同上 |
| c. 对向车道 50 m 内有来车 | GT | N1 的 2W 世界 |

特征：openpilot `temporal`（Cinque、Lebowski）、`driving_vision` 输出、YOLO26x-seg image-plane token 集（对照）。
**判据**：AUC ≥ 0.70 记「有信息，可激发」；≤ 0.60 记「没有，要外接」；之间如实报。

**desire 执行器检查**（零训练）：在 P5 v1 直行帧（v ≥ 5 m/s，按速度分档 5–10 / 10–15 / > 15 m/s）上重跑 openpilot，desire = laneChangeLeft / Right 的上升沿脉冲
（`OPModel` 已有 rising-edge），读原生 plan 3 s 处相对不带 desire 的横向偏移。**判据**：每档 |Δ_lat(3 s)| 中位 ≥ 1.5 m 且方向正确 ≥ 90% → 执行器可用，
第三层可以走「模式头 → desire → openpilot plan」；中位 < 0.8 m → 记「openpilot 不按 desire 变道，执行器另找」。N1 的 x₁₀ 帧到后再补一遍（有障碍时）。
09-25 的 2b 是拿 desire 当导航（有害），这里是反应式触发，两者分开写。

- 2026-09-26 09:55 CST [A-N2] P5 v1 部分的操作性选择（写于任何 probe / desire 数字之前；代码 `jevdrive/night2_n2.py`）：
  - **帧**：两个 expert 集（`processed/carla_p5v1_{ba,pdm}`）各自 index 里 `source == p5` 的全部行（x⁺ / x⁻ / null 都用；P4 训练路线不算 P5 v1 帧）。
  - **几何**：每帧在本 run 的 `route.json` 折线上投影 ego 与 `actors.npz` 同一 server frame 的所有 vehicle / walker，得 (Δs, d)。ego 车道 = |d| ≤ 1.75 m，相邻车道 = 1.75 < |d| ≤ 5.25 m（按 3.5 m 车道；不查 CARLA map，因为不起 server）。
    藏在地下的 actor（|z − z_ego| > 5 m）不算。
  - **probe a 标签**：存在 vehicle 或 walker，速度 < 0.5 m/s，0 < Δs ≤ 30 m，|d| ≤ 1.75 m。**probe b 标签**（登记口径）：存在 vehicle，|Δs| ≤ 20 m，1.75 < |d| ≤ 5.25 m；
    另报「只算前方 0 ≤ Δs ≤ 20 m」一列作补充（三路相机都朝前，后方车看不见），判格按登记口径。
  - **probe**：训练折内标准化 + L2 logistic regression（C = 1），GroupKFold(5) 按 base 路线分组；判格用 5 折 AUC 的均值，另报折间标准差和 OOF 合并 AUC。
  - **特征**：openpilot `temporal`（Cinque / Lebowski，512 维）；「`driving_vision` 输出」取 tap 表里的 `vision`（temporal 模块之前的 pooled vision encoder 输出：Cinque `mean` 512 维、
    Lebowski `view_40` 3072 维），都是 `op_streams_vis` 已抽好的；YOLO26x-seg image-plane token 集 = E5 的检测（score ≥ 0.25，行人 / 骑车人 / 车）每路相机按 score 取前 8 个，
    每个 (类别 one-hot 3, u_c/W, v_c/H, w/W, h/H, score, mask) → 3 × 8 × 9 = 216 维（初稿误写 192，10:10 改正，未出数），不 lift、不筛。**YOLO 只有 BA 集**（PDM 集没有检测，补跑约 1.6 GPU·h，超本节预算，写「未测」）。
  - **desire 目标帧**：index 的 p5 行里 v_ego ≥ 5 m/s、intent = GO_STRAIGHT、前方 60 m 路线航向变化 < 10°、该 attempt 里目标帧之前至少 40 个相机帧；每个集每档最多 150 帧
    （每 attempt 最多 2 帧、相隔 ≥ 10 s，seed 0 抽样）。
  - **desire 协议**：每个目标帧三臂（无 / laneChangeLeft = 3 / laneChangeRight = 4），每臂都从目标帧前 40 帧（8 s）的零状态起跑，前 40 帧三臂逐字节相同；
    desire one-hot 从目标帧起一直保持（modeld 的 DesireHelper 在变道期间就是这样保持的），`OPModel` 只在上升沿发脉冲。
    **主读数 = 目标帧那一步的输出**（含脉冲的那次 forward；图像仍是直行 log，开环），另报 +0.2 s、+1.0 s 两个读点作补充，不用于判格。
    Δ_lat(3 s) = y_left(3 s | desire) − y_left(3 s | 无)，y_left = −plan_pos[:, 1] 在 T_IDXS 上插值到 3 s；方向正确 = Left 时 Δ > 0、Right 时 Δ < 0。
    每档的中位 |Δ| 和方向正确率把左右两臂合在一起算；Cinque、Lebowski 分别判格；两个 expert 集分别报并报合并。

## N3. 榜单 head × 反应：能力包的 2 × 2（CPU 为主，< 2 GPU·h）

同一个冻结 `temporal` 上，回答「榜单最优的 head 保不保反应、能力 head 掉不掉榜单分、叠起来两边能不能都留住」。

| head | NAVSIM PDMS / EPDMS（navtest） | WOD RFS | P5 v1 BA 行人 / cut-in 翻转 | I3 |
|:--|:--|:--|:--|:--|
| `ridge_late` | 有 | 有 | 有 | 有 |
| `cls_late` | 有 | 有 | **补** | **补** |
| Hydra 打分头 | 有（单 seed）→ **补 3 seed + Lebowski 行** | 不适用（WOD 无 PDM 子分，写「不适用」） | **补** | **补** |
| pair-Δ（M-C）不加 gate | 有（有害） | 有（有害） | 有 | 有 |
| Hydra + gated Δ（gate 取 G1 出的主 arm；G1 没出就用 P2(e) gate） | **补** | — | **补** | **补** |

- Hydra 头零样本上 P5 / I3 之前先做**兼容检查**：Hydra 训练用的 NAVSIM `temporal` 是 2 Hz 输入下抽的，P5 / I3 是 5 Hz；对照 = 同一 head 在 P5 null 帧上的
  输出分布与 navtest 上的分布（候选 argmax 的 top-10 重叠、走廊内 DAC 子分均值差）。差得离谱（top-10 重叠 < 30%）就写「不可比」，这格不读。
- **判据**（写死）：Hydra 头 P5 BA 行人翻转 CI 上界 < `ridge_late` 的点估计 → 「榜单 head 压掉反应」；CI 重叠 → 「同一水平」；
  Hydra + gated Δ 的 NAVSIM PDMS 掉 ≤ 1.0 且 P5 行人翻转 ≥ M-C 的 80% → 「能力包成立（CARLA 内）」。
- 顺带补两格：nuScenes 冻结 head 的 collision 率；E5 student 的 E4c latency 曲线。特征、标签、预测都在 box 上。

**[B] 执行记录（执行员 B；时间为 box 时钟 CST，每条早于它影响的数字）**

- 2026-09-26 09:55 [B] **分步估时**（N3 + N4，墙钟）：N3 代码（NAVSIM 头搬到 P5 / I3、兼容检查、P5 训的 `cls_late`、Hydra + gated Δ）2.5 h，与下面的 CPU 打分并行；
  Hydra seed 1 / 2 的逐 anchor 子分（E6 的 20 000 token × 1024 anchor，每个新词表约 37 core·h，≤ 48 核）约 1.5–2 h；6 次 Hydra 拟合 GPU 4 约 15 min；
  P5 / I3 考试与兼容检查 30 min；devkit（v1.1 PDMS + main EPDMS，navtest）约 22 个 job、4 路并行约 1 h；nuScenes collision + E4c student 1.5 h；表、图、decisions 1.5 h。N3 合计约 7 h，GPU < 0.5 GPU·h。
  N4：检测 embedding 补跑（YOLO26x-seg 在 P5 v1 BA 140 109 张图上重跑并取逐检测特征，GPU 4 约 20–30 min）+ 代码 1.5 h；60 次 student 拟合约 15 min GPU；延迟 20 min；写表 1 h。N4 合计约 4 h，< 0.5 GPU·h。
- 2026-09-26 09:58 [B] **N3 的操作化**（写于 N3 的任何新数字之前；此前只读过 G1 已提交的 `selection_auc.csv` 与 E6 / E1 / E5 / I3 已发表的数）。
  (0) **gate = G1 的主 arm g₂**：G1 按登记的训练行 AUC 已选出主 arm（`runs/real-data-transfer/select/20260926-091241/selection_auc.csv`：WOD、NAVSIM、两个模型都是 g₂，AUC 0.74 / 0.73；G1 的结果节尚未写，若它之后改主 arm，本节另记一行）。所以不用 P2(e) gate。
  (1) **Hydra 的 seed**（沿用 [SEEDS] (b) 的 NAVSIM 薄 head 口径）：seed s = CPU 确定性 k-means 词表 seed s + 留出 log 划分 seed s + 1；带子分标签的 20 000 个 navtrain token（E6 的 seed 0 子集）与它的 v1.1 metric cache 固定不变（相当于训练数据固定）。
  s = 0 直接用 E6 的逐 anchor 子分；s = 1、2 用 `scripts/elicit_e6_score.py` 原样给新词表重打分。拟合是 `elicit_e6.fit` 原样，只加「存 head 权重与标准化统计量」和「对任意特征矩阵出 logits」。
  s = 0 重拟合对 E6 已存 navtest 选择的一致率报出（GPU L-BFGS 非逐位确定，[SEEDS] 01:31 已定性；≥ 99% 同一 anchor 视为复现，否则停下报）。Lebowski 用同一批子分，`--model lebowski`，同样 3 seed。
  NAVSIM 读数 = navtest 官方 devkit（PDMS v1.1、EPDMS main @ 0a380a9），逐 token 配对 bootstrap 10 000 次（E6 `report` 的代码）；navhard 不做（表的列是 navtest）。
  (2) **P5 / I3 上的 NAVSIM 头**（零样本）：NAVSIM 训的 `ridge_late`（`navsim_heads.fit_ridge` 原样重拟合，确定性）、`cls_late`（同 seed 的 (a′)，与 Hydra 共用词表与模仿 logits）、Hydra。
  32 维 NAVSIM ego 输入由 P5 / I3 的 `past`（16 × 0.25 s，t0 帧的后轴位置、速度、每步速度变化）造：t = −1.5 / −1.0 / −0.5 / 0 s 取 past 第 9 / 11 / 13 / 15 步的位置；朝向 = 该步速度方向（速度 < 0.5 m/s 时沿用更晚一步的朝向，t0 为 0）；
  速度与加速度（past 的每步速度变化 / 0.25 s）旋到该步自己的车体系；command：GO_LEFT → left、GO_STRAIGHT → straight、GO_RIGHT → right、UNKNOWN → unknown。
  特征 = P5 `op_streams_vis` 的 `temporal`（M-C 同一份）、I3 `op_streams` 的 `temporal`（I3 考试同一份），都用 navtrain 的标准化统计量。输出的 8 个 0.5 s 位姿用 `elicit_e1._grid20` 放到 0.25 s 网格（1.75 s 取 1.5 / 2.0 s 的中点，与 E1 / G1 在 NAVSIM 上的激活率口径相同），
  judge = `p5_exam.exam` 原样（τ 由各自的 null 定）；I3 = `elicit_i3` 的 judge（去掉 TFv6 列）。
  (3) **兼容检查**（在任何 Hydra 的 P5 / I3 翻转数之前算、先写进结果）：P5 null 表引用的全部帧（fn_plus 与 fn_null）上，Hydra 选中 anchor 的频次 top-10 与 navtest 上的 top-10 的交集 / 10；
  「走廊内 DAC 子分均值差」操作化为：选中 anchor 的 σ(DAC head) 均值，P5 null 帧减 navtest（另报全部 1024 个 anchor 上的均值差，描述）。按模型 × seed 各报，**主判 seed 0**：top-10 重叠 < 30% → P5 列 Hydra 与 Hydra + gated Δ 写「不可比」、不读。
  I3 列同样在 I3 null 帧上检查一次。另报描述性 sanity（消融矩阵建议的）：NAVSIM 训的 `ridge_late` 在 P5 cut-in 上的翻转对 P5 训的 prior（76.9%），不作门。
  (4) **表里 `cls_late` 的 P5 / I3 格**：按「每列用该基准自己训的 head」读（`ridge_late` 的 P5「有」就是 P5 训的 prior），补 **P5 训的 `cls_late`**：P5 路线 5 折（`p5_exam.folds`，seed s），训练行未来 20 × 2 上 K = 1024 k-means（seed s），
  `cls ego`（ego 输入 = `p5_exam.ego_input`）λ 按 20% base 路线内层划分（seed s）上的 top-1 ADE 选，`cls ego` 在训练行上的 5 折 OOF logits 作 `cls_late` 的 offset，`cls_late` 输入 = 标准化 op `temporal`，λ 同规则；
  obs 行取所在 fold 的 top-1 anchor；I3 = 5 个 fold head 选中 anchor 的平均（与 `elicit_i3` 的 5 fold 平均同）。NAVSIM 训的 `cls_late` / `ridge_late` 零样本并列作 Hydra 的同训练源参照。
  (5) **判格**：主判按原文，`ridge_late` 的点估计 = P5 训的 prior（M-C 同 fold seed s 的 `prior [m]`，seed 0 为 Cinque 0.2% / Lebowski 2.7%）；Hydra 每个 seed 各判一次，三个 seed 一致取那一格，否则写「随 seed 变」（[G0] 的规则）；
  另把同一规则对 NAVSIM 训的 `ridge_late` 零样本、以及对 cut-in 各算一次，只作描述。
  (6) **Hydra + gated Δ**：pred = Hydra_s + g₂ · Δ，Δ = M-C 配对双流（seed 0，三处同一个 head 族）：NAVSIM 用 E1 的 `navtest_delta_<m>.npz`（加在 0.5 … 4.0 s，heading 不动，同 E1 / G1），P5 用 M-C seed 0 run 的逐 fold 交叉拟合 Δ（`M-C pair [m]` − `prior [m]`），
  I3 用 I3 考试的 `M-C pair [m]` − `ridge_late op-m temporal`。g₂：NAVSIM = G1 的 `g2_nav_eval.npz`；P5 与 I3 = **同一个 navtrain 训的 g₂ probe**（`g2_nav.pkl`）作用在 G0 格式的 embedding 上（I3 = `real_g0.load_embed('i3')`；P5 = 用 `real_g0.geom_check` (1) 的代码在 E5 检测上重算历史圆弧走廊的 embedding），
  即整个包（Hydra + Δ + gate）在三处是同一套权重；WOD 训的 g₂ 在 P5 / I3 上只作描述。判据原文：NAVSIM PDMS 对同 seed Hydra 的配对 Δ 点估计 ≥ −1.0，且 P5 行人翻转 ≥ 0.8 × M-C（seed 0 为 0.8 × 43.3% = 34.6%）→「能力包成立（CARLA 内）」；按 seed 各判，合并规则同 (5)。
  另报不加 gate 的 Hydra + Δ（P5 / I3 全部 seed，NAVSIM 只 seed 0）作描述。
  (7) **顺带两格**：nuScenes = `runs/nusc_backbones/ladder/20260925-144240/nusc_preds.npz` 里的冻结 head 预测（0.5 … 3.0 s 六个点，从后轴 t0 系换到 LIDAR_TOP 系，第 39 条考试的 `nuscenes_zs.per_sample` / `horizons` 原样），
  报 VAD 与 BEV-Planner 两种 collision（1 / 2 / 3 s）与 L2，scene bootstrap；只在第 39 条 index 里有 GT 框的 val 样本上算（n 照报）。E4c = `elicit_e4c` 的曲线代码原样，加 E5 student A / B（seed 0 主，1 / 2 并报），L = 0.1 s（与 openpilot 同，端到端 30 ms）。

## N4. 快通道去 lift：E5-b image-plane token（CPU + < 0.5 GPU·h）

第 45 条把召回缺口归到 flat-ground 放置；E5 的 embedding 又用同一条 lift 加手写走廊筛检测。改成不做几何、让配对差分自己学：

| arm | 检测 token（每路相机前 k = 8） |
|:--|:--|
| A（现 E5） | lift → 走廊筛 → (类别, x, y, 尺寸, score) |
| B | image-plane：(相机 id, u, v, w, h, 类别, score, 检测 embedding)，不 lift、不筛 |
| C | B ⊕ A |

P5 v1 BA 集，配对差分，3 seed，Cinque 与 Lebowski。**判据**：B 的行人翻转 CI 与 A 重叠或更高，且 DOC 非反应误翻不比 A 多 3 pp 以上 → 快通道改为 image-plane，
lift 只留在测量里；B 明显更低（CI 不重叠）→ 记「几何先验在这里有用」。cut-in 对 prior 的 Δ 并列报。端到端延迟同 E5 口径重测一次。

## N5. 放置修复只用于测量（GPU 1–2 h）

第 45 条要求登记的一项。metric depth（Depth Anything 3 metric 或 UniDepth v2，给相机内参）替代 flat-ground lift，只改**召回测量**，不进模型。
数据同第 45 条：P5 hazard 行人帧 + nuScenes 子集。**判据**：20–40 m 行人 BEV 召回（2 m 容差）从 0.12–0.22 升到 ≥ 0.40 → 第 45 条「缺口在放置」得到修法；
仍 < 0.30 → 记「单目 metric depth 也不够，需要地面高度」。第 45 条就地补一行。

- 2026-09-26 09:55 CST [C] 开工（执行员 C，N5 → N6，GPU 3；N6 抽特征时若 GPU 4 空着借来分片）。分步估时（写于任何 N5 / N6 数字之前）：

  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | N5-1 | `envs/depth`（torch 2.13 cu130 同 `envs/sam3`）+ UniDepth v2 ViT-L 与 DA3METRIC-LARGE 权重（各约 1.3 GB） | 30–45 min | CPU、网络 |
  | N5-2 | 子集 S 上逐图 metric depth（P5 9 918 张 + nuScenes 6 024 张 × 2 个模型），在 YOLO26x-640 与 SAM 3.1 的已存检测接地点上取深度 | 30–45 min | GPU 3，≤ 20 核 |
  | N5-3 | `fastperc.evaluate --contact depth` 原样（2 模型 × 2 检测器 + 固定 2 m 门 side）+ 表 | 20–30 min | CPU |
  | N6-1 | 抽特征模块 + 与 `features.py` 原类的逐位等价检查 + 吞吐 smoke | 45 min | Mac + GPU 3 |
  | N6-2 | V-JEPA 2 / DINOv2 / SigLIP2：46 703 行 × 3 路（去重后约 14 万个 clip），解码与推理重叠 | 30–60 min | GPU 3（+ GPU 4），≤ 48 核 |
  | N6-3 | openpilot small `temporal`：P5 v1 BA 的 858 个 stream、7.4 万帧，6 个分片（与 N6-2 并行） | 20–40 min | GPU 3，≤ 30 核 |
  | N6-4 | `ridge_late` + pair-Δ：5 个 backbone（含 Qwen 复现行）× 3 seed | 30 min | GPU 3 |
  | N6-5 | 表、图、decisions、todo 结果 | 60 min | Mac |

  合计墙钟约 5 h（到约 15:00），GPU 约 2.5 GPU·h。特征体积：三个图像 backbone 主 + 副 tap 共约 1.7 GB（float16），small `temporal` < 0.2 GB；数据盘现剩 168 GB（不是 260 GB），够。
  任何一步超估计 2 倍就停下写日志。

- 2026-09-26 09:55 CST [C] **N5 的操作性选择**（写于任何深度数字之前）：
  - **两个模型都测，主读数 = UniDepth v2**（`lpiccinelli/unidepth-v2-vitl14`，作者仓库 `infer(rgb, K)`，给针孔内参 fx / fy / cx / cy，输出沿光轴的 z 深度）；
    副读数 = Depth Anything 3 `DA3METRIC-LARGE`（作者 API，默认处理分辨率，按作者 README 的焦距换算成米）。理由：N5 写「给相机内参」，UniDepth v2 把内参当输入条件，
    DA3 metric 只在输出上按焦距缩放。两个权重 2026-09-26 都核实可下载（HF 未 gated）。判格按主读数下；副读数不一致时两个都写。
  - P5 相机有 Waymo k1 / k2 畸变，给深度模型的 K 不含畸变；接地点的射线仍由 `fusion_q4.lift` 去畸变，深度沿这条射线放（`fastperc._depth_place` 原样，射线光轴分量为 1，所以深度即 z 深度）。
  - 检测不重跑：主 = YOLO26x-seg 640（快通道检测器，E5 用的也是它）的已存检测，副 = SAM 3.1 原样（第 43 / 45 条基线）。取深度的规则沿用 `fastperc.depth_sample`：接地点上方 3 px 起 5 × 5 窗口的中位数，深度图在原图分辨率上。
  - 「2 m 容差」按第 45 条那组 0.12–0.22 实际用的匹配门理解：`fusion_q4.gate` = max(2 m, 0.1 d)（20–40 m 处 2–4 m），不改；另报固定 2 m 门作 side（平地与深度两边都算）。
  - 判读对象：行人 recall (ii)（≤ 40 m、在图内，P5 = 背景 actor，nuScenes = visibility ≥ 3）20–40 m 档，P5 与 nuScenes 各判：≥ 0.40 → 修法成立，< 0.30 → 不够，之间如实报。
    本条判格：两个数据集都 ≥ 0.40 才写「得到修法」，都 < 0.30 才写「需要地面高度」，否则分数据集写。并列报 0–10 / 10–20 m、hazard 行人（全部 / ≤ 20 m）和深度 / 平地放置距离比的中位数，
    因为 YOLO26x-depth 那次近处被弄坏（第 45 条），修远处不能以坏近处为代价，这一条只描述、不进判格。

## N6. backbone 行补到 P5 v1 BA（GPU 2–4 h）

`ablation-matrix.md` 的空格：V-JEPA 2（权重在 box）、DINOv2、SigLIP2、openpilot small 在 P5 v1 BA 集上抽特征（29 757 张，三路），拟合 `ridge_late` 和 pair-Δ，
读行人 / cut-in 翻转与 null。回答「视频 / 图像自监督特征里有没有 E 层信号」，与 WA-JEPA 的 +6 对照（`nohack-mechanisms.md` 说那是微调 encoder，与我们冻结不同口径，报时注明）。
**判据**：pair-Δ 行人翻转 CI 下界 > 该 backbone 自身 null p95 + 10 pp → 「有 E 层信号」。DINOv3 无权重，写「未测」。

- 2026-09-26 09:55 CST [C] **N6 的操作性选择**（写于任何 N6 特征或数字之前）：
  - **行数**：P5 v1 BA 索引 46 703 行（P4 训练 9 529 + P5 训练 17 746 + 观测 19 428），每行三路相机；本节写的「29 757 张」与索引对不上，按索引全量抽
    （`ridge_late` 与 pair-Δ 的训练行都要）。按 JPEG 真实路径去重后约 14 万个（行, 相机）单元。
  - **输入与主 tap**（每个 backbone 一个主 tap 进判格，副 tap 只并列）：V-JEPA 2 ViT-L（`vjepa2-vitl-fpc64-256`）每路相机用索引里同一个 4 帧 clip（5 Hz、跨 0.6 s，
    与 WOD 阶梯的 4 帧 × 0.2 s 相同），256² 拉伸，主 `mean`（WOD 的主 tap）、副 `last_mean`；DINOv2-base 当前帧，保持宽高比缩到 350 × 322（25 × 23 patch，
    约等于 nuScenes 配方 252 × 448 的 576 个 patch；原配方是横图，直接用会把 972 × 1079 的竖图压扁），主 `patch_mean`、副 `cls`；SigLIP2 so400m 当前帧 384² 拉伸（原生），
    主 `patch_mean`、副 `pooled`；openpilot small `temporal`（`scripts/p5_openpilot.py` 原样，5 Hz 每帧 hold 4 步，同 Cinque）。前三个的三路按 front / front_left / front_right 拼接。
    预处理用 `jevdrive/features.py` 里的原类（`VJepaFeatures`、`DinoFeatures`、`SiglipFeatures`），只换外面的循环。
  - **head**：`ridge_late <bb>` = `p5_exam.heads` 原样；**pair-Δ 主读数 = 单流**：`reactivity_mc.fit_fold` 原样、把 Qwen 流换成该 backbone，prior = openpilot Cinque `ridge_late`，
    Δ 只看该 backbone（= 第 42 条 M-C「只 Qwen」那一格，Qwen 行人 42.1%）；双流（backbone ⊕ Cinque `temporal`）与 Lebowski prior 并列作 side。λ 网格、μ、fold 规则一字不改。
    Qwen `L18_last` 作为第 5 个「backbone」在同一驱动里重跑一遍，seed 0 必须逐位复现已存的 M-C「pair qwen [cinque]」，不复现就先查驱动。
  - **seed**：route fold 排列 0 / 1 / 2（`2026-09-26-overnight-queue.md` [SEEDS]）。
  - **判据的操作化**：τ = 考生自身 null |Δ| 的 p95（`p5_exam` 原样），所以「null p95」对应的翻转率就是该考生的样本外 null false-flip（构造上约 5%）；
    判据读作：行人翻转的路线 bootstrap CI 下界 > 该考生（同一 seed）样本外 null false-flip + 10 pp。3 个 seed 都过 → 「有 E 层信号」，都不过 → 「没有」，混合 → 「不稳定」如实写。
    cut-in 报翻转率与对 prior 的配对 Δ（M-C `criteria` 口径），不进判格。

## N7. state-space policy（另一执行员，已派，`todos/2026-09-26-state-space-policies.md`）

找公开权重、装机、预登记后量 bypass / negotiation。与本文件其余节不共享 GPU 以外的东西。

## 分派与顺序

| 执行员 | 节 | 卡 |
|:--|:--|:--|
| A | N1 → N2（N2 的 probe 先在 P5 v1 上跑，N1 数据到后重跑） | 两张最空的卡（CARLA 6 server / 卡） |
| B | N3 → N4 | CPU 为主，零散 GPU |
| C | N5 → N6 | 一张卡 |
| D | N7 | 一张卡 |

顺序图：N2(P5 v1 部分) / N3 / N4 / N5 立即；N1 smoke 过后批量；N6 在 N5 后；N2 的 N1 部分最后。

- 2026-09-26 09:55 CST [main] A / B / C 由中央调度员（real-data-transfer 那个 session）派出，D 已在跑。卡：G0 / G3 已收工、5 张卡全空；
  A 用 GPU 0–2（每卡 ≤ 6 个 CARLA server，全 box ≤ 30，与 D 的 CARLA 合计），C 用 GPU 3，B 的零散 GPU 用 GPU 4；空出来的卡谁需要谁用，开工前看 `nvidia-smi`。

## 结果

（按节追加，每条带出处路径。）
