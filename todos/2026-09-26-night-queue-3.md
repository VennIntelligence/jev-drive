# 夜间队列 3（2026-09-26 晚 → 09-27 上午）：第三层读考生与激发、闭环（P7）、P6 扩容、真实数据上的 Δ 抑制、hack 核查与主表口径

状态: registered（16:15 CST 首版；16:50 CST 按用户要求重排：闭环恢复、用 P7 执行层，四条 lane 按卡 / 核 / 时间排好，每条 lane 是一次性脚本。两版都写于本队列的任何数字之前）

box 现在基本空着（7 卡各占 7–20 GB / 96 GB，load 28 / 175 核，线程 852 / 20 480，盘剩 543 GB）。仍在跑的 T2 尾巴（WA-JEPA NAVSIM 导出，GPU 6）与 T3 的 SimLingo 一格（GPU 6 + 4）不动，收工后卡和核归本队列（见「时间表」）。

## 这一轮的起点（中期结论，出处在 decisions）

1. **榜首的增量不在 E 层**（第 46 条 T1 / T2 / T3）：BridgeDrive 与 TFv6 分不开，BLUE 行人 5.9%（比 TFv6 低 24 pp），DrivoR 在 P5 上 3%、I3 上 34%；
   WA-JEPA 在 I3 上 66% 但非反应帧误翻 40%（openpilot `ridge_late` 70% / 18%），是「见车就减速」。出了 navtrain 生态，DrivoR / SparseDriveV2 / ZTRS 在 WOD 上都比 cv 差。
2. **反应来自训练信号，不来自 backbone 或 head**（第 48、53 条）：V-JEPA 2 / Qwen / SigLIP2 在均匀 imitation 下行人翻转都是 0，配对差分下 34–47%；分类头（`cls_late`、Hydra）比回归头更保不住反应。
3. **第三层：信息在，考卷有了，解码器和执行器缺**（第 49、52、51 条）：openpilot 冻结特征读得出本车道静止障碍 / 旁车道车 / 对向来车（AUC 0.85–0.92）；P6 v0 的 bypass 与 negotiation 对比成立；
   `cls_late` 词表没有绕行 anchor；desire 脉冲开环里方向对、幅度不够；公开 state-space policy 不能当执行层。
4. **真实数据仍是主阻塞**（第 44、53 条）：CARLA 激发的 Δ 在真实分布上是有方向的系统偏置，gate 缩小它但不消除，叠在 Hydra 上 NAVSIM 掉 5–8 分。
5. **快通道不需要 BEV**（第 50 条、N5）：image-plane token 与抬升版同一水平；真实相机上按内参的单目 metric depth 修好了 20–40 m 行人放置（nuScenes 0.22 → 0.57），只用于测量。

主线是第三层：P6 上读所有考生（榜首方法会不会绕）→ 在 openpilot 上用配对差分激发绕行 → **闭环里看激发出来的反应和绕行是否变成 SR**。
开环的配对考卷回答「有没有」，闭环回答「有用没用」，这一轮第一次两边都有。

## 通用规则

照抄夜间队列 2 的通用规则 1、3–6（判据先于数字、估时 + 2 倍停、3 seed 才能写「更好」、口径、结果路径），规则 2「不跑闭环」撤销（用户 16:45），另加：

7. **P6 的判卷口径（Q1、Q2 共用）**。帧窗 = 通过 t_div ≥ t_vis 的 bypass 对里，障碍首次可见 t_vis 到 expert 横向分叉 t_div_lat + 2 s 的 5 Hz 帧，只取 9 类可用障碍（第 52 条；InvadingTurn、开门车、Emergency 单列）。
   Δ_lat(k) = 考生在 x₁₀ 与 x₀₀ 同一 tick 输出的 3 s 处横向位置之差（ego 系，左正）；τ_lat = 该考生在天气 null 对上 |Δ_lat| 的 95 分位数。
   **bypass 翻转** = |Δ_lat| > τ_lat 且方向与 expert 绕行方向相同；**stop 替代** = 纵向 2 s 速度 Δ < −τ_lon（τ_lon 同 `p5_exam`）且不是 bypass 翻转。
   主读数 = 合并逐帧 bypass 翻转率，路线整组 bootstrap；判格「有 bypass」= CI 下界 > 天气 null 样本外误翻率 + 10 pp，**且**选择性：放置 null 的 40 个 keep 世界（第 52 条事后口径）上 bypass 翻转率 ≤ x₀₀ 误翻率 + 10 pp（不满足写「对『有东西』起反应，不是绕行」）。
   negotiation（x₁₁ − x₁₀，2W 5 类）：|Δ_lat| 首次过 τ_lat 的时刻 x₁₁ 比 x₁₀ 晚 ≥ 1 s 的对的比例，对 expert 0.65 报一致率，描述性。镜像题：向左 Δ_lat > τ_lat 的「借对向车道」率，> 50% 标「会冲进来车」。
8. 新写的考生适配器 / 闭环 agent 开考前做等价检查（native vs 我们的路径：同一条轨迹 ≥ 90% 或读数逐位相同；闭环 agent 另在 3 条路线上核对它每 tick 的模型输入输出与离线路径逐位相同），不过就停。
9. **闭环的执行层与归因（写在任何闭环数字之前）**。按第 41 条末段的建议：非共训的 planner（openpilot 原生 plan、我们所有的 head、Alpamayo 1.5）一律走 **P7**
   （`todos/2026-09-23-tfv6-controller/controller-eval/P7.json`，`controller_preset: pursuit`，plan 节奏 = 考生原生节奏，openpilot 与我们的 head 5 Hz、Alpamayo 2 Hz）；
   与作者执行层共训的 TFv6 / BridgeDrive / SimLingo / BLUE 用各自自带的执行层（作者 agent 原样）。
   **P7 没有过基础设施验收**（纵向跟随落后 0.73–0.81 s，专家 plan 经 P7 在 20 条验收路线上 DS 77.8–87.1，[closed-loop-acceptance](../docs/closed-loop-acceptance.md)），所以：
   (a) P7 列内的考生之间只做同路线同 seed 的配对差，差异归 planner；(b) 每张闭环表都带「专家轨迹经 P7 replay」这一行作执行层天花板；(c) P7 列与作者执行层列不直接比总分，只并列；
   (d) 这一轮不调 P7 的任何参数。
10. **闭环读数**：Bench2Drive 官方 220 路线、官方评测器（4000-tick 截断、完成阈值 99%），报 DS、SR、按第 38 条 hazard family 分组的 SR（突发 hazard / 让行与博弈 / obstacle bypass / 其余），
    配对差按路线整组 bootstrap（10 000）；多 seed 时先对 seed 取均值再 bootstrap。单次运行的噪声约 ±3 DS（第 41 条 P6 对 P5），单 seed 的差只写「同一水平」。
    **2026-09-26 18:10 CST 补（main，写于任何 CL3 / CL4 / CL6 / CL5 分数被读之前）**：P5 v0 / v1 与 P6 的路线都取自 `bench2drive220.xml`，所以我们的 head（P5 v1 BA 上训）在闭环里有一部分路线**训练时见过它的录像**，
    这正是我们要在榜单模型上查的「背题」。所以我们所有 head 臂（CL3–CL6、CL9）的每张表都按路线拆成三组分开报：seen（该 head 的训练集里有这条路线的录像）、unseen（220 里从未录过的路线）、合并；
    与 openpilot 原生、Alpamayo、作者执行层各臂的配对差**以 unseen 组为主读数**，合并只作参照。seen / unseen 名单由执行员从训练集的路线号直接导出，写进 `runs/nq3/b/route_split.json`，先于读分。
    判据 1–3 的主读数相应改为 unseen 组（unseen 组若少于 30 条路线，判据降为描述，写明）。干净的做法（交叉拟合）在 [队列 4](2026-09-26-night-queue-4.md) 的 K 里做。

## 各节的目标与判据

### Q1. P6 上读所有考生：谁会绕、谁只会停

- **不用重录的**（P6 v0 已有 Waymo 式三路相机与 TFv6 shadow）：openpilot Cinque / Lebowski 原生 plan（605 流已抽）、TFv6 waypoint 与 route + target speed、Alpamayo 1.5（nav，E[1 sample]）、
  DrivoR、WA-JEPA、SparseDriveV2、ZTRS（T1 / T2 的虚拟相机适配器原样）、我们的 `ridge_late` / `cls_late` / M-C / E5 student（P5 v1 上训的 checkpoint 原样，零样本）。
- **要挂各自 rig 重录的**：BridgeDrive、BLUE、SimLingo（T3 的 recorder 与离线 runner 原样，E1 逐 tick 核对 expert 轨迹）。v0 只重录主读数要用的世界（9 类的 x₁₀ / x₀₀、2W 的 x₁₁、放置 null、天气 null，约 400 个，录到最后引用 tick 为止）；v1 与 Q3 的新世界一起录。

判据：规则 7。另报每个考生的世界级模式分布（考生自己 5 s 轨迹按第 52 条 expert 的分类规则），与 expert 的一致率。**读法（写在数字之前）**：
- TFv6 waypoint「有 bypass」→ 第 38 条在 obstacle_bypass 格改写（第 47 条推翻条件之一）；route + target speed 通道只看纵向。
- NAVSIM 族「有 bypass」而 openpilot 原生 plan 没有 → 第三层是 navtrain 上 PDM 子分数学得到的配方，对我们是「可移植的 R 层配方」。
- 全部考生只有 stop 替代 → 第三层是公开方法的空白，Q2 是唯一的正例来源。
- Alpamayo 的 CoT 说 nudge 而轨迹不绕，单列「语言与轨迹不一致率」。

- 2026-09-26 16:40 CST [C] 开工（执行员 C，lane C），分步估时（写于本 lane 任何数字之前）。资源：GPU 6（T2 已收工，与 lane D ≤ 20 GB、T3 SimLingo 尾巴共卡），
  核段 `taskset -c 150-179`（30 核），OMP / MKL / OpenBLAS / NUMBA 线程 ≤ 30（并行子步骤按份分）。脚本 `scripts/nq3_c.sh`（tmux `jev:nq3-c`），状态 `runs/nq3/c/`。
  | 步 | 内容 | 估墙钟 | 资源 |
  |:--|:--|:--|:--|
  | C0 | 代码：P6 判卷模块（规则 7）与考卷帧集、考生适配（openpilot 原生 plan、TFv6、P5 v1 上训的读出零样本；NAVSIM 族与 Alpamayo 由两个子执行员写）、Q2 模块、链式脚本；子集 profiling 与数值一致检查 | 16:40–19:00 | Mac + GPU 6 小量 |
  | C1 | P6 全部 37 317 帧的特征：Qwen `L18_last`、V-JEPA 2 `mean`、YOLO26x image-plane token | 1.5 h | GPU 6 |
  | C2 | Q1 推理：openpilot 原生 plan（605 流）、NAVSIM 族 4 个（考卷帧）、Alpamayo 1.5（考卷帧，按实测吞吐定范围）、P5 v1 读出（CPU / GPU 分钟级）；TFv6 读已录的 shadow | 2.5 h（GPU 6 上并行） | GPU 6 |
  | C3 | Q1 判卷 + 小表 + 图 | 0.3 h | CPU |
  | C4 | Q2 pilot（v0）：A0–A4 × 2 模型 × 3 seed，按障碍类留一 + 按路线 5 折；backbone 对照 A1 / A3 | 1 h（与 C2 并行起跑） | CPU + GPU 6 |
  | C5 | 选臂、全部 v0 重训、写 `runs/nq3/q2/closed_loop_head/READY` | 0.2 h，目标 ≤ 00:30 | |
  | C6 | 等 `runs/nq3/a/v1/DONE`（约 03:00）；v1 帧的 openpilot 流与 backbone 特征 | 0.5 h + 1.5 h | GPU 6 |
  | C7 | Q2 v1（加 town 留出，必要时 A5） | 1.5 h | CPU + GPU 6 |
  任何一步超估计 2 倍，脚本停该步写 `runs/nq3/c/ERROR`。
- 2026-09-26 16:45 CST [C-nav] NAVSIM 族 4 个考生（SparseDriveV2、ZTRS、DrivoR、WA-JEPA）在 P6 v0 考卷帧上的操作口径（写于这四个考生的任何 P6 数字之前）。
  (1) 帧：`nq3_exam_frames.parquet` 的全部 18 782 个唯一帧（三档 priority 都跑），每帧读一次，同一帧出现在几个 reading 里共用同一条预测。估时超 2.5 h 才退到 priority 0 + 1，另写一条。
  (2) 输入与 P5 v1 BA 的 T1 / T2 路径逐项相同（P6 由 P5 v1 的 recorder 录，索引布局与相机 rig 相同）：T1 = 当前帧三路 JPEG → `carla_calib()` rig 的虚拟 nuPlan 相机，ego = `nav_ego(past, intent)`；
  T2 = `req_p5` 的构造（DrivoR 当前帧三路 + 黑后视；WA-JEPA 历史取最近的 5 Hz 帧 −1.4 / −1.0 / −0.4 / 0 s，早于流起点的钳到起点；ego 按 [T2] choice 3；命令 = `NAV_CMD[intent]`）。WA-JEPA 照 P5 用 bf16 autocast、batch 1。
  (3) 输出 `processed/top10_exam/p6/<model>.npz`（frame_name、raw、grid (n, 20, 2)，rear axle ego 系，x 前 y 左，米）。grid 沿用各自 P5 的转换：T1 `spline_grid`（4 s 之后按末两点直线外推），
  T2 `grid(traj)`（不平移，4 s 之后 NaN）。规则 7 只读 2 s / 3 s，两种约定对判卷没有影响。

### Q2. 在 openpilot 冻结特征上激发绕行

特征 = Cinque / Lebowski `temporal`（主），Qwen `L18_last`、V-JEPA 2 `mean` 作 backbone 对照（只跑 A1、A3）。每臂 3 seed × 2 模型。

| 臂 | 读出 | 训练信号 |
|:--|:--|:--|
| A0 | `ridge_late` 在 P6 expert 未来上重训（横纵向 1–5 s） | 均匀 imitation |
| A1 | A0 + 横向 pair-Δ（x₁₀ − x₀₀ 的 expert 横向差作 Δ 目标，M-C 的 `fit_fold` 换横向目标） | 配对差分 |
| A2 | 模式头（keep / stop / bypass-L / bypass-R / wait，按第 52 条规则从 expert 5 s 未来贴标）+ 每模式一条横向模板轨迹 | 均匀分类 |
| A3 | A2 + 配对一致性（同一 tick x₁₀ / x₀₀ 的模式 logit 差受 expert 模式差监督） | 配对差分 |
| A4 | `cls_late` 换词表：K-means 词表强制加入 P6 x₁₀ 与 WOD train 的 bypass 形状 anchor 各 64 条 | 均匀分类（vocabulary 对照） |
| A5（v1 上加跑，条件见下） | A1 / A3 + 第二个 Δ 项（x₁₁ − x₁₀，negotiation） | 配对差分 |

切分：**按障碍类留一**（9 折）为主读数，按路线 5 折为副；放置 null、镜像题、天气 null 只当考题。v0 做 pilot，v1 到了按同一代码重跑并加 town 留出。判据：规则 7，在测试折上：
- 「绕行被激发」= A1 或 A3 过判格 **且**比 A0 / A2 的 bypass 翻转率高（同帧配对差 CI 下界 > 0，3 seed 都成立）。
- 「只是词表问题」= A4 过判格而 A0 / A2 不过：第 47 条 `cls_late` 的 0 / 21 归 vocabulary。
- 镜像题借对向车道率 > 50% → 「激发出的是见障碍就绕，没有 gap 判断」，v1 上开 A5。
- 副读数（只报）：最好一臂零样本上 WOD 的 21 个 nudge 帧与 18 个双模式帧，报 nudge 预测率与 RFS。
- **交付给闭环**：v0 pilot 里过判格的最好一臂（没有过的就用 A1）按全部 v0 数据重训一版 checkpoint，写 `runs/nq3/q2/closed_loop_head/READY`，CL 队列看到它就把 CL5 插进去。

### Q3. P6 v1 扩容：更多路线、town 留出、recovery 题

- 路线：Bench2Drive 全量里 9 类障碍 scenario 的全部可用路线，每类 ≥ 20 条；按 town 留出测试组（占 20–30% 路线，定好写日志再生成）。世界类型同 v0，录制窗口延到障碍后 15 s 或路线结束。
- 放置 null 判卷主口径改为「放置 null 的模式 = 同 case x₀₀ 的模式」，登记门槛 0.90 仍报（看过 v0 数字后的改动，明示为事后口径）。
- **recovery 题**：每条 1W 路线另造 x_shift（出生点横移 ±1.0 / ±1.5 m，无障碍）与 x_center；先 smoke 10 个世界，PDM-Lite 3 s 内回到 |d| < 0.3 m 的比例 ≥ 0.80 才开，否则写日志不开。
  考生读数 = 出生后 1–3 s 的横向回线量（对 x_center 的差），「会回线」= 回线量 / 初始偏移中位 ≥ 0.5 且 CI 下界 > x_center 天气 null 抖动。
- recorder 同时挂 TFv6 shadow + BridgeDrive shadow + BLUE 相机（T3 证明加传感器不改仿真，E1 在 v1 上全量核）。
- 规模：**约 1 200 个世界**（首版写 2 000，按时间表缩：v0 实测每世界约 0.12 server·h → 约 140 server·h，18 个 server 约 8 h；60 MB / 世界 → 约 75 GB）。
- 门照 v0：每类 x₁₀ bypass ≥ 0.70 才进主读数；x₀₀ ≥ 95% 帧 |d| < 0.3 m；t_div ≥ t_vis。

### Q4. 真实数据上压住 Δ（第 53 条的翻案条件）

- **Q4a 训练时的真实帧零约束**：M-C 的 pair-Δ 训练加 λ·‖Δ(x)‖²，x = navtrain 与 WOD train 里走廊 ±4 m、30 m 内没有行人 / cyclist / 切入车的帧（GT 框只用于筛选），λ ∈ {0.1, 1, 10}，按 navtrain 留出 10% 的 PDMS 选 λ（先于任何 navtest 数字）。
  判「能力包成立」照第 53 条原登记：Hydra + Δ navtest PDMS 掉 ≤ 1.0 **且** P5 v1 BA 行人翻转 ≥ M-C 的 80%；另报 WOD RFS cluster mean 配对差、I3 车辆翻转。3 seed × 2 模型。
- **Q4b NAVSIM 协议兼容检查**：P5 v1 BA 帧按 NAVSIM 2 Hz sample-and-hold 重抽 openpilot 特征，重做第 53 条 top-10 重叠检查（门槛 30%）；过线则把 Hydra 的 P5 翻转按原口径补上。
- 读法：Q4a 过 → 系统偏置来自训练时没见过真实非 hazard 帧，可在训练里修；不过 → 偏置在特征分布差本身，需要真实配对数据，写进论文限定。
- Q4a 若过线，过线的 Δ（λ 选定版）作为 CL 队列的一个追加臂（`mc_real0`），与 M-C 同路线同 seed 配对。

- 2026-09-26 16:45 CST [D] 开工（执行员 D，lane D），分步估时（写于本 lane 任何数字之前）。资源：核段 `taskset -c 180-199`（20 核），OMP / MKL / OpenBLAS / NUMBA 线程 ≤ 20（并行子步骤按份分，devkit 的 ray worker 每个 1 个 BLAS 线程）；
  GPU 6 的空档，显存 ≤ 20 GB（与 lane C 共卡）。脚本 `scripts/nq3_d.sh`（tmux `jev:nq3-d`），状态 `runs/nq3/d/`，每步产物在 `runs/nq3/<q4a|q4b|q5|q6>/`。
  | 步 | 内容 | 估墙钟（box 时钟） | 资源 |
  |:--|:--|:--|:--|
  | D0 | Q4a 代码与链式脚本；λ_r = 0 对已存 M-C 的逐 fold 复现检查、devkit 子集打分对全量的逐位核对 | 16:45–18:45 | Mac + 小量 box |
  | Q4a-1 | 真实干净帧筛选（navtrain GT 框、WOD YOLO 检测）、navtrain 留出划分、navtrain 子集的 Qwen 抽取清单 | 15 min | CPU |
  | Q4a-2 | navtrain 约 8 000 token 的 Qwen `L18_last`（NAVSIM 2 Hz clip，2 个进程） | 50 min | GPU 6 ≤ 18 GB |
  | Q4a-3 | M-C + 真实帧零约束：3 seed × 2 模型 × 3 λ_r × 5 fold（GPU `eigh`）；Hydra 3 seed × 2 模型重拟合出留出 token 的选择 | 30 min | GPU 6 + CPU |
  | Q4a-4 | navtrain 留出（约 2 000 token）devkit v1.1：Hydra 与 Hydra + Δ_λ 共 24 个 job → 选 λ | 30 min | CPU 16 worker |
  | Q4a-5 | navtest devkit v1.1（6 个主 job + 6 个 g₂ 描述 job）、P5 / WOD / I3 读数、表、判格、`PASS` | 45 min | CPU + GPU 6 |
  | Q4b | P5 v1 BA 考卷帧按 NAVSIM 2 Hz sample-and-hold 重抽 openpilot `temporal`（两个模型）、Hydra 兼容检查；过线补 P5 翻转 | 1.5 h | GPU 6 + CPU |
  | Q5 | DrivoR / WA-JEPA 的 ego status 扰动（nuScenes main + navtest，GPU 推理 + devkit 打分）、选择性表（纯 CPU）、scorer argmax 扰动（navtest 256 token） | 4 h | GPU 6 + CPU |
  | Q6 | 主表口径统一（已存预测重算）、V-JEPA 2 单帧对照（P5 v1 BA 抽取 + 3 seed 拟合）、V-JEPA 2 流上真实数据（navtest 抽取 + 读数） | 3.5 h + 写作 | GPU 6 + CPU |
  预计 Q4a 在 22:00 前出判格，Q4b / Q5 在 22:00–03:30，Q6 在 03:30–07:30，之后写结果与 decisions。任何一步超估计 2 倍，脚本停该步写 `runs/nq3/d/ERROR`。
- 2026-09-26 16:50 CST [D] **Q4a 的操作化**（写于 Q4a 的任何数字之前；此前只读过第 42、44、53 条与 N3 / E1 / G0–G2 已发表的数）。
  (1) **真实干净帧**：走廊 = 该帧自己的 log 未来路径（navtrain 8 个 0.5 s 位姿，WOD 20 个 0.25 s 点）从原点起、沿末端朝向延长到 30 m（E3 的 `extend`），半宽 4 m，0 < s ≤ 30 m。
  物体：navtrain = t0 的 GT 框（E3 的 `extract` 缓存，五点判入廊）；WOD train 没有 GT 框，用 G0 已有的 YOLO26x-seg 三路前视检测（平地抬升 `lift_ok`，中心点）。
  干净 = 走廊内没有行人 / 骑车人，**且**侧带 1.5 m < |d| ≤ 4 m 内没有车辆（「切入车」的操作化：30 m 内贴在路径旁的车都当潜在切入，偏保守；|d| ≤ 1.5 m 的前车保留）。WOD 的筛选受 YOLO 召回限制（远处行人会漏），写作限定。
  (2) **留出与 navtrain 特征**：navtrain 全部 log 按 rng 0 分 90 / 10；λ 选择只用 10% 留出 log 里属于 E6 两万 token 子集的 token（有 v1.1 metric cache `v1_e6sub`）；零约束集不含留出 log。
  navtrain 没有 Qwen 特征，按 `navsim_qwen` 原配方（NAVSIM 2 Hz clip，P3(d″)）只抽：干净帧里随机 6 000 个（rng 0）+ 全部留出 token。WOD 用 `qwenvid_train_t4` 里全部干净帧。
  (3) **损失**（每个路线 fold）：Σ_pair |D W − R|² + μ Σ_train |Z_c W|² + λ_r · n_pair · [½ mean_{navtrain 干净}|(z − z̄) W|² + ½ mean_{WOD 干净}|(z − z̄) W|²] + λ |W|²；
  z 用该 fold 的 CARLA 训练行统计量标准化（与 E1 把 Δ 搬到真实数据的映射相同），z̄ = CARLA 训练行均值，所以约束的正是部署时加上去的 Δ(x)。两个数据集各占一半权重。
  λ_r ∈ {0.1, 1, 10}；ridge 的 λ 仍按原来的 3 折路线内层 CV 在配对 MSE 上选（零约束项在内层拟合里照加），μ、网格、fold 一字不改。**λ_r = 0 必须逐 fold 复现已存 M-C（seed s）的预测（≤ 1e-3 m、λ 相同）**，不过就停。
  (4) **seed**：seed s = M-C 路线分折 seed s（第 42 条三个 run）配 Hydra seed s（N3）。真实数据上的 Δ = 5 个 fold head 的 Δ 平均（E1）。
  (5) **选 λ**：每个模型 × seed 各选一次，取 navtrain 留出上 Hydra_s + Δ_λr 的 v1.1 PDMS（官方 devkit）最高的 λ_r。Hydra_s 在留出 token 上的选择来自 N3 `fit` 代码原样重拟合、把留出 token 作额外评测矩阵
  （navtest 上的选择对 N3 已存的同一 anchor ≥ 99% 才用；Hydra 训练时见过这些 token，对 Hydra 是样本内、对 Δ 是样本外）。Δ 加法同 N3：加在 0.5 … 4.0 s 的 xy，heading 不动。
  选择结果连同时间写进 `runs/nq3/q4a/lam_select.json`，**先于任何 navtest 打分**；navtest 只打选定的 λ。
  (6) **判格**：主臂 = Hydra_s + Δ_λ\*，**不加 gate**（零约束就是来替代 gate 的）；g₂ 门控版只作描述。NAVSIM = navtest v1.1 PDMS 对 N3 已存的同 seed Hydra 分数逐 token 配对（bootstrap 10 000），点估计 ≥ −1.0；
  P5 = 约束版 M-C 在 P5 v1 BA 上按 fold 出预测，`p5_exam.exam` + `reactivity_mc.criteria` 原样，行人翻转 ≥ 0.8 × 同 seed 已存 M-C 的行人翻转（43.3 / 43.6 / 42.4%）。两条都满足 →「能力包成立」。
  每个模型 × seed 各判，三个 seed 一致取那一格，否则写「随 seed 变」。描述：WOD RFS（E1 的 19 663 帧、prior = WOD train `ridge_late`；cluster mean 与 frame mean 的配对差，cluster 分层 bootstrap）、
  激活率，I3 车辆翻转（`elicit_i3` 的 judge，5 fold 平均）。
  (7) **交给 CL**：Cinque 的合并判格是「成立」才写 `runs/nq3/q4a/PASS`（CL 的 M-C 臂是 Cinque）；`mc_real0` = Cinque seed 0、λ\*(seed 0) 的 5 个 fold head（W、z̄、两路标准化统计量、λ，E1 `fold_heads` 的格式）放 `runs/nq3/q4a/mc_real0/`，附说明。
- 2026-09-26 17:20 CST [D] **Q4b 的操作化**（写于 Q4b 的任何数字之前）。
  (1) **只换时间协议**：P5 v1 BA 全部 obs 行（19 428 帧，兼容检查用其中 null 表引用的 6 352 帧，翻转用全部）按 NAVSIM 的输入协议重抽 openpilot `temporal`：NAVSIM 的 4 个 2 Hz 历史槽（−1.5 / −1.0 / −0.5 / 0 s）取最近的 5 Hz 录制帧
  （等距取较晚的：−1.4 / −1.0 / −0.4 / 0 s，与 T2 给 WA-JEPA 的取法相同；早于流起点的钳到第一帧，比例照报），每帧仍用 P5 自己的三路 rig 渲染（`p5_openpilot.render` 原样，与 `op_streams_vis` 同一渲染），
  然后是 NAVSIM 那条 rollout 原样（`navsim_zs_openpilot.schedule / rollout`：零状态、每个 2 Hz 帧在 20 Hz 时钟上保持到下一帧，Cinque 31 步、Lebowski 8 个 context-rate 相位，desire 无，右侧通行）。
  所以和第 53 条的 N3 检查相比只差「2 Hz sample-and-hold + 1.5 s 零状态历史」这一项；NAVSIM 只给 openpilot 前视一路（CAM_F0）这一差别不动（P5 前视 47° 覆盖不了宽视野帧，单路渲染反而离 NAVSIM 更远），写作限定。
  (2) **等价检查**：新 runner 按 P5 原协议（整条流、5 Hz、`run_stream`）在 8 条流的末尾 target 上复现已存 `op_streams_vis` 的 `temporal`（最大差 ≤ 1e-3），不过就停。
  (3) **兼容检查**：N3 `fit` 原样重拟合 Hydra（3 seed × 2 模型，navtest 选择对 N3 已存的同一 anchor ≥ 99%），P5 行换成新特征（navtrain 统计量、N3 的 32 维 ego 构造）；统计量同 N3 `compat`：P5 null 帧上 Hydra 选中 anchor 的频次 top-10 与 navtest top-10 的交集 / 10，
  **主判 seed 0，≥ 30% → 可比**，与 N3（5 Hz 协议）同表并列；同词表 `cls_late` 的重叠、选中 anchor 的 σ(DAC) 差作描述。
  (4) **过线才读**：可比的模型按 N3 [B] 09:58 (5) 的原口径补 P5 翻转：Hydra 每个 seed 对同 seed 的 P5 prior 用 `p5_exam.exam` + `reactivity_mc.criteria`，第 53 条判据 1：Hydra 行人翻转 CI 上界 < prior 的点估计 →「榜单 head 压掉反应」，CI 重叠 →「同一水平」，
  三个 seed 一致取那一格；NAVSIM 训的 `ridge_late`（2 Hz 协议特征）与 `cls_late` 并列作描述。不可比就维持「不可比」，读法写「P5 上 NAVSIM 训的读出选不同轨迹不是时间协议造成的，是场景 / rig 分布差」。

### Q5. 六族的 hack 核查

- **ego status 依赖**：WA-JEPA 在 nuScenes 上 L2 0.41（cv 0.71，全部考生最好），DrivoR 0.70。ego 速度 / 加速度 / 命令分别置零、置 cv 常数、换成同场景另一帧，重跑 nuScenes main 与 NAVSIM navtest。
  判：置零后相对 cv 的 L2 优势缩掉 ≥ 50% → 「nuScenes 分数主要来自 ego prior」（第 35 条 (a) 类）；NAVSIM PDMS 掉 ≥ 5 同理标注。
- **选择性表**：反应帧翻转 − 非反应帧误翻，六族 + openpilot + M-C，P5 与 I3 各一张（已有读数，纯 CPU）；< 10 pp 标「见车就减速」。
- **scorer argmax 脆弱性**：DrivoR、WA-JEPA 补 T1 同款扰动（±0.5° yaw、±5 cm 高度），报换轨迹率与平均位移，描述性。

- 2026-09-26 16:50 CST [D] **Q5 的操作化**（写于 Q5 的任何数字之前；此前只读过 T1 / T2 / T3 已发表的表）。代码 `jevdrive/nq3_q5.py`（请求、扰动、读数、表）与 `scripts/nq3_d/{drivor,wajepa}_arms.py`（模型 env 里的推理器），入口 `scripts/nq3_d/q5.sh`，run 在 `runs/nq3/q5/`，小表 `research/results/nq3/q5/`。
  (1) **请求**：nuScenes = T2 的 `runs/top10_t2/requests/nusc.npz` 原样（main 4 636，ego 由位姿差分，[T2-real] 10:25）；navtest = 我们冻结的 navtest 索引（`navsim_zs.load_index`，12 146 token）：4 个历史帧 × L0 / F0 / R0 / B0 的原图、相对当前帧的历史位姿、当前帧的 velocity / acceleration、`driving_command` 的 argmax。
  开跑前核对：16 个 token 上这些字段与 NAVSIM `SceneLoader` 给 agent 的逐项相同，WA-JEPA fp32 在这些 token 上的输出与 T2 已导出的 fp32 轨迹逐位相同；不过就停。
  (2) **扰动臂**：变量 S = (vx, vy)、A = (ax, ay)、C = 命令 one-hot。「置零」：S、A 置 0，C 置全零向量；「置常数」（todo 里的「cv 常数」）：S、A 换成该评测集全体的均值（nuScenes 4 636 帧、navtest 12 146 token 各算一次），C 换成该集最常见的命令，
  即一个不带逐帧信息的数据集常数（A 的均值约为 0，等于匀速假设）；「换帧」：换成同一 scene（nuScenes）/ 同一 log（navtest）里另一帧的值，组内按时间排序后循环平移 ⌊n/2⌋ 位，组里只有一帧的保留原值（个数照报）。
  每个模型 9 个单变量臂 + base + ALL0（S、A、C 同时置零）；WA-JEPA 另加 H0（history_trajectory 四个位姿置零，即「一直停着」），因为它的历史位姿本身带着速度，只扰 ego_status 会低估它对 ego 的依赖。
  **判格臂 = S0、A0、C0**（todo「分别置零」），其余全部是描述。
  (3) **精度与路径**：每个模型所有臂（含 base）走同一条路径：WA-JEPA = 推理器 bf16 autocast、batch 1（T2 考 nuScenes 的路径；nuScenes 的 base 必须与 T2 已存的 `nusc_wajepa` 逐位相同），T2 的 navtest fp32 EPDMS 91.71 只作参照；
  DrivoR = 推理器 fp32、batch 16（base 与 T2 的 `nusc_drivor` 逐位相同）。NAVSIM 一律用我们的 devkit v1.1 PDMS（`navsim_zs_score.sh`，回放 agent），base 也在同一条路径上重打。
  (4) **判据的操作化**：nuScenes 读数 = `top10_t2_real.exam_nusc` 的指标原样（VAD 口径 L2 1 / 2 / 3 s 均值、两种 collision），scene bootstrap 10 000，臂 − base 配对。优势 adv = L2(CV) − L2(模型)；
  S0 / A0 / C0 任一臂的 (adv_base − adv_arm) / adv_base ≥ 0.5（点估计，配对 CI 并报）→「nuScenes 分数主要来自 ego prior」。**只在 base 对 CV 的优势 CI 整体 > 0 时判**：DrivoR 在 T2 里对 CV −0.011 [−0.078, +0.059]，写「不适用（本来没有对 CV 的优势）」，只报 ΔL2。
  NAVSIM：navtest 全部 token 的 v1.1 PDMS，臂 − base 逐 token 配对 bootstrap 10 000；任一判格臂 base − arm ≥ 5（点估计）→ 同一标注。
  (5) **选择性表**：只读已存的合并行（`research/results/top10-exams/` 的 `t1_*`、`t2_*`、`p5_t3_*`，M-C 与 openpilot `ridge_late` 的 P5 run `runs/reactivity/mc-carla_p5v1_ba*/flip_rates.csv`、I3 考试 `runs/elicitation/i3-exam/*/flip_rates.csv`），
  选择性 = 反应帧翻转率 − 非反应帧误翻率（点估计；已存的是汇总表，不重做逐帧 bootstrap）。标注顺序：翻转率 CI 下界 ≤ 该考生样本外 null false-flip →「无反应」；否则选择性 < 10 pp →「见车就减速」。
  P5 行：SparseDriveV2、ZTRS、DrivoR、WA-JEPA、BridgeDrive / BLUE / SimLingo / TFv6 的 waypoint 2 s 通道、openpilot `ridge_late` 与 M-C 双流（Cinque / Lebowski，seed 0）；I3 行：前四个 + openpilot + M-C，其余 I3 没考过，写「未考」。
  (6) **scorer argmax 扰动**（描述）：navtest 索引顺序里每 ⌊12 146 / 256⌋ 个取一个，共 256 token。整个 rig 一起动（四路相机、四个历史帧同一个扰动），5 个臂：恒等、yaw ±0.5°（绕 ego z）、高度 ±5 cm。
  yaw 用 `navsim_rig` 的纯旋转重投影（源 = 该相机自己的原图，边缘看不到的像素为黑）；高度没有深度就无法精确渲染，用「地平面 + 无穷远」近似：虚拟相机的光线若在 200 m 内打到地面（NAVSIM 后轴系 z = −0.36 m，G0 的值）就取交点再投回原相机，否则按无穷远方向处理（地面以上的物体被当成无穷远，是近似）。
  五个臂都经过同一套渲染 + JPEG q95 落盘，比较对象是恒等臂（恒等臂对原图另报一次，是重编码的噪声地板）。读数：DrivoR 换选中候选的比例（任一位姿差 > 1e-3 m）与平均位移；WA-JEPA 是连续输出，报平均位移与位移 > 0.1 m 的比例。
  (7) **资源**：GPU 6（≤ 20 GB，与 lane C 共卡），CPU 全部 `taskset -c 180-199`，devkit ≤ 16 个 worker、每个 1 个 BLAS 线程。

### Q6. 主表口径统一 + 第 48 条的两个后续

- 按 [ablation-matrix-inventory](../research/ablation-matrix-inventory.md) 的 15 处不一致逐条定口径，从已存特征与 checkpoint 重算 backbone × head × 考卷主表（P5 v1 BA / PDM、I3、WOD、NAVSIM；P6 与闭环列等 Q1 / Q2 / CL 出来后补）。与旧表不同的格子逐个写原因；与 decisions 冲突的就地修正。
- V-JEPA 2 单帧对照：当前帧重复成 4 帧 clip，同一 pair-Δ，3 seed。行人翻转 < 10% → 「是时间不是视频预训练」；与 4 帧版 CI 重叠 → 「视频预训练本身」。
- V-JEPA 2 进 E5 student 上真实数据：Qwen 流换 V-JEPA 2，G0 口径上 WOD 与 NAVSIM；对不加 Δ 的配对差 CI 覆盖 0 → 快通道 backbone 换 V-JEPA 2 进候选。

- 2026-09-26 16:50 CST [D] **Q6 的操作化**（写于 Q6 的任何新数字之前；此前只读过盘点、第 42–53 条与各 todo 已发表的数）。
  **(a) 主表**。表 = 行（backbone × head）× 列（考卷），每格一个主数 + seed 数 + 出处；P6 与闭环两列写「待 Q1 / Q2 / CL」。「重算」= 从 box 上已存的逐帧预测 / checkpoint 用统一 judge 重新出数；已经按统一口径出过的格直接收，每格记「收 / 重算」。
  行：ego-only（`ridge ego`、`cls ego`）；Qwen3-VL-4B、V-JEPA 2、SigLIP2、DINOv2、openpilot small（`ridge_late`、pair-Δ 单流、pair-Δ 双流）；openpilot Cinque / Lebowski（`ridge_late`、`cls_late`、Hydra、M-C 双流 / 只 Qwen / 只 op / hard / 均匀、E5 student A / B、原生 plan）；
  榜单与 CARLA 族（SparseDriveV2、ZTRS、DrivoR、WA-JEPA、TFv6 waypoint / target speed、BridgeDrive、BLUE、SimLingo、Alpamayo 1.5，原生、零样本）。列：P5 v1 BA、P5 PDM、I3、WOD、NAVSIM、P6（待）、闭环（待）。15 条不一致逐条定为：
  1. RFS：绝对值一律 cluster mean（榜单口径）；配对 Δ 用 frame mean（E1 / G0 登记的 judge），并在同格括注 cluster mean（有逐帧预测就重算）。n = 479 为主，W-xfit 的 478 只在附录。
  2. `cls ego`：主表取 W-train 协议的 7.262（heads-train，3 seed）；P0 的 7.311 进附录，不混用。
  3. WOD 主协议 = W-train（train 训、完整 val 评；pre-onset 第 1–9 档 ΔADE 对 `ridge ego`，n = 1 291，V-JEPA 行 n = 1 249 照注）；W-half / W-xfit 只进附录。只有 W-half / W-xfit 数的行（DINOv2、SigLIP2、op small、Alpamayo 特征）WOD 格写「无 W-train」。
  4. s_ego 分档来自 `ridge ego` 自己的残差，所有「第 1–9 档」格带脚注，不改算法。
  5. PDMS（v1.1）为 NAVSIM 主列，EPDMS（main @ 0a380a9）并列，对文献只写「同量级」。
  6. navhard 不进主表（无 CI、seed 极差与行间差同量级），进附录。
  7. seed：每格写 seed 数；3 seed 的格报 seed 均值 [最小, 最大] 并附 seed 0 的 CI；单 seed 格标「1」，按通用规则只能读「同一水平」；确定性的 `ridge_*` 标「确定性」。
  8. Lebowski `cls_late`：3 seed 均值，不取 seed 0。
  9. nuScenes 不进主表（三套 L2 口径不能同列）；U-zs / U-head 进附录，各自带口径。
  10. P5：BA 逐帧定向翻转为主数（行人 / cut-in / 合并 + 样本外 null false-flip），A₃（[L, 3 s] 面积减 null）有就并列；PDM 集只报按对（E4 的决定）。
  11. τ 与 null：每张卷用各自的 null 定 τ，null false-flip 与翻转并列；`ridge ego` 在 P5 的 6–8% 标「τ = 0 的标签伪影，不是反应」，I3 上 0% 标「按构造」。
  12. expert：列头写明 expert（P5 BA = BehaviorAgent，PDM = PDM-Lite，I3 = 规则 expert，WOD = log / rater）；不同列的「翻转率」不跨列比较。
  13. Qwen 行按卷写 tap：P5 / I3 / M-C 是 P3(d″) 视频 clip 的 `L18_last`，WOD W-train 是单帧 `L18_mean`（arm A），NAVSIM 是 0.5 s 间隔 clip；行名带脚注。
  14. openpilot 输入时钟：NAVSIM 列的 openpilot 行标「2 Hz sample-and-hold 协议读数」（原生 plan 与冻结特征都是）。
  15. E2 的 1.44 是 R1（head Δ 幅值比），不进主表。
  代码 `jevdrive/nq3_q6.py table`：主表 `research/results/nq3/q6/main_table.csv`，逐格对旧表（盘点第 1–5 节里有数的格）`diff_vs_old.csv`（旧值、新值、原因代码：cluster-vs-frame / seed-mean / protocol-switch / new-cell / same），原因的文字由执行员逐格写进结果。
  **(b) V-JEPA 2 单帧对照**：N6 的抽取配方原样（`features.VJepaFeatures(frames=4)`、256² 拉伸、bf16、三路 front / front_left / front_right 拼接、主 tap `mean`、副 `last_mean`），唯一改动是 clip = 当前帧重复 4 次（每个 unit 只解码当前帧一次，同一张量复制 4 份，与把同一 PIL 图传 4 次逐位相同）。
  拟合 = `n6_backbones.fit` 原样（`ridge_late` + pair-Δ 单流 / 双流，prior = Cinque，Lebowski 只作 side，λ 网格、μ、fold 不改，`--eigh cuda`），route-fold seed 0 / 1 / 2。判格（todo 原文）：单帧版行人翻转（3 seed 均值）< 10% →「是时间不是视频预训练」；
  每个 seed 的单帧版行人翻转 CI 与同 seed 4 帧版（N6 已存）CI 重叠 →「视频预训练本身」；两条都不满足写「部分来自时间」，seed 间不一致写「随 seed 变」。
  等价检查（批量前）：同一驱动对 4 帧 clip 在 256 个 unit 上重抽，对 N6 已存特征报最大相对差（bf16 batch 形状噪声，N6 记 0.3–2%），> 5% 就停；同一驱动的 4 帧 seed 0 重拟合，行人翻转对 N6 已存值差 ≤ 1 帧（GPU `eigh` 的 0.5 mm 不确定性）、预测最大差 ≤ 1e-3 m，否则停。
  **(c) V-JEPA 2 流上真实数据**：读作「M-C 双流里的 Qwen 流换成 V-JEPA 2 `mean`」（= N6 的双流臂 V-JEPA ⊕ openpilot `temporal`），E5 student 不含 Qwen 流，所以「student」这里按第 48 条的推翻条件理解为快通道的配对差分读出。
  head = N6 各 seed 的 5 个 fold head（E1 `fold_heads` 的做法用 V-JEPA 替换 Qwen 重算，逐 fold 对 N6 已存预测 ≤ 1e-3 m、λ 相同），真实数据上的 Δ = 5 个 fold 的平均，CARLA 统计量；读数 = E1 / G0 的原样：
  WOD 用 E1 的 19 663 帧（已有 `vjepa2_p3` / `_fl` / `_fr`，4 帧 × 0.2 s、256²、ViT-L，与 N6 配方一致，三路按 front | front_left | front_right 拼），prior = WOD train `ridge_late`；
  NAVSIM navtest 需新抽 V-JEPA 2（三路 CAM_F0 / L0 / R0，NAVSIM 2 Hz 的 4 帧历史 −1.5 … 0 s，与 E1 的 Qwen NAVSIM clip 相同的「记录下来的输入差」：0.5 s 间隔对 P5 的 0.2 s），prior = NAVSIM `ridge_late`，官方 devkit v1.1 PDMS（EPDMS 不跑，省 CPU）。
  3 seed × 2 模型。判格：WOD 全部 rater 帧 RFS 配对差与 navtest 全部 token PDMS 配对差的 95% CI **都覆盖 0（或整体 > 0）**，3 个 seed 都成立 →「V-JEPA 2 进快通道候选」；另报 G0 的四格判定（有害 / 有用 / 无害无用 / 都不是）与同口径的 Qwen M-C（E1）对照。激活率的 τ 取该臂自己在 P5 null 上的 τ。

### CL. 闭环（Bench2Drive 220，官方评测器）

考生与执行层（规则 9）：

| 臂 | 考生 | 执行层 | seed | 优先级 |
|:--|:--|:--|:--|:--|
| CL0 | 全部下列 agent 各 3 条路线的 smoke + 规则 8 的逐 tick 等价检查；openpilot 适配器用 [openpilot-migration](2026-09-24-zeroshot-exam/openpilot-migration.md) A 部分修过的版本 | — | — | 最先 |
| CL1 | 专家轨迹经 P7 replay（执行层天花板） | P7 | 0 | 1 |
| CL2 | openpilot Cinque 原生 plan | P7 | 0 | 2 |
| CL3 | Cinque `temporal` + `ridge_late`（R 层）| P7 | 0 | 3 |
| CL4 | Cinque `temporal` + M-C（R + E 层，第 42 条） | P7 | 0 | 4 |
| CL5 | Cinque `temporal` + Q2 的绕行 head（`READY` 出现后插队到当前位置） | P7 | 0, 1, 2 | 插队 |
| CL5d | openpilot 原生 + desire：Q2 的模式头判 bypass-L / R 时发 laneChange desire 上升沿（开环第 49 条的执行器问题，闭环里图像会跟着走） | openpilot 自己的 plan → P7 | 0, 1, 2 | 与 CL5 同时插队 |
| CL6 | E5 student（Cinque ⊕ YOLO26x image-plane，第 50 条 B arm） | P7 | 0 | 5 |
| CL7 | openpilot Lebowski 原生 plan | P7 | 0 | 6 |
| CL8 | Alpamayo 1.5（第 33 条暂停在 13 / 220，重跑全量） | P7 | 0 | 7 |
| CL9 | CL2 / CL3 / CL4 的 seed 1、2 | P7 | 1, 2 | 8 |
| CL10 | TFv6、BridgeDrive、SimLingo、BLUE（top-10 todo 5.1 建议的那一项） | 作者自带 | 0 | 9（截止线，时间不够就顺延到上午） |

- 我们的 head 用 P5 v1 BA 上训、P5 表里的同一个 checkpoint；导航输入与 openpilot 原生同一套 desire（`b2d_zeroshot_agent` 的路口 desire）；head 输出的轨迹原点对齐到后轴后交给 P7（第 33 条那个 1.78 m 的 bug 不许再出）。
- CL5 / CL5d 只跑 220 里的 obstacle 类路线（Accident / Construction / ParkedObstacle / HazardAtSideLane 及其 TwoWays，约 40 条）× 3 seed，对照 = 同路线同 seed 的 CL4（M-C）与 CL2（openpilot 原生），CL9 跑完后对照也有 3 seed。
- **判据（写在数字之前）**：
  1. 「E 层激发变成闭环收益」= CL4 − CL3 在突发 hazard family 上 SR 配对差 CI 下界 > 0（3 seed），且总 DS 配对差 CI 下界 > −3。只有 1 seed 时写「同一水平 / 方向」。
  2. 「第三层激发变成闭环收益」= CL5 − CL4 在 obstacle 类路线上 SR 配对差 CI 下界 > 0（3 seed），且这些路线上碰撞数不多于 CL4。
  3. 「desire 执行器闭环可用」= CL5d 里触发的 desire 在 8 s 内完成横移 ≥ 2.5 m 的比例 ≥ 70%，且 obstacle 类路线 SR ≥ CL5 的 SR − 10 pp。
  4. 「openpilot 能开」（第 33 条悬而未决的那一半）：CL2 完成率与 DS 如实报，对 CL1 天花板写差距，不设门。
  5. CL10 与 P7 列只并列，按规则 9 (c) 不比总分；作者执行层列内部可以配对比（BridgeDrive − TFv6、BLUE − SimLingo，对第 46 条 T3 的开环结论作闭环复核）。

## 时间表与资源（box 时钟 CST；每条 lane 固定卡与核段，`taskset` 绑核，OMP / MKL / OpenBLAS / NUMBA 线程按 lane 上限设）

| lane | 内容 | 卡 | 核上限 | 17:30–22:00 | 22:00–03:00 | 03:00–09:00 |
|:--|:--|:--|--:|:--|:--|:--|
| A（CARLA 生成） | Q1 的 v0 重录（约 400 个世界）→ Q3 smoke + profiling → v1 批量 → BLUE / SimLingo 离线推理与 Q1 CARLA-rig 考生判卷 | 3、4、5（每卡 ≤ 6 server） | 60 | v0 重录（约 2 h）、Q3 smoke | v1 批量 | v1 收尾（约 03:00）、离线推理与判卷；**卡 3–5 在 v1 结束时自动交给 B** |
| B（闭环） | CL 队列 | 0、1、2（每卡 ≤ 6 server）；A 交卡后扩到 0–5 | 50 → 90（A 结束后） | CL0、CL1、CL2 | CL3、CL4、CL6（CL5 插队） | CL7、CL8、CL9，有余量做 CL10 |
| C（推理 / 训练） | Q1 不重录的考生 → Q2 pilot（v0）→ 交付闭环 head → Q2 v1（等 A 的 v1） | 6（T2 / T3 尾巴收工前与其共卡，≤ 40 GB） | 30 | Q1 推理与判卷 | Q2 pilot，约 00:30 交 `READY` | Q2 v1（03:00 后） |
| D（CPU 为主） | Q4a → Q4b → Q5 → Q6 | 6 的空档 | 20 | Q4a | Q4b、Q5 | Q6 |

- **核**：A 60 + B 50 + C 30 + D 20 = 160 / 175，留 15 给系统与 T2 / T3 尾巴；A 交卡后 A 只剩判卷（≤ 10 核），B 升到 90。
- **线程**（`pids.max` = 20 480，数线程不数核）：一个 CARLA server 约 430 线程，route client 用 `--client-threads 8` 约 16 线程；A 18 + B 18 = 36 个 server 约 16 000 线程，到顶了。
  B 扩到 36 个 server 只在 A 的 server 全部退出之后；每条 lane 的脚本启动新 server 前读 `/sys/fs/cgroup/pids.current`，> 17 000 就等。
- **盘**：Q3 约 75 GB、闭环 dump 关掉（`dump_every 0`）只存 per-route json，剩余 > 400 GB。
- **估时依据**：v0 生成每世界约 0.12 server·h（605 个世界约 12 卡·h）；闭环 220 条一轮按 full220 实测（8 worker 3.1 h，含大量 4000-tick 超时）估 15–25 worker·h，18 worker 约 1–1.5 h / 轮，obstacle 子集 40 条 × 3 seed 约 0.8 轮。
  到 09:00 B 能跑约 13–15 轮，CL0–CL9 约 12 轮，CL10 的 4 轮在截止线上。任何一步超估计 2 倍，脚本停该步、写 ERROR。

## 一次性脚本与轮询（用户 16:45：各执行员写一次性脚本，不用一直盯）

每条 lane 开工时写**一个**链式脚本（`scripts/nq3_<lane>.sh`，进 tmux `jev` 的 `nq3-<lane>` 窗口），把本 lane 的全部步骤按顺序串起来，执行员启动后不再手动推进：

1. **步骤之间自动衔接**：每步结束写 `runs/nq3/<lane>/<step>/DONE`（含墙钟、产物路径），下一步检查前一步的 `DONE` 再开；跨 lane 的依赖只靠这些文件（C 的 `q2/closed_loop_head/READY` → B 插 CL5 / CL5d；A 的 `a/v1/DONE` → B 扩卡、C 开 Q2 v1；D 的 `q4a/PASS` → B 追加 `mc_real0`）。
2. **自动重试与熔断**：单条路线 / 单个世界崩溃自动重试 ≤ 2 次后跳过并记录；同一批失败率 > 10%，或一步墙钟超估计 2 倍，停该步、写 `runs/nq3/<lane>/ERROR`（原因、最后 50 行日志、已完成比例），不再往下走。
3. **状态文件**：`runs/nq3/<lane>/STATUS.md` 每 10 min 刷新（当前步、完成比例、ETA、GPU / 核 / 线程占用），`events.jsonl` 照 long-runs 规范写。
4. **执行员的轮询**：每 1–2 h 看一次 `STATUS.md`；另起一个只等 `ERROR` 或整条 lane 的 `DONE` 出现的等待（Monitor 的 until-loop），出现就立刻处理。其余时间不看、不发进度消息。
5. **结果落盘也在脚本里**：每步结束自动生成小表到 `research/results/nq3/<节>/`；commit、decisions 回填由执行员在轮询时做（结论要人写）。

## 分派

| 执行员 | lane | 先做 |
|:--|:--|:--|
| A | A | 写 `nq3_a.sh`（v0 重录清单 → Q3 路线清单与 town 留出 → smoke → 批量 → 离线推理 → 判卷），先跑 v0 重录 |
| B | B | 写 CL 的 agent 适配（head 模式、desire 触发、P7 接线）与 `nq3_b.sh` 的优先级队列（含插队与扩卡逻辑），先跑 CL0 |
| C | C | 写 `nq3_c.sh`（Q1 推理 → 判卷 → Q2 pilot → 交付 head → 等 A 的 v1 → Q2 v1） |
| D | D | 写 `nq3_d.sh`（Q4a → Q4b → Q5 → Q6 的 CPU 部分） |

每节结果写回下面「结果」，结论回填 decisions（新条或就地修正，标**待定**）。

## 结果

（待写）
