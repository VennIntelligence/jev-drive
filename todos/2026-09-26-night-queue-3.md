# 夜间队列 3（2026-09-26 晚）：第三层读考生与激发、P6 扩容、真实数据上的 Δ 抑制、hack 核查与主表口径

状态: registered（2026-09-26 16:15 CST 写于任何本队列的数字之前；执行员按节领活，结果写回本节末尾，结论回填 decisions）

box 现在基本空着（7 卡各占 7–20 GB / 96 GB，load 28 / 175 核，盘剩 543 GB），本队列按宽裕排。仍在跑的 T2 尾巴（WA-JEPA NAVSIM 导出）与 T3 的 SimLingo 一格不动，它们收工的卡和核直接归本队列。

## 这一轮的起点（中期结论，每条的出处在 decisions）

1. **榜首的增量不在 E 层**（第 46 条 T1 / T2 / T3）：BridgeDrive 与 TFv6 分不开，BLUE 行人 5.9%（比 TFv6 低 24 pp），DrivoR 在 P5 上 3%、I3 上 34%，
   WA-JEPA 在 I3 上 66% 但非反应帧误翻 40%（openpilot `ridge_late` 70% / 18%），是「见车就减速」而不是有选择的反应。出了 navtrain 生态，DrivoR / SparseDriveV2 / ZTRS 在 WOD 上都比 cv 差。
2. **反应来自训练信号，不来自 backbone 或 head**（第 48、53 条）：V-JEPA 2 / Qwen / SigLIP2 在均匀 imitation 下行人翻转都是 0，配对差分下 34–47%；榜单常用的分类头（`cls_late`、Hydra）比回归头更保不住反应。
3. **第三层：信息在，考卷有了，解码器和执行器缺**（第 49、52、51 条）：openpilot 冻结特征读得出本车道静止障碍 / 旁车道车 / 对向来车（AUC 0.85–0.92，锥桶、事故车也读得出）；
   P6 v0 的 bypass 与 negotiation 对比成立；`cls_late` 词表没有绕行 anchor；desire 脉冲方向对、幅度不够；公开 state-space policy 不能当执行层。
4. **真实数据仍是主阻塞**（第 44、53 条）：CARLA 激发的 Δ 在真实分布上是有方向的系统偏置，gate 缩小它但不消除，叠在 Hydra 上 NAVSIM 掉 5–8 分。
5. **快通道不需要 BEV**（第 50、45 条补 N5）：image-plane token 与抬升版同一水平；真实相机上按内参的单目 metric depth 把 20–40 m 行人放置修好了（nuScenes 0.22 → 0.57），只用于测量。

所以这一轮的主线是第三层：先在 P6 上读所有考生（榜首方法会不会绕），再在 openpilot 上用配对差分激发绕行；同时扩 P6、试一个真正针对第 53 条的 Δ 抑制，
并把论文主表要的 hack 核查与口径统一做掉。闭环仍暂停，执行器问题（绕行要闭环才能验证）本轮只登记不跑。

## 通用规则

照抄夜间队列 2 的通用规则 1–6（判据先于数字、不跑闭环、估时 + 2 倍停、3 seed 才能写「更好」、口径、结果路径），另加：

7. **P6 的判卷口径（Q1、Q2 共用，写死于此）**。帧窗 = 通过 t_div ≥ t_vis 的 bypass 对里，障碍首次可见 t_vis 到 expert 横向分叉 t_div_lat + 2 s 之间的 5 Hz 帧，只取 9 类可用障碍（第 52 条；InvadingTurn、开门车、Emergency 不进主读数，单列）。
   读数 Δ_lat(k) = 考生在 x₁₀ 与 x₀₀ 同一 tick 输出的 3 s 处横向位置之差（ego 系，左正）；τ_lat = 该考生在天气 null 对（x₁₀ 对自己的天气 null）上 |Δ_lat| 的 95 分位数。
   **bypass 翻转** = |Δ_lat| > τ_lat 且方向与 expert 绕行方向相同；**stop 替代** = 纵向 2 s 速度 Δ < −τ_lon（τ_lon 同 `p5_exam`）且不是 bypass 翻转。
   主读数 = 合并逐帧 bypass 翻转率，路线整组 bootstrap；判格「有 bypass（第三层）」= CI 下界 > 该考生天气 null 的样本外误翻率 + 10 pp，
   **且**选择性：放置 null 40 个 keep 世界（第 52 条事后口径）上同一帧窗的 bypass 翻转率 ≤ x₀₀ 误翻率 + 10 pp（否则写「对『有东西』起反应，不是绕行」）。
   negotiation（x₁₁ − x₁₀，2W 5 类）读横向起动延迟：考生 |Δ_lat| 首次过 τ_lat 的时刻在 x₁₁ 比 x₁₀ 晚 ≥ 1 s 的对的比例，对 expert 的 0.65 报一致率，描述性不设门。
   镜像题读「借对向车道」率（向左 Δ_lat > τ_lat），描述性，> 50% 标「会冲进来车」。
8. 所有新写的考生适配器在开考前做与 T1 同款的等价检查（native vs 我们的渲染路径，同一条轨迹 ≥ 90% 或读数逐位相同），不过就停。

## Q1. P6 上读所有考生：谁会绕、谁只会停（GPU 约 8–10 卡·h + CPU）

目标：填第三层这一列。考生分两批：

- **不用重录的**（P6 v0 已有的 Waymo 式三路相机与 TFv6 shadow）：openpilot Cinque / Lebowski 原生 plan（605 流已抽，第 49 条）、TFv6 waypoint 与 route + target speed（shadow 已录）、
  Alpamayo 1.5（nav，E[1 sample]，与 WOD 零样本同配置）、DrivoR、WA-JEPA、SparseDriveV2、ZTRS（T1 / T2 的虚拟相机适配器原样）、我们的 `ridge_late` / `cls_late` / M-C / E5 student（P5 v1 上训的 checkpoint 原样，零样本）。
- **要挂各自 rig 重录的**：BridgeDrive、BLUE、SimLingo（T3 的 recorder 与离线 runner 原样，E1 逐 tick 核对 expert 轨迹；与 Q3 的扩容批量合并录，见 Q3）。先在 v0 的 605 个世界上录，Q3 的新世界录出来后同一口径补。

判据：通用规则 7。另报每个考生的世界级模式分布（bypass / wait-then-bypass / stop / keep，用考生自己的 5 s 轨迹按第 52 条 expert 的同一分类规则），与 expert 的一致率。
**读法（写在数字之前）**：
- TFv6 waypoint 若「有 bypass」，第 38 条「TFv6 高分主要不是 E 层」在 obstacle_bypass 这一格要改写（第 47 条推翻条件之一）；route + target speed 通道只看纵向（它的 route 就是导航路线，预期没有横向）。
- NAVSIM 族（DrivoR、SparseDriveV2、ZTRS、WA-JEPA）的词表 / scorer 里本来就有横向候选，若它们「有 bypass」而 openpilot 原生 plan 没有，说明第三层是 navtrain 上 PDM 子分数学到的配方能给的，对我们是「可移植的 R 层配方」而不是 E 层。
- 全部考生都「没有 bypass」、只有 stop 替代，则第三层确实是公开方法的空白，Q2 的激发是唯一的正例来源。
- Alpamayo 的 CoT 若说 nudge 而轨迹不绕（第 47 条 WOD 180 帧的现象），单列「语言与轨迹不一致率」。

## Q2. 在 openpilot 冻结特征上激发绕行（CPU 为主，< 2 GPU·h；先 v0 做 pilot，Q3 的数据到后按同一代码上 v1）

目标：回答「能不能像行人那样把绕行激发出来」。特征 = Cinque / Lebowski `temporal`（主），Qwen `L18_last` 与 V-JEPA 2 `mean` 作 backbone 对照行（第 48 条同款抽取）。

臂（每臂 3 seed × 2 openpilot 模型；backbone 对照行只跑 A1 与 A3）：

| 臂 | 读出 | 训练信号 |
|:--|:--|:--|
| A0 | `ridge_late` 在 P6 expert 未来上重训（横纵向 1–5 s） | 均匀 imitation |
| A1 | A0 + 横向 pair-Δ（x₁₀ − x₀₀ 的 expert 横向差作 Δ 目标，M-C 的 `fit_fold` 把纵向目标换成横向） | 配对差分 |
| A2 | 模式头（keep / stop / bypass-L / bypass-R / wait，按第 52 条规则从 expert 5 s 未来贴标）+ 每个模式一条横向模板轨迹 | 均匀分类 |
| A3 | A2 + 配对一致性（同一 tick 的 x₁₀ / x₀₀ 模式 logit 差受 expert 模式差监督） | 配对差分 |
| A4 | `cls_late` 换词表：K-means 词表里强制加入 P6 x₁₀ 与 WOD train 的 bypass 形状 anchor（各 64 条），其余同 `cls_late` | 均匀分类（第 52 条 4 的 vocabulary 对照） |

切分：**按障碍类留一**（9 折，每折测一类没见过的障碍）为主读数，按路线 5 折为副；训练集不含放置 null、镜像题、天气 null（它们只当考题）。
判据：通用规则 7，在留一类的测试折上；另两条（写在数字之前）：
- 「绕行被激发」= A1 或 A3 过判格 **且**比 A0 / A2 的 bypass 翻转率高（同帧配对差 CI 下界 > 0，3 seed 都成立）。
- 「只是词表问题」= A4 过判格而 A0 / A2 不过：那第 47 条 `cls_late` 的 0 / 21 归 vocabulary，第三层不需要配对监督。
- 镜像题上激发后的头借对向车道率 > 50%：写「激发出的是『见障碍就绕』，没有 gap 判断」，negotiation 需要单独的 x₁₁ − x₁₀ 配对（A1 / A3 的第二个 Δ 项），作为 A5 在 v1 上加跑。
副读数（只报不判）：CARLA 上训的最好一臂零样本上 WOD 的 21 个 nudge 帧与 18 个双模式帧（第 47 条），报 nudge 预测率与 RFS；按第 44 条的先例，预期真实数据上有偏置，只作 sanity。

## Q3. P6 v1 扩容：更多路线、town 留出、recovery 题（GPU 6 卡 × 约 6 h，CPU 约 100 核）

目标：v0 每类只有 5 条路线 × 3 seed，Q2 的留一类在 135 对上 CI 会很宽；v1 让每类 ≥ 20 条路线，并补 recovery 这一格。

- **路线来源**：Bench2Drive 全量（不止 220 集）里这 9 类障碍 scenario 的全部可用路线；按 town 留出一组测试 town（执行员按各 town 的路线数定，测试 town 占 20–30% 的路线，定好写进日志再生成）。
  世界类型同 v0（x₁₀ / x₀₀ / 2W 的 x₁₁ / x₀₁、放置 null、镜像题、天气 null），录制窗口延长到障碍后 15 s 或路线结束（v0 很多帧没有回正段）。
- **放置 null 的背景交通**：v0 剩下的 5 个 stop 是背景交通（第 52 条 3）。v1 的放置 null 与它的 x₀₀ 用同一 TM seed（本来就是），判卷口径改为「放置 null 的模式 = 同 case x₀₀ 的模式」为主、登记门槛 0.90 仍报。这条是看过 v0 数字后的改动，在此明示为事后口径。
- **recovery 题（新）**：每条 1W 路线另造 x_shift（ego 出生点横移 ±1.0 m / ±1.5 m，无障碍）与 x_center。expert = PDM-Lite；先 smoke 10 个世界确认 PDM-Lite 会回线（3 s 内 |d| < 0.3 m 的比例 ≥ 0.80，不过则 recovery 题不开，写日志）。
  考生读数 = 出生后 1–3 s 的横向回线量（对 x_center 的差），判格「会回线」= 回线量 / 初始偏移的中位 ≥ 0.5 且 CI 下界 > x_center 的天气 null 抖动。
- **Q1 的重录合并**：recorder 同时挂 TFv6 shadow + BridgeDrive shadow + BLUE 相机（T3 已证明加传感器不改仿真，E1 逐 tick 核对仍在 v1 上全量做）。
- 规模：目标约 2 000 个世界（v0 实测 605 个世界约 12 卡·h、60 MB / 世界 → 约 40 卡·h、120 GB）；超过 3 h 的批量先按 v0 的逐 tick 分项做 profiling pass，CPU 是上一轮的瓶颈（25 次渲染超时），这次按空余核开，每卡 ≤ 6 server。
- 门（照 v0）：每类 x₁₀ bypass ≥ 0.70 才进主读数；x₀₀ smoke ≥ 95% 帧 |d| < 0.3 m；t_div ≥ t_vis。

## Q4. 真实数据上压住 Δ：第 53 条的翻案条件（CPU 为主，< 1 GPU·h）

目标：第 53 条写了两条翻案条件，这一节各做一个。

- **Q4a 训练时的真实帧零约束**：M-C 的 pair-Δ 训练加一项 λ·‖Δ(x)‖²，x 取 navtrain 与 WOD train 里「走廊 ±4 m、30 m 内没有行人 / cyclist / 切入车」的帧（用各自的 GT 框筛，GT 只进训练的筛选，不进特征），λ ∈ {0.1, 1, 10}，按 navtrain 留出 10% 的 PDMS 选 λ（先于任何 navtest 数字）。
  判「能力包成立」照第 53 条原登记：Hydra + Δ 的 navtest PDMS 掉 ≤ 1.0 **且** P5 v1 BA 行人翻转 ≥ M-C 的 80%；另报 WOD RFS cluster mean 对不加 Δ 的配对差、I3 车辆翻转。3 seed × 2 模型。
- **Q4b NAVSIM 协议下的兼容检查**：在 P5 v1 BA 帧上按 NAVSIM 的 2 Hz sample-and-hold 重抽 openpilot 特征，重做第 53 条的 top-10 重叠检查（门槛 30%）；过线则第 53 条第 1 点变为可判，把 Hydra 的 P5 翻转按原口径补上。
- 读法：Q4a 过线 → 第 44 / 53 条的「系统偏置」是训练时没有见过真实非 hazard 帧造成的，可以在训练里修，真实数据这条路重新打开；不过线（PDMS 仍掉 > 1）→ 偏置在特征分布差本身，需要真实配对数据（G2 那条路），写进论文的限定。

## Q5. 六族的 hack 核查（CPU + < 1 GPU·h）

目标：论文主张「榜单分 ≠ 能力」需要对交集的六族各有一条直接的 hack 量，不只是「反应低」。

- **ego status 依赖**：WA-JEPA 在 nuScenes 上 L2 0.41（cv 0.71，所有考生最好），DrivoR 0.70。把 ego 速度 / 加速度 / 命令分别置零、置 cv 常数、换成同场景另一帧的值，重跑 nuScenes main 与 NAVSIM navtest。
  判（写在数字之前）：置零后相对 cv 的 L2 优势缩掉 ≥ 50% → 「nuScenes 分数主要来自 ego prior」（第 35 条 (a) 类 hack）；NAVSIM PDMS 掉 ≥ 5 同理标注。
- **WA-JEPA 的选择性**：I3 上非反应帧误翻 40%（static 类 58%），报「选择性 = 反应帧翻转 − 非反应帧误翻」给六族 + openpilot + M-C 一张表；选择性 < 10 pp 的标「见车就减速」。只用已有读数，纯 CPU。
- **scorer argmax 脆弱性**：T1 量到亚度级相机安装变化让 60% / 34% token 换轨迹；对 DrivoR、WA-JEPA 补同一扰动（±0.5° yaw、±5 cm 高度），报换轨迹率与平均位移，描述性。

## Q6. 主表口径统一 + 第 48 条的两个后续（CPU 为主，GPU 约 2 h）

- **口径**：按 [ablation-matrix-inventory](../research/ablation-matrix-inventory.md) 的 15 处不一致逐条定口径（RFS cluster mean、3 seed、τ 定义、「Qwen L18_last」统一为一种特征并改名另一种），
  从已存特征与 checkpoint 重算一张 backbone × head × 考卷的主表（P5 v1 BA / PDM、I3、WOD、NAVSIM，加 P6 列等 Q1 / Q2 出来后补）。数字与旧表不同的格子逐个写原因；与 decisions 冲突的就地修正。
- **V-JEPA 2 单帧对照**（第 48 条推翻条件）：当前帧重复成 4 帧 clip，同一 pair-Δ 读出，3 seed。判：行人翻转掉到 DINOv2 水平（< 10%）→ 写「是时间不是视频预训练」；与 4 帧版 CI 重叠 → 「视频预训练本身」。
- **V-JEPA 2 进 E5 student 上真实数据**（第 48 条推进条件）：E5 student 的 Qwen 流换成 V-JEPA 2，G0 的口径原样上 WOD 与 NAVSIM。判：WOD RFS / NAVSIM PDMS 对不加 Δ 的配对差 CI 覆盖 0（不再有害）→ 快通道 backbone 换 V-JEPA 2 进候选。

## 本轮登记但不跑

- **执行器**：绕行 / 回线要闭环验证（desire 持续多步、我们自己的横向轨迹 + 第 41 条控制器）。闭环暂停期间不跑；Q2 过线后再立 todo。
- 第二个 expert（非特权的绕行 teacher）：没有现成可用的（BehaviorAgent 不绕、state-space policy 不可用），等 Q1 看哪个考生绕得最像 expert 再说。

## 分派与顺序

| 执行员 | 节 | 卡 / 核（按开工时 `nvidia-smi` 与 load 选最空的，记在 run 目录） | 依赖 |
|:--|:--|:--|:--|
| A | Q3（含 Q1 的重录部分）→ Q1 重录考生的判卷 | GPU 0–3 + 5，CPU ≤ 110 核 | 无；最长的一条，先开 |
| B | Q1 不重录的考生 → Q2 pilot（v0）→ Q2 v1 | GPU 4，CPU ≤ 24 核 | Q2 v1 等 A 的数据 |
| C | Q4a → Q4b | GPU 6 空档，CPU ≤ 24 核 | 无 |
| D | Q5 → Q6 | GPU 6 / 4 空档，CPU ≤ 16 核 | Q6 的 P6 列等 Q1 / Q2 |

T2 / T3 的尾巴收工后，它们的卡给 A。每节结果写回下面「结果」，结论回填 decisions（新条或就地修正，标**待定**）。

## 结果

（待写）
