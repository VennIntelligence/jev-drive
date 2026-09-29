# nq3 三条机器判格的人工复核单（Q5 速度置零 / Q1 bypass / Q6 单帧对照）

2026-09-29 编制。对象是 [decisions.md](decisions.md) 里三处标着「GPT 按登记判格，待人复核」的新读数（第 35 条后、第 46 条后、第 48 条后各一处），
登记文本在 [夜间队列 3](../todos/2026-09-26-night-queue-3.md)。本单不改 decisions.md，每条末尾的「建议改写」只是提案，由你定。
预登记（Q1 / Q5 / Q6 的目标与判据原文）首次提交在 159151b（2026-09-26 17:11:55 +0900），8e66075（17:24:14 +0900）重组后的文字是下面各节引用的版本，三条的阈值与两次提交一致（我逐条比对过），都早于任何 nq3 数字；
下面引用的执行员条目自带 box 时钟（CST）时间戳。

**怎么用**：每节 (a) 是登记原文，(b) 是原始数字与我的复算，(c) 是机器给的判格，(d) 是我对「判格是否从规则和数字推得出来」的检查与可疑处，(e) 是建议改写。
先看下面的总表，再决定要不要读细节。

## 总表

| # | decisions 位置 | 机器判格 | 按登记规则是否成立 | 主要红旗 |
|:--|:--|:--|:--|:--|
| 1 | 第 35 条后（Q5 S0） | DrivoR −22.1、WA-JEPA −8.2，「ego prior」 | 成立（数字逐项复现，规则只要求点估计掉 ≥ 5） | 「主要来自 ego prior」措辞过强：S0 后仍保留 71.6 / 84.3 PDMS（cv 为 20.7 / 22.1，无视觉 ego-only 头 65.6–68.4）；S0 是自相矛盾的输入，comfort 分项一项就掉 0.29；没有图像遮挡的同口径对照 |
| 4 | 第 46 条后（Q1） | 除 PDM-Lite 外 no bypass，target speed longitudinal only | 成立（22 行逐格复算，无一行例外） | 判据对噪声大的考生没有功效：SparseDriveV2 的 τ_lat = 8.5 m，openpilot 5.7 m，此时「no bypass」是「不可测」；四个 CARLA 自带 waypoint 的模型比 null 高 3–4 pp（弱信号），二分判格看不到 |
| 6 | 第 48 条后（Q6） | 单帧 0.35–0.40 对 4 帧 0.46–0.48，「视频预训练本身」 | 按登记文字成立，但登记的判据太弱 | 「CI 重叠」是两个独立 CI 的比较；同帧同路线配对后单帧显著更低（seed 均值 −9.7 pp [−13.8, −5.7]），三个 seed 的配对 CI 都不含 0。单帧 V-JEPA 2 还与 SigLIP2（图像模型）同一水平，「视频预训练」并没有被分离出来 |

---

## 1. Q5 速度置零 S0（decisions 第 35 条后）

### (a) 预登记原文

Q5 的目标与判据（队列 3 登记，159151b 首提交，8e66075 重组后措辞）：

> - **ego status 依赖**：WA-JEPA 在 nuScenes 上 L2 0.41（cv 0.71，全部考生最好），DrivoR 0.70。ego 速度 / 加速度 / 命令分别置零、置 cv 常数、换成同场景另一帧，重跑 nuScenes main 与 NAVSIM navtest。
>   判：置零后相对 cv 的 L2 优势缩掉 ≥ 50% → 「nuScenes 分数主要来自 ego prior」（第 35 条 (a) 类）；NAVSIM PDMS 掉 ≥ 5 同理标注。

2026-09-26 16:50 CST [D] 的操作化（写于 Q5 任何数字之前）里与判格直接相关的一段：

>   (4) **判据的操作化**：nuScenes 读数 = `top10_t2_real.exam_nusc` 的指标原样（VAD 口径 L2 1 / 2 / 3 s 均值、两种 collision），scene bootstrap 10 000，臂 − base 配对。优势 adv = L2(CV) − L2(模型)；
>   S0 / A0 / C0 任一臂的 (adv_base − adv_arm) / adv_base ≥ 0.5（点估计，配对 CI 并报）→「nuScenes 分数主要来自 ego prior」。**只在 base 对 CV 的优势 CI 整体 > 0 时判**：DrivoR 在 T2 里对 CV −0.011 [−0.078, +0.059]，写「不适用（本来没有对 CV 的优势）」，只报 ΔL2。
>   NAVSIM：navtest 全部 token 的 v1.1 PDMS，臂 − base 逐 token 配对 bootstrap 10 000；任一判格臂 base − arm ≥ 5（点估计）→ 同一标注。

2026-09-26 17:40 CST [D] 因算力对 16:50 条目的修改（写于任何 Q5 考试数字之前）：

>   **对 16:50 条目的修改（写于任何 Q5 考试数字之前，理由是算力）**：WA-JEPA 全量（12 臂 × 16 782 个请求）按上面的速度要 30–70 GPU·h，GPU 6 只有空档，所以：
>   (a) WA-JEPA 只跑子集：nuScenes main 每 4 个取 1 个（1 159），navtest 每 10 个取 1 个（1 215，devkit 只打这些 token，base 也在同一子集上打）；
>   (b) WA-JEPA 的臂减为 base、S0、A0、C0（判格臂不变）与 ALL0、H0（描述），去掉置常数与换帧（DrivoR 保留全部 11 臂、全量）；
>   (c) WA-JEPA 走「8 个样本 × 全部臂一次前向」的 bf16 路径，base 也在其中，判格用的都是同一路径里的臂 − base；原登记「nuScenes base 与 T2 逐位相同」对 WA-JEPA 改为报告 base 对 T2 单独调用的差（路径噪声地板，不作门），fp32 的 navtest 等价核对（逐位）已过。
>   判据阈值不变。估时：DrivoR 约 1.5 h、WA-JEPA 约 2.5 h（两者在 GPU 6 上并行），devkit 约 1 h（与推理重叠），合计约 3 h 墙钟；每个 GPU 步骤开跑前等 GPU 6 空出显存（DrivoR 4 GB、WA-JEPA 12 GB），OOM 时 WA-JEPA 的 batch 减半重试。

2026-09-26 18:40 CST [main] 对缩规模的复核（写于任何 Q5 / Q6 数字之前）：

> - 2026-09-26 18:40 CST [main] 复核 lane D 的两处（写于任何 Q5 / Q6 数字之前）：(1) 同意 Q5 对 WA-JEPA 的缩规模（nuScenes 1/4、navtest 1/10、3 个判格臂 + 2 个描述臂）与 bf16 批量前向；
>   条件：置零臂与基线臂必须走同一条批量 bf16 路径，使 3 cm 级的数值差在配对差里抵消；报告里写明子样本与 CI 变宽。(2) 同意 Q6 (c) 的读法：E5 student 没有 Qwen 流，

### (b) 原始数字与复算

术语：PDMS（NAVSIM v1 的 PDM Score，把碰撞、可行驶区域、进度、TTC、舒适度等按规则合成 0–100 的分）；cv（constant velocity，匀速外推基线）；S0 / A0 / C0 分别把 ego 速度 (vx, vy)、加速度、驾驶指令置零；
CI 是逐 token 配对差的 bootstrap 95% 区间。

**NAVSIM navtest（判格用）**，出处 [ego_navtest.csv](results/nq3/q5/ego_navtest.csv)。我在 box 上直接读 devkit 逐 token 分数（`runs/navsim/eval/v1_navtest_nq3q5_*`）重算：
点估计逐位相同，CI 端点差 ≤ 0.07（我的 bootstrap 是 4 000 次、种子不同，原脚本 10 000 次）。

| 模型 | n（token） | 臂 | PDMS | 配对差 [95% CI]（CSV） | 我的复算 CI | ≥ 5 |
|:--|--:|:--|--:|:--|:--|:--|
| DrivoR | 12 146（全量） | base | 93.691 | — | — | — |
| | | **S0** | 71.575 | **−22.115 [−22.657, −21.578]** | [−22.655, −21.587] | 是 |
| | | A0 | 90.056 | −3.635 [−3.991, −3.288] | [−3.981, −3.289] | 否 |
| | | C0 | 92.897 | −0.794 [−1.002, −0.572] | [−1.016, −0.578] | 否 |
| WA-JEPA | 1 215（每 10 个取 1，bf16） | base | 92.423 | — | — | — |
| | | **S0** | 84.264 | **−8.159 [−9.406, −6.942]** | [−9.336, −6.929] | 是 |
| | | A0 | 91.809 | −0.613 [−1.153, −0.107] | [−1.126, −0.117] | 否 |
| | | C0 | 92.625 | +0.202 [−0.199, +0.572] | [−0.206, +0.576] | 否 |

参照（同一批 token 上的 cv，我读自 devkit 的 `v1_navtest_cv`）：DrivoR 那 12 146 个 token 上 20.65，WA-JEPA 那 1 215 个 token 上 22.15。
仓库里已有的无视觉 ego-only 头：`cls ego` PDMS 68.4 / 67.8，文献 Ego Status MLP 65.6（[decisions.md](decisions.md) 第 37 条附近）。

**描述臂**（不进判格，同表）：DrivoR Sk（换成数据集平均速度）−11.38、Ss（换成同 log 另一帧的速度）−17.97、ALL0 −26.90；WA-JEPA ALL0 −7.12、H0（历史位姿置零）+0.07 [−0.01, +0.24]。

**S0 掉分的分项**（我从 devkit 分项读出，未入库；子分数均值）：

| | 模型 | base | S0 |
|:--|:--|--:|--:|
| DrivoR | comfort / ego_progress / TTC / no-collision / drivable | 1.000 / 0.899 / 0.967 / 0.990 / 0.989 | 0.710 / 0.687 / 0.823 / 0.962 / 0.945 |
| WA-JEPA | 同上 | 1.000 / 0.859 / 0.987 / 0.997 / 0.981 | 0.933 / 0.737 / 0.959 / 0.991 / 0.967 |

**nuScenes（附带的另一半判格）**，出处 [ego_nusc.csv](results/nq3/q5/ego_nusc.csv)：DrivoR 对 cv 的 L2 优势 +0.011 [−0.059, +0.078]，CI 含 0，登记规则规定「不适用」。
WA-JEPA（n = 1 159，nuScenes main 每 4 个取 1）base L2 0.421（cv 0.712），优势 +0.291 [+0.237, +0.345]；S0 后 L2 4.533，优势 −3.82，缩减 1 412%（≥ 50% 线）；A0 缩 33%、C0 缩 34%（未过线）。
路径噪声地板：WA-JEPA 的 bf16 批量 base 对 T2 单独调用，平均位移 2.7 cm（[verify.json](results/nq3/q5/verify.json)）；DrivoR base 与 T2 逐位相同。

### (c) 机器判格

decisions 该段原文：「Q5 NAVSIM速度置零S0：DrivoR PDMS差−22.115 [−22.657,−21.578]；WA-JEPA −8.159 [−9.406,−6.942]，均下降≥5，按原判格 ego prior。nuScenes DrivoR原verdict not applicable，WA-JEPA原verdict nuScenes score mainly from the ego prior (S0)」。
CSV 里的标签是 "PDMS mainly from the ego prior (S0)"。

### (d) 我的检查

**规则层面：成立。** 登记只要求「任一判格臂 base − arm ≥ 5（点估计）→ 同一标注」，两个模型的 S0 都过线，CI 下界（−22.7、−6.9）也都离 5 很远；A0、C0 都没过线；
nuScenes 一半按规则「DrivoR 不适用、WA-JEPA 过线」也逐项对得上。数字与 CSV 完全一致，没有发现偏离登记：缩规模发生在数字之前并经 main 批准，置零臂与 base 走同一条 bf16 路径。

**可疑处（按重要性）**：

1. **标签措辞超出证据。** 「ego prior」标注源自第 35 条 (a) 类，含义是「分数主要靠 ego 先验拿到」。但 S0 之后 DrivoR 仍有 71.6、WA-JEPA 仍有 84.3，
   远高于 cv（20.7 / 22.1）；掉的是 22.1 / 93.7 = 24% 和 8.2 / 92.4 = 9% 的分数。DrivoR 的 71.6 与无视觉 ego-only 头（65.6–68.4）同量级，所以「DrivoR 的 PDMS 里大约有 2/3 到 3/4 是不看图也拿得到的」这个读法有支持，
   而「主要来自 ego prior」对 WA-JEPA（掉 9%）几乎没有支持。规则里 ≥ 5 的线是 todo 从 nuScenes 的「优势缩掉 ≥ 50%」类比过来的，两者不同量纲，PDMS 这半边没有对应的「缩掉多少比例」判据。
2. **S0 是自相矛盾的输入。** 速度置 0 但相机、历史位姿（DrivoR）与加速度都在显示车在动。任何把速度当输入的规划器都会掉分，这是「依赖 ego 速度」，不等于「用 ego 先验代替看图」。
   分项也支持这个读法：DrivoR 的 comfort 从 1.000 掉到 0.710，一项就占了很大一块（轨迹起点速度与真实速度不连续），ego_progress 掉 0.21，TTC 掉 0.14。
   一致性更好的臂（Sk、Ss）DrivoR 仍掉 11.4 / 18.0，方向稳；WA-JEPA 的 Sk / Ss 因算力被去掉了，只有 S0 一个臂。
3. **缺同口径的对照。** 第 35 条的旧说法是「速度 ×0.5 的伤害是图像全黑的 35 倍」，来自另一套读数；本轮 Q5 没有同一条路径上的图像遮挡臂，所以「相对看图，ego 有多重要」这个比较在这一轮里没有。
4. **WA-JEPA 是 1/10 子样本、单 seed、单 checkpoint，bootstrap 是逐 token 独立重采样**（同一 log 内的 token 相关，CI 偏窄；但下界 −6.9 离 −5 线还有余量，结论不敏感）。DrivoR 是全量。
5. nuScenes 一半：WA-JEPA 的缩减 1 412% 只说明「L2 优势翻成负数」，本身没有量级信息；另外 nuScenes 只有 1/4 子样本，DrivoR 一半「不适用」不等于「没有 ego 依赖」（它的 S0 使 L2 从 0.70 到 5.79）。

### (e) 建议改写（提案，未采用）

> ### 2026-09-27 新读数：Q5 ego status 置零（登记判格，待人复核）
> navtest v1.1 PDMS，S0（ego 速度置零，其余输入不动）：DrivoR 93.69 → 71.58（−22.1 [−22.7, −21.6]，n = 12 146，全量）；WA-JEPA 92.42 → 84.26（−8.2 [−9.4, −6.9]，n = 1 215，1/10 子样本，bf16，单 seed）。
> 两者都过登记的「掉 ≥ 5」线，A0、C0 都不过（−3.6 / −0.8；−0.6 / +0.2）。按登记标注：两个模型的 PDMS 对 ego 速度输入**高度敏感**。
> nuScenes：DrivoR 对 cv 无优势（不适用）；WA-JEPA 的 L2 优势在 S0 下翻成负（0.42 → 4.53，cv 0.71）。
> **限定**：(1) 登记的标注文字是「ego prior」，但 S0 后两者仍有 71.6 / 84.3（cv 20.7 / 22.1，无视觉 ego-only 头 65.6–68.4），不能读成「分数主要来自 ego 先验」；DrivoR 的一半以上分数与无视觉头同量级，WA-JEPA 只丢了 9%。
> (2) S0 与图像、加速度矛盾，DrivoR 的 comfort 分项 1.000 → 0.710 占了掉分的一大块，掉分里有一部分是接口不连续，不是「靠先验」；DrivoR 的换值臂 Sk / Ss 也掉 11.4 / 18.0。
> (3) 没有同路径的图像遮挡臂，「ego 与图像谁更重要」不能由本读数回答。（**待定**）

---

## 4. Q1 bypass 判格（decisions 第 46 条后）

### (a) 预登记原文

规则 7（P6 的判卷口径，队列 3 通用规则；159151b 首提交，8e66075 重组后措辞，阈值不变）：

> 7. **P6 的判卷口径（Q1、Q2 共用）**。帧窗 = 通过 t_div ≥ t_vis 的 bypass 对里，障碍首次可见 t_vis 到 expert 横向分叉 t_div_lat + 2 s 的 5 Hz 帧，只取 9 类可用障碍（第 52 条；InvadingTurn、开门车、Emergency 单列）。
>    Δ_lat(k) = 考生在 x₁₀ 与 x₀₀ 同一 tick 输出的 3 s 处横向位置之差（ego 系，左正）；τ_lat = 该考生在天气 null 对上 |Δ_lat| 的 95 分位数。
>    **bypass 翻转** = |Δ_lat| > τ_lat 且方向与 expert 绕行方向相同；**stop 替代** = 纵向 2 s 速度 Δ < −τ_lon（τ_lon 同 `p5_exam`）且不是 bypass 翻转。
>    主读数 = 合并逐帧 bypass 翻转率，路线整组 bootstrap；判格「有 bypass」= CI 下界 > 天气 null 样本外误翻率 + 10 pp，**且**选择性：放置 null 的 40 个 keep 世界（第 52 条事后口径）上 bypass 翻转率 ≤ x₀₀ 误翻率 + 10 pp（不满足写「对『有东西』起反应，不是绕行」）。
>    negotiation（x₁₁ − x₁₀，2W 5 类）：|Δ_lat| 首次过 τ_lat 的时刻 x₁₁ 比 x₁₀ 晚 ≥ 1 s 的对的比例，对 expert 0.65 报一致率，描述性。镜像题：向左 Δ_lat > τ_lat 的「借对向车道」率，> 50% 标「会冲进来车」。

Q1 的判据与读法（同一文件，写在数字之前）：

> 判据：规则 7。另报每个考生的世界级模式分布（考生自己 5 s 轨迹按第 52 条 expert 的分类规则），与 expert 的一致率。**读法（写在数字之前）**：
> - TFv6 waypoint「有 bypass」→ 第 38 条在 obstacle_bypass 格改写（第 47 条推翻条件之一）；route + target speed 通道只看纵向。
> - NAVSIM 族「有 bypass」而 openpilot 原生 plan 没有 → 第三层是 navtrain 上 PDM 子分数学得到的配方，对我们是「可移植的 R 层配方」。
> - 全部考生只有 stop 替代 → 第三层是公开方法的空白，Q2 是唯一的正例来源。
> - Alpamayo 的 CoT 说 nudge 而轨迹不绕，单列「语言与轨迹不一致率」。

2026-09-26 17:32 CST [C] 的操作性选择里，对「选择性」参照与 τ 的解释（并披露看过一个 smoke 数）：

>   **披露**：17:23 验证判卷代码时看到过一次 smoke 数（LOCO、seed 0、Cinque、A0–A3：bypass 翻转约 4%、τ_lat ≈ 3.0 m、四臂都「no bypass」）；下面的判卷参数与各臂定义在那之前已写进代码，之后没有改；A4 不进闭环候选是 17:35 因为没有导出路径加的（技术原因，写明）。

>   2. **读数**：Δ_lat = 考生自己 ego 系（后轴、左正）y(3 s) 之差；视野不到 3 s 的考生（TFv6 waypoint 8 × 0.25 s）用 y(2 s)。τ_lat、τ_lon = 天气 null 对上 |Δ_lat|、|Δv(2 s)| 的 95 分位；「天气 null 样本外误翻率」= `p5_exam` 的两半交叉（一半路线定 τ、另一半算任意方向误翻，取均值）。
>      todo 的「x₀₀ 误翻率」P6 没有第二个 x₀₀ 可比，按「无障碍参照的同向误翻率」理解：用天气 null 对的同向（expert 绕行方向）误翻率作选择性的参照，另报 x₀₀ 自身 y(3 s) 越过 τ_lat 的比例（只报）。bootstrap 10 000 次，路线整组。

### (b) 原始数字与复算

术语：bypass 翻转（考生在障碍出现窗内 3 s 处的横向位移 Δ_lat 超过 τ_lat 且方向与 expert 绕行方向一致的帧比例）；τ_lat（该考生在「只换天气」的 null 对上 |Δ_lat| 的 95 分位，即它自己的噪声地板）；
CI 是按路线整组（40 条路线）的 bootstrap；null 是样本外误翻率。出处 [summary.csv](results/nq3/q1/summary.csv) 与 [carla_rig_summary.csv](results/nq3/q1/carla_rig_summary.csv)（BridgeDrive / BLUE / SimLingo 在各自 rig 上重录，读 2 s 处而不是 3 s）。
我逐行按规则复算了门槛（CI 下界 > null + 0.10 且选择性过线），22 行的判格与 CSV 全部一致；原始逐帧数据在 box 上，未重跑 bootstrap。

| 考生 | n 帧 | τ_lat（m） | bypass 翻转 [95% CI] | 天气 null | 门槛 null + 0.10 | 判格 |
|:--|--:|--:|:--|--:|--:|:--|
| openpilot Cinque 原生 plan | 3 920 | 5.75 | 5.1% [3.6, 7.0] | 5.1% | 15.1% | no bypass |
| openpilot Lebowski 原生 plan | 3 920 | 4.80 | 4.9% [3.4, 6.8] | 5.0% | 15.0% | no bypass |
| TFv6 waypoint（2 s） | 3 920 | 1.18 | 7.3% [5.6, 9.3] | 5.3% | 15.3% | no bypass |
| TFv6 target speed | 3 920 | — | 无横向输出 | — | — | longitudinal only |
| ridge_late [Cinque / Lebowski] | 3 920 | 3.56 / 2.91 | 3.7% [2.4, 5.3] / 5.9% [4.3, 7.9] | 5.3 / 5.8% | 15.3 / 15.8% | no bypass |
| M-C pair [Cinque / Lebowski] | 3 920 | 3.29 / 2.79 | 4.1% [2.6, 5.8] / 6.1% [4.5, 8.0] | 4.8 / 5.6% | 14.8 / 15.6% | no bypass |
| E5 student B [Cinque / Lebowski] | 3 920 | 4.10 / 3.38 | 2.9% [1.8, 4.3] / 5.6% [4.0, 7.4] | 5.1 / 5.5% | 15.1 / 15.5% | no bypass |
| cls_late [Cinque / Lebowski] | 3 920 | 3.44 / 3.49 | 3.1% [2.2, 4.1] / 4.4% [3.3, 5.5] | 5.3 / 5.3% | 15.3 / 15.3% | no bypass |
| SparseDriveV2 | 3 920 | **8.49** | 2.5% [1.7, 3.3] | 5.0% | 15.0% | no bypass |
| ZTRS | 3 920 | 1.29 | 2.8% [0.9, 5.1] | 5.4% | 15.4% | no bypass |
| DrivoR | 3 920 | 3.32 | 5.2% [3.6, 7.0] | 7.3% | 17.3% | no bypass |
| WA-JEPA | 3 920 | 1.84 | 4.8% [3.0, 7.0] | 5.3% | 15.3% | no bypass |
| Alpamayo 1.5 | 3 623 | 4.76 | 5.2% [3.7, 7.0] | 5.8% | 15.8% | no bypass |
| BridgeDrive waypoint（2 s） | 3 682 | 1.17 | 9.2% [6.9, 11.6] | 6.0% | 16.0% | no bypass |
| BLUE speed waypoints（2 s） | 3 682 | 1.17 | 10.0% [7.0, 13.3] | 6.2% | 16.2% | no bypass |
| SimLingo speed waypoints（2 s） | 3 682 | 1.20 | 9.3% [6.3, 12.6] | 6.5% | 16.5% | no bypass |
| BridgeDrive target speed | 0 | — | 无横向输出 | — | — | longitudinal only |
| **PDM-Lite expert（future）** | 3 920 | 0.015 | **40.3% [34.4, 46.0]** | 6.4% | 16.4% | **has bypass** |

选择性一栏（放置 null 的 40 个 keep 世界上的翻转率 ≤ 参照 + 10 pp）20 个有横向读数的行全部为 True，没有一行被选择性门拦下。
另有一份看过 smoke 之后加的**描述性**读数 [tau_alt.csv](results/nq3/q1/tau_alt.csv)：换成「同天气、expert 保持车道的 x₀₁ − x₀₀ 对」上的 95 分位当 τ，
openpilot 原生 plan 12.3 / 10.8%（τ_alt 1.6 / 1.7 m），TFv6 waypoint 12.1%（0.53 m），ridge_late 9.2 / 11.4%，cls_late 10.4 / 11.9%，E5 student B 9.4 / 11.9%，Alpamayo 9.3%（3.28 m），expert 24.8%（0.90 m）；
所有考生的 CI 下界都在 6.8–9.9% 之间（expert 20.7%），仍低于 15% 门槛。SparseDriveV2 / ZTRS / DrivoR / WA-JEPA / M-C 没有匹配的 x₀₁ / x₀₀ keep 预测，未读。

分 scenario（[per_scenario.csv](results/nq3/q1/per_scenario.csv)）：公开考生在 8 个主类里最高的单类点估计是 AccidentTwoWays 13.0%（M-C Lebowski，n = 531）、ConstructionObstacleTwoWays 13.1%（Alpamayo）、ParkedObstacleTwoWays 11.5%（DrivoR），都不过 15% 线；expert 在同 8 类里 26–53%。

**Alpamayo 的 CoT 与轨迹不一致**（登记的单列项，[alpamayo_cot.csv](results/nq3/q1/alpamayo_cot.csv)）：3 623 个 x₁₀ 帧里 676 帧（18.7%）的 CoT 说要 nudge / 绕，其中 93.5% 的轨迹没有 bypass 翻转；没有 CoT nudge 的帧里 bypass 翻转为 4.9%。

### (c) 机器判格

「除特权 PDM-Lite expert 外，具横向读数的公开考生均 no bypass；target-speed 为 longitudinal only。Q5 选择性差 <10pp 的登记标注逐格见 todo Q5 表；保留原 CSV 的 no reaction label，不新增解释」。
（同一段里的 Q5 选择性标注不在本次复核范围，见文末备注。）

### (d) 我的检查

**规则层面：成立。** 22 行逐格复算门槛，只有 PDM-Lite 过线，其余最高下界（BLUE 7.0%）离门槛 16.2% 还差 9 pp；选择性门没有起作用；
偏离登记的两处都已披露且发生在读数之前（17:32 对「x₀₀ 误翻率」的解释；17:23 看过一个 smoke 数但参数早于它写入代码），τ_alt 是事后描述性读数、不进判格。

**可疑处**：

1. **判据对噪声大的考生没有功效。** τ_lat 是考生自己在只换天气的 null 对上 3 s 横向位置差的 95 分位。SparseDriveV2 是 8.5 m，openpilot 原生 plan 4.8–5.7 m，Alpamayo 4.8 m，DrivoR 3.3 m；
   一个绕行只有几米的横向位移，超过这个噪声地板几乎不可能。对这几行「no bypass」应该读成「在这个噪声地板下测不出」，不是「没有绕行倾向」。
   τ_alt（1.4–1.7 m）把 openpilot / ridge_late / cls_late / E5 的地板压下来后结论方向没变（下界 6.8–10% < 15%），所以对这些行结论稳；但 SparseDriveV2、ZTRS、DrivoR、WA-JEPA、M-C 没有 τ_alt，SparseDriveV2 的 8.5 m 尤其不可判。
2. **二分判格吞掉了弱信号。** 四个 CARLA 自带 waypoint 的模型（TFv6 waypoint 7.3%、BridgeDrive 9.2%、SimLingo 9.3%、BLUE 10.0%）都比各自的 null（5.3–6.5%）高 2–4 pp，下界略高于 null；
   放置 null 上它们的翻转（5.0–5.3%）也比参照（2.6–3.4%）高。这是「有一点横向反应，远不到绕行」，判格 no bypass 没错，但「均无 bypass 倾向」说得过头。
3. **「longitudinal only」是通道没有横向输出，不是实测阴性。** TFv6 与 BridgeDrive 真正驱动车的是 route + target speed 通道，那个通道没有横向读数（n = 0）；所以「target-speed longitudinal only」是对读数范围的说明，不该被读成「这两个模型不会绕」——它们的 waypoint 通道才有横向读数，那一行是 no bypass。
4. **「谁只会停」没有被判。** 登记读法第三条（全部考生只有 stop 替代 → 第三层是公开方法的空白）需要 stop 替代率对照 null 才能判，rule 7 没有给 stop 的门槛；表里的 stop 替代率 14–38%（最高 M-C Lebowski 38%）与 PDM-Lite expert 自己的 33%（它是先等再绕）同量级，说明 stop 替代分不开「等」和「停」。
   所以这一条结论目前只有「没有 bypass」，没有「只会停」的证据。
5. **覆盖不齐。** Alpamayo 3 623 / 3 920 帧（priority 0 跑完为止）；BridgeDrive / BLUE / SimLingo 3 682 帧（重录世界经 E1 等价核对后整体剔除一部分），且读 2 s 而不是 3 s。各行帧集不完全相同，不做横向排名。
6. **考生在 P6 渲染画面上是零样本**（Waymo 式三路相机 + CARLA 外观），每个考生都同时吃到 rig 与外观两层分布外；这决定了本读数只能回答「开环、零样本、这套 rig 上有没有横向反应」，回答不了「闭环里能不能绕」。
7. 40 条路线、8 个主类的 CI 已是路线整组 bootstrap，n 与 CI 口径没有问题；没有多 seed（考生都是确定性的或 seed 0）。

### (e) 建议改写（提案，未采用）

> ### 2026-09-27 新读数：Q1 P6 上读所有考生的 bypass（登记判格，待人复核）
> 规则 7，40 条路线、8 个主类、每个考生 3 623–3 920 帧（BridgeDrive / BLUE / SimLingo 3 682，读 2 s）。**只有特权 PDM-Lite expert 判「有 bypass」**（40.3% [34.4, 46.0]，null 6.4%）；
> 其余 19 个具横向读数的考生判「no bypass」，bypass 翻转 2.5%–10.0%，CI 下界最高 7.0%（BLUE），门槛为各自 null + 10 pp（≈ 15–17%）。
> **限定**：(1) 判据以考生自己的天气 null 定噪声地板，SparseDriveV2（τ_lat 8.5 m）、openpilot 原生 plan（4.8–5.7 m）、Alpamayo（4.8 m）的地板高于典型绕行位移，这几行读作「测不出」；
> 换成同天气 x₀₁ − x₀₀ 的 τ_alt（描述性）后 openpilot / ridge_late / cls_late / E5 / Alpamayo / TFv6 的翻转为 9–12%，仍低于门槛。(2) 四个 CARLA 自带 waypoint 的模型比 null 高 2–4 pp（7.3–10.0%），是弱的横向反应，不是绕行。
> (3) TFv6 / BridgeDrive 的 route + target speed 通道没有横向输出（longitudinal only），是读数范围的说明，不是「不会绕」的证据；「只会停」没有被判（stop 替代率 14–38%，与 expert 的 33% 同量级，无 null 门槛）。
> (4) Alpamayo：18.7% 的 x₁₀ 帧 CoT 说要 nudge，其中 93.5% 轨迹没有 bypass 翻转。(5) 零样本、开环、渲染画面，不代表闭环。（**待定**）

---

## 6. Q6 单帧 vs 4 帧 V-JEPA 2（decisions 第 48 条后）

### (a) 预登记原文

队列 3 Q6 的登记（159151b 首提交，8e66075 重组后措辞；判格阈值不变）：

> - V-JEPA 2 单帧对照：当前帧重复成 4 帧 clip，同一 pair-Δ，3 seed。行人翻转 < 10% → 「是时间不是视频预训练」；与 4 帧版 CI 重叠 → 「视频预训练本身」。

2026-09-26 16:50 CST [D] 的操作化（写于 Q6 任何新数字之前）里 (b) 的判格部分：

>   **(b) V-JEPA 2 单帧对照**：N6 的抽取配方原样（`features.VJepaFeatures(frames=4)`、256² 拉伸、bf16、三路 front / front_left / front_right 拼接、主 tap `mean`、副 `last_mean`），唯一改动是 clip = 当前帧重复 4 次（每个 unit 只解码当前帧一次，同一张量复制 4 份，与把同一 PIL 图传 4 次逐位相同）。
>   拟合 = `n6_backbones.fit` 原样（`ridge_late` + pair-Δ 单流 / 双流，prior = Cinque，Lebowski 只作 side，λ 网格、μ、fold 不改，`--eigh cuda`），route-fold seed 0 / 1 / 2。判格（todo 原文）：单帧版行人翻转（3 seed 均值）< 10% →「是时间不是视频预训练」；
>   每个 seed 的单帧版行人翻转 CI 与同 seed 4 帧版（N6 已存）CI 重叠 →「视频预训练本身」；两条都不满足写「部分来自时间」，seed 间不一致写「随 seed 变」。

等价检查（批量前，登记为门）：

>   等价检查（批量前）：同一驱动对 4 帧 clip 在 256 个 unit 上重抽，对 N6 已存特征报最大相对差（bf16 batch 形状噪声，N6 记 0.3–2%），> 5% 就停；同一驱动的 4 帧 seed 0 重拟合，行人翻转对 N6 已存值差 ≤ 1 帧（GPU `eigh` 的 0.5 mm 不确定性）、预测最大差 ≤ 1e-3 m，否则停。

第 48 条自己写的推翻条件（2026-09-26，第 48 条末段）：「V-JEPA 2 只喂当前帧（1 帧重复成 clip）行人翻转掉到 DINOv2 的水平（那就是「时间」而不是「视频预训练」）」。

### (b) 原始数字与复算

术语：pair-Δ（用同一场景加 / 不加行人的一对帧的特征差去拟合轨迹差，`reactivity_mc.fit_fold`）；行人翻转（反应帧上轨迹变化超过考生自己的 τ 且方向与 expert 一致的比例，n = 406 个「必须反应」的行人帧，21 条路线）；
seed 只是 route-fold 的划分与内部随机，帧与特征在三个 seed 里是同一批。出处 [vjepa_single_frame.csv](results/nq3/q6/vjepa_single_frame.csv)、[vjepa_single_frame_verdict.json](results/nq3/q6/vjepa_single_frame_verdict.json)。
主臂 = Cinque prior 的 pair-Δ 单流 mean tap（第 48 条表里 V-JEPA 2 那一行）。

| seed | 单帧 [CI] | 4 帧 [CI] | 两个 CI 重叠 | **配对差（单帧 − 4 帧）[95% CI]，我的复算** | 单帧 < 10% |
|:--|:--|:--|:--|:--|:--|
| 0 | 36.7% [28.6, 45.9] | 45.8% [36.8, 54.9] | 是 | **−9.1 pp [−15.1, −3.5]** | 否 |
| 1 | 35.2% [27.8, 43.4] | 47.5% [39.4, 56.1] | 是 | **−12.3 pp [−16.3, −8.1]** | 否 |
| 2 | 40.2% [31.3, 50.4] | 47.8% [40.1, 55.8] | 是 | **−7.6 pp [−11.9, −3.4]** | 否 |
| 3 seed 均值 | 37.4% | 47.0% | 3 / 3 | **−9.7 pp [−13.8, −5.7]** | — |

我的复算：在 box 上用 `p5_exam` 的同一口径（同一批 406 个反应帧、各考生在天气 null 上定的 τ 取自 `flip_rates.csv`、按 base 路线整组 bootstrap 10 000 次），
对**同一批帧**取「单帧翻转 − 4 帧翻转」的配对差。单帧与 4 帧的点估计逐位复现 CSV（0.3670 / 0.3522 / 0.4015 与 0.4581 / 0.4754 / 0.4778），所以只有 CI 是新算的。

旁证（同一 CSV 与第 48 条表）：

| 项 | 单帧 | 4 帧 | 备注 |
|:--|:--|:--|:--|
| cut-in 翻转（Cinque 单流，seed 0 / 1 / 2） | 77.5 / 79.0 / 79.0% | 75.6 / 78.0 / 77.5% | 车辆 cut-in 基本不受影响 |
| Lebowski 单流行人（seed 0 / 1 / 2） | 35.2 / **29.3** / 35.5% | 46.6 / 45.3 / 45.3% | seed 1 的 CI 不重叠（[20.7, 37.2] 对 [37.5, 52.3]） |
| Cinque 双流行人 | 42.9 / 41.1 / 41.9% | 47.0 / 47.5 / 47.8% | 差距更小 |
| `ridge_late`（均匀 imitation）行人 | 0.0–0.3% | 0.0% | 与第 48 条一致：单靠 ridge_late 都是 0 |
| 同表其他 backbone（第 48 条）：SigLIP2（单帧图像模型） | 33.8%（3 seed 均值，seed 极差 30.5–37.0） | — | 与单帧 V-JEPA 2 同一水平 |
| DINOv2-B | 7.1%（seed 极差 4.2–10.3） | — | 第 48 条设的「掉到这里才算是时间」的参照 |

特征层面：单帧 clip 的特征与 4 帧特征的中位相对差 31.5%，余弦 0.78（box 上 `runs/nq3/q6/sf_check.json`，未入库）。
等价门都过：4 帧路径对 N6 已存特征最大相对差 0.38%（门槛 5%），4 帧 seed 0 重拟合对 N6 已存预测逐位相同。

### (c) 机器判格

decisions 该段原文：「Q6 Cinque pair-Δ mean 主臂：单帧行人翻转 seed0/1/2=0.3670/0.3522/0.4015，4帧=0.4581/0.4754/0.4778，各seed CI重叠；原登记读法「视频预训练本身」。全部CI见 todo Q6 表，其他臂不外推」。
`vjepa_single_frame_verdict.json`：video pre-training itself (single-frame CI overlaps the 4-frame one in every seed)。它**尚未并入第 48 条的结论 2 和推翻条件**（第 48 条末段那句括注只说「已跑过、读数在下面」）。

### (d) 我的检查

**规则层面：机械上成立。** 登记的三档是「单帧 < 10% → 时间」「每 seed 的 CI 与 4 帧版重叠 → 视频预训练本身」「都不满足 → 部分来自时间」，本次单帧 37.4% ≥ 10%，三个 seed 全部重叠，落在第二档；第 48 条自己的推翻条件（掉到 DINOv2 的 7% 水平）也没有触发。
执行没有偏离登记：抽取配方原样、只改 clip = 当前帧 ×4，等价门过，3 seed 全跑。

**但登记的第二档判据太弱，配对检验推翻了它给出的读法。**

1. **「CI 重叠」不是检验。** 两个独立 CI 各自宽 ±9 pp（n = 406、21 条路线），重叠并不表示差为 0。两个版本跑在同一批帧、同一批路线上，应当做配对差：三个 seed 的配对差都是负的且 CI 不含 0
   （−9.1 / −12.3 / −7.6 pp，seed 均值 −9.7 pp [−13.8, −5.7]）。所以时间信息（4 帧里的运动）确实贡献了约 10 pp，相当于 4 帧翻转的 21%；登记里的第三档「部分来自时间」才是配对数据下的读法，机器的第二档措辞（「视频预训练本身」）忽略了这一点。
   （seed 只是 route-fold 划分，三个 seed 不是独立重复；配对差在三个 seed 里的一致性是「同一批帧的稳定性」，不是三次独立证据。）
2. **「视频预训练本身」没有被分离出来。** 单帧 V-JEPA 2（37.4%）与单帧 SigLIP2（33.8%，seed 极差 30.5–37.0）同一水平，而 DINOv2-B（7.1%）低得多。这个结果能说的是「语义 / 预训练得好的特征在单帧上就够 30–40%，DINOv2 不够」，并不能说这是视频预训练带来的。
   V-JEPA 2 与 SigLIP2 之间除了「视频 vs 图像」，还差预训练目标（重建 vs 图文对比）、数据与模型规模。
3. **4 帧与 1 帧的差异不只是「时间信息」。**（i）重复 4 次的静态 clip 对视频模型是分布外输入：训练里没见过零运动的 clip，特征与 4 帧特征余弦只有 0.78、中位相对差 31.5%，掉下来的 10 pp 里有多少是「没有运动信息」、有多少是「输入离分布」，这个对照分不开。
   （ii）mean-pool 在 4 帧下平均了 4 个时刻的 token，单帧没有这种时间平均带来的降噪。（iii）pair-Δ 的一对特征在 4 帧下包含了「行人进入视野的过程」，单帧只有当前时刻；这也是「时间」，但与「视频预训练」是两回事。
   一个能干净分离的对照需要「单帧输入的图像模型 vs 同规模的视频模型」或「打乱帧序 / 换成时间上有真实运动但去掉顺序的 clip」，本轮没有。
4. **DINOv2 的比较有规模混杂。** 第 48 条里 DINOv2 是 ViT-B（350 × 322 输入、`patch_mean`），V-JEPA 2 是 ViT-L（256² 拉伸、`mean`）；把 7.1% 当作「单帧特征的下限」隐含了「规模与输入分辨率不影响行人翻转」这个没验证的假设。
5. **只在主臂上下结论。** Lebowski 单流 seed 1 的单帧 29.3% 与 4 帧 CI 不重叠（该臂机械上应判「随 seed 变」）；decisions 段已注明「其他臂不外推」，这一点写得对。cut-in 翻转单帧几乎不变（77–79% 对 76–78%），说明这个 4 帧优势只出现在行人这一族，与「运动线索对行人更重要」一致，是推测，未验证。
6. 与第 48 条的关系：第 48 条结论 2 写的「V-JEPA 看 4 帧、图像模型只看当前帧，视频预训练与有时间没有分开」——单帧对照做完后，这句话可以更新为：单帧保留 79%，配对下显著低于 4 帧，仍未与图像模型分开。

### (e) 建议改写（提案，未采用）

> ### 2026-09-27 新读数：Q6 V-JEPA 2 单帧对照（登记判格，待人复核）
> 当前帧重复成 4 帧 clip，其余同第 48 条（Cinque prior、pair-Δ 单流 mean tap，P5 v1 BA，n = 406 个行人反应帧，21 条路线，3 个 route-fold seed）：行人翻转 单帧 36.7 / 35.2 / 40.2%，4 帧 45.8 / 47.5 / 47.8%。
> 登记的机械判格：单帧 ≥ 10%，且三个 seed 的两个独立 CI 都重叠，登记读法「视频预训练本身」（不是「时间」）。
> **配对复核**（同一批帧、同一 τ、路线整组 bootstrap，复算读数）：单帧 − 4 帧 = −9.1 [−15.1, −3.5] / −12.3 [−16.3, −8.1] / −7.6 [−11.9, −3.4] pp，三个 seed 都显著为负，seed 均值 −9.7 pp [−13.8, −5.7]（seed 只是 route-fold 划分，不是独立重复）。
> 所以单帧 V-JEPA 2 保留了约 79% 的行人翻转（37.4 / 47.0），第 48 条的「掉到 DINOv2 的 7%」推翻条件未触发；但有约 10 pp 需要多帧，「是时间不是视频预训练」和「全是视频预训练」两个极端都不成立，更接近登记的第三档「部分来自时间」。
> **限定**：(1) 单帧 V-JEPA 2（37.4%）与单帧 SigLIP2（33.8%）同一水平，视频预训练本身没有与图像—文本预训练分开；DINOv2-B（7.1%）与 V-JEPA 2 ViT-L 还差规模与输入分辨率。
> (2) 静态重复 clip 对视频模型是分布外输入（与 4 帧特征余弦 0.78），单帧 − 4 帧的差混着「没有运动信息」与「输入离分布」。(3) 只对 Cinque pair-Δ 主臂下结论；Lebowski 单流 seed 1 的 CI 不重叠。
> (4) cut-in 翻转单帧几乎不变（77–79% 对 76–78%），4 帧的增益只出现在行人（推测：运动线索，未验证）。
> **第 48 条结论 2 可同步改写为**：「V-JEPA 2 与 Qwen 同一水平；单帧对照显示其行人翻转约 79% 在单帧上就有，约 10 pp 依赖多帧；与图像模型（SigLIP2）单帧同一水平，视频预训练本身仍未分离」。（**待定**）

---

## 备注

- 第 4 项那一段 decisions 里还夹着一句 Q5 选择性标注（P5 v1 BA 上 SparseDriveV2 / ZTRS / DrivoR / WA-JEPA 的「反应帧翻转 − 非反应帧误翻」为 3.6 / 4.4 / 2.2 / 7.8 pp，标「见车就减速」，I3 上 14–35 pp），不在这次的三项里，我没有复核。
- Q6 的另一半（V-JEPA 2 进快通道候选：WOD 与 NAVSIM 的配对差 CI 都为负，两个模型 6 / 6 seed 判 harmful）不在这三项里；它与第 48 条「会推进本条的证据」第二项直接相关，若第 48 条要一起更新，可以并在一起。
- 我在 box 上的复算脚本是一次性的（放在会话临时目录，不入库）：Q5 是读 devkit `v1_navtest_nq3q5_*` 与 `v1_navtest_cv` 的逐 token 分数；Q6 是对 `runs/nq3/q6/fits/sf-seed*/preds_obs.npz` 与 `runs/night2/n6/fit-seed*-cuda/*/preds_obs.npz` 用 `p5_exam` 的 τ 与翻转定义重算配对差。需要的话可以整理成 `scripts/` 里的正式脚本。
