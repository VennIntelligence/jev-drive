# 路口眼：执行稿（交协调者与三个执行者）

2026-10-14 定稿：main 与用户三轮讨论后重写，取代 10-13 稿（旧稿在 `tmp/junction-eyes-plan-2026-10-13.md.bak`，以及 [junction-eyes.md](junction-eyes.md) 的方案部分；那份稿子的第 1、2、7 节仍是证据背景）。
状态：**已定，待协调者建 topic、派三个执行者（第 8 节）。** 本稿没有提交任何 job，没有改代码。
记号同 plan.md：d### 指 `research/decisions/###.md`；「测」有记录读数，「推」由测得的数算出，「估」没有测量依据。盒上路径以 `$DATA_DIR` 起（10-14 为 `/root/autodl-tmp/ujs`）。

## 0. 一句话

**用地图标签教 Cinque 自己的眼睛看路口：先换尺子（第 0 步，CPU 为主，可能直接关线），再只训眼睛（第 1 步，一夜），过线才碰 policy（第 2 步）。**
训练行 = navtrain 真实帧 + AlpaSim train 划分的 on-policy 重建帧，另训一只只用 navtrain 的眼作 drop-one 与论文可报臂。
读数以榜为目标：AlpaSim held 452 的 18 决策零分与横向 GT、navtest / navhard；HUGSIM 与 WOD 只要求不退。跑过的臂都不过第 1 步真实帧上的判线，眼睛线关闭，不开第二轮。

## 1. 用户已定的事（不再讨论）

10-13：
1. 教 **Cinque 自己的眼**（LoRA 为主，整段解冻作对照臂）；不做从零小眼（它是 nuPlan 专用感知模块，不是学生的能力）。
2. 关闭规则：第 1 步的臂都不过判线，眼睛线关闭，不开第二轮。
3. 侧视图不作默认：冻结编码器读侧视已死（d248 RW0 精确为 0，d255 负）；侧视只以「训过的读图器 + 前视填不满的目标」进入，作臂。
4. 本稿不碰正在跑的其他任务；任务安排由用户另定。

10-14：
5. **论文不急（11 月后），涨分优先。** Z-0 判「冻结眼在公平尺子上看得见」即关线，不为论文单独跑第 1 步。HUGSIM 只作不退线（d138：它的差距是卡死 / 打转 / 起步，不是路口眼）。
6. 第 1 步判线的**数值**在第 0 步读完 Z、blind、WA-Cf 之后、第 1 步任何训练之前写进预登记 B；加 **WA-Cf 同款探针作参照行**（只参照，不进方法与 driver）；判线**不过严**。
7. 训练行**加 AlpaSim train 划分的 on-policy 行**（1 020 scene、28 个 log 单元）；held 452 与 98 个零分窗口一行不进；L-F0 加 drop-one 臂 **L-F0-nav**（只 navtrain）。comma1M 没有任何几何标注（10-14 盒上核对），只作保持。
8. 第 2 步 navtest / navhard 的基底是 **P2H10S**（d255 的 (c)），不是 SH30；AlpaSim 侧用 **YS-L18 配方，不叠路线支路**（d282 已判支路被策略忽略）。
9. 第二轮与退路先不议；先把第 0 步与工程项落下去。

## 2. 为什么要先修尺子（证据）

- d275 / rp_real（`experiments/road_probe/results/rp_real.md`）：在「20 m 固定横线上的 ego 车道边界」这把尺子上，不看图的 blind 头在路口内也从 0.64 m 跳到 1.47 m，探针 0.46 → 1.18 m；标签在 navtest 转弯里只有 33 % 有定义，98 个零分窗口只有 45 个可评。路口鼓包大半是标签难度；token 在路口内比 blind 多看出的只有 0.29 m（路口外 0.18 m）。
- 几何事实：wide 槽 58.7°（焦距 455，512 px）在 20 m 处只盖左右各 11.3 m，road 槽 31.4°。90° 转弯的出口车道在 20 m 横线上多半在画面外。旧尺子可能在要求眼睛看见画面里没有的东西。
- d245：三种载体都把航向误差压低而 DAC 失败率不动，失败从转不过去翻成切内侧。policy 缺的不是「弯在哪结束」，是「路沿和对向车道在哪」。所以尺子要量近处走廊，不量远处一个点。
- d275 的限定：训练的 on-policy 行排除了离开 corridor 的状态，零分窗口的状态在训练里少见。这也是 10-14 把 train 划分的 on-policy 行加进训练的理由（4.2）。
- 结论：**尺子先换，判定线建在新尺子上。** 旧尺子（横线）保留为描述行，便于与 d275 对照。

## 3. 第 0 步：新尺子（CPU 为主，不训练 encoder；这一步本身就可能关线）

### 3.1 三把尺子

**R-A 弧长尺（route-conditioned）。** 车道序列 = 日志路径逐点所在的车道（含 connector）按顺序去重得到的序列；navtrain / navtest 用 trajdata MapAPI（`rp_real.py` 的 `lab_log` 同一批元素，d240 / d243 的 M 标签做法），AlpaSim 侧用 `lm.py route_regen` 的 RouteGenerator（MAP）把 `route_start_offset_m` 设为 0 取稠密车道中心折线，或同一 MapAPI 做法，两侧必须同一语义（G-lab 比对一批共同 scene）。
从相机位置在该序列中心线上的**投影点**起量弧长 s ∈ {5, 10, 15, 20, 30} m，取中心点 C(s) 与该处车道沿法向的左右边界点，全部表示为自车后轴系 (x, y)，y 左正。标签 5 × 3 点 × 2 = 30 个数。误差 = 预测点与标签点的欧氏距离（m），逐决策取中位。车离开日志（on-policy 行、零分窗口）时标签仍由日志路径定义，与 d275 一致。它几乎处处有定义，问的是「我的车道往哪走」。

**R-B 近处栅格尺（route-free）。** 自车后轴系 BEV 栅格：x ∈ [0, 24) m，y ∈ [−12, 12) m，0.5 m 格，48 × 48。三类：0 出路（ROAD_AREA 之外）、1 可行驶（ROAD_AREA 之内且非 2）、2 对向（非路口车道，且车道朝向与自车朝向差 > 120°）。路口内一律记 1（connector 方向混杂，不判对向）。读数：三类各自 IoU；**主量是出路类与对向类在 0–20 m 内的 IoU**，分桶：路口内 / 路口前 0–20 m / 20–60 m / 无路口，直行与转弯分开；路口内只报出路类。这把尺子量的正是 S 课训练期用的那张栅格的近处部分（d268 臂 B 的带方向栅格）。

**R-C 视场核查（只算几何，不训练）。** 对三组读数集的转弯决策：R-A 标签点在 s = 10 / 15 / 20 m 的相机方位角，|方位角| > 29.4°（wide 槽外）与 > 15.7°（road 槽外）的比例，按到路口距离分桶；R-B 里出路 + 对向格在 0–20 m 内落在 ±29.4° 之外的比例。它给出 F0 臂的天花板，并**决定第 1 步跑不跑视场臂**（4.1）。

### 3.2 行与读数集（定义一次，全文引用）

| 名字 | 内容 | 帧 | 在哪 | 用途 |
|:--|:--|:--|:--|:--|
| **J_nav** | navtrain 里路线到路口 ≤ 60 m 或 40 m 内路线转角 ≥ 10° 的 token，加等数量随机直行 token（估 5–6 万行；按 `rp_real.py` 的 `jdist` / `turn40` 列） | 真实帧，t0 图像对（协议 W 的 t0 槽，`vt_px.py`） | 像素已删，重渲（5.1） | 训练 + 探针行 |
| **J_on** | AlpaSim train 划分（`alpasim_nuplan/onpol-train`，1 020 scene / 28 个 log 单元）三个 label 的 18 决策 on-policy rollout 里 jdist ≤ 60 m 或转弯的决策，三个 label 的行都保留，按 scene 均匀抽（估 2–3 万行） | 重建帧；编码对 = (slot 6, slot 7)（`on_rows.py make_front` 的配对） | `$DATA_DIR/runs/onpol_long/roll/{YR10m10-F-s0,YS-L18-s0,YS-L18-s1}/c*/pool/*.npz`（34 GB，10-14 在盘）；d275 的 token 缓存 `runs/op_parity/cache/on1_{L18Y,L18YS}@warp/front.npy` | 训练 + 探针行 |
| **J** | J_nav ∪ J_on | | | 默认训练行 |
| **val** | J_nav 里 sha256 取模约 10 % 的 drive log，J_on 里 3 个 log 单元 | | | 选 epoch、选探针 step；不进拟合 |
| **P** | J 去掉 val | | | 探针训练行（Z 与所有臂同一个 P） |
| **保持集** | comma1M 602 段，`fov_replay.py scan` 的转弯 / 直行事件各半，约 20 k 个 t0 图像对；**留出 60 段**不进训练 | 原生 1.22 m 相机，原生两路焦距（`frames.segment_model_frames`） | `$DATA_DIR/datasets/comma1M/<seg>/{fcamera,ecamera}.hevc + frame_info + localizer`；没有几何标注 | 防忘记原生相机；E5 |
| **读：navtest-16** | navtest 里 16 个 held drive log 的 token（`rp_real.py` 的 held units 映射；转弯决策 1 542 个）；**L-F0-nav 另读全部 12 146 个** | 真实帧 | 像素重渲（5.1） | 真实帧判线 |
| **读：held-452** | AlpaSim held 452 scene 的 8 129 个 18 决策决策 | 重建帧 | `$DATA_DIR/runs/road_probe/{roll,tok}/`（d275 的 held 重跑带帧 dump） | 重建帧判线、第 2 步入口 |
| **读：zero-98** | d272 的 98 个出路零分窗口 × 5 决策 | 重建帧 | `$DATA_DIR/runs/eyes_vs_head/msgs/`（driver 消息含 JPEG，`rp.py replay` 重放出 token） | 描述与 BEV 样例 |

读数集的 scene、log、token 一行都不进 J、P 或保持集（G-split）。

### 3.3 读数、探针与判线（写进预登记 A，push 后才能读任何数）

**探针**：`experiments/corridor/scripts/head1_train.py` 的 `build()`（3 层 encoder + 2 层 query decoder，d 256，3.4 M 参数）原样 import，读冻结 token（8 slot × 32 × 512）+ 20 维 ego（含 command）。
- R-A：`nq = 15, nc = 2`（15 个点的 (x, y)），目标按 P 的均值 / 标准差标准化，Huber δ = 1，缺失屏蔽。
- R-B：`nq = 36, nc = 192`：36 个 query 各管一个 8 × 8 格的块（6 × 6 块铺满 48 × 48），每格 3 类 logit，交叉熵，reshape 回 48 × 48 × 3；多出来的参数只在输出层，< 0.1 M。
- 训练：8 000 step、batch 256、lr 3e-4、wd 0.05、warmup 300，每 500 step 在 val 上取最好；**2 seed，登记读数 = 两 seed 预测的平均**，单 seed 另报。行 = P。
- **blind 底**：同头 `vision=False`（只 ego + command），1 seed。
- **WA-Cf 参照行（只参照，不进方法）**：`experiments/op_probe/scripts/opb_wajepa.py` 的 Cf token（前视、最新 tubelet、4 × 4 池化 → 32 × 512，与 view_39 同形）放进 slot 7、其余 slot 置零，用 `build()` 的 `keep` 只读最后 32 个 token；**同口径配一行「Cinque 只读 slot 7」**，两者才可比。WA-Cf 特征只在真实帧上（J_nav 的 navtrain token 训、navtest-16 读；不碰重建帧与 driver）；缺的约 77 k navtrain token 按 d160 估 20–30 卡分钟。限定写明：WA 的 nuPlan 视频 SSL 是否含 navtest log 未查证（d160）。
- **旧尺子描述行**：d275 的横线尺在同一个 P 上重训一次，复现 rp_real 的 held 转弯 0.90 / navtest 近路口 1.18（G-repro）。

**读法**：三组读数集；按到路口距离分桶（沿日志路径的弧长，`ev.py Geo.junction`：路口内 / 0–10 / 10–20 / 20–40 / 40–60 / 60 m 内无），转弯（`turn40 ≥ 20°`）与直行（< 5°）分开；中位数与 IoU 的 95 % CI 用聚类 bootstrap（navtest 按 log 段，AlpaSim 按 scene），B 2000。

**闸**（全过才读判线）：
- G-lab：R-A 在三组读数集的转弯决策上有定义的比例 ≥ 0.9；符号检查（后轴系里左边界 y > 中心 y > 右边界 y）；navtrain 侧与 AlpaSim 侧的标签器在 20 个共同 log 的同一时刻上互差中位 < 0.2 m。
- G-dump：抽 500 个 J_on 决策，npz 里 8 槽帧完整，(slot 6, slot 7) 对经 `pp_prep.encoder` 的 token 复现 d275 缓存 `front.npy` 到 fp16 噪声。
- G-split：读数集与 J / P / 保持集零交集（scene、log、token 三级查）。
- G-repro：旧尺子描述行的读数在 rp_real 两 seed 的范围内。
- G-ref：WA-Cf 新抽特征在 200 个 navtest token 上复现 d160 的缓存（`representation.md` 的等价检查）。

**判线**：
- **Z-0 关闭线**：冻结探针 R-A 在 s = 20 m、路口内 + 路口前 0–20 m 桶、**held-452 转弯与 navtest-16 转弯两组中位都 < 0.5 m**，且 R-B **出路类** IoU 路口内 / 20–60 m 之比 ≥ 0.8 → 判「冻结的眼在公平尺子上看得见」，**眼睛线不开**；读数记入知行差距一侧（眼睛看见、policy 出路），进 decisions。
- 否则进第 1 步。无论哪种，R-C 的比例进决策条，并按 4.1 定臂的集合。
- 探索（读后、不进判词）：各桶的 blind 与 WA-Cf 行、R-A 按 s 的曲线、R-B 的 BEV 样例四格。

### 3.4 代价（估）

标签：J_nav 全部 navtrain 候选 + J_on + 三组读数集的 R-A / R-B 标签，CPU 全盒并行 1–3 h（横线标签 rp_real 已跑过同规模；栅格是新增的多边形栅格化，shapely / rasterio 一类）。探针：R-A / R-B × 2 seed、blind、WA-Cf 两行、旧尺子行，约 10 个 × 10 分钟 GPU；WA-Cf 特征 20–30 卡分钟。报告半天。**合计一个 Opus 执行者约一天，GPU < 2 卡时。**

## 4. 第 1 步：只训眼睛（一夜，6 卡）

### 4.1 臂

所有臂从出厂 Cinque 权重出发，训练目标 = R-A 回归 + R-B 栅格（两个头接在 `view_39` 之上：每帧 32 × 512 的视觉主干输出，adapter 读的就是它；出处 `jevdrive/op_torch.py` 模块 docstring 与 `jevdrive/op_adapt.py`）+ 保持项（4.3）。**不加 plan loss，不动 policy、不动 adapter。**

| 臂 | 视觉权重 | 输入 | 训练行 | 问什么 | 对照 |
|:--|:--|:--|:--|:--|:--|
| **Z** | 不动 | F0：road + wide 槽，wide 58.7° | （第 0 步的冻结探针本身） | 对照 | — |
| **L-F0** | 整段主干 LoRA r 16、alpha 16，**含 Conv**（5.2） | 同 Z | J（+ 保持） | 眼睛学不学得动；默认候选 | Z |
| **L-F0-nav** | 同 L-F0 | 同 Z | J_nav（+ 保持），步数与 L-F0 相同 | on-policy 重建行值多少；论文可报的眼 | L-F0、Z |
| **U-F0** | 整段主干解冻，lr 1e-5，视觉组无 weight decay | 同 Z | J（+ 保持） | LoRA 容量够不够；遗忘对照 | L-F0 |
| **L-W90**（R-C 定） | 同 L-F0 | road 槽不变；**wide 槽 90°**（焦距 256）由 CAM_L0 / F0 / R0 纯旋转拼成（5.1）；**保持行的 wide 槽也从 comma 120° 鱼眼 warp 到 90°**，蒸馏目标仍是出厂模型在 58.7° 帧上的输出 | J_nav（J_on 没有侧相机）+ 保持 | 学生生来就有的那只眼看宽一点够不够 | **L-F0-nav**（同数据） |
| **L-3V**（R-C 定） | 同 L-F0，加 3 个相机 id 嵌入（加在 view_39 token 上，trunk 本身不知视图） | F0 + L0 + R0 各走一次主干（`vt_side.py` 的渲染），token 拼接进头 | J_nav + 保持 | 侧视图在训过的读图器下是否变正 | L-F0-nav |

- **臂的集合由 R-C 定**：L-F0、L-F0-nav、U-F0 必跑；L-W90、L-3V 只在 R-C 说 s = 15 / 20 m 的 R-A 标签点**多半**（> 50 %）落在 ±29.4° 之外时才渲染、才跑；否则不渲染这两条路径（各省 40–90 GB 与约一天工程）。R-C 说 20 m 多在视场外时，F0 臂的 G1 只在 s = 15 上判。
- 各 2 seed：必跑 6 个 job，加视场臂 10 个。同 seed 的臂用同一行顺序、同一保持行流。
- **视场臂的限制写明**：J_on 与 held-452 的 dump 只有 F0 的 road / wide 槽，没有侧相机，所以视场臂只在真实帧上判（navtest-16），没有重建帧读数；它们若赢过 L-F0-nav 且要进第 2 步的 AlpaSim 侧，需要新的 driver 渲染路径（模拟器每决策本来就渲 8 相机），估一天，只在赢了之后做。

### 4.2 训练行与 batch

- 默认 J = J_nav ∪ J_on（3.2）；每 batch 64 = **24 行 J_nav + 24 行 J_on + 16 行保持**（L-F0-nav 与视场臂：48 行 J_nav + 16 行保持）。J_on 按 scene 均匀抽（一个 scene 的行共享一个权重），不让长 rollout 多占。
- 步数：12 epoch of J（推：J 约 7–9 万行，48 行 / step，约 20 k 步）；**所有臂同一步数**（L-F0-nav 因此是约 17 epoch of J_nav）。
- **选 epoch**：每 1 000 步在 val 上算 R-A / R-B 的 loss，取最好的 checkpoint 作该臂的读数对象；navtest-16 与任何读数集不参与选择（d243 的 fold 规则）。15 epoch 级的 LoRA 几乎必然过拟合，没有这一条，G1 读到的可能只是末 epoch 的运气。
- 两个数据源的帧协议不同（navtrain 协议 W 的 t0 对；J_on 是 driver 实际喂的槽对），不做对齐：两者都是这只眼在各自板上实际吃的输入。

### 4.3 保持项（防遗忘，comma 数据，无几何标注）

- 行：3.2 的保持集，20 k 个 t0 图像对，分段抽样，60 段留出。
- 目标：出厂 Cinque 在同一帧、**带真实 20 Hz 历史**（`op_adapt.Stepper`，modeld 式步进）的**全部输出**（plan、lane lines、road edges、lead、pose、desire state ……）的蒸馏；教师输出离线算一次并缓存。学生侧：8 个上下文帧对过当前 trunk 无梯度，t0 对带梯度，再过**冻结的出厂 policy**（VT 臂 C 与 d145 U2 的做法），各头按 vt.py 的 `lam_d` 口径蒸馏，**plan 头与其它头同权**（vt.py 原来只蒸非 plan 头；这里改的是眼睛，plan 头必须一起保）。
- smoke 上核对：保持行的蒸馏梯度范数与 J 行头损失的梯度范数同量级（d245 的辅助梯度只占 1–5 % 是失败样例）；不同量级就调权重，调完写进预登记 B。
- 这不是 10-10 关掉的「眼睛课 / 多身体」线：保持项不教尺度不变性，只防忘记原生相机。L-W90 的保持行 wide 槽 warp 到 90°（4.1），是为了同一个槽只有一种内参，d137 的 W116 就是混合视场的失败样例。

### 4.4 训练超参（所有臂共用，差别只在 4.1 的列）

- LoRA：r 16、alpha 16、B 零初始化（起点逐位等于出厂），lr 3e-4（d145 U1L 的 LoRA lr），无 weight decay；depthwise conv 直接放开（5.2），lr 1e-5。U-F0：视觉全部 lr 1e-5，无 weight decay（VT 预登记第 3 点的理由）。两个头：lr 3e-4、wd 0.05。
- warmup 300 步，cosine 到末步；bf16 autocast（op_torch 支持）；梯度裁剪分组各 1.0（视觉 / 头）。batch 64。
- **恒等闸**（训练前）：LoRA B = 0 且 depthwise 位移为 0 时，512 帧的 view_39 与冻结前向逐位相等（bf16 下 fp16 噪声内）；像素路径复现缓存 token（vt_px 的 `check`）。
- staged：先 1 个 300 步 smoke（恒等、吞吐、两类梯度范数、保持行 loss 下降）→ 全部入池。smoke 不读任何判线量。

### 4.5 读数协议与判线（数值写进预登记 B：第 0 步读后、第 1 步任何读数前 push）

**读数协议**：每个臂取 4.2 选出的 checkpoint，重新编码 P 与三组读数集的 token（J_on、held-452、zero-98 的槽对按 dump 配对），**在 P 上重训一只 HEAD1 探针**（3.3 同头、同超参、同 seed 规则，2 seed 平均），在读数集上读；这去掉联合头过拟合自己 encoder 的混杂。臂自带的联合头读数作描述行。Z 的读数 = 第 0 步的探针。全部按决策配对、聚类 bootstrap B 2000。

- **G1 教进去了**（主判线，对 Z）：R-A 在 s = 15 与 20 m（R-C 说 20 m 多在视场外时 F0 臂只判 15）、路口内 + 路口前 0–20 m 桶、转弯决策。**真实帧组（navtest-16）决定「可教 / 关闭」；重建帧组（held-452）只决定能否进第 2 步的 AlpaSim 侧。** 两组分开判：LoRA 若只在真实帧上过，那是域差，不是教不动。形状：臂 − Z 的中位差 CI 上界 < 0，且中位减少 ≥ **max(0.10 m, (Z − WA-Cf 参照) / 3)**；并且 R-B 出路类 IoU（0–20 m，路口内）臂 − Z 的 CI 下界 > 0。数值（含 WA-Cf 参照差）在第 0 步读完后填进 B。
- **G2 没坏**：(a) 直路 R-A s = 20 m 臂 − Z 的 CI 上界 < +0.05 m；(b) **E5** comma1M 留出 60 段：plan ADE@4 s（对 localizer 位姿）臂 / 出厂 ≤ 1.05，action 曲率增益 A_act 与出厂差在 ±0.05 内（口径沿 `experiments/op_wide_ft` 的 comma 读数与 `op_fov` 的 A_act；d137 的 ×2.6 是失败样例）；(c) **WOD 金丝雀**：过 G1 与 (a)(b) 的臂，导出合并权重的 ONNX（5.3）跑 `scripts/wod_zeroshot_openpilot.py --onnx` 的 WOD-E2E val 零样本，对出厂 8.005（d183）的配对 RFS 差 CI 下界 ≥ −0.20。
- **G3 数据轴与视场轴**（描述为主，定第 2 步用哪只眼）：L-F0 − L-F0-nav 在 G1 两个量上的差与 CI（on-policy 重建行值多少，真实帧与重建帧两组都读）；L-W90 − L-F0-nav、L-3V − L-F0-nav 同口径，另在 R-C 判为「标签在 ±29.4° 之外」的决策子集上单独读。
- **关闭规则（用户已接受）**：跑过的臂都不过真实帧组的 G1 → 眼睛线关闭，不开第二轮。判词三选一，由读数定：WA-Cf 参照也不优于 Z → 「单帧前视像素里没有」；WA-Cf 优于 Z 而所有臂不过 → 「Cinque 的眼睛教不动」；R-C 说标签多在视场外且视场臂未跑或不过 → 「几何不在视场里」。
- **进第 2 步**：至少一个臂过真实帧 G1 + G2 → 带它进第 2 步。navtest / navhard 侧任何过线臂都能进；AlpaSim 侧只有同时过重建帧 G1 的 F0 臂能进。多个过线按 G1 的 s = 20 m 减少量选；视场臂与 F0 臂分不开时取 F0（省 serving 工程）。

### 4.6 代价（估，按 VT 实测外推，±50 %）

| 项 | 量 | 时间 |
|:--|:--|:--|
| 训练 | 约 20 k 步，F0 臂只过一次主干，估 4–6 it/s（VT 三视图带 policy 与锚行 3.3 it/s，测）；U 臂全反传约 1.3 × | F0 / nav 臂 60–90 min，U 臂约 2 h，3V 臂约 3 h |
| job | 必跑 6 个，视场臂加 4 个 | 6 卡一夜（推） |
| 读数 | 每臂重编码 P + 三组读数集（约 10 万帧对）+ 2 只探针 | 每臂约 20 min；全部 2–3 h |
| E5 / WOD | 过线臂各一次 | E5 分钟级；WOD 每臂 ≤ 1 卡时 |

## 5. 工程项（与第 0 步并行；估两到三个工作日，分给执行者 ② ③）

### 5.1 像素
- **navtrain 前视 t0 图像对**（`runs/vis_train/px` 已清）：给 `vt_px.py build` 加 t0-only 选项，只出 (N, 2, 2, 6, 128, 256)，786 KB / 行；J_nav 约 5–6 万行 ≈ 45 GB，navtest 12 146 行 ≈ 10 GB（旧稿测：24 核 169 s）。像素路径以 `vt_px.py check` 复现缓存 token。
- **宽槽 90°**（R-C 之后才做）：CAM_L0 / F0 / R0 纯旋转重投影到 `frames.wide_K(256)` 的 512 × 256 帧（`experiments/op_parity/scripts/wa_xboard.py reproject` 的三相机纯旋转重投影；`vt_side.py` 的三相机标定；三视图合起来 ≥ ±82.4°，无缝，测）。竖直方向 NAVSIM 约 38.6° 填 53.1° 的约 73 % 行，其余填黑，所有行一致。只存 wide 帧（road 帧与 F0 共用），≈ 45 GB。comma 保持行的 wide 从 120° 鱼眼直接 warp 到同一 K。
- **三视图**（R-C 之后才做）：`vt_side.py build` 对 J_nav（1.57 MB / 行，≈ 90 GB）。
- **保持集**：流式解码（一分钟 1928 × 1208 解码 8 GB，不落盘原图），只存模型帧，≈ 16 GB；教师输出缓存（Stepper）另存。
- **J_on / held / zero-98**：帧已在盘，只做 G-dump 的核对与槽对抽取。
- 盘：10-14 核对余 988 GB；全部缓存最多约 210 GB；每次 build 前核对余量 ≥ 300 GB，否则缩 J_nav。

### 5.2 Conv 的 LoRA（按层型）
`jevdrive/op_torch.py` 的 LoRA 现只接在 MatMul / Gemm 上。移植图里 Cinque 的视觉主干是 **ConvNeXt 式**（unfreeze 预登记：stage 4 = 3 个 ConvNeXt block、2048 通道 4 × 8；d145 的 U1L 就是 stage 4 的六个 MLP matmul），stage 1–3 全是 Conv，只做 MatMul 的 LoRA 等于又只动 stage 4（d145 已判无效）。给 `op_Conv` 按层型加可训练项：
- 1 × 1 pointwise conv：等价 Linear，LoRA 在 (out, in) 上，W += A @ B · alpha / r，reshape 回 (out, in, 1, 1)。
- 下采样 / stem 的 k × k conv：LoRA 在 (out, in·k·k) 上，reshape 回卷积核。
- 7 × 7 depthwise conv（group = in）：每层只有 C × 49 个参数，**直接作为可训练参数**（fp32 主副本，初值 = 出厂），不做低秩。
- 等价检查：B = 0 且 depthwise 位移为 0 时逐位等于冻结前向（4.4 的恒等闸）。半天。

### 5.3 导出与 serving
- LoRA 合并：W' = W + A @ B · alpha / r（Conv 按 5.2 reshape），depthwise 直接替换；写回出厂 ONNX 的同名 initializer，其余图不动；**不改 Cinque 一个字节的 base 文件**，合并只发生在导出副本上。
- 等价闸：torch 口径与合并 ONNX 在 512 帧上 view_39 的差在 fp16 噪声内、plan xy ≤ 0.5 m（沿 `adapt_h` 族的等价闸）；B = 0 的合并 ONNX 对出厂 ONNX 的 plan 逐位相同（eager，d282 G-serve 的口径）。
- 第 1 步只需要这个导出给 WOD 金丝雀与 E5；第 2 步才需要 bench 的 navtest / navhard / HUGSIM 与 AlpaSim driver 吃新 encoder（bench `vt` 族已有「自带 encoder 的臂」的路径，`VTCP2` 曾导出到 HUGSIM；AlpaSim driver 按 `JEV_TAG` 选 checkpoint，encoder 随 ONNX 替换）。

## 6. 第 2 步：policy（只在过线后；另写预登记 C，此处只定形状）

- 新眼睛**冻结**，重算 token bank（navtrain、navtest、navhard；S 的 hinge-only 离轨行；AlpaSim 的 NAVSIM 行、yaw 行、on-policy 池），policy 与 adapter **从出厂权重联合训**（d248 范围声明：不续训已训好的 policy）。
- **navtest / navhard**：`P2H10S`（`experiments/body1/scripts/bd4_train.py` 的配方，10 k × 128，2 seed），对照 = 同配方、旧眼睛（存档 `P2H10S-F-s{0,1}`）。新眼接上去就是 d255 那张 2 × 2 的第三列：同一只眼 + 车身课，对 换眼 + 车身课。**前置核对**：S 的 hinge-only 离轨行（`bd4`）要重编码，先确认它们的像素还在；不在则按 body1 的重投影路径重渲，或先只跑 `P2H10`。
- **AlpaSim**：按 YS-L18 的两段原样重走——从出厂权重训 `YR10m10-F` 配方（10 k × 128）→ 在同一 on-policy 池（`on1_L18Y`，帧重编码）上续训 S 两项（YS-L18 的步数与 lr），2 seed；对照 = 旧眼的 `YR10m10-F` → `YS-L18`。driver 吃合并 ONNX（5.3 的 G-serve 先过）。**不叠路线支路**（d282）。
- 读法按 d245 / d280 的规则：navtest / navhard 的 > 20° 与 > 45° DAC 失败率、切内侧 / 转不过去的翻转数、EPDMS 护栏（L-F0-nav 的眼可报全板；默认眼只报 navtest-16 与 navhard 里不含 train 划分 log 的部分）；AlpaSim held 452 的 18 决策零分三类、横向 GT（过终点部分单列）、at-fault km，选模型按 plan.md 3.3；WOD 零样本不退；HUGSIM 64 `spec` 不退（bench）。**不读航向误差**。
- 代价：bank 约 3 min / 套（测，VT）+ on-policy 池重编码约 4 min / 2 万行（测）；训练约 40 min / 个；bench 读约 1 h / 板（测）。

## 7. 本稿不做的事（有意，不要补）
- 从零小眼；冻结编码器的侧视 token；任何 heading 误差作闸门或代理量；续训已训好的 policy 来评眼睛；把 HUGSIM 写成预期收益；用 AlpaSim **held** scene 或零分窗口的帧训眼睛；用 10 决策公开 scene 的日志回放行训眼睛（它们只是 navtest 日志的重渲染，状态在日志上）；在 R-C 出来之前渲染 W90 / 3V；在第 0 步判线出来之前训任何会被读数的臂。
- 不改 Cinque 一个字节的 base 文件：LoRA 单独存，合并只发生在导出副本上。

## 8. 三个执行者（协调者派；目标、硬约束、交回什么；过程由执行者自己定）

共用：topic `experiments/junction_eyes/`（README 按 `experiments/TEMPLATE.md`，INDEX.md 一行），run dir `$DATA_DIR/runs/junction_eyes/<step>/`，job tag `JE-<臂>-s<seed>`，探针 `JEP-<臂>-s<seed>`。代码进 `experiments/junction_eyes/{scripts,lib}/`；两个 topic 都要的进 `lib/`。所有 GPU 任务走 `python -m jevdrive.cl submit`，> 1 min 的任务在 tmux `jev` 里走 `scripts/tmux_run.sh`，超 1 h 的先估、先 profile、分级启动（docs/long-runs.md）。每 3–5 h 报一次，出错、停滞、判线出来、完成时立刻。按 CLAUDE.md 的分工，① ③ 派 Opus；② 的 F0 重渲与 G-dump 是机械活可派 Sonnet，W90 / 3V 的新渲染路径（若 R-C 触发）换 Opus。

| | 目标 | 硬约束 | 交回 |
|:--|:--|:--|:--|
| **① 尺子与第 0 步** | 建 topic；写预登记 A（3.1–3.3 逐字，push 后才读数）；R-A / R-B / R-C 标签器（两侧同语义）；J_nav / J_on / val / P 的行表；三组读数集的标签；探针、blind、WA-Cf 两行、旧尺子行；Z-0 判线与 R-C；决策条草稿 | 读数集一行不进 P；判线一个数不许读后改；WA-Cf 只作参照 | 行表（parquet）与标签；`results/step0.md`：闸、Z-0 判词、R-C 比例表、分桶表、两张图（R-A 按到路口距离分桶含 blind / WA-Cf 底；R-B IoU 分桶）；臂的集合建议（按 R-C） |
| **② 像素** | `vt_px.py` 的 t0-only 选项；J_nav 与 navtest 的 t0 对重渲并过 `check`；保持集的模型帧流式解码与 Stepper 教师输出缓存（60 段留出单列）；J_on / held / zero-98 的槽对抽取与 G-dump；R-C 触发后才做 W90 / 3V 渲染（5.1） | build 前盘余量 ≥ 300 GB；不改 vt_px 的既有协议；W90 / 3V 只在 ① 的 R-C 说要之后 | 像素缓存路径与 `check` 读数；保持集清单；G-dump 读数 |
| **③ 训练器与导出** | `op_torch` 的 Conv 可训练项（5.2）与恒等闸；trainer：两个头、两源 batch、保持项蒸馏、val 选 epoch、LoRA / U 两种臂、cam_emb（备 3V）；合并导出与 G-serve（5.3）；E5 与 WOD 金丝雀的读数脚本；300 步 smoke（恒等、吞吐、梯度范数） | 不碰 base ONNX；不加 plan loss；smoke 不读判线量；第 1 步的判线数值由 ① 在第 0 步后填进预登记 B，③ 不先训会被读数的臂 | trainer 与导出代码、恒等 / 等价闸读数、吞吐实测、每臂的预计时长 |

第 0 步判线出来后：① 填预登记 B 并 push；③ 把臂入池（staged）；读数协议（4.5）由 ① 跑，③ 提供重编码脚本。第 2 步另派。

## 9. 日历（估）

| 日期 | 事 |
|:--|:--|
| 10-14 | 协调者建 topic、派 ① ② ③；① 预登记 A push；三人并行（第 8 节） |
| 10-15 | ① 第 0 步读数、Z-0、R-C → 臂的集合；预登记 B push；③ smoke；若进，必跑 6 个 job 入池；视场臂若触发，② 开渲染 |
| 10-16 | 第 1 步读数协议、G1–G3、关闭规则或进第 2 步；决策条 |
| 10-16 – 10-18 | 第 2 步（若进）：bank、P2H10S 与 YS-L18 两条链、各板读；AlpaSim held 452 读数赶在 10-24 冻结前进选模型 |
| 10-18 | 决策条；research/junction-eyes/ 中文页面（docs/html-reports.md） |

与比赛线的关系：不等它、不阻它；10-24 冻结前若第 2 步在 AlpaSim held 上有读数，按 plan.md 3.3 的规则参与选模型，否则不参与。

## 10. 交付物
- 预登记 A（第 0 步，任何读数之前 push）、B（第 1 步判线数值，第 0 步读后、第 1 步任何读数之前 push）、C（第 2 步，另写）；每条补记带时间。
- 第 0 步：决策条一条（新尺子上的冻结眼读数、blind 与 WA-Cf 行、R-C 比例、判词），图两张。
- 第 1 步：决策条一条（臂 × 判线表、真实帧 / 重建帧两组、翻转方向、E5、WOD），图一张（各臂在新尺子上的分桶误差），BEV 样例四格（零分窗口里各臂的栅格对地图）。
- 第 2 步（若进）：另一条决策与页面。
- 全程按 docs/long-runs.md 报进度。

## 11. 给执行者的硬约束（CLAUDE.md 之外的本线专项）
- 地图只作训练标签与探针真值，测试期没有地图；不借任何外部模型权重进方法或 driver（WA-Cf 只作参照行）。
- held 452、98 个零分窗口、navtest-16 的任何 scene / log / token 不进训练、不进探针行、不参与任何选择。
- 判线一个数都不许在读数后改；读后探索单列「探索」，不进判词。
- 任何与本稿的偏离（数据量、臂、线）写进预登记补记并说明原因，读数之前。

## 12. 修订记录

| 日期 | 改了什么 | 依据 |
|:--|:--|:--|
| 2026-10-13 夜 | 定稿（旧稿备份在 tmp/） | main 与用户两轮讨论 |
| 2026-10-14 | 全文重写：定位改为涨分优先；训练行加 AlpaSim train 划分 on-policy 行、加 L-F0-nav drop-one；臂集合由 R-C 定，视场臂对 L-F0-nav 比、只判真实帧；G1 真实帧 / 重建帧分判，数值第 0 步后定（max(0.10 m, 参照差 / 3)），加 WA-Cf 参照行与「Cinque 只读 slot 7」配行，关闭判词三选一；Z-0 第三条只比出路类；探针行 P = J 去 val，val 选 epoch，所有臂同步数；保持项带真实历史的 Stepper 教师、plan 头同权、梯度范数核对；L-W90 保持行也 90°；Conv-LoRA 按 ConvNeXt 层型；第 2 步基底 P2H10S、AlpaSim 两段链、去路线支路、bd4 像素前置核对；三个执行者分工；盘 988 GB | 用户 10-14 答复（第 1 节 5–9）；main 第三轮审读；d282、d255、d137、d145、d160、d245、d275、unfreeze 预登记；盒上核对（roll dump、comma1M、盘） |
