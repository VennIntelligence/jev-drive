# op_parity 解冻视觉预登记（2026-10-06，任何打分跑之前写定）

## 问题

第 144 条：输入对齐、全量 navtrain、视觉冻结、只训 plan 通路 + adapter 的 P2 在 navtest 上 EPDMS 88.21，WA-JEPA 91.71，差 −3.55 [−4.27, −2.84]，剩余差在 DAC −2.7、TTC −0.9、NC −0.8。候选 (a) 冻结的前视 encoder（第 104 条：CAM_F0 离地 1.87 m 对 openpilot 的约 1.22 m，模型把世界缩到约 0.7）；(b) W 帧协议（B 阶段 G − W +0.90）。本登记测 (a)：**前视 encoder 允许适应之后，3.55 分能收回多少**；可选臂 V 把「encoder 适应」和「几何修正」分开。

## 不重测的既有结论

- 第 137 条：rc-bear-fix（stage 4 + plan + action 放开、只在 1.86 m 视角训）在 comma1M 原生相机上直路 ADE ×2.6。本线保留锚行（同 navtrain 帧、新输入置零、全部输出头蒸馏回 shipped），并把锚行上的「输入关时对 shipped 的漂移」作护栏。
- 第 142 条 B 阶段的协议选择（W）不变；侧 / 后相机无用（第 144 条），所以底座是 P2（无侧机）。
- 第 104 条第 9 点：旧模型上虚拟相机 1.40 m 在 navtrain 上 +0.72 [−0.14, +1.60]、1.30 m −0.09，降高换来 EP、丢 DAC；V 臂在训练过的 P2 上重测，不是重测零样本机制。

## 臂（同一配方：第 144 条 P2 的全量配方，只差视觉部分）

| 臂 | 视觉 | 可训视觉参数 | 输入 | 视觉 lr |
|:--|:--|--:|:--|--:|
| F | 冻结（= P2-frozen） | 0 | 缓存的 `view_39` token（`<shard>@warp/front.npy`） | — |
| U1 | stage 4（3 个 ConvNeXt block，2048 通道 4 × 8）+ head（LN、fc 2048 → 512）全放开；stage 1–3 与 stage 4 的下采样冻结 | 110 M | 缓存的 `conv2d_36`（下采样输出，2048 × 4 × 8，每 token 8 槽 1 MB） | 1e-5 |
| U1L | stage 4 六个 MLP matmul 上 LoRA rank 16（α = r） | 3.0 M | 同 U1 | 3e-4 |
| U2 | 整个视觉编码器放开 | 349 M | 像素在 CPU 上在线渲染（4 个 CAM_F0 key + 6 帧 warp，与 pp_prep 的 W 路径逐字节同一函数），每步编码；梯度只走 t0 图像对，7 个旧图像对用同一（当前）编码器无梯度前向（op_adapt 的 C′） | 5e-6 |
| V | 冻结；帧来自 1.40 m 的虚拟相机（`scripts/op_lb.py _vcam_job`：key 按低机位重渲，warp 用该机位的路面） | 0 | 缓存的 `view_39` + shipped 在同帧上的锚定教师（`<shard>@vh140/`） | — |

选型理由：
- **U1**：stage 4 是四段里最后一段，空间只有 4 × 8，切在下采样之后缓存，训练成本与冻结臂同级；库里已有 stage 4 放开的配方（it_dw3、rc-bear-fix），第 137 条的遗忘也是这个范围，正好检验锚行能否压住。
- **U1L**：同一缓存、参数少 37 倍。U1L ≈ U1 → 增益是小幅重标定；U1L ≪ U1 → 需要容量。
- **U2**：尺度错误来自图像到特征的整条映射（相机高度改变的是地面纹理的透视），只有全量放开才能在早层改；低 lr 加锚行防漂。只让 t0 对带梯度是显存与吞吐的取舍（8 对全带梯度约 8 倍激活）。
- **V**：不动 encoder，只把输入几何改对（第 104 条的修复第一步）。V 与 U 对照回答「encoder 适应」还是「几何修正」。

其余配方与第 144 条 P2 相同：每批 0.75 模仿行（plan → 日志 8 位姿，Huber 归一）+ 0.25 锚行（新输入置零，plan 与全部头蒸馏回同帧 shipped），λ_i 1、λ_c 3、λ_d 30；AdamW wd 0.01，plan 通路 lr 3e-5、adapter 3e-4，cosine，clip 1，fp16 + GradScaler。

## 磁盘与数据（决定了 pilot 的形状）

盒子数据盘剩约 104 GB（2026-10-06 08:00，共享 5 TB 盘 98% 满）。U1 的缓存每 token 1 MB：全量 navtrain 103 k 约 105 GB，放不下；pilot 用全量缓存的 2 个分片 `navtrain_full.s0of12`、`s1of12`（17 216 token，日志混合，dev = `navsim/op-parity-full-dev` 在其中的部分），U1 缓存约 17.6 GB，V 缓存约 4.4 GB。F 用已有的 W token 缓存，同分片同行序。

## pilot（早停闸门）

- 每臂 3 000 步、batch 64（约 11 epoch over 17 k）、warmup 100、seed 0 与 1；同 seed 各臂行流相同（U2 的 rank 行流同 F）。
- 读数：navtest 全量 12 146 token，W 帧（V 用 1.40 m 虚拟相机帧），**所有臂走同一条像素路径**（`pp_unfreeze.py plans`：渲染 → 各自的编码器 → policy），devkit navsim main @ 0a380a9 v2 EPDMS（复现 WA-JEPA 91.71 的同一 harness 与 metric cache），按 navtest log 聚类的逐 token 配对 bootstrap（B 10 000），两 seed 逐 token 平均。
- 打分前的等价检查（不过不打分）：(i) 冻结 stage 4 在 `conv2d_36` 缓存上复出 W token 缓存（fp16 噪声级）；(ii) 像素路径上 P0 与全量 P2-F-s0 的计划对缓存路径计划（pp_eval 已存）平均差 < 0.01 m。

**闸门**（每个 U 臂各自判；先写在这里）：
1. U − F（seed 平均）≥ **+0.5** EPDMS（点估计；CI 一并报告）；
2. 护栏：计划速度比（计划 4 s 路程 / 日志，日志 > 2 m 的 token 中位）在 1.00 ± 5%；
3. 护栏：EP 与 DAC 不出现交换（两者配对差异号且各自 CI 不含 0）；
4. 护栏（防遗忘）：dev 行上输入关时计划对 shipped 的漂移（`drift_off`，8 位姿平均 L2）≤ 0.10 m（第 144 条全量 P2 为 0.04）。

全过的 U 臂里取 U − F 最大者进全量估算；没有一个过 → 停、报告。V 按同一规则判 V − F；V 过闸 → 向 main 报「V + 最优 U」组合臂的估算。

## 全量（pilot 之后，先估算；> 约 6 GPU·h 先报 main）

主对比：**U − P2-frozen（第 144 条全量 P2，2 seed）** 在 navtest 上的配对差与 CI；读数：距 WA-JEPA 的差、navhard two-stage（results/navhard.md 同一 harness）、HUGSIM 64 `exam` + `spec`（第 144 条同一 harness，U 臂的训练过的视觉权重经 pp_hugsim.py onnx 进 ONNX）。护栏同 pilot。全量数据的形状（全 navtrain 在线算 `conv2d_36`、或腾盘缓存）由 pilot 实测吞吐决定，一并报。

## 判读（写在结果之前）

- U − F ≥ +0.5 且 DAC / TTC 涨：冻结 encoder 是剩余差距的一部分；按 U − F 占 3.55 的比例报告。
- U ≈ F（< 0.5）：剩余差距不在前视 encoder 的适应上（在预训练表征、W 协议或 policy 容量），trick or trade 偏 trade 一侧。
- V ≥ F + 0.5 而 U ≈ F：几何（相机高度）是可修的输入问题，不需要动 encoder。
- U 涨但 drift_off 超线：增益以遗忘换来，不进全量。

## 限定（预先写明）

pilot 只有 17 k token（全量的 1/6）、3 000 步，U 的增益可能随数据变化；U2 的梯度只走 t0 对；原生相机（comma）上的遗忘本轮不测（范围：本轮不用 comma 数据），只用 navtrain 锚行的 drift_off 代理；V 的 warp 路面用正确的机位高度，W 用的是 CAM_F0 相对后轴原点的高度（1.53 m，op_lb 的约定），V − F 含这一点差别。
