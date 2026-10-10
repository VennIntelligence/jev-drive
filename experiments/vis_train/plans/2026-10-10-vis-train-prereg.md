# VT：在 navtrain 上真正训练视觉参数，navtest 转弯出界差距会不会收（预登记）

2026-10-10 23:30 JST，任何读数之前提交并 push。lane VT，topic `experiments/vis_train/`。接第 145、147、160、204、243、244 条。

## 问题

driver 的 Cinque 视觉 encoder 一直冻结；今天在冻结特征上试过的方法对大弯 off-road（DAC 失败）率的改动都约为 0。唯一一次解冻（第 145 条 U2）3 000 步、lr 5e-6、25% 锚行，权重只动 0.1%，没有测到这个问题。第 160 条：在 navtrain 上微调过的 encoder（WA-Cf，诊断上限）作 memory，> 20° DAC 失败 −1.33 pp、EPDMS +0.95。WA-JEPA 的配方：整段 encoder 微调约 240 epoch、encoder lr 1e-5、无锚、无增广，加未来帧 teacher 特征的稠密预测 co-loss（权重 0.5）。

今晚剂量约为它的 1/12，主输出是**趋势**：off-road 率随步数与权重位移是在降还是平。

## 共同设置

- 起点：`SH30-F-s{0,1}`（P2 + hinge λ 30 / margin 0.5，10 k 步 × 128，W 协议）。各臂从同 seed 的 SH30 续训，同一行流（seed 决定），navtrain 全量 103 288 token，split `navsim/op-parity-full`，batch 64。
- 配方 = SH30（imitation + hinge 30 / 0.5 + 25% 锚行 + 非 plan 头蒸馏 λ_d 30），**偏离一处**：学习率在 300 步 warmup 后保持常数（policy 1e-5、adapter 及新模块 1e-4、视觉 1e-5），不做 cosine。理由：读数是「随步数的趋势」，退火到 0 会让后段按构造变平；常数 lr 下各臂可在相同步数比较，也可续跑。
- 步数：目标 30 k（约为第 145 条剂量的 10 倍）；按实测吞吐取 5 k 的整数倍，使 A / B 在 05:00 JST 前结束、C 在 07:00 前结束，下限 15 k，上限 60 k。A 与 B 同步数；对照臂取其中最大者。每 5 k 步存一个 checkpoint（tag `VT-<臂>-s<seed>-k<NN>`，末尾 `VT-<臂>-s<seed>`，都在 `runs/op_parity/runs/` 下；提交前验证 tag 不存在）。
- 像素：W 协议帧预渲染成缓存（训练与 navtest 各一份），训练时不在线渲染；原版 Cinque 的 token 用现有 `front.npy` 缓存，不在线跑。

## 臂

| 臂 | seed | 视觉 | 说明 |
|:--|:--|:--|:--|
| F | 0, 1 | 冻结，无支路 | 同步数同数据的续训基线 |
| A0 | 0, 1 | 冻结；memory 通道喂 Cinque 自己的 t0 token（= 支路 lr 0） | 「加通道」对照：A − A0 才是「训练了视觉参数」的效应 |
| A | 0, 1 | 原版 Cinque 冻结且逐位不变；支路 = Cinque 视觉 encoder 的可训练副本（lr 1e-5），读同一对 t0 前视图，输出 32 × 512 token 经 memory 通道进 policy | 主臂 |
| B | 0, 1 | A + 未来特征损失 | 见下 |
| C | 0 | 无支路；Cinque encoder 原地解冻，lr 1e-5；无锚行（d_frac 0）、无非 plan 头蒸馏（λ_d 0）；8 个 slot 都过当前 encoder，梯度只过 t0 一对（同 U2） | 「加」对「训」，以及原生能力损失 |
| F0 | 0 | 冻结，无锚行、λ_d 0 | C 的对照：C − F0 是 encoder 训练本身，F0 − F 是去锚 |

- **memory 通道**（第 160 / 204 条配方）：`ParityAdapter(use_side, n_cam = 1, n_t = 1)`，SH30 的 adapter 权重原样载入，新增的 `side_in` 与 embedding 新初始化；训练时 25% 行屏蔽 memory（`MEM_DROP`，独立 rng 流）；锚行 bias 恒为 0。
- **支路自己的 head**（第 204 条：tokenizer 必须带自己的 head 才稳定被读）：A / B / A0 的支路 token 上接第 147 / 160 条的 thin decoder（2 层 MLP，[token, ego]，imitation + hinge λ 10 / 0.3），权重 1.0，与 policy 联合训练；推理不用。它同时给支路一条不依赖「policy 读不读通道」的监督梯度（否则支路与原版 token 初始相同，policy 没有理由读它，支路可能不动）。A0 上该 head 只训 head。
- **B 的未来特征损失**：predictor（2 层 transformer，d 512，输入支路 t0 token + ego 20 维（含 command）+ horizon embedding；不给未来位姿）预测同一 log 里 t0 + 1.0 s 与 + 2.0 s 两帧的 teacher 特征（冻结 Cinque 的 t0-slot token，取自现有 `front.npy` 缓存里同 log 的后续 token；缺的行跳过），目标按训练集逐通道标准化，smooth-L1，权重 0.5。报告覆盖率与「未来 = 当前」的 copy 基线损失。推理不用。
- 特权信息：无。没有用 WA-JEPA 的权重或特征。日志未来轨迹只作模仿标签。

## 启动前检查（不过不启动）

1. **吞吐**：对整条像素训练管线做 profile（读盘、搬运、forward、backward、optimizer、日志），前后数字写进 README；it/s、samples/s、GPU 利用率决定上面的步数。
2. **恒等**：(A / B) memory 屏蔽时、(A0) 同、(C) lr 0 跑若干步后、(F) 步 0，navtest 前 2 048 token 的 plan 与 `SH30-F-s{seed}` 的缓存 token 路径比较，只看 ≤ 4 s 的 plan 点，逐位相同或在 fp16 / bf16 舍入内（超过 0.03 m 的行占比 < 0.1%，并给错模型对照：另一 seed 的 SH30）。像素缓存路径复现缓存 token（同 `pp_unfreeze` 的等价检查）。
3. **分级启动**：先 A-s0 跑 300 步，查：loss 有限、dev ADE 不比 SH30 差 > 0.1 m、支路权重在动（相对 L2 > 0）、memory 屏蔽与否 dev ADE 有差或至少支路 head 的 loss 在降、吞吐与估计一致；然后其余。

## 读数

每个 checkpoint（A / B / C / F 每 5 k；A0 / F0 每 10 k）与末尾，全部经 `jevdrive.bench`，navtest 12 146 token，按 log 的 cluster bootstrap（B 10 000），同 seed 配对：

1. > 45° 与 > 20° 桶：DAC 失败率、切内侧率、转不过去率（定义同 `experiments/corridor/`、第 240 条）、桶 EPDMS。
2. 全榜 EPDMS 与全部子项、< 5° 桶（护栏）。
3. 末尾：navhard two-stage；A / B 的测试时屏蔽 memory 与跨 log 打乱 memory（通道是否被读、是否读的是本行内容）。
4. 权重位移（encoder / 支路相对初始的 L2，按 stage）；支路 token 上的第 160 条 Stage-0 probe（> 20° DAC 失败率；参照 V 10.59%、WA-Cf 6.88%）。
5. C：原生帧直路 ADE 对 shipped（第 137 条的遗忘读数）。

## 判定线

主量：> 20° DAC 失败率，X − 对照（A − A0、B − A0、B − A、C − F0；另报对 F）。

- **在收**：末尾差 ≤ −0.5 pp 且 CI 上界 < 0，并且后三个 checkpoint 的均值低于前三个（趋势向下）。参照：诊断上限 −1.33 pp。
- **平**：|差| < 0.3 pp、CI 含 0，同时位移 ≥ 1%（A / B 另需测试时屏蔽 memory 掉 ≥ 0.2 EPDMS）。
- **没测到**：位移 < 0.5%，或 A / B 的通道没被读（屏蔽掉分 < 0.2 且 Stage-0 probe 不动）。此时结论只能写「剂量 / 通道不足」，不写「视觉训练无用」。
- 其余情形写「未定」并给数。
- 护栏：全榜 EPDMS 与 < 5° 桶对对照 ≥ −0.3（CI 下界）；不过则该臂的转弯收益按「换来的」记。
- 两个 seed 方向相反的结论降一档。

## 停止规则

loss 非有限、分级启动检查不过、dev ADE 比起点差 > 0.3 m 持续两个 eval：该臂停并报告，其余继续。任何臂不因中途读数好坏而提前停或加码（续跑只按上面的时间规则）。

## 补记 1（2026-10-11，第二波臂 W；在任何读数之前提交）

来由：`experiments/corridor/results/fov.md`：模型最宽的输入帧水平 58.7°（±29.4°，只由 CAM_F0 渲染）。navtest > 45° 上 SH30 切内侧失败（占该桶 DAC 失败的 47%）所越过的内侧路沿，t0 时 93.8% [86.3, 98.7] 在视场外，中位方位角 43°。该量不区分失败与通过，所以它只说明模型看不到那条路沿，不说明看到了就有用。第 144 条的侧视臂（P3 − P2 −0.05）走的是冻结 encoder、读的是全榜。W 是直接的测试。

- **臂 W**（卡够则 2 seed，否则 seed 0）：与臂 A 完全相同，只有可训练支路的输入不同：t0 的 CAM_L0、CAM_F0、CAM_R0 三个视图各自过**同一个**（共享权重的）支路 encoder，得 3 × 32 个 token 经 memory 通道进 policy（`ParityAdapter(n_cam = 3, n_t = 1)`，相机 embedding 区分三路）。F0 视图与 A 的输入逐字节相同；侧视图按 P3 的渲染方式（沿各自安装 yaw 渲染成 openpilot road + wide 对）。合起来的水平视场由代码量出并写进 state.md 与结果，须 ≥ ±60°，否则改为拼接宽帧并再补记。冻结的 Cinque 通路输入不变。支路自己的 head 读三路 token。若 W 建好时 B 的趋势已明显好于 A，则 W 以 B 为底（加未来特征损失，只对 F0 视图），对照相应换成 B；用哪个在启动前写进 state.md。
- **恒等**：memory 屏蔽时与 `SH30-F-s{seed}` 相同，标准同 A。
- **登记的比较**：W − A（或 W − B），> 45°：切内侧率与 DAC 失败率；全榜 EPDMS 与 < 5° 桶为护栏（线同上）。另读：> 45° 切内侧率按「被越过的内侧路沿 t0 时是否在常规视场内」分层（`$DATA_DIR/runs/corridor/fov/` 的 tok.parquet、unit.parquet）。
- **线**：W − 对照的 > 45° 切内侧率 ≤ −1.0 pp 且 CI 上界 < 0 记「看到路沿有用」；|差| < 0.5 pp 且 CI 含 0、且测试时屏蔽侧视 token 掉分 < 0.2 记「没被用」；其余「未定」。
- **次序**：不推迟、不重启第一波。W 只需 t0 的侧视帧（不要历史帧），只用空闲 CPU 建，且共享盘保持 ≥ 300 GB 余量；卡或卡的份额空出时由 pool 放置。步数与其对照臂相同；若时间不够，按相同步数处的 checkpoint 与对照比较。

## 补记 1（2026-10-11 00:30 JST，builder；任何训练步与任何读数之前）

实现时定下、预登记正文没有写明或需要偏离的几处：

1. **恒等检查的容差**。A0（memory 屏蔽）对 `SH30-F-s{0,1}` 的缓存 token plan，前 2 048 个 navtest token、前 15 个 plan 点：超过 0.03 m 的行占 0.20% / 0.29%，登记线是 < 0.1%，按原文不过。这些行的最大差全部是 0.03125 m，即 32–64 m 处两个 fp16 ulp（0.03 m 这个数取自「32 m 处两个 ulp」，实际是 0.03125，略大于 0.03）；中位差 0.002–0.003 m；错模型对照（另一 seed）77% 的行超线、最大 0.31 m。来源是 adapter 多了被屏蔽的 memory token 后 attention 的 fp32 末位不同，再经 fp16 的 H 相加。判据改为「超过 0.032 m（两个 ulp 之上）的行 < 0.1%，且错模型对照 > 50%」，原线的读数照报（`ident/*.json` 的 `pass` 与 `pass_2ulp` 两栏）。F 逐位相同（0 行，最大 0）。A / B / C 的读数在像素缓存建好后补在 state.md。
2. **navhard 用 W 帧**，所有臂（含 F、F0）一致。支路与 C 读像素缓存，缓存只有 W 协议；G 帧要另渲一套 GIMM 像素。SH30 的 navhard 参照行（`SH30-F-s{0,1}@gimm`）因此不能直接对比，读数阶段补跑 `SH30-F-s{0,1}`（W）作参照。
3. **视觉组不加 weight decay**（policy、adapter、head、predictor 仍是 SH30 的 0.01）。lr 1e-5 × wd 0.01 × 3 万步会让权重整体收缩约 0.3%，与判定线里的位移阈值（0.5% / 1%）同量级；去掉后位移只来自梯度。
4. **梯度裁剪分组**：policy + adapter 一组（与 F 相同，1.0），视觉一组（1.0），支路 head + predictor 一组（1.0）。C 也如此（U2 当时是合并裁剪）。
5. **停止规则的「起点」**取 SH30 自身的 dev ADE（memory 臂：步 0 时屏蔽 memory 的 ADE）。memory 臂在步 0 打开 memory 时 dev ADE 约 1.08 m（`side_in` 与 embedding 新初始化，SH30 的 adapter 输出层非零），不是起点。
6. **checkpoint 命名**：中间每 5 k 步一个 `VT-<臂>-s<seed>-k<NN>`，最后一步只写 `VT-<臂>-s<seed>`（不重复写一个 `-k<末>`）。
7. **支路 head** 在所有有日志未来的行上训练（含锚行；锚行只是 policy 的 ego 置零），输入是真实 ego。A / B 的支路实现为 port 自身的视觉权重可训练、原版 Cinque 的 8 个 slot token 全部取自缓存（逐位不变）。
8. **B 的 predictor**：32 个支路 token（逐通道标准化）+ 1 个 ego token，加 slot 与 horizon embedding，2 层 pre-norm transformer encoder（d 512，8 头），线性输出 32 × 512；两个 horizon 各前向一次。未来行按 log + 时间戳（± 0.1 s）在 12 个 shard 内查找。

## 补记 2（2026-10-11 00:30 JST，用户改范围，经 main 转达；任何读数之前）

这不只是转弯实验：任何方向的提升都算，靠降低别处换来转弯收益的臂不接受。读数与判定增加：

1. **每个 checkpoint 的全分解**（navtest，对各自对照配对，按 log 的 CI）：EPDMS 的全部子项（NC、DAC、DDC、TLC、EP、TTC、LK、HC、EC）与四个转角桶（< 5°、5–20°、20–45°、> 45°），不只 > 20° DAC。
2. **纵向读数**：NC 失败按第 196 条的类别拆分（尤其「本车道前方静止或慢车」，SH30 的 NC 失败里占 53%；复用该条的分类代码与 token 表），以及 plan 4 s 弧长对对照、对日志的比值。
3. **末尾的板**（每臂最终 checkpoint，全部经 `jevdrive.bench`）：navhard two-stage（stage 1 / stage 2 分开报）、HUGSIM 64 闭环（与 SH30 参照行同协议，`spec_plan_smooth`）。F / F0 是普通 P2 checkpoint，链里直接排（等最后一个训练任务结束、核空出来之后）。A0 / A / B / C 的 HUGSIM 需要 serving 端新代码（bias server 里跑支路 encoder；C 要导出带训练后视觉权重的 ONNX），bench 目前不支持，读数阶段先核实 C 能否走现有 `pp_hugsim.py onnx`，不能的臂如实写「未跑」。WOD-E2E val 零样本：只在现有脚本能直接读 navtrain 训练的 checkpoint 时跑（F / F0 可能可以，memory / 支路臂需要新代码则跳过并写明）。
4. **不退步规则**：一个臂只有在没有任何子项、任何桶、任何板相对其对照「CI 整体在 0 以下」时才是候选。退步与收益同等位置报告，每臂一张表：升了什么、降了什么、没动什么。
5. 最终报告对每个臂写明收益（如有）来自哪个方向：转弯、纵向 / 前车、还是别处。

原判定线（> 20° DAC 的「在收 / 平 / 没测到」）保留为转弯方向的读法；候选资格由第 4 点决定。

## 补记 3（2026-10-11，臂 W 的实现选择，arm-W agent；W 的任何训练步与读数之前）

「补记 1（第二波臂 W）」没有写明或需要改的几处：

1. **侧视图的图像对 = W 协议的 0.2 s 对**，不是 P3 的 0.5 s 对（key k − 1, key k）。前视 t0 slot 的对是（t0 key 按 ego 运动 warp 到 t0 − 0.2 s 的位姿，t0 key）（`op_interp` 的 `warp` 在两个 key 之间取较近的 key，t0 − 0.2 s 取 t0 key）。侧视图用同一构造：t0 key 按 P3 的方式渲染（`OpenpilotMaps(cam, yaw_deg = 安装 yaw)`，与 `pp_prep.render_side` 逐字节相同），前一帧 = 同一张 t0 key 经 `op_interp.warp_frame` warp 到 t0 − 0.2 s。warp 的几何不改代码：位置 c、朝向 ψ 的相机等价于绕竖直轴转过 ψ 的车体系上的前向相机（相机位置 R(−ψ)c，位姿 (x, y, yaw + ψ)），`warp_frame` 原样适用（`lib/side_store.virtual`）。理由：三个视图过同一个共享权重的支路 encoder，第 142 条表明图像对的间隔决定 encoder 读出的速度；0.5 s 对会让侧视图读出 2.5 倍的速度，与 F0 视图不一致。代价：前一帧不含真实的第二帧信息（F0 视图同样如此），路面以上的近处结构按 60 m 球面 warp，侧向视差比前视大。与 P3 缓存 token 的预期差异见 state.md（同配对时复现，换成 W 配对后 mean |d| 约为跨行差异的 0.7 倍）。
2. **只存 t0 的一对**（每 token 2 相机 × 2 帧 × 0.39 MB = 1.57 MB；navtrain 151 GiB、navtest 18 GiB、navhard 9 GiB），`$DATA_DIR/runs/vis_train/px_side/<data>/side_t0.npy`，行序 = `tab.npz`。
3. **盘余量线由 300 GB 改为 50 GB**（用户 2026-10-11 改，经 main 转达）。建完后预计剩约 290 GiB。`vt_side.py build` 在写每个文件前按「当前空闲 − 第一波像素缓存尚未写入的部分 − 本文件」核对，低于 50 GiB 即停。
4. 视图顺序与 embedding：memory 的相机维为 [CAM_F0, CAM_L0, CAM_R0]；测试时「屏蔽侧视」= 只屏蔽后两路（`side_mask`），训练时的 memory 屏蔽（25% 行）三路同屏蔽。

## 补记 4（2026-10-11 00:05 JST，接手的 lane agent；A / A0 / B 的任何训练步与任何读数之前）

停止规则的计数从 warmup 结束（步 ≥ 300）之后的 eval 开始，步 0 的 eval 不计。原因：memory 臂在步 0 打开 memory 时 dev ADE 约 1.08–1.12 m（补记 1 第 5 点：`side_in` 与 embedding 新初始化，不是起点），比起点（屏蔽 memory 的 0.57 m）高 0.5 m，按原实现计作一次「差 > 0.3 m」。6 步的 preflight smoke（eval 在步 0 与步 3）因此连续两次超线、触发停止规则，A / A0 / B 六条链在 22:53 CST 全部以 preflight 失败结束（F、F0、C 没有 memory 通道，不受影响）。300 步分级启动里 memory-on ADE 在步 100 已降到 0.64 m、步 300 为 0.58 m，正式训练（eval 每 1 000 步）不会因此触发；改动只是不让步 0 的构造性差值占掉两次机会中的一次。规则本身（比起点差 > 0.3 m 持续两个 eval 即停）不变。
