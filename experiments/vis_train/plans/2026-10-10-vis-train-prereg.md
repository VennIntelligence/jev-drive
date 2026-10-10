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
