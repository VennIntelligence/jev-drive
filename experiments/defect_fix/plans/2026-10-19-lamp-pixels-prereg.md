# LAMP-PIXELS 预登记（2026-10-19 晚，写在任何分类器输出、任何新 dump 读数之前）

## 问题

第 373 条：冻结 Cinque token 读灯只到「运动学 + lead 头」的水平（TOK − LEAD +0.031），红灯站 vs 绿灯起步 0.531，判为输入问题。
剩下的分叉是**哪一种**输入问题：

- (a) 渲染像素里就没有灯色（MTGS 把背景烘焙进 3DGS，灯冻结在采集时的样子 / 被糊掉 / 颜色不对；第 340 条看到人已起步时渲染仍是红灯），
  任何模型在这个模拟器里都读不到，只能靠先验；
- (b) 像素里有灯色，是冻结的眼睛（openpilot 模型帧分辨率、YUV 打包、2 Hz 槽）带不出来 → 我们自己在真实 nuPlan 相机上训的读灯通路有望加分。

用户规则：线索是否可见只用模型探针判，不做人看的页面。不训练策略。

## 标签（全部来自日志，已存在的标签管线）

- **真实训练 / 留出帧（TV）**：NAVSIM trainval 全量 CAM_F0（第 TV-AUDIT 下载），`runs/self_iter/tv_census/frames.npz`（`tv_audit.py census`，
  `lc_ctx.label_log` 的规则）：路线前方第一条停止线为灯线（`ln_type == 2`）、找到路线所走的 lane connector（`ln_conn`）、
  车头到线 `ln_dF` ∈ [-3, 60) m、t0 时该 connector 状态 `ln_st0` ∈ {红 1, 绿 0}，且 CAM_F0 在盘上（`cam4`）。
  y = 红。log 划分：训练 `navsim/tv-clean-train-logs`，留出 `navsim/tv-clean-hold-logs`（两者与大考场 / held 460 / navtest 无 segment、无 drive 重叠）。
- **格（cell）**：自车速度 v0 ∈ [0, 0.3) / [0.3, 3) / [3, 8) / ≥ 8 m/s × 到线距离 ∈ [-3, 10) / [10, 25) / [25, 60) m，共 12 格。
  红灯帧大多是站着、绿灯帧大多在走，分类器可以靠「站在线前」而不是灯。**训练集逐格平衡**：每格红、绿各取 min(n红, n绿, 1500)
  帧（seed 0）；留出集每格各取 min(·, 300)。所有 AUC 主报**格内分层 AUC**（只数同一格内的（红, 绿）对），自然 AUC 只作参照。
- **大考场 held 窗（arena）**：census（`native_gap/set_census.parquet`，nx_census 规则）有灯线、connector 已知、t0 车头到线 [-3, 60) m 的 held 窗。
  每窗 10 个决策 k = 0..9（t0 + 0.5 k），每个决策的标签 = 原始 nuPlan test DB 该时刻 connector 状态（`nx_census.Log.status`，±0.5 s 最近已知值），
  人车头到线距离 = t0 距离减人走过的弧长；只保留 状态 ∈ {红, 绿} 且 人车头到线 ∈ [-3, 60) 的决策行。
  - **渲染帧**：JTN-L18-s1 原生（无 hold）大考场闭环 `SH30_DUMP` 的 1920x1080 CAM_F0 JPEG（render 原图，不经模型打包）。
    k = 0 / 1 是 force-GT，自车在人的位姿上，帧与模型无关；k ≥ 2 是 JTN1 自己的位姿。已有 dump（`redline_probe/dump` 的 k = 0 / 3、
    `native_gap/dump/JTN1` 的全部 k）直接复用，缺的窗用 redline_probe.sh dump 同一条命令新读。
  - **配对真实帧**：同一 log（39 个 held log 全在 navsim test sensor_blobs 里）、DB 时间戳最近的 2 Hz OpenScene 帧的 CAM_F0
    （时间差 ≤ 0.26 s，超出则丢弃该行）：同一时刻、人的位姿、同一盏灯的真实照片。

## 分类器（我们自己的，不借外部检测器 / 规划器当老师）

- **CNN-N（主）**：torchvision ResNet-18 ImageNet 初始化（通用骨干，`resnet18-f37072fd.pth`），全网微调，1 个 logit。
  输入 = CAM_F0 原生分辨率上半部裁切：行 0-640、全宽 1920（1920x640，不缩放）。3 epoch，batch 32，AdamW 3e-4、wd 1e-4、余弦，bf16。
  增广：裁切窗口平移 ±24 px、水平翻转、亮度 / 对比度 ±20 %、p = 0.5 的高斯模糊 σ ∈ [0, 1.5]、p = 0.3 的 1-2 倍降采样再升回；
  **不做色相 / 饱和度扰动**（灯色就是信号）。1 个 seed，不调参。
- **CNN-L（分辨率对照）**：同一网络与配方，输入是同一裁切缩小 4 倍（480x160）：像素密度约等于 Cinque wide 帧的亮度（0.29x）。
- **TOK（冻结 token 基线）**：Cinque `view_39` 在 serve 打包（`navsim_zs.OpenpilotMaps`，road + wide，YUV 6 通道 128x256）上的 32 x 512 token，
  单帧（prev = cur，同一帧复制，真实与渲染同一协议）；真实帧按 pickle 相机标定打包，渲染帧用 dump 里该决策的槽 7 打包帧（即 serve 自己的打包）。
  探针 = redline_probe 的 token 网络（逐 token LN → 16 → MLP 256），在同一平衡训练集上训，10 epoch。说明：openpilot 帧有自己的相机模型，
  不能喂 1920x640 裁切；「同样的裁切」对 TOK 的含义是同一批帧、同一标签、同一格平衡。
- 三者都只在 TV 训练 log 上训，原样用于 TV 留出、配对真实帧、渲染帧。

## 主比较与判读（k = 0，每窗一行；红 vs 绿；格内分层 AUC；95 % CI = log 聚类 bootstrap，B = 2000，配对差用同一组抽样）

先过两道门，否则判「探针太弱，不下结论」：
- **G1**：CNN-N 在 TV 留出上的分层 AUC ≥ 0.85。
- **G2**：CNN-N 在配对真实帧（大考场 held 窗、k = 0）上的分层 AUC ≥ 0.80（测试域真实照片上也读得出）。

分支（看 CNN-N 在渲染 k = 0 上的分层 AUC，记 R）：
- **(b) 像素里有灯色、冻结眼睛带不出**：R ≥ 0.85 且 R − TOK(渲染) ≥ +0.05、差的 CI 下界 > 0。→ 写读灯通路接进 planner 的具体方案和代价，**不训练策略**，回报。
- **(b0) 像素里有、token 也已带**：R ≥ 0.85 但 R − TOK(渲染) < +0.05。→ 与第 373 条冲突，记录并复查 373 的标签口径。
- **(a) 模拟器里没有灯色**：R ≤ 0.65 且 G2 过。→ 结论：此模拟器里线索缺失，只能用先验；并给出「烘焙灯色」统计（次要 3）。
- **(m) 中间**：0.65 < R < 0.85。→ 灯色部分残留 / 变质，报数，不写通路方案；用次要 1、2 判断更接近哪一边。

## 次要（全部预先列出，只描述，不改判读，除 (a) 的佐证条款）

1. **跟踪测试（冻结灯检验）**：窗内灯态变化的窗（同一窗的决策行既有红又有绿，主要是红转绿起步窗），只数**同一窗内**的（红, 绿）决策对的 AUC，
   CNN-N 真实帧 vs 渲染帧。灯若冻结，渲染接近 0.5，真实应高。(a) 的佐证条款：渲染 ≤ 0.65 且真实 ≥ 0.80 时写「灯被冻结」，否则只写「灯色缺失」。
2. **渲染上的 log 折微调（排除域偏移）**：CNN-N 在渲染 k = 0 行上按 sha1(log) 5 折微调（lr 1e-4、4 epoch），每窗由没见过其 log 的折打分。
   若真实训练的 CNN-N 在渲染上低、而渲染微调后 ≥ 0.80，则不是「没有」而是「有但域偏移」，(a) 改记为 (m)。
3. **烘焙灯色统计**：阈值 τ = TV 留出上 CNN-N 平衡准确率最高的分数；报渲染与配对真实帧被判「红」的比例：
   红灯线前等待（census `red_stop & red_cls == C`，k = 0）、起步窗（`red_cls == G`）k = 0 与日志已转绿之后的决策、日志绿灯的 k = 0 行。
4. 分项：CNN-N / CNN-L / TOK 在 TV 留出、配对真实、渲染上的分层与自然 AUC；渲染 − 真实的差（同一批窗配对）；按到线距离三档的分层 AUC。
5. CNN-L 对 CNN-N：真实帧上分辨率掉多少（给 (b) 方案定输入分辨率用）。

## 冒烟与产出

- 冒烟（几分钟）：`SMOKE=1`：TV 每格 40 帧、训练 60 步，1 个新 dump 列表 4 窗，跑通全链（不看数）。
- 代码 `experiments/defect_fix/scripts/lamp_pixels.{py,sh}`；盒上 `$DATA_DIR/runs/defect_fix/lamp_pixels/`；
  结果 `experiments/defect_fix/results/lamp_pixels.{md,json}`，decisions.md 下一个空号。所有作业走 `python -m jevdrive.cl submit`，priority 12，
  一条 `--after` 链一次提交，等完成只用一个后台阻塞命令，不轮询。
