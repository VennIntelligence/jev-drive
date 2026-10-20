# NU-FLAT 预登记：把今天 PAI 上的历史拉平搬到 nuPlan 候选 B2-s1（CTRL / F / FR，续训，一个 seed）

**状态：已登记，2026-10-20 21:40 CST（box 时间）。** 登记时三个臂都没有任何训练、identity、smoke 或读数输出。盘上有：B2 `STANDFMXH-L18-s1`
（第 391 条，nuPlan 原生候选）的检查点与存档大考场 held 18 块原生读数、held 460 读数；STAND-R（补记 20，第 399 条）的 `on1_JTX/stand_gap.npz`
与离线闸 G1–G3（已过）；SIMP-G 的起刹行表 `runs/defect_fix/simp/flat_onset.txt`；ROUND2 加的 `rm_train --flat-p`（main 上 3039f4c8 起有，
ROUND2 的 identity 已证 off 与 `--flat-p 0` 逐位相同）。代码：`scripts/nu_flat.{sh,py}`；结果 `results/nu_flat/`。**rm_train 不改。**
本 lane 不碰 ROUND2（`experiments/round2/`，作业 `r2-*` / `pai_fix*`）的文件与作业，池优先级低于它（见「优先级」）。不读 PAI、不做镜像、不改 decisions.md。

## 为什么

- 第 407 条：适配器的 ego 状态通路压掉了前车刹车；navtrain 需刹车的行 84 % 历史已在减速 → 起刹靠历史捷径。第 409 条：起刹行历史拉平（SIMP-F）在 PAI 上
  lead_break +0.15、修 3 个 STOP 零分；ROUND2 预登记把它写成 `--flat-brk <SIMP-G 起刹行表> --flat-p 0.5`。拉平作用在 navtrain 行上，即 nuPlan 域的数据，
  所以应当能搬到 nuPlan 赛道。第 404 条：速度复制（惯性）在所有头里都有。
- 第 399 / 400 条：STAND-R（× 1）在 pilot B2 上把红灯线易题零分减半，但起步护栏不过。开放问题：拉平（教「看到前车刹就开始刹」、不抄历史的减速）是否抵消
  STAND-R 的起步代价，或至少不加重。

## 臂（同一命令，只差下表）

底命令 = B2-s1 训练时的原命令（盒上 `runs/defect_fix/stand_b/cmd_FHS.txt`，on1_JTX / JTX00 / on-k 70 / STAND W_S 0.0068 / FREE-M 0.001 move 0.95 /
seed 1），只换 `--tag`、`--init RB0-F-s0` → `--init STANDFMXH-L18-s1`、`--steps 9500` → N、`--warmup 500` → W（`nu_flat.sh check` 打印 diff 并核对新 token）。

| 臂 | tag | 加的开关 |
|---|---|---|
| CTRL | `B2C-L18-s1` | 无（B2 配方续训） |
| F | `B2F-L18-s1` | `--flat-brk runs/defect_fix/simp/flat_onset.txt --flat-p 0.5`（ROUND2 的设置） |
| FR | `B2FR-L18-s1` | F 的开关 + `--on-stand-roll 3.0`（第 399 条 R1：W_S 不变 = × 1，同一 stand_gap 标签、同一行） |

## 步数规则（写定，不再改）

- 每批 128 行 = 13 yaw-rate + 70 池行（on-k 70）+ nN = 45 个 navtrain 普通行；普通行里 d_frac 0.25 是 anchor（从不拉平），所以每批可拉平的抽取是
  0.75 × 45 = 33.75 行，在 navtrain 训练行 T（训练器记录的 `flat_brk.train_rows_total`，SIMP-G 记的是 101 499）里均匀抽。
- **N = round(T / (0.75 × nN), 到 100)** → 每个起刹行作为可拉平行平均被抽到约 1 次（p 0.5 下平均拉平约 0.5 次；连 anchor 在内约 1.33 次）。
  T = 101 499 时 N = 3 000。简报预期约 2 000 步，差在 anchor：按「可拉平的抽取」数，规则给 3 000。实际 T 由 smoke 的 DONE 读出，写 `results/nu_flat/steps.json`。
- W = round(0.05 N)（B2 全长 500 / 9 500、B2 pilot 50 / 1 000 都是 5 %）= 150。lr-scale 0.1、cosine 到 floor 0.1 不变。三个臂同 N、同 W。
- 期望每批拉平行数 0.5 × 45 × 0.75 × (37 847 / 101 499) ≈ 6.3。

## 训练前的闸（不过不训练）

1. **关掉即逐位相同**（60 步 eager，CTRL 的命令）：(a) B2-s1 的训练代码 5858f899（`git worktree`，**从旧工作树运行**）对当前代码，不加新开关；
   (b) 同一旧运行对当前代码加 `--flat-brk <表> --flat-p 0`。两项 `df.py bitcmp` 标量、权重、dev ADE、抽样 hash 全部逐位相等，且 meta.json 的 git sha 与 cwd
   两边不同 → `results/nu_flat/identity.json`。
2. **smoke**（FR 的命令 60 步 eager）：标量全有限；`flat/rows` 均值在期望 6.3 的 0.6–1.4 倍；`on/lon_standr_pos` 至少一步 > 0；`flat_brk.p` = 0.5；
   meta 的 init = `STANDFMXH-L18-s1` → `results/nu_flat/smoke.json`。
3. STAND-R 的离线闸 G1–G3 在同一池同一标签上已过（`results/stand_r/gate.json`），不重跑；`stand_gap.npz` 原样复用。

## 读数（每臂，一次提交，原生）

- 大考场 held 18 块原生（`jt.sh aread <tag> off c`，2 228 窗，第 371 条的选模型尺子）。
- held 460 @18（`rm_chain.sh read`，6 块）。
- 等待集 v2（`wait_set.sh readc`，a / b 变体）。另读一次 B2-s1 的等待集作参照（盘上没有）。
- 盒上汇总：`free_m.py ext / arena`（对 CTRL，`--ref STANDFMXH-L18-s1`）、`stand.py drift / pilot`（对 CTRL）。Mac 上：冻结配池 PCS 与易题成因
  （`stand_r.py` 的 `_ctx` / `classes`，同 2 106 冻结题、同分层与成因规则），全 18 块与 716 窗子集（d399 的 0 / 3 / 6 / 9 / 12 / 15 块，按
  `results/stand_r2/arena_windows.csv.gz` 里对照的窗口定义）各算一次 → `results/nu_flat/gate.{json,md}`。
- **不跑** lead_break / inertia：两个脚本只在 PAI 状态上（lead_break = RB0-F 的 PAI 281 撞车状态，inertia = P160），没有 nuPlan 状态的入口；
  nuPlan 上与之对应的是 held 460 前撞与大考场 AF-front，已在下面的护栏里。

## 线（每臂对 CTRL；读之前写定）

**护栏（有约束力，d399 / d400 第 1 段的线原样）**：
- 等待集 b 起步数 other_lead / red_queue ≥ 0.95 × CTRL；b 起步中位延迟 ≤ CTRL + 0.15 s（两边先取到 0.001 s 再比，免得第 400 条那种 0.567 对 0.417 + 0.15
  的浮点误判）；a 等住率 ≥ CTRL − 0.03。**起步护栏是 binding。**
- held 460 零分 ≤ CTRL + 2、前撞 ≤ CTRL + 2；held 460 右漂（非转弯 k 11–17）> −0.05 m。
- 716 窗子集上（同 d399 的口径）：AF-front ≤ CTRL + 2；红灯线以外每类易题失分 ≤ CTRL + 2；慢部分分窗 ≤ CTRL + 3；k = 9 速度比 ≥ CTRL − 0.03。

**主读数（全 18 块）**：
- 冻结配池 PCS ≥ CTRL + 15（d399 主 2 的线，搬到全集）；均分配对差与 log 聚类 CI、零分，对 CTRL 与对存档 B2-s1 都报。
- FR 另加：全 18 块红灯线易题零分 ≤ CTRL − 3（B2 全集只有约 9–10 窗，d399 的「对照 − 6」是按 pilot 的 30 窗定的；× 1 在 pilot 上减半，这里要求约三分之一）。
  716 子集的红灯线数照 d399 的线（≤ 对照 − 6）只作描述：强模型在子集上只剩约 3 窗，这条线到不了。

**判定**：护栏全过 + 主读数过 = PASS；否则 FAIL 并写明哪条。本 lane 不做采用、不起草镜像，交主会话。

**预期方向（写在读之前）**：CTRL ≈ B2-s1（续训在配方 lr 上重新热身，可能有 ±0.005 的漂移，这正是要 CTRL 的原因）。F：前撞 / AF-front 不增或降、等待集起步不降
（PAI 上 G-LAUNCH 0.708 → 0.831），红灯线不动。FR：红灯线零分约减半；起步低于 F。若 FR 过起步护栏，就是拉平抵消了 STAND-R 的起步代价；若 FR 不过而 F 过，
记「拉平不抵消」。种子散布未知（一个 seed）：大考场均分 |差| < 0.005 记为「分不开」（B2 s0 / s1 差 0.001，第 391 条）。

## 优先级与花费

- 池里没有抢占。本 lane 训练 PRIO 4、读数 5，ROUND2 是 12 / 13，所以排队时 ROUND2 永远先起；最坏情况是 ROUND2 的作业等本 lane 一个已在跑的读数结束
  （约 15–25 分钟，池的 15 分钟头部保留规则兜底）。
- 估：训练 3 × 约 10 分钟（9 500 步 compile 约 27 分钟）；每臂读数 ≈ 大考场 18 块 4.3 + held 6 块约 1.4 + 等待集 8 块约 1 ≈ 6.7 job 小时，三臂加 B2-s1
  等待集 ≈ 21 job 小时 ≈ 5.5 卡时（4 读 / 卡）；identity + smoke 约 0.2 卡时。空池 5 卡约 1.5–2 小时墙钟。
