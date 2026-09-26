# 多榜前 10 交集：哪些方法族在多个榜上都靠前，装机 smoke，考试预登记

状态: draft（榜单汇总、装机与 smoke 已完成；**考试一个都没跑**，交给中央执行员按下文预登记执行）
主题: ../research/leaderboard-vs-ability.md
前提: [hack 审计](2026-09-24-hack-audit/README.md)、[榜单文本分析](2026-09-24-leaderboard-text-analysis/README.md)（204 个跨榜分数）、decisions 第 35、38 条
附件: [boards.md](2026-09-26-top10-intersection/boards.md)（12 张榜逐条前 10 明细）、`2026-09-26-top10-intersection/boards/*.csv`（完整字段，含 notes）、
`2026-09-26-top10-intersection/raw_snapshots/`（HF NAVSIM 榜原始抓取）；smoke 代码 `scripts/top10_smoke/*_smoke.py`，表格生成 `scripts/make_top10_tables.py`

## 目标

用户的判断是：真正好的模型应该在开环榜上都靠前，或者开环、闭环都靠前。之前的审计每个榜只取了前 4–7，且漏了几个榜的第一名。
这里把 8 类榜单（12 张子榜）都放大到前 10，按方法族（同一代码库，或同一作者组的衍生）求交集，挑出 ≤ 6 个值得研究的族，
把其中代码 + 权重开源、box 上还没有的装好并 smoke，最后给出进我们考卷的预登记方案。

## 0. 一页结论

1. **上次漏掉的第一名**（全部本次补齐，出处见第 1 节）：Bench2Drive 是 BridgeDrive（96.34，2026-09-24 才进官方榜），不是 TFv6；
   NAVSIM v1 与 navhard 的 HF 官方榜第一是匿名队 EABOT.AI&NJU（95.61 / 58.63），可查身份的是 FPD-Drive-Pro（95.57 / 58.07），TOAD 实际第 8；
   WOD-E2E 是 ZSD-Titan（8.1667），DriveMA-4B 第 5，RAP 掉出前 10；CARLA LB2 MAP 赛道是 Kyber-E2E（12.455）；HUGSIM 是 DriveZero-Scale（46.6，但与 WA-JEPA 的 436 场景协议不同，同协议下第一仍是 WA-JEPA）。
   漏统计的根因是上次用 sota2 转录和论文表，没有直接拉官方榜的 JSON（HF competition space、Waymo 页面、Bench2Drive README、leaderboard.carla.org）。
2. **没有一个族同时在「真实数据开环」和「CARLA 闭环」两类榜上都稳居前 10。** CARLA 系（B2D / Longest6 / LB2）由 TransFuser-LEAD 和 SimLingo 两族主导；
   真实数据系（NAVSIM / HUGSIM / WOD）被 DrivoR 系和 AFARI（WA-JEPA）占据；nuScenes 前 10 与其他所有榜零交集；nuPlan 三榜全是吃 GT 目标 + 地图的 planner，与传感器榜零交集。
   唯一两边都挂名的是 SparseDrive（B2D #10、navtest-v2 #7，但 B2D 版本读 CARLA 真值位姿）。**跨榜交集主要由训练数据和输入契约决定**，不是能力的直接信号。
3. HUGSIM 前 10 几乎全是 navtrain 上训、零样本进 HUGSIM 的 NAVSIM 模型，所以「NAVSIM + HUGSIM 双榜」不是两份独立证据；它是唯一一个真实外观的闭环，仍有价值。
4. 建议研究的 6 个族（第 3 节）：**DrivoR 系、AFARI（WA-JEPA）、TransFuser-LEAD（TFv6 + BridgeDrive）、SimLingo（SimLingo + BLUE）、SparseDrive、NVlabs Hydra（ZTRS，作对照）**。
   本次新装 6 个模型，全部装通并在真实输入上 smoke 过：DrivoR、WA-JEPA、SparseDriveV2、ZTRS、BridgeDrive、BLUE（第 4 节）；TFv6、SimLingo、Alpamayo 1.5、openpilot、AutoVLA 核对能跑。

## 1. 各榜前 10（2026-09-26 快照）

每格是名次对应的方法，括号里是族的缩写；`*` 表示无代码或无权重。逐条的分数、日期、出处、代码 / 权重、license、输入、hack 机制、是否本次新补在 [boards.md](2026-09-26-top10-intersection/boards.md)。
族缩写：**DR** = DrivoR 系（valeo DrivoR/TOAD，及在其架构 / 代码上衍生的 Xiaomi DriveZero、Mila Gigapixel-DrivoR、NUS NTR）；**AF** = AFARI（WA-JEPA / ChainFlow-VLA / CoWorld-VLA）；
**LEAD** = TransFuser-LEAD（TF++、TFv5/v6、LTF、BridgeDrive，及他组衍生的 LTFv7）；**SL** = SimLingo（SimLingo / CarLLaVA、BLUE，及按数据 / agent 归入的 LinkVLA、FIVE-VLA、SteerVLA、RoG-DAgger、TakeVLA）；
**SD** = SparseDrive；**HY** = NVlabs Hydra（Hydra-MDP / GTRS / ZTRS）；**FPD** = FPD-Drive（NIO/HKUST）；**UV** = UniAD/VAD 系；**ZR** = Zeron；**MA** = DriveMA；**VAIL** = 基于 Alpamayo 1.5 的 VAIL；**anon** = HF 匿名队。

| 榜（主指标，出处） | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| nuScenes 开环（avg L2，各论文表；协议不统一，全部用了 ego status） | IK-VLA* 0.06† | OWMDrive* 0.18 | AutoDrive-R² 0.19 | Senna*(UV) 0.22 | Reasoning-VLA* 0.22 | SparseOccVLA 0.23 | EvoDriveVLA 0.26 | ORION+plug-in* 0.26 | VLA-World* 0.26 | DiMA+*(UV) 0.27 |
| WOD-E2E test（RFS，[官方榜](https://waymo.com/open/challenges/2025/e2e-driving/) JSON） | ZSD-Titan*(ZR) 8.167 | ZSD*(ZR) 8.090 | PlusAI-WorldVLA* 8.087 | VAIL+*(VAIL) 8.082 | DriveMA-4B*(MA) 8.079 | DriveMA-2B(MA) 8.075 | VAIL*(VAIL) 8.061 | Zero-1*(ZR) 8.048 | NTR ens.*(DR) 8.046 | TTVLM* 8.043 |
| NAVSIM v1 navtest（PDMS，[HF 榜](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)） | EABOT*(anon) 95.61 | FPD-Drive-Pro*(FPD) 95.57 | CooWAIM*(anon) 95.43 | minfish*(anon) 95.37 | DriveZero-Scale*(DR) 95.31 | iDriveVLA*(anon) 94.95 | EvaDrive* 94.9 | TOAD+DrivoR(DR) 94.88 | FPD-Drive*(FPD) 94.88 | ChainFlow-VLA(AF) / TransDiffuser* 94.85 |
| NAVSIM v2 navhard two-stage（EPDMS corrected，[HF 榜](https://huggingface.co/spaces/AGC2025/e2e-driving-navhard)） | EABOT*(anon) 58.63 | FPD-Drive-Pro*(FPD) 58.07 | CooWAIM*(anon) 57.61 | guest9527*(anon) 57.41 | Aqua10086*(anon) 57.41 | zzzzz*(anon) 57.24 | DriveZero-Scale*(DR) 56.81 | TOAD+DrivoR(DR) 56.51 | FPD-Drive*(FPD) 55.87 | DriveFuture* 55.5 |
| NAVSIM v2 navtest（EPDMS，只有论文自报，无官方榜） | WA-JEPA(AF) 91.7 | NTR*(DR) 90.9 | BeyondDrive 90.5 | CLOVER / Discrete-WAM* 90.4 | — | SimWAM 90.2 | SparseDriveV2(SD) / READ* / LTFv7(LEAD) 90.1 | — | — | CoWorld-VLA(AF) 90.0 |
| Bench2Drive（DS，[官方榜](https://github.com/autonomousvision/Bench2Drive-Leaderboard)） | BridgeDrive(LEAD) 96.34 | TFv6(LEAD) 95.28 | LinkVLA*(SL) 91.01 | FIVE-VLA*(SL) 90.95 | SteerVLA*(SL) 90.71 | BLUE(SL) 90.58 | RoG-DAgger*(SL) 90.34 | TakeVLA*(SL) 89.72 | AutoMoT+ (部分开源) 89.42 | SparseDriveV2(SD) 89.15 |
| Longest6 v2（DS，论文表；只找到 7 个学习式传感器方法） | TFv6(LEAD) 62 | RoG-DAgger*(SL) 44 | CLEAR* 39.89 | BLUE(SL) 36 | TFv5/TF++(LEAD) 23 | SimLingo(SL) 22 | HiP-AD 7 | — | — | — |
| CARLA LB2 官方 test（DS，[leaderboard.carla.org](https://leaderboard.carla.org) JSON；SENSORS / MAP） | S: CarLLaVA(SL) 6.87；M: Kyber-E2E* 12.46 | S: TF++(LEAD) 5.18；M: CarLLaVA(SL) 6.25 | S: CaRINA* 1.23；M: TF++(LEAD) 5.56 | 其余为匿名或 < 3 DS | | | | | | |
| nuPlan Val14（CLS-NR；吃 GT 目标 + 地图） | Mosaic 95.56 | LAP* 95.03 | CarPLAN* 95.0 | FlowDrive* 94.81 | Plan-R1* 94.72 | CoPlanner* 94.45 | CaRL 94.38 | EMoE* 94.35 | Flow Planner 94.31 | Diffusion Planner 94.26 |
| nuPlan Test14-hard（CLS-NR） | CarPLAN* 82.2 | FlowDrive* 81.86 | EMoE* 80.96 | G2DP* 80.76 | SDD* 80.32 | BIBeR* 80.30 | EMoE w/o post* 80.12 | PLUTO 80.08 | G2DP w/o refine* 80.05 | DFP-FM* 79.43 |
| interPlan（score） | RAD-LAD* 74 | RAD（规则）72 | TerraZero* 70.87 | SAH-Drive 64 | SPDM（规则）63.66 | Diffusion-ES* 57.41 | BIRDriver* 55.29 | Mosaic 54.1 | HybridLLMPlanner* 53 | PLUTO 48.92 |
| HUGSIM（HD-Score ×100；至少 3 种协议） | DriveZero-Scale*(DR) 46.6 | WA-JEPA(AF) 44.62‡ | DriveZero*(DR) 39.4 | Gigapixel-DrivoR*(DR) 38.5 | DrivoR+SimScale(DR) 38.1 | DrivoR(DR) 35.7 | ZTRS(HY) 32.9 | UniAD(UV) 32.7 | MM-Future* 32.3 | Latent-WAM* 28.9 |

† IK-VLA 的 1 s L2 = 0.04 m，且只评了 5869/6019 个样本，基本可判无效；可信度高的第一是 OWMDrive 0.18（无代码），有代码 + 权重的最好是 SparseOccVLA 0.23。
‡ WA-JEPA 用 436 场景加权平均且修过 HUGSIM 控制器（审计 HUG-WA-001），DriveZero / DrivoR 用 345 场景协议，数字不直接可比；同一个 UniAD 在三种协议下是 32.7 / 31.24 / 28.9。

**本次新补**（相对 204 个跨榜分数 + 审计抽样）：nuScenes 前 10 里 8 个、WOD 前 10 里 7 个、NAVSIM v1 前 10 里 10 个、navhard 前 10 里 8 个、navtest-v2 大部分、
B2D 的 BridgeDrive / TakeVLA / AutoMoT+、Longest6 的 CLEAR 与 BLUE 分数、LB2 的 Kyber-E2E 12.455 与匿名条目、nuPlan 三榜全部（base 里没有 nuPlan）、HUGSIM 的 DriveZero 系 / ZTRS / MM-Future / Latent-WAM。
逐条 `new` 列见 [boards.md](2026-09-26-top10-intersection/boards.md)。

**读这张表要记住的口径问题**（逐条出处在 CSV 的 notes 列）：
- NAVSIM v1 有 9 个学习式条目超过 human 94.8；能查到机制的新条目（FPD-Drive、DriveZero、ChainFlow、CLOVER、EvaDrive）都用 learned PDM 子分数 scorer 选轨（`metric_proxy_candidate_selection`，依据是论文 / 代码，未正式审计）。
  TransDiffuser 94.85、EvaDrive 94.9 用其自报子分数按公式只能算到约 92.6 / 93.4，存疑。前 10 里 4 个（v1）/ 6 个（navhard）是匿名队，输入是否 non-privileged 无法核验。
- navhard 前 8 个学习式条目全部超过 privileged PDM-Closed 56.6（上一轮的 v2 榜首判断「PDM-Closed #1」因此作废）。
- B2D：SteerVLA 论文 90.71、官方榜四舍五入为 91；SimLingo / BLUE 自带评测器注释掉了 4000-tick 截断（simlingo#44），BLUE 另把完成阈值改为 90%（官方 99%，`atomic_criteria.py:1553`）；
  BridgeDrive 用 route + target speed 控车并带 creep（`sensor_agent_bridgedrive.py:840-905`）；SparseDriveV2 的 B2D agent 读 CARLA lidar actor 真值位姿（`sparsedrive_b2d_agent.py:360-361`，审计 B2D-SPARSE-001 由本次代码复核确认）。
  B2D 第 2–10 名总分差 < 7 DS，按第 38 条单次噪声 ±2.3 DS，名次本身多数分不开。
- CarLLaVA 就是 SimLingo-BASE，上次把 SENSORS 6.87 与 MAP 6.25 当成两个方法。Kyber-E2E 12.455 这次提交无文档，是模块化栈不是 E2E。
- WOD 新条目全部只有榜单行（无论文、无代码），single / ensemble、是否拿 RFS 当奖励都不知道；能确认拿 RFS 当奖励的只有 DriveMA（审计 REAL-WOD-001）和 Poutine（论文）。
  2025 挑战赛的奖项不按 RFS 排（1st UniPlan 7.779），Poutine 7.986 是截止时最高分但只拿 special mention。
- nuScenes 前 10 **全部用了 ego status**；纯 ego 的 AD-MLP 就有 0.29，所以 0.29 以下的差距要先扣掉 ego prior（第 35 条 (a)）。不用 ego 的最好是 GeRo 0.27。
- nuPlan 前排几乎全是 hybrid（学习式候选 + PDM 式规则 scorer / refine），纯学习且无后处理的只有 CaRL（Val14 #7）和 TerraZero（interPlan #3）；R2LPL 83.51 在测试场景上做 rollout 更新，没排名。

## 2. 交集：方法族 × 榜单

格子里是该族最好成员的名次（同族多条列前两个）；「—」是不在前 10。开环 = nuScenes、WOD、NAVSIM v1、navhard、navtest-v2；闭环 = B2D、Longest6、LB2、nuPlan（三张子榜算一个）、HUGSIM。
排序：先按上榜总数，同数按「代码 + 权重开源」优先。「开源」指该族至少一个前 10 成员两者都有。

| 族（代表成员） | nuSc | WOD | v1 | navhard | navtest-v2 | B2D | L6 | LB2 | nuPlan | HUGSIM | 开环 | 闭环 | 开源 | box |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|:--|:--|
| **DR** DrivoR 系（DrivoR / TOAD；DriveZero*、Gigapixel*、NTR*） | — | 9 | 5, 8 | 7, 8 | 2 | — | — | — | — | 1, 3 | **4** | 1 | 是（valeo DrivoR / TOAD；衍生成员都没有权重） | DrivoR 本次新装 |
| **LEAD** TransFuser-LEAD（BridgeDrive、TFv6、TF++；LTFv7） | — | — | — | — | 7（LTFv7，他组衍生） | 1, 2 | 1, 5 | S2 / M3 | — | — | 1 | **3** | 是 | TFv6 已有；BridgeDrive 本次新装 |
| **SL** SimLingo（BLUE、SimLingo/CarLLaVA；5 个无代码成员） | — | — | — | — | — | 3, 4 | 2, 4 | S1 / M2 | — | — | 0 | **3** | 是（SimLingo、BLUE） | SimLingo 已有；BLUE 本次新装 |
| **AF** AFARI（WA-JEPA、ChainFlow-VLA、CoWorld-VLA） | — | — | 10 | — | 1, 10 | — | — | — | — | 2（436 协议下 1） | 2 | 1 | 是 | WA-JEPA 本次新装 |
| **FPD** FPD-Drive（NIO/HKUST） | — | — | 2, 9 | 2, 9 | — | — | — | — | — | — | 2 | 0 | 否（仓库只有 README） | — |
| **SD** SparseDrive（SparseDriveV2） | — | — | —（92.2，第 20 名外） | — | 7 | 10 | — | — | — | — | 1 | 1 | 是（NAVSIM 与 B2D 两套权重） | NAVSIM 版本次新装；B2D 版未装（mmcv 1.7 / torch 2.0 栈） |
| **UV** UniAD/VAD 系（UniAD；Senna*、DiMA+*） | 4, 10 | — | — | — | — | — | — | — | — | 8 | 1 | 1 | 部分（UniAD / VAD 原版 nuScenes 权重；前 10 的衍生成员无） | — |
| **HY** NVlabs Hydra（ZTRS、GTRS） | — | — | — | 第 15（去掉匿名队后第 10） | — | — | — | — | — | 7 | 1（弱） | 1 | 是 | ZTRS 本次新装 |
| **anon** EABOT / CooWAIM（HF 匿名） | — | — | 1, 3 | 1, 3 | — | — | — | — | — | — | 2 | 0 | 否 | — |
| **VAIL**（基于 Alpamayo 1.5 + Qwen3-VL） | — | 4, 7 | — | — | — | — | — | — | — | — | 1 | 0 | 否（底座 Alpamayo 1.5 开源） | Alpamayo 1.5 已有 |
| **ZR** Zeron（ZSD、Zero-1） | — | 1, 2 | — | — | — | — | — | — | — | — | 1 | 0 | 否 | — |
| **MA** DriveMA | — | 5, 6 | —（91.2） | — | — | — | — | — | — | — | 1 | 0 | 是（仅 2B） | — |
| nuPlan planner 各族（KIT FlowDrive/Mosaic、CarPLAN、EMoE、PLUTO、Diffusion Planner、G2DP、RAD-LAD、autonomousvision PDM / CaRL） | — | — | — | — | — | — | — | — | 各 1–3 张子榜 | — | 0 | 1 | 部分 | —（输入是 GT 目标 + 地图，不吃传感器） |
| 只上一个榜的其余方法（nuScenes 8 个、WOD 3 个、v1 4 个、navtest-v2 5 个、B2D AutoMoT+、L6 CLEAR / HiP-AD、HUGSIM 2 个） | | | | | | | | | | | 1 | 或 1 | 多数否 | — |

**读法**

- **开环、闭环都靠前的族：严格意义上没有。** 若把 HUGSIM 算闭环，DR、AF、HY、UV 四族算「两边都有」；但 HUGSIM 前 10 除 UniAD 外全是 navtrain 上训的 NAVSIM 模型零样本迁过去的，
  和 NAVSIM 共享训练数据、相机契约和 PDM 式评分（HD-Score 本身就是 NC × DAC × (TTC, C)），所以它更像 NAVSIM 的「真实外观、会反应的版本」，不是一个独立的能力来源。
  若只认 CARLA 闭环，两边都挂名的只有 SD（B2D #10 + navtest-v2 #7），而它的 B2D 版读真值位姿，LEAD 在开环上只靠他组衍生的 LTFv7 挂名（LEAD 自己的 LTFv6 是 v1 85.4 / navhard 28.3）。
- **只在一类榜上靠前的族**：CARLA 闭环由 LEAD 与 SL 两族主导（B2D 前 8 名全是这两族、Longest6 前 7 名里 5 个、LB2 两条赛道端到端方法的前两名），这两族在所有真实数据榜上都不在前 10；
  真实数据开环被 DR、AF、FPD 与匿名队占据；WOD 的前 10 与 NAVSIM 只共享 DR（NTR）；nuScenes 前 10 与其余所有榜零交集，而且全部靠 ego status。
  所以「多榜靠前」首先是「训练数据属于哪个生态（CARLA 专家数据 / nuPlan-navtrain / WOD / nuScenes）」，其次才可能是能力。第 35 条 (b)「跨榜排序只在同代码族内和代际分层上一致」在前 10 的尺度上仍然成立。
- **多榜覆盖最广的 DR 系，也是配方证据最多的一族**：它的成员全部靠 learned PDM 子分数 scorer 选轨；TOAD 的 CEM 在 v2 上把 5 个基座都收敛到 49–56（第 35 条），DrivoR 的长目标 v1 +0.6 / v2 −1.6（审计 NAV1-DRIVOR），
  NAVSIM 顶部 4 分基本是配方。它的多榜覆盖可能主要是「同一个 scorer 在同一套 PDM 式公式上反复拿分」。这正是要用我们的配对考卷去分开的。

## 3. 建议研究的 6 个族与理由

| # | 族（我们用的成员） | 为什么选 | 能力的证据 | 配方 / hack 的证据 | 要回答的问题 |
|:--|:--|:--|:--|:--|:--|
| 1 | **DR**（DrivoR v1 ckpt，可选 TOAD） | 上榜最多（开环 4 + HUGSIM）、衍生成员占 HUGSIM 第 1/3/4 | DrivoR SimScale 合成困难场景 +6.3 EPDMS（W1，C 级，预算未配平）；衍生 DriveZero 在 HUGSIM 零样本第一 | scorer 选轨、v2 手工改权、交集 warmup 调参、长目标（审计 3 个 `yes`）；TOAD 收敛 49–56 | 多榜覆盖在我们的 E 层配对（P5 v1 / I3）上还剩多少：scorer 会不会对突发车辆 / 行人翻转 |
| 2 | **AF**（WA-JEPA） | navtest-v2 第一、HUGSIM 436 协议第一、v1 靠 ChainFlow 进前 10；唯一「表征驱动」的多榜族 | V-JEPA 2 预训练 +5.7–6.4 PDMS（W1，C）；无 hack finding | HUGSIM 分数是在修过的控制器上跑的（HUG-WA-001），不能与原版控制器 baseline 直接比 | 视频预训练的世界模型在配对考卷上是否比 scorer 族更会反应；和我们第 24 条「V-JEPA 2 效应在 train split 上没活下来」对照 |
| 3 | **LEAD**（TFv6 已有 + BridgeDrive） | CARLA 三榜第一（B2D 1–2、Longest6 1、LB2 前 3） | TFv6 waypoint 通道 P5 翻转 39% / 按对 73%（第 32 条）；LiDAR +3.1 DS、SR +6.1 | route + target speed 接口 +14.3 DS（第 31 条）；拿分通道只翻 2%；BridgeDrive 同样用 route + target speed + creep 控车 | BridgeDrive 的 +1 DS 是网络（diffusion bridge）还是接口；它的 waypoint / target speed 两通道各翻多少 |
| 4 | **SL**（SimLingo 已有 + BLUE） | CARLA 三榜第 2 梯队且成员最多；**BLUE 是第 38 条里唯一在突发 hazard 上显著更好的方法**（SR 93.8%，比 SparseDriveV2 高 18.5 [7.7, 29.2]） | BLUE gate +5.51 DS / SR +8.91（W1，B 级） | 一半来自「关掉语言」；评测器删 4000-tick 截断、完成阈值 90%、creep；simlingo#43 DS +11 而 SR 不变 | BLUE 的突发 hazard 优势在 P5 配对上是否复现；gate 是否在 hazard 帧上改变输出 |
| 5 | **SD**（SparseDriveV2 NAVSIM 版） | 唯一在真实数据开环（navtest-v2 #7）和 CARLA 闭环（B2D #10）都进前 10、两套权重都公开的族 | 词表打分式规划，参数 50M，14 ms | B2D 版读真值位姿（B2D-SPARSE-001，本次复核确认）；标题即「Scoring is All You Need」 | 同一族在两个生态里的反应能力是否一致（先考 NAVSIM 版，B2D 版按需另装） |
| 6 | **HY**（ZTRS，对照） | HUGSIM #7、navhard 去匿名后 #10，GTRS 自述为 CVPR 2025 NAVSIM 挑战赛冠军（分数未查到；HF 2025 私榜第一是 SimpleVSF 53.06，两者关系未核） | — | **只用 PDM 奖励训练（EPO 策略梯度，无模仿）**，固定 8192 / 16384 条词表 + 8 个子分数头选轨（本次代码确认） | 纯指标训练、完全没有人类轨迹的 policy 有没有 E 层反应：是 scorer 族的「配方上限」对照 |

没选的：FPD-Drive、DriveZero、Zeron、匿名队（都没有代码或权重；**DriveZero 若放出 NAVSIM / HUGSIM 权重应补进 DR 族**）；UniAD/VAD（B2D 只有 42–46 DS，代际落后）；DriveMA（只上 WOD 一个榜，且 RFS 当奖励，审计 REAL-WOD-001）；
nuPlan planner（不吃传感器，进不了我们的考卷）。Alpamayo 1.5 与 openpilot 不在任何前 10 里，但 VAIL（Alpamayo 1.5 + Qwen3-VL）在 WOD 第 4 / 7，与第 34 条「Alpamayo 在 WOD 上超过 ego-only」方向一致。

## 4. 装机与 smoke（2026-09-26 09:00–09:50，全部是新的独立 env，没动已有 env）

所有 env 都是 `~/data/envs/<name>`（uv venv；box 上已有 env 都是 uv venv，照同一惯例），代码在 `~/data/third_party/<name>`，权重在 `~/data/models/<name>/`，smoke 输出在 `~/data/runs/top10_smoke/<name>/<时间>/summary.json`。
latency 口径同 [docs/baselines.md](../docs/baselines.md)：batch 1，5 次 warmup + 20 次计时，`torch.cuda.synchronize()`；计时范围写在表里（都不含 CPU 端特征构建）。测时各卡多为空闲，所以是偏乐观的数字。

| 模型 | env / 代码 commit | checkpoint（来源，license） | 装机耗时 | 依赖冲突与处理 | smoke 样本与结果 | latency mean / p50 / p95（范围） | 峰值显存 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| DrivoR | `drivor`，py3.10 torch 2.8 cu128；valeoai/DrivoR `fc6e5aa` | `drivor_Nav1_25epochs.pth`（[release](https://github.com/valeoai/DrivoR/releases/tag/model_weights)，PDMS 93.7 那一行；Apache-2.0）+ timm ViT-S DINOv2 reg4 | 约 10 min（env 69 s） | 上游 py3.9/torch 2.1 → torch 2.8 直接能跑；无编译扩展；GitHub release 直连卡住，走 Clash | navtest `0f206a62842b59b2`：ADE 0.20 m、FDE 0.19 m（CV 2.89 m） | 85 / 90 / 92 ms（`agent.forward`，fp32）；特征构建 136 ms CPU | 0.48 GB |
| WA-JEPA | `wajepa`，py3.10 torch 2.14 cu130；AFARI-Research/WA-JEPA `bec2966` | `model_state_dict.pt` 1.58 GB（[HF](https://huggingface.co/AFARI-Research/WA-JEPA)，Apache-2.0）；自带 V-JEPA 2.1 encoder，缓存里的 V-JEPA 2 用不上也不需要 | 约 12 min | navsim 钉的 torch 2.0.1 / numpy 1.23.4 去掉，navsim、nuplan-devkit `--no-deps` | navtest `8781cde1032354cb`：ADE 0.67 m、FDE 1.97 m | NAVSIM 评测路径（fp32）614 / 549 / 867 ms；HUGSIM 路径（bf16 autocast）151 / 151 / 152 ms | 3.7 / 4.7 GB |
| SparseDriveV2 | `sparsedrivev2`，py3.10 torch 2.8 cu128；swc-17/SparseDriveV2 `696ef77` | `sparsedrive_navsimv1_92p2.ckpt`（[HF](https://huggingface.co/wenchaosun/SparseDriveV2)，Apache-2.0） | 约 12 min（env 376 s） | 编译 deformable_aggregation（`TORCH_CUDA_ARCH_LIST=12.0`）一次过；torch 2.8 的 `weights_only` 需手动 load | navtest `b111bb8716b756d2`（9.1 m/s）：ADE 2.75 m、FDE 6.41 m，方向对、速度偏保守（第二个 log 2.73 m 同样） | 13.8 / 13.8 / 14.0 ms（`agent.forward`）；特征 84 ms CPU | 1.0 GB |
| ZTRS | `gtrs`，py3.10 torch 2.8 cu128；woxihuanjiangguo/ZTRS `50a52f2`（仓库也含 GTRS / Hydra-MDP） | `ztrs_vov.ckpt` 958 MB（[HF](https://huggingface.co/Zzxxxxxxxx/ZTRS)，Apache-2.0）；V2-99 init 取 HF 同源副本 | 约 5 min | 不装 xformers，补 diffusers；nuplan-devkit 用本地 v1.2 | navtest `ff8c1dd3f49254f6`（3.0 m/s 加速）：8192 词表 ADE 2.98 m（CV 3.89 m），直行但偏慢 | 68 / 68 / 69 ms（原样，含一次只进 loss 的 t−0.5 s 前向；去掉后 34 ms，输出逐位相同） | 0.87 GB |
| BridgeDrive | `bridgedrive`（复制 scout-tfv6 的 freeze + einops）；BridgeDrive `85aa089` + 作者钉的 lead `a41d116` | `model_BridgeDrive_m2_k60_0030.pth`（[HF](https://huggingface.co/liushu-ethz/BridgeDrive)，README 称 m2 更好；**哪个变体对应 96.34 未写明；仓库无 LICENSE 文件**） | 约 17 min | 无编译扩展；作者代码 3 处 bug 用运行时 shim 绕过（GPU 名单、未定义的 `debug_mode`、anchor 路径） | 自跑 B2D 路线 26966（Town05，路口右转）254 tick，RC 100%；dump 两帧离线前向：ADE 0.25 / 0.66 m（对照是它自己开出的路径，只是自洽性） | 118–131 ms（`ClosedLoopInference.forward`，含 20 步 ODE bridge 与两个 PID）；每 tick 含预处理 381 ms | 1.5 GB |
| BLUE | `blue`（照搬 simlingo env 的 90 条 pin）；George-Ling3/BLUE `6970cb6` | 仓库自带 `gate/weights/blue_simlingo_gate.pt`（sha256 对上，Apache-2.0）+ 已缓存的 SimLingo ckpt | 约 1 min（全部命中 uv cache） | 无；gate .pt 在 torch ≥ 2.6 下需 `add_safe_globals` | 自开 CARLA Town10HD 抓一帧（红灯前 4.7 m/s）：gate 0.032 → 关语言、直出动作，预测 2.8 m 内停车（autopilot 1.36 m），ADE 1.22 m | 95 / 94 / 97 ms（gate 判直出）；强制开语言 658 ms；预处理 60 ms | 2.2 GB |

作者代码里的附带发现（都会影响考试怎么读，出处是本次代码复核）：
- BridgeDrive 的 ODE 采样把第 10 个 route 点的 y 恒置 0（`model_diffusion_head_ddbm.py:371,383`），route 与 waypoint 两个 head 共用同一组 decoder query（`planning_decoder_bridgedrive.py:171-178`，自增被注释）；
  控车用 route + target speed，P(target speed = 0) > 0.9 时强制刹车（`open_loop_inference_bridgedrive.py:113`），creep 默认开、停牌规则默认关（`config_closed_loop.py:59,67`）。
- BLUE 的直出分支要做两次完整 prefill（一次给 gate，一次带驾驶 token）；creep：静止 800 tick 后强制油门 ≥ 0.4 共 15 帧（`agent_simlingo.py:864-877`）。
- ZTRS 选轨公式（`HydraTrajHead.forward`）：0.05·log softmax(rl) + 0.5·log σ(TLC / NC / DAC / DDC 各项) + 8·log(5σ(TTC) + 5σ(EP) + 5σ(LK))，comfort 预测了但不用。
  它的 checkpoint README 写 EPDMS 45.5，是旧协议；HF 榜 48.1 是修复后协议（boards 里两者分列）。
- WA-JEPA 的 NAVSIM 评测路径没读 config 里的 `tf32: true`（推测是 fp32 慢 4 倍的原因，未验证）。

**已在 box 上的模型只核对能跑**（2026-09-26 10:00 前后）：TFv6（`envs/scout-tfv6`，torch 2.8 cu128，CUDA 可用；最近一次闭环 `runs/tfv6_rules/B1.rep0`，09-25）、
SimLingo（`envs/simlingo`，transformers 4.46.3；`runs/simlingo-catalogue`，09-25）、Alpamayo 1.5（`third_party/alpamayo1.5/.venv` import 通过；`runs/wod_zeroshot`、`runs/navsim_zs` 里有 09-24 的 alpamayo run）、
openpilot（`envs/openpilot`，onnxruntime 1.30 带 TensorRT EP；`runs/p5_openpilot`，09-26 01:14）、AutoVLA（`envs/autovla`，torch 2.8 cu128）。都没有重新推理。

## 5. 考试预登记（交给中央执行员；**本文件不跑任何考试**）

### 5.1 每个模型进哪几张卷

| 模型 | P5 v1 BA 集配对翻转 | I3 HUGSIM 车辆配对 | WOD-E2E val 零样本 | NAVSIM navtest devkit | nuScenes |
|:--|:--|:--|:--|:--|:--|
| DrivoR | 进（缺后视：B0 槽喂黑图，标「适配折中」） | 进（先补渲 CAM_BACK，见 5.2） | 进 | 进（复现 PDMS 93.7；v2 ckpt 复现 EPDMS） | 进（需补解压 CAM_BACK） |
| WA-JEPA | 次要（缺后视 + 0.5 s 历史不在 5 Hz 网格上，标「domain 混杂」） | 进（补渲 CAM_BACK + 10 Hz） | 进 | 进（复现 navtest EPDMS 91.7，用它的评测路径原样） | 进（需补解压 CAM_BACK） |
| SparseDriveV2 | **进，契约直接对得上** | **进** | 进 | 进（复现 PDMS 92.2） | 进 |
| ZTRS | **进** | **进** | 进 | 进（复现 navhard EPDMS 48.1） | 进 |
| BridgeDrive | 进，但要**重录**：P5 recorder 只存了 TFv6 的输出（`tf_*` 列），没存 TFv6 rig 的原始输入 | 不进（LiDAR + 4 radar 契约） | 不进 | 不进 | 不进 |
| BLUE | 进，同样要重录（单前视 1024×512、FOV 110） | 不进 | 不进（单前视 110° 与 WOD 47° 针孔差太多） | 不进（BLUE 自报的 NAVSIM 87.0 不是这个 agent） | 不进 |

BridgeDrive 与 BLUE 另建议一项闭环：Bench2Drive 官方 220 路线，**用官方评测器（保留 4000-tick 截断、完成阈值 99%）**，与它们自带目录各跑一次（第 35 条 7.4 的实验 2，BLUE 属于 SimLingo 系）。这一项不在本文件的判据范围内，另立 todo。

### 5.2 输入契约怎么对到我们的记录

| 记录 | 我们有什么 | 对 NAVSIM 式模型（DrivoR / WA-JEPA / SparseDriveV2 / ZTRS） | 对 CARLA 原生模型（BridgeDrive / BLUE） |
|:--|:--|:--|:--|
| P5 v1 BA（`$DATA_DIR/runs/p5v1/gen-ba/attempts/<route>/<n>/cams/<front,front_left,front_right>/<frame>.jpg`，索引 `processed/carla_p5v1_ba`） | 5 Hz、Waymo 式标定三路前视（左右 ±45°）；past / future 为 P4 / WOD 约定 | f0 ← front，l0 / r0 ← front_left / front_right（nuPlan 的 L0 / R0 约 ±55°，视角差预登记为适配损失）；SparseDriveV2 按真实标定喂 `projection_mat`；ZTRS 用它的裁剪拼接规则拼全景；DrivoR 的 B0 与 WA-JEPA 的 back 喂黑图。命令：route intent → NAVSIM 4 维 one-hot（left / straight / right / unknown）；ego vx, vy, ax, ay 取自 `past.npy`。WA-JEPA 的 4 帧 0.5 s 历史取最近的 5 Hz 帧（t, t−0.4, t−1.0, t−1.4 s，最大误差 0.1 s） | 5 Hz 三路图用不上。按 TFv6 shadow 的方式在 recorder 上挂它们自己的 rig **重录**：BridgeDrive = 3×384² FOV 60 相机（yaw ±54.5°）+ 2 LiDAR + 4 radar + 3 个 target point，要用作者钉的 lead `a41d116`；BLUE = 单前视 1024×512 FOV 110（x = −1.5, z = 2.0）+ 2 个 target point + 速度文本。P5 生成是确定性的（v0 的确定性检查），重录同一批 322 个 run |
| I3（`processed/hugsim_pairs/scenes/<key>/<world>/cams/<front,front_left,front_right>/<frame>.jpg`） | 5 Hz、800×450、HUGSIM 前三路，原点在前相机 | 同上映射；DrivoR / WA-JEPA 需要后视：I3 渲染器按 HUGSIM rig 补渲 CAM_BACK（WA-JEPA 自己的 HUGSIM 适配器就用 CAM_BACK），WA-JEPA 另需 0.5 s 间隔的历史，建议整体重渲为 10 Hz 四路；输出轨迹从后轴换到前相机原点 | 不适用 |
| WOD-E2E val（479 rater 帧 + ADE-extra，10 Hz，8 路针孔） | FRONT、FRONT_LEFT/RIGHT（±45°）、SIDE（±90°）、REAR 全有；intent 命令 | F0 ← FRONT，L0 / R0 ← FRONT_LEFT / RIGHT（或按 Alpamayo 适配器的纯旋转重投影到 ±55°，二选一，跑前写死），B0 ← REAR；2 Hz 历史从 10 Hz 精确取帧；intent → 4 维 one-hot；ego 取自 WOD 的 4 Hz 历史。**这些模型只出 4 s（2 Hz 或 10 Hz），RFS 要 5 s 点：预登记用最后两点匀速外推到 5 s**，这是已知对它们不利的折中 | 不适用 |
| NAVSIM navtest / navhard | 原生 | 原生；它们都是 2 Hz 输入训出来的，第 36、37 条「2 Hz 协议惩罚」只针对 openpilot 这类原生高帧率模型，不适用于这四个 | 不适用 |
| nuScenes val main（4636） | 关键帧 2 Hz；box 上只解压了 FRONT / FRONT_LEFT / FRONT_RIGHT / BACK_LEFT / BACK_RIGHT，**没有 CAM_BACK** | F0 / L0 / R0 ← CAM_FRONT / FRONT_LEFT / FRONT_RIGHT（nuScenes 侧前约 ±55°，与 nuPlan 接近）；B0 ← CAM_BACK（先 `scripts/extract_nuscenes.sh trainval samples/CAM_BACK`）；命令沿用 VAD converter 规则（GT 3 s 横向位移，属于 `future_label_conditioning`，与 nuscenes-physicalai 同口径）；ego 取 CAN bus | 不适用 |

输出到考卷量的换算：P5 / I3 的 v2 = 1.75 → 2.0 s 位移 / 0.25 s。2 Hz 输出的模型（DrivoR、WA-JEPA、SparseDriveV2）先对 (0, 0) + 8 个点做三次样条插到 0.25 s 网格再算；ZTRS 是 10 Hz、BridgeDrive / BLUE 的 waypoint 是 0.25 s 间隔，直接算。
BridgeDrive 与 TFv6 一样两个通道都报：waypoint 通道（2 s 处）与 route + target speed 通道（target speed 标量），P5 v1 的「每个考生固定申明读数通道，多通道全部报」照用。BLUE 报 speed waypoints，另记每帧 gate 是否开语言。

### 5.3 判据（照抄现有 todo，不新设门槛）

- **P5 v1 BA 集**：照抄 [p5-v1-e-layer](2026-09-24-p5-v1-e-layer.md) 的「指标与判据」与「读法」：纵向定向翻转率（expert 真反应的帧里，考生 Δ 与 expert 同号且超过考生自己在 null 上的 95 分位数的比例，逐帧与按对各报）、
  横向定向翻转率（只在 expert 横向反应的帧上）、反应类别一致率（4×4 混淆矩阵）、null false-flip；按 family 分报加合并，CI 按路线整组 bootstrap。
  「有 E 层能力」= 纵向或横向翻转率的 CI 下界明显高于它自己的 null false-flip，且反应类别一致率高于 expert 类别分布的随机基线。BA 集为主判定集，PDM-Lite 集只报按对（第 42 条 E4 的决定）。
  judge 是 `p5_exam.exam` 原样。TFv6 规则触发帧单独报的做法推广到 BridgeDrive（creep）与 BLUE（creep）。
- **I3**：照抄 [i3-hugsim-pairs](2026-09-25-reactivity-program/i3-hugsim-pairs.md)「现有考生零样本考试」一节：judge 是 `p5_exam.exam` 原样（去掉 TFv6 列），τ 由 I3 的 24 个 null 场景定，CI 按场景 bootstrap；
  报合并与逐 family（static / cutin / oncoming）的定向翻转、反方向、非反应帧误翻、样本外 null false-flip。一个 family 至少 5 个场景有 reactive 帧才进合并数。只有车辆 family。
- **WOD-E2E val**：照抄 [zeroshot-exam/wod-e2e](2026-09-24-zeroshot-exam/wod-e2e.md) 的「指标与统计」与「怎么判」（对 cv 配对 Δ 的 CI 整体 > 0 / 跨 0 / < 0 三档；≥ logged future 8.13 的 CI 下界），
  并按第 22 条：**主判用 s_ego 第 1–9 档的 ADE，RFS 并排，顶档单独报**。配对对象 cv、logged future、`cls ego`；另加 Alpamayo 1.5 与 openpilot Cinque 的已存逐帧预测作配对行。
- **NAVSIM navtest**：照抄 [zeroshot-exam/navsim](2026-09-24-zeroshot-exam/navsim.md) 的 devkit 版本（v1.1 出 PDMS，main @ 0a380a9 出 EPDMS）、n = 12146、`OPENBLAS_CORETYPE=Haswell`、官方打分原样。
  那份的阈值表（EPDMS ≥ 77 等）是给零样本模型的，这四个模型在 navtrain 上训，阈值自动满足，不作判据；只报与论文值的差，作为复现检查（第 6.1 节的复现差距表口径），不设门槛。
- **nuScenes**：照抄 [zeroshot-exam/nuscenes-physicalai](2026-09-24-zeroshot-exam/nuscenes-physicalai.md) 的指标（VAD / ST-P3 为主，BEV-Planner 口径并报）、main 4636、按 scene 的 bootstrap、「怎么判」表（对 CV 的配对 Δ；≤ AD-MLP；collision CI 跨 0 只报方向）。
  这四个模型都没在 nuScenes 上训，是零样本。

### 5.4 算力估计（GPU·h 按一张 RTX PRO 6000 计；CPU 打分另列）

| 考试 | DrivoR | WA-JEPA | SparseDriveV2 | ZTRS | BridgeDrive | BLUE | 共用的准备 |
|:--|--:|--:|--:|--:|--:|--:|:--|
| P5 v1 BA（离线，观测帧 + null 约 2 万帧） | 0.5 | 1.0（bf16） | 0.2 | 0.4 | 重录内 | 重录内 | 重录：P5 v1 BA 生成原本 322 run、23.6 server·h（`runs/p5v1/gen-ba/summary.json`），挂 shadow 后估 30–35 server·h，按每卡 6 server 约 5–6 卡·h，墙钟约 1.5 h（5 卡 30 server）；两个模型 shadow 前向合计约 1.5 GPU·h |
| I3（约 1 万帧） | 0.3 | 0.5 | 0.1 | 0.2 | — | — | 补渲 CAM_BACK + 10 Hz：原渲染 20 min（4 进程一卡），四路 × 2 倍帧率估 1 GPU·h |
| WOD val（479 + 958 帧） | < 0.1 | 0.1 | < 0.1 | < 0.1 | — | — | 相机映射适配器（按 Alpamayo 适配器改），主要是工程时间 |
| NAVSIM navtest + navhard | 0.5 | 2.5（原样 fp32 路径） | 0.2 | 0.4 | — | — | metric cache 已有；每次 devkit 打分 CPU 10–15 min × 2 |
| nuScenes main | 0.2 | 0.4 | 0.1 | 0.2 | — | — | 解压 CAM_BACK（CPU / IO，约 20–30 min） |
| 合计 | 约 1.6 | 约 4.5 | 约 0.7 | 约 1.3 | 约 3.5（含重录分摊） | 约 3.5 | 约 2 GPU·h + CPU |

总计约 17 GPU·h，另加重录与补渲的约 7 卡·h。除 P5 重录外每项都远低于 3 h 的 profiling 门槛；**P5 重录超过 3 h 墙钟时按 CLAUDE.md 先做 2 对的 profiling**（v0 的流程）。
建议顺序：先做契约直接对得上、最便宜的 SparseDriveV2 / ZTRS 的 P5 + I3（半天内出数），再做 DrivoR / WA-JEPA 的 I3 补渲与 WOD，最后做 BridgeDrive / BLUE 的重录；NAVSIM 复现与 nuScenes 可以插空跑。

### 5.5 这些考试能改变什么（写在数字之前）

- 若 DR / HY（scorer 族）在 P5 与 I3 上的定向翻转不高于它们自己的 null false-flip，而 AF（WA-JEPA）明显更高：多榜覆盖主要是 PDM 式评分器的配方，表征驱动的族才带 E 层能力，第 35 条的判断推广到前 10 的多榜族。
- 若 scorer 族在 I3 车辆配对上也与 openpilot `ridge_late`（70%）同量级：PDM 子分数头学到了对车辆的反应，「scorer = 纯配方」要降级。
- 若 BridgeDrive 的 target speed 通道翻转仍像 TFv6 一样接近 0、waypoint 通道 ≥ 30%：B2D 榜首的增量属于接口与规则，不属于 E 层。
- 若 BLUE 在 P5 v1 突发 family 上显著高于 SimLingo：第 38 条「BLUE 在突发 hazard 上真的更好」从公开逐路线数据升级为我们的配对证据。

## 执行分派（中央执行员）

- 2026-09-26 10:00 CST [main] 按 5.4 的建议顺序拆成三路，插进空出来的卡（用户：谁先完成、卡空了就插谁）：
  T1 = SparseDriveV2 + ZTRS：P5 v1 BA + I3 先出数，再 WOD / NAVSIM 复现 / nuScenes；契约直接对得上，最便宜，现在卡空着就先开，放 GPU 4（与 night-queue-2 B 的零散 GPU 共用）。
  T2 = DrivoR + WA-JEPA：I3 补渲 CAM_BACK + 10 Hz，再 P5 / WOD / NAVSIM / nuScenes；等下一个执行员收工空出卡再开。
  T3 = BridgeDrive + BLUE 的 P5 重录（约 30 个 CARLA server）：等 night-queue-2 A 的 N1 批量结束、CARLA 容量空出来再开。
  各路在本节下写 [T1] / [T2] / [T3] 条目，结果写「结果」。B2D 闭环那一项仍按 5.1 另立 todo，不在这里跑。

## 结果

跑完再填。smoke 的 run dir：`~/data/runs/top10_smoke/{drivor,wajepa,sparsedrivev2,gtrs,bridgedrive,blue}/`。
