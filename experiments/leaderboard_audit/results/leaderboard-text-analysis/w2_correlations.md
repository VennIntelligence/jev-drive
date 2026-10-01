# W2 跨榜/跨协议机械相关

数据：[`w2_cross_board.csv`](w2_cross_board.csv)。`papers/` 相对于 `/data/hack_audit/`；`PDF p.` 指 PDF 文件页序，不是印刷页码。长表记录已发表分数，没有运行模型。

计算：对两个协议与指标列按 `method` 精确匹配；每列按原始数值升序赋平均秩（并列取平均），再计算两列秩的 Pearson 相关，即 Spearman ρ。L2 数值越低越好，其余所列主指标越高越好；这里没有倒转 L2 的符号。`n < 5` 只列配对数值。

## n ≥ 5 的配对

| 列 A | 列 B | n | Spearman ρ | 方法与数值（A → B） |
|---|---|---:|---:|---|
| Bench2Drive base-set 开环 Avg L2 | Bench2Drive base-set 闭环 Driving Score | 9 | -0.1590 | AD-MLP 3.64→18.05；DriveAdapter* 1.01→64.22；TCP* 1.7→40.7；TCP-traj w/o distillation 1.96→49.3；TCP-traj* 1.7→59.9；ThinkTwice* 0.95→62.44；UniAD-Base 0.73→45.81；UniAD-Tiny 0.8→40.73；VAD 0.91→42.35 |
| Bench2Drive base-set 开环 Avg L2 | Bench2Drive base-set 闭环 Success Rate (%) | 9 | -0.0546 | AD-MLP 3.64→0；DriveAdapter* 1.01→33.08；TCP* 1.7→15；TCP-traj w/o distillation 1.96→20.45；TCP-traj* 1.7→30；ThinkTwice* 0.95→31.23；UniAD-Base 0.73→16.36；UniAD-Tiny 0.8→13.18；VAD 0.91→15 |
| Bench2Drive base-set 开环 Avg L2 | Bench2Drive base-set 闭环 Efficiency | 9 | -0.6946 | AD-MLP 3.64→48.45；DriveAdapter* 1.01→70.22；TCP* 1.7→54.26；TCP-traj w/o distillation 1.96→78.78；TCP-traj* 1.7→76.54；ThinkTwice* 0.95→69.33；UniAD-Base 0.73→129.21；UniAD-Tiny 0.8→123.92；VAD 0.91→157.94 |
| Bench2Drive base-set 开环 Avg L2 | Bench2Drive base-set 闭环 Comfortness | 9 | -0.2510 | AD-MLP 3.64→22.63；DriveAdapter* 1.01→16.01；TCP* 1.7→47.8；TCP-traj w/o distillation 1.96→22.96；TCP-traj* 1.7→18.08；ThinkTwice* 0.95→16.22；UniAD-Base 0.73→43.58；UniAD-Tiny 0.8→47.04；VAD 0.91→46.01 |
| NAVSIM v1 navtest PDMS | NAVSIM v2 navhard two-stage 修复后 EPDMS | 20 | +0.6281 | DiffusionDrive 88.1→24.2；DriveFuture 90.7→55.5；DrivoR 94.6→54.6；DrivoR + TOAD 94.7→56.3；GTRS (V2-99) 90.4→45.4；GTRS (V2-99) + TOAD 90.9→51.7；Hydra-MDP 90.9→40.9；Hydra-MDP + TOAD 91.4→49.7；LTF 83.8→25.1；LTFv6 85.4→28.3；LTFv6 + LEAD 86.4→31.4；PDM-Closed 89.1→56.6；RAP-DINO 93.8→39.6；RAP-DINO + TOAD 93.9→49；TransFuser 84→23.1；World4Drive 85.1→34.9；ZTRS (V2-99) 86.9→48.1；ZTRS (V2-99) + TOAD 89→49.2；iPad 91.7→34.7；iPad + TOAD 93.4→49.8 |
| Bench2Drive DS | CARLA Longest6 v2 DS | 13 | +0.9296 | HiP-AD 86.8→7；LEAD 96.8→73；PDM-Lite 97→73；RoG-DAgger 90.34→44；SimLingo 85.1→22；TF++ 84.21→23；TFv5 (RegNetY-032, 110C+L) 83.5→23；TFv6 (RegNetY-032, 140C+L+R) 95.2→62；TFv6 (ResNet34, 140C+L+R) 94.7→57；TFv6 (ResNet34, 360C) 91.6→43；TFv6 (ResNet34, 360C+L) 94.7→52；TFv6 (ResNet34, 360C+L+R) 95→54；TFv6 (ResNet34, 360C+R) 94.2→52 |
| Bench2Drive DS | CARLA Longest6 v2 RC | 13 | +0.8985 | HiP-AD 86.8→56；LEAD 96.8→93；PDM-Lite 97→100；RoG-DAgger 90.34→88；SimLingo 85.1→70；TF++ 84.21→70；TFv5 (RegNetY-032, 110C+L) 83.5→70；TFv6 (RegNetY-032, 140C+L+R) 95.2→91；TFv6 (ResNet34, 140C+L+R) 94.7→99；TFv6 (ResNet34, 360C) 91.6→85；TFv6 (ResNet34, 360C+L) 94.7→88；TFv6 (ResNet34, 360C+L+R) 95→89；TFv6 (ResNet34, 360C+R) 94.2→88 |
| Bench2Drive SR | CARLA Longest6 v2 DS | 13 | +0.9076 | HiP-AD 69.1→7；LEAD 96.6→73；PDM-Lite 92.3→73；RoG-DAgger 73.51→44；SimLingo 67.2→22；TF++ 67.27→23；TFv5 (RegNetY-032, 110C+L) 67.3→23；TFv6 (RegNetY-032, 140C+L+R) 86.8→62；TFv6 (ResNet34, 140C+L+R) 82.1→57；TFv6 (ResNet34, 360C) 79.5→43；TFv6 (ResNet34, 360C+L) 85.6→52；TFv6 (ResNet34, 360C+L+R) 84.3→54；TFv6 (ResNet34, 360C+R) 85.3→52 |
| Bench2Drive SR | CARLA Longest6 v2 RC | 13 | +0.8223 | HiP-AD 69.1→56；LEAD 96.6→93；PDM-Lite 92.3→100；RoG-DAgger 73.51→88；SimLingo 67.2→70；TF++ 67.27→70；TFv5 (RegNetY-032, 110C+L) 67.3→70；TFv6 (RegNetY-032, 140C+L+R) 86.8→91；TFv6 (ResNet34, 140C+L+R) 82.1→99；TFv6 (ResNet34, 360C) 79.5→85；TFv6 (ResNet34, 360C+L) 85.6→88；TFv6 (ResNet34, 360C+L+R) 84.3→89；TFv6 (ResNet34, 360C+R) 85.3→88 |
| NAVSIM v1 navtest PDMS | WOD-E2E test RFS Overall | 5 | +0.1000 | AutoVLA 89.11→7.5566；DriveMA-2B 90.5→8.06；DriveMA-4B 91.2→8.079；NTR 94.1→7.9982；RAP-DINO 93.8→8.04 |

`DrivoR (+134k SimScale)` 与 TOAD 表中的 `DrivoR` 在两榜上均为 94.6/54.6；相关计算只计一次，原始两种出处仍保存在长表。Bench2Drive ↔ Longest6 的 DS 对 DS 相关含 TFv6 Table 5 的多个同论文传感器/骨干变体（ρ 约 +0.9296），并非独立方法观测。该表中的 LEAD 为论文估计的特权 expert：Bench2Drive DS 96.8 是估计值，不是统一公开提交。TOAD 的基座与搜索变体、DriveMA 的两个模型规模也分别计入，观察值并非统计独立。配对允许不同训练数据或 checkpoint；`same_checkpoint` 列记录证据，ρ 本身只描述表内排序。

## n < 5 的配对数据

以下协议各自单列，没有把 NAVSIM v2 navtest 与 navhard、修复前与修复后、CARLA 官方 MAP 与 SENSORS、或不同 nuScenes 指标实现合并。

| 列 A | 列 B | n | 方法与数值（A → B） |
|---|---|---:|---|
| NAVSIM v1 navtest PDMS | NAVSIM v2 navtest 修复后 EPDMS | 3 | DriveFuture 90.7→89.9；SparseDriveV2 92→90.1；WA-JEPA 91.8→91.7 |
| NAVSIM v1 navtest PDMS | HUGSIM 436 场景 HD-Score | 4 | DrivoR 94.6→0.3252；LTF 83.8→0.231；UniAD 83.4→0.3124；WA-JEPA 91.8→0.4462 |
| NAVSIM v2 navhard 修复后 EPDMS | HUGSIM 436 场景 HD-Score | 2 | DrivoR 54.6→0.3252；LTF 25.1→0.231 |
| Bench2Drive DS | CARLA LB2 official MAP DS | 2 | SimLingo-BASE 85.94→6.25；TF++ 84.21→5.56 |
| Bench2Drive DS | CARLA LB2 official SENSORS DS | 2 | SimLingo-BASE 85.94→6.87；TF++ 84.21→5.18 |
| NAVSIM v1 navtest PDMS | WOD-E2E validation RFS | 2 | LTFv6 85.4→7.51；LTFv6 + LEAD 86.4→7.76 |
| Bench2Drive DS | NAVSIM v1 navtest PDMS | 2 | AutoVLA 78.84→89.11；SparseDriveV2 89.15→92 |
| Bench2Drive DS | NAVSIM v2 navtest 修复后 EPDMS | 1 | SparseDriveV2 89.15→90.1 |
| Bench2Drive DS | HUGSIM 436 场景 HD-Score | 1 | VAD 42.35→0.1393 |
| Bench2Drive DS | WOD-E2E test RFS | 1 | AutoVLA 78.84→7.5566 |
| NAVSIM v2 navhard 修复后 EPDMS | WOD-E2E test RFS | 1 | RAP-DINO 39.6→8.04 |
| NAVSIM v2 navhard 修复后 EPDMS | WOD-E2E validation RFS | 2 | LTFv6 28.3→7.51；LTFv6 + LEAD 31.4→7.76 |
| Bench2Drive DS | nuScenes val, ST-P3 metric, 1/2/3s mean Avg L2 | 2 | AD-MLP 18.05→0.29；AutoVLA 78.84→0.4 |
| Bench2Drive DS | nuScenes val, OmniSpace Table 1 metric, 1/2/3s mean Avg L2 | 3 | OmniSpace (Qwen2.5-VL 7B) 79.65→0.28；OmniSpace (Qwen3-VL 4B) 81.4→0.28；VAD-Base 42.35→0.37 |
| Bench2Drive DS | nuScenes val, 1/2/3s mean Avg L2 | 1 | SteerVLA 90.71→0.4 |
| nuScenes ST-P3 Avg L2 | NAVSIM v1 navtest PDMS | 1 | AutoVLA 0.4→89.11 |
| NAVSIM v1 navtest PDMS | HUGSIM zero-shot KITTI360 HDS (%) | 2 | DrivoR 94.6→17.7；DrivoR + TOAD 94.7→18.4 |
| NAVSIM v1 navtest PDMS | HUGSIM zero-shot nuScenes HDS (%) | 2 | DrivoR 94.6→39.7；DrivoR + TOAD 94.7→49.9 |
| NAVSIM v1 navtest PDMS | HUGSIM zero-shot PandaSet HDS (%) | 2 | DrivoR 94.6→35.1；DrivoR + TOAD 94.7→38 |
| NAVSIM v1 navtest PDMS | HUGSIM zero-shot Waymo HDS (%) | 2 | DrivoR 94.6→44.2；DrivoR + TOAD 94.7→42.8 |

## 榜单官方论文自身的开环/闭环分析

- **Bench2Drive 官方论文**：Table 3 同时列出 2 秒、2 Hz 的开环 Avg. L2 与 220 条路线的闭环 DS/SR 等。正文称 UniAD-Base 的 L2 低于 VAD、闭环表现较差；同表中 UniAD-Base 的 DS/SR 高于 VAD，而 Efficiency/Comfortness 低于 VAD，原句没有指明所说的闭环指标。原文没有报告总体相关系数。出处：`papers/bench2drive_benchmark.pdf`，Table 3 与其后讨论，PDF p.9。
- **NAVSIM v1 官方论文**：Section 4.1 在 396 个 navmini 场景上比较 37 个规则式与 114 个学习式 planner 的 nuPlan closed-loop score（CLS）对 PDMS 和 nuPlan open-loop score（OLS）的关系。Fig. 3/4 同时给 Spearman 与 Pearson 图示；正文称 PDMS 对 CLS 的相关性在五类 planner 中均高于 OLS。图中没有印出每个系数的精确数值，本表不从图像估读。出处：`papers/ltf.pdf`，Fig. 3，PDF p.6；Fig. 4，PDF p.7。
- **NAVSIM v2 官方论文**：Section 4.1 称在 83 个 planner、244 个 Stage 1 与 4164 个 Stage 2 观察上比较（原文五类 planner 数 10+15+15+22+24 合计 86，与 83 不一致） EPDMS、ADE、OLS 与 8 秒 nuPlan CLS；这里用于相关性实验的是删去 TLC/LK/EC 的简化 EPDMS。文中给两阶段 2×4 秒与单阶段基线的 Pearson `r=0.89`（`R²=0.8`）和 `r=0.83`（`R²=0.7`）；Fig. 4 还展示 Spearman 排序相关，但没有印出精确系数。出处：`papers/navsim_v2_benchmark.pdf`，Fig. 3 与 Section 4.1，PDF p.6；Fig. 4 与续文，PDF p.7。

## 已知边界

1. `NAVSIM v2 navtest` 是单阶段旧测试集上的扩展指标；`navhard-two-stage` 是官方两阶段榜单。SparseDriveV2、DriveFuture、WA-JEPA 在 navtest 表中明确区分修复前的 `EPDMS*` 与修复后的 `EPDMS`。NTR 的 navtest EPDMS 90.9 未注明修复状态，故独列。DrivoR Table 14 也分别列修复前后数值；这里用后者。
2. 不同论文重报的同名方法可有版本或四舍五入差异。长表只选一个可定位来源作为每个键的主记录；例如 LTF v2 navhard 用官方修复后快照 25.1，RAP 早期表的修复前 23.12 不混入。
3. Bench2Drive 官方开环/闭环是同一训练集下基准作者重评；跨榜值多为不同模型训练、独立提交或未知 checkpoint。尤其 `RAP-DINO` 的 Bench2Drive 结果实际为 `RAP-ResNet`，保留分名而不配对。
4. HUGSIM 的 WA-JEPA Table 2 是统一 436 场景，TOAD Table 4 是分 KITTI360/nuScenes/PandaSet/Waymo 的另一协议；nuScenes L2 也有不同实现和输入协议。没有跨这些不同列凑足样本数。
5. OmniSpace 的 nuScenes/Bench2Drive 表、DriveMA 的 NAVSIM 表未在各表标题逐字写明所有 split；长表按论文评测上下文及官方协议标注，严格跨协议对照时应回查原文。
6. 这些相关性是方法级已发表数字的描述统计，含多种模型变体和来源，不附显著性检验，也不作因果或能力解读。
