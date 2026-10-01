# W3 `paper_locator` 与论文数字独立复核

复核日期：2026-09-24。逐条核对 `w3_issues.csv` 中 `paper_locator` 非 `N/A` 的 **35 行**：其中 34 行指向论文，carla_garage #73 仅指向 issue。页码均按 PDF 阅读器从 1 开始计数。用本地 PDF 的指定页核对表格、章节和数字；另打开 arXiv 原链接与相关 GitHub issue，特别核对论文版本及数字是论文、用户还是作者所述。未运行模型、仿真或评测。

**当前结果：35/35 的定位位置可核；没有剩余的论文数字或 URL 错配。** 以下“通过”只代表引文核对，不代表用户复现实验或作者解释已被独立证实。末尾列出初稿中发现、现已修正的问题及仍需谨慎表述之处。CSV 数据行号不含表头。

| CSV 行 / issue | 结论 | 核对证据 |
|---|---|---|
| 1 [WA-JEPA #1](https://github.com/AFARI-Research/WA-JEPA/issues/1) | 通过 | [WA-JEPA Table 2，PDF p.7](https://arxiv.org/pdf/2608.20974#page=7) 的 HUGSIM HD-Score 为 **0.4462**；与帖内 NAVSIM EPDMS 是不同榜单。 |
| 4 [carla_garage #11](https://github.com/autonomousvision/carla_garage/issues/11) | 通过，须固定 v1 | [TF++ arXiv v1 Table 6，p.8](https://arxiv.org/pdf/2306.07957v1#page=8) 为 Longest6 **DS 72±3、RC 95±2、IS 0.74±0.04**，帖主引其中心值；CSV 已指向 v1。[v2 同表](https://arxiv.org/pdf/2306.07957v2#page=8) 则为 69±0、94±2、0.72±0.01。 |
| 7 [carla_garage #46](https://github.com/autonomousvision/carla_garage/issues/46) | 通过 | [TF++ v2 Table 6，p.8](https://arxiv.org/pdf/2306.07957v2#page=8) 有 69±0、94±2、0.72±0.01；[Table 13，p.16](https://arxiv.org/pdf/2306.07957v2#page=16) 有 `+3x data` 69±1/97±1 及 `TF++ WP` 70±4/94±2，对应帖主列的行。 |
| 8 [carla_garage #47](https://github.com/autonomousvision/carla_garage/issues/47) | 通过 | [TF++ v2 Table 6，p.8](https://arxiv.org/pdf/2306.07957v2#page=8) 的 Longest6 为 **DS 69±0、RC 94±2**。CSV 对用户单次运行的 checkpoint 对应关系保留了不确定性。 |
| 11 [carla_garage #73](https://github.com/autonomousvision/carla_garage/issues/73) | 通过，issue 来源 | 无可比论文数字；作者在[帖内回复](https://github.com/autonomousvision/carla_garage/issues/73)称 **25 DS** 看起来在波动范围内。CSV 未把此数写成论文值。 |
| 13 [carla_garage #103](https://github.com/autonomousvision/carla_garage/issues/103) | 通过 | [Hidden Biases of End-to-End Driving Datasets Table 4，p.6](https://arxiv.org/pdf/2412.09602#page=6) 的 TF++ Bench2Drive 行是 **DS 84.21、SR 67.27**。 |
| 15 [carla_garage #120](https://github.com/autonomousvision/carla_garage/issues/120) | 通过 | [同论文 Table 5，p.8](https://arxiv.org/pdf/2412.09602#page=8) 的“Town13 withheld / TF++ / ET 否”行是 **RC 50.20、DS 1.08**，支持“约 50/约 1”；限定到这一行才准确。 |
| 16 [navsim #35](https://github.com/autonomousvision/navsim/issues/35) | 通过 | [NAVSIM Table 1，p.7](https://arxiv.org/pdf/2406.15349#page=7) TransFuser **84.0**；[Hydra-MDP Table 1，p.4](https://arxiv.org/pdf/2406.06978#page=4) Transfuser **78.0**。帖主自己的约 80.0 属用户实测。 |
| 19 [navsim #62](https://github.com/autonomousvision/navsim/issues/62) | 通过，指标口径见下 | 两篇论文同上，**78.0/84.0 都是 TransFuser 行**；[帖文](https://github.com/autonomousvision/navsim/issues/62)没有本人复现实测。当前 `related_methods` 已把 TransFuser 标为帖文主体，PDM-Closed 仅标仓库关联。 |
| 30 [AD-MLP #4](https://github.com/E2E-AD/AD-MLP/issues/4) | 通过 | [AD-MLP Table 1，p.3](https://arxiv.org/pdf/2305.10430#page=3) 的 `Ours` 全输入行平均 L2 是 **0.29**；issue 未给精确复现 L2，CSV 已说明。 |
| 31 [AD-MLP #5](https://github.com/E2E-AD/AD-MLP/issues/5) | 通过 | 同一[Table 1，p.3](https://arxiv.org/pdf/2305.10430#page=3) 有平均 L2 **0.29**；帖主并未报告可直接相减的同配置分数。 |
| 35 [BLUE #4](https://github.com/George-Ling3/BLUE/issues/4) | 通过，须区分版本 | [BLUE arXiv v1 Table 1，p.4](https://arxiv.org/pdf/2606.08684v1#page=4) 的 DeLL expert data 是 **Think2Drive**；[现行 Table 1，p.4](https://arxiv.org/pdf/2606.08684#page=4) 改为 **PDM-Lite**，与[作者勘误回复](https://github.com/George-Ling3/BLUE/issues/4)一致。CSV 现列两个版本。 |
| 36 [BLUE #5](https://github.com/George-Ling3/BLUE/issues/5) | 通过 | [BLUE Table 6，p.6](https://arxiv.org/pdf/2606.08684#page=6) 的 SimLingo + SimLingo gate 行为 **DS 90.58±0.12**；中心值 90.58 正确。 |
| 39 [Senna #11](https://github.com/hustvl/Senna/issues/11) | 通过，变体需说明 | [Senna Table II，p.6](https://arxiv.org/pdf/2410.22313#page=6) 的 **Senna\*** 平均 L2 为 **0.22**；星号表示使用 ego status。帖内用户报的是动作准确率，不是 L2。 |
| 40 [Senna #26](https://github.com/hustvl/Senna/issues/26) | 通过，变体需说明 | 同一[Table II，p.6](https://arxiv.org/pdf/2410.22313#page=6) 的 **Senna\*** 平均 L2 为 **0.22**；用户的 40%、39.13%、68.41% 为动作准确率，不应写作论文 L2。 |
| 44 [LEAD/TFv6 #89](https://github.com/kesai-labs/lead/issues/89) | 通过，归属已修 | [TFv6 Table 5，p.7](https://arxiv.org/pdf/2512.20563#page=7) 最佳 140°/LiDAR/Radar 配置 Bench2Drive **DS 95.2±0.3**。**95.28 仅是帖主引用值**；论文未报 95.28，CSV 现不把它说成 95.2 的舍入值。 |
| 45 [LEAD/TFv6 #90](https://github.com/kesai-labs/lead/issues/90) | 通过，估计值 | 同一[Table 5，p.7](https://arxiv.org/pdf/2512.20563#page=7) 确有 LEAD expert 行（Bench2Drive **DS 96.8、SR 96.6**）；[作者回复](https://github.com/kesai-labs/lead/issues/90)说明它由标准 leaderboard 结果估计，未直接在 Bench2Drive 上评测。 |
| 48 [LEAD/TFv6 #99](https://github.com/kesai-labs/lead/issues/99) | 通过，归属已修 | 论文最佳配置 **95.2±0.3** 见[Table 5，p.7](https://arxiv.org/pdf/2512.20563#page=7)；[该帖正文和评论](https://github.com/kesai-labs/lead/issues/99)只称低于发布分数，**没有 95.28 或明确 DS**。CSV 现已取消错误的 95.28 帖内归属。 |
| 61 [SimLingo #25](https://github.com/RenzKa/simlingo/issues/25) | 通过 | [SimLingo §3.4，p.5](https://arxiv.org/pdf/2503.09594#page=5) 明写桶采样使每 epoch 为 **650,000 samples**；用户的加载数属于用户报告。 |
| 63 [SimLingo #43](https://github.com/RenzKa/simlingo/issues/43) | 通过 | [SimLingo Table 2，p.7](https://arxiv.org/pdf/2503.09594#page=7) 中 **SimLingo-BASE (LB2.0 model) DS 85.94**。 |
| 65 [SimLingo #49](https://github.com/RenzKa/simlingo/issues/49) | 通过 | [§3.4，p.5](https://arxiv.org/pdf/2503.09594#page=5) 为 **650,000/epoch**；[附录 Table 7，p.14](https://arxiv.org/pdf/2503.09594#page=14) 的 Epochs 为 **14**。用户配置 `max_epochs=15` 属用户报告。 |
| 66 [SimLingo #66](https://github.com/RenzKa/simlingo/issues/66) | 通过 | [§3.4，p.5](https://arxiv.org/pdf/2503.09594#page=5) 写 VQA **50%**、commentary prediction **35%**、commentary in prompt **7.5%**、no-language **7.5%**；帖中代码概率是用户所引代码。 |
| 67 [SimLingo #72](https://github.com/RenzKa/simlingo/issues/72) | 通过 | [Table 2，p.7](https://arxiv.org/pdf/2503.09594#page=7) 为 SimLingo-BASE **85.94** 和 full SimLingo **85.07±0.95**，支持 CSV 的概略范围；若需计算差距，应指定变体。 |
| 74 [SparseDriveV2 #16](https://github.com/swc-17/SparseDriveV2/issues/16) | 通过 | [SparseDriveV2 Table 4，p.13](https://arxiv.org/pdf/2603.29163#page=13) Bench2Drive 的 SparseDriveV2 行 **DS 89.15**。 |
| 76 [DriveMA #2](https://github.com/Tsinghua-MARS-Lab/DriveMA/issues/2) | 通过 | [DriveMA Table 3，p.7](https://arxiv.org/pdf/2605.31271#page=7) 的 Meta-Action SFT w/ ACP 行 RFS Overall **7.893**，以两位小数报 **7.89**；用户复现为 7.28。 |
| 77 [DriveMA #3](https://github.com/Tsinghua-MARS-Lab/DriveMA/issues/3) | 通过 | [DriveMA 附录 RL training details，p.14](https://arxiv.org/pdf/2605.31271#page=14) 明写 KL 系数 **β=0.4**；**0.04** 是[帖主报告的代码参数](https://github.com/Tsinghua-MARS-Lab/DriveMA/issues/3)，论文没有此值。 |
| 89 [AutoVLA #47](https://github.com/ucla-mobility/AutoVLA/issues/47) | 通过 | [AutoVLA Table S4，p.30](https://arxiv.org/pdf/2506.13757#page=30) 第 3/4 数据行为 Multi camera、Action-only / CoT-enhanced；[帖主](https://github.com/ucla-mobility/AutoVLA/issues/47)仅点名行号，未转录分数。 |
| 90 [AutoVLA #48](https://github.com/ucla-mobility/AutoVLA/issues/48) | 通过 | [AutoVLA Table 1，p.7](https://arxiv.org/pdf/2506.13757#page=7) 的 Post-RFT **PDMS 89.11**；用户的 83.69 属发布权重复现。 |
| 92 [DrivoR #4](https://github.com/valeoai/DrivoR/issues/4) | 通过 | [DrivoR Table 1，p.5](https://arxiv.org/pdf/2601.05083#page=5) `DrivoR (trainval)` 的 navtest **PDMS 93.7**。作者在帖内给的 0.931288 是另一个训练/评测陈述，CSV 已说明不可直接相比。 |
| 96 [DrivoR #31](https://github.com/valeoai/DrivoR/issues/31) | 通过 | [DrivoR Table 3，p.6](https://arxiv.org/pdf/2601.05083#page=6) 的无 SimScale `DrivoR (ViT-S)` navhard EPDMS **48.3**；“修 bug 后”由[作者回复](https://github.com/valeoai/DrivoR/issues/31)支持。 |
| 97 [DrivoR #34](https://github.com/valeoai/DrivoR/issues/34) | 通过 | [DrivoR Table 2，p.5](https://arxiv.org/pdf/2601.05083#page=5) 为 HUGSIM 结果；[p.6 正文](https://arxiv.org/pdf/2601.05083#page=6) 明写 pre-challenge test set **345 scenarios**。 |
| 100 [DrivoR #40](https://github.com/valeoai/DrivoR/issues/40) | 通过 | [DrivoR Table 1，p.5](https://arxiv.org/pdf/2601.05083#page=5) 有 `+134k SimScale data` 行；[p.6](https://arxiv.org/pdf/2601.05083#page=6) 写提取 65k/134k 注释。用户的 236k/约 200k 是用户统计。 |
| 101 [DrivoR #41](https://github.com/valeoai/DrivoR/issues/41) | 通过 | [DrivoR Table 1，p.5](https://arxiv.org/pdf/2601.05083#page=5) `DrivoR (train)` 的 navtest 为 **93.1**；[Table 4a，p.7](https://arxiv.org/pdf/2601.05083#page=7) 的 DINOv2 消融在 navval 为 **90.0**。帖主仅引用两项论文值，未报告本人实测。 |
| 105 [DrivoR #47](https://github.com/valeoai/DrivoR/issues/47) | 通过 | [DrivoR Table 3，p.6](https://arxiv.org/pdf/2601.05083#page=6) `+134k SimScale` navhard EPDMS **54.6**；“指定发布权重”对应关系来自[作者在 issue 的回复](https://github.com/valeoai/DrivoR/issues/47)，不是 Table 3 单独证明。 |
| 106 [DrivoR #54](https://github.com/valeoai/DrivoR/issues/54) | 通过 | [DrivoR Table 2，p.5](https://arxiv.org/pdf/2601.05083#page=5) HUGSIM `DrivoR` Avg. 为 **RC 49.8、HD-Score 35.7**；[p.6](https://arxiv.org/pdf/2601.05083#page=6) 明写 **345** 场景。 |

## 版本与归属问题的修订记录

初次读取 CSV 时，下列 4 个 issue 的定位或数字归属不正确；主表为修订后的复核结果：

1. **carla_garage #11**：72/95/0.74 只在 TF++ arXiv **v1** Table 6 p.8；未定版 URL 的现行 v2 为 69/94/0.72。现已固定 v1。
2. **carla_garage #46**：v2 Table 13 实际在 PDF **p.16**，初稿写 p.15。现已更正；#47 对应 v2 Table 6 p.8。
3. **BLUE #4**：现行 PDF p.4 已改成 PDM-Lite，不能证明“原写 Think2Drive”。现已加入 v1 Table 1 p.4 证明旧写法，同时保留现行版说明修正。
4. **LEAD/TFv6 #89/#99**：论文只报 95.2±0.3；#89 帖主另引 95.28，不能称为论文值或对 95.2 的舍入；#99 帖文根本未给 95.28。现已分别限定。

## 仍需保留的口径限定

- **Senna #11/#26**：Table II 的 **0.22** 属 **Senna\***（ego status 输入）平均 L2；不带星的 Senna 为 **0.59**。CSV 现已区分两行和用户所报的另一种指标。
- **navsim #35/#62**：[Hydra-MDP Table 1 与 §3.1，p.4](https://arxiv.org/pdf/2406.06978#page=4) 的 **78.0** 列题为 `Score`，是忽略 DDC 的 PDM score；[NAVSIM Table 1，p.7](https://arxiv.org/pdf/2406.15349#page=7) 的 **84.0** 列题为 `PDMS`。CSV #62 现已注明实现口径未核同；论文和 issue 均未证明 78→84 的原因。
- **LEAD/TFv6 #90**：表内 LEAD expert 数字是论文所列值，但作者明确说明是估计，并非 Bench2Drive 直接评测。`paper_numbers` 可补 DS 96.8/SR 96.6 及“估计”以便脱离 issue 阅读。
- **DrivoR #41/#47**：93.1 主结果与 90.0 消融分别在 navtest/navval；#47 指定 checkpoint 的关系由 issue 作者回复支持，论文 Table 3 本身只支持 54.6。
