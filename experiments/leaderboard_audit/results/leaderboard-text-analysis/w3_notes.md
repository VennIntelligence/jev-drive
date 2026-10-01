# W3 GitHub issues 与 discussions：抓取及筛选记录

抓取时间：2026-09-24T12:50:39.081197+00:00。范围是 `out/sampling.csv` 中 26 个带 `repo_url` 的条目，去重后 **20 个仓库**，含 C 档的 `RenzKa/simlingo`、`hustvl/Senna`、`valeoai/DrivoR`。对每仓用已登录 `gh api --paginate --slurp` 拉取 `state=all` 的 issue 列表与全部 issue 评论，剔除 REST issues 接口混入的 pull request。另用 GraphQL `discussions.totalCount` 对 20 仓逐一核验，均为 **0**。本次共 **658 个 issue、1589 条 issue 评论、0 个 discussion**；四类口径筛出 **113 个 issue**。批量抓取 API 全部成功，无失败仓库。首次逐条 API HEAD 校验 113/113 返回 200；二次并发 HEAD 检查中 `swc-17/SparseDriveV2#18` 有一次 20 秒超时，单条重试返回 200。

CSV 一行对应一个 issue；`category` 可复合，值为 `score_gap`（用户复现实测与论文的分数差）、`evaluation_config`（复现/评测配置）、`author_disclosure`（作者补充）、`protocol_split`（协议/划分疑点）。`user_reported_numbers` 与 `paper_numbers` 分列；若用户只引用两篇已发表论文而无自身复现实测，明确标为“非用户复现差”。数字无可比关系时显式说明；`author_quote` 是与 `author_reply_url` 对应的原文短摘，`case_summary` 才是中文转述。`resolved` 的 `yes` 表示原问题有可执行的明确答复或提问者确认复现，`partial` 表示仅部分解释或仍有差距，`no` 表示问题或取得代码/权重等复现障碍仍未解决（即使作者作过回复）；它与 GitHub 的 `issue_state` 不是同一字段。

`paper_locator` 对真正引用的论文数字补上 Table/Section 与 PDF 页序；若数字只来自用户或作者在 issue 中的陈述，则 `issue_url`/`author_reply_url` 才是出处，`paper_locator` 标 `N/A`。字段中的「未报告数字」「该 issue 无可比论文数字」和「N/A（无作者回复）」是明确的缺失值标记，不是分数或回复原文。

以下统计均按唯一仓库计算；C 档与跨榜同仓不会重复计数。筛选旨在记录与四类问题明确相关的原文，不把所有安装、下载或运行报错自动算作复现分数差。

### [kesai-labs/lead](https://github.com/kesai-labs/lead)

全部 issue **26**（open 2、closed 24），discussions **0**，拉取评论 **69** 条；筛出 **7** 条。样本中对应：TFv6[A]。
较值得关注：[# 90](https://github.com/kesai-labs/lead/issues/90) 论文表中 LEAD expert 的 Bench2Drive 行由标准 leaderboard 结果估计，作者明确说未直接跑 B2D；[# 72](https://github.com/kesai-labs/lead/issues/72) Longest6 复现差距及每个训练 seed 只评一次的原文说明；[# 89](https://github.com/kesai-labs/lead/issues/89) 202 条完成路线的单权重复现及未训练的 YieldToEmergencyVehicle 场景。

### [George-Ling3/BLUE](https://github.com/George-Ling3/BLUE)

全部 issue **6**（open 3、closed 3），discussions **0**，拉取评论 **10** 条；筛出 **4** 条。样本中对应：BLUE[B]。
较值得关注：[# 2](https://github.com/George-Ling3/BLUE/issues/2) 发布包漏文件后补；作者给正式评测单 A100 的资源设置；[# 5](https://github.com/George-Ling3/BLUE/issues/5) 作者确认 gate 的 direct 模式沿用 CoT prompt，未单独消融其影响；[# 3](https://github.com/George-Ling3/BLUE/issues/3) MinSpeedTest 大值的指标解释。

### [swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)

全部 issue **18**（open 7、closed 11），discussions **0**，拉取评论 **42** 条；筛出 **7** 条。样本中对应：SparseDriveV2[A]。
较值得关注：[# 16](https://github.com/swc-17/SparseDriveV2/issues/16) 发布权重在 B2D 复现 DS 86.03、SR 65.45，论文 DS 89.15，仓库未回复；[# 7](https://github.com/swc-17/SparseDriveV2/issues/7) 作者明确说 NAVSIM 推理评分设计为最大化 PDMS；[# 9](https://github.com/swc-17/SparseDriveV2/issues/9) 评测入口缺 _global_plan_far，作者给配置修法。

### [RenzKa/simlingo](https://github.com/RenzKa/simlingo)

全部 issue **99**（open 25、closed 74），discussions **0**，拉取评论 **198** 条；筛出 **8** 条。样本中对应：CarLLaVA[C,excluded], SimLingo-BASE[C,excluded]。
较值得关注：[# 43](https://github.com/RenzKa/simlingo/issues/43) B2D 定制目录使用户 DS 由 75.50 到 86.53；作者两次评测为 87.40/87.67；[# 44](https://github.com/RenzKa/simlingo/issues/44) 作者承认关闭 4000 tick 截断；[# 66](https://github.com/RenzKa/simlingo/issues/66) 公开训练混合比例与论文叙述差异仍无回复。

### [autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage)

全部 issue **117**（open 0、closed 117），discussions **0**，拉取评论 **337** 条；筛出 **13** 条。样本中对应：TF++[A]。
较值得关注：[# 11](https://github.com/autonomousvision/carla_garage/issues/11) Longest6 单次 DS 57.58/62.68 与九次均值 72 的比较及漏配 BENCHMARK；[# 19](https://github.com/autonomousvision/carla_garage/issues/19) 作者逐表说明同一权重与 seed/ensemble 对应；[# 103](https://github.com/autonomousvision/carla_garage/issues/103) 作者澄清 TF++ B2D 训练路线不是 B2D 或 Longest6-v2 评测路线。

### [valeoai/TOAD](https://github.com/valeoai/TOAD)

全部 issue **0**（open 0、closed 0），discussions **0**，拉取评论 **0** 条；筛出 **0** 条。样本中对应：TOAD+DrivoR[A], DrivoR+TOAD[A]。
仓库在抓取时没有 issue 或 discussion，因此无匹配记录。

### [ZebinX/DriveVLA-M0](https://github.com/ZebinX/DriveVLA-M0)

全部 issue **0**（open 0、closed 0），discussions **0**，拉取评论 **0** 条；筛出 **0** 条。样本中对应：DriveVLA-M0[A]。
仓库在抓取时没有 issue 或 discussion，因此无匹配记录。

### [vita-epfl/RAP](https://github.com/vita-epfl/RAP)

全部 issue **20**（open 12、closed 8），discussions **0**，拉取评论 **36** 条；筛出 **7** 条。样本中对应：RAP-DINO[A]。
较值得关注：[# 10](https://github.com/vita-epfl/RAP/issues/10) 发布权重分数差追至 deformable attention bug；用户更新后确认复现；[# 8](https://github.com/vita-epfl/RAP/issues/8) 作者给 PDM CPU/Ray 评测设置及增强、扰动数据量；[# 14](https://github.com/vita-epfl/RAP/issues/14) 作者披露栅格化相机外参有人为调整且仍可能错位。

### [valeoai/DrivoR](https://github.com/valeoai/DrivoR)

全部 issue **32**（open 6、closed 26），discussions **0**，拉取评论 **93** 条；筛出 **15** 条。样本中对应：DrivoR[A], DrivoR[C,excluded]。
较值得关注：[# 54](https://github.com/valeoai/DrivoR/issues/54) HUGSIM 345 场景两组复现仍有差距；作者给旧场景 commit、指定 checkpoint 和 LTF 管线；[# 47](https://github.com/valeoai/DrivoR/issues/47) Nav2 重新缓存后用户确认与论文一致；[# 31](https://github.com/valeoai/DrivoR/issues/31) Nav2 自训差距、官方权重复现及作者实际库版本；[# 4](https://github.com/valeoai/DrivoR/issues/4) navtrain 85k 子集训练对照及作者的划分说明。

### [autonomousvision/navsim](https://github.com/autonomousvision/navsim)

全部 issue **194**（open 41、closed 153），discussions **0**，拉取评论 **526** 条；筛出 **14** 条。样本中对应：PDM-Closed[A,excluded]。
较值得关注：[# 151](https://github.com/autonomousvision/navsim/issues/151) 维护者明确承认 human_penalty_filter bug，随后称 v2.2 修复；[# 40](https://github.com/autonomousvision/navsim/issues/40) warmup_test 与 mini train logs 可重叠，维护者说明该榜的用途；[# 35](https://github.com/autonomousvision/navsim/issues/35) TransFuser 复现差对应单卡 batch 64 与八卡每卡 batch 64 的设置差。

### [NVlabs/GTRS](https://github.com/NVlabs/GTRS)

全部 issue **13**（open 5、closed 8），discussions **0**，拉取评论 **28** 条；筛出 **5** 条。样本中对应：GTRS[A]。
较值得关注：[# 4](https://github.com/NVlabs/GTRS/issues/4) EP 对 pinv 与 solve 及 NumPy 环境敏感，双方给了单帧对照；[# 5](https://github.com/NVlabs/GTRS/issues/5) 作者追加 teacher/student 初始化对照，EPDMS 43.5/43.6；[# 12](https://github.com/NVlabs/GTRS/issues/12) 推理组合权重 grid search 步骤仍无回复。

### [hustvl/Senna](https://github.com/hustvl/Senna)

全部 issue **50**（open 31、closed 19），discussions **0**，拉取评论 **109** 条；筛出 **5** 条。样本中对应：Senna[C,excluded]。
较值得关注：[# 10](https://github.com/hustvl/Senna/issues/10) 作者明确说 Senna-E2E 部分基于内部框架，可能无法完整开源；[# 11](https://github.com/hustvl/Senna/issues/11) 发布的 VLM 权重仅在私有 DriveX 预训练，与 nuScenes 动作准确率复现差的说明；[# 39](https://github.com/hustvl/Senna/issues/39) 导航命令坐标方向疑点仍无作者确认。

### [MSunDYY/SparseOccVLA](https://github.com/MSunDYY/SparseOccVLA)

全部 issue **8**（open 5、closed 3），discussions **0**，拉取评论 **6** 条；筛出 **5** 条。样本中对应：SparseOccVLA[A]。
较值得关注：[# 7](https://github.com/MSunDYY/SparseOccVLA/issues/7) 公开 stage3 配置是否启用规划未获作者回复；[# 8](https://github.com/MSunDYY/SparseOccVLA/issues/8) stage2 use_gen_token 设置疑问未获回复；[# 2](https://github.com/MSunDYY/SparseOccVLA/issues/2) 作者说缺失的数据准备文件来自 OmniDrive。

### [E2E-AD/AD-MLP](https://github.com/E2E-AD/AD-MLP)

全部 issue **6**（open 3、closed 3），discussions **0**，拉取评论 **15** 条；筛出 **3** 条。样本中对应：AD-MLP[A]。
较值得关注：[# 4](https://github.com/E2E-AD/AD-MLP/issues/4) 用户提出未来 GT 字段混入 CAN bus 的指控；作者承认重审数据并修订流程，未逐项承认该指控；[# 5](https://github.com/E2E-AD/AD-MLP/issues/5) CAN bus 重建数据与发布 pkl 不一致，作者解释 ST-P3/VAD 预处理差别；[# 6](https://github.com/E2E-AD/AD-MLP/issues/6) 作者否认“因前两帧表现差而排除”的因果说法。

### [Tsinghua-MARS-Lab/DriveMA](https://github.com/Tsinghua-MARS-Lab/DriveMA)

全部 issue **3**（open 1、closed 2），discussions **0**，拉取评论 **9** 条；筛出 **2** 条。样本中对应：DriveMA-4B[A]。
较值得关注：[# 2](https://github.com/Tsinghua-MARS-Lab/DriveMA/issues/2) Stage 2 RFS 7.28 对论文 7.89，作者确认各阶段只训 1 epoch；[# 3](https://github.com/Tsinghua-MARS-Lab/DriveMA/issues/3) 作者给 2B SFT 前后分数但 4B 档案暂缺，KL 系数问题仍未答。

### [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA)

全部 issue **58**（open 6、closed 52），discussions **0**，拉取评论 **107** 条；筛出 **14** 条。样本中对应：AutoVLA[B]。
较值得关注：[# 48](https://github.com/ucla-mobility/AutoVLA/issues/48) 官方权重 navtest PDMS 83.69 对论文 89.11，作者强调 CoT 开/LoRA 关；[# 42](https://github.com/ucla-mobility/AutoVLA/issues/42) 作者最终说明 nuScenes 附录结果需额外 RFT，公开权重基于 NAVSIM；[# 22](https://github.com/ucla-mobility/AutoVLA/issues/22) test 路径训练配置为作者确认的误贴，已修正；[# 43](https://github.com/ucla-mobility/AutoVLA/issues/43) 作者确认 B2D 管线仍有未发布内部依赖。

### [AFARI-Research/WA-JEPA](https://github.com/AFARI-Research/WA-JEPA)

全部 issue **2**（open 1、closed 1），discussions **0**，拉取评论 **1** 条；筛出 **2** 条。样本中对应：WA-JEPA[A]。
较值得关注：[# 1](https://github.com/AFARI-Research/WA-JEPA/issues/1) 作者给固定 seed 下去噪步数、FPS、NAVSIM EPDMS 的对照，公开默认改为 4 步；[# 2](https://github.com/AFARI-Research/WA-JEPA/issues/2) 第一阶段 checkpoint 发布请求尚无回复。

### [hyzhou404/UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM)

全部 issue **2**（open 2、closed 0），discussions **0**，拉取评论 **1** 条；筛出 **0** 条。样本中对应：UniAD[B]。
现有 issue 主要涉及环境依赖或一般使用问题，未见符合四类筛选口径的明确陈述。

### [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM)

全部 issue **0**（open 0、closed 0），discussions **0**，拉取评论 **0** 条；筛出 **0** 条。样本中对应：LTF[B]。
仓库在抓取时没有 issue 或 discussion，因此无匹配记录。

### [NVlabs/BEV-Planner](https://github.com/NVlabs/BEV-Planner)

全部 issue **4**（open 3、closed 1），discussions **0**，拉取评论 **2** 条；筛出 **2** 条。样本中对应：BEV-Planner++[A]。
较值得关注：[# 2](https://github.com/NVlabs/BEV-Planner/issues/2) 用户称评测需要 gt_fut_segmentations 并请求生成文件，未回复；[# 3](https://github.com/NVlabs/BEV-Planner/issues/3) CCR 计算代码位置未获回复。

## 限制

- 帖子里的用户推测均保持为报告或疑点；作者没有确认的原因不作为事实。某些数字仅见截图，CSV 不转录截图中无法逐字校验的数值。
- 论文分数若出自 issue 用户引述或第一轮 `sampling.csv`，CSV 保留该语境，并给样本论文 URL；跨协议或不同权重不作直接相减。
- issue 关闭只代表 GitHub 状态，不自动视为复现问题解决。抓取是时间快照，后续新增回复、修改或删除不在本次统计内。
