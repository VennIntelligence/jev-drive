# W5 修订版独立复核（2026-09-24）

## 抽样与结果

- 输入：`/data/hack_audit/out2/w5_paper_only.csv`，45 条数据行，SHA-256 `85b8dab47847d77981b76ff6f4d2b58424b6239d7e41626bfacba52db3cd3e96`；codebook SHA-256 `d7efd751095edb9c96fe0ffbf98a15c0e2d41b46dee72996880220d2db748f8d`。
- 按 `random.Random(20260924).sample(range(45), max(10, ceil(0.1*45)))` 抽取 10 条；零基行号依抽样顺序为 `[0,42,34,40,9,35,36,3,27,17]`，对应 `W5-001, W5-043, W5-035, W5-041, W5-010, W5-036, W5-037, W5-004, W5-028, W5-018`。
- **结果：10/10 PASS，一致率 100%，达到 README 要求的 ≥90%。** PASS 表示该行主要设计、类别、报告的数字或定性对照、论文出处及有限措辞的代码状态可由资料支持；不表示论文结果有独立复现。
- 仅阅读本地论文 PDF、固定提交的本地仓库和公开元数据；未运行模型、仿真、评测或 GPU 作业。

## 逐条核对

| ID | 判定 | 论文、类别及分数证据 | 代码状态与 URL 证据 |
| --- | --- | --- | --- |
| `W5-001` | **PASS** | [LinkVLA Fig. 2，PDF p.3](https://arxiv.org/pdf/2603.01441#page=3) 与 §3.1–3.2，pp.4–5 说明统一 action/text 词表及从动作反推语言的辅助损失；[Table 5，p.8](https://arxiv.org/pdf/2603.01441#page=8) 为 DS `85.07→89.57→89.85→91.01`。CSV 的 tokenization `85.07→89.57` 和在 C2F 后加 alignment `89.85→91.01` 正确，`language_action_alignment_training` 适用。 | 所列 [GitHub 搜索](https://github.com/search?q=LinkVLA&type=repositories) 可访问；`gh search repos LinkVLA` 返回空列表，只能支持“未检出”，CSV 未断言绝对无仓库。 |
| `W5-043` | **PASS** | [DrivoR Table 4a，PDF p.7](https://arxiv.org/pdf/2601.05083#page=7) 的 NAVSIM-v1 navval PDMS 为随机初始化 `70.1`、ImageNet21k `87.5`、DINOv2 `90.0`，且正文说后续实验使用 DINOv2；`pretrained_backbone` 适用。[Table 2 注释，p.5](https://arxiv.org/pdf/2601.05083#page=5) 和 p.6 正文支持 NAVSIM-v1 模型零样本评 HUGSIM，不能把 Table 4a 差值直接归于 HUGSIM。 | [DrivoR 固定提交 README L140–185](https://github.com/valeoai/DrivoR/blob/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a/README.md#L140-L185) 提供 NAVSIM 评测命令；本地该提交未检出 HUGSIM 入口，CSV 的“未见”限定成立。 |
| `W5-035` | **PASS** | [NTR Fig. 2/§3.1，PDF p.4](https://arxiv.org/pdf/2605.31116#page=4) 及 §3.2 p.5 说明重建梯度经过 scene token bottleneck，EMA teacher 给 latent target，辅助分支仅训练时使用。[Table 4，p.8](https://arxiv.org/pdf/2605.31116#page=8) 列 baseline `7.652`、Frozen+Random `7.754`、EMA+Random `7.817`、完整 NTR `7.974` RFS；CSV 的定性对照准确，类别适用。 | 所列 [GitHub 搜索](https://github.com/search?q=NTR+driving&type=repositories) 可访问；`gh search repos 'NTR driving'` 返回空列表，支持“未检出”措辞。 |
| `W5-041` | **PASS** | [Poutine Table 1 与 Implementation Details，PDF p.4](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf#page=4)：Poutine-Base 验证集 no-CoT `8.12`、CoT `8.08`；同页最终推理文字写 CoT 温度 `1e-6`、轨迹 greedy decoding、移除 intent。CSV 已将基座对照与最终推理区分，`inference_reasoning_mode` 适用。 | 所列 [GitHub 搜索](https://github.com/search?q=Poutine+driving&type=repositories) 可访问；`gh search repos 'Poutine driving'` 返回空列表，支持“未检出对应模型仓库”。 |
| `W5-010` | **PASS** | [RoG-DAgger Fig. 3，PDF p.3](https://arxiv.org/pdf/2608.24525#page=3) 与 §4.1 p.6 支持学生闭环状态采样、expert 接管及新旧数据混合微调。[Table 1，p.6](https://arxiv.org/pdf/2608.24525#page=6)：SimLingo `85.07/67.27`，RoG-DAgger `90.34/73.51`，相差 DS `5.27≈5.3`、SR `6.24≈6.2` 个百分点；类别适用。 | 所列 [GitHub 搜索](https://github.com/search?q=RoG-DAgger&type=repositories) 可访问；`gh search repos 'RoG-DAgger'` 返回空列表，支持“未检出”。 |
| `W5-036` | **PASS** | [NTR §3.3，PDF pp.5–6](https://arxiv.org/pdf/2605.31116#page=5) 说明语义先验决定遮蔽重建位置，推理时移除；p.3 明确说先验来自 SAM3 mask。[Table 4，p.8](https://arxiv.org/pdf/2605.31116#page=8) 比较 Frozen+Random `7.754` 与 Frozen+Semantic `7.781`、EMA+Random `7.817` 与 EMA+Semantic `7.974` RFS。`masked_scene_token_reconstruction` 适用。 | NTR 的 [仓库搜索链接](https://github.com/search?q=NTR+driving&type=repositories) 可访问；同 `W5-035`，未检出而非证明不存在。 |
| `W5-037` | **PASS** | [NTR §3.1，PDF p.4](https://arxiv.org/pdf/2605.31116#page=4) 明确为 DrivoR 式候选轨迹加 trajectory-conditioned scorer，并称 NTR 保持评分模块不变；[Table 1，p.7](https://arxiv.org/pdf/2605.31116#page=7) 只给整体 WOD 成绩。论文未说明该 scorer 的监督目标是 RFS/子分数，故修订后的 `learned_candidate_scoring` 类别及“无同配置单独消融”备注准确，符合与原 `metric_proxy_candidate_selection` 的边界。 | NTR 的 [仓库搜索链接](https://github.com/search?q=NTR+driving&type=repositories) 可访问；同 `W5-035`。 |
| `W5-004` | **PASS** | [LinkVLA Table 7，PDF p.8](https://arxiv.org/pdf/2603.01441#page=8)：GPS target point DS/SR `91.01/74.55`，navigation command `91.25/73.18`；这是导航输入形式比较，修订后的 `navigation_input_conditioning` 正确。CSV 未把两个方向不同的指标合成优劣结论。 | LinkVLA 的 [仓库搜索链接](https://github.com/search?q=LinkVLA&type=repositories) 可访问；同 `W5-001`。 |
| `W5-028` | **PASS** | [Senna Table II，PDF p.6](https://arxiv.org/pdf/2410.22313#page=6) 注释 `*` 表示输入 ego status；Senna* 平均 L2/碰撞 `0.22/0.08`，无星 Senna `0.59/0.18`，表中确列有/无状态的分数。视觉规划器加 ego 状态与原 codebook `ego_state_fusion` 相符；CSV 未声称跨行差值是单因素效应。 | [Senna 固定提交 README L25–34](https://github.com/hustvl/Senna/blob/31a3a2336e9494c254b612665fe55e95f67e9cae/README.md#L25-L34) 只宣布 Senna-VLM 代码和权重；本地该提交有 VLM/meta-action 脚本，未见对应 `0.22` L2 的 Senna-E2E 规划实现。 |
| `W5-018` | **PASS** | [SimLingo Appendix Table 9a，PDF p.17](https://arxiv.org/pdf/2503.09594#page=17)：waypoint-only DS `3.21`、加 path `4.49`，静态布局碰撞 `0.68→0`；§3.3 p.4 说明 path 与时序 waypoint 解耦，并由两个 PID 控制器转为转向/加速。`output_representation` 适用，CSV 未混同完整 SimLingo 分数。 | [SimLingo 固定提交 README L86–93、L193](https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L86-L93) 确认共用闭环 agent 与 Base 训练目录；[模型文件列表](https://huggingface.co/api/models/RenzKa/simlingo) 仅见 `simlingo` checkpoint，未见能对应旧 Base 官方 `6.25` DS 的 checkpoint/提交配置。 |

## 应修订的非阻断事项

1. `W5-001` 的 `source` 建议补 LinkVLA PDF pp.4–5（§3.1–3.2）：现有 Fig. 2 p.3 与 Table 5 p.8 足以定位架构和数字，但反向语言损失的公式在 pp.4–5。
2. `W5-036` 的 `source` 建议补 NTR PDF p.3：§3.3 pp.5–6 描述 foundation segmentation model，具体 **SAM3** 名称在 p.3。`reported_score_effect` 的“随机目标”宜写成“随机重建位置”，以对应 Table 4 的 `Recon. Target Selection` 列。
3. `W5-018` 的 `source` 建议补 SimLingo PDF p.4（§3.3）以定位 PID 转车控；`reported_score_effect` 可直接填 Table 9a 数值 `3.21→4.49`，减少读者二次查表。
4. `W5-010` 的 Table 1 差值是论文所报的跨行比较；§4.1 同时披露其实现将原 SimLingo 的 InternVL-2 backbone 换成 Qwen3-VL-2B。建议在 `notes` 补这一比较限制，避免把 `+5.3/+6.2` 解读为严格单因素 DAgger 消融。

以上四项均不改变当前记录可核对的数值、类别或有限措辞的代码状态，故未计 FAIL。另查看非样本 `W5-045`：它已将 DrivoR 的 scene token 压缩单列，依据同论文 Table 4b p.7；未发现与 `W5-043` 重复合并的旧问题。

## 链接与范围

本次抽样的 `source` 和 `code_status_source` 共 20 个唯一 URL，按 CSV 原链接逐一请求；首轮 4 个连接超时，重试后 **20/20 返回 HTTP 200**。三个有仓库的方法，其本地 HEAD 分别为 SimLingo `743b243...`、Senna `31a3a233...`、DrivoR `fc6e5aa...`，与 CSV permalink 一致。GitHub 搜索链接是随时间变化的查询结果，只支持所记日期和关键词下的“未检出”；代码状态不等同于完整复现性结论。
