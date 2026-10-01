# W5 留给综合阶段的问题

- `W5-013` 和 `W5-017`：CarLLaVA / SimLingo-BASE 的主动提前停路，对官方 LB2 DS 的收益及路线完成率损失该如何共同呈现？重复提交方差见 CarLLaVA PDF p.5。
- `W5-023`–`W5-025`：DriveFuture 的 future latent 使用训练期 GT、测试期预测值；与直接读取测试期未来真值的机制应如何分界？GTRS-Dense scorer 对 navhard 55.5 的独立贡献未量化。
- `W5-026`–`W5-028`：Senna 带 ego status 的 0.22 L2 与不带 ego status、不同碰撞实现的比较如何处理？
- `W5-032`–`W5-034`：LVLDrive 的 LiDAR 输入和自定义碰撞计算与 camera-only nuScenes 排名是否属于同协议？
- `W5-025`、`W5-037`：DriveFuture 与 NTR 的 trajectory scorer 各自是否以榜单总分或子分数训练？两篇论文未交代足以直接归入榜单分数代理的细节。
- `W5-040`：Poutine 的 GRPO 使用 416 个 preference-labeled validation scenarios，对正式 test 成绩的解释边界是什么？
- `W5-041`：Poutine-Base 验证集 no-CoT 优于 CoT，但最终推理段仍描述低温 CoT；具体提交配置如何对应？
- `W5-042`–`W5-045`：DrivoR 论文称 NAVSIM-v1 模型零样本用于 HUGSIM；NAVSIM 的预训练、压缩与评分消融能否解释 HUGSIM 分数？确切 checkpoint 文件和适配入口未公开。
