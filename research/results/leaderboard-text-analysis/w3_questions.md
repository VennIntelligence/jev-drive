# W3 留给综合阶段的问题

以下只列需要跨论文、代码与基准协议综合判断的问题；issue 本身的明确陈述已经在 `w3_issues.csv`，这里不作结论。

1. [TFv6 #90](https://github.com/kesai-labs/lead/issues/90) 中作者称论文 Table 5 的 LEAD expert Bench2Drive 数字是估计，而非实际 Bench2Drive 评测。综合阶段应如何标注该行与直接测得的 DS 的可比性？相关记录：`kesai-labs/lead#90`。
2. [AD-MLP #4](https://github.com/E2E-AD/AD-MLP/issues/4) 的用户指控未来 GT 混入 CAN bus，作者确认重审数据并修订训练流程，但未逐项确认用户的具体指控。应如何把旧/新论文分数、发布 pkl 和代码证据对应起来？相关记录：`E2E-AD/AD-MLP#4,#5`。
3. [NAVSIM #151](https://github.com/autonomousvision/navsim/issues/151) 明确承认 human penalty filter bug；挑战期保留，后修复。哪些榜单快照与论文分数对应修复前或修复后协议？相关记录：`autonomousvision/navsim#151,#158,#172`、`valeoai/DrivoR#31,#47`。
4. [DrivoR #54](https://github.com/valeoai/DrivoR/issues/54) 多名用户在旧 345 场景 HUGSIM 上得到不同复现值，作者指定权重和 LTF 管线后仍有未解释差距。哪些配置差异可由原文核实，哪些仍不可判定？相关记录：`valeoai/DrivoR#34,#38,#54`。
5. [SimLingo #43](https://github.com/RenzKa/simlingo/issues/43) 中改用 Bench2Drive 定制目录后 DS 75.50→86.53，且 Town13 进入训练。综合阶段如何与 CARLA LB2 和 Bench2Drive 的其他分数分开比较？相关记录：`RenzKa/simlingo#43,#72`、`autonomousvision/carla_garage#95,#103,#108`。
6. [AutoVLA #42](https://github.com/ucla-mobility/AutoVLA/issues/42) 中作者最终说 nuScenes 附录结果需要额外 RFT，发布的 checkpoint 是 NAVSIM RFT；[B2D #43](https://github.com/ucla-mobility/AutoVLA/issues/43) 的评测管线也有未发布依赖。哪些已发表数字能对应到可取得的权重与代码？相关记录：`ucla-mobility/AutoVLA#30,#42,#43,#48,#56`。
7. [GTRS #4](https://github.com/NVlabs/GTRS/issues/4) 与 [NAVSIM #140](https://github.com/autonomousvision/navsim/issues/140) 提出 `pinv`/`solve`、NumPy 版本影响 EP。该差别影响哪些已发表 EPDMS/PDMS 对照，是否有固定环境可重算？相关记录：`NVlabs/GTRS#4`、`autonomousvision/navsim#81,#140`。
8. [SparseDriveV2 #7](https://github.com/swc-17/SparseDriveV2/issues/7) 作者称推理评分设计为最大化 PDMS，[#13](https://github.com/swc-17/SparseDriveV2/issues/13) 称 Bench2Drive 未用该监督。综合阶段如何解读该模型两榜分数的训练目标差异？相关记录：`swc-17/SparseDriveV2#7,#13,#16`。
