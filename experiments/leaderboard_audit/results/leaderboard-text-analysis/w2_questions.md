# W2 留给综合阶段的问题

以下问题只定位记录，不在 W2 回答；主代理可合并进 `questions_for_synthesis.md`。

1. Bench2Drive 官方 Table 3 中，`UniAD-Base` 的开环 Avg L2 为 0.73、闭环 DS/SR 为 45.81/16.36；`VAD` 分别是 0.91、42.35/15.00。正文却称 UniAD-Base 闭环较差，可能指其低于 VAD 的 Efficiency/Comfortness。综合阶段应如何解释该语句与不同闭环指标的关系，而不把机械 Spearman 当作因果或部署能力？涉及 `w2_cross_board.csv` 中 Bench2Drive `base-set open-loop 2s@2Hz` 与 `base-set closed-loop 220 routes` 的对应行，以及 `w2_correlations.md` 首四项。
2. NAVSIM v1 `navtest` PDMS 与 NAVSIM v2 `navhard-two-stage` 修复后 EPDMS 的排序变化，究竟来自场景分布、指标、两阶段合成观测、训练目标，还是 checkpoint/重评版本？涉及 `w2_cross_board.csv` 中这两列、TOAD Tables 5/6、DrivoR Tables 13/14 及 `w2_correlations.md` 第五项。
3. 对同方法跨榜的统计，什么证据足以确认同一 checkpoint？TOAD 说使用公开 checkpoint 且不更新权重，但没有逐一对应 v1/v2 的文件或 hash；RAP-DINO 自述直接评估同一训练模型，而其 WOD 和 Bench2Drive 分别微调和更换骨干。涉及 `RAP-DINO`、`RAP-ResNet`、`DrivoR`、`iPad` 等行的 `same_checkpoint` 字段。
4. CARLA LB2 官方 `MAP`/`SENSORS` 的 DS、Bench2Drive 220 短路线 DS 与 Longest6 v2 长路线 DS 对同一方法分别代表哪些可迁移行为？是否应按路线长度、早停规则、传感器轨道和专家数据来源分层？涉及 `SimLingo-BASE`、`TF++`、TFv6 Table 5 的变体、RoG-DAgger Table 1，以及 `w2_correlations.md` 的 Longest6 四项。
5. NAVSIM v2 旧 `navtest EPDMS*`、修复后 `navtest EPDMS` 和官方 `navhard-two-stage EPDMS` 应如何进入统一综合表？NTR 的 `navtest` 90.9 未说明修复状态。涉及 `SparseDriveV2`、`DriveFuture`、`WA-JEPA`、`NTR`、`LTF` 和 `DrivoR` 行的 `protocol/split`。
6. `WOD-E2E test RFS` 与 `NAVSIM v1 navtest PDMS` 的五个重名方法中，有两个 DriveMA 模型规模，且至少部分跨数据集重新训练。这个样本级相关是否值得用于跨榜一致性判断，还是只应作已发表分数索引？涉及 `AutoVLA`、`DriveMA-2B`、`DriveMA-4B`、`NTR`、`RAP-DINO`，以及 `w2_correlations.md` 最后一项。
7. HUGSIM WA-JEPA 的统一 436 场景协议，与 TOAD 分数据源的四组 HDS、HUGSIM 官方论文原始难度分组如何对齐？是否可对某一固定场景/控制器的结果比较 NAVSIM 排序？涉及 `hugsim` 行及 `w2_correlations.md` 的小样本配对。
8. nuScenes 的 L2 表中 ST-P3、UniAD、OmniSpace 等报告采用不同评测实现、ego 输入或修正，和 Bench2Drive 的重训模型能否视为一对可比 checkpoint？涉及 `AD-MLP`、`VAD-Base`、`AutoVLA`、`OmniSpace`、`SteerVLA` 的 `nuscenes` 与 `bench2drive` 行。
