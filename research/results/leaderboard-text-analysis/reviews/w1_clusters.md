# W1 专项复核：BLUE、BEV-Planner++、DriveMA

复核对象为 `out2/w1_ablation_ledger.csv`，下列行号是**含表头的 CSV 物理行号**。依据 `out2/w1_component_codebook.md` 的字段定义、三篇本地论文及相关项目 README；`out2/` 下没有单独的 W1 README 文件。未运行模型、仿真或 GPU，也未改主账本。

## BLUE：训练 gate 不等于全部权重不变

论文 [blue.pdf](../../papers/blue.pdf#page=3) PDF p.3 Figure 3 明确：视觉编码器及 VLA 主干冻结，但单隐层 MLP gate 用 BCE **另行训练**。PDF p.4 Table 1 标 gate 有 0.11M 可训练参数；PDF p.6 Table 6 明说 SimLingo 与 CriticVLA 各有自己训练的 gate，转移 gate 是在另一主干训练的。PDF p.9 Table 11 及正文又说明 ReCogDrive gate 在 NAVSIM 训练集 90% 上训练，10% 用于阈值选择。因此“主干冻结”不能概括为“完整系统同权重”。

| CSV 物理行 | 当前 `same_weights_or_retrained` | 建议值 | 具体原因／附注 |
| --- | --- | --- | --- |
| **557–564**（Table 1/3/10）、**565–566**（Table 5）、**567–574**（Table 6） | `same_weights` | **`retrained`** | 基线是无 gate 的 SimLingo/CriticVLA，变体含另行训练的 gate。567–568、571–572 是**转移已训练 gate**，并非在目标主干重新训练；建议 `notes` 明说 gate 由另一主干训练、目标 VLA 主干冻结。569–570、573–574 是当前主干的 matched gate。改字段依据 codebook 中“训练模块改变，另行训练”的定义；不能把这些差值称为固定全套权重的纯推理收益。 |
| **575–592**（Table 7） | `same_weights` | **保留 `same_weights`，但补 `notes`** | 这组是在相同 SimLingo **主干**上换门控决策策略，规则 gate/随机 gate 无新神经网络训练，符合 codebook 的“固定基础 checkpoint，仅改规则”口径。不过 BLUE 基线使用已训练的 gate，规则臂不用该 gate，故非完整系统的逐参数同权重消融。建议逐行加：“同一 SimLingo backbone；BLUE 臂含已训练 gate，变体臂改用规则/随机策略；两臂语言激活率不同（BLUE 21.44%，见 Table 7），策略效果与计算量未严格配平。”若团队把 `same_weights` 定义为**完整推理栈**权重同一，应统一改这 18 行为 `mixed_checkpoints` 并同步修改 codebook；当前 codebook 的“基础 checkpoint”定义下保留最一致。 |
| **593–598**（Table 8）及 **603**（Figure 7） | `retrained` | **保留 `retrained`** | gate dropout/宽度或训练数据改变，需要另训 gate；Table 8 明确其他训练数据、标签、阈值相同，PDF p.8。 |
| **599–600**（Table 11） | `same_weights` | **`retrained`** | 同一 ReCogDrive backbone，但 BLUE gate 被另行训练，PDF p.9 正文明确 90%/10% 训练与阈值选择。`notes` 应写“主干相同，训练新 gate；不是全系统同权重”。 |
| **601–602**（Figure 6） | `same_weights` | **保留 `same_weights`** | 比较同一个已训练 BLUE gate 的 θ=0/1 与 θ=0.66，仅在推理时调阈值。PDF p.7 正文给出端点 SR 66.91/69.55 与 θ=0.66 的 76.18，p.8 Figure 6 给出连续曲线。可在 `notes` 加“同一 gate checkpoint，仅阈值变化”。 |

上表明确建议改字段的 BLUE 行共 **20** 行（557–574、599–600）；Table 7 的 18 行是语义边界及限定语风险，并非另训 18 个 gate 的证据。

## BEV-Planner++：Table 1 的 checkpoint 来源混杂

[bev_planner.pdf](../../papers/bev_planner.pdf#page=5) PDF p.5 Table 1 的 `ckpt. source` 逐项为 UniAD **ID1 Reproduce / ID2 Official / ID3 Reproduce**，VAD-Base **ID4 Reproduce / ID5 Official / ID6 Official**。表注称 ID1、3、4 来自作者对官方代码的修改；附录 PDF p.11 说明 ID1/4 关闭 BEV 的 CAN bus，ID3 在规划头拼接 ego 状态。`Official` 是来源标签，**不能推出 ID5 与 ID6 为同一个 checkpoint**，更不能推出跨 `Official`/`Reproduce` 的配对同权重。Table 2 的 VAD-Base*（对应 ID5）与 VAD-Base（ID6）检测 NDS/mAP 也不同（46.0/47.5 对 45.5/47.0），进一步不支持简单视作完全相同模型仅切一个推理开关。

| CSV 物理行 | 配对（Table 1 ID） | 当前值 | 建议值与应加 `notes` |
| --- | --- | --- | --- |
| **604–606** | UniAD ID1 → ID2 | `retrained` | 建议 **`mixed_checkpoints`**：`Reproduce → Official`；同时变化 ego 输入与 checkpoint 来源，差值不能纯归因 ego 状态。若保留 `retrained`，至少必须写出来源混杂。 |
| **607–609** | UniAD ID2 → ID3 | `retrained` | 建议 **`mixed_checkpoints`**：`Official → Reproduce`；ID3 是在规划头增加状态的修改版，跨来源，不能称同权重。 |
| **610–612** | VAD ID4 → ID5 | `retrained` | 建议 **`mixed_checkpoints`**：`Reproduce → Official`；ID4 修改 BEV 输入，来源与 ID5 不同。 |
| **613–615** | VAD ID5 → ID6 | `retrained` | **保留 `retrained`**；两者虽都标 `Official`，Table 1 未给相同 checkpoint 标识，且规划头输入配置不同。`notes` 应写“ID5/6 均标 Official，但不能据此断定同一权重；Table 2 两模型的检测指标亦不同”。 |
| **616–624** | BEV-Planner ID9 → ID10 → ID11 → ID12 | `retrained` | **保留 `retrained`**。Table 1 表注指出 ID9 无历史信息、ID10 用四帧历史、ID11/12 再逐级加 ego 状态；论文没有声称这是同一 checkpoint 的推理开关。 |

`mixed_checkpoints` 在这里编码的是 Table 1 **明确给出的不同 checkpoint 来源**；它不主张已核实两个 checkpoint 文件的 hash 或训练随机种子。若 codebook 要求只有识别了两个具体文件才能用该值，可把 604–612 记 `unknown`，但仍需逐行注明 `Official/Reproduce` 来源混杂。关键是不要在分析中把这 9 行作为严格配对的同权重 ego 状态消融。

## DriveMA：收窄 `REAL-WOD-001`

原 finding [`REAL-WOD-001`](../../out/findings.jsonl) 记录 `trajectory_reward.py` 用官方 RFS 计算器构造轨迹奖励；[drivema.pdf](../../papers/drivema.pdf#page=5) PDF p.5 §4.1 明说 WOD-E2E 的 `Rtraj` 由 RFS 实例化。PDF p.7 Table 3 说明所有 RL 变体都从 `Meta-Action SFT w/ ACP` 初始化，Vanilla GRPO 使用完整奖励但没有 turn-level credit assignment。故 finding ID 应连到**比较两端首次加入 RFS 轨迹奖励**的增量；仅在两个已用 `Rtraj` 的 RL 变体之间加 `Rcons` 或 `Rmeta`，不是该代码发现的增量。

| CSV 物理行 | 当前 `hack_finding_id` | 建议值 | 原因 |
| --- | --- | --- | --- |
| **236–238**（Meta-Action SFT w/ ACP → Vanilla GRPO w all rewards） | `REAL-WOD-001` | **保留 `REAL-WOD-001`** | 从 SFT 到 RL 首次引入包括官方 RFS 的 `Rtraj`。但还同时引入 RL、`Rcons`、`Rmeta`，`notes` 应明确“RFS 是新增奖励之一；Table 3 不隔离其纯贡献”。 |
| **239–241**（Vanilla GRPO → turn-level RL full rewards） | `N/A` | **保留 `N/A`** | 两端已有全部奖励，包括 RFS；比较的是 credit assignment。 |
| **242–253**（各类 Rcons/Rmeta 边际比较） | `REAL-WOD-001` | **全部改 `N/A`** | 每个配对两端均含 `Rtraj=RFS`；实际增加的是一致性奖励或 meta-action 奖励，不能把 RFS 奖励代码发现重复链接到这些增量。 |
| **254**（Figure 5 多阶段束） | `N/A` | **保留 `N/A`** | 无可读数值，且不是隔离 RFS 奖励的实验。 |

若希望账本显式记录更直接的 RFS 引入对照，可在现有 **238 与 239 之间**新增三条 Table 3 比较：`Meta-Action SFT w/ ACP → turn-level RL: Rtraj`，RFS overall **7.893→8.009（+0.116）**、RFS spotlight **7.189→7.146（−0.043）**、ADE@5s **2.774→2.627（−0.147 m）**，链接 `REAL-WOD-001`。这仍是“SFT→RL 且使用 RFS”的联合变化，不能声称识别 RFS 奖励相对其他 RL 奖励的单独因果贡献。
