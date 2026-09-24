# WOD-E2E / AutoVLA

- 仓库：[ucla-mobility/AutoVLA，固定 commit ba34eed](https://github.com/ucla-mobility/AutoVLA/tree/ba34eed74ce6729e7986592d0e66cbaca397b4fa)。
- 论文：[AutoVLA](../../papers/autovla.pdf)，核对 §3/§4、Fig. 6、附录 B.1。
- 读过：根 README、Waymo 数据预处理 `dataset_utils/preprocessing/waymo_e2e_dataset.py`、`cot_prompts.py`、`dataset_utils/sft_dataset.py`、`tools/preprocessing/cot_sample_generation.py`、`models/autovla.py` 的奖励路径、公开配置与脚本清单。
- 未读：全部 NAVSIM 上游代码、动作码本细节及未发布的 Waymo 提交脚本；未运行代码或评测。

## 整体印象

Waymo 输入路径从 `past_states` 取自车速度、加速度和历史行为，从 `frame.intent` 取导航指令，并读取历史相机帧。论文称 Waymo 使用 ADE 奖励做 RFT、提交 RFS 7.5566；仓库公开的 RFT 配置和模型奖励调用主要是 nuPlan PDMS 路径，未见完整的 Waymo RFS 提交入口。故其公开程度已经从 A 降为 B，不能复核提交 seed、选择策略和后处理。

## 阶段 1 候选观察（阶段 2 未纳入最终发现）

- **REAL-WOD-004**，阶段 1 的 `answer_conditioned_rationale` 候选：生成 CoT 训练标注时，教师提示直接给出从未来 GT 派生的最佳动作。论文附录披露，尚无去除该提示的独立消融；该条仅涉及训练监督的解释可信度，不宣称评测推理时输入未来标签。阶段 2 认为这属于常规训练监督的可能形式，缺少独立榜单增益证据，因此仅保留在原代理记录，不进入最终 `findings.jsonl` 或矩阵。

阶段 1 原始记录及固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)；排除理由见 [重编码裁决](../sources/recode_decisions.md)。

## 局限

Waymo test 预处理代码尝试由之后帧的历史状态重建目标未来轨迹，但目标写入 `gt_trajectory` 字段，已读推理提示未将其放入模型输入，故没有认定测试标签泄漏。论文在 NAVSIM 提到 oracle best-of-N；这不能移植到 WOD-E2E 的 RFS 记录，故未作为 Waymo 发现。
