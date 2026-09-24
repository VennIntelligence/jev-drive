# WOD-E2E / DriveMA-4B

- 仓库：[Tsinghua-MARS-Lab/DriveMA，固定 commit de20e6c](https://github.com/Tsinghua-MARS-Lab/DriveMA/tree/de20e6c64878bc3413d26d69362c3be9b3a91533)。
- 论文：[DriveMA](../../papers/drivema.pdf)，核对 §4、Table 1、Table 3、附录方法和提示词。
- 读过：根 README、`README_DriveMA_Pipeline.md`、`examples/train/grpo/internal/gspo.sh`、`examples/train/grpo/plugin/trajectory_reward.py`、`tools/infer_scripts/run_vllm_infer_mutil_turn.py`、`tools/other/convert_to_submission.py`、RFS 工具。
- 未读：捆绑的通用 ms-swift 实现全集和数据包中的每条标注；未运行代码、模型或评测。

## 整体印象

WOD-E2E 是预测未来 5 秒轨迹的开放环 RFS 任务，不是闭环驾驶。DriveMA 将元动作和轨迹分两轮生成，并在 RL 中直接使用提供人工偏好轨迹的官方 RFS 函数作奖励。论文对此披露且 Table 3 给出训练阶段消融，但不能把整个 RL 差额只归于 RFS 项。模型还用历史轨迹、速度、加速度和路线意图；这些是 WOD 提供的输入，单独使用不能自动判为 hack。

## 发现

- **REAL-WOD-001**，`metric_reward_finetuning`：用 RFS 直接训练轨迹生成器，将评分协议纳入学习目标。

完整记录及固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)。

## 局限

提交转换器将模型的 1 Hz 五点轨迹线性插值为 4 Hz 二十点，并在点不足时按最后一步外推；没有独立证据证明此后处理提高 RFS，因此未记为发现。公开脚本中的局部路径需替换，未用权重或数据核验 4B 提交配置、seed 和实际 RFS 差额。
