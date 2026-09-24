# NAVSIM v2：DrivoR

- 榜单口径：**navhard 双阶段 EPDMS 54.6**；论文 Table 3 和 README scaling 表的版本为 85k navtrain + 134k SimScale、30 epochs。
- 固定版本：[`fc6e5aa`](https://github.com/valeoai/DrivoR/tree/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md` 的 v2 训练、评测、scaling 与提交命令；`navsim/agents/drivoR/drivor_model.py` 评分选择、`drivor_agent.py` 的评分标签接口、`layers/losses/drivor_loss.py`；`navsim/planning/script/config/common/agent/drivoR.yaml`。
- `papers/drivor.pdf` 提取文本的 §3–4、Table 3/6/7、附录 Table 8/10。

## 未读与局限

- README 说明本 repo 基于 Nav1，v2 要在官方 NAVSIM-v2 仓库中复制 agent/config 才能评测。故这里能核对作者给出的 v2 命令及模型选择代码，不能把固定 clone 当成完整自包含的 v2 运行环境。
- 初始 v2 评测命令示例写 `Nav2_10epochs`，但 **54.6** 对应后文 SimScale 134k、30 epochs；不能用前者复现 54.6。
- 未读外部 v2 运行环境、SimScale 原始样本、checkpoint 和日志；没有同一模型原权重与调权的独立 EPDMS 数字。

## 整体印象与发现

v2 模型仍用 v1 式六项评分器选轨，而推理权重在与 navhard 有交集的 warmup 上被重新调整；论文和 README 均披露。改权与交叠验证划分分别编码，主办方认可后者作开发集，不将它称作违规。分数增强至少包含特定榜单的候选排序，54.6 的合成数据扩充本身未被当作 hack。发现：**NAV2-DRIVOR-001、NAV2-DRIVOR-002、NAV2-DRIVOR-003**。
