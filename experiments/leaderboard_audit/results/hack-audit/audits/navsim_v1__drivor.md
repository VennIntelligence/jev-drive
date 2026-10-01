# NAVSIM v1：DrivoR

- 榜单口径：navtest，PDMS 93.7 对应论文 Table 1 的 **trainval**。同表 SimScale 版 94.0/94.6 是不同训练规模，不混入本抽样行。
- 固定版本：[`fc6e5aa`](https://github.com/valeoai/DrivoR/tree/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md` 的 v1 训练与评测命令、scaling 表；`navsim/agents/drivoR/drivor_features.py` 的较长未来目标、`drivor_model.py` 的评分选轨、`layers/losses/drivor_loss.py` 的双目标/评分损失；`navsim/planning/script/config/common/agent/drivoR.yaml`。
- `papers/drivor.pdf` 提取文本的 §3–4、Table 1/5/6/7、附录 Table 8/10。

## 未读与局限

- 未读完整 DINOv2 骨干、SimScale 缓存生成、全部 `nuplan-devkit/`、权重及日志；未运行目标开关的反事实评测。
- Table 7 的 +0.6 是 **navval** 消融，不是 navtest 93.7 的直接分解。

## 整体印象与发现

评分器拟合官方子指标并从 64 候选中挑选；v1 另构造偏进度的更长目标。论文公开披露了两者，而且较长目标在 v2 验证上变差，形成较强的榜单依赖证据。发现：**NAV1-DRIVOR-001、NAV1-DRIVOR-002**。
