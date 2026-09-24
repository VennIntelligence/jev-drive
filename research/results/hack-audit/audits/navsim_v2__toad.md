# NAVSIM v2：DrivoR+TOAD

- 榜单口径：**navhard 双阶段 EPDMS 56.3**；不是 v1 的 navtest PDMS。
- 固定版本：[`cfa88e0`](https://github.com/valeoai/TOAD/tree/cfa88e008080d0799b2859b5935c257048ec0adf)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md` 的 v2 Evaluation/Submission 命令与 Results；`navsim/agents/drivoR/drivor_model.py` 的候选聚合、CEM 采样/舒适/锚定及最终回退；`navsim/planning/script/config/common/agent/drivoR.yaml`；`navsim/planning/script/run_pdm_score_gpu_v2.py` 的双阶段入口。
- `papers/toad.pdf` 提取文本的 §3–5、Table 2/3/6、Figure 4、附录 A。

## 未读与局限

- 未读便携打包目录的全部副本、评分缓存生成全路径、权重、数据和运行输出；未实测新城市或闭环交通。
- 论文 Figure 4 在 warmup 上消融，其中与 navhard 有场景交集且作者说明得到主办方认可。该事实是验证划分依赖，不等同于作弊，也不等于能量化交集带来的 navhard 提分。

## 整体印象与发现

本提交有完整 v2 命令：5 轮 CEM、每轮 64 样本、64 个原始候选，54.6→56.3 EPDMS 在论文 Table 2 明确给出。另一独立可见机制是用与测试集有交集的 warmup 划分做超参消融。独立复核还补出评测命令把 PDM 子分数权重改为 10/13/6/14/15/2，源码确实用这组值决定候选排序；其贡献没有单项消融。发现：**NAV2-TOAD-001、NAV2-TOAD-002、NAV2-TOAD-003**。
