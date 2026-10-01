# NAVSIM v1：DriveVLA-M0

- 榜单口径：navtest，PDMS；抽样 94.1 为论文 Table 1 的 **Scale**，该表 Base 为 92.3。
- 固定版本：[`7fabe16`](https://github.com/ZebinX/DriveVLA-M0/tree/7fabe160fc9bb41f9278845b36d457bf871f697a)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md`；`configs/base_model_navtest.yaml`；`scripts/run_base_pdms.sh`；`eval.sh`；`navsim/agents/EpisodeDrive/drivevla_backbone.py`、`drivevla_base_agent.py`、`episodedrive_agent.py`、`action_decoder.py`、`retrieve_model/retrieve_agent.py` 的入口和选轨片段。
- `papers/drivevla_m0.pdf` 提取文本的 §3–4、Table 1/3/4、附录 A/C。

## 未读与局限

- 固定提交的 `run_base_pdms.sh` 指向 `episode_drive` Base，Retrieve Model 是独立检索可视化/验证入口；未找到把 memory 检索与 TTT 接入 navtest 轨迹生成的完整代码。论文 **94.1 Scale** 所需的 10K memory 和适配链无法由这份公开提交静态复现。README 也明确称本次 release 聚焦 Base 和 Retrieve 路径。
- 独立复核补出入口不一致：`drivevla_base_agent.py:490` 的 `compute_trajectory` 读取 `predictions["pred_traj"]`，而已读 `forward`/评分头返回和 `run_create_submission_pickle.py:82–87` 使用 `predictions["trajectory"]`；README 的直接 CPU PDMS 路径可能因此失败。另一提交脚本与多 GPU `predict_step_drivor` 正确读取 `trajectory`，故确认选轨机制仍成立，但没有运行任何入口来验证哪条用于论文分数。
- 系统提示虽被送入推理模板，未有有无提示消融；因此 NAV1-DVM0-002 只作低置信度机制记录，不推断其定量贡献。
- 未读全部 InternVL 外部实现、训练数据生成、权重、日志、第三方依赖；未验证 Scale 实际推理。

## 整体印象与发现

可审代码显示两种榜单定制：按 PDMS 公式选 64 候选轨迹，以及把评分和非反应式仿真细节写入 VLM 提示。论文披露了指标和评分头，但固定提交尚不足以把这两点直接归因到 Scale 的 94.1。发现：**NAV1-DVM0-001、NAV1-DVM0-002**。
