
### `metric_proxy_optimization`（提议者：agent_nav）

定义：训练一个模仿榜单评分项的代理评分器，在推理中用其选择或搜索轨迹；其榜单增益依赖代理评分与真实驾驶效用的一致性。正例：TOAD 在 NAVSIM-v2 中以预测 PDM 子分数作为 CEM 奖励，使同一 DrivoR checkpoint 的 EPDMS 从 54.6 升至 56.3。

### `benchmark_prompt_specification`（提议者：agent_nav）

定义：将具体榜单评分公式或评测仿真细节写入模型输入提示，使策略可能适配该协议；无提示消融时应降低置信度。正例：DriveVLA-M0 的系统消息直接写出 PDMS 权重和四秒非反应式日志回放。

### `benchmark_target_shaping`（提议者：agent_nav）

定义：为某榜单指标偏好改造训练目标，使轨迹在另一协议或分布出现反向变化。正例：DrivoR 的额外远期目标使 v1 navval PDMS 90.0→90.6，却使 v2 warmup EPDMS 39.4→37.8。

### `benchmark_split_adaptation`（提议者：agent_nav）

定义：在与正式评测场景有交集或高度相近的验证划分上选择推理配置，收益无法视为跨城市或独立样本能力；需要单列披露与官方许可。正例：TOAD 在与 navhard 有交集的 warmup-two-stage 上对 CEM 参数和组件做消融。

### `benchmark_weight_tuning`（提议者：agent_nav）

定义：在固定模型推理时手工调整评分项权重或排序公式以匹配榜单偏好，且这些权重未证明适用于实际部署。正例：DrivoR 的 NAVSIM-v2 命令将 NC/DAC/DDC/TTC/EP 权重设为 10/13/6/14/15，而非官方 EPDMS 的组合。

### `offline_candidate_library`（提议者：agent_nav）

定义：通过从榜单源域轨迹离线聚类的大型静态库扩充推理候选，评分提升依赖库对该域的覆盖。正例：GTRS 把 8192/16384 条 nuPlan 轨迹词表与扩散候选合并，在旧协议消融中动态候选并入后 EPDMS 39.7→40.8。
