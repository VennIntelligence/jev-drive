# CARLA Leaderboard 2.0 · SimLingo-BASE（排除核验记录）

- 论文：[SimLingo](../../papers/simlingo_base.pdf)；Table 1 在 **MAP** 区列 Base DS 6.25，在 **SENSORS** 区列 Base DS 6.87。抽样所用 6.25 不可与 SENSORS 数字混排。
- 原抽样仓库：[RenzKa/simlingo `743b243afd6cf5ff51b9fa1f8cac86f22d569684`](https://github.com/RenzKa/simlingo/tree/743b243afd6cf5ff51b9fa1f8cac86f22d569684)；本地 `repos/carla_lb2__simlingo_base/`。
- 核验结论：论文 Table 1 脚注原文：“we changed the naming of our model from CarLLaVA to SimLingo-BASE”；README L193 同样确认“previously CarLLaVA”。这是旧名对应的 Base 模型，非新增独立方法。仓库没有可对应 MAP 6.25 官方提交的闭环 Base agent/参数，故降 C、排除、发现数 0。

## 已读与未读

已读 `README.md` 方法关系、`simlingo_base_training/config.py`、`simlingo_base_training/models/driving.py`、`start_eval_simlingo.py`、`team_code/agent_simlingo.py` 的模型初始化/推理入口，以及论文 §3–4、Appendix B/C 的 Base、计分、早停与消融。`start_eval_simlingo.py` 的公开示例指向 full SimLingo；其 `agent_simlingo.py` 直接引用 `simlingo_training` 模型类型，不能据此断定 MAP Base 的实际闭环实现。

未逐行读所有模型层和专家数据生成文件；未运行代码。没有 MAP 官方提交参数和 checkpoint，因此不能确认该数字的后处理与复现路径。

## 论文可见但缺少源码定位的榜单机制

SimLingo 论文 §4 写明 Leaderboard 暂停后 full SimLingo 未正式提交；官方榜单数字来自此前的 Base。论文 §4 与 Appendix C.1 披露提前停止，表 9(c) 再列 1300/1800/2100/2400 m 对应官方 DS 3.93/4.49/6.87/6.35。论文 Appendix C.1 解释长路线 DS 偏好不完整路线，并报告 2100 m 和 2400 m 重复提交的均值约 5.9/5.83。**当前仓库无法找到此 Base 的在线早停代码或 MAP 6.25 对应配置**，不将论文描述伪装成固定源码发现。

## 局限

同一模型在 MAP 与 SENSORS 的官方分数不可混比，也不能把 Base 和 full SimLingo 的 Bench2Drive 表现混为同一提交。若取得原官方提交包，可检查早停、集成、控制器和传感器轨道，再补充代码行证据。
