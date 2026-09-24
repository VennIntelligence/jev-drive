# CARLA Leaderboard 2.0 · CarLLaVA（排除核验记录）

- 论文：[CarLLaVA 技术报告](../../papers/carllava.pdf)；官方 **SENSORS** DS 6.87、RC 18.08、IS 0.42（Table 1）。
- 原抽样仓库：[RenzKa/simlingo `743b243afd6cf5ff51b9fa1f8cac86f22d569684`](https://github.com/RenzKa/simlingo/tree/743b243afd6cf5ff51b9fa1f8cac86f22d569684)；本地 `repos/carla_lb2__carllava/`。
- 核验结论：当前仓库明确把 CarLLaVA 称作 SimLingo-BASE 的旧名，非独立方法；只能定位 Base 训练代码，不能定位 2024 官方提交的闭环 agent/配置。因此该条从 A/B 可审计抽样降为 C 并排除，发现数 0。

## 已读与未读

已读仓库 `README.md` 中方法关系、训练与评测说明，`simlingo_base_training/config.py` 和 `simlingo_base_training/models/driving.py` 的入口，`start_eval_simlingo.py`、`team_code/agent_simlingo.py` 的入口与模型加载段，以及论文 §3–4、表 1–3 和早停/方差段。仓库 `README.md` L193 原文：“SimLingo-Base (previously CarLLaVA - without language capabilities)”。`start_eval_simlingo.py` 指向 full SimLingo 的 `agent_simlingo.py`；当前没有可辨识的 CarLLaVA 官方 agent。

未读全部 Base 网络层和数据生成细节；未下载旧权重或运行代码。不能由现有 full SimLingo agent 证明 2024 CarLLaVA 提交的具体在线行为。

## 论文可见但缺少源码定位的榜单机制

CarLLaVA 论文 §4 主动披露：为适应长路线乘积式 DS，达到指定距离后停止，避免后续碰撞；表 2(c) 的官方 DS 为 1300 m→3.93、1800 m→4.49、2100 m→6.87、2400 m→6.35。论文还报告 2100 m 三次提交 5.5/6.8/5.3、平均 5.87±0.81，说明单次 6.87 不能当稳定效果。**当前 commit 未找到对应 2100 m 在线实现**；按验收的固定 commit 代码行要求，不在 `findings.jsonl` 中硬配来源。该机制可作为报告中的“论文已披露、源码未复现”关键证据与候选实测，但不是已验证代码发现。

## 局限

CarLLaVA 与 SimLingo-BASE 是同一 Base 方法的命名沿革，不能把两个官方数字当两个独立模型横比。其 6.87 属 SENSORS；SimLingo 论文还列同一 Base 的 MAP 6.25。若未来获得 2024 提交包或对应历史 commit，应复查早停开关和 agent 代码，再决定是否重编码。
