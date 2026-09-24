# 榜单 hack 审计：只读代码和论文，找出分数里不属于驾驶能力的部分

状态: draft
执行者: Codex（Tokyo box），本文自包含，不需要读其他上下文也能做
主题: 分数与驾驶能力的分离；审计结果将用于 [research/](../../research/README.md) 下的方向论述

## 目标

从 7 个自动驾驶榜单各抽 3–5 个公开了代码的方法，只读代码、对照论文，找出每个方法的分数里
**不来自可迁移驾驶能力**的部分，记到具体的 `file:line`，并判断论文有没有披露、有没有量化它的影响。
分类体系由审计过程自己长出来，不预设。

为什么做：我们的判断是「榜单分数高」和「会开车」是两件不同的事。一个模型可能分数很高但不会开车，
也可能很会开车但分数一般。我们要先弄清楚高分里有多少来自针对榜单的工程，才知道哪些东西值得从这些模型里提取，
哪些只是配方（可以直接移植，或者不该移植）。

## 什么算 hack

**定义**：一项设计或做法，它提高了榜单分数，但这份提高**不会随模型一起迁移到真实部署的驾驶上**。
原因可能是它依赖仿真器、数据集、计分规则或评测协议的某个特点，或者它在评测时做了部署时做不到或不会做的事。

判断时看三个问题，每个发现都要回答：

1. **部署时还成立吗**：换一个城市、换一套传感器、上真车，这个做法还带来同样的好处吗？（是 / 部分 / 否）
2. **论文披露了吗**：论文、附录或 README 有没有写？（披露 / 部分披露 / 未披露，给出原文出处）
3. **影响有证据吗**：论文 ablation 里有没有量化它对分数的贡献？（给出数字 / 无）

hack 不等于作弊。许多条目是合理的工程选择，只是只对榜单有效。审计的目的是**把它们看见并记下来**，不是给作者定罪。
「部分成立」的灰色地带照样记录，并写清楚理由。

### 样例（只是举例，不是清单，不要照着逐项打勾）

- **输出接口和控制器**：我们实测过 TFv6（LEAD，Bench2Drive DS 约 95）。同一个网络，用「route + target speed 加作者的 PID」
  来开，比用它自己的 waypoint 来开，DS 高约 14 分；而在一个反事实配对测试里，target speed 这一路几乎不对行人和 cut-in 做反应，
  waypoint 那一路反而会反应。高分主要来自接口和控制，不来自对场景的理解。
- **手写规则覆盖模型输出**：TFv6 代码里有 creeping（卡住时缓慢前挪）、stop sign 强制停车之类的启发式。
  这类规则常常专门针对某个榜单的计分项或 scenario 类型。
- **开环榜单的 ego status**：nuScenes 开环规划上，只吃 ego 历史速度和加速度、不看图像的 MLP，L2 可以和视觉方法持平
  （AD-MLP，"Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes"）。
  一个方法在这类榜上的分数有多少来自 ego status，就是需要审计的问题。
- **评测协议**：比如只报最好的 seed、评测时用 ensemble 但论文写的是单模型、改了 timeout 或路线子集。

你很可能会发现样例里没有的类型，那正是我们想要的。

## 抽样

### 榜单（7 个）

Bench2Drive（闭环）、CARLA Leaderboard 2.0（闭环）、NAVSIM v1（PDMS）、NAVSIM v2（EPDMS）、
nuScenes open-loop planning、Waymo Open Dataset E2E（WOD-E2E，RFS）、HUGSIM（闭环，3DGS 仿真）。nuPlan 不做。

### 规则

1. 对每个榜单，找一份**公开排行**：官方 leaderboard 页面、Papers with Code，或最近一篇 survey/论文里的对比表。
   记下来源 URL 和快照日期。
2. 从高分往下，按**代码公开程度**挑 3–5 个：
   - A 档：训练代码、评测代码、agent/推理代码都公开
   - B 档：只有推理或评测代码
   - C 档：没有代码，或者只有空仓库、README（不选，但要记录）
   优先 A 档，其次 B 档。同一个名次附近 A 档优先于 B 档；不要为了凑数去选分数远低于前列的方法，不够 3 个就如实报告。
3. 同一个 repo 出现在多个榜单上时只 clone 一次，但要按榜单分别审计（不同榜单的评测代码往往不同）。
4. 如果某个榜单的 leaderboard 同时有 privileged 的规则 baseline（例如 PDM-Lite、PDM-Closed），把它记入抽样日志作为参照，
   不作为审计对象。

预计 25–35 个 repo。

## 做法

分类体系采用 open coding，边读边建：

1. 读第一个 repo 时自由记录发现，写入 `taxonomy.md`，每类一个简短定义和一个例子。
2. 往后每个 repo，有合适的类就归进去，没有就新建；合并或拆分类别时在 `taxonomy.md` 里记一行变更日志。
3. **全部 repo 读完后，用最终版 `taxonomy.md` 从头再过一遍所有 repo**，确保早读的 repo 没有因为当时类别还没建出来而漏记。
   第二遍后，每个 repo 对每个最终类别都要给出 yes / no / NA。

读代码的重点（建议，不是规定）：agent 或推理入口、输出后处理、控制器、评测配置和脚本、训练数据的 split 定义、
config 里的 `if`/查表分支、论文里没提到的超参数。

对照论文：下载论文（arXiv 或会议版本，以 repo README 里引用的为准），对每个发现检查论文是否提到它，
引用原文位置（section / table / 附录编号）。同时核对论文报告分数的配置（seed 数、是否 ensemble、评测子集）
与代码里默认或 README 推荐的评测配置是否一致，不一致本身就是一个发现。

## 交付物

全部写在 Tokyo box 的 `/data/hack_audit/out/` 下：

| 文件 | 内容 |
|---|---|
| `sampling.csv` | 每行一个榜单条目：board, rank, method, venue/年份, score, 排行来源 URL, 快照日期, repo URL, 公开程度 A/B/C, 是否选中, 未选原因 |
| `repos.csv` | 每个选中的 repo：repo URL, commit hash, clone 日期, 论文 URL, 对应的榜单 |
| `taxonomy.md` | 最终分类体系：每类的定义、判定标准、正例和一个容易混淆的反例；末尾是变更日志 |
| `findings.jsonl` | 每行一个发现，字段见下 |
| `matrix.csv` | repo × 最终类别，取值 yes / no / NA（第二遍的结果） |
| `repo-notes/<repo>.md` | 每个 repo 一页：整体印象，以及读了哪些文件、没读哪些文件 |
| `report.md` | 中文总结：每个榜单最常见的几类、分类体系概览、最值得注意的 5–10 条发现、「候选实测」清单 |

`findings.jsonl` 的字段：

```
id, board, repo, commit, category, title,
location        # 固定到 commit 的 GitHub permalink，精确到行范围
snippet         # 不超过 15 行
description     # 它做了什么，为什么它影响分数
deploy_valid    # yes / partial / no，加一句理由
disclosed       # yes / partial / no
paper_quote     # 论文原文出处和引文；未披露时写明查过哪些部分
impact_evidence # 论文 ablation 的数字，或 "none"
confidence      # high / medium / low
```

## 验收

1. `sampling.csv` 覆盖全部 7 个榜单，每个未选的条目都写了原因。
2. `findings.jsonl` 的每条 `location` 都能打开，并且在该 commit 的那个行范围里确实能找到 `snippet`。
   写一个脚本在本地 clone 上自动检查，把检查输出附在 `report.md` 末尾。
3. `matrix.csv` 没有空格；每个 yes 在 `findings.jsonl` 里至少对应一条记录。
4. 每条发现的 `disclosed` 和 `paper_quote` 都已填写。
5. **独立复核**：另开一个全新的 Codex session，只给它 `taxonomy.md` 和被抽中的条目，
   复核随机 20% 的 yes（发现是否成立、归类是否正确）和 10% 的 no（是否确实没有）。
   一致率 ≥ 90% 才算通过，不一致的条目逐条写进 `report.md`。

## 约束

- **只读代码和论文**：不下载权重，不跑模型，不跑评测。
  如果某个发现的影响只能靠实际运行来确认，写进 `report.md` 的「候选实测」清单，说明要跑什么、预期看到什么，然后停在那里。
  Tokyo box 能跑 Bench2Drive，但运行需要另外批准。
- 工作目录是 `/data/hack_audit/`，repo clone 在 `/data/hack_audit/repos/`（`git clone --depth 1` 之后记下 commit）。
  不要在 `~/mycode/jev-drive` 里写任何东西，不要 commit 或 push。交付物由 Mac 这边拉回并入库。
- Tokyo box 的 GPU 和 CARLA 可能正在被其他实验使用，不要启动或关闭任何 CARLA 或 GPU 进程。
- 网络：GitHub 可以访问。论文或其他站点下载失败时，在 `repos.csv` 里记下来，接着做别的，不要卡住。
- 大文件（数据集、权重、LFS 对象）一律不拉：用 `GIT_LFS_SKIP_SMUDGE=1`。
- 全部说明文字用中文，代码、字段名、category 名称用英文。

## 预算与汇报

- 预计每个 repo 30–60 分钟（含对照论文），总计 1–2 天。
- 每完成一个榜单，在 `out/progress.md` 里追加一行：已审 repo 数、新增类别、耗时。
- 遇到需要决定的问题（例如某榜单找不到任何 A/B 档代码、需要跑代码才能判断），写进 `out/questions.md` 继续做别的，不要停下来等。

## 启动

```bash
ssh ujs@100.108.238.8
cd ~/mycode/jev-drive && git pull
mkdir -p /data/hack_audit/{repos,out}
tmux new -s hack_audit
cd /data/hack_audit && codex
# 在 codex 中输入：
#   读 ~/mycode/jev-drive/todos/2026-09-24-hack-audit/README.md，按其中的目标、约束和验收完成审计。
```

## 结果

跑完再填：交付物拉回 Mac 后放在 `research/results/hack-audit/`，结论写进 research 下的笔记。
