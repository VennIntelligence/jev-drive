# 榜单 hack 审计：只读代码和论文，找出分数里不属于驾驶能力的部分

状态: done（2026-09-24 18:12–19:25，Codex gpt-6-sol ultra，约 80 万 token）
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

### 组织：先备料，再并行审计，最后汇总

这个任务很大（约 30 个 repo 加论文），要分给多个子代理并行完成。所有代理在同一个共享目录 `/data/hack_audit/` 里协作，
靠文件互相配合，不靠对话传递信息。

**阶段 0：备料（主代理自己做，串行）**
1. 检查网络：GitHub、arXiv、OpenReview、各 leaderboard 页面能否访问（2026-09-24 实测 Tokyo 上 GitHub、arXiv、OpenReview 都能直连）。
2. 按下面的「抽样」规则写出 `out/sampling.csv`。
3. clone 全部选中的 repo 到 `repos/<board>__<method>/`，下载论文 PDF 到 `papers/<method>.pdf`，记入 `out/repos.csv`。
   **备料全部完成、每个 repo 和论文都确认到位之后，才进入阶段 1。** 下载失败的条目在 `repos.csv` 里标出，不阻塞其他条目。
4. 写好 `out/INDEX.md`：一张表，每行一个审计单元（repo × 榜单），列出负责的代理、状态（todo / in-progress / done / verified）、
   审计页链接。它是所有代理共用的任务板。

**阶段 1：并行审计（多个子代理）**
- 每个子代理领一批审计单元（建议按榜单分，一个代理负责一个榜单的 3–5 个 repo，这样同榜单的评测代码只需要读懂一次）。
- 子代理只读代码和论文，**不运行任何东西**。每个审计单元写一页 `out/audits/<board>__<method>.md`，
  同时向 `out/findings.jsonl` 追加发现（每行一条 JSON；多个代理同时追加时用文件锁，或者先写各自的 `out/findings/<agent>.jsonl`，最后合并）。
- 分类体系采用 open coding：`out/taxonomy.md` 是共享的 codebook。子代理遇到现有类别装不下的发现时，
  在 `taxonomy.md` 末尾的「提议」区追加新类别（定义 + 例子 + 提议者），不要直接改已有定义；已有类别照常使用。
- 子代理开工时把 `INDEX.md` 里自己的单元标成 in-progress，写完标成 done。

**阶段 2：统一 codebook 并重新编码（主代理或一个专门的代理）**
- 阶段 1 全部 done 后，合并、拆分和重命名 `taxonomy.md` 里的类别，得到最终版（变更日志记在末尾）。
- 用最终版 codebook 把全部审计单元重新过一遍，确保早期的审计没有因为当时类别还没建出来而漏记，然后填出 `out/matrix.csv`。
  这一步同样可以按榜单并行。

**阶段 3：独立复核（全新的子代理，不继承之前的上下文）**
- 只给它 `taxonomy.md`、repo、论文和被抽中的条目，复核随机 20% 的 yes 和 10% 的 no，见「验收」第 5 条。

**阶段 4：汇总（一个汇总代理）**
- 读全部审计页、`findings.jsonl`、`matrix.csv` 和复核结果，写 `out/report.md`，并跑「验收」里的自动检查。

### 读什么

读代码的重点（建议，不是规定）：agent 或推理入口、输出后处理、控制器、评测配置和脚本、训练数据的 split 定义、
config 里的 `if`/查表分支、论文里没提到的超参数。

对照论文：对每个发现检查论文（正文、附录、README）是否提到它，引用原文位置（section / table / 附录编号）。
同时核对论文报告分数时的配置（seed 数、是否 ensemble、评测子集）和代码里默认或 README 推荐的评测配置是否一致，不一致本身就是一个发现。

## 交付物

全部写在 Tokyo box 的 `/data/hack_audit/out/` 下：

| 文件 | 内容 |
|---|---|
| `INDEX.md` | 任务板：每个审计单元（repo × 榜单）的负责代理、状态、审计页链接；最后也作为交付物的目录 |
| `audits/<board>__<method>.md` | 每个审计单元一页：读了哪些文件、没读哪些、整体印象、发现列表（链到 `findings.jsonl` 的 id） |
| `sampling.csv` | 每行一个榜单条目：board, rank, method, venue/年份, score, 排行来源 URL, 快照日期, repo URL, 公开程度 A/B/C, 是否选中, 未选原因 |
| `repos.csv` | 每个选中的 repo：repo URL, commit hash, clone 日期, 论文 URL, 对应的榜单 |
| `taxonomy.md` | 最终分类体系：每类的定义、判定标准、正例和一个容易混淆的反例；末尾是变更日志 |
| `findings.jsonl` | 每行一个发现，字段见下 |
| `matrix.csv` | repo × 最终类别，取值 yes / no / NA（第二遍的结果） |
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
5. **独立复核**：由一个不继承之前上下文的全新子代理来做，只给它 `taxonomy.md` 和被抽中的条目，
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

模型用 `gpt-6-sol`，reasoning effort 设为 `ultra`（它会自动把任务分给子代理）。
在 Tokyo box 的 tmux session `hack_audit` 里用非交互模式运行，日志写到 `/data/hack_audit/codex.log`：

```bash
ssh ujs@100.108.238.8
cd ~/mycode/jev-drive && git pull
mkdir -p /data/hack_audit/{repos,papers,out}
tmux new -s hack_audit
cd /data/hack_audit && codex exec -m gpt-6-sol -c model_reasoning_effort='"ultra"' \
  -s workspace-write -c sandbox_workspace_write.network_access=true --skip-git-repo-check \
  "读 ~/mycode/jev-drive/todos/2026-09-24-hack-audit/README.md，按其中的目标、约束、组织方式和验收完成审计。" \
  2>&1 | tee -a codex.log
```

## 结果

全部交付物已拉回 [research/results/hack-audit/](../../research/results/hack-audit/)，总报告是 [report.md](../../research/results/hack-audit/report.md)。

7 个榜单共 36 个抽样条目，入选 20 个审计单元（A 档 16、B 档 4）。CARLA Leaderboard 2.0 只有 TF++ 一个符合代码公开条件，按规则如实少报。
最终 codebook 19 类，35 条发现，矩阵 yes 33 / no 190 / NA 157。三轮独立复核中，最终一轮一致率 25/26 = 96.2%，5 项自动验收全过。
没有运行任何模型或评测。

四个机制簇：榜单指标直接进入决策（评分头重排候选、测试时 CEM 搜索、用 RFS 做 RL 奖励），主要在 NAVSIM 和 WOD-E2E；
训练目标、评分权重与划分针对榜单调整；开环里的 ego 状态与未来标签（nuScenes）；闭环接口、手写规则与仿真协议（CARLA 系、HUGSIM）。
报告只能说明分数由哪些机制共同产生，多数机制缺同权重的单项消融，**给不出「分数里多少比例不是驾驶能力」**；
需要实测的 7 组对照列在报告的「候选实测设计」一节，其中 TFv6 的接口和规则拆分与 [R 层测量](../2026-09-24-r-layer-routine.md) 重合。
