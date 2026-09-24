# 榜单文本分析（第二轮）：正面归因、跨榜一致性、复现差距、指标攻击面、无代码方法

状态: done（2026-09-24 21:46–22:52，Codex gpt-6-sol ultra，约 54 万 token）
执行者: Codex（Tokyo box，gpt-6-sol ultra），本文自包含
前置: [第一轮 hack 审计](../2026-09-24-hack-audit/README.md)，交付物在 Tokyo 的 `/data/hack_audit/out/`（Mac 上的副本在 `research/results/hack-audit/`）

## 目标

第一轮回答了「高分里有哪些不属于驾驶能力的机制」。这一轮收集**另一半材料**：高分由什么成分带来，
这些成分在不同榜单之间是否一致，社区复现时暴露了什么，榜单指标本身奖励什么、容许什么，以及没有公开代码的高分方法在论文里自己披露了什么。

这一轮**只收集和整理材料，不下结论**。每条记录都要可核验（论文 table/figure 编号加页码、`file:line` permalink、issue 链接）。
凡是需要深度判读的问题（哪个成分最重要、某个差距说明什么、某个指标该不该信），不要自己回答，写进 `questions_for_synthesis.md`，
留给后面的综合阶段由另一位专家处理。

## 范围

**沿用第一轮的抽样**：`/data/hack_audit/out/sampling.csv` 里的 36 个条目，包括 20 个入选单元和 14 个 C 档排除项（W5 专门处理 C 档）。
论文和 repo 已经在 `/data/hack_audit/papers/` 和 `/data/hack_audit/repos/`，不要重复下载。

**默认不扩大范围。** 如果某项工作确实离不开样本外材料，可以扩大，但要在 `out2/expansion_log.md` 里逐条写明：扩了什么、为什么没有它就做不下去。
这些情况不算扩大：同一篇在样本内的论文里对比表中其他方法的数字（W2 会用到），以及七个榜单自己的官方论文和评测代码（W4 会用到）。

## 五项工作（W1–W5 互相独立，可以并行）

### W1 增益归因账本（正面材料）

对样本内每一篇有论文的方法（含 C 档），把论文里的每一个 ablation 和对照实验抽成一条记录：
**哪个成分，相对什么基线，在哪个榜单、哪个划分、哪个指标上，带来多少变化。**

字段：`method, board, split, metric, component, baseline_config, variant_config, baseline_score, variant_score, delta,
same_weights_or_retrained, seeds_or_runs, n_samples, source（论文 table/figure 编号 + 页码）, notes`。

成分的分类采用 open coding（codebook 放在 `out2/w1_component_codebook.md`），例如数据规模或数据来源、训练目标、RL 或偏好优化、
DAgger 与 expert、backbone 或预训练、输出表示（waypoint / target speed / 轨迹词表）、传感器配置、辅助任务、测试时计算、后处理和规则。以上只是例子，不是清单。
如果某条记录对应第一轮的一条 hack 发现，在 `hack_finding_id` 字段里填上它的 id。

### W2 跨榜一致性（已发表数字）

收集同一方法在不止一个榜单或协议上的已发表分数，重点是「开环 + 闭环」都报了分的方法。
来源限于样本内论文的正文和对比表，以及七个榜单的官方论文（例如 Bench2Drive 和 NAVSIM 的论文里有开环与闭环指标的对照）。

交付一张长表：`method, board, protocol/split, metric, score, source, 是否作者自报, 是否与其他行同一 checkpoint`。
可以做**机械计算**：对每一对榜单，取两边都有分数的方法，算 Spearman 相关，报告 n 和方法列表；n < 5 的只列数据、不算相关。
如果榜单官方论文里本身就有开环与闭环相关性的分析，照原文转录，并注明出处。解读留给综合阶段。

### W3 GitHub issue 挖掘（复现差距）

对样本内每个有 repo 的方法（第一轮已 clone 的 repo，加上 C 档里有 repo 地址的），用 `gh`（已登录）拉取全部 issue 和 discussion，包括已关闭的。
挑出以下几类：
- 复现不出论文分数，或复现出的分数不同；
- 询问评测配置、seed、checkpoint、ensemble、没有写进论文的超参数；
- 作者承认或解释了论文里没写的东西；
- 指出评测协议或数据划分有问题。

每条记录：`repo, issue_url, 日期, 类别, 用户报告的数字, 论文数字, 作者是否回复及回复要点（引原文）, 是否解决`。
不要把用户的猜测当成事实，只记录 issue 里明确说了什么。

### W4 榜单侧审计（指标攻击面）

读七个榜单的**官方计分代码**（Bench2Drive 的 leaderboard 与 scenario_runner 的计分部分、CARLA Leaderboard 2.0、NAVSIM v1 的 PDMS、NAVSIM v2 的 EPDMS、
nuScenes 开环规划常用的 L2 与碰撞率实现（注明是哪家的实现）、WOD-E2E 的 RFS、HUGSIM 的 HD-Score）。
官方代码不在本地的，clone 下来放到 `/data/hack_audit/repos/benchmarks/`，并记下 commit。

每个榜单交付一页 `out2/w4/<board>.md`：
1. 指标公式，逐项对应到 `file:line`；
2. **攻击面**：在不改善驾驶的情况下，哪些做法能提高这个分数（例如乘积结构使提前停车有利），每一条都要从代码里给出依据；
3. **盲区**：这个指标看不到哪些驾驶行为；
4. 对照第一轮：每个攻击面是否在第一轮的发现里实际出现过（填 finding id，没出现过就写「未观察到」）。

### W5 C 档方法的纯论文审计

对 `sampling.csv` 里 14 个 C 档排除项，只读论文（正文、附录、项目页），记录它**自己披露的**技巧和对分数有影响的设计，
编码用第一轮的最终 codebook（`/data/hack_audit/out/taxonomy.md`），装不下的在 `out2/w5_codebook_additions.md` 里提议新类。
每条注明「仅论文披露，无代码可核」。同时记录它的代码公开状态（空仓库、只有 README、承诺后续发布但一直没发布等），附日期。

## 组织

与第一轮相同：主代理先做备料检查（确认论文和 repo 都在，W3 的 `gh` 可用，W4 的官方代码拉取成功），然后给 W1–W5 各派子代理并行工作。
共享任务板是 `out2/INDEX.md`，每项工作一行加子条目，写明状态（todo / in-progress / done / verified）。
多个代理同时写同一个文件时用锁，或者各写各的、最后合并。

**独立复核**：每项工作完成后，由一个不继承上下文的全新子代理，按固定 seed 抽查 10% 的记录（每项至少 10 条），
核对数字和出处是否与原文一致。一致率 ≥ 90% 算通过，不一致的逐条改正并记入 `out2/review.md`。

## 交付物（全部在 `/data/hack_audit/out2/`）

| 文件 | 内容 |
|---|---|
| `INDEX.md` | 任务板与交付物目录 |
| `w1_ablation_ledger.csv`、`w1_component_codebook.md` | 增益归因账本和成分 codebook |
| `w2_cross_board.csv`、`w2_correlations.md` | 跨榜分数长表；机械计算的相关，含 n 和方法列表 |
| `w3_issues.csv`、`w3_notes.md` | issue 记录；每个 repo 一段摘要（issue 总数、筛出的条数、最值得注意的几条） |
| `w4/<board>.md`、`w4_attack_surface.csv` | 每个榜单的指标页；攻击面汇总表（board, 攻击面, 代码依据, 是否观察到, finding id） |
| `w5_paper_only.csv`、`w5_codebook_additions.md` | C 档方法的纯论文审计 |
| `questions_for_synthesis.md` | 所有需要深度判读、留给综合阶段的问题，每条写明涉及哪些记录 |
| `expansion_log.md` | 扩大范围的记录（没有就写「无」） |
| `review.md` | 各项工作的独立复核结果 |
| `HANDOFF.md` | 给综合阶段专家的导读：每个文件是什么、怎么读、已知的局限，控制在一页以内 |
| `DONE` | 全部完成且验收通过后最后写入，内容是一行摘要 |

## 验收

1. 每条数字记录都有出处，出处可以定位（table/figure 编号 + 页码，或 permalink / issue URL）。写脚本检查必填字段非空、URL 能访问。
2. W4 的每条攻击面都有 `file:line` 依据。
3. 五项工作的独立复核一致率都 ≥ 90%。
4. `HANDOFF.md` 和 `questions_for_synthesis.md` 都存在且不为空。

## 约束

- **只读文本**：论文、代码、issue、网页。不运行任何模型、仿真器或评测，不下载权重和数据集（`GIT_LFS_SKIP_SMUDGE=1`）。
- 所有产出写在 `/data/hack_audit/` 下；不要在 `~/mycode/jev-drive` 里写东西，不要 commit 或 push。
- 不要启动或关闭任何 CARLA 或 GPU 进程；Tokyo box 上还有别的项目在跑。
- 说明文字用中文，字段名、类别名、代码用英文。
- 下载失败或拿不到的材料记下来，继续做别的，不要卡住；需要决定的问题写进 `out2/questions.md`，不要停下来等。

## 启动

```bash
# Tokyo box 上，tmux session hack_audit2
cd /data/hack_audit && codex exec -m gpt-6-sol -c model_reasoning_effort='"ultra"' \
  --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
  "读 /home/ujs/mycode/jev-drive/todos/2026-09-24-leaderboard-text-analysis/README.md，按其中的目标、范围、组织和验收完成 W1–W5，产出写在 /data/hack_audit/out2/。" \
  2>&1 | tee -a /data/hack_audit/codex2.log
```

## 结果

交付物已拉回 [research/results/leaderboard-text-analysis/](../../research/results/leaderboard-text-analysis/)，导读是 [HANDOFF.md](../../research/results/leaderboard-text-analysis/HANDOFF.md)。
没有扩大范围。W1 账本 1,121 行，W2 已发表分数 204 个、机械 Spearman 10 组，W3 筛出 issue 113 条，W4 七个榜单的指标页和 24 条攻击面，W5 无代码方法的论文披露 45 条。
五项独立复核最终都 ≥ 90%（W3 和 W5 初次未过，改正后重新复核通过，过程见 review.md）；170 个来源链接验收通过。
45 个待判问题在 `questions_for_synthesis.md`，交给[综合判读](../2026-09-24-leaderboard-synthesis.md)。
