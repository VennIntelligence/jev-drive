# 晚转弯 / 分叉：按类别读转弯完成率（nuPlan @18 + PAI）

**日期** 2026-10-13。**对象** RSG-L18-s0、RSGboth-L18-s0、RB0-F-s0，@18 读数。
**脚本** [experiments/route_follow/scripts/rf_lateturn.py](../../experiments/route_follow/scripts/rf_lateturn.py)（`scan` / `pick` / `shape` / `scores` / `report` / `pai` / `page-*`，规则全在 docstring），
链 [rf_lateturn.sh](../../experiments/route_follow/scripts/rf_lateturn.sh)，相机重跑 [rf_lateturn_video.sh](../../experiments/route_follow/scripts/rf_lateturn_video.sh)。
**结果** [experiments/route_follow/results/lateturn/](../../experiments/route_follow/results/lateturn/)：`summary.md`（总表）、`scenes.csv`（1491 个 scene 的类别、入口站位、角度、方向、split）、
`rollouts.csv`（model × scene）、`pai.md` / `pai_classes.csv`。**看片** Tokyo `~/Desktop/late-turns/index.html`（Mac 上 `tmp/lateturn/late-turns/`）。

## 0. 结论

- **nuPlan @18 上几乎没有真正的晚转弯，这是窗口决定的，不是挑法的问题**。18 个决策里日志中位只走 47 m（18 已经是重建能支撑的最长读数），
  入口在片头 40 m 以后、而且人在窗口里至少转过一半的转弯，1473 个 scene 里只有 **4 个**（train 3、held 1）。所以把入口 20–40 m 的 **delayed** 当晚转弯的替身（18 个），
  合起来 **22 个**。另有 86 个 scene 的第一个弯人在窗口里没转过一半（其中 22 个入口 ≥ 40 m），@18 测不了。真正的晚转弯要去 PAI（20 s 片段，36 个里 15 个 ≥ 40 m）。
- **入口 ≥ 20 m 的弯，RSG 和 RB0 掉得很厉害，RSGboth 没掉**：完成率 RSG **18 %**、RB0 **14 %**、RSGboth **68 %**（22 个）。入口 < 20 m 的分别是 52 % / 45 % / 68 %。
  RSGboth - RSG 在晚弯上是 **+50 pp [+30, +76]**，早弯上是 +16 [+8, +24]。
- **但 RSG / RB0 掉的这部分主要是"太慢"，不是"不转"**：22 个里 RSG 有 10 个、RB0 有 7 个是 short（到 decision 17 还没走到目标站位），到入口时比人慢 2.1 m/s（RSGboth 只慢 0.2）。
  只数 event + off（真出事或走错路）：RSG 8/22、RSGboth 5/22、RB0 12/22，早弯是 44/147、35/147、58/147，比例差不多。
  也就是说，@18 能测到的范围内，**看不出"弯在片子后段就转不好"**；晚弯的差距是车速，RSGboth 修好的正是车速。
- **分叉（路线离开本车道的直行后继）不比普通转弯难**：97 个分叉完成率 RSG 57 %、RSGboth 74 %、RB0 48 %；没有分叉的转弯是 42 % / 64 % / 36 %。
  有分叉减没分叉 RSG +15 pp [-1, +29]，没有一个模型是分叉更差。只有分叉、没有 60° 弯的 30 个（Y 形分叉、先变道再分出），RSG 57 %、RSGboth 73 %，零分只有 1 / 3 个。
- **PAI 上晚转弯确实更差**（不跑新 rollout，只按类别重分 rf_pai_follow 的 36 个弯）：RSG 晚弯 **1/15 = 7 %**，早弯 4/21 = 19 %；RBSG2 晚弯 1/15 = 7 %，早弯 8/21 = 38 %。
  晚弯失败多是 event（8 个）和 short（4 个）。PAI 上的晚弯和分叉大量重叠（36 个里 17 个分叉），而且 PAI 本身整体就低，所以这个差距是模型 × 数据集一起造成的，不能只归到"晚"上。
- **对下一步的含义**：想在 nuPlan 闭环里测"自己决定什么时候转"，@18 做不到（窗口太短）。能做的有两条：在 PAI 上测（片子长）；或者在 nuPlan 上把起点往前挪（rollout 从 token 前面开始），
  但这要先确认重建覆盖得到。分叉不需要单独补救，它的失败和普通转弯是同一种。

## 1. 怎么分类

- 路线 = AlpaSim 自己的 RouteGenerator（MAP，40 m），用日志在 18 个决策时刻的位置做锚点生成（rf_trainread 的做法），窗口 = 日志走到 decision 17 的行程 [s0, s_end]。
- **转弯** = rf_follow 的规则（30 m 内净转角 ≥ 60°）。取从 s0 往后第一个入口 ≤ s_end 的弯：exit + 10 m ≤ s_end 是 strict（rf_follow 原口径），
  否则要求人在 decision 17 时已经过了弯中点 5 m（half），目标站位取 s_end - 5 m。入口站位 entry_rel = entry - s0。
  **late** = entry_rel ≥ 40 m，**delayed** = 20–40 m，**early** = < 40 m（主对照用 < 20 m）。
- **分叉** = 从自车车道出发，每到车道末端都选转角最小的后继，得到"直行链"；路线到这条链的距离第一次 ≥ 3 m 的站位是分离点，要求分离点在片头 5 m 以后、s_end - 5 m 以前，
  并且路线真的离开了这条路（最大分离 ≥ 5 m，或 s_end 处航向差 ≥ 15°；只偏一个车道并且平行的算 lane_offset，不计入）。
  类型：branch 74（从链上车道的后继分出去：路口转出、Y 形分叉），lc 11（先换到相邻车道再分出），other 12。一个 scene 可以既是转弯也是分叉。
- **完成** = 在第一次零分事件之前越过目标站位，并且离路线 ≤ 2.5 m；纯分叉的目标是 min(分离点 + 10, s_end - 5)。strict 转弯上这个口径和 rf_follow 的结果逐个一致。

## 2. 读数

- held：已有的 held 460 读数（三个模型都有）。train：trainread 读过的 104 个，加上 52 个新读（所有还没读过的 late / delayed / fork train scene，没到 80 的上限，所以没有抽样），
  三个模型、走 pool 的 1–5 号卡，rm_chain readlist 口径。早弯里 41 个 half 口径的 train 弯没读，所以早弯 n = 165（共 206 个）。
- 跑的过程中出过一次事故：21:11 有人把 pool 里排队的作业全部取消后又重新提交了一遍，我也重新提交了一遍，结果 4 个 chunk 被两个作业同时写。这 4 个 chunk 已删除，干净地重跑过；表里的数都来自干净的读数。

## 3. 总表（完成率 [对数据 log 做 cluster bootstrap 的 95 % CI]（n））

| 类别 | scene（train / held） | RSG-L18-s0 | RSGboth-L18-s0 | RB0-F-s0 |
|:--|:--|:--|:--|:--|
| late（≥ 40 m） | 4（3 / 1） | 0 % | 100 % | 0 % |
| delayed（20–40 m） | 18（15 / 3） | 22 % [0, 39] | 61 % [43, 85] | 17 % [0, 33] |
| late + delayed（≥ 20 m） | 22 | **18 %** [0, 32] | **68 %** [52, 88] | **14 %** [0, 27] |
| early < 20 m | 147（84 / 63） | 52 % [43, 61] | 68 % [59, 75] | 45 % [36, 54] |
| fork（全部） | 97（63 / 34） | 57 % [46, 67] | 74 % [65, 82] | 48 % [38, 59] |
| fork，没有 60° 弯 | 30（16 / 14） | 57 % [38, 78] | 73 % [59, 87] | 50 % [33, 69] |
| 转弯，没有 fork | 102（55 / 47） | 42 % [33, 52] | 64 % [53, 73] | 36 % [27, 46] |

结果拆分（completed / event / off / short）：late + delayed：RSG 4/6/2/10，RSGboth 15/5/0/2，RB0 3/9/3/7；early < 20 m：RSG 77/33/11/26，RSGboth 100/30/5/12，RB0 66/46/12/23。
"太慢"占比：late + delayed 上 RSG 45 %、RSGboth 9 %、RB0 32 %；fork 上 18 % / 3 % / 18 %。内切（只算 strict 转弯）各类都在 50–67 %，三个模型之间没有差别。
转弯形状（相对人类路径，rf_ledger）：late + delayed 上到入口时自车减人 RSG -2.07 m/s、RSGboth -0.20、RB0 -2.05；入口前 20 m 是 -0.51 / -0.06 / -0.50。
早弯入口处的差距只有 -0.14 / +0.03 / -0.13。入口前 40 m 的速度在 @18 窗口里几乎没有（n = 2–3）。配对差、组内对照和每个类别的零分类型见 summary.md。

## 4. PAI（36 个 strict 转弯，按类别重分，不跑新 rollout）

| 类别 | RSG | RBSG2 |
|:--|:--|:--|
| late（≥ 40 m，15 个） | 1/15 = 7 % [0, 20] | 1/15 = 7 % [0, 20] |
| early（21 个） | 4/21 = 19 % [5, 38] | 8/21 = 38 % [19, 57] |
| fork（17 个） | 1/17 = 6 % [0, 18] | 1/17 = 6 % [0, 18] |
| early，没有 fork（14 个） | 3/14 = 21 % | 7/14 = 50 % |

## 5. 看片

`late-turns/`（Tokyo 桌面）放的是 22 个 late / delayed，加上 RSGboth 最差的 38 个分叉（RSGboth 结果由差到好）。每行有四个视频：BEV RSG、BEV RSGboth，以及在 0 号卡上重跑的前视相机 RSG / RSGboth。
列：类别、入口站位、结果 RSG → RSGboth、分数、形状数字。视频 2 倍速自动播放，滚出屏幕就暂停。相机重跑的闭环结果不保证和表里的分数逐位一致。
