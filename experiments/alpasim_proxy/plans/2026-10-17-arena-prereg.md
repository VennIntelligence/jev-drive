# ARENA 预登记：在全部公开重建上均匀取窗的大闭环竞技场（2026-10-17，任何模型读数之前）

写于任何 arena 模型读数出现之前（select 与 pick 只看窗口与人的状态，不看任何分数；smoke 只验管线）。第一次提交后只追加修正，不改上文。
代码：`scripts/arena.py`（定义写在文件头）、`scripts/arena.sh`；集合 `results/arena/{set.csv.gz, meta.json}`；box `$DATA_DIR/runs/alpasim_proxy/arena/`。

## 1. 问题

1. **主要目的（用户定）：这个竞技场能否把 private 分数已知的 anchor 排对。** 六个：garage G0014（private 0.848）、YR10m10-F-s0（0.800）、
   go-straight starter（0.799）、DD-sample（0.772）、LTF-sample（0.738）、APY10m10-AB-s0（0.640）。公开 1491 对这五个非平凡 anchor 的 Kendall tau 是 −0.6
   （decision 262），YR / APY 排反（0.925 / 0.935），等待集合也没排对（wait_set.md）。
2. 我们的两个模型 RSG-L18-s0 与 NR1F-L18-s0 在一个比 held 460 大一个量级、静止 / 等待窗口按自然比例出现的集合上：零分率、失败类型分布、
   分数分布与对人速度（defect_fix/results/zero_overlap.md 在已有读数上的问题放大版）。NR1FB10、V2、P0 不读（用户定：不重复读，只读这两个 + anchor）。
3. held 与 non-held 分开：L18 的 on-policy 池在 non-held 公开场景上滚过，non-held 窗口对 RSG / NR1F 是训练污染的（同一重建、同一段时间附近）；
   anchor 都不受污染。**RSG 对 NR1F 的判读以 held 部分为主；anchor 排序看全部窗口，held / non-held 分开也报。**

## 2. 集合（定义见 arena.py 文件头，在此固定）

- 来源：1491 个公开场景各自的重建资产（MTGS asset，一个场景一个），窗口 5.5 s = 111 帧（10 个决策），trajdata 窗口移进重建内，
  资产、评分器、serving 与公开读数相同（wait_set.py 的 build 做法，只改 central_tokens_config）。
- 取窗：每个 log 从第一帧起的全局时间网格，步长 step（由第 5 节定）；窗口必须**在重建的相机视频内**：首尾帧都在相机时间戳范围内（±0.1 s），
  且人的 20 Hz 位姿离最近相机帧 ≤ 1.0 m（ego_cov）。同一 log 的两个重建在时间上重叠时同一窗口只取一次（id 相同），split 取保守的 train。
- 排除（与等待集合相同）：本车道前方走廊里或作为前车的、渲染里画不出来的车（inv_path > 0 或前车无模型）；窗口出 log。
  另加读后排除（4.1）：任何一个读数在开车之前就失败的窗口（route-fail：地图路线生成失败，与策略无关，公开集上 12 个场景的同一类）。
  不读人类回放（省一个读数；与策略无关的失败从任一读数的 failure_reason 就能认出）。
- 层（人在窗口内的状态，第一个满足的）：static（最大位移 < 1 m）、launch（v0 < 1 m/s）、braking（v_end < v0 − 2）、turn（航向变化 ≥ 30°）、
  cruise（v0 ≥ 5、|Δv| < 1、航向变化 < 5°）、other。与 defect_fix/df_zero.py 同一规则。

## 3. 读什么

8 个读数，每个窗口每个 tag 一次（读数是确定性的：RSG held 460 读三次 0/460 翻转，zero_overlap.md），按这个顺序提交（池里同优先级按提交顺序起）：
RSG-L18-s0 → NR1F-L18-s0 → YR10m10-F-s0 → APY10m10-AB-s0（ap2 driver）→ DD-sample → LTF-sample → starter（go-straight）→ G0014（garage，最后，夜里来不及就不完整）。
驱动路径同等待集合（RSG / NR1F / YR：sh30 + c0b_chain.OVERRIDES，即公开 10 决策读数路径 + 本集合 cache；APY：ap2；LTF / DD / garage：与它们公开读数相同的原生驱动
run.sh ltf / dd / garage；go-straight：AlpaSim starter kit 原样）。
每块 ≤ 150 个窗口一个池作业（我们的驱动 --vram 20 --cpu 6 --ram 30，CL_HIST_KEY=alpasim-read4，与 df.sh 相同；第三方驱动 --vram 24 --cpu 8 --ram 40，与 wait_set.sh 相同）；每个 shard（按 log 分组、约 1200 窗口）一个 cache 构建作业，
构建后用 2 个公开窗口对照已发布 cache（build_cache.py compare 必须逐位相同）并检查每个窗口的 cache 轨迹（56 步、资产、起始帧）。

## 4. 怎么读（读出脚本 arena.py report，写进 results/arena/）

1. 有效窗口 = 集合 − replay 失败 / 0 分的窗口。某 tag 读完 ≥ 98 % 有效窗口才算完整读数；不完整的标出，不进配对比较以外的结论。
2. 每个模型 × {全部, held, non-held}：均分（按 log 聚类 bootstrap 95 % CI）、零分率、零分类型（AF-front / AF-side / offroad / corridor / 其他）、
   非零低分类型（被追尾、其他接触、不起步、爬行 < 30 %、慢）、分数分布（= 1、(0, 0.5)、[0.5, 0.9)、[0.9, 1)）与 CDF 图。
3. 对 RSG 配对（共同窗口，模型 − RSG）：分数差、零分率差（log 聚类 CI）；任一方 < 1.0 的窗口上的速度差 dv。
4. 分层：每层每模型均分 / 零分率，及对 RSG 的配对分数差 [CI]。
5. 对人速度（用户追加）：每个模型在自己 < 1.0 的窗口上 dv =（自车弧长 − 人弧长）/ 5.5 s 与弧长比（人弧长 > 5 m）的分位数，按零分 / 部分分、按层；直方图。
   注意选择效应：没有接触时分数只因进度不足而 < 1，所以部分分窗口按构造偏慢；配对比较才是比较。
6. anchor（主读数）：竞技场均分对 private 分数的 Kendall tau-b（读完整的 anchor，≥ 3 个才算；全部 / held / non-held 各一个），并列公开 1491 的 tau；
   YR − APY 共同窗口配对差：CI 下界 > 0 = 这对排得像 private；CI 含 0 = 分不开；上界 < 0 = 排反。go-straight 的公开数（0.522）来自 N1 的 300 场景样本。

预设的判读（写在读数之前）：
- 竞技场「排得像 private」= 六个 anchor（或读完整的全部，至少 5 个）的 tau ≥ +0.6 且 YR − APY CI 下界 > 0；tau ≤ 0 = 不像；其间 = 部分。
  这条线在读数之前定；即使排对，anchor 只有 6 个、两个是我们一族，不据此单独选提交。
- NR1F 对 RSG 的 held 配对分数差 CI 下界 > 0 才说「在竞技场上更好」；上界 < 0 说「更差」；其他「分不开」。
  non-held 的同样数字只作描述（污染）。零分类型里哪类增长，以 held 的计数差和配对零分率差为准。
- 没有过 / 不过的 bar：这是测量，结果进 decisions 作为证据，不单独触发训练或提交。

## 5. 规模与预算（读数之前定）

select（1.0 s 网格，box 20261017-192605）：34 209 个候选窗口，排除后 23 188 个（held 6 723）；丢掉的：不可见车 3 054，重叠重建的重复 id 7 945，
自车出相机范围 22。人的状态：static 9 404（41 %），other 4 840，braking 3 615，launch 2 324，turn 1 723，cruise 1 282。静止窗口这么多，
是因为长的重建（最长 136 s）多是人停着的场景；这就是「在重建时间内均匀」的自然比例（N2 在原始 log 上均匀取窗是 30 % 静止）。

吞吐（等待集合同一驱动的实测）：每个窗口每个读数 5.5-6.9 s 运行时（每卡 4 个并发读，渲染为瓶颈；APY 的 ap2 最慢），加每块约 45 s 启动。
今晚约 20 个读槽（我们的驱动每卡 4 个，第三方每卡 3 个），NR1HV 训练占两张卡的一部分。step = 3 s → **7 741 个窗口（held 2 246，136 个 log），55 块，
8 个 tag = 440 个读作业，约 6.2 万窗口读，约 108 卡槽小时**：满槽约 6 h，连同争用预计 02:00-04:00 CST 结束。step = 2 s（11 600 窗口）要约 9-10 h，
赶不上早上且留不出失败重跑的余量，不取。相邻窗口重叠 2.5 s，有效样本小于窗口数；CI 按 log 聚类。
shard：按 log 打包约 1 200 个窗口一个 cache（7 个），每个 16 核构建；每个 shard 用 2 个公开窗口对照已发布 cache。
可行性：smoke（12 个窗口，各层都有；starter、RSG、APY（ap2）、DD 各一个读）只验管线，不看分数做任何选择；通过后一次提交整条链，优先级 9，
两个汇总作业（core = RSG / NR1F / YR / APY 读完时；all = 全部）。

## 6. 不做的

不做重加权（每个重建等权等等只作事后描述）；不读 seed 1；不做长时程；不改任何模型或 serving；不读 NR1FB10 / V2 / P0 / 回放；
窗口与公开场景窗口重叠的部分不单独拆（pub_off_s 记在集合里，可以事后描述）。

## 补充 1（2026-10-18，任何 hold 读数之前）：ARENA-HOLD，serve 端静止保持，在 held 窗口上读

**为什么。** 走廊规则已从评分器源码与存档读数的逐步指标核实（`results/corridor_rule.md`，`scripts/corridor_rule.py`）：评分器的 GT 是本窗口
5.5 s 内人的轨迹（不含窗口之后）；人在窗口末尾仍静止时（最后一个 10 Hz 段 < 1 mm，抖动），走廊退化为以人停车点为心的 4 m 圆盘，
任何一个 0.5 s 指标步上本车离它 ≥ 4 m 即零分（held 静止层 RSG：382 次越过 4 m，379 次走廊零分；< 4 m 的 446 窗 0 次）；人末尾在动时
越过路径末端只算横向偏移，前滚免费。等待集（第 344 条）的 a 变体 146 窗全部以人已在蠕动（末段 8–20 mm / 0.1 s，沿航向）结束，
所以前滚不判零，「等待是小杠杆」的结论来自窗口选择，不是评分器。

**变体**（`JEV_HOLD`，`experiments/alpasim/lib/static_hold.py`；由 `lib/sh30_hold.py` 包住 sh30_driver.py 提供，sh30_driver.py 不改；
关闭时 = sh30_driver.main() 原样，`scripts/hold_check.py` 核对）。锁存：交接调用（会话第一次调用后 ≥ 0.5 s 的那次，同 JEV_ACC 的 nuPlan
交接）上实测车速 < 0.3 m/s 则本会话锁存于该点，否则本会话完全不变。锁存期间把模型计划沿其路径截到弧长 s_lim：
- (a) `full`：s_lim = 0，整窗不动。静止收益的上界，也是起步代价的上界。
- (b) `lead`：s_lim = 0，直到「前车动了」：模型自己的 2 s 弧长 ≥ 2 m 且模型自己的 lead 输出里出现过的前车现在以 > 1 m/s 前进
  （radard 滤波概率 > 0.5），或出现过的前车消失；之后本会话放开。从没见过前车（首车、停车）= 一直保持。只用模型自己的输出，不用日志。
- (c) `leash3`：s_lim = max(0, 3.0 − 本车离锁存点距离)：整个会话离锁存点不超过 3.0 m（走廊 4 m 内留 1 m），之内照模型计划起步。

**读什么。** RSG-L18-s0，三个变体各读一遍 arena 的 2 246 个 held 窗口（`scripts/arena_hold.sh`：按 arena shard 重新分块，每块 ≤ 150，
复用 arena 的 cache，只读；池优先级 10）。先 smoke（12 个 held 窗口：关闭读一次、full 读一次）+ 闸门：关闭读与存档 RSG 读分数最多 1 个窗口不同
（闭环重读不是逐位相同：arena smoke 与链上 RSG 读 120 个决策中 9 个计划不同，最大 0.20 m）；full 在交接车速 < 0.3 m/s 的每个窗口都锁存且
这些窗口行驶 ≤ 0.5 m。闸门过了才放读数作业。

**读出**（`arena_hold.py report` → `$W/hold/report/hold.md`）：只用 held 窗口，与存档 arena RSG-L18-s0 读数在同窗口配对（变体 − RSG），
log 聚类 bootstrap 95 % CI（同 arena.py）；去掉任一读数 route-fail 的窗口。
- **杠杆大小** = 全部 held 窗口上的配对均分差 [CI]。
- **收益 / 代价** = 各层（static、launch、braking、turn、cruise、other）的贡献（该层配对差之和 / 全部 held 窗口数，各层相加 = 杠杆大小），
  加该层均分 / 零分率、新增零分 / 修复零分数。**起步代价** = launch 层贡献；other / braking 层的负贡献也算代价。

**读数前的估计**（`hand.csv.gz` 的离线算术，不是读数：交接车速 = 窗口 0.5 s 时人的车速；held 2 228 个有效窗口，RSG 均分 0.786）：
交接 < 0.3 m/s 的 999 窗（static 854、launch 145，其中人 5.5 s 内走 ≥ 5 m 的 80 窗）。(a) 若 static 全拿 1.0、人走 ≥ 5 m 的窗得 0：
约 +0.138 / held 窗口（收益 +379，launch 代价 −72，launch 层均分 0.929 → 约 0.45）。(c) 按进度截到 3 m 估算约 +0.139（代价约 −42）。
(b) 走廊零分的 387 窗里只有 54 窗开头 20 m 内有记录的前车，所以 (b) 在静止层应接近 (a)，代价比 (a) 小多少取决于 launch 窗里前车是否被
模型看到并起步。实际值会因被追尾（不判零）、静止时被侧碰、交接后蠕行等偏离。

**判读线**（读前定）：配对差 CI 下界 > 0 = 该变体在 held arena 上是正杠杆；上界 < 0 = 负；否则分不开。「不跟人动就不动」是不是主杠杆，
看 (a) 的杠杆大小与 static 层贡献；推荐哪个变体看净值，同时报 launch 代价。阈值（0.3 m/s、3.0 m、1 m/s、2 m）读后不调；
不读 seed 1、不读 NR1F 等其他模型、不在 private / 公开集上读（这些另立）。

## 补充 2（2026-10-18，stop3 读数之前）：leash3 之后剩下的代价，和一个变体 stop3

**诊断**（`arena_hold.py diag` → `$W/hold/diag.md`，只用存档的逐窗结果与 full / lead / leash3 的逐决策 hold 日志；serve 时可得的量 = 每次调用
实测车速、模型计划的 2 s 弧长、checkpoint 自己的 lead 输出）：
- (a) 起步代价 −32.9 分全部来自锁存后人 5.5 s 内走 ≥ 5 m 的 82 个 launch 窗（leash3 0.497，RSG 0.899）。与人不动到窗尾的 807 个静止窗
  分不开：模型计划弧长、lead 速度、lead 概率、lead 距离的 AUC 0.59–0.85；75 条放行规则（lead 概率 / lead 速度 / 计划弧长的合取）里
  最好的一条在样本内、按 RSG 分数记放行窗，也只有 +6.8 分（+0.003），而静止窗里被放行的几乎全是 RSG 零分窗（一放就滚出 4 m）。不做。
- (b) leash3 新增的 11 个 AF-front 全在静止层、3 个 log（一个 log 占 9 窗），都是停着的前车在 2.36–2.88 m（RSG 也撞），模型 lead 头把它读成
  约 21 m（概率 0.63），serve 时没有间距信号；全局缩短 leash 到 2.5 m 样本内 +5.7，由单个 log 撑着。不做。
- (c) 留下的 34 个走廊零分全在未锁存会话（full / lead / leash3 同一批，RSG 自己的）：交接车速 0.32–4.5 m/s，braking 16、other 9、static 6、
  turn 3；多数是人在窗内匀减速停下（中位末尾静止 1.8–3.9 s），本车刹得晚、冲过人停点 4–5 m。交接时（0 s 与 0.5 s 两次调用的实测车速）
  匀减速外推的停车距离 v²/(2|a|) 与人交接后的实际弧长逐窗吻合（例如 0.26 / 0.37 m、1.25 / 1.21 m、2.56 / 2.61 m）。提高锁存阈值到
  0.5 / 0.8 / 1.0 m/s 估计 −1.2 / −6.9 / −11.2 分，不做。

**变体 stop3**（`JEV_HOLD=stop3`，`experiments/alpasim/lib/static_hold.py`，同一个 sh30hold driver）：= leash3，另加停车 leash：交接时未按
0.3 m/s 锁存的会话，若 a = (v_交接 − v_上一次调用) / dt < −0.5 m/s² 且 v / |a| ≤ 3 s（窗尾前至少 2 s 停下），则在交接点锁存，s_lim =
max(0, v²/(2|a|) + 3 − 本车离锁存点距离)：人的匀减速停点再加与 leash3 相同的 3 m。低于 0.3 m/s 锁存的会话与 leash3 逐位相同
（`scripts/hold_check.py` 第 4 项）。只用 driver 自己的实测车速与位置。

**读前估计**（diag 里把 static_hold 模块本身在 leash3 日志的车速上重放；leash3 在未锁存窗口上与 RSG 逐位相同，所以基线用 RSG 分数；
被截的窗：RSG 的非走廊零分保留，人末尾静止且截点超过人停点 ≥ 4 m 判零，否则进度分 min((交接前弧长 + leash)/(0.8 gt), 1)，人走 < 5 m 记 1）：
142 个会话锁存，68 个被截；修复走廊零分 25 / 34，新增零分 0；截点超过人停点 0.44–2.90 m（中位 2.62）。**对 leash3 +18.5 分，
+0.0083 / held 窗**（收益 +24.7，进度代价 −6.3；braking +11.0、static +6.0、other +3.6、turn −1.8、cruise −0.4）。阈值敏感性：
a_stop 0.3–0.5、t_stop 2–4 s、余量 2–3 m 时 +13.5 到 +19.0；a_stop 1.0 时 +7 到 +11。阈值 0.5 m/s²、3 s、3 m 在读之前定下，读后不调。

**读什么**：RSG-L18-s0 + stop3，arena 的 18 个 held 块（与补充 1 同一批列表与 cache），池优先级 13。先 smoke（shard 6 的 12 个 held 窗：
6 个预计被截的停车锁存窗、3 个静止锁存窗、3 个不锁存窗）+ 闸门 gate2：每个会话过交接；每个锁存（静止或停车）与按日志车速重算的规则一致
（停车 leash 误差 ≤ 0.01 m）；锁存会话的 d0 不超过各自 leash + 0.5 m；至少 1 个停车锁存。闸门过了才放读数（`arena_hold.sh chain2`）。

**读出**（`$W/hold/report2/hold.md` → `results/arena_hold/hold2.md`）：held 窗、与 leash3 读数和 RSG 存档读数分别配对，log 聚类 bootstrap 95 % CI，
各层贡献，新增 / 修复零分。**判读线**（读前定）：对 leash3 配对差点估计 ≥ +0.004（估计的一半）且新增零分少于修复零分 → 推荐 stop3 进
提交镜像（替换 leash3，镜像需重建）；否则 leash3 留下。只读这一次，不读其他 seed / 模型 / 阈值。
