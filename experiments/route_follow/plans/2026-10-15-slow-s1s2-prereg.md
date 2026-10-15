# 预登记：SLOW S1 / S2（慢的训练修法：只在空路上打断速度历史回路 + 补 PAI 高速直路行），2026-10-15

背景：[决定 315](../../../research/decisions/315.md)、[research/lon-diag.md](../../../research/lon-diag.md)：慢有两段。(1) 交接即短：67 个起点里 45 个在 k ≤ 2，
这些场景日志车速中位 21 m/s，训练 store 行速 p99 12.8、最高 17.1 m/s，≥ 16 只有 8 行；(2) 回路偏移：自由决策上 plan 对喂入速度 g 0.936、3–19 m/s 常数 −0.3 到 −0.5 m/s，
不动点 3 m/s。已试过的修法都把分换成零分：serve hold（前撞 39）、holdlead（held GT +0.10）、`--ego-drop-speed 0.5` 全行（公开零分 +8.5、GT +0.056，allrows）、
`--on-mode both`（零分 +7、GT +0.64，rsgboth_v）；多出的零分 75 % 发生在比人快的时候（弯里跑宽、人停着往前蹭、追尾）。用户 10-15 晚批准：跑 S1 + S2。
本文件在任何训练（60 步 smoke 除外）之前写好、提交、推送；修改只追加补记。

## 臂（底座 RSG-L18 配方，`--init RB0-F-s0`，2500 步，`--lr-scale 0.1`，`--on on1_L18RB0 --on-k 13 --on-mode self --on-split alpasim_nuplan/onpol-train`，
`--yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --warmup 1 --eval-every 500`，与 rf_onk.sh 的 REC 逐项相同）

| Tag | 改动 | 问题 |
|:--|:--|:--|
| `RSGS1-L18-s{0,1}` | `--ego-drop-speed 0.5 --ego-drop-open`：速度历史只在**空路行**上以 0.5 的份额置零，其余行历史照旧 | 只在没有东西的路上打断回路，是否不再付 allrows 的横向零分 |
| `RSGS2-L18-s{0,1}` | S1 + PAI train160 人类**高速直路**行（池 `on1_PAITV`，模仿人的速度剖面；nuPlan 行仍是 S-only） | 补上训练里没有的 ≥ 12 m/s，交接即短是否消失 |

**空路行**（`--ego-drop-open`，训练时逐行判定，serve 不变）：
- 池行（on-policy）：on_lon 的 `p_open_design`（60 m 内无 A-eligible 前车、无任何停止线、无 ≥ 45° 转弯峰）。
- 离线 navtrain 行：按 rf_lonol.py 的 free 格子口径（40 m 内无 A-eligible 前车、60 m 内无红绿灯线、60 m 内路线转角 < 30°），按 token 与 LONG-PROBE `lc_ctx` 连接；连不上标签的行**不丢**。
- 报每个 batch 实际置零的份额（预期 ≈ 0.5 × 空路行占比）与两类行各自的占比；`--ego-drop-open` 关掉时与 `--ego-drop-speed 0.5` 逐位相同（allrows 的实现不动）。

**`on1_PAITV`**：从 `on1_PAIT`（PAI train160 沿人类日志滚出的行，带帧与 `imit` = 日志路径按日志自己的时序，turning-school prereg 的定义）取：行的 t0 车速 ≥ 12 m/s、
走廊 40 m 内无同向车（asl 框）、路线 60 m 内转角 < 30°、不在考题 72（`hard_exam.txt`，按构造 train160 本来就不在）。目标 = `imit`（模仿，`both` 的模仿项），agent hinge 照 PAIT 的规则。
nuPlan 池行保持 `self`（hinge only）：需要按池分模式；实现成 `--on-imit-pool <tag>` 一类的最小改动，关掉时与 S1 逐位相同。报 PAITV 的行数、场景数、车速分布（p50 / p90 / max）；
行数 < 300 则 S2 不跑、记「数据不够」。每 batch 的 PAITV 行数按 `--on-weight pool=...` 调到约 10 % 的池行。

## 闸（训练侧，先 pilot）

Pilot = 两臂 seed 0 各 1000 步（其余同配方）。过闸才跑 2500 步 × 2 seed：
- dev ADE（memory on）≤ RSG-L18 的 0.528 + 0.01（allrows 的 drop 在 60 步把 ADE 打到 0.671、2500 步回到 0.538；1000 步 ≤ 0.538 即过）；
- G-rely 读为 true；
- S1：置零份额在登记预期的 ±20 % 内；S2：PAITV 行的 imit 损失在 1000 步内下降（起点的 0.8 倍以下）。
任一臂两条不过就停该臂；两臂都停就只写决定。

## 读数与线（全部先登记；参照 = RSG-L18 的存档读：held `runs/route_mem/read/RSG-L18-s{0,1}`、公开护栏 `runs/route_follow/guard/read10`、PAI Tokyo `v2pai/RSG-L18-s0-gate`）

1. **held 460 @18**（`rm_chain.sh submit-read`，两 seed，seed 均值对 RSG-L18 seed 均值，452 配对，log 聚类 bootstrap B 10 000）：
   零分 ≤ RSG + 3（点）；GT ≤ RSG + 0.03 m（点，CI 一并报）；被追尾 ≤ 0.7 × RSG；均分 ≥ RSG（CI 报）。
2. **公开 1491 @10 护栏**（rf_allrows.sh 的 ol_chain readlist + ol_confirm 法，两 seed）：均分差 CI 下界 > −0.01；零分 ≤ RSG + 5（点）；plan 2 s 短量（rf_deficit）CI 下界 > 0。
3. **PAI 281**（Tokyo，gate serve = 现默认，gains 20，`JEV_VCONT=1.0 JEV_LEAD=1`；只 seed 0，Tokyo 两卡在跑别的、排队后跑；对 `v2pai/RSG-L18-s0-gate` 逐场景配对）：
   73 个 `low_slow_free` 场景的位置对齐速度比中位 0.59 → **≥ 0.80**；交接处（k 0–2）≥ 16 m/s 箱的 plan / 喂入 0.867 → ≥ 0.95（S2 的专门线，S1 一并报）；
   前撞零分 ≤ 参照 + 2；全 281 零分 ≤ 参照 + 3；考题 72 均分差 CI 下界 > −0.02；被追尾 ≤ 0.7 × 参照；`cor_route_end` 不多于参照 + 2（变快会多闯人停的点，这是代价线）。
4. 判定：一臂三组线全过 → 采用为新的底座候选（替换 RSG-L18 要再过 d286 的 L 线，另登记）；过 1、2 不过 3 → 「nuPlan 上安全但 PAI 不动」，记下、不采用；
   不过 1 或 2 → 与 allrows 同一结局，慢的训练侧修法关闭，剩 serve 侧。S2 对 S1：若只有 S2 过交接线，「交接即短 = 数据缺口」成立。

## 代价（估）

pilot 2 × 0.4 卡时；训练 4 × 0.8 卡时；held 4 × 1.2 卡时；护栏 4 × 2 卡时；confirm / deficit CPU；PAI 2 × 2 h Tokyo。盒上约 16 卡时，三张卡一天内；Tokyo 排在 expert 读之后。
池 job 全部 `--after` 串起来，一次提交；主会话按 3–5 h 看一次。

## 补记 1（主会话，2026-10-15 晚，任何训练之前；用户：盒上的卡零碎地空着，把活派满）

臂扩成 2 × 2 加一个对照，每臂 2 seed，pilot 也直接两 seed（闸按 seed 0 判，seed 1 一并报）：

| Tag | 速度历史 | PAITV 行 | 作用 |
|:--|:--|:--|:--|
| RSG-L18（存档） | 照旧 | 无 | 底座 |
| `RSGS1-L18-s{0,1}` | 只空路行置零 0.5 | 无 | 主臂 |
| `RSGSall-L18-s{0,1}` | **全行**置零 0.5（allrows 的 RSGallV 配方，但池换成 RSG-L18 自己的 `on1_L18RB0`） | 无 | S1 的对照：RSGallV 的 +8.5 零分有一半可能是 all-1491 池的（RSGall 自己就 +3），同池才分得开「全行」与「只空路」 |
| `RSGS3-L18-s{0,1}` | 照旧 | 有 | S2 的 drop-one：交接即短若只靠数据能修，S3 就该过交接线 |
| `RSGS2-L18-s{0,1}` | 只空路行置零 0.5 | 有 | 两者合一 |

线不变（第 1–3 组对 RSG-L18）；另登记两条交互读法：S1 − Sall 的公开零分差（CI）回答「横向代价是不是来自在该读历史的行上丢了历史」；S2 − S3 与 S1 − RSG 的速度比差回答两段是否可加。
PAI 281 的 Tokyo 读按 seed 0、顺序 S1、S2、S3、Sall（Tokyo 排在 expert 读之后，能读几个读几个，先读的先判）。
代价改为：训练 8 × 0.8、held 8 × 1.2、护栏 8 × 2 ≈ 32 卡时，三张卡约 11 h；全部 `--after` 串起来一次提交，不等 pilot 过闸就先把 seed 1 的 pilot 和其余臂的训练排进队列（pilot 不过的臂由主会话取消它后面的 job）。
