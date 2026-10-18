# PAI-STOP-PROBE 预登记（2026-10-18 晚，写在任何探针输出之前）

## 问题

PAI 上「人停处车也停」的课程能不能做，取决于一个前提：**「人马上要停」这件事在策略自己看到的输入里有没有**，
也就是在模型自己的特征上，一个探针能不能读出来，而且读出来的时候还来得及停。用户规则：不靠人看视频，
只看模型特征里有没有线索。d357：PAI 停车类零分 17 个（C 类 15 + F 类 2），车以 3.8–15.7 m/s 开过人的停点；
另外 18 个走廊零分（C 类）是这里的命名子集。

## 数据（全部是已有的，或者从今天 18 场景相机重跑里复建）

| 源 | 状态 | 场景 | 行 | 来源 |
|---|---|--:|--:|---|
| `log` | 人的状态（ego 跟着日志走，模型的 serve 不执行） | 80（train160 内） | ~15.9k | `runs/route_follow/pait/cache_log`（10-11 log 回放 pass） |
| `p1` | RSG-L18-s0 自己的状态（10-13 Tokyo PAI 281 读数，ACC 之前的 serve） | 48（test281 内） | ~9.5k | `runs/route_follow/pait/cache/p1rsg` |
| `c18` | RSG-L18-s0 自己的状态，d330 ACC-GAP serve | 18（d357 C 类） | ~3.5k | `runs/route_follow/pai_stop_cues/read`（今天 20:43 的重跑，有 rollout.asl）→ `onp_rows` rows + cold → `onp_cache build` |

冻结 token 是 Cinque 视觉编码器的输出，跟 checkpoint 无关（RSG / JTN 只训头和 adapter），所以三个源的 token 可以混用。
p1 与 c18 共有 4 个场景，按场景同组。场景是 PAI 的 clip，没有更上层的 log 编号可用：按场景分组（偏差：不是 log 聚类，
PAI 没有可用的 log 映射）。

## 标签（逐行，10 Hz 决策）

- `a_stop` = 沿人的日志路径，从 ego 投影点往前第一个 `path_v < 0.5 m/s` 的弧长（`path` 网格 0..100 m；`path_v` 是该弧长上日志速度的最小值）。
- 主标签 **y5**：`a_stop <= v0 × 5 s`（按当前车速 5 s 内到达人停的地方）；次标签 **y3**：`<= v0 × 3 s`。
- 行的纳入：`usable`、评分器读到的行（不在 `after_event` / `after_end` 之后）、`v0 >= 2 m/s`（在走，停才是个决策）、
  `|e0| <= 3 m`（ego 在人的路径附近，路径标签才对得上）、`path_ok[S0]`。
- 线索分层（只给正例；互斥，按顺序）：
  1. **agent**：任一 actor 框中心（9 个时刻任一）距人路径横向 < 1.8 m，且弧长在 `[a_stop - 2, a_stop + 20]` m；
  2. **line**：一条可用停止线（`stop_s`）在 `[a_stop - 15, a_stop + 5]` m；
  3. **other**：其余。
  「无前车」= line + other：ACC 和 hold 都抓不到的那部分，是本次重点。
- 负例：同样纳入条件下 y = 0 的行（各层 AUC 都是「该层正例 vs 全部负例」）。

## 特征集（同一套行）

| 名 | 内容 | 维度 |
|---|---|--:|
| KIN | `ego_in`（速度 / 位姿历史、加速度、命令） | 20 |
| LEAD | KIN + RSG 自己的 lead 头（3 个 lead × 6 时刻 × x/y/v/a 的均值 + 3 个 lead_prob） | 95 |
| PLAN | KIN + RSG 的 plan 均值（495） | 515 |
| HID-RSG | KIN + RSG-L18-s0 的 `hidden_state`（512，头前特征） | 532 |
| HID-JTN | KIN + JTN-L18-s0 的 `hidden_state` | 532 |
| TOK | KIN + 冻结 token：最新槽 32 × 512 | 16k+20 |

RSG / JTN 的前向在缓存 token + `ego_in` 上跑（pp_train.PModel，rmem.load_pmodel）。**偏差**：路线记忆不喂（行缓存里没有，
RouteParityAdapter 在没有路线特征时跳过路线支路），所以 HID / LEAD / PLAN 是「路线关掉」的版本；TOK 不受影响，是主读数。

## 探针（固定超参，不调）

- 线性：标准化 + L2 logistic（C = 0.1），KIN / LEAD / PLAN / HID 用；TOK 用每 token 线性降到 16 维 → 512 → MLP(256) → 1 的小网络（AdamW lr 1e-3，
  wd 1e-2，dropout 0.2，3 epoch，batch 256，正例重加权），KIN 拼在 MLP 输入上。另报 TOK-lin（32 token 平均 512 维 + KIN 的 logistic）。
- 交叉拟合：所有场景按 sha1(scene) 分 5 折；每行的分数来自没见过它场景的那一折模型。
- AUC：池化跨折分数；95 % CI = 按场景聚类 bootstrap（B = 2000，seed 0；偏差：stats 默认 10 000，AUC 每次重排太贵）。
- 报：y5 / y3 × 源（log、own = p1 + c18、全部）× 分层（全部、agent、line、other、无前车）× 特征集。
- 「图像加了什么」= TOK − KIN、TOK − LEAD 的 AUC 差，场景配对 bootstrap。

## 18 场景命名子集

c18 的行（交叉拟合分数）。阈值 τ = 该折训练负例上 FPR 10 % 的分数。在 ego 越过人停点之前（`a_stop > 0`）的决策上，
「首次触发」= 连续 3 个决策（0.3 s）分数 ≥ τ 的第一个；报此时到停点的距离 d、车速 v、所需舒适停车距离 v² / (2 × 3 m/s²)，
「来得及」= d ≥ v² / 6。按场景列 TOK、LEAD、KIN 三套。

## 判读线（事先定）

- **readable**：own 状态上无前车层（line + other）TOK 的 y5 AUC ≥ 0.75、CI 下界 > 0.65，且 TOK − LEAD 的 CI 下界 > 0；
  并且 18 场景里至少 9 个「来得及」触发。
- **readable only for leads**：agent 层 TOK 或 LEAD ≥ 0.75，但无前车层不过上面那条。
- **not readable**：两层都不过。
- log 源（人的状态）只作参照：人在停之前就减速，KIN 会泄漏答案；课程关心的是 own 状态（车没减速）上是否读得出。

## 预算

GPU：c18 编码 ~1 分钟、两个 checkpoint 在 ~29k 行上前向、TOK 探针 5 折，合计 < 1 卡时；全部走 pool（priority 11，--after 串）。
