# 预登记：RECOVERY-REWIND on every onpol-train turn（不训练，只跑 rollout），2026-10-16

**为什么。** 决策 316 在 nuPlan @18 训练池的 64 个严格转弯上读到：规则专家从策略自己失败前 Δ 2 s 的状态接手，A 层完成 33 / 41，
完成且过 d293 闸的只有 15–23 个弯、122–230 行（2 Hz）。这对恢复课太少。下一轮训练的恢复臂（策略自己的失败状态 + 老师标签，DAgger 式）
需要同一份材料覆盖整个 onpol-train 里策略会到达的每一个失败转弯。本读数只产出这份材料（单元、交接 session、标签行清单）并报产量；不训练，
不下训练判词。写于任何非 smoke rollout 之前；只看过 d316 的结果文件与 lon_inventory。

定义与 d316 预登记（`2026-10-16-rewind-train64-prereg.md`，含补记 1 / 2）相同，下面只写不同之处和本读数自己的量。

## 范围

- 全部 onpol-train 场景：RSG-L18-s0 训练 rollout（`$DATA_DIR/runs/onpol_long/roll/RSG-L18-s0/c0..c5`）的 1 021 个场景，一个不排除。
- d316 读过的 64 个**重跑**（同一代码、同一 stage 结构，读数一致；不复用 d316 的 session）；d316 的 A 单元有多少在这里仍是 event / off 单列。
- **转弯窗口（唯一的定义改动）**：rf_follow 的转弯要求 exit + 10 m 落在日志于第 K_END 个决策时走到的路段内。d316 用 K_END = 17（18 决策读数），
  所以 1 021 个场景里只有 64 个是严格转弯（`entry_speed/follow/scenes.csv`：957 个 turn = False）；用 17 重跑全部场景只会再找到同样的 64 个。
  本读数的 session 是 30 个决策，所以取 rf_follow 的长读数模式 `RF_K_END=29`（按场景日志长度截断，rf_follow.kend），即「15 s 内日志走过的严格转弯」。
  其余 turn / entry / exit / completed / outcome 规则不变。

## 服务与 session

- 同 d316 补记 2：held read 的服务（sh30、plain RSG-L18-s0、c0b_chain OVERRIDES），30 个决策（after280）。trajdata 缓存用同一做法
  （`alpasim_long/build_cache.py --after 280`）给 1 021 个场景新建 `$DATA_DIR/datasets/alpasim_nuplan_long/after280_op`（d316 的 after280_rw 不动）。
- 打包同前：`--vram 20 --cpu 6`，history key alpasim-read4，每卡 4 个；GPU 池，优先级 6。

## 单元（取本读数**新跑的 30 决策 identity**，不取存档 18 决策 rollout）

- A：event（k_ref = k_z）、off（k_ref = 第一个站点 ≥ entry − 20 m、|横向| > 2.5 m、k < k_z 的决策；没有时 k_cross；再没有时 29）。
- short：列出，不倒回（d316：short 是慢，不是转弯错误，策略自己 15 s 内多半完成）。completed（B）：只计数，不交接。
- h = k_ref − 2Δ，Δ ∈ {2, 3} s（d316 的候选），要求 1 ≤ h ≤ 29；不满足的 (单元, Δ) 列为「不可倒回」，另报。
- 专家 XP-L12-E46（rule_expert 默认）。**专家包从本 stage 的 identity run 的 generated config 打**（after280 缓存，日志到 15 s）：d316 的包来自
  after160 配置，日志只到约 8 s，之后是 150 m 直线延长，转弯若在第 8 s 之后，旧包会把它画成直路。
- 交接：`JEV_EXPERT_FROM`（`<scene> <h>` 文件），同一 session，交接时位姿、速度、控制器状态、历史和路线记忆都是策略自己的。D launch 只跑有 h 的场景。

## 检查

- identity 容差（d316 补记 2）：h 前 served plan 最大差 ≤ 0.15 m、控制器轨迹 ≤ 5 cm、|v(h) − v_id(h)| ≤ 0.05 m/s，对**每个**单元算（对照 = 同 stage 的 id）；
  smoke 加一个 idB 看 run-to-run 散布。超出容差的单元照常读、单列；超出比例 > 20 % 时停下报告。
- 交接是否真的在 h：D session 第一个专家决策 = h，且 h 前没有零分；否则该 (单元, Δ) 记为无效，不进分母。

## 量（每个 Δ）

- 专家完成率（rf_follow completed），日志聚类 bootstrap 95 % CI；completed / event / off / short 四列。
- h 之后的零分（rf_follow k_z：at-fault 前 / 侧碰、offroad、dist_to_gt ≥ 4 m）与其中 at-fault，追尾（不是零分）单列。
- 蠕变：v(h + 3 s) ≥ 2 m/s（剩余 < 3 s 的不计，报分母）。
- d293 形状闸（只在 h 之后的轨迹段，入口在 h 前已过时入口速度项记为过）两套门限：
  (1) PAI 门限（`rf_rewind.n_gate`：kmax 0.184、dk 0.0425、alat 4.42、入口 5.6 m/s）；
  (2) **nuPlan 重标定**：本读数全部严格转弯的人类日志轨迹（同一 route、同一闸函数）上 kmax / dk / alat / 入口速度的 p95。人类过两套闸的比例都报。
- **标签资格**：交接有效、rf_follow completed、且 h 之后整个 session 没有任何零分（at-fault 或其他）。有零分的续段整段排除。标签是专家在当时回放车辆下
  算出的 served plan（专家读 3 s 内的回放框，模拟器里的车也是日志回放）。
- **行产量**：资格单元里 k ≥ h、k < k_z（零分前；资格单元没有零分）、站点 ∈ [entry − 20, exit + 10] m（rf_follow junction 段）的决策数，2 Hz。
  三个嵌套类：完成 + 闸（PAI）/ 完成 + 闸（nuPlan）/ 完成且无零分、不看闸。每类报单元数、行数、不同转弯数、不同日志数、行数最多的 3 个转弯所占份额；
  Δ 2 与 Δ 3 分开报，再报两者合计（同一转弯两个 Δ 是两个不同的交接状态，行不重复）。
- 没有登记线，不下训练判词：这是材料清点。

## 规模、成本与分阶段

- 参考：d316 一个 job 13–14 个场景 30 决策 258 s（含约 45 s 启动），约 16 s / rollout；每卡 4 个 job。
- identity 1 021 个 ≈ 4.5 job-h；D2 + D3 预计 A 单元 100–250 个 × 2 ≈ 1–2 job-h；合计约 6 job-h ≈ 1.5–2 card-h（4 job / 卡）。CPU：缓存、打包、
  tfail、read 各 < 15 min 墙钟。超过 8 card-h 时在全量 stage 之前报告。
- 分阶段（docs/long-runs.md）：smoke 2 个场景（4109b79d event、19488eb3 off，加 idB）→ 检查交接与 tfail → 50 个随机场景（seed 0，从其余场景抽）→
  检查容差、完成率、产量是否合理 → 其余 969 个 → 三个 stage 合并读数。每个 stage：id（chunk）→ tfail + pack（CPU）→ D2 / D3（chunk）→ read（CPU），
  一次排队、`--after` 串起来。

## 交付

- 代码：`experiments/route_follow/scripts/rf_rewind.sh train64 {scenes-all, cache, ostage, oread}`（train64 的 nsim 复用；缓存 / 包 / owner 走环境变量），
  `rf_rewind.py n-tfail --kend --deltas --rewind`、`n-read`（多个 tfail）、`n-yield`。
- 结果：`experiments/route_follow/results/rewind_onpol_all/`（rewind_onpol_all.md、逐单元 CSV、units.csv、read.json / summary.json、from_D 文件）。
- 盒：`$DATA_DIR/runs/route_follow/rewind_onpol_all/`（stage 目录、包、`rows.csv` = 训练器要的行清单：scene、k、Δ、h、闸标志、专家 session 的 run 目录 + rollout id）。
- research/next-round/turn-night.md 一条、README 一段、决策文件 + decisions.md 一行。
