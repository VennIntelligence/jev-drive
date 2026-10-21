# HARM-MIX：PAI 人类状态行的 Harmonizer 开渲染孪生行（混合训练数据，先不训练）

写于 2026-10-22 凌晨（CST），在任何渲染之前。代码：`scripts/hmix.sh`、`scripts/hmix.py`、`lib/twin_tokens.py`（仓库根 `lib/`）、
`experiments/route_mem/scripts/rm_train.py --twin`。结果：`results/harm_mix/`。box 运行目录 `$DATA_DIR/runs/round7/hmix/`。

## 为什么

- 决赛复测开 Harmonizer，公开榜和我们全部训练渲染都是关。NTV0 开 / 关不变；用 PAI 人类行 + DAgger 行训练的族（R5LC、R6DAG）开时变差：
  R6DAG-s1 闭环 0.815 → 0.780，损失在纵向（被追尾、起步 35 → 32；第 439 条）；开环开时计划变短（4 s 汇总 −0.10 m，直路 −0.24 m；
  harmonizer_openloop.md 的 R6DAG-s1 一节）。
- 假设：适配器从关渲染的 PAI 行里学到了关渲染特有的线索。用户 10-21 夜选混合训练（同一状态开 / 关两种渲染都喂），不选 serve 端滤波
  （滤波要知道 Harmonizer 开没开，或者会改公开榜的输入）。
- 已有证据：同一个 PAI 人类状态重渲染是逐位确定的。第 435 条关 / 关 300 / 300 帧相同；本 lane 先查了 harm 的关渲染池
  `on1_HARMOFFA` 与 `on1_PAIH`：6 223 行（39 个场景，分块不同）token、ego、标签、cam、teacher 全部逐位相同。所以开渲染的人类行与关行状态
  完全对齐，只差画面。

## 哪些行有孪生行

- **只做 PAI 人类行**（`on1_PAIH` + `PAIH05` + `PAIH15`）。人类行沿人的日志开（`onp_log_driver.py`：每步返回日志，MPC 跟），ego 状态与画面
  无关，开渲染与关渲染的状态完全一样（第 435 条 ego 差 0）。
- **DAgger 行这次不做。** DAgger 状态来自学生闭环：每步的 ego 由学生当时 serve 的计划经 MPC 决定。精确重放需要每步原样的参考轨迹，但
  `drive.jsonl` 里的 `poses` 只存到 1e-4 m / 1e-4 rad（`pai_driver.py` 的 round(4)），rollout.asl 也已删除（dag.sh clean）。
  `rec_driver.py` 是「日志 + 侧偏」的 MPC 跟踪，不是重放。写一个「返回记录 poses」的近似重放驱动只要几十行，但它给出的是近似状态
  （ego 差约亚毫米级、帧不逐位相同），闸 C（关重渲逐位复现）过不了，只能用容差。按任务要求：不存在精确重放 → 只做人类行。若主会话要
  DAgger 行，下一步是这个近似重放驱动加一个 6 场景探针（关重放 token 对存档相对 L2 中位 < 2 %、ego 最大差 < 1 cm 才继续），另行登记。
- 场景抽样（`hmix.py select`，种子 0，渲染前定死）：`scenes.tsv` 中 held False 且在 `on1_PAIH` 里的 717 个场景，按场景类分层随机：
  launch 50 %、rest（直路巡航为主）40 %、bend / stop_nolead / turn_sharp 30 % → **246 个场景**。整场景开渲染，同场景的 PAIH / 05 / 15 行
  一起得到孪生行。held 场景（训练的 `--on-hold-pct 15` 留出）不渲染。
- 覆盖（留出外的行，行分层用 harm.py 的规则作用在人的 4 s 日志未来上；拉伸行继承其 f = 1 底行的层）：

| 池 | 选中行 / 可用行 | 占比 | 起步 | 停车 | 路口转 | 弯道 | 直路巡航 | 其他 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `on1_PAIH` | 39 299 / 114 453 | 34 % | 40 % | 33 % | 27 % | 29 % | 41 % | 34 % |
| `on1_PAIH05` | 35 855 / 104 415 | 34 % | 42 % | 34 % | 28 % | 30 % | 41 % | 34 % |
| `on1_PAIH15` | 28 750 / 83 671 | 34 % | 42 % | 34 % | 27 % | 29 % | 42 % | 34 % |

  （起步 1 566 / 3 899 行、直路 8 540 / 20 994 行在 PAIH 里。）共约 10.4 万孪生行。
- 在 R6DAG 的抽样权重下（`round6/dag/w/R6DAG.npz`，每批 72 个池行），PAI 人类行占池抽样质量 0.444（约 32 行 / 批），其中有孪生行的占
  0.155（约 11 行 / 批）。所以 `--twin PAIHTW:p` 每批约 11 p 行换成开渲染 token（p = 0.5 时约 5.6 / 136 行）。p 由主会话定。

## 管线（`hmix.sh`，全部经池，优先级 24，`--after` 串成一条链）

1. `select`（上面）→ `chunks/c000`（1 个场景）、`c001`（10 个）、其余每块 32 个；`offcheck.txt` = 选中场景里随机 6 个。
2. 开渲染：pr.sh H 的配方原样（`onp_log_driver.py`、A0 serve `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 PAI_DUMP=100000`、
   g10 + `send_recording_ground_truth=true`、NTV0-L18-s0 在环、`ALPASIM_FAST=1`、CONC 4、KEEP_ASL）加 `HARMONIZER=1`，VRAM 30 / RAM 40
   （harm.sh 的开渲染设置）。分阶段：c000 → 闸 B 查渲染日志 → c001 → 其余（同时最多 W = 6 块持有 rollout.asl / dumps）。
   关重渲：offcheck 的 6 个场景一块，配方相同、不加 HARMONIZER。
3. 每块：`onp_rows rows` + `cold`（同 pr.sh）→ clean（删 rollout.asl 与帧 dump，留 drive.jsonl）。
4. 尾：`onp_cache build`（普通缓存路径）→ `rf_pai_pool --fut log_fut --prefix paih --src human` 建 `on1_PAIHON`（名字与 `on1_PAIH` 同为
   `on:paih:<scene>:<k>`）→ `pr.py stretch --f 0.5 / 1.5` 建 `on1_PAIHON05` / `on1_PAIHON15`（前缀自动为 paih05 / paih15）；关重渲同样建
   `on1_PAIHOFFC`（只 f = 1）。→ `hmix.py gates` → 全过才 `hmix.py twin` 建孪生库 `on1_PAIHTW`。
5. 孪生库 `on1_PAIHTW`：`tab.npz`（names、src 池）+ `@warp/front.parts.json`（三个开池 token 文件的零拷贝拼接，不复制 token）。
   训练只从这里取 token，其余（标签、ego、路线特征、teacher、行名、时间缩放）一律用关行自己的，所以孪生行与关行除 token 外按构造相同；
   闸 A 证明这些 token 属于同一个状态。

## 闸（线在渲染前写定）

- **A 对齐**：三对池（PAIH / PAIHON、05、15）按行名配对，开池每一行都在关池里，且关池在这 246 个场景里的行数 = 开池行数（一行不丢）；
  tab（ego、fut、cam、lht、speed、log、src、f、base 等全部键）、agent / road / route 全部键、路线特征 feat 全部键 **逐位相同**。
  `served`（司机在环看画面出的计划）与 teacher out（shipped Cinque 在 token 上的输出）依赖 token，只报告差，不判。
- **B 权重载入**：每个开渲染块的 `renderer-*.log` 含 `"enable_harmonizer": true` 与 `checkpoint ready at ... harmonizer_nontemporal.pt`
  （harmcl.sh check 的规则）；关重渲块不含 enable。
- **C 位姿对齐**：`on1_PAIHOFFC`（6 个场景关重渲）对 `on1_PAIH` 共有行：token、ego、标签、served、teacher、全部 tab / agent / route 键
  逐位相同，行数相同。
- **D token 距离**：每对池开 vs 关最新图像对（slot 7）相对 L2 中位在 [0.22, 0.36]（第 435 条 0.29）、p90 ≤ 0.5、p10 ≥ 0.1，且没有一行开 =
  关（= Harmonizer 没起作用）；按层分报。
- 任一闸不过：不建孪生库，记录原因，报主会话。

## 训练端开关（只做恒等检验，不训练）

- `rm_train.py train --twin <孪生库>:<p>`（默认空 = 关）。抽到的池行若有孪生行，以概率 p 用孪生 token 替换（在 fetch 里替换，ego / 标签 /
  teacher 等仍是关行的）。掷币用独立随机流：种子 = [seed, 0, 223, 本批池抽样哈希]，与线程顺序无关，不动任何已有随机流。只认池行；
  合并池的行名 `on:<part>:<name>` 去掉前缀后与孪生库的 `<name>` 配对（open_rows 的规则）。日志记：配上的孪生行数、在 `--on-wfile` 权重下
  的抽样质量、实际替换行数 / 有孪生的抽中行数（应 ≈ p）。
- 一致性项（同一行关 / 开 token 的计划互相拉近）**不做**：要在主训练前向旁再加一次前向并接进 ST.terms 的损失路径，不是小改动；
  若主会话要，单独登记。
- 恒等检验（joint.sh 的做法）：R6DAG-L18-s0 的 argv 60 步 eager，(a) 我改动之前的提交在 worktree 里跑，(b) 当前 HEAD 不加 `--twin`，
  `df.py bitcmp` 逐位比较 → `results/harm_mix/identity.json`。孪生库建好后再跑一个 60 步烟测 `--twin PAIHTW:0.5`（只看替换率与能跑通，
  不留 checkpoint）。
- rm_train.py 正由第 7 轮联合臂同时修改：每次改前 pull，只加一个独立、默认关的开关，不动他们的代码。

## 成本与盘

- 渲染：pai_rows H 关渲染 8.95 作业时 / 853 场景 ≈ 38 s / 场景；开 2.2×（第 435 条）≈ 83 s → 246 个场景约 **5.7 作业时**，关重渲 6 个场景
  约 0.1。开渲染每作业约占三分之一到四成张卡 → 约 **2.3 卡时**。行 / 缓存 / 池 / 拉伸约 0.5 作业时（编码）。恒等 2 × 60 步 + 烟测约 0.3
  作业时。合计约 6.5 作业时、约 3 卡时。墙钟取决于池里空卡，联合臂优先。
- 盘：开行文件约 54 MB / 场景 → 13 GB；缓存约 10 GB；拉伸分片约 15 GB；**开池约 27 GB**（PAIHON 10.3、05 9.4、15 7.5）；关重渲池 < 1 GB；
  渲染临时文件按块清。峰值约 70 GB（剩 1.5 TB）。闸全过、孪生库建好后删行文件、缓存、拉伸分片与关重渲的行 / 缓存（记入
  `experiments/cl_infra/cleanup/`），留开池与孪生库约 28 GB。

## 不做的事

- 不开训练臂：什么时候和联合臂合并由主会话定（先看联合臂开 / 关 Harmonizer 的 281 读数）。
- 不改任何关池、不改公开榜 serve。
