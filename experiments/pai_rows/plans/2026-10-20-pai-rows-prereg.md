# PAI-ROWS 预登记：盘上全部可训练 NuRec 场景的 PAI 人类状态行 + 帧间隔拉伸变体 + 新场景 S 行（2026-10-20 晚，写于任何输出之前）

## 为什么

- 第 404 条：计划抄自身速度，约 90 % 走帧通道（帧间运动），原厂 Cinque 也有，停车与转弯因此失败。
- 第 407 条：起刹缺失，训练历史里需要刹车的行 84 % 已在减速。
- 用户要：更多数据多样性 + 帧通道的修法，信号保持简单。下一轮（主会话定）是一个联合臂 J = SIMP 胜出配方 + 广义历史拉平（SIMP-G）
  + 本 lane 的 PAI 人类状态行（含拉伸变体），全量数据 / 2N 步；J-half（半 A / N 步）作数据规模对照；同步数对照。本 lane 只建行和闸，不训练。

## 场景（`results/scenes.tsv`，`scripts/pr.py scenes`）

- 下载器已按用户决定停止（可续）；只用盘上现有的：box `datasets/nurec/all-usdzs/` 1 301 个 usdz（清单 `results/usdz_20261020.txt`，大小与
  nurec_select.csv 逐个一致，0 截断）。其中 curated-train 860 个；去掉 `experiments/pai_fix/scenes/` 的 exam_scenes.txt（0）、train_exclude.csv（0）、
  drives_val160.csv 的 scene_id 列（7），以及 curated-val / test281（0）→ **853 个可训练场景**，登记为
  `physical_ai_av/pai-rows@v1:bcfba1aeeca8`（与 curated-val、test281 不相交，已检查）。
- 第 402 条池的 570 个里 565 个仍可用；**5 个现在因 drives_val160 被剔除**（1be86721、8d19b573、9c628a6c、b8566b44、bdf2c763）：SIMP 池若扩张，应同时去掉这 5 个场景的行（这里只报告，不改别人的池）。
- 新场景（不在第 402 条池）288 个：rest 201、bend 87（下载器按类别排序，停 / 急弯 / 起步类已在前 570 里取完）。
- 类别取 nurec_select.csv 的 `cls`（第 402 条的类）：stop_nolead 218、turn_sharp 216、launch 81、bend 137、rest 201。

## 数据规模切分（主会话 10-20 补充）

- **半 A** = 每个类别内按 sha256("pai-rows-half:" + scene) 排序后隔一取一：428 个场景，登记 `physical_ai_av/pai-rows-half-a@v1:70c6f83364b3`
  （半 B = 其余 425）。各类 A / B：stop_nolead 109 / 109、turn_sharp 108 / 108、launch 41 / 40、bend 69 / 68、rest 101 / 100。同一切分用于 H、拉伸与 S 行。
- 训练器的 held（crc32(scene) % 100 < 15，`--on-hold-pct 15`）在 853 个里占 134 个，两半里各自照常扣掉；行数表同时报「可抽」（held 外）行数。
- **每行遍数**（passes per row）= 步数 N × 每步从该池抽的行数 k ÷ 可抽行数 R。行数按每场景约 155 行（199 个决策减去最后 4 s 无完整人类未来的行）先估，建完后以 `results/rows.md` 为准回填：

  | 臂 | 场景 | 可抽行 R（估，plain / 含两个拉伸变体） | 步数 | k = 16 时每行遍数（plain / 含拉伸） |
  |---|--:|--:|--:|--:|
  | J-half | 半 A 428 | 约 5.6 万 / 约 16 万 | N | 16 N / 5.6 万（N = 1 000：0.29 / 0.10） |
  | 全量 / N（本轮不跑） | 853 | 约 11.2 万 / 约 32 万 | N | 0.14 / 0.05 |
  | J | 853 | 约 11.2 万 / 约 32 万 | 2N | 0.29 / 0.10（= J-half 的遍数） |

  k 是 J 配方里人类 PAI 行的每步抽取数（第 402 条：on-k 32 的一半 = 16）；J 与 J-half 每行遍数相等，只差数据量。第 402 条 1 000 步约 0.2 遍、
  第 368 条在更大池上沿用 9 500 步，都是固定步数加数据，这里改成步数随数据走。

## 第 1 步：人类状态模仿行 H（`scripts/pr.sh chain H`）

- 渲染：第 381 条的 H 读法原样（`pai_gap.sh`）：`pai_box.sh`，NTV0-L18-s0 照 A0 serve 喂入并记录（`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`、g10），
  `DRV_PY=onp_log_driver.py` 让 ego 跟人的 log（MPC 跟踪），`send_recording_ground_truth=true`，`PAI_DUMP=100000`、`KEEP_ASL=1`、CONC 4、ALPASIM_FAST=1，10 Hz。
  模型只被喂入不驾驶，行的状态与模型无关；记录 NTV0 是为了 R1。
- 行：`onp_rows rows / cold`（与第 402 条相同）→ `onp_cache build`（冻结 Cinque 编码，驱动器的槽协议：8 个槽 = 渲染帧 t0 − 1.4 .. t0、0.2 s 间隔，
  对子 = (槽 j − 1, 槽 j)，冷启动行用驱动器的回扭帧；与第 333 / 367 / 402 条 PAI 训练行同一路径）→ `rf_pai_pool --fut log_fut --prefix paih --src human`
  → 池 `on1_PAIH`。
- 标签 `fut` = **人的 log 未来位姿**（`log_fut`：log 在 t0 + 0.5 .. 4 s 的后轴位姿，t0 的 rig 坐标系），只留 8 个都在 log 内的行；与 `imit`（PAIT 用的
  恢复目标）的差别只在跟踪误差（盘上旧 H 行 p99 0.32 m）。行筛选同 `rf_pai_pool`：8 槽有效、在第一个评分事件与 log 结束之前。
  `ego` = 驱动器喂的 20 维行（未取整），`speed` = v0，actor 框、路线记忆与路线标签照 `rf_pai_pool`；PAI 无道路栅格（road ok False）。
  训练时按模仿行用（`--on-mode imit` 或 `both`，由训练 lane 定）。

## 第 2 步：帧间隔拉伸变体（`pr.py stretch`）

- f ∈ {0.5, 1.5}：8 个槽改取 t0 + f × (−1.4 .. 0) s 的渲染帧（0.1 / 0.3 s 间隔，10 Hz 上精确，容差 30 ms，同 inertia.py 的 W_f），对子 = 新的
  (槽 j − 1, 槽 j)，用 `onp_cache.encode`（从 build 里原样抽出的函数）编码，teacher 在新 token 上重算。
- ego 行一致缩放（时间压缩 f）：历史位姿 = ego 在 t0 + f × (−1.5, −1.0, −0.5) s 的**实际位姿**（各决策 anchor 在时间上线性插值，换到 t0 的 rig 坐标系），
  速度 × f，加速度 × f²（与 rm_train `--ego-scale-speed` 的幂次一致），指令不变，经 `parity_adapter.ego_features` 组装。f = 1 时用存档历史。
- 标签不变：人的 log 未来位姿（同一时刻同一位置，停车点仍在人的停车点）。路线记忆、actor 框、路线标签照抄原行。
- 丢弃：冷启动行（没有可重取的帧）、所需帧或历史早于会话开始的行（f = 1.5 约前 2.3 s）。名字 `on:paih05:..` / `on:paih15:..`，池 `on1_PAIH05` / `on1_PAIH15`，
  tab 多 `f`、`base`（原行名）；`speed` = f × v0。
- **可行性**（只报告，不改）：`a05 = 2 (|fut_0.5s| − f v0 × 0.5) / 0.25`（把喂入速度 f v0 带到标签第一个位姿所需的匀加速度），以及到 1 s 的同量；
  报 |a| > 3 / 6 / 9 m/s² 的份额，按类别与车速分。若 f = 1.5 上 |a05| > 6 的份额 > 10 %，在结果里给出最小修法的建议（不实现）。

## 第 3 步：新场景 S 行（`scripts/pr.sh chain S`，优先级低于 H）

- 288 个新场景，第 402 条管线不变（`cp.sh` 的滚动：NTV0-L18-s0、PAI 探针 serve、帧 dump、`onp_rows` → `onp_cache` → `rf_pai_pool` 默认参数）→ 池 `on1_CPAIX`
  （前缀 `pai`，与 on1_CPAI570 相同的格式；合并进 SIMP 池时由训练 lane 定前缀）。

## 闸（读之前写定）

- **分阶段**：每个 pass 第一个 chunk 1 个场景 → 检查（打分、无 RPC 错误、dump 与 rollout.asl 在；H 还要 k ≥ 20 的每条记录都返回 log 且带 log_poses，
  即 pai_gap.sh 的闸）→ 10 个场景 → 检查（≥ 9 / 10）→ **probe**（11 个场景上把整条尾巴跑一遍，池 `on1_PAIHT*` / `on1_CPAIXT`）→ 其余（每 chunk 32，至多 W = 8 个 chunk 同时持有 rollout）。
  任一检查不过 → 写 `$DATA_DIR/runs/pai_rows/<pass>/STOP`，其后不提交。
- **R1（训练器前向）**：`rf_recov.py fwd`（rm_train 的 Store 与模型，即训练路径）在池的 token + ego 上前向 NTV0-L18-s0，对驱动器记录的 `poses_model`；
  线 = `pp.py r1`（第 367 / 402 条）：中位 ≤ 0.02 m、超出 max(0.03 m, 4 fp16 ulp) 的行 ≤ 10 %、错模型 RB0-F-s1 超出 ≥ 90 %、无缺行。probe 与全量各判一次。
- **f = 1 位等**：同一拉伸代码在 f = 1 下（槽按时间重取、ego 经 ego_features 重组）对 plain 池逐行：ego、fut、8 个 token、teacher 全部 bit 相等，重取的槽与存档槽全同。
  不过 → 停（说明编码批组合或槽 / ego 路径有差，先查清再建拉伸池）。另报 f = 1 时 anchor 插值历史对存档历史的最大差（拉伸历史来源的精度）。
- 不判训练效果；行数、类别、半切分、磁盘、花费写进 `results/`。

## 花费与磁盘（估）

- H 滚动 853 场景 ≈ 27 个 32 场景 chunk × 约 0.35 作业小时（第 402 条 c002 1 267 s）≈ 9.5 作业小时（每卡 3 个 ≈ 3.2 卡时）；S 288 场景 ≈ 1.1 卡时；
  编码（cache 约 20 min、两个拉伸各约 20 min）、前向 R1 约 15 min → 合计约 **6 卡时**，低于 10 卡时，不需在全量前停下请示。
- 磁盘：H 行文件约 43 GB（含帧，拉伸与追加都要用，保留）、H cache 约 32 GB、三个 H 池约 115 GB（每行 front 0.26 MB）、拉伸 parts 约 22 GB、S 约 37 GB
  → 约 250 GB 净增；临时 rollout.asl / dump ≤ 8 × 3.6 GB，按 chunk 删。剩余约 1.1 TB（下载器已停）。

## 追加

`scenes.tsv` 增加场景后重跑 `pr.sh chain H|S`：只列新场景（32 一个 chunk，不再分阶段），cache 与拉伸 parts 按场景续，池整体重建。半切分只对 `pai-rows@v1` 有效；新场景进来需新版本切分。

## 补记 1（2026-10-20 19:40，probe 的 f = 1 闸不过之后、任何拉伸池使用之前；设计与闸不变）

- probe（11 个场景，1 759 行）：两级滚动检查过（k ≥ 20 的 179 / 1 789 条记录全部返回 log）；**R1 过**（中位 0.0096 m、超出 3.4 %、错模型 99.8 %）。
- **f = 1 位等闸不过**：重取的槽 1 605 / 1 605 与存档全同，ego、fut 逐位相等，anchor 历史对存档历史最大差 4e-15；但 8 个 token 只有 99.81 % 的行逐位相等
  （差的行最大差 0.0156 = 该量级的 1 个 fp16 步），teacher 最大差 0.031。原因：拉伸代码只编码保留行的对子，对子顺序与每批 8 个的组合与 `onp_cache build`
  不同，而编码器在最后一个 fp16 步上不是批组合不变的（teacher 的 256 行一批同理）。
- 修法（只改编码顺序，不改定义）：拉伸代码改为编码整个行文件的槽表（保留行用拉伸后的槽，其余行用存档槽与冷启动帧），对子顺序与批组合因此与
  `onp_cache build` 完全相同，再取保留行。f = 1 时应逐位等于 plain；重跑 probe 判同一条闸。若仍不过，停。
- 读法：plain 行的 token 本身也只在「同一批组合」意义下确定；拉伸行与 plain 行之间的数值差异上限就是这一个 fp16 步，对训练无实际影响，但闸按登记仍要位等。
