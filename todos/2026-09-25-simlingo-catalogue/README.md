# SimLingo 同一 checkpoint：官方 Bench2Drive 协议 vs SimLingo 仓库自带的 Bench2Drive 副本

状态: running（预登记与所采用的设计 (a') 已写死；计分 run 排在 slot simlingo-exp，预计 06:30 起）
主题: [research/leaderboard-vs-ability.md](../../research/leaderboard-vs-ability.md)；回答 [decisions.md](../../research/decisions.md) 第 35 条「怎么才能定下来」的 (2)

## 目标

[simlingo#43](https://github.com/RenzKa/simlingo/issues/43) 里，有用户用官方 Bench2Drive 评 SimLingo 得 DS（Driving Score，
每条路线 RC × 违规折扣，再对路线平均）75.50，改用 SimLingo 仓库里自带的 `Bench2Drive/` 目录后得 86.53，
而 SR（Success Rate，走完且除 min speed 外没有违规的路线比例）两次都是 67.12%（147/219），一条都没变。
第 35 条据此推测：SimLingo 系四个方法（SimLingo、BLUE、RoG-DAgger、FIVE-VLA 都用这个 agent 和这套评测）的 B2D 分数对官方协议偏高，
上限约 11 DS。本实验用同一个 checkpoint 在两套评测上各跑 3 遍 220 条，直接量这个差，并把差拆到具体的代码改动上。

## 两套评测逐行对比（跑之前做完）

对比对象：官方 `Thinklab-SJTU/Bench2Drive` 分支 `0.0.3` 的 `21d85ee`（SimLingo README 说它的副本基于 0.0.3；
按 `diff -rq` 逐 commit 比对，0.0.3 上 2025-02 之后的 commit 与副本差异最少，且 2025-02 到仓库首次提交 2025-05-08 之间没有新 commit）
和 SimLingo 仓库 `RenzKa/simlingo@743b243`（2025-08-25）的 `Bench2Drive/`。
box 上官方臂实际跑的是 `0.0.4`（`7ec25d1`）：它和 0.0.3 在 `leaderboard/`、`scenario_runner/` 里唯一的差别是多了一个 dev 路线 XML
和一个评测不 import 的 `srunner/scenariomanager/utils.py`，评测代码逐字节相同。行尾（CRLF）和文档图片的差异不列。

**路线目录本身没有改。** SimLingo 用的是 `leaderboard/data/bench2drive_split/` 下 220 个单路线 XML，把它们逐条和官方
`bench2drive220.xml` 对比（去空白后逐 route 比 XML），220 条完全一致；`bench2drive220.xml` 和 `weather.xml` 也逐字节一致。
所以 #43 里说的「定制目录」不是路线或 scenario 参数，而是评测代码。全部差异如下：

| # | 文件 | 官方 | SimLingo 副本 | 影响什么 |
|---|---|---|---|---|
| D1 | `leaderboard/scenarios/scenario_manager.py` `_tick_scenario` | `tick_count > 4000` 时抛 `TickRuntimeError`，路线以 `Failed - TickRuntime` 结束，RC 停在 200 s 时的值 | 这两行被注释掉：路线一直跑到 route timeout（`RouteTimeoutBehavior`，至少 300 s 游戏时间，随前进延长）、到达终点、或 60 s 不动（`AgentBlockedTest`） | 慢路线的 RC 和成败；这就是 [simlingo#44](https://github.com/RenzKa/simlingo/issues/44) 作者承认的改动 |
| D2 | `srunner/.../atomic_criteria.py` `RouteCompletionTest.PERCENTAGE_THRESHOLD` | 99 | **90** | RC > 阈值且离终点 < 10 m 即判完成、RC 记 100、路线结束。B2D 路线约 150 m，10 m 约是 7%，所以 90 比 99 能让「停在终点前几米」的路线被判 `Completed`：既加 RC，也可能加 SR。#43 和 #44 都没提到这一条 |
| D3 | `tools/merge_route_json.py`（算总分的脚本） | DS = Σ/220，SR = 成功数/220；`Failed - Agent crashed` 的路线按 0 分计入 | 跳过 `Failed - Agent crashed` 的记录，分母改成实际计入的路线数 | 只影响计分，不影响仿真；#43 里「219 条」就是这样来的 |
| D4 | `leaderboard_evaluator.py` 启动服务器后的等待 | 30 s | 60 s | 不影响分数（我们的 runner 自己管服务器，两臂都不走这段） |
| D5 | 7 个 scenario 文件、`carla_data_provider.py` | — | 往 `CarlaDataProvider.active_scenarios` 里登记 scenario actor（给 PDM-Lite 专家采数据用） | 只记账，不改任何 actor 行为；SimLingo agent 不读它 |
| D6 | `autonomous_agent.py` `set_global_plan` | — | 多存一份稠密路线 `org_dense_route_world_coord` | SimLingo agent 不读它（只用 50 m 降采样的 `_global_plan`），对本实验无影响 |
| D7 | `route_parser.py` ×2、`scenario_parser.py`、`leaderboard_evaluator.py` 的 import | `getchildren()`、`pkg_resources` 版本检查 | `list(elem)`、注释掉版本检查 | Python 版本兼容，无行为差异 |

SimLingo 的评测启动脚本（`start_eval_simlingo.py`）另有几处不属于目录的设定：一个进程一条路线、`--timeout=600`（客户端 RPC 超时）、
TM（traffic manager）seed 取 `[1, 2, 3]`、失败路线重交一次。本实验两臂共用同一个 runner，这些设定两臂相同（见下）。

**预期（只从代码推，登记在这里）**：D1、D2 在仿真里起作用，D3 只在计分时起作用。D1 只影响跑过 4000 tick（200 s）的路线；
这类路线在官方臂必然失败（`TickRuntime` 不是 Completed），在 SimLingo 臂里有机会走完，所以**D1 也可能改 SR**，
#43 里 SR 一条没变不是从代码能推出来的必然结果。D2 同理。

## Setup

| 项 | 两臂共用 |
|---|---|
| checkpoint | `RenzKa/simlingo@26c7c89e797d4e25bbf640013317af8da26a5454`，`simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`（sha256 `ec894372…a700d28`，2 569 679 322 B）+ `.hydra/config.yaml`；基座 `OpenGVLab/InternVL2-1B`（ModelScope 拉取，按 HF 的 sha256 / git oid 校验） |
| agent | `RenzKa/simlingo@743b243` 的 `team_code/agent_simlingo.py`，原样；`--agent-config <ckpt>+/run`（agent 从 `+` 后取可视化输出的子目录名，否则 `setup()` 拿到 `None` 拼路径会崩） |
| 环境 | `~/data/envs/simlingo`：Python 3.10、torch 2.8.0+cu128、transformers 4.46.3，其余按 `environment.yaml` 的 pin。**偏离**：作者 pin 的是 Python 3.8 / torch 2.2，不支持 Blackwell（sm_120），只能换；这可能让绝对分数和作者的 87.4 有出入，但两臂用同一个环境，不影响配对差 |
| CARLA | 0.9.15，同一个二进制，`-RenderOffScreen -quality-level=Epic`，固定 `-graphicsadapter` |
| runner | `scripts/b2d_run.py`（一条路线一个进程、服务器复用、看门狗、基础设施失败最多重试 3 次），经 `scripts/simlingo_catalogue_run.sh` 启动。两臂只差 `BENCH2DRIVE_ROOT`（哪棵 leaderboard + scenario_runner）和路线 XML 的路径（内容相同） |
| world 设置 | runner 统一：同步模式 20 Hz、`deterministic_ragdolls`、TM 同步 + hybrid physics（与两棵树各自的 `_setup_simulation` 一致） |
| 只读记录 | `b2d_hooks` 的计时 tick 副本（对两棵树各自的 `_tick_scenario` 逐行复制，按源码 md5 区分，SimLingo 那棵不带 4000 截断）；每 20 tick 记一次 RC 与违规事件数（`rc_trace.jsonl`）；criterion 事件带 frame（`criterion_events.json`） |

### 条件

| 臂 | leaderboard + scenario_runner | 4000 tick 截断 | 完成阈值 | 主计分脚本 |
|---|---|---|---|---|
| **O**（official） | Bench2Drive 0.0.4 `7ec25d1`，原样 | 有 | 99 | 官方 `merge_route_json.py`（/220，崩溃算 0） |
| **S**（simlingo） | `RenzKa/simlingo@743b243` 的 `Bench2Drive/`，原样 | 无 | 90 | SimLingo 的 `merge_route_json.py`（跳过 crash，/N） |

- n = 220 条路线 × TM seed {1, 2, 3} × 2 臂 = 1320 个 route run。seed 取 SimLingo 启动脚本的默认值，两臂按 (路线, seed) 配对。
- 基础设施失败（服务器段错误、看门狗）自动重试，最多 3 次；仍失败的路线记缺失，在官方口径里算 0 分，缺失路线逐条列出。
- 两臂交错排队（同一时间两臂都在跑），避免时段负载差异偏向一臂。

### 指标

- 每臂每个 seed：DS、RC（Route Completion）、SR，以及两种计分脚本各算一遍（O 用 SimLingo 脚本、S 用官方脚本，这就是 D3 的效应）。
- 违规分项：碰撞（行人 / 车辆 / 静物）、闯红灯、闯停车牌、出车道、偏离路线、`Agent got blocked`、`TickRuntime`、route timeout、
  scenario timeout、未让急救车、min speed；路线结束状态分布；每条路线的 tick 数。
- **主比较**：Δ = S − O，按 (路线, seed) 配对，先对 3 个 seed 平均得每条路线的差，再对 220 条平均。
  95% CI 用按路线整组的 cluster bootstrap（重抽路线，3 个 seed 一起带走，10 000 次，percentile）。
- **run-to-run 噪声**：每臂内 3 个 seed 的 220 条 DS 的标准差，以及逐路线 seed 间差的分布（给兄弟实验的噪声估计用）。
- **归因**（把 Δ 拆到 D1 / D2 / D3）：
  - D3：同一批记录换计分脚本，精确。
  - D1：用 S 臂的 `rc_trace` 和带 frame 的违规事件，把每条 S 路线在第 4000 tick 截断重新计分
    （RC 取第 4000 tick 的值，只保留 frame 不晚于第 4000 tick 的违规，没在 4000 tick 前走完的判 `TickRuntime` 失败），
    得到「同一条轨迹、只加上截断」的分数。这是同一次仿真上的反事实，不含重跑噪声。
  - D2：trace 记下 RouteCompletionTest 判完成那一 tick、被覆盖成 100 之前的真实完成度（`_route_accum_perc[_index]`）；
    S 臂里这个值 ≤ 99 的完成（只有阈值 90 才会判、阈值 99 当时不会判）逐条列出，计它们贡献的 DS 和 SR。
  - 剩余 = Δ − D1 − D2 − D3，理论上只有重跑噪声（两臂的仿真在 4000 tick 前除 D2 外完全相同）。

### 判据（跑之前写死）

预登记的假设（来自 #43）：**S 比 O 高最多约 11 DS，SR 不变。**

| 结果 | 读法 | 对第 35 条 |
|---|---|---|
| ΔDS 的 CI 下界 > 0，且 ΔSR 的 CI 含 0 | 假设成立：SimLingo 副本抬高 DS 而不改成功数；SimLingo 系在 B2D 上报的分数要按 ΔDS 下调才能和官方协议的方法比 | 推测升为结论（仍标待定，直到有第二个 checkpoint），写入 ΔDS 与 CI |
| ΔDS 的 CI 下界 > 0，且 ΔSR 的 CI 下界 > 0 | 副本同时抬高 DS 和 SR；「SR 不变」这半句被否定 | 改写推测：两项都要下调 |
| ΔDS 的 CI 含 0，且 CI 上界 < 3 | 同一 checkpoint 上看不出目录效应；#43 的 11 分差来自别处（环境、机器、那次运行的噪声） | 删去这条推测，说明原来说了什么、为什么删 |
| ΔDS 的 CI 含 0 但上界 ≥ 3 | 没检出，也不能排除几分的效应 | 降级为「未检出」，写明可排除的上限 |

幅度另报：ΔDS 与 11 相比（CI 覆盖 11 就说「与 #43 一致」，CI 上界 < 11 就说「比 #43 小」）。
归因部分只描述各项占比，不设判据；D1 + D2 + D3 占 Δ 的比例写进 decisions 条目。

## 成本估计与分批

**smoke 实测**（2026-09-25 00:58–01:09，官方臂，路线 24240 / Town10HD，GPU 0 上另有 Alpamayo 特征抽取等任务，利用率 60–100%）：
323 tick、wall 634 s，DS 100（Completed）。每 tick 1854 ms，其中 **agent 1812 ms**，`world.tick` 14 ms，scenario tree 27 ms；
路线外的开销（加载模型、load world、清理）约 36 s。每 worker 显存约 11.5 GB（CARLA 7.5 + agent 3.9），CPU 约 2 核。
agent 慢的原因是作者的推理路径：`config_simlingo.use_cot = True`，每个 tick（20 Hz）先用 InternVL2-1B greedy 生成最多 100 个
token 的 commentary，再做一次带 driving token 的 forward。这是作者评测时的设置，按规则原样跑，不改。
（中途排除过一个嫌疑：torch 按宿主机 208 核开线程池，单进程 465 个线程；把 OMP/MKL 线程压到 2 之后每 tick 没有变化，瓶颈不在 CPU。）

S 臂 smoke（同一路线 24240，01:13–01:21）：313 tick、wall 453 s、agent 1345 ms/tick，DS 100。它在 **完成度 90.63% 时就被判完成**
（trace 记下的覆盖前数值），比官方臂早 10 tick 结束（官方臂在 >99% 时才判）。所以 D2 不是边角情形：**几乎每条走完的路线在 S 臂里都是
按 90% 阈值提前判完成的**。只有「走到最后 10 m 停住或出事」的路线分数会因此不同；这部分不能从 S 轨迹反推，只能靠真实官方 run 对照。

两 worker 同卡并发测速（01:28，GPU 0 上只有这两个 worker）：每个 worker 约 1.1 s/tick，并发几乎不拖慢单个 worker。
Town12 路线 1711（S 臂）：353 tick、wall 546 s，路线外开销约 72 s（大地图加载）。

**工作量**：同一个 checkpoint 的第三方复测（`research/results/b2d-family/public/simlingo_userrerun`，SimLingo 目录、seed 1）里，
220 条路线的游戏时长合计约 19.1 万 tick（按 4000 截断算约 18.0 万），其中 12 条超过 200 s。所以一遍 220 条约 18–19 万 tick。

| 方案 | tick 量 | 按实测 1.85 s/tick、10 worker | 假设 GPU 空闲时 0.6 s/tick（未测） |
|---|--:|--:|--:|
| 完整设计 2 臂 × 3 seed × 220 | 约 111 万 | 约 57 h | 约 19 h |
| P1：S 臂 seed 1 全 220 条（带 trace） | 约 19 万 | 约 9.8 h | 约 3.2 h |
| P2：O 臂 seed 1，只跑两臂可能不同的路线（S 超 4000 tick、阈值判完成）+ 20 条随机对照 | 约 6–8 万 | 约 3–4 h | 约 1–1.3 h |

完整设计远超可用窗口。P1 + P2 能回答原问题，理由是两棵树的代码在第 4000 tick 之前、以及 RC 落进 90–99% 之前完全相同
（逐行对比结论）：S 臂的一条轨迹同时给出 S 分数，以及「同一条轨迹加上截断」时官方口径的反事实分数（D1，精确、无重跑噪声）；
D3 靠重新计分，精确；D2 逐条标出。P2 里的真实官方臂 run 用来核对反事实（截断后的分数是否与真实官方 run 一致到重跑噪声以内），
20 条随机对照路线给出重跑噪声。时间有余，再在受影响的路线上补 seed 2、3（两臂都跑）。
### 采用的设计（2026-09-25 01:55 定，main 按 (a') 排期；用户若改选，main 通知后重排）

**(a')：两臂各 1 个 TM seed（seed 1）× 全 220 条，按路线配对。** 取代上面「条件」一节里的 3 个 seed：
- n = 220 条 × 2 臂 × 1 seed = 440 个 route run，约 37 万 tick，10 worker 约 15 h。
- 主比较、CI、判据**不变**：Δ = S − O 按路线配对，95% CI 用重抽路线的 bootstrap（10 000 次）；判据表原样适用。
  每条路线只有一次运行，所以 CI 里包含了重跑噪声，但没法把它单独分出来；「run-to-run 噪声」一项改为：
  S 臂反事实截断分数（同一条轨迹）与真实官方 run 在 4000 tick 内都走完、且没有 D2 差异的路线上的逐路线差，作为重跑噪声的估计，并写明它的来源。
- 归因不变：D1 用 S 轨迹精确反事实；D3 重新计分；D2 = 「S 在截断下的反事实」− O，其中混有重跑噪声，按路线列出。
- 若 15 h 内有余量，再在「两臂分数不同」的路线上补 seed 2（两臂都跑），只作补充，不改主结果。
- 并发：10 worker，每卡 5 个；每个臂在两张卡上各有一个 runner（O：GPU 0 3 个 + GPU 1 2 个；S：GPU 0 2 个 + GPU 1 3 个），
  两臂同时跑，任一臂不会独占较空的卡。批量脚本 `scripts/simlingo_catalogue_batch.sh`，经 `scripts/slot_run.sh simlingo-exp` 在
  `alp-b2d-full`、`op-b2d-full` 结束后启动。

## 给兄弟实验（hazard family 拆分）的逐路线结果

`research/results/simlingo-catalogue/per_route.csv`：一行一个 (臂, seed, 路线)，列与 `research/results/b2d-family/public_routes.csv` 对齐
（`method, run, route_id, ds, rc, success, source`），另加 `status, penalty, ticks, n_<infraction>…`。
`method` 取 `SimLingo-official` / `SimLingo-simlingo`，`run` 取 `seed1..3`，`source` 为 `jev-rerun`。
runner 目录本身也可以直接喂给 `jevdrive/tfv6_rules.collect()`（同一个 `b2d_run.py` 目录结构）。

## 步骤

- [x] 两套评测逐行对比（上表）
- [x] 环境 `~/data/envs/simlingo`、权重、SimLingo 仓库 `~/data/third_party/simlingo@743b243`
- [x] smoke：两臂各 2 条（24240、1711）跑通，外加 2 worker 并发测速；GPU 0 ≤ 20 GB（并发测速时 23 GB，当时 GPU 0 空闲）
- [x] 本文件提交（任何计分 run 之前）；成本估计补进上一节；gpu-plan.md 登记
- [ ] 批量（slot simlingo-exp，排在 alp-b2d-full、op-b2d-full 之后，约 06:30 起，约 15 h）
- [ ] 分析、图、per_route.csv、decisions 条目（第 35 条就地更新）

## 结果

跑完再填。run dir：`$DATA_DIR/runs/simlingo-catalogue/{official,simlingo}/seed{1,2,3}/`。
