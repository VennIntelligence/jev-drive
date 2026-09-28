# G lane 交接（2026-09-28 13:00 CST，给 Sonnet 执行员）

G lane 是 night-queue-4 的 ghost / 扰动闭环测试：4 个榜单考生（TFv6、BridgeDrive、BLUE、SimLingo）加 PDM-Lite 基线，在 G 的 80 条路线上跑
`shift`、`swap`、`ghost`、`orig` 四种世界。从现在起它是纯运行任务：lane 自己推进、自己重试，执行员只看状态、处理常规动作，遇到第 9 节的情况交回 main。
背景和登记的判据在 `todos/2026-09-26-night-queue-4.md` 的 G 节，过程记录在 `tmp/2026-09-27-box-restart.md` 和 `tmp/2026-09-27-gk-restart.md`。

## 1. 布局（13:00）

| 卡 | 用途 | 谁的 |
|:--|:--|:--|
| 0、3 | G 批量，每卡最多 6 个 CARLA | nq4-g |
| 6 | 测试卡：G 的分级 pilot、别的 lane 的 pilot 和 smoke | debug 行 |
| 4、5 | world-model fork 生成 | wm-loop |
| 1 | Cosmos-Transfer2.5 pilot（不跑 CARLA） | cosmos-pilot |
| 2 | openpilot 可训练性探针（不跑 CARLA） | op-train |

G 的 runner 按卡绑 24 核：卡 g 用 `24g`–`24g+23`（0-23、72-95，测试卡 144-167）。CARLA server 默认带减线程参数
（`-RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4`），每个 worker 约 220 线程，线程上限不是约束。

## 2. lane 怎么自己跑

- 进程：box 上 tmux `jev:nq4-g-r13`，命令 `env G_PIDS_PER_WORKER=250 G_CAPS=tfv6=6,pdm=6,bridgedrive=6 G_SERVER_MAX=6 .venv/bin/python scripts/nq4_g_lane.py run`。
  owner 的 PID 在 `$DATA_DIR/runs/nq4/g-lane/pid`（此刻 31494）。
- 每 20 s 一轮：重读调度表里自己的行（卡、每卡上限、核、index 段），量每张卡上实际的 CARLA 数，有空位就给最靠前、还有路线没被领的格子起一个 `b2d_run` runner。
  多个 runner 共用一个格子，靠 b2d_run 的路线 claim 分片。
- 每卡预算：每个 worker 占 1/6 张卡，外加切片 CPU 85% 和每卡最多 6 个 server 的上限。重模型（SimLingo、BLUE）在各卡之间摊平：
  一张卡的重 worker 数不超过最少那张 + 1，且最多 3 个；其余位置给轻模型（TFv6、BridgeDrive）。
- 全机同时最多 2 个 CARLA 在启动（`b2d_run` 的 flock 槽位 `$DATA_DIR/runs/sched/carla-start/`）。这是 10:1x 驱动卡死之后加的，**不要去掉**。
- owner 退出不会停 runner；重启 owner 时它按记录的 PID + start ticks 认领还活着的 runner，已完成的路线不会重跑。
- 全部格子结束后写 `DONE`；每一档结束时在后台跑一次 `jevdrive.nq4_g report`（结果在 `runs/nq4/gk/results/g/`）。

## 3. 要看的文件（box 上，`D=/root/autodl-tmp/ujs/runs/nq4`）

| 文件 | 内容 |
|:--|:--|
| `$D/g-lane/STATUS.md` | 每分钟刷新：每卡 CARLA 数、显存；每个格子的状态、完成数、在跑的 worker 数 |
| `$D/g-lane/log.txt`、`events.jsonl` | launch / drain / runner_end / pilot_check / cell_end / grant / loop_error |
| `$D/g-lane/DONE`、`ERROR` | 全部结束，或调度进程连续 10 次出错 |
| `$D/gk/ERROR.<考生>.<世界>`、`$D/gk/arms_blocked/` | 某个 pilot 没过清单（只挡这一格） |
| `$D/gk/ERROR.cell.<格子>` | 某格 > 10% 路线用完 attempt 仍没完成 |
| `$D/gk/arms/<考生>/<世界>/s<seed>/` | 路线输出：`done/<id>.json`、`attempts/`、`runner-g<卡>.log`、`servers/carla-*.log` |

等待要靠事件，不要定时轮询：在 box 端对 `$D/g-lane` 和 `$D/gk` 做 inotify（新建 `DONE` / `ERROR` / `ERROR.*`），再用 pidfd 等 owner 退出，
另加一个「60 min 没有新路线完成」的截止时间（最多每小时醒一次）。一次性检查用 `cat $D/g-lane/STATUS.md`。

## 4. 推进顺序（用户 2026-09-28 定）

档 0 `shift` seed 0 → 档 1 `swap` seed 0 → 档 2 seed 0 剩下的（已空）→ 档 3 `ghost` + `orig` seeds 1–2（殿后，3 seed 幽灵判格要用）。
PDM-Lite 只保留 `ghost` seeds 0–2；它的 orig 1–2、shift、swap 是 `DROPPED`，不要恢复。

每个「考生 × 世界」先过分级 pilot：1 条路线 → 查清单 → 10 条 → 查清单 → 全量。清单是 `jevdrive.nq4_g pilot-check`，没过就只挡这一格。
pilot 只在测试卡 6 上跑，批量只在 0、3 上跑。13:00 的 pilot 状态：

| 格子 | pilot |
|:--|:--|
| tfv6.shift、tfv6.swap、bridgedrive.swap、pdm.shift | PASS |
| blue.shift、blue.swap、simlingo.swap | 第 2 阶段 |
| bridgedrive.shift（第 2 阶段）、simlingo.shift（第 1 阶段） | 重跑中：11:17 的 BLOCKED 是驱动卡死期间 CARLA 崩溃造成的，判决存档在 `$D/gk/arms_blocked.storm-20260928/`（附 README） |

## 5. 重试规则

- 一条路线最多 6 次**真正跑了**的 attempt（每个 runner 3 次，加登记的一次重试）。server 在路线第一个 tick 前就崩（`server_died_*` 且 ticks 为空）不算在这 6 次里，
  但每条路线全部 attempt 合计上限 12 次，总是崩的路线也会结束。
- 某格 > 10% 的路线用完 attempt 仍没完成 → `ERROR.cell.<格子>`，这一格记 FAILED，其余格子照跑。
- **只因 CARLA 崩溃造成的失败可以重开一次**（登记的「可直接重试」）：先核实没完成的路线的失败 attempt 全是 `server_died_*` 且没有 tick，
  然后把 `ERROR.cell.*` 或 `arms_blocked/<格子>` 改名存档（后缀加日期和原因，写一个 README 说明），把 `state.json` 里对应的 cell / pilot 重置，再重启 owner。
  先例：`ERROR.cell.bridgedrive.ghost.1.storm-20260928`、`arms_blocked.storm-20260928/`、`arms_blocked.rc139-20260927/`。
  失败 attempt 里只要有一次是考生自己的结果（blocked、碰撞、超时、route_timeout 等），就不是常规情况，交回 main。
- 永远不写 PASS，不改清单、判格或阈值。

## 6. 调度表（`$DATA_DIR/runs/sched/table.tsv`，`python3 scripts/sch_table.py show|check|grant`）

| 行 | GPU | 每卡 CARLA | index | 核 |
|:--|:--|--:|:--|:--|
| nq4-g | 0,3（pilot 用 6） | 6 | 每卡 300 + 20g（0:300 … 6:420），span 20 | `0:0-23,3:72-95,6:144-167` |
| debug | 6 | - | - | 144-167 |
| wm-loop | 4,5 | 6 | 4:60、5:80，span 20 | 184-207 |
| cosmos-pilot | 1 | - | - | 24-47 |
| op-train | 2 | - | - | 48-71 |
| nq4-p3-pool | 3（sky） | - | - | 178-183 |

lane 每轮重读 nq4-g 行，改行即生效，不用重启。任何改动之后跑 `python3 scripts/sch_table.py check`，必须是 0。
wm-loop 要 G 让位时，会写 `runs/sched/demand/wm-loop.json`（如 `{"4": 6, "5": 6}`）；G 只在它自己的卡上让位。

## 7. 常用命令（box 上，先 `export DATA_DIR=/root/autodl-tmp/ujs; cd ~/data/jev-drive`）

**让 G 离开一张卡 g（drain，不杀路线）**：先从 nq4-g 行去掉这张卡（不再新起），再让那张卡上的 runner 跑完手上的路线退出：

```bash
python3 scripts/sch_table.py grant nq4-g --gpus 0,3 --status "<why, date>"   # the new card list without g
.venv/bin/python - <<'EOF'
import json, sys; from pathlib import Path
sys.path.insert(0, "scripts"); import nq4_g_lane as L
from cx_controller import process_snapshot
G = 3   # the card to drain
rows = process_snapshot(); st = json.loads((L.OUT / "state.json").read_text())
for k, r in st["runners"].items():
    if r["gpu"] == G and L.alive(r, rows) and r.get("drain"):
        Path(r["drain"]).parent.mkdir(parents=True, exist_ok=True); Path(r["drain"]).touch(); print("drain", k, r["key"])
EOF
python3 scripts/sch_table.py check
```

**把卡交给别的 lane**：drain 完以后，等这张卡上没有 CARLA、没有 `b2d_run --gpu-rank g`、显存 < 2 GB，再写
`$DATA_DIR/runs/sched/released-gpu<g>`（内容写时间和显存），并在表里给新 lane 加一行（`sch_table.py grant <lane> --gpus g --cpus <slice> --status ...`）。
13:00 的 `released-gpu1` / `released-gpu2` 就是这样写的（tmux `release-gpu12`，已结束）。

**把卡还给 G**：`python3 scripts/sch_table.py grant nq4-g --gpus 0,1,3 --cpus 0:0-23,1:24-47,3:72-95,6:144-167`，下一轮自动补位。

**重启 lane 的调度进程**（改代码后，或它退出了）：

```bash
git pull --ff-only
pid=$(cat $DATA_DIR/runs/nq4/g-lane/pid); kill -TERM $pid     # exact PID only; runners keep running
scripts/tmux_run.sh nq4-g-rN env G_PIDS_PER_WORKER=250 G_CAPS=tfv6=6,pdm=6,bridgedrive=6 G_SERVER_MAX=6 \
    .venv/bin/python scripts/nq4_g_lane.py run
```

窗口名每次换一个新的（`nq4-g-r14`、`r15` …）。环境变量不能省：没有它们时每卡上限和线程预算会回到旧的默认值。

**剩余工作量**：`.venv/bin/python scripts/nq4_g_lane.py plan`（只读）。

**停进程**：只按精确 PID（先 `tr '\0' ' ' < /proc/<pid>/cmdline` 核对），不用 `pkill -f`、不用 `pgrep -f` 去匹配会出现在自己命令行里的字符串，不杀进程组。
停 runner 用 `kill -INT <b2d_run 的 PID>`（它会取消路线、释放 claim、停掉自己的 server），或者用上面的 drain（更好，不丢路线）。

## 8. ETA（13:00，批量卡 0、3 约 12 个 worker，外加测试卡上的 pilot，剩 165.7 worker·h）

| 档 | 剩余 | 预计 |
|:--|--:|:--|
| shift seed 0 | 31.7 worker·h | 约 3 h，要等 BridgeDrive / BLUE / SimLingo 的 pilot 过；约 16:00 |
| swap seed 0 | 15.6 worker·h | 约 1.5 h；约 17:30 |
| ghost + orig seeds 1–2 | 118.4 worker·h | 约 10 h；明早 04:00 前后 |

worker·h 是按旧 box 的每路线耗时先验算的，SimLingo / BLUE 在新卡上更慢，偏乐观。每 3–5 h 用实际完成数重估一次。

## 9. 交回 main（不是常规情况）

- 一个 pilot 没过清单，而失败里有考生自己的结果（不全是 CARLA 崩溃）；或者 `ERROR.cell.*` 里有非崩溃的失败。
- `$D/g-lane/ERROR`，或者 owner 反复退出，或者 `loop_error` 在同一个原因上反复出现（这是代码问题，要 Opus 修）。
- 多个进程在 D 状态、nvidia-smi 卡住、新 server 在装图时成批 RenderThread 超时（10:1x 驱动卡死的特征）；或者 60 min 没有新路线完成。
- 任何人要求改每卡上限、调 `G_CAPS` / `G_SERVER_MAX`、改启动槽位数，或者重做 per-card knee 测量（那是新测量，不是运行）。
- 用户对范围、顺序或考生的新决定；所有要写进 `todos/` 的判断。
- 所有格子结束（`DONE`）：报告给 main，小表由 Mac 拉回 `research/results/nq4/g/`。
