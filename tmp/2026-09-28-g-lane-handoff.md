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

## 当前状态（2026-09-28 晚，给下一个 Sonnet 执行员）

前一个执行员被换下（省 token），这是交接快照，写于 20:53 CST。

**owner**：PID `770170`，box 上 `.venv/bin/python scripts/nq4_g_lane.py run`，tmux 窗口 `jev:nq4-g-r15`（`r13`、`r14`
是之前两次重启留下的空壳，进程已退出，窗口可能还挂着但没有内容）。`$DATA_DIR/runs/nq4/g-lane/pid` 是当前 owner 的准的来源，别看窗口名猜。

**box 端等待脚本**：tmux 窗口 `jev:g-lane-watch`，跑 `$DATA_DIR/runs/nq4/g-lane/g_lane_watch.sh > .../watch.out 2>&1`，15:51:26 起挂着，
watching `owner_pid=770170`（当前有效，不用重挂）。它是本执行员写的一次性脚本（不在 git 里），逻辑：`tail -F log.txt` 事件驱动 + 每 9000 s（2.5 h，按用户 2026-09-28
晚间新规矩改的，原来是 1 h）兜底检查一次 owner 存活 / DONE / ERROR / 新的 `ERROR.cell.*` / `ERROR.<key>` / `arms_blocked/*`；触发条件命中就写一行 `TRIGGER=...` 到
`watch.out` 然后自己退出（连带 tmux 窗口关掉，需要重开）。**新执行员请直接接上这个脚本**（不用重写）：`ssh autodl 'tail -F -n0 $DATA_DIR/runs/nq4/g-lane/watch.out | grep -m1 "TRIGGER="'`
挂一个阻塞等待即可；如果 `jev:g-lane-watch` 窗口已经不在了（脚本已经触发过一次退出），照第 7 节的方式重开：
`tmux new-window -t jev -n g-lane-watch "bash $DATA_DIR/runs/nq4/g-lane/g_lane_watch.sh > $DATA_DIR/runs/nq4/g-lane/watch.out 2>&1"`。
脚本本体在 box 上：`$DATA_DIR/runs/nq4/g-lane/g_lane_watch.sh`（没提交到仓库，纯 ops 脚本）。

**Mac 端监听**：本执行员挂的 `ssh ... tail -F | grep TRIGGER` 后台任务已按精确 task id 停掉（`beamen6f9`），没有留下任何本地后台进程 —
换人之后是干净的，新执行员自己重新挂一个指向上面 watch.out 的监听即可。

**格子表快照（20:53 CST，`cat $D/g-lane/STATUS.md`）**：`shift`/`swap`/`ghost`/`orig` seed 0 基本全 DONE，只剩 SimLingo 还在跑
（`simlingo.shift.0` 20/80、`simlingo.swap.0` 46/50，在卡 0、3 上共 11 个 worker）；`ghost`+`orig` seeds 1–2 里没起的还有
`simlingo.ghost.1`(0/80)、`simlingo.orig.1`(0/80)、`blue.ghost.2`(0/80)、`blue.orig.2`(0/80)、`simlingo.ghost.2`(0/80)、`simlingo.orig.2`(0/80)——
这 6 格排在 SimLingo/BLUE 前面的活干完之后应该自动起。PDM-Lite 的 `shift`/`swap`/`orig.1`/`orig.2` 仍是登记的 `DROPPED`，原样不动。
GPU 4、5 已经被 wm-loop 的 demand file 要回去（20:53 的 STATUS 显示这两张卡上「this lane」是 0 worker、「CARLA others」各 6 个）——drain 机制照设计工作，不是异常。

**blue.swap 的修正与归档**：main 2026-09-28 下午判定「zone reached ≥ NQ3 − 30 pp」这条清单项要按世界判（同路线上其它考生都到区就是该考生自己的结果，不是评测器坏了）。
核对：TFv6、BridgeDrive 在 blue.swap 用的同 10 条 pilot 路线上都是 100% 到区（TFv6 没有 NQ3 tfv6 臂做对照，用的是它自己的到区比例；BridgeDrive 有臂，两边一致）。
处理：
- 原判决归档到 `$D/gk/arms_blocked.behaviour-20260928/blue.swap`（附 README）、`$D/gk/ERROR.blue.swap` 改名为 `ERROR.blue.swap.behaviour-20260928`（和 storm 系列的归档同一手法）。
- `state.json` 里 `pilots["blue.swap"]` 手动改成 `PASS`（`how` 字段写清楚是 main override，不是清单本身过了），`cells["blue.swap.0"]` 清空重置，然后重启 owner 让它接上——
  10 条 pilot 路线的 `done/*.json` 本来就在批量要读的同一个 `arm_dir` 里，所以批量只补剩下的 40 条，没有重跑 pilot。现在 `blue.swap.0` 已经 DONE（50/50）。
- 判据原文本身没有改（下次别的考生在别的世界也这样卡，还是要按第 5/9 节走一遍核实，不能直接套用这次的结论）。
- 决定写在 `todos/2026-09-26-night-queue-4.md` 的 G 节（`[main] 2026-09-28 post-hoc amendment`），已 commit/push（`e63d3a1`）。

**这次交接学到的、原笔记没写的事**：
1. **`state.json` 有写竞态**：owner 每 `POLL_S=20 s` 一轮会用自己内存里的状态整份覆写 `state.json`。在 owner 还活着的时候直接改这个文件会在下一轮被它自己的旧内存值盖掉
   （亲身验证过一次：改完文件、等了几秒发 `kill -TERM`，owner 在这几秒里又存了一次盘，把我的改动冲掉了）。正确顺序是：**先 `kill -TERM` + 确认进程真的没了（`ps -p` 查不到），再改 `state.json`，最后重启 owner**——
   笔记第 5 节写的顺序（先改文件再重启）在这台机器上不安全，除非能保证改完立刻杀、中间不隔一轮。
2. **`CARD_CAP`（比如 6）不是硬性的 server 数上限**：代码里轻模型（TFv6、BridgeDrive）按 per-candidate 预算走，能比 CARD_CAP 走得更远，卡 0、3 长期跑在 6→7 个 CARLA
   是设计内的正常状态，不是残留/僵尸进程。`sch_table.py check` 真正卡的是显存（75 GB）、pids、CPU 核数三项，不查 server 数。之前一次 GPU3 撞到 76 GB 是因为 7 个里混进了两个重模型（BLUE），
   给其中一个打了 drain（路线边界退出，没杀路线），显存就回落了——这不是 bug，遇到类似情况不用当成异常处理，先看是不是「重模型混进去」再决定要不要 drain。
3. **GPU 1（cosmos-pilot 的卡，不归 G lane）晚上 `sch_table.py check` 报过 75 GB 超限**：只读到，没有动它（第 GPU1/2 不归我们，硬约束里写了）——交接给下一个人时提一句，免得被当成 G lane 自己的问题去查。
4. 临时诊断脚本 `$DATA_DIR/runs/nq4/g-lane/reopen_blue_swap.py` 留在 box 上（记录了上面第 2 条 state.json 改动的确切内容），没提交到仓库，纯留痕，可以留着或删，不影响 lane。
