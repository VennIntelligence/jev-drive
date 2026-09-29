# Cosmos G4 全量生成：交接（2026-09-29，随阶段更新）

最后更新：2026-09-29 12:25 CST（接手 agent）。登记、规则与读数都在 [todos/2026-09-28-cosmos-pilot.md](../todos/2026-09-28-cosmos-pilot.md) 的「全量生成」节；决策是 [decisions](../research/decisions.md) 第 56 条。

## 现在处于哪一步（12:25 CST）

- stage 1 / stage 10 已完成，checklist 5 条全过（表在 [todo 执行记录](../todos/2026-09-28-cosmos-pilot.md) 末尾，数字在 `research/results/cosmos/full/`）。
- main 11:59 已经自己 `touch lane/COSMOS_GO`（不等 stage 10 checklist）；5 个 Cosmos worker 在 GPU 0–4 上稳态 **219 s / 对**，卡上 VRAM 57–62 GB、峰值 ≤ 65 GiB，util 100%。
- 全量 CARLA 12:21 起是 **每卡 6 个 server**（idx 块 266..271），带 `B2D_KEEP_STREET_LIGHTS=1`；正在做的 invocation 4 是 612 个变体的第 1 遍（1224 条 route，之前的 chunk 被 DRAIN 打断后补完）。
- 预计的风险点：手上 READY 的 93 对够 Cosmos 干约 1.1 h（12:00 起），而第 1 遍 chunk 要 ~80 min，之后第 2 遍才出新对，所以 13:10 前后 Cosmos 可能有一段空档。之后每个 invocation 先排第 2 遍（p3, p2, 再 pass 1），应当接得上。
- 外推：Cosmos 5 卡 24.4 h（约 12:00 → 次日 12:30），CARLA 约 890 server·s / 对，30 个 server 约 16.5 h（共卡时会慢）。GPU 5 归世界模型训练，GPU 6 归另一条线，都不用。

## 做好了什么、在哪

| 东西 | 位置 |
|:--|:--|
| lane 驱动（选窗、controls、QC、调度 CARLA / Cosmos） | `jevdrive/cosmos_full.py`（`build` / `run` / `summary` / `checklist` / `controls-test`） |
| 启动脚本 | `scripts/cosmos_full.sh`（环境变量见文件头） |
| Cosmos worker（每 slot 一个进程，claims 协调，可以随时手工多开） | `scripts/cosmos_full_worker.py` |
| 分级 checklist | `scripts/cosmos_full_check.sh <stage>` → `research/results/cosmos/full/<stage>/checklist.json` |
| 录制器开关 | `scripts/p5_pair_agent.py`（`rig`），`scripts/cosmos_pair_agent.py`（`cosmos_stop`、`cosmos_rgb_attrs`） |
| 场景池（Town12 209 个实例）/ 考卷排除清单 | box `runs/cosmos_full/pool.csv`；repo `research/results/cosmos/full/scenes.csv` |
| controls 与 pilot 逐位等价 | 24211 上 rgb / edgeE / 锚定 mask / alpha / GT 最大差全 0（box `runs/cosmos_full/equiv/24211-s0/equiv.json`） |

调度表行 `cosmos-full`：GPU 0–4，每卡 server index 块 260 + 12k（全量用 +6..+11，stage 用 +0..+5），cores 0-47,72-143。`sch_table.py check` 通过。

## 在跑什么

| tmux 窗口（session `jev`） | 做什么 | 根目录（box `$DATA_DIR/runs/…`） | 信号文件 |
|:--|:--|:--|:--|
| `cosmos-full`（驱动 PID 用 `tmux list-panes -t jev:cosmos-full -F "#{pane_pid}"` 再取它的子进程） | 全量：CARLA 两遍 + controls + 5 个 Cosmos worker | `cosmos_full/` | `cosmos_full/lane/{STATUS,DONE,ERROR,DRAIN,CARLA_DONE,CONTROLS_DONE,COSMOS_GO}` |
| `cf-stage1`、`cf-stage10`、`cf-warm` | 已结束（窗口里只剩 shell），可以 `tmux kill-window` | `cosmos_full/stage1`、`stage10` | 无 |

全量驱动的启动命令（重开要一字不差，环境变量要放在 `env` 后面，`tmux_run.sh` 不继承 ssh 的环境；漏了就会退回默认目录，出过一次事故）：

```bash
ssh autodl 'cd ~/data/jev-drive && scripts/tmux_run.sh cosmos-full env B2D_KEEP_STREET_LIGHTS=1 CARLA_W=0:6,1:6,2:6,3:6,4:6 CARLA_IDX_OFF=6 CARLA_SPAN=6 COSMOS_SLOTS=0,1,2,3,4 scripts/cosmos_full.sh --target 2000'
```

重开前：`touch lane/DRAIN` 让驱动收尾，等驱动 PID 退出（约 3–6 min），`rm lane/DRAIN lane/ERROR`，`tmux kill-window -t jev:cosmos-full`。DRAIN 打断的第 2 遍对会被记成 `pass2_failed`（现在写进正确的列），**重开前要把 ctl.csv 里这些行删掉**（先备份 ctl.csv），否则那些对永远不会被重排；第 1 遍被打断的变体重开后自动补完。

每次 CARLA 调用的 b2d_run PID 在 `<root>/carla_logs/inv<NNN>-gpu<g>.pid`，日志同名 `.log`；Cosmos worker 的 PID 在 `<root>/cosmos_logs/<slot>.pid`。
进度：`<root>/lane/STATUS`（每次调用一行，Cosmos 阶段每小时一行）；机器可读 `runs/cosmos_full/lane/<时间戳>/events.jsonl`。
状态表：`variants.csv`（排了哪些变体）、`sel.csv`（选窗结果）、`ctl.csv`（controls / QC 结果），`pairs/<pair>/done.json`（Cosmos 完成）。

## 监看命令

```bash
# one-shot status
ssh autodl 'D=$DATA_DIR/runs/cosmos_full; tail -n 4 $D/lane/STATUS | cut -c1-250; ls $D/pairs/*/done.json | wc -l; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -5; df -h $DATA_DIR | tail -1'
# Cosmos seconds / pair and card peak of the last pair per slot; READY stock vs done pairs
ssh autodl 'D=$DATA_DIR/runs/cosmos_full; for g in 0 1 2 3 4; do grep -h card_peak $D/cosmos/g$g/*/events.jsonl | tail -1 | cut -c1-220; done; echo READY $(ls $D/clips/*/READY | wc -l) done $(ls $D/pairs/*/done.json | wc -l)'
# blocking wait for the whole run (returns only on DONE / ERROR)
ssh autodl 'L=$DATA_DIR/runs/cosmos_full/lane; timeout 10800 bash -c "until [ -e $L/DONE ] || [ -e $L/ERROR ]; do sleep 60; done"; tail -5 $L/STATUS'
```

看什么：Cosmos 每对应在 210–230 s；VRAM 每卡 < 80 GB（过去峰值 65 GiB）；READY 库存别降到 0（降到 0 说明 CARLA 供不上，可以在 invocation 边界给某张卡加 server，见下）；`STATUS` 里每个 invocation 的 route 完成率、选窗通过率（约 0.42–0.46）和 QC 标记率。盘低于 250 GB 驱动自己暂停 CARLA 输出。

## 还没做 / 待 main 决定

- 加卡：GPU 5、6 现在不是这条 lane 的。要加卡就是三步：调度表 `cosmos-full` 行的 `gpus` 加上那张卡（idx 块是 `260 + 12k` 按在 gpus 里的位置 k 算，先 `sch_table.py check`），`CARLA_W` 里加 `5:6`，`COSMOS_SLOTS` 里加 `5`，然后按上面的 DRAIN 流程重开驱动。已有 worker 不需要停手，`COSMOS_SLOTS` 多出来的 slot 是新进程；但驱动重开时会把旧 worker 一起收掉再起，所以要在 invocation 边界做。
- 全量结束后：`python -m jevdrive.cosmos_full summary`，把 `research/results/cosmos/full/` 拉回 Mac，decisions 第 56 条补全量结果；stage 10 里没开路灯开关渲染的日落对 c007v01 的旧输出在 `stage10/pairs_preflag/`（可以由 main 决定删不删）。

## 每种信号怎么办

- `DONE`：`python -m jevdrive.cosmos_full summary`，把 `research/results/cosmos/full/`（variants.csv、summary.json）拉回 Mac 提交，todo 执行记录与 decisions 第 56 条补结果，报 main。
- `ERROR`：看 `ERROR` 内容与 `STATUS` 末尾。驱动出错时会自动 `touch lane/DRAIN`（在跑的 route 和 Cosmos 对做完就停）。修好后 `rm lane/DRAIN lane/ERROR`，用同样的命令重开，已完成的全部跳过。
- 卡 / 盘：盘低于 250 GB 时驱动自动暂停新的 CARLA 输出，等 Cosmos 追上；低于 150 GB 报 ERROR。

## 坑

- 停进程只按 PID（上面的 pid 文件），不要 `pkill -f`。运行中的 bash 脚本不要被 `git pull` 改掉（`cosmos_full.sh` 一开始就 `exec` 成 python，可以 pull）。
- 两个 lane 实例（全量 + stage）共用调度表的一行，靠 `CARLA_IDX_OFF` / `CARLA_SPAN` 分 server index 子块，不要让两个实例用同一个子块。
- 每卡 VRAM：CARLA（带 Cosmos 相机）约 6.5 GB / 个，Cosmos worker 稳态约 30 GB，遇到新 prompt 时文本编码器临时上卡再多约 16–20 GB（stage 1 峰值 50.5 GB）。每卡 6 个 CARLA + 1 个 Cosmos 稳态约 60 GB，实测峰值 65 GiB；文本编码缓存已预热（`runs/cosmos_full/te_cache`，stage 目录里是指向它的软链）。
- 渲染 QC 只比 x⁺ / x⁻ 两边的亮度，两个世界同时出同一种故障看不出来（已登记为接受的漏检）。另一个 Opus 在 GPU 6 上查渲染故障根因；如果 main 转来修正（比如固定曝光），只影响故障 clip 的，用 `COSMOS_RGB_ATTRS` 重开驱动（之后的第 2 遍与重渲都带上），已标记的对会自动重渲一次；会改变正常帧的，先报 main。路灯修正已经用 `B2D_KEEP_STREET_LIGHTS=1` 上了（11:43 起），别忘了每次重开都要带。
- `runs/cosmos_full/lane/DRAIN` 存在时驱动不会开新的 CARLA 调用，重开前要删掉。
- 不要用 `pgrep -f` / `pkill -f`；用 tmux pane 的子进程或 pid 文件。
