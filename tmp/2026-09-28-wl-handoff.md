# WL 全量生成：交接说明（2026-09-28 13:35 CST）

给接手看守 WL 全量的 agent（Sonnet 看守，需要判断的交回 Opus / main）。设计与登记在 [todos/2026-09-28-wm-loop.md](../todos/2026-09-28-wm-loop.md)，
第 3 级结果和 13:30 的 checklist 修正也在那里。这里只讲怎么看、什么时候叫人。

## 现在的状态

- box tmux 窗口 `jev:wl-full`，脚本 `scripts/wl_full.sh`，13:30 启动。
- 启动时 `sch_table.py check` 返回 1（GPU 6 显存 77–81 GB 超过 75，GPU 6 不是本 lane 的卡），脚本每 5 min 重查一次，每 30 min 在 STATUS.md 写一行 `waiting: ...`；
  check 通过后才写 demand 文件、开始生成。GPU 6 的显存由 G lane 那边处理。
- 已完成：第 3 级 259 个 run（全量会自动跳过），停止线计数从 20 worker·h 起算（`runs/wl/pipe/gen_wh`）。

## 链怎么跑

`wl_full.sh` 按顺序做这几步，每步往 `$DATA_DIR/runs/wl/pipe/STATUS.md` 追加一行：

1. 等 `sch_table.py check` = 0。
2. 写 `runs/sched/demand/wm-loop.json` = `{"4": 6, "5": 6}`。G lane（`scripts/nq4_g_lane.py`）读到后在 route 边界 drain 自己在 GPU 4、5 上的 runner，不杀 route。
3. `STAGE=full SETS="ba p6 d2" scripts/wl_gen.sh`：每张卡一条 chain，每个 runner 起之前等到本卡 CARLA 总数 + 本 runner 的 6 个 ≤ 6（G lane 让干净了才起）。
   卡、编号段、核全部取 SCH 表 `wm-loop` 行（GPU 4:60、5:80，span 20，核 184–207）。
   同时一个守护循环每 5 min：记本 lane 在线 CARLA 数到 worker·h；磁盘 < 150 GB 或 worker·h ≥ 290 就 `touch runs/wl/gen/DRAIN`（runner 不再领新 route，跑完手上的退出）并写 ERROR；
   90 min 没有新 run 完成写 `STALL`（有新 run 完成时自动删掉）；每小时写一行 progress，并跑一次前缀剔除检查（`python -m jevdrive.wl drops`，超 5% 同样 drain + ERROR）。
4. 生成结束：每个集合数没有 done 记录的 run，harness 失败 > 5% 就 ERROR；再跑一次剔除检查（按集合不设下限）；全量 checklist 写到 `runs/wl/pipe/sanity_full.json`（描述，不判）。
5. demand 缩到 `{"4": 2}`，在 GPU 4 上跑 `wl_pipeline.sh STEPS="index opspec op"`；openpilot `temporal` 出来后再跑一次剔除检查（这次含余弦）；再跑 `STEPS="index vjepa z"`。
6. 删 demand 文件，写 `DONE`。

结束标志：`runs/wl/pipe/DONE`（全部完成）或 `runs/wl/pipe/ERROR`（内容是原因）；`STALL` 是提醒，不是结束。
脚本是可续跑的：`wl_gen` 跳过有 done 记录的 run，`wl_pipeline` 跳过已有的特征块，`gen_wh` 接着累加。出错修好后在同一窗口名重开即可（先关旧窗口）：

```
ssh autodl 'tmux kill-window -t jev:wl-full; cd ~/data/jev-drive && scripts/tmux_run.sh wl-full scripts/wl_full.sh'
```

## 崩溃与重试

- 每个 route 在一个 runner 里最多 3 次尝试（`--max-attempts 3`），每个 chain 对每个集合跑两遍（pass 2 补 pass 1 没跑成的）。
- 第 3 级的实测：279 次尝试里 17 次装图时 `server_died_rc139`、3 次 `hung_no_tick_progress_600s`，重试后 harness 失败 0。尝试失败率 7% 左右是正常的，不用报。
- CARLA 启动走 box 级的 start slot（最多 2 个同时启动，`b2d_run` 默认），server 用精简线程池（默认）。
- 看某张卡的 route 结果：`$DATA_DIR/runs/wl/gen/<set>/events.jsonl` 里 `kind == "route_end"` 的 `status`。

## 什么时候删 demand 文件

脚本自己会删：正常结束（第 6 步）和任何 ERROR 路径都会删。只有一种情况要手动删：窗口被人为关掉或脚本被杀，没走到收尾。
这时 `rm $DATA_DIR/runs/sched/demand/wm-loop.json`，并把 SCH 表 `wm-loop` 行的 status 改成 waiting（`python3 scripts/sch_table.py grant wm-loop --status "waiting ..."`），G lane 下一轮就拿回卡。

## 手动停

`touch $DATA_DIR/runs/wl/gen/DRAIN`，runner 跑完手上的 route 自己退出。**不要 `pkill -f` / `pgrep -f`**；非停不可时只按 `runs/wl/gen/pids.txt` 里的精确 PID 停。

## ETA

- check 通过、G lane 让出两张卡之后开始算：剩约 2 555 个分叉 run × 0.073 worker·h ≈ 186，加 D2 380 个 × 约 0.1 ≈ 38，合计约 225 worker·h；12 个 worker 约 **19 h** 墙钟。
- 之后特征约 2–3 h（openpilot 流很快；V-JEPA 2 按 150 clip/s 约 1–2 h）。
- 停止线 290 worker·h 对应约 22 h 纯生成墙钟；如果 progress 行显示 worker·h 走得比 run 数快（每个 run 远超 0.073），提前报。

## 怎么看（信号触发，不要每分钟轮询）

一个后台阻塞等待就够，box 端循环，只在 DONE / ERROR / STALL 出现时退出：

```
ssh -o ServerAliveInterval=60 autodl 'P=/root/autodl-tmp/ujs/runs/wl/pipe; until [ -e $P/DONE ] || [ -e $P/ERROR ] || [ -e $P/STALL ]; do sleep 120; done; ls $P; tail -5 $P/STATUS.md'
```

ssh 断了就重新挂上。另外最多每 1–3 h 看一眼 `tail -3 runs/wl/pipe/STATUS.md`，正常就不报。

## 生成之后：哪些是机械的，哪些要 Opus

机械的（脚本里已经包含，失败时照 STATUS.md 的命令重跑即可）：`index`、`opspec`、`op`、`vjepa`、`z`、剔除检查，以及把小表拉回 Mac：

```
scp autodl:/root/autodl-tmp/ujs/runs/wl/drops.json research/results/wl/drops_full.json
scp autodl:data/jev-drive/research/results/wl/{runs_full.csv,sanity_full.json} research/results/wl/
```

要 Opus 的（不在链里，需要判断）：
- `wl_pipeline.sh STEPS="outcomes train report"`：登记的第 5 级是先 C1 的单 seed、单折 smoke 只看训练 loss，再 3 seed 全部读数，要有人看 smoke 再放行；
- C1–C3 读数与判格、写进 todo 与 `research/decisions.md`；
- 下一阶段（在世界模型里训练 openpilot 的动作）的登记，取决于 C1 和 [op-adapt](../todos/2026-09-28-op-adapt.md) 的探针。

## 什么时候叫人（报给 main）

- `ERROR` 出现（原因写在文件里：停止线、磁盘、harness > 5%、前缀剔除 > 5%、某一步失败）；
- `STALL` 出现且 30 min 后仍在；
- STATUS.md 的剔除检查里 P6 或 BA 的剔除比例持续上升（每小时一行），哪怕还没到 5%；
- check 一直不过、`waiting` 超过 2 h（生成还没开始）；
- `DONE`：报完成，交 Opus 做训练和读数。
