> **2026-09-29 18:45 已结束**：判格写在 todo 执行记录最后一条与 decisions 第 65 条；本文件只作历史。

# carla-rewind 交接状态（2026-09-29 约 14:45 CST，写给接手的 agent）

登记与判据：[todos/2026-09-29-carla-rewind.md](../todos/2026-09-29-carla-rewind.md)（R1–R5，地图复用 U0–U3 / UR，zygote Z，都写在数字之前）。
代码：`scripts/carla_rewind.py`（快照 / 回退，方法 `poc`、`teleport`、`tree`、`respawn`，每种可加 `+w<N>` / `+w<N>v` warm-up）、`scripts/wl_fork_agent.py`（`rewind` job 键 / `wl_rewind`，默认关）、
`scripts/rewind_eval.py`（prep / cut / eval / opspec / opcos / cost / report）、`scripts/rewind_gen.sh`（env GEN、OUTGEN、IDS、SETS、WORKERS、IDX0、IDX_SPAN、ZYGOTE）、
`scripts/b2d_mapreuse.py`（`B2D_PHASES`、`B2D_REUSE_MAP`，从 b2d_route 装上）、`scripts/b2d_run.py --zygote` + `scripts/b2d_route.py` 的 `B2D_ZYGOTE`（预热的 route 进程），
`scripts/rewind_physics_probe.py` / `rewind_probe.sh`（独立 server 上的物理 probe；server 在装图时 Signal 11 两次，没跑出来，已改成在 route 里原地测）。
box 数据：`$DATA_DIR/runs/rewind/`（plan.parquet、jobs.json、routes-*.xml、gen/、gen_old/（旧代码的 warm-up 试跑）、gen_smoke/、branches/、eval/）。

## 正在跑

- tmux `jev:rw-gen`：`GEN=gen`，IDS = poc + `tree+w10` + `tree+w20` 全 11 个分叉点 + 14 个 floor 从头 run，GPU 6，3 个 server（index 440–442）。
  PID：bash 197179 → rewind_gen.sh 197182 → b2d_run 197610。停：`kill -TERM 197182 197179; kill -INT 197610`（按精确 PID）。约 14:45 时 BA 已完成 53 个（含旧的），P6 还没开始。
- 调度行 `carla-rewind`：GPU 6、3 worker、idx 440 span 3、核 184–207（14:10 从 GPU 5 迁来）。main 说 op-drive 会暂停、GPU 6 会让出，届时改成 6 个 worker（`sch_table.py grant carla-rewind --workers 6 --span 6`，先 check 冲突）。

## 已经看到的（box `runs/rewind/eval/`，`rewind_eval.py report`）

- **floor = 0**：同配置从头重跑（每个回退 run 的第 0 分支 + floor run），ego、hazard、标签与 WL 真值逐位相同。所以判据线退化为固定下限（R1 0.25 m、R2 0.3 m / 0.3 m/s / 2°）。
- **提案的 `poc` 与无 warm-up 的 `teleport` / `tree` / `respawn` 全不过**：`set_transform` + `set_target_velocity` 之后 ego 在交接那一刻就差 0.5 m/s（中位），0.5 s 时速度差 p95 3–5 m/s，3 s 时位置差 p95 3.6–8.9 m；
  hazard 行人「开始走」的时刻一致率 0–16%；unsafe 一致 70–92%（`poc` 70%、collision 73%）。POC 表里自己的 64.1 vs 58.7 km/h 就是这个传动系状态没恢复。
- **warm-up 修好了 ego 与 cut-in**：`tree+w20` 的 ego 3 s 位置差 p95 0.33 m、速度差 p95 0.68 m/s，cut-in 车 p95 0.30 m，unsafe / collision 一致 100%（4 个分叉点、24 个分支）；`w10` 稍差（ego p95 1.1 m）。
  最后一步只设速度、不 teleport 的 `+w10v` 更差。
- **行人还不过（R1 不过）**：`tree+w10/w20` 的行人开始走的一致率只有约 53%，行人位置 p95 约 4 m。现象：回退后行人的状态**跟上一个分支走**——上一个分支里行人走了，下一个分支一开始就以满速走（没有加速过程）；上一个分支没走，这个分支就不走（真值会走）。
  fork 78 的旧版 `tree+w10`（`gen_old`，warm-up 期间树自由跑、只在最后一 tick 恢复）行人 3/3 对；新版（warm-up 每个 tick 都 `_restore_python`、第 0 个 warm-up tick 就 `restore()` 重生）坏了。
  嫌疑在这两处改动；另一个线索：行人 teleport 后移动组件的速度不归零（apply_control speed 0 不能立刻停下）。**下一步先用 `CARLA_REWIND_DEBUG=1` 在 fork 78 / 42 上各跑一次 `tree+w20`**
  （`OUTGEN=gen_dbg GEN=gen IDS=70078060,70042060 CARLA_REWIND_DEBUG=1 WORKERS=1 IDX0=442 IDX_SPAN=1 scripts/rewind_gen.sh`，读 rewind.jsonl 的 debug 行：行人 id / 速度 / ctl_speed、Crossing 序列的 current_index），
  然后试：回到「warm-up 期间树不恢复、只在 InRouteTest 那种会结束 route 的判据上兜住」（例如 warm-up 期间把 scenario_tree 的 tick 换成 no-op，比每 tick thaw 更干净），以及在最后一步对行人 destroy + respawn（新生成的行人移动组件速度为 0）而不是 teleport。
- 另修过的坑：非 ego 车辆的最后一次控制会留在 server 上（cut-in 车回退后带着油门开走）→ 现在快照 / 恢复所有非 ego 车的 VehicleControl（`poc` 保持提案原样不恢复）；
  warm-up 期间 InRouteTest 看到跳回会失败并结束 route → 现在 warm-up 期间保持 Python 状态。

## 成本（实测，WL 全量 2 814 个 run 的 route.log / done 记录，`runs/rewind/wl_cost.csv`、`wl_phases.csv`）

| BA 每 run（均值） | 秒 |
|:--|--:|
| 进程启动 + import（到 LEAD 的 logging 行） | 18 |
| load_world + RouteScenario 构建 | 60（中位 69；P6 39） |
| tick（473 tick × 0.21 s） | 98（前缀约 110 tick ≈ 23 s，3 s 分支 ≈ 13 s，20 s 续跑 ≈ 62 s） |
| 收尾 | 5 |
| 合计 | 181 |

- 提案说「192 s 里 160 s 是前缀」不对：前缀只有约 5 s 模拟时间、约 23 s 墙钟；装载约 60 s、import 约 18 s，续跑约 60 s（续跑依赖分支，回退省不掉，除非回退连 expert 状态一起恢复）。
- 每 tick 分解（BA 中位，ms）：server `world.tick` 27、传感器等待 + 传输（三路 1088×1560 相机，在 agent 里）约 100、存图 49（每个相机 tick，折合约 12 / tick）、expert 11、scenario 树 8。剩下的瓶颈是渲染 / 传输三路大图。
- 回退 run 实测：warm-up 20 的 7 分支 run 墙钟 136–315 s（Town05 136 s）；从头 7 × 181 ≈ 1 270 s。不带续跑时可比的从头成本 ≈ 7 × (18 + 60 + 23 + 13 + 5) ≈ 830 s；回退 w20 ≈ 18 + 60 + 23 + 7 × 13 + 6 × 20 × 0.2 + 5 ≈ 220 s（约 3.8×）。`rewind_eval.py cost` 出逐 run 表。
- Town05 路线的 phases.json：import 21 s、load_world 6 s、构建 3 s；Town12 的 load 大头要等 gen_reuse / phases 的数。

## 下一步（按顺序）

1. 修行人（上面的 debug 与两个试法），在 78、42、60、96 上验，再在 11 个分叉点上跑最终方法（改 `rewind_eval.py` 的 METHODS / UR_METHOD，`prep`）。
2. 地图复用 U + zygote：`GEN=gen_reuse ZYGOTE=1 scripts/rewind_gen.sh`（B2D_REUSE_MAP 自动开；按 town 排序），Z：`GEN=gen_zyg ZYGOTE=1 scripts/rewind_gen.sh`。U 的 UR 行用最终方法。
3. openpilot：`rewind_eval.py opspec`，再 `CUDA_VISIBLE_DEVICES=6 P5_SET=rewind OMP_NUM_THREADS=2 $DATA_DIR/envs/openpilot/bin/python scripts/p5_openpilot.py --models cinque --arrays temporal --out-sub op_streams --workers 10`（taskset 184-207），然后 `opcos`、`report`（R4）。
4. `cut` → `eval` → `report` → `cost`；结果写进 todo、`research/decisions.md`、`research/carla-rewind-branching.md` 末尾的更正节（不改原文）；过了再写 docs 段落（何时准确、何时不准）并把默认关的选项说明补全。
5. 收尾：`tmux kill-window -t jev:rw-gen`，调度行 `sch_table.py finish carla-rewind`。
