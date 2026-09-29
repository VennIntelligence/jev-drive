# Cosmos G4 全量生成：交接（2026-09-29，随阶段更新）

最后更新：2026-09-29 10:40 CST。登记、规则与读数都在 [todos/2026-09-28-cosmos-pilot.md](../todos/2026-09-28-cosmos-pilot.md) 的「全量生成」节；决策是 [decisions](../research/decisions.md) 第 56 条。

## 现在处于哪一步

- **stage 1（1 对）**：跑完第一对 `c000v00`（Town12 DynamicObjectCrossing，实例 0，v0）。确定性 0.000 m，QC 亮度差中位 0.01，GT 可见 22 帧，Cosmos x⁻ 116 s + x⁺ 107 s = 223 s（v2 是 213 s），controls 50 s。人工看图、eval 还没做。
- **全量 CARLA 已经开跑（Cosmos 未开）**：main 10:2x 要求立刻用空闲卡跑全量 CARLA 渲染，Cosmos 等 10 对 checklist 过了再开（门控文件 `lane/COSMOS_GO`）。
- **stage 10（10 对，铺在 GPU 0–4 上）**：还没启动。

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
| `cosmos-full`（pane 32530，驱动 python 32534） | 全量：CARLA 两遍 + controls；Cosmos 等 `COSMOS_GO` | `cosmos_full/` | `cosmos_full/lane/{STATUS,DONE,ERROR,CARLA_DONE,CONTROLS_DONE}` |
| `cf-stage1`（pane 20171，驱动 20175） | stage 1，收尾中 | `cosmos_full/stage1/` | `cosmos_full/stage1/lane/…` |

每次 CARLA 调用的 b2d_run PID 在 `<root>/carla_logs/inv<NNN>-gpu<g>.pid`，日志同名 `.log`；Cosmos worker 的 PID 在 `<root>/cosmos_logs/<slot>.pid`。
进度：`<root>/lane/STATUS`（每次调用一行，Cosmos 阶段每小时一行）；机器可读 `runs/cosmos_full/lane/<时间戳>/events.jsonl`。
状态表：`variants.csv`（排了哪些变体）、`sel.csv`（选窗结果）、`ctl.csv`（controls / QC 结果），`pairs/<pair>/done.json`（Cosmos 完成）。

## 等待命令（box 侧阻塞，一个阶段一次）

```bash
ssh autodl 'L=$DATA_DIR/runs/cosmos_full/lane; timeout 10800 bash -c "until [ -e $L/DONE ] || [ -e $L/ERROR ]; do sleep 60; done"; tail -5 $L/STATUS'
```

## 下一步

1. stage 1：看 `c000v00` 的并排图，跑 `scripts/cosmos_full_check.sh stage1 1`（注意：stage 1 的 worker 是旧代码，G4b npy 在 `stage1/eval/out/G4b/`，先 `mv` 到 `stage1/out/G4b/`）。
2. 预热文本编码缓存（避免 worker 在卡上临时多占 16 GB）：`python -m jevdrive.cosmos_full prompts` 写 `te_prompts.txt`，再用 worker 的 `--warm` 在一张卡上算一遍（代码待提交）。
3. stage 10：`COSMOS_FULL_DIR=cosmos_full/stage10 CARLA_W=0:1,1:1,2:1,3:1,4:1 CARLA_IDX_OFF=0 CARLA_SPAN=6 COSMOS_SLOTS=0,1,2,3,4 COSMOS_GO=1 scripts/cosmos_full.sh --target 10 --insts 0,68,123,167,11,42,7,99,4,96 --keep-npy`（tmux 窗口 `cf-stage10`），跑完 `scripts/cosmos_full_check.sh stage10 1`，按 todo 的 checklist 判。
4. checklist 过：`touch $DATA_DIR/runs/cosmos_full/lane/COSMOS_GO`，全量的 5 个 Cosmos worker 自己起来。不过：停批（`touch …/lane/DRAIN`），报 main。

## 每种信号怎么办

- `DONE`：`python -m jevdrive.cosmos_full summary`，把 `research/results/cosmos/full/`（variants.csv、summary.json）拉回 Mac 提交，todo 执行记录与 decisions 第 56 条补结果，报 main。
- `ERROR`：看 `ERROR` 内容与 `STATUS` 末尾。驱动出错时会自动 `touch lane/DRAIN`（在跑的 route 和 Cosmos 对做完就停）。修好后 `rm lane/DRAIN lane/ERROR`，用同样的命令重开，已完成的全部跳过。
- 卡 / 盘：盘低于 250 GB 时驱动自动暂停新的 CARLA 输出，等 Cosmos 追上；低于 150 GB 报 ERROR。

## 坑

- 停进程只按 PID（上面的 pid 文件），不要 `pkill -f`。运行中的 bash 脚本不要被 `git pull` 改掉（`cosmos_full.sh` 一开始就 `exec` 成 python，可以 pull）。
- 两个 lane 实例（全量 + stage）共用调度表的一行，靠 `CARLA_IDX_OFF` / `CARLA_SPAN` 分 server index 子块，不要让两个实例用同一个子块。
- 每卡 VRAM：CARLA（带 Cosmos 相机）约 6.5 GB / 个，Cosmos worker 稳态约 30 GB，遇到新 prompt 时文本编码器临时上卡再多约 16–20 GB（stage 1 峰值 50.5 GB）。每卡 6 个 CARLA + 1 个 Cosmos 稳态约 69 GB，临时峰值会超；所以要先预热文本编码缓存。
- 渲染 QC 只比 x⁺ / x⁻ 两边的亮度，两个世界同时出同一种故障看不出来（已登记为接受的漏检）。另一个 Opus 在 GPU 6 上查渲染故障根因；如果 main 转来修正（比如固定曝光），只影响故障 clip 的，用 `COSMOS_RGB_ATTRS` 重开驱动（之后的第 2 遍与重渲都带上），已标记的对会自动重渲一次；会改变正常帧的，先报 main。
- `runs/cosmos_full/lane/DRAIN` 存在时驱动不会开新的 CARLA 调用，重开前要删掉。
