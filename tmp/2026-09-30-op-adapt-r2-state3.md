# op-adapt r2：状态交接三（2026-09-30 18:40，box 暂时没有 GPU）

**状态：v5 已写、V4 重跑过、关卡全过；M1 没有成功跑出任何东西，没有训练。** 没有任何后台等待或触发器在运行（box 上 tmux 服务已不在，没有 op_adapt 进程）。

## 已完成

| 项 | 结果 |
|:--|:--|
| v5 修订（prereg「v5 修订」节，commit 0cba201，先于重跑） | 去掉偏离起点 slot 集（V6、`rej`、B-off、孪生帧）；P 规则改成「移动挡路者 → 全部 P = 1」；静止挡路者 dwell gate T_w = 5 s；不加绕行候选；V4 口径不变 |
| V4 重跑（唯一一次） | simC 0.950（1 029 slot）、simK 0.954（1 027），线 0.90，过；按实例平均 0.987 / 0.989（描述、事后） |
| 其余检验 | V3 sim 72.2% / 74.9%，V3-P5 0.790，V2 0.980，V5a 1.0，V5b 0.9947，全部过（`checks/gate.json`）；V1 仍是 D1 |
| 旧表 | `score/run1/`、`checks/run1_v346/`（未删）；新表在 `score/{simC,simK,nus,p5}.npz` |
| 代码 | `op_adapt_score.py`（P 规则、`Actors.still`）、`op_adapt_score_data.py`、单测 12 项过；训练 / 读数 / lane 里偏离 slot 集已关（`Arm.offset` 默认 False，`TRAIN_DOMAINS`、`DOMAINS`、readout 的 `offdev` 去掉）；链脚本 `scripts/op_adapt_r2_chain.sh` |

## M1 的 ERROR 不是缩容造成的

18:24 的 `m1_rater failed` 发生在 box 缩容之前，是代码 bug：`cmd_rater` 用了 `op_plan.json` 的 span（只有 WOD 子集），rater 帧的历史不在里面（`KeyError '…-129'`）。已改成读 `Z.root()/sets.json` 的 span（commit 见下）。同时发现 `m1_eval` 那一步其实什么都没做：所有读数域都「not packed, skipped」，因为 pack 与 teacher 只在 lane 的 `prep` 里做。链脚本已加 `prep`（新 lane 子命令）、swap 域 pack、rater，顺序：gate → prep → m1_rater → m1_swap → m1_eval → m1_read → m1_nav → stage1 →（PAUSE 停在 stage10 之前）→ stage10 → full。**这些修改和 rater 修复都还没在真实数据上试过**（`rater`、`pack simC_swap …`、`read`、`navsim navtest` 都是第一次在真实集上跑，出错要按报错修；这是登记里预期的）。

## 恢复前要清的

box 上 `R=$DATA_DIR/runs/op_adapt_r2`：

1. `rm $R/chain/m1_eval.ok $R/chain/ERROR`（`m1_eval.ok` 是空跑留下的假标记，必须删；`m1_rater.log` 可留作记录）。
2. `chain/PAUSE` 现在内容是 `stage10`（链只在 stage10 之前暂停，main 说过 stage 1 之后要等 main）；缩容期间想让链一个新步骤都不启动，`echo > $R/chain/PAUSE`（空内容 = 任意步骤前暂停）。
3. `echo <两张卡的编号> > $R/chain/GPUS`（现在是 `0,6,4`；链会丢掉不存在的编号，都不存在就用现有全部卡）；`CORES` 现在 `48-95`，主机现有 `nproc` = 208、内存 754 GB（free 348 GB），核心数不变就不用改（无效列表会回退到全部）。
4. box 上 repo 落后 origin 两个 commit，先 `cd ~/data/jev-drive && git pull --ff-only`（此时没有 r2 进程，可以拉）。
5. 调度表 `op-adapt-r2` 行现在写着 GPUs 0,6,4，缩容后改成实际的卡：`python3 scripts/sch_table.py grant op-adapt-r2 --gpus <ids> --cpus 48-95 --status "..."`。

## 恢复命令（缩容后，2 张卡）

```
ssh autodl 'C=$DATA_DIR/runs/op_adapt_r2/chain; rm -f $C/m1_eval.ok $C/ERROR; echo "<id1>,<id2>" > $C/GPUS; echo stage10 > $C/PAUSE; cd ~/data/jev-drive && git pull --ff-only && scripts/tmux_run.sh r2-chain scripts/op_adapt_r2_chain.sh'
```
链读 `.ok` 标记续跑，出错写 `chain/ERROR`，暂停写 `chain/READY_FOR_RESIZE`。之后要放行 stage 10：`rm $C/PAUSE $C/READY_FOR_RESIZE` 并重新启动同一命令。

## 估计（2 卡）

M1 含 navtest 与 rater 约 1.5 h；stage 1 约 1 h（含 prep）；stage 10 约 1 h；full 约 13 arm-seed × 约 2 GPU·h（RTX 6000D 约老卡 55% 算力，估计，没测），约 13 h。合计约 16 h，范围 12–18 h。每个 run 约 27 GB，一张卡放 3 个在 78 GB 之内（stage 1 的实测显存出来后再改）。

## 其他

- 约束照旧：PID 总数 < 17.5k、按精确 PID 停进程、不删数据、不 pull 到运行中的脚本上、每个 devkit 进程 `OPENBLAS_CORETYPE=Haswell`、每次看 score 表记日志。
- 教训：tmux 登录 shell 里有 `WORKERS=2` 环境变量（脚本里用 `R2_WORKERS`）；今天 P5 的 re-score 因此白跑了约 1 h。
- 临时目录（box `/tmp/r2synth`、`/tmp/r2real`、`/tmp/r2unit`、`R2/equiv/`、`R2/swap-pilot-diff30/`、`/tmp/r2code`、`/tmp/r2diag*`）未动。
