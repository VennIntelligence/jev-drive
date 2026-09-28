# NAVSIM 全量跑交接（2026-09-29）

上一个 agent 的 context 到了 290k，交接给一个新的 Sonnet。这份文档要让接手的人不看别的上下文就能把任务跑完。

## 任务背景（一句话）

在 openpilot 开环补帧研究（[research/openpilot-openloop-integration.md](../research/openpilot-openloop-integration.md)）
2000-token navtest 子集诊断（结论：hold 51.9 → GIMM-VFI 84.7 / ego-motion warp 83.1 PDMS）之后，跑默认 pipeline
（Cinque 原生 plan、不 retime、只 base 适配器：杠杆臂变换 + 线性重采样）在**全量** navtest（12146 token，v1 PDMS）和
**全量** navhard_two_stage（5912 token，v2 EPDMS，v1.1 devkit 没有这个 split）上，两种补帧器都跑。优先级低
（"反正空着也是空着"），跑完把结果写进 doc 和 `research/results/op-interp/`，并在 [decisions.md](../research/decisions.md)
第 37 条追加一行（那条已经有 hold 输入的全量数字：navtest PDMS 52.1 / EPDMS 46.2，navhard EPDMS 9.3，Cinque）。

## 现在跑着什么（box 上，tmux 会话 `jev`）

用 `ssh autodl` 登录（sshd 偶尔拒连，重试 + 等几秒）。`DATA_DIR=/root/autodl-tmp/ujs`。

| 窗口 | 在干什么 | run dir | 卡 / 核 |
|---|---|---|---|
| `opi-navfull-score` | navtest 全量的补算：re-score（PDMS）+ nav-report + by_command | `$DATA_DIR/runs/op_interp/navfull` | CPU 96-117,120-132,134-138 |
| `opi-navhard` | navhard_two_stage 全量：`scripts/op_interp_full.sh navhard_two_stage` 整条链（nav-cache → warp/GIMM synth → run → export → score → report） | `$DATA_DIR/runs/op_interp/navhard` | GPU 2（≤15GB）+ 同一批 CPU |

调度表（`python3 scripts/sch_table.py show`）里已经登记了两行：`op-interp-warp`（CPU，`96-117,120-132,134-138`）、
`op-interp-gimm`（GPU 2，`170-171,174-176`，≤15GB）。不用重新 grant，除非这两行被别人 revoke。

**navtest（`navfull`）的状态**：预测已经算完并核对过（`preds/{warp,gimm_g0.2}-cinque__base.npz`，`poses.shape=(12146,8,3)`，
token 数对得上）——这部分不用重跑。当前 `opi-navfull-score` 窗口在补跑打分（这步之前因为两个 bug 没做对，见下）。

**navhard（`navhard`）的状态**：2026-09-28 23:23 刚启动，此时只做完 `nav-cache`。全新一条链，从头跑。

## 信号文件（不要 poll，一次阻塞等待）

```bash
# navtest 补算（score + report + by_command）
until [[ -f $DATA_DIR/runs/op_interp/navfull/SCORE_DONE || -f $DATA_DIR/runs/op_interp/navfull/SCORE_ERROR ]]; do sleep 180; done

# navhard 全链
until [[ -f $DATA_DIR/runs/op_interp/navhard/DONE || -f $DATA_DIR/runs/op_interp/navhard/ERROR ]]; do sleep 180; done
```
两个可以合并成一条 ssh 命令、一次挂后台（`run_in_background: true`），例如：
```bash
ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 autodl '
until [[ -f $DATA_DIR/runs/op_interp/navfull/SCORE_DONE || -f $DATA_DIR/runs/op_interp/navfull/SCORE_ERROR ]]; do sleep 180; done
until [[ -f $DATA_DIR/runs/op_interp/navhard/DONE || -f $DATA_DIR/runs/op_interp/navhard/ERROR ]]; do sleep 180; done
echo BOTH_DONE
[[ -f $DATA_DIR/runs/op_interp/navfull/SCORE_ERROR || -f $DATA_DIR/runs/op_interp/navhard/ERROR ]] && exit 1
exit 0
'
```
出错就看对应 tmux 窗口（`tmux capture-pane -t jev:<窗口名> -p -S -200`）和 `$R/STATUS` / `$R/score.log` / `$R/report.log`。

## ETA（粗略，按 navtest 12146 token 实测速度线性换算到 navhard 5912 token）

- navtest 补算：只是打分（CPU，官方 devkit，ray 16 线程），navtest 12146 token 量级预计几十分钟；export/report/by_command 各 ~1 分钟。
- navhard 全链：warp synth（CPU）~2-3 min；GIMM synth（GPU2）navtest 12146 token 实测约 2h39min，按比例 navhard 5912 token 约 **1.3 h**；
  run（openpilot 推理，两种插帧）navtest 实测 ~36 min，navhard 约 **17 min**；export/score/report 未知（v2 EPDMS 用 `run_pdm_score.py`
  reactive two-stage 打分，可能比 v1 one-stage 慢，没有实测数据，留出余量）。总计粗估 navhard 从 23:23 起 **2-3 小时**内跑完，
  也就是大约 2026-09-29 01:30-02:30 CST。

## 打分命令（如果哪一步需要手动重跑）

```bash
CPUS="96-117,120-132,134-138"
PJ="taskset -c $CPUS $DATA_DIR/envs/jevdrive/bin/python scripts/op_interp.py"

# navtest（v1 PDMS）
scripts/op_interp_score.sh "$CPUS" navfull v1 navtest
$PJ nav-export --data navfull --adapters base
$PJ nav-report --data navfull --ver v1 --split navtest --refs warp-cinque__base
taskset -c "$CPUS" $DATA_DIR/envs/jevdrive/bin/python scripts/op_interp_by_command.py --data navfull --ver v1 --split navtest

# navhard_two_stage（v2 EPDMS）—— 正常情况下 scripts/op_interp_full.sh navhard_two_stage 会自己做完这几步，
# 这里只是手动重跑单步时用
scripts/op_interp_score.sh "$CPUS" navhard v2 navhard_two_stage
$PJ nav-export --data navhard --adapters base
$PJ nav-report --data navhard --ver v2 --split navhard_two_stage --refs warp-cinque__base
taskset -c "$CPUS" $DATA_DIR/envs/jevdrive/bin/python scripts/op_interp_by_command.py --data navhard --ver v2 --split navhard_two_stage
```
结果落在 `$DATA_DIR/runs/op_interp/{navfull,navhard}/{results.csv,report.log,by_command.txt}`。

## 结果表放哪、怎么排版

1. **`research/results/op-interp/`**（Mac 上，小文件）：
   - `navtest_full_results.csv` ← 从 box 拉 `runs/op_interp/navfull/results.csv`（只留 `gimm_g0.2-cinque__base`、
     `warp-cinque__base` 两行核心结果 + `exam human`/`exam cv`/`exam cinque_none`（hold 参照）即可，别把全部消融行搬过来）。
   - `navtest_full_by_command.txt` ← `runs/op_interp/navfull/by_command.txt`。
   - `navhard_full_results.csv` / `navhard_full_by_command.txt` ← 同理从 `runs/op_interp/navhard/`。
   - `拉取方式`：`scp` 不用，直接 `ssh autodl cat <path>` 重定向写本地文件（数据小，几十行）。

2. **`research/openpilot-openloop-integration.md` 第 9 节**（已经占好位置，搜索 `## 9. 全量跑`，当前是占位段落
   "跑完后这里贴 PDMS / EPDMS 主表"）：换成两张小表，格式仿第 4 节：

   ```markdown
   **navtest 全量（12146 token，v1 PDMS）**：

   | 输入 | PDMS [CI] | NC | DAC | EP | TTC | + 2000-token 子集参照 |
   |:--|:--|--:|--:|--:|--:|--:|
   | GIMM-VFI | xx.x [lo, hi] | .. | .. | .. | .. | 84.7（第 4 节，n=2000） |
   | ego-motion warp | xx.x [lo, hi] | .. | .. | .. | .. | 83.1（第 4 节，n=2000） |
   | 参照：hold（decisions.md #37，Cinque） | 52.1 | | | | | 51.9（第 0 节） |

   直行 / 左转 / 右转 / 起步（by_command.txt）：

   | | 直行 | 左转 | 右转 | 起步 v0<1 |
   |:--|--:|--:|--:|--:|
   | GIMM-VFI PDMS | .. | .. | .. | .. |
   | warp PDMS | .. | .. | .. | .. |

   **navhard two-stage 全量（5912 token，v2 EPDMS，v1.1 devkit 无此 split）**：

   | 输入 | EPDMS [CI] | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
   |:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
   | GIMM-VFI | .. | | | | | | | | | |
   | ego-motion warp | .. | | | | | | | | | |
   | 参照：hold（decisions.md #37，Cinque） | 9.3 | | | | | | | | | |
   ```
   subscore 列名对照见 `scripts/op_interp.py` 里的 `V1_SUBS`/`V2_SUBS`（v2 短名：NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC）。
   写完表，在表下面加 2-3 句读法（直行/转弯差多少、navhard 上收回了多少、和 hold 基线比涨了多少），跟第 2、4 节的文风一致。

3. **`research/decisions.md` 第 37 条**：不要新开条目，在现有表格下面加一行或一段"就地修正"风格的补充
   （参照第 37 条已有的"2026-09-26 就地修正"写法）：GIMM-VFI / ego-motion warp 补帧后，openpilot 原生 plan 在全量
   navtest 上从 hold 的 52.1 PDMS 涨到多少、navhard EPDMS 从 9.3 涨到多少，呼应"openpilot 原生 plan 的行基本是输入协议的读数"
   这句话——full-scale 数字进一步坐实（或修正）第 4 节 2000-token 子集的结论是否在全量上还成立。

## 参照数字（拿到新分数先跟这些比，量级对不上就是有 bug）

- navtest 2000-token 子集（同一套代码，seed 0）：hold 51.9、GIMM-VFI **84.7**、ego-motion warp **83.1**（PDMS，
  第 0/4 节）。全量 12146 token 的数字应该在这个量级附近（token 集合不同，允许有几分的移动，但不应该差出 10+ 分）。
- decisions.md 第 37 条，hold 输入、全量：Cinque PDMS **52.1**、EPDMS **46.2**、navhard EPDMS **9.3**；
  human PDMS 94.6 / EPDMS 94.5；cv PDMS 20.7 / EPDMS 25.9；navhard 上 cv EPDMS 11.5（这几个是 "exam" 参照行，
  `nav-report`/`op_interp_by_command.py` 会自动从 `runs/navsim/eval/{v1_navtest,v2_navhard_two_stage}_{cinque_none,cv,human,...}`
  读进来，不用手抄）。
- **合法性判断标准**：GIMM/warp 补帧后的 PDMS/EPDMS 必须显著高于同一 split 上的 hold 参照行（补帧本来就是为了修时间轴契约，
  第 4 节的效应量是 +25~34 PDMS），且不超过 human 上限。如果补帧后的分数反而低于或接近 hold，说明补帧管线在全量 token 上出了问题，
  先别写进 doc，回来查。

## 已知的坑（都是这次交接前踩过、已经修好的，但接手的人要知道历史，因为可能还有没踩到的同类问题）

这次任务里 `scripts/op_interp.py`/`scripts/op_interp_score.sh` 从只支持 2000-token navtest 子集，改成支持任意
`--split`/`--out`/`--data` 的全量跑（commit 历史：`op_interp: generalize nav-cache/nav-export/nav-report...` →
`fix(op_interp): keys_of() ignored --data...` → `fix(op_interp): eval-directory name collision...` →
`fix(op_interp_score): drop the per-token filter for full-benchmark run dirs...`）。改动过程中连续踩了三个坑：

1. **`keys_of()` 曾经硬编码 `root("nav")`**：不管 `--data` 传的是什么，都去读旧的 2000-token `nav/keys.npy`。
   已修（现在用 `root(data)`）。现象：`plan_pos` 数组长度跟 `meta.json` 的 token 数对不上，`nav-export` 里
   `IndexError: index N out of bounds`。**如果再遇到「某个数组长度跟 token 数对不上」的报错，先怀疑是不是又有函数
   忘了用 `root(a.data)` 而是写死了某个目录名** —— 搜 `root("nav"`/`root('nav'` 确认没有漏网的。
2. **不同 run dir 的同名 pose 文件（比如 `warp-cinque__base.npz`）打分时会撞同一个 eval 目录名**
   （`v1_navtest_opi_warp-cinque__base`），`op_interp_score.sh` 的「已打分就跳过」逻辑会把旧 run dir（`nav/`）的
   2000-token 分数当成新 run dir（`navfull/`）的结果，静默返回错的行（`n=2000`，数字跟子集一模一样）。已修
   （现在除了 `nav` 自己，其他 `--data` 的 eval 名都会加前缀，如 `opi_navfull_warp-cinque__base`）。
   **拿到 `results.csv` 先看 `n` 这一列，navtest 应该是 12146、navhard 应该是 5912，不是 2000。**
3. **给全量 token 集合传 `TOKENS_FILE`（`scene_filter.tokens=[...]` 命令行参数）会因为 12146+ 个 token 超出
   `ARG_MAX`，报 `Argument list too long`**。已修：只有 `nav`（2000-token 子集）还传这个参数，`navfull`/`navhard`
   之类全量 run dir 不传（本来就是打整个 split，不需要过滤）。

**还没验证过、可能有坑的地方**（navhard 是第一次真正跑到这几步）：
- `cmd_nav_report` 的 v2 EPDMS 分支（`V2_SUBS` 那几个 subscore 列名）没有实测过，列名是照抄
  `scripts/navsim_zs_report.py` 里的 `V2` 常量，理论上该对，但没有跑通过一次确认列名跟 `run_pdm_score.py`
  实际吐出的 CSV 列完全一致。
- `op_interp_by_command.py` 同理，v2 分支没跑过。
- navhard 的 `run_pdm_score.py`（reactive two-stage）比 navtest 的 `run_pdm_score_one_stage.py` 慢多少、内存/CPU
  占用多少，没有实测，ETA 里的打分时间是纯猜的。
- `runs/op_interp/navhard` 全链跑完后记得删掉大缓存（`warp.npy`、`gimm_g0.2.npy`、`keys.npy`，可再生），
  参照这次 navtest 的做法：`rm -f "$R"/{warp,gimm_g0.2,keys}.npy`，保留 `meta.json`/`tokens.txt`/`results.csv`/`report.log`/
  `by_command.txt`。

## 收尾

两条都 DONE 之后：
1. 按上面「结果表放哪」把 `research/results/op-interp/` 的四个文件、doc 第 9 节两张表、decisions.md 第 37 条更新好。
2. `git add` 只加自己动的文件（doc、results、这份 handoff 如果还留着就删掉或者在 PR 里说明），`commit` + `push`，
   再 `ssh autodl 'cd ~/data/jev-drive && git pull'`。
3. 清理两条 run dir 的大文件（见上一节最后一条）。
4. 报告给 main：full navtest / navhard 的 PDMS / EPDMS 主数字（GIMM、warp，各自跟 hold 参照的差），5 行以内，中文。
