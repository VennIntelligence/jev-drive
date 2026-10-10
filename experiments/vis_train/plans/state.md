# VT 工作状态（builder 与吞吐工程师共用；各写各的小节）

## 吞吐

（吞吐工程师维护；builder 请勿改这一节。）

### 接口（已定，照此写 trainer）

```python
import pixel_store as PX                      # lib/pixel_store.py
PS = PX.PixelStore(datas, dev, mode="t0")     # datas 与 pp_train.Store 相同的列表；rows = 拼接后的 tab.npz 行号
prev, cur = PS.t0(rows)                       # uint8 (B, 2, 6, 128, 256) ×2，已在 dev 上：slot 7 的图像对
prev, cur = PS.all(rows)                      # uint8 (B, 8, 2, 6, 128, 256) ×2：8 个 slot（prev[:, 0] 全零）
g = PS.frames(rows)                           # uint8 (B, 8, 2, 6, 128, 256)：8 帧本体；slot j 的对 = (g[:, j-1], g[:, j])，省一次 cat
row2, ok = PS.future(rows, k)                 # 同一 log 里 0.5·k 秒之后的行（B 臂：k = 2, 4）；ok=False 时 row2 = 本行
h = PX.fast_encode(net, prev, cur, grad=True) # (n, 2, 6, 128, 256) uint8 → (n, 32, 512)；net = A.load("cinque", ...)
```

- rows 可以是 numpy 或 tensor；`t0 / all / frames / future` 都可以在 `pp_train.Prefetch` 的 worker 线程里调用（读盘走 `os.preadv` 进 pinned 内存，再 non_blocking 上卡）。
- 缓存位置：`$DATA_DIR/runs/vis_train/px/<data>/frames.npy`，(N, 8, 2, 6, 128, 256) uint8，行序 = `cache/<data>/tab.npz`；`meta.npz` 存 names / log / ts。
- B 臂的 teacher：`S.front[row2][:, 7]`（`row2` 是同一 Store 的全局行号，直接索引现有 `front.npy` 缓存）。
- 解释器：box 上所有 torch / onnx 任务用 `$DATA_DIR/envs/op-train/bin/python`（base python 没有 onnx）。

### 进度

- 22:20 CST：`runs/navsim_zs/index/` 下的 `navtrain.pkl`、`navtest.pkl`、`navtrain_future.npz` 以及 navtest 的 keyframe 缓存 `runs/navsim_zs/openpilot/navtest/frames.npy` 在 box 上已经不存在（10-08 之后被清掉；`pp_unfreeze.py` 的在线渲染路径 `Frames` / `full_meta` 因此现在跑不起来）。已用归档脚本 `experiments/zeroshot_openloop/archive/navsim_zs_index.py` 重建 `navtest.pkl`、`navtrain.pkl`（pool 任务 `vt-index-*`；`navtrain_future.npz` 没有重建）。token 列表取自各 shard 的 `tab.npz`。
- 22:28 CST：`lb_navtest` 像素缓存完成（12 146 行，35.6 GB，24 核 169 s）。等价检查（512 行，冻结 encoder 过缓存像素 对 `lb_navtest@warp/front.npy`）：mean |d| 2.9e-4，max 0.031，RMS 1.73（与 `pp_unfreeze` 的 x4 检查同量级）。`lb_navhard`（5 912 行）同时在建。
- 22:29 CST 起：navtrain 12 个 shard 在渲染（各 10 核，约 13 token/s/shard，预计 22:45 CST 前后完成）。完成前 `PixelStore(FULL, ...)` 会因文件不存在而报错。

### 恒等检查的容差（给 builder：prereg 的「> 0.03 m 的行 < 0.1%」在像素路径上按字面过不了）

SH30-F-s0，navtest 前 2 048 行，plan 点 ≤ 4 s（21 个点），对缓存 token 路径（`front.npy`）：

| 路径 | 均值 m | 行内最大的中位数 m | 最大 m | > 0.03 m 行占比（≤ 4 s） | > 0.03 m 行占比（前 15 点） | > 0.07 m 行占比 |
|:--|--:|--:|--:|--:|--:|--:|
| 像素缓存 + 未优化 encoder（`pp_prep.enc_dev`，每次 128 对） | 0.0020 | 0.011 | 0.063 | 11.2% | 0.24% | 0 |
| 像素缓存 + `fast_encode`（7 个旧 slot 每次 128 对，t0 一次 64 对） | 0.0023 | 0.014 | 0.063 | 13.7% | 0.49% | 0 |
| 错模型对照：SH30-F-s1，缓存 token | 0.0475 | 0.141 | 1.05 | 98.2% | 79.4% | 80.8% |

差异来自 fp16 下 batch 组成不同（token 上 mean |d| 约 1e-3）；32 m 以外一个 fp16 ulp 就是 0.031 m，所以 0.03 m 的阈值在 ≤ 4 s 的远点上被单个 ulp 触发，未优化路径也一样。建议恒等检查用「> 0.07 m 的行 = 0」或「均值 < 0.005 m」，并保留错模型对照（两者差 20 倍以上）。`fast_encode` 相对未优化路径没有额外可测的偏差。

### 标签栅格（main 转来的问题）

VT 是「每行一个标签」的情形：12 个 shard 共 103 288 行，`runs/op_probe/labels/navtrain_all.npz` 也是 103 288 个 token，没有重复标签，`Hinge(..., bank=True)`（commit 61cb40c2 的 `RowBank`）在这里不省显存；稠密栅格 103 288 × 128 × 96 fp16 = 2.5 GB / 任务，保持现状。

### 给 builder 和 arm-W agent 的转告（来自 main）

- 约到 00:30 CST 前不要改、不要 commit / stash / 覆盖：`experiments/body1/scripts/bd4_train.py`、`experiments/op_parity/scripts/ot_rows.py`、`experiments/body1/scripts/tokyo_bench.sh`（另一个 session 的未提交改动）。`lib/drivable_hinge.py` 已经解禁。
- 那个 session 今晚还会改 `experiments/INDEX.md`（alpasim / body1 行）和 `research/decisions.md`（一行）：按路径 `git add`，push 前 `git pull --rebase`，不要 `--autostash`。工作区有别人的未提交改动时 `git pull --rebase` 会拒绝；可用 `git fetch && git merge --ff-only origin/main`（提交之前做），然后提交并立刻 push。

## W 臂

（arm-W agent 维护；其他人请勿改这一节。登记：prereg「补记 1（第二波臂 W）」与补记 3。）

### 侧视像素缓存（阶段 1）

- 代码：`experiments/vis_train/scripts/vt_side.py`（`split` / `build` / `fov` / `check`），loader `lib/side_store.py`：

```python
import side_store as SD
SS = SD.SideStore(datas, dev)        # datas、行号与 PixelStore 相同（行序 = tab.npz；SS.names 可核对）
prev, cur = SS.t0(rows)              # uint8 (B, 2, 2, 6, 128, 256) ×2，已在 dev 上：[CAM_L0, CAM_R0] 的 t0 图像对；可在 Prefetch 线程里调
```

- 位置：`$DATA_DIR/runs/vis_train/px_side/<data>/side_t0.npy`，(N, 2 相机, 2 帧 [t0 − 0.2 s, t0], 2, 6, 128, 256) uint8，1.57 MB / token；`meta.npz`（names、cam_t、yaw、pose）、`ents.pkl`（渲染输入）。navtrain 151 GiB、`lb_navtest` 17.8 GiB、`lb_navhard` 8.7 GiB。
- **图像对的时间**：W 协议的 0.2 s 对，（t0 key warp 到 t0 − 0.2 s 的位姿，t0 key），与前视 t0 slot 的构造相同；不是 P3 的 0.5 s 对。理由与几何见 prereg 补记 3 第 1 点。侧相机的 W 协议 warp 可以直接用 `pp_prep` / `op_interp` 现有的 `warp_frame`：把相机位置与位姿换到绕竖直轴转过安装 yaw 的车体系里（`side_store.virtual`），不改 warp 代码。
- 渲染走 pool（`vt-side-*`，owner `vis_train-W`，每 shard 4 核、nice 19、priority −5，解释器 op-train）。盘：用户把余量线改为 50 GB（经 main）；`build` 写前核对。

### 三个视图合起来的水平视场（由渲染代码的射线量出，`vt_side.py fov`）

navtrain 103 288 token 与 navtest + navhard 18 058 token，各 16 套标定；每个视图是 openpilot wide 帧（focal 455，512 px），像素中心张角 58.63°（road 帧 31.37°），原生相机（63.7°）对三个视图的两种帧覆盖率都是 100%。方位角相对 ego 朝向，左正：

| 量（度） | navtrain 最小 / 中位 / 最大 | navtest + navhard 最小 / 中位 / 最大 |
|:--|:--|:--|
| CAM_L0 安装 yaw | +54.3 / +54.9 / +56.8 | +53.5 / +55.2 / +56.8 |
| CAM_R0 安装 yaw | −56.0 / −55.3 / −53.2 | −56.4 / −55.5 / −53.2 |
| L 视图范围 | [+25.6, +84.3]（中位） | [+25.9, +84.5]（中位） |
| F 视图范围（沿 ego x 轴渲染，不随安装 yaw） | [−29.27, +29.36] | 同左 |
| R 视图范围 | [−84.6, −25.9]（中位） | [−84.7, −26.1]（中位） |
| L–F 重叠 | 1.85 / 3.73 / 4.31 | 1.85 / 3.46 / 5.15 |
| F–R 重叠 | 2.64 / 3.32 / 5.46 | 2.27 / 3.17 / 5.46 |
| 左侧到达 | 83.7 / 84.3 / 86.2 | 82.9 / 84.5 / 86.2 |
| 右侧到达 | 82.4 / 84.6 / 85.3 | 82.4 / 84.7 / 85.6 |

合起来 **至少 ±82.4°（中位 ±84.5°），视图之间没有缝**（最小重叠 1.85°），满足 ≥ ±60°，不需要拼接宽帧。`fov.md` 里切内侧失败越过的路沿中位方位角 43°，落在侧视图内（侧视图光轴 ±55°）。竖直方向与前视相同（wide 帧地平线行 151.8，沿光轴 8.2 m 以内的地面在帧外）。原始数：`px_side/fov-navtrain.json`、`fov-eval.json`。

### 与 P3 缓存侧视 token 的等价（`vt_side.py check`，冻结 Cinque encoder，256 行 × 2 相机）

| 数据 | 缓存的 t0 key 对现渲染 | 缓存的前一帧对现算 CPU warp（16 行） | (t0 − 0.5 s key, 缓存 t0 key) 对 `side.npy` 的 k = 3（P3 配对） | 缓存的 W 对 对 `side.npy` | 另一行的 `side.npy`（跨行尺度） |
|:--|:--|:--|:--|:--|:--|
| `lb_navtest` | 逐字节相同 | 逐字节相同 | mean \|d\| 4.2e-4，max 0.0625（RMS 1.84） | mean \|d\| 0.857 | mean \|d\| 1.246 |
| `navtrain_full.s0of12` | 逐字节相同 | 逐字节相同 | mean \|d\| 1.3e-4，max 0.0625（RMS 1.81） | mean \|d\| 0.828 | mean \|d\| 1.290 |

配对相同时复现 P3 的 token（与前视缓存检查的 2.9e-4 同量级，来自 fp16 的 batch 组成）；换成 W 配对后 token 差 0.857，是跨行差异的 0.69 倍：图像对间隔对 encoder 输出的影响很大（第 142 条），所以 W 的侧视 token 不等于 P3 的侧视 token，这是预期差异。图像对的平均像素差：W 对 12.8，P3 对 20.2（灰度级）。

### 给读数 agent：> 45° 切内侧按「被越过的内侧路沿 t0 时是否在常规视场内」分层

`$DATA_DIR/runs/corridor/fov/`（1 517 个 navtest > 45° token）：

- `tok.parquet`：每 token 一行，键 `token`（另有 `log`、`sgn` 转向符号、`dpsi`）。对所有 token 都有定义的分层量：`edge_in_w_fov_t0`（内侧路沿采样点在 wide 帧水平视场内的占比，t0；`edge_in_n` = 0 时无内侧路沿命中）、`brg_edge_in_s10 / s20`（内侧路沿方位角）。建议分层：`edge_in_w_fov_t0 == 0` 对 `> 0`，各层内报 W 与对照的切内侧率及配对差。
- `unit.parquet`：只含 SH30 两个 seed 与 WA-JEPA 的失败单元，键 `token` + `unit`（`sh0_pp`、`sh1_pp`、`wa_pp`）。切内侧 = `lqr_out` 且 `lqr_side == sgn`；越出点是否在常规视场内 = `lqr_pt_w_fov_t0`（水平）/ `lqr_pt_w_img_t0`（含行范围），越出点 10 m 内路沿在视场内的占比 = `lqr_edge_w_fov_t0`，方位角 `lqr_brg`。用法：取 `unit == f"sh{seed}_pp"` 的切内侧 token，按 `lqr_pt_w_fov_t0` 分成两组固定 token 集，报 W 与对照在这两组上的切内侧率（同 seed 配对）。W / A 自己的越出点不在表里，要它们自己的方位角需用 `experiments/corridor/scripts/fov_build.py` 对其 plan 重放。

### 进度

- 22:41 CST：`lb_navtest` 侧视缓存完成，检查通过。
- 22:50 CST 前后：navtrain 12 个 shard（103 288 行，151.3 GiB）与 `lb_navhard`（5 912 行，8.7 GiB）全部完成，`navtrain_full.s0of12` 检查通过（结果 `px_side/check-s0.json`、`check-navtest.json`）。阶段 1 结束；盘余 285 GiB。
- 阶段 2（trainer 的 W 臂、`VT-W-*` 的 bench 路由、链）在第一波启动之后才动 `vt.py` / `vt_chain.sh` / bench 文件。

## 第一波的链（接手的 lane agent 维护；2026-10-10 23:20 CST 起）

### 怎么续

- 每臂一条链：box 上 `scripts/tmux_run.sh vt-<臂>-s<seed> experiments/vis_train/scripts/vt_chain.sh <臂> <seed> <步数>`，状态在 `$DATA_DIR/runs/vis_train/chain/<臂>-s<seed>/{STATUS,DONE,ERROR,jobs.txt,pool/}`。同一条命令重跑即续（训练 `--resume`，步数必须与首次相同，否则 `claim` 拒绝）。重跑前先 `tmux kill-window -t jev:vt-<臂>-s<seed>`（窗口在脚本退出后留着）。
- box 上 `python` 不在 PATH：pool / bench 用 `~/data/jev-drive/.venv/bin/python -m jevdrive.cl|bench`，torch 任务用 `$DATA_DIR/envs/op-train/bin/python`。

### 22:53 CST 的失败与修复

A / A0 / B 六条链的 preflight（6 步 smoke，eval 在步 0 与步 3）真失败，不是被杀：停止规则把 memory 臂步 0 的 memory-on ADE（1.08–1.12 m，对起点 0.57 m，构造性差值）计了两次。修复 = 停止规则从 warmup 之后计数（commit d630dc45，prereg 补记 4）。pool 的 `rc None` 只是「依赖失败」的转述，preflight 自身 rc 1。

### 步数与当前任务（23:09 CST 重启后）

| 臂 | 步数 | 卡（pool 放置） | 23:15 实测 it/s | 预计结束（CST / JST） |
|:--|--:|:--|--:|:--|
| F-s0 / F-s1 / F0-s0 | 60 000（22:52 启动，未动） | 1 / 2 / 0 | 6.6 / 7.6 / 6.2（独占时 15） | 约 01:10–01:20 / 02:10–02:20 |
| A0-s0 / A0-s1 | 50 000 | 5 / 2 | 6.3 / 7.5 | 约 01:20 / 02:20 |
| A-s0、B-s1 | 50 000 | 4（两个同卡） | 2.76 / 2.77 | 约 04:15 / 05:15 |
| A-s1、B-s0 | 50 000 | 3（两个同卡） | 2.76 / 2.73 | 约 04:15 / 05:15 |
| C-s0 | 40 000 | 0（与 F0 同卡） | 1.23（F0 结束后预计约 1.9） | 约 05:45 / 06:45 |

- 步数的来由：A / B 先按分级启动的独占速度（5.28 it/s）以 60 k 提交；pool 把四个任务两两放在同一张卡上，实测 2.76 it/s，60 k 要到 06:10 JST，于是 23:07 取消（各跑了约 1 000 步，无 snapshot，run 目录只有 `vt_run.json` / `evals.json`，已删）并以 50 k 重新提交（50 000 / 2.76 = 5.0 h）。A0 同步数。F 已在跑 60 k，保留（对照取最大者；与 A / B 的比较用 F 的 k50）。
- C 取 40 k：F0 每 10 k 一个 snapshot，C 的末尾对 `VT-F0-s0-k40`；1.23 it/s（与 F0 同卡约 2 h）→ 1.9 it/s。
- F / F0 的 HUGSIM 等 `VT-C-s0/ckpt-final.pt` 出现后才排（`VT_LAST` 默认）。
- 恒等：`runs/vis_train/ident/{A,A0}-s{0,1}.json`、`B-s0.json`、`C-s0.json` 均 `pass_2ulp` = true（> 0.032 m 的行 0，错模型对照 70%），F-s0 逐位相同。分级启动：A / B / C 的 `train-smoke-vt-stage-*` 均完成（A：步 300 dev ADE 0.579 对起点 0.561，位移 0.11%，支路 head ADE 9.85 → 1.16）。

### 吞吐（batch 64，navtrain 12 shard；`runs/vis_train/prof-step-*`）

| 步型 | 第 145 条 U2（在线 CPU 渲染） | 像素缓存 + U2 的 encoder 路径 | + `fast_encode` | + 编译（上线配置） | 分级启动实测（独占） | 正式运行（两个同卡） |
|:--|--:|--:|--:|--:|--:|--:|
| A / B（只过 t0 一对） | 0.7 it/s | 1.69（GPU 70%） | 4.20（87%） | 5.04（90%） | 5.28 | 2.76 × 2 = 5.5 / 卡 |
| C（8 个 slot 全过 encoder） | 0.7 it/s | 1.08（71%） | 1.71（91%） | 1.92（89%） | 1.89 | 1.23（与 F0 同卡） |

对 0.7 it/s：A / B 独占 7.5 倍、每卡合计 7.9 倍、单任务同卡时 3.9 倍；C 独占 2.7 倍。

## 读数

（reads builder 维护；其他人请勿改这一节。脚本 `scripts/vt_read.py`，产物 `results/`。）

### 怎么重跑

- Mac 上一条命令：`.venv/bin/python experiments/vis_train/scripts/vt_read.py sync`。它经 ssh 把 `vt_read.py box` 作为一个 pool CPU 任务提交（`vt-read`，owner `vis_train`，16 核，envs/navsim2），等它结束，再把 `$DATA_DIR/runs/vis_train/reads/out/` 的六个小文件拷回 `results/`（`reads.parquet`、`reads.md`、`disp.csv`、`probe.csv`、`trend.png`、`trend.pdf`）。任何时候都可以重跑，只处理新出现的 snapshot；之后按路径 `git add experiments/vis_train/results/...` 提交。`sync --no-replay` 只重建表，`sync --copy-only` 只拷上一次的结果。
- 给报告用的首个 snapshot 短表：`.venv/bin/python experiments/vis_train/scripts/vt_read.py first`（读 `results/reads.parquet`，Mac 上即可）。
- Stage-0 probe（第 160 条）：box 上 `.venv/bin/python experiments/vis_train/scripts/vt_read.py probe --name <名字> --tags VT-A-s0 VT-B-s0 ...`，提交三段 pool 任务（`vt.py tokens` 占一张卡 → `rep.py` 的 thin decoder 占一张卡 → `opb_score.py` 24 核打 3 154 个 > 20° token），结果 `$DATA_DIR/runs/vis_train/reads/probe/<名字>/score_t20.csv`，下一次 `sync` 自动读进 `reads.md` 与判定。同名重跑即续。
- 自检：box 上 `$DATA_DIR/envs/navsim2/bin/python experiments/vis_train/scripts/vt_read.py check`（向量化 bootstrap 对 `jevdrive.stats.paired`，132 格最大差 2.2e-16）。

### 缓存在哪

- `$DATA_DIR/runs/vis_train/reads/tok/<模型>.parquet`：每个模型一张逐 token 表（bench 子分、plan 4 s 弧长、heading gain、LQR 回放首次出界的侧、NC 类别），`jevdrive.cache` 按 `units.csv` 与预测文件的 size / mtime 作键，bench 重跑后自动失效；改定义时改脚本里的 `VERSION`。回放只跑 DAC / NC / TTC 失败的 token（每模型约 650 个），8 个模型 16 核 84 s。
- `$DATA_DIR/runs/vis_train/reads/out/`：每次 `box` 全量重建（bootstrap 是矩阵乘法，秒级）。run 目录在 `runs/vis_train/reads/<时间>/`，pool 日志在 `runs/vis_train/reads/pool/<时间>/`。

### 定义的出处（全部复用，没有重推）

转角桶 = `jevdrive.bench.tables.navtest_strata`；切内侧 / 转不过去 = `corr_report.py` 的 `inside` / `cannot`（`fd_navsim.work` 回放的 `lqr_side`、`plan_kin` 的 gain < 0.9）；NC 类别 = `nc_tax.classify`；FoV 分层 = `runs/corridor/fov/` 的 `tok.parquet` / `unit.parquet`。校验：SH30 的 NC 分类计数与第 196 条逐项相同（176.5 / A 93.5 / A1 58.5 / A2 35 / B 23 / C 13.5 / D 16.5 / E 30，TTC-only 87.5）；> 45° 切内侧 65、转不过去 37.5（corr0 在匹配上的 1 476 个 token 里是 64 / 36）；回放的 8 个子分与 bench 存档最大差 1.2e-12。

### 对照取法与判定

- 同 seed、同步数；对照在该步没有 snapshot 时取其最近的登记 snapshot（并列取早的），表里 `ctrl` 列标出。A / B 的 k05、k15… 对 A0 就是这种情况（A0 每 10 k）。A / B 的末尾（50 k）对 F 用 `VT-F-s*-k50`；C 的末尾对 `VT-F0-s0-k40`。navhard / HUGSIM 只有末尾 checkpoint 有，对照取其末尾（步数不同时 `ctrl step` 列可见）。
- 判定线、护栏、不退步规则、seed 反向都在 `reads.md` 末尾按登记的阈值机械算出；输入缺失的线写 `pending`，末尾 checkpoint 之前写 provisional。「Stage-0 probe 不动」取为「支路 token 对同一次 decode 的 V 的配对 CI 含 0」（预登记没有给数，这是读数脚本的取法）。趋势 = 差值序列后三个 snapshot 的均值对前三个，少于 6 个 snapshot 时写 n/a。

### 还缺什么

- prereg 读数 5（C 的原生帧直路 ADE 对 shipped，第 137 条）不在脚本里。
- A0 / A / B / C / W 的 HUGSIM：bench 没有支路 encoder 的 serving，只有 F / F0 在排。
- `SH30-F-s{0,1}` 的 navhard（W 帧）参照行已于 2026-10-10 23:40 CST 经 `bench run` 提交，跑完后下一次 `sync` 进板表。
- W 臂：`VT-W-*` 在 bench 里能 resolve 之后脚本自动纳入（登记 50 k、每 5 k、对照 A；步数不同时改脚本顶部的 `STEPS` / `EVERY` / `SEEDS`）；`:sideoff` 的读数名按 `VT-W-s<seed>:sideoff` 取。
- 链目录里 `A / B` 的 `pool/b-k50`、`b-k55` 与 `A0` 的 `b-k50` 下的 `ERROR` 是 23:07 取消 60 k 提交时留下的（`cancelled while queued`），不是读数失败。
