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
- 阶段 2 已完成并启动，见下一小节。

### 阶段 2：trainer、bench、链（2026-10-10 23:20–23:50 CST；登记：prereg 补记 5）

**实现**（commit 180d2730、2a3bc178）。`vt.py` 的臂 `W` = A + `views=3`：`VT.branch` 把 F0 的 t0 对（`PixelStore.t0`，与 A 逐字节相同）与 L0 / R0 的 t0 对（`SideStore.t0`）按视图拼成 192 对过同一个支路 encoder，得 (B, 3, 32, 512)，经 `ParityAdapter(n_cam=3, n_t=1)`；支路 head 输入 3 × 32 × 512 + 20；训练时的 memory 屏蔽是同一个 (B, 1) 抽样（三路同屏蔽）。`Cfg` 没有加字段，F / A0 / A / B / C / F0 的行为与 checkpoint 格式不变：`VT-A-s0-k05` 在 navtest 12 146 行上，改动前（14052da3）与改动后的 `vt.py plans` 输出 `plan_mu` / `plan_std` **逐位相同**。

**定下的三件事**

1. 显存与梯度 pass 的切法（单 shard 200 步，各与一个 F / A0 轻任务同卡，`max_memory_reserved`）：

| 配置 | it/s | 峰值 GB |
|:--|--:|--:|
| `--enc-compile --compile`（192 对一次过） | 1.50 | 42.9 |
| `--enc-compile --compile --enc-chunk 64`（每视图一次） | 1.46 | 42.5 |
| `--compile --enc-ckpt 64` | 1.06 | 20.3 |
| `--compile --enc-ckpt 96` | 1.15 | 25.5 |

   前两者在噪声内相同（不同卡、不同同卡任务），比 checkpointing 快 1.3–1.4 倍；42–45 GB 能放进有一个轻任务的卡（空余 52–66 GB）。取 **`--enc-chunk 64`**：F0 视图那次调用就是 A 的那次调用，且复用 64 对形状的编译图。全量数据上峰值 45.1 GB（nvidia-smi 46 GB），链里预订 `--vram 50`、`--vram-cap 49`（分配器硬上限，超了自己 OOM）、10 核、RAM 64 GB、`--train`、优先级 5。192 对的显存远小于 3 × A（A 的 25 GB 里 encoder 的图只占约 15 GB，且那是 reserved）。
2. bench 选项 `:sideoff`（仅 `VT-W-*`）= `vt.py plans --mem sideoff`，mask (B, 3) = [F0 保留, L0 / R0 屏蔽]；`:noside` 三路全屏蔽，`:mshuf` 三路都取自另一 log 的同一 token。链的末尾读数：navtest、navhard、navtest 的 `:noside` / `:sideoff` / `:mshuf`。dev eval 多一栏 `ade_sideoff`。
3. 支路 head 的标准化统计：F0 视图用 `front.npy` 缓存 token（与 A 相同的 rng、相同的 8 192 行）；侧视图用同一批行经初始化时的支路 encoder（= 冻结 Cinque）现算（全量数据上含在 32 s 的初始化里）。读数：三路 `mu` 的 RMS 1.09 / 1.18 / 1.21，`sd` 均值 1.13 / 0.99 / 0.99。

**恒等**（`runs/vis_train/ident/W-s{0,1}.json`，navtest 前 2 048 行、前 15 个 plan 点）

| 量 | s0 | s1 |
|:--|--:|--:|
| memory 屏蔽 对 SH30：> 0.032 m 的行 / 最大 / 中位 | 0 / 0.03125 m / 0.0029 m | 0 / 0.03125 m / 0.0020 m |
| 同上按原登记线（> 0.03 m 的行） | 0.20% | 0.29% |
| 错模型对照（另一 seed）> 0.032 m | 70.0% | 70.0% |
| 屏蔽后的 bias 对 SH30 自己的 adapter（fp32）最大差 | 0 | 0 |
| `:sideoff` 对 memory 全开：> 0.03 m 的行 / 中位 | 92.7% / 0.16 m | 92.6% / 0.22 m |
| W 的 F0 视图 token 对 A 的支路 token（512 行）最大差 | 0 | 0 |
| 像素缓存 token 对 `front.npy`：mean \|d\| / max | 2.6e-4 / 0.031 | 同量级 |

`pass_2ulp` 两个 seed 都为 true（含新增的「侧视被读」一条）。初始化时侧视 token 与 F0 token 的 mean |d| 1.30（RMS 1.65），不是同一内容的复制。

**分级启动**（`train-smoke-vt-stage-W-s0`，全量数据 300 步，card 5 与 A0-s0 同卡）：loss 全程有限、无跳过步；dev ADE 1.233（步 0，memory 新初始化）→ 0.724 → 0.596 → **0.583**（步 300），起点（屏蔽 memory）0.561，差 +0.022 m（线 0.1 m）；视觉位移 0.09%（各 stage 0.09–0.12%），policy 0.07%，adapter 2.98%；支路 head loss 4.98 → 0.50、head ADE 9.85 → 1.23 m；步 300 时 `ade_masked` 0.585、`ade_sideoff` 0.592；1.54 it/s，45.1 GB。参照 A 的分级启动：步 300 ADE 0.579、位移 0.11%、head ADE 1.16。

**正式运行**（23:43 CST 提交）

| 链 | 训练任务 | 状态（23:51 CST） | it/s | 峰值 | 预计结束 |
|:--|:--|:--|--:|--:|:--|
| W-s0，50 000 步 | `vt-t-W-s0` 1010-234311-a449 | card 5，与 A0-s0 同卡（pool 放置） | 1.56 | 45.2 GB（卡上 46 / 预订 50） | 一直同卡 08:40 CST；A0-s0 约 01:20 CST 结束后独占，速度未测，按 A 的独占 / 同卡比例估约 2.3 it/s → 约 06:20 CST（07:20 JST） |
| W-s1，50 000 步 | `vt-t-W-s1` 1010-234311-23ce | 排队：`--when-exists runs/op_parity/runs/VT-A0-s1/ckpt-final.pt`（`VT_WHEN`），即 A0-s1 结束（约 01:20 CST）后由 pool 放置；preflight 已过 | – | – | 独占约 2.3 it/s 时约 07:20 CST |

- W-s1 加门的原因：现在放它只能与第一波的 F / A0 轻任务三个挤一张卡，会拖慢它们；门开时 F / A0 已结束，有空卡。
- 到早上没跑完：按相同步数的 snapshot（`VT-W-s<seed>-k<NN>` 对 `VT-A-s<seed>-k<NN>`）比较，每个 snapshot 的 navtest 读数由链里的 `vt-b-W-*` 任务自动提交。

**怎么续**：与第一波相同。box 上 `tmux kill-window -t jev:vt-W-s<seed>` 后重跑同一条命令 `scripts/tmux_run.sh vt-W-s<seed> experiments/vis_train/scripts/vt_chain.sh W <seed> 50000`（训练 `--resume` 从最近的 snapshot 续，步数必须仍是 50 000；W-s1 的 `VT_WHEN` 在 A0-s1 结束后可省）。状态在 `$DATA_DIR/runs/vis_train/chain/W-s<seed>/{STATUS,DONE,ERROR,jobs.txt,pool/}`，训练日志 `pool/train/log.txt`，dev 曲线 `runs/op_parity/runs/VT-W-s<seed>/evals.json`。建造期的任务日志在 `chain/W-build/`（ident、四个测量 smoke、分级启动、回归比对）。若 `--vram-cap 49` 触发 OOM（峰值 45.2 GB，余量 3.8 GB）：`VT_VRAM=56` 重跑同一条命令。

### 交接（00:10 CST / 01:10 JST，arm-W builder 收尾；由 jev-night 接手）

W 的建造已结束，没有半成品、没有未提交的改动（Mac 与 box 的 checkout 里都没有我的未跟踪或已改文件）。两条链的 tmux 窗口 `jev:vt-W-s0`、`jev:vt-W-s1` 在等各自的训练任务，不要关。

**仍在 pool 里的任务**（owner `vis_train-W`；日志目录 `$DATA_DIR/runs/vis_train/chain/W-s<seed>/pool/<子目录>/`）

| 任务 | id | 状态（00:09 CST） | 做什么 / 等什么 | 日志子目录 |
|:--|:--|:--|:--|:--|
| `vt-t-W-s0` | 1010-234311-a449 | running，card 5，与 A0-s0 同卡 | 训练 50 000 步；步 2 000，1.53–1.56 it/s，峰值 45.4 GB reserved（卡上 46 / 预订 50） | `train/` |
| `vt-b-W-s0-k05 … k45` | 1010-234311-2fd8、234312-ff65、234312-322f、234312-31d6、234313-bd49、234313-5988、234313-0e92、234314-54d8、234314-748a | queued | 各等 `runs/op_parity/runs/VT-W-s0-k<NN>/ckpt-final.pt` 出现，然后 `bench run --model VT-W-s0-k<NN> --bench navtest` | `b-k<NN>/` |
| `vt-b-W-s0-final` | 1010-234314-000e | queued | `--after` 训练任务且等 `VT-W-s0/ckpt-final.pt`；navtest + navhard，navtest 的 `:noside` / `:sideoff` / `:mshuf` | `b-final/` |
| `vt-t-W-s1` | 1010-234311-23ce | queued（门） | 等 `runs/op_parity/runs/VT-A0-s1/ckpt-final.pt`（A0-s1 结束，约 01:20 CST），之后由 pool 放置（需要 50 GB 空余且该卡训练任务未满）；preflight 已过 | `train/` |
| `vt-b-W-s1-k05 … k45` | 1010-234311-9122、234312-d623、234312-9eec、234312-166d、234313-e068、234313-b709、234313-8b1f、234314-5ff6、234314-fc90 | queued | 同 s0，等 `VT-W-s1-k<NN>/ckpt-final.pt` | `b-k<NN>/` |
| `vt-b-W-s1-final` | 1010-234315-cd67 | queued | 同 s0，`--after` 1010-234311-23ce | `b-final/` |

已结束的建造任务（ident、四个测量 smoke、分级启动、回归比对、两个 preflight）全部 rc 0，日志在 `chain/W-build/` 与 `chain/W-s<seed>/pool/train/preflight/`。

**W-s0 的 dev 读数**（步 1 000）：ADE 0.5669，`ade_masked` 0.5652，`ade_sideoff` 0.5656，head ADE 0.99 m，视觉位移 0.24%；起点 0.5606。停止规则未触发。

**预计**：W-s0 一直与 A0-s0 同卡则 08:40 CST；A0-s0 约 01:20 CST 结束后独占，独占速度没有测过（按 A 的独占 / 同卡比例估约 2.3 it/s，约 06:20 CST）。W-s1 在门开后独占一张卡时约 07:20 CST。早上没跑完按相同步数的 snapshot 对 A 比较。

**重启命令**（box 上，先 `tmux kill-window -t jev:vt-W-s<seed>`；同一条命令即续，训练 `--resume`，步数必须仍是 50 000）

```bash
scripts/tmux_run.sh vt-W-s0 experiments/vis_train/scripts/vt_chain.sh W 0 50000
scripts/tmux_run.sh vt-W-s1 env VT_WHEN=$DATA_DIR/runs/op_parity/runs/VT-A0-s1/ckpt-final.pt experiments/vis_train/scripts/vt_chain.sh W 1 50000   # A0-s1 结束后 VT_WHEN 可省
```

**已知问题**

- `--vram-cap 49`（链里 = 预订 50 − 1）是 CUDA 分配器的硬上限，峰值 45.4 GB，余量 3.6 GB。若训练任务报 CUDA OOM（`pool/train/log.txt`；pool 会按 `--tries 4` 带 `--resume` 重试，但同一上限下会再 OOM）：kill 窗口后用 `VT_VRAM=56 scripts/tmux_run.sh …` 重跑同一条命令。`VT_VRAM` 只改预订与上限，不进 `Cfg`，`claim` 不受影响。
- W-s0 与 A0-s0 同卡，两者互相拖慢（A0-s0 约 01:20 CST 结束）；这是 pool 的放置，没有手动干预。
- 独占时的 it/s 是估计，不是实测；A0-s0 结束后读 `pool/train/STATUS` 即可得到。
- 链的 `STATUS` 文件停在「queued: train job …; waiting」直到训练结束，进度看 `pool/train/log.txt` 与 `runs/op_parity/runs/VT-W-s<seed>/evals.json`。
- 我的 smoke 目录还在 `runs/op_parity/runs/`：`smoke-vt-stage-W-s0`（分级启动）、`smoke-vt-W-s{0,1}` 与 `-k00`（链的 preflight，重跑链时 `--scratch` 会重建）；各约 1.7 GB，可删。
- 没做的：W 的 HUGSIM（family vt 不上 HUGSIM）、`vt_read.py` 里是否已认 `VT-W-*` 与 `:sideoff`（读数 agent 的文件，我没有改也没有核对）、按「路沿是否在常规视场内」分层的读数（上面「给读数 agent」一节）。

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

### 交接（2026-10-11 00:10 CST，lane 交给 jev-night；本节之后我不再动任何任务）

**没有取消任何任务。** 00:09 CST 的全部在跑 / 排队 / 带门任务（262 行：id、状态、名字、在等什么）在 [handover-jobs.txt](handover-jobs.txt)；现状以 `cl queue | grep -E "vis_train|VT-"` 为准。分组：

| 组 | id | 状态 / 在等什么 |
|:--|:--|:--|
| 训练 F-s0 / F-s1 / F0-s0（60 k） | 1010-225245-4c70 / -6d51 / -9b86 | 在跑，步 46.0 k / 38.7 k / 32.2 k |
| 训练 A0-s0 / A0-s1（50 k） | 1010-230854-fb48 / -523d | 在跑，步 26.7 k / 25.3 k |
| 训练 A-s0 / A-s1 / B-s0 / B-s1（50 k） | 1010-230854-22a0 / -4952 / -b6ec / -c74d | 在跑，步 9.6 k，2.76 it/s，card 4 / 3 / 3 / 4 |
| 训练 C-s0（40 k） | 1010-230021-65dc | 在跑，步 4.5 k，1.23 it/s（card 0，与 F0 同卡） |
| 训练 W-s0 / W-s1（50 k） | 1010-234311-a449 / -23ce | W-s0 在跑（card 5，1.55 it/s，45 GB）；W-s1 等 `VT-A0-s1/ckpt-final.pt` |
| snapshot 读数的启动任务 `vt-b-<臂>-s<seed>-k<NN>`（约 90 个，各链 `jobs.txt`） | 见清单 | 各等自己的 `VT-…-k<NN>/ckpt-final.pt`；出现后跑 `bench run --model … --bench navtest` 并退出 |
| 末尾读数 `vt-b-*-final`（11 个） | 见清单 | `--after` 各自的训练任务；navtest + navhard（A / B / W 另有 `:noside` / `:mshuf`，W 另有 `:sideoff`） |
| HUGSIM 64 `vt-b-{F-s0,F-s1,F0-s0}-hugsim` | 1010-225249-aa34 / -f3b9、1010-225247-fcd6 | `--after` 训练，且等 `VT-C-s0/ckpt-final.pt`（约 05:45 CST） |
| bench 自己的 `bn-navtest-VT-*` 阶段（约 150 个） | 见清单 | **被内存门挡住**，见下 |
| Stage-0 probe 链（k05，seed 0） | 1010-234950-0f52、-faa0（tokens）→ 1010-235054-3000（decode）→ -e225（score） | tokens 被内存门挡住，其余 `--after` |
| `vt-read`（读数表重建） | 1010-235353-395d | 被内存门挡住；不是本 session 的任务（接手方的 reads builder 提交） |
| `vt-native-onnx-test` | 1011-000816-17fb | 被内存门挡住；不是本 session 及其子 agent 提交的（owner `vis_train`，00:08 CST；应属接手方） |
| `SH30-F-s{0,1}` 的 navhard（W 帧）参照行 | – | 已完成（本 session 的 reads builder 经 `bench run` 提交，8 个 stage 全 done） |

**已知问题**

1. **内存门（最要紧）**：00:09 CST box 的 working set 496 GiB，对 kill line 541；pool 以「memory working set … > kill line（page cache; trimming）」把所有新的 CPU / 读数任务挡在队列里，最早的已等 28 min（`bn-navtest-VT-A-s1-k05` 的评分 worker、F 的 k25–k45、A0 的 k2x、probe、`vt-read`）。训练不受影响，但 snapshot 读数在积压，第一批 A / B 读数因此还没出齐。来源未查实（疑为 534 GB 像素缓存的 page cache 加 12 个训练各自的 `front.npy` host 映射）；我没有动它。接手后先看 `cl top` 第二行与 `cl usage`，确认 trimming 是否在降；不降就要让训练结束（F / A0 约 01:20 CST）腾出内存，或找 pool 的维护者。
2. B-s1 没有单独的恒等 json（只有 `ident/B-s0.json`）；A / A0 / W 两个 seed 都有，C 只有 seed 0（本来就只有一个）。B-s1 与 B-s0 走同一代码，训练已在跑；补一个 `vt.py ident --arm B --seed 1 --lr0-steps 10 --enc-compile --compile`（pool，约 2 min）即可。
3. 恒等按原登记线（> 0.03 m 的行 < 0.1%）不过，按补记 1 的两个 ulp 线过；两栏都在 json 里。
4. W 的 `--vram-cap 49`：峰值 45.2 GB，余量 3.8 GB；OOM 时 `VT_VRAM=56` 重跑同一条链命令（W 节「怎么续」）。
5. A / B 的步数是 50 k 不是 60 k（两个同卡 2.76 it/s）；F 的对照行用 k50。若 F / A0 结束后 A / B 所在的卡没有变快（它们两两同卡，不会迁移），结束时间仍是约 04:15 CST。
6. 第一个 snapshot 的短表（给 main 的报告）还没发：`vt_read.py first`，等 A / B / C 的 k05 读数出齐。
7. Stage-0 probe 的 decode 任务（-3000）只 `--after` A 的 token 任务；若在 B 的 token 落盘前启动会因缺 `VT-B-s0-k05/*.npy` 报错，同一条 `vt_read.py probe --name k05-s0 --tags VT-A-s0-k05 VT-B-s0-k05` 重跑即续。整条 probe 路径、判定线、`:noside` / `:mshuf` / `:sideoff`、W 的 FoV 分层、navhard / HUGSIM 配对行都还没有在真实输入上跑过（见「读数交接」节）；`vt_read.py` 是否认 `VT-W-*` 未核对。
8. 可删的 smoke 目录：`runs/op_parity/runs/smoke-vt-*`（W 的各约 1.7 GB）。
9. 没做的：读数的结论、趋势图的说明、decision 条目（先 pull，取当时的下一个空号）、`research/vis_train/index.html` 及第二读者核对、README 的 Conclusion。

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

## 读数交接（第一任 reads builder，2026-10-11 00:10 CST 收尾；之后不再改）

读数由 jev-night 的 reads builder 接手，「读数」一节归它维护；本节只记我留下的东西。

### 我提交、仍在 pool 里的任务（不要取消，接手方直接用）

Stage-0 probe 链 `k05-s0`（第 160 条协议，验证用：第一个 A / B snapshot 的支路 token），四个任务都还在排队，卡在 box 的内存线（`memory working set ... > kill line 541`），没有失败：

| job id | 名字 | 做什么 | 等什么 | 日志目录 |
|:--|:--|:--|:--|:--|
| 1010-234950-0f52 | `vt-probe-tok-VT-A-s0-k05` | `vt.py tokens --tag VT-A-s0-k05`（一张卡，16 GB） | 内存线 | `$DATA_DIR/runs/vis_train/reads/probe/k05-s0/pool/vt-probe-tok-VT-A-s0-k05/` |
| 1010-234950-faa0 | `vt-probe-tok-VT-B-s0-k05` | 同上，`VT-B-s0-k05` | 内存线 | `.../pool/vt-probe-tok-VT-B-s0-k05/` |
| 1010-235054-3000 | `vt-probe-dec-k05-s0` | `vt_read.py probe-decode`：`rep.py` 的 thin decoder 在 V 与两份支路 token 上（一张卡，12 GB） | `--after` 0f52（提交时写错成只跟第一个 token 任务，见下） | `.../pool/vt-probe-dec-k05-s0/` |
| 1010-235054-e225 | `vt-probe-score-k05-s0` | `opb_score.py`，24 核，3 154 个 > 20° token × 3 个 key | `--after` 3000 | `.../pool/vt-probe-score-k05-s0/` |

- 产物：`$DATA_DIR/runs/vis_train/tokens/VT-{A,B}-s0-k05/`（各约 1.2 GB）→ `$DATA_DIR/runs/vis_train/reads/probe/k05-s0/{decoder_poses.npz, tokens_t20.txt, score_t20.csv}`；`jobs.json` 记着四个 id。`score_t20.csv` 出现后下一次 `vt_read.py sync` 把它读进 `probe.csv` / `reads.md`。
- 已知问题：decode 任务的 `--after` 我是在修 `--after` 的逗号格式之后补交的，`jobs.json` 是手写的，实际提交的依赖要以 `cl queue` 的 why 列为准；若 decode 在 B 的 token 落盘之前启动，它会因缺 `VT-B-s0-k05/*.npy` 报错，此时同一条命令重跑即续（已完成的阶段跳过）：box 上 `.venv/bin/python experiments/vis_train/scripts/vt_read.py probe --name k05-s0 --tags VT-A-s0-k05 VT-B-s0-k05`。
- 这条 probe 路径没有在真实数据上跑通过：token dump、`rep.cmd_decode` 的 monkey-patch（`_load_tok` / `fit_pca` / `OUT` / `ARMS`）、V 的复现（应接近 10.59%）都未验证。

已结束、无需再管：`bench run --model SH30-F-s0 SH30-F-s1 --bench navhard`（W 帧参照行，23:39 CST 提交，八个阶段全部 done，`$DATA_DIR/runs/bench/navhard/SH30-F-s{0,1}@warp/`）；两次 `vt-read`（1010-232856-1d45、1010-233925-e409）。队列里的 `vt-read` 1010-235353-395d 不是我提交的。

### 读数命令（我交付时的版本，commit a67eecf0；之后以接手方的脚本为准）

- Mac：`.venv/bin/python experiments/vis_train/scripts/vt_read.py sync`（提交 `vt_read.py box` 为一个 pool CPU 任务，等待，拷回 `results/`）；`sync --no-replay`、`sync --copy-only`；`vt_read.py first`。
- box：`$DATA_DIR/envs/navsim2/bin/python experiments/vis_train/scripts/vt_read.py check`；probe 见上。
- box 上的落点：`$DATA_DIR/runs/vis_train/reads/tok/<模型>.parquet`（逐 token 缓存）、`reads/out/`（六个小文件）、`reads/pool/<时间>/`（pool 日志）、`reads/<时间>/`（run 目录）、`reads/probe/<名字>/`。

### 我交付时已验证与未完成

- 已在真实 snapshot 上验证：navtest 全分解与转角桶、off-road / 切内侧 / 转不过去、NC 类别与弧长比、位移表、趋势图、`first`、不退步表（F k05–k20、F0 k10–k20、A0 k10、SH30 k00；14 个模型的回放与存档子分最大差 4.9e-12）。
- 写了但没有真实输入、未验证：A / B / C / W 的任何行（第一个 `VT-A/B-*-k05` 的 navtest 读数在我收尾时还没出）、判定线（转弯线、W 线、seed 反向）、`:noside` / `:mshuf` / `:sideoff`、FoV 分层在 W 上的读法、navhard / HUGSIM 板表的配对行、Stage-0 probe。
- 没做：prereg 读数 5（C 的原生帧直路 ADE）；A0 / A / B / C / W 的 HUGSIM（bench 无 serving）。
- box checkout 里没有我留下的未跟踪或改动文件；Mac 上我没有未提交的改动。
