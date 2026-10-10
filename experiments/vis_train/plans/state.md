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

配对相同时复现 P3 的 token（与前视缓存检查的 2.9e-4 同量级，来自 fp16 的 batch 组成）；换成 W 配对后 token 差 0.857，是跨行差异的 0.69 倍：图像对间隔对 encoder 输出的影响很大（第 142 条），所以 W 的侧视 token 不等于 P3 的侧视 token，这是预期差异。图像对的平均像素差：W 对 12.8，P3 对 20.2（灰度级）。

### 给读数 agent：> 45° 切内侧按「被越过的内侧路沿 t0 时是否在常规视场内」分层

`$DATA_DIR/runs/corridor/fov/`（1 517 个 navtest > 45° token）：

- `tok.parquet`：每 token 一行，键 `token`（另有 `log`、`sgn` 转向符号、`dpsi`）。对所有 token 都有定义的分层量：`edge_in_w_fov_t0`（内侧路沿采样点在 wide 帧水平视场内的占比，t0；`edge_in_n` = 0 时无内侧路沿命中）、`brg_edge_in_s10 / s20`（内侧路沿方位角）。建议分层：`edge_in_w_fov_t0 == 0` 对 `> 0`，各层内报 W 与对照的切内侧率及配对差。
- `unit.parquet`：只含 SH30 两个 seed 与 WA-JEPA 的失败单元，键 `token` + `unit`（`sh0_pp`、`sh1_pp`、`wa_pp`）。切内侧 = `lqr_out` 且 `lqr_side == sgn`；越出点是否在常规视场内 = `lqr_pt_w_fov_t0`（水平）/ `lqr_pt_w_img_t0`（含行范围），越出点 10 m 内路沿在视场内的占比 = `lqr_edge_w_fov_t0`，方位角 `lqr_brg`。用法：取 `unit == f"sh{seed}_pp"` 的切内侧 token，按 `lqr_pt_w_fov_t0` 分成两组固定 token 集，报 W 与对照在这两组上的切内侧率（同 seed 配对）。W / A 自己的越出点不在表里，要它们自己的方位角需用 `experiments/corridor/scripts/fov_build.py` 对其 plan 重放。

### 进度

- 22:41 CST：`lb_navtest` 侧视缓存完成，检查通过；navtrain 12 shard 与 `lb_navhard` 在 pool 里（等核）。
- 阶段 2（trainer 的 W 臂、`VT-W-*` 的 bench 路由、链）在第一波启动之后才动 `vt.py` / `vt_chain.sh` / bench 文件。
