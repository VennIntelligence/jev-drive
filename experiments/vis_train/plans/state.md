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
