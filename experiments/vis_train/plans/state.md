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

- 22:20 CST：`runs/navsim_zs/index/` 下的 `navtrain.pkl`、`navtest.pkl`、`navtrain_future.npz` 以及 navtest 的 keyframe 缓存 `runs/navsim_zs/openpilot/navtest/frames.npy` 在 box 上已经不存在（10-08 之后被清掉；`pp_unfreeze.py` 的在线渲染路径 `Frames` / `full_meta` 因此现在跑不起来）。已用归档脚本 `experiments/zeroshot_openloop/archive/navsim_zs_index.py` 重建 navtest / navtrain 两个 index（pool 任务 `vt-index-*`）；token 列表取自各 shard 的 `tab.npz`，不依赖 `navtrain_future.npz`。
- 渲染任务已排队（`vt-px-split` → 14 个 `vt-px-build-*`，每个 10 核）：navtrain 12 个 shard 先跑，然后 `lb_navtest`、`lb_navhard`。
