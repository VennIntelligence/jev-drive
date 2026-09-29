# WL 空间 token 抽取（token 副臂的数据侧）

目标：给 WL-2 预注册里的 `T` 臂（[wl2-prereg](2026-09-29-wl2-prereg.md)「副臂：V-JEPA 2 空间 token」）准备数据。先在 WL-1 全部 run 上抽，之后 WL-2 新数据用同一份代码（`--set` 换目录）。不训练任何东西。

## 规格（取自 prereg 当前草稿，未改）

- 每个 index 行、每个相机：4 帧 clip 走 V-JEPA 2 ViT-L（`features.VJepaFeatures(frames=4)`，与 WL-1 的 `mean` 同一 extractor、同一预处理、同一 clip 采样）。
- 取最后一个 temporal slice 的 16 × 16 patch token，块均值池化成 2 × 4 网格（行 × 列，row-major），每相机 8 个 token。
- 输出 `processed/wl_gen/tok.npy`，shape (307 587, 24, 1024)，fp16，15.1 GB（约 48 KB / 行）。token 序号 = 相机 × 8 + 行 × 4 + 列，相机顺序 front、front_left、front_right。
- 行与 `index.parquet`、`z.npy`、`z_ok.npy` 逐行对齐（z_ok 全真，307 587 行）。

## 前提核查

WL-1 的 JPEG 没被删：`wl_data prune` 没跑过，按 154 个等间距行抽样，三相机 4 帧全部存在。所以不需要重新生成。

## 实现

- `jevdrive/features.py`：`VJepaFeatures(grid=(rows, cols))` 多一个 `grid` tap，默认关，不影响别处。
- `jevdrive/wl_tokens.py`：`extract`（按 3 000 行一个 chunk，可续跑，可 `--shard i/n`）、`assemble`、`check`。每个 chunk 同时存同一次前向算出的完整 `mean`（`tok/m*.npy`，共约 1.9 GB），只用于等价检查。

## 等价检查

chunk 0（3 000 行，9 000 个 clip）：重算的 `mean` 对 `z.npy` 里存的 `mean`，余弦最小 0.9999999，全部 ≥ 0.999。说明行对齐和预处理都与 WL-1 一致。

另外记一条：网格 token 只取最后一个 temporal slice，所以 token 的均值不等于 `mean`（`mean` 是两个 slice 的平均）。实测 chunk 0 上网格均值对 `mean` 的余弦中位数 0.985、最小 0.938，达不到 0.999。这不是对齐问题，是 slice 选择的后果；`mean` 与 last-slice 均值的差就是第一个 slice。要让网格均值严格复现 `mean`，得再存第一个 slice（数据翻倍到 30 GB），目前没做。

## 规模与算力

- 单 chunk 实测 62 s / 9 000 clip，约 144 clip/s（含启动），瓶颈是 CPU 端 JPEG 解码，与 WL-1 `vjepa` 步相同；显存峰值约 4 GB。
- 共 103 个 chunk，2 个 shard 并行：GPU 5（12 核，48-59）和 GPU 4（12 核，60-71），CPU 用 op-train 的空闲行 48-71，避开 carla-rewind 占着的 184-207。table.tsv 已加 `wl-tokens` 行。

## 结果

抽取 13:29 到 14:15 完成，wall 约 46 min（2 shard，单 shard 每 chunk 约 45 s），103 个 chunk，全部 rc 0。`assemble` 后 `tok.npy` 为 (307 587, 24, 1024) fp16，15.1 GB。

全量等价检查（922 761 个 clip，全部相机、全部行）：

| 比较 | 余弦最小 | 中位数 | ≥ 0.999 的比例 |
|---|---|---|---|
| 重算 `mean` 对 `z.npy` 存的 `mean` | 0.9999999 | 1.0 | 100% |
| 网格 token 均值对 `mean` | 0.930 | 0.985 | 0% |

第一行证明对齐和预处理一致；第二行是 last-slice 池化的固有差异，见上文。`tok/` 下的 `m*.npy`（重算 mean，约 1.9 GB）是我自己的检查用副本，留着未删，需要时问过再清。
