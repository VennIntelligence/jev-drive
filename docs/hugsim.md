# HUGSIM data

Read this when you need the HUGSIM closed-loop benchmark data on the box, or need to re-download it.

## Where it lives

`$DATA_DIR/datasets/hugsim/`, mirroring the HF repo paths (not the HF hub cache layout, so eval scripts
can address files by plain relative path):

| Path | Content | Source |
|---|---|---|
| `3DRealCar/` | vehicle 3DGS assets (330 files, `gs.pth` + `wlh.json` pairs) | `XDimLab/HUGSIM/3DRealCar` |
| `scenes/kitti360/`, `scenes/nuscenes/`, `scenes/pandaset/`, `scenes/waymo/` | reconstructed scene 3DGS assets, one per sequence | `XDimLab/HUGSIM/scenes/*` |
| `nusc_map_cache.zip` | nuScenes map cache | `XDimLab/HUGSIM` |
| `scenarios.zip` | 400+ scenario configs (yaml) | `XDimLab/HUGSIM` |
| `sample_data/data.zip` | smoke-test data for the install/port | `hyzhou404/HUGSIM/sample_data` |

This is the already-reconstructed, exported benchmark (`export_scene.py` output), not raw
KITTI-360/nuScenes/PandaSet/Waymo sensor data — running `closed_loop.py` needs nothing else. Raw sensor
data is only needed to reconstruct new scenes with `train.py`. See
[research/lit/research_notes/开源驾驶模型与OpenPilot打榜现状/hugsim_resources.md](../research/lit/research_notes/开源驾驶模型与OpenPilot打榜现状/hugsim_resources.md)
for the full survey (size breakdown, Blackwell/CUDA install risk, agent interface).

Total ≈ 61 GB benchmark + 2.4 GB sample data ≈ 63.4 GB.

## How to (re)fetch

`scripts/hugsim_fetch.py`, built on `jevdrive/hfdl.py` (the same parallel range-request downloader used
by `scripts/alpamayo_fetch.py`):

```bash
python scripts/hugsim_fetch.py sample_data   # ~2.4 GB, fetch first (needed for the install/port smoke test)
python scripts/hugsim_fetch.py benchmark     # ~61 GB, everything else
```

Run in tmux `jev` window `hugsim-dl`: `scripts/tmux_run.sh hugsim-dl python scripts/hugsim_fetch.py sample_data benchmark`.

Resumable: a file already at the right size is skipped; a stalled chunk within a file is retried. Every
LFS file is sha256-checked against the HF API's reported oid, every non-LFS file against the git blob
sha1 (see `jevdrive/hfdl.download`); a mismatch raises instead of leaving a silently-corrupt file.

## Network

Both repos (`XDimLab/HUGSIM`, `hyzhou404/HUGSIM`) serve LFS files through Xet storage, redirected to
`cas-bridge.xethub.hf.co`. Measured 2026-09-24 (West-D box), 4 parallel streams, 95 s sustained:

| Route | Sustained |
|---|---:|
| `hf-mirror.com` direct | 9 MB/s |
| `proxy_on` (Clash) + `huggingface.co` | 8 MB/s |

No ModelScope mirror of either repo exists. Direct `hf-mirror.com` wins narrowly and costs no Clash
quota, so it is the default (`jevdrive.hfdl.HF_MIRROR`); no proxy needed. At ~9 MB/s, 63.4 GB is
roughly 2 hours.

Last verified: 2026-09-24
