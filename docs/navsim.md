# NAVSIM data

Read this when you need the NAVSIM / OpenScene data on the box, or need to re-download or extend it.

## Where it lives

`$DATA_DIR/datasets/navsim/` (= `OPENSCENE_DATA_ROOT`), in the layout the NAVSIM devkit expects:

| Path | Content | Source archives |
|---|---|---|
| `maps/` | nuPlan maps v1.0 (from `nuplan-maps-v1.1.zip`) | Motional S3 |
| `navsim_logs/trainval/`, `navsim_logs/test/` | OpenScene log pickles (full splits) | `openscene_metadata_{trainval,test}.tgz` |
| `sensor_blobs/trainval/` | navtrain frames, current + history | `navsim/navtrain_{current,history}_{1..32}.tgz` |
| `sensor_blobs/test/` | test split frames (navtest, and the real frames of navhard_two_stage) | `openscene_sensor_test_camera_{0..31}.tgz` |
| `navhard_two_stage/` | `sensor_blobs/`, `synthetic_scene_pickles/`, `synthetic_scenes_attributes.csv` | `navsim-v2/navsim_v2.2_navhard_two_stage_*.tar.gz` |
| `_archives/` | in-flight downloads and `state/<archive>.done` markers | |

**Camera only.** We never use LiDAR, so `MergedPointCloud` is not on disk: the test LiDAR archives are
not downloaded, and LiDAR files in mixed archives (navtrain, navhard) are skipped at extraction.
Agents must use a camera-only sensor config (no `lidar_pc`).

Devkit env:

```bash
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps
export OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim
export NAVSIM_EXP_ROOT=$DATA_DIR/runs/navsim   # metric caches and eval outputs
```

`navhard_two_stage` uses `data_split: test`, so it reads `navsim_logs/test` and `sensor_blobs/test` plus
`navhard_two_stage/`. Metric caches are built locally from logs + maps; nothing else is downloaded for them.

## How to (re)download

`scripts/tmux_run.sh navsim scripts/download_navsim.sh [maps logs navhard navtest navtrain]`

- Idempotent: archives with a `.done` marker are skipped, `.part` files resume. Re-run the same command after a failure.
- Each archive: download, size + sha256 check (from the HF API), extract with `pigz` (LiDAR excluded), delete.
  8 archives in flight (`JOBS`), so extraction overlaps with the other downloads and disk use stays bounded.
- Logs: `$DATA_DIR/runs/download/navsim/<ts>/{log.txt,events.jsonl,jobs.tsv}` (per-archive size, time, MB/s).

## Network

HF files come from `https://hf-mirror.com` (direct, it redirects to `cas-bridge.xethub.hf.co`).
Measured on 2026-09-20 (West-D box):

| Route | 1 stream | 8-16 streams |
|---|---|---|
| huggingface.co direct | blocked | |
| hf-mirror.com direct | 8 MB/s | 16 MB/s total |
| `/etc/network_turbo` + huggingface.co | 0.6 MB/s | 2.8 MB/s total |
| Clash + huggingface.co | | 16.6 MB/s total |
| hf-mirror direct + Clash at the same time | | 16.2 MB/s total |

Every route tops out at ~16 MB/s total (~130 Mbit/s), so the box's bandwidth is the limit, not the route.
Motional S3 (maps) is unreachable direct; the script fetches that one file through Clash (8.7 MB/s).

## Sizes

TBD

Last verified: 2026-09-20
