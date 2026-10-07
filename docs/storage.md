# Storage

Read this when you need to decide where a model, dataset, checkpoint or output goes on the box.

## Layout

`$DATA_DIR` (= `~/data` = `/root/autodl-tmp/ujs`) is the local data disk. Code reads paths from env, never hard-coded.

| Dir | Holds |
|---|---|
| `cache/` | HF (`$HF_HOME`), torch, pip, uv, modelscope caches |
| `models/` | models that do not come from the HF cache (local copies, converted weights) |
| `datasets/` | extracted datasets we read at training time |
| `processed/` | our own preprocessed data (features, tokens, caches) |
| `ckpt/` | our own checkpoints |
| `runs/` | experiment outputs and logs |

HF models live in the HF cache, so `from_pretrained("Qwen/Qwen3-VL-4B-Instruct")` works without a path.
Add new models to `scripts/download_models.sh`.

## Tiers

- Re-downloadable (HF models, public datasets): not precious. The repo keeps the ids and download scripts.
- Ours (`processed/`, `ckpt/` worth keeping): precious. Once we have any, turn on AutoDL file storage
  (`/root/autodl-fs`, shared by all instances in the same region) and keep an offsite backup too.
- Hot data for training: the local data disk. Grow it when training starts, shrink it after.

## Public data

`/autodl-pub/data` (read-only; West-B and West-D both checked) already has nuScenes (full, 549G of `.tgz`), KITTI,
SemanticKITTI, cityscapes and more. They are archives: extract what you need into `datasets/`.
`scripts/extract_nuscenes.sh mini` or `trainval [member ...]` -> `datasets/nuscenes`
(default member `samples/CAM_FRONT`, ~6 GB; the full trainval is 300 GB+).
Do not download these again.

## Region

The box is in West-E since 2026-09-28 (West-B until 2026-09-20, then West-D). autodl-fs only mounts inside one region,
so rent further boxes in West-E too. autodl-fs is not turned on (`/root/autodl-fs` does not exist; checked 2026-09-28).
`/autodl-pub/data` in West-E carries the same datasets (nuScenes, KITTI, ...).

Last verified: 2026-09-20

## Cleanup candidates (scan 2026-10-06, nothing deleted)

Disk 5.0T, 4.6T used. Largest removable items from concluded lines, all regenerable: `runs/openpilot_rigs/frames` 141G
(model_smoke), `datasets/nuscenes/sweeps` 123G (no repo ref; re-extract from `/autodl-pub`), `runs/cosmos_full` 100G
(cosmos no-go, d56/d63). Live but regenerable (ask first): `runs/op_adapt_H/{fixbank,bank}` 151G + 122G,
`processed/op_adapt` 224G, `runs/op_route_cmd/carla_pairs_s10000` 65G. Re-scan before deleting anything.
