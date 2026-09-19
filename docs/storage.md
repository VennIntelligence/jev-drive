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

`/autodl-pub/data` (read-only, region West-B) already has nuScenes (full, 549G of `.tgz`), KITTI,
SemanticKITTI, cityscapes and more. They are archives: extract what you need into `datasets/`.
Do not download these again.

## Region

The box is in West-B. autodl-fs only mounts inside one region, so rent multi-GPU boxes in West-B too.

Last verified: 2026-09-19
