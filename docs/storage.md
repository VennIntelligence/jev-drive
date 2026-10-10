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

Small-VLM first look (2026-10-08, d182): `models/gemma-4-E2B-it` 9.9 GB kept (ModelScope copy, bf16); `Qwen/Qwen3.5-2B` (4.5 GB) and `Qwen/Qwen3.5-4B` (9.2 GB) were
downloaded the same way and deleted 2026-10-08 after the first look (re-download with `small_vlm_pdl.py`, ~10 min each); `envs/svlm-extra` is a `--target` dir with flash-linear-attention / fla-core / einops
that the Qwen3.5 runs put on `PYTHONPATH` (the shared `.venv` is untouched). hf-mirror gave ~1 MB/s in total, ModelScope caps
one connection at ~0.6 MB/s but scales with connections: `experiments/vlm_arb/scripts/small_vlm_pdl.py` (16 MB ranges, 64
streams) reached 14-15 MB/s.

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

## Cleanup candidates (scan 2026-10-08)

Disk 5.0T, 4.5T used, 529G free after the 2026-10-08 cleanup (user-approved, script and log in `tmp/cleanup_1008.*` on
the data disk), which removed about 339G and kept small samples:

| Removed | Kept |
|---|---|
| `runs/cosmos_full/{pairs,gen,clips,stage10}` (cosmos no-go, d56/d63) | `pairs_sample/` (20 pair dirs), `stage1/`, logs |
| `runs/cosmos/{out,clips,gen}` | `anchor/`, `gen-cabincam/`, small dirs |
| `models/cosmos` (Cosmos-Reason1-7B, siglip2 naflex, Predict2.5), `envs/cosmos-transfer` | nothing; `vlm_arb_models.py`'s `cosmos-reason1-7b` arm needs a re-download |
| `datasets/nuscenes/sweeps` camera and lidar dirs (re-extract from `/autodl-pub`) | `CAM_BACK` (0.3G), radar |
| `runs/factor_wm/clips/{g0a,g0b,g1s,train}`, `runs/factor_wm/p2op/{logged,roll}` (line dropped, d177) | json indexes, reports, `onnx/`, `runs/` |

Still removable from concluded lines: `runs/openpilot_rigs/frames` 141G (model_smoke). Live but regenerable (ask first):
`runs/op_adapt_H/{fixbank,bank}` 151G + 122G, `processed/op_adapt` 224G, `runs/op_route_cmd/carla_pairs_s10000` 65G.
Bench2Drive / CARLA data is about 580G in total (`runs/{op_route_cmd,op_route_ft,b2d_collect,p5v1,p6,nq4}`,
`processed/carla_*`, `third_party/carla`, the four leaderboard-model envs); untouched. Re-scan before deleting anything.
