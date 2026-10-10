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

2026-10-09 cleanup (user-approved, script and log in `tmp/cleanup_1009.*` on the data disk): 473G -> 892G free.
Removed `runs/openpilot_rigs/frames` (138G, model_smoke) and the per-episode `scene.ply` / `ground.ply` dumps under
`runs/{bench/hugsim,op_parity/hugsim,op_adapt_H/hugsim,op_guard,hugsim-derot}` (11058 files, ~300G; scores, logs and
videos untouched). `runs/hugsim-exam` keeps its ply (35G): `experiments/hugsim/scripts/` spin attribution and
`export_review_cases.py` read them. HUGSIM writes two ply per episode (~27 MB each), so new bench runs refill this.

2026-10-10 cleanup (user-approved, scripts and log in `tmp/cleanup_1010*` on the data disk): 268G -> 827G free.
CARLA / Bench2Drive bulk data of the closed CARLA lines (374G): `runs/b2d_collect/data`,
`runs/op_route_cmd/{carla_pairs_s10000,carla_pairs_s10000ol,carla_pairs_s2000,carla_render_s200}`,
`runs/op_route_ft/{bank,carla_near,synth,gif,plan_track_gif}`, `runs/nq4/{k,gk,cx}`, `runs/b2d_privileged_ceiling/arms`,
`processed/carla_*`, `datasets/bench2drive-mini`. Kept: `runs/op_route_ft/{runs,onnx}` (trained weights, 15G), small
results and logs, `third_party/carla` (30G), the leaderboard-model envs, `runs/op_parity/cache/b2d_v2` (24G).
Full rollout logs of concluded AlpaSim runs (185G): `rollout.asl` under `runs/alpasim/{pai2,c0,cf1,fix1,ot3,c0c,ot2,m1,c0b}`,
one `.asl` per `rollouts` dir kept as a sample (16G); scores, metrics parquet, csv, logs, videos and `aggregate/`
untouched, so these runs can be re-read but no longer replayed. `runs/alpasim/ap2` (120G) is the AP2 token cache, not
rollouts; untouched.

2026-10-11 cleanup (user-approved, script and log in `tmp/cleanup_1011.*` on the data disk; the disk had filled to 280G
free after `runs/vis_train/{px,px_side}` grew to 534G). Removed only material of concluded or paused lines (plan.md section 5)
with no live reader in `jevdrive/`, `lib/` or non-archive topic scripts, about 560G: the CARLA parts of paused lanes
(`runs/vlm_arb/arms`, `runs/op_img_cmd/{carla,ft/bank/skycarla}`, the `*carla*` entries of `runs/op_adapt_H/{fixbank,samples,bank}`,
`processed/op_adapt/p5`, `ckpt/nq4_p3*`, `runs/{jfa,rig122}`); the dropped world-model line (`runs/wl`, `processed/wl*_gen`,
`envs/comma-wm`, `models/comma`); envs and weights of concluded third-party lines (drivestudio, r3d2, p3-wodprep, statepol,
qwen-drive, autovla, drivor, gtrs, sparsedrivev2, gdino, efficientsam3, hugsim-ltf, depth, openjev); the early frozen-feature
caches (`processed/waymo_e2e/features/{qwenvid_train_t4,qwen_front3,qwen_grid_p2}`, `processed/drive_backbones/op_trainval`,
`processed/{elicit_e2,waymo_ds,waymo_ds_veh,night2,hugsim_pairs,hugsim_pairs_10hz}`); concluded runs (`runs/{op_adapt_r2,
infra-accept,scale_check,unified,nq3,waymo_heads,op_cosmos_probe}`); and the per-episode ply of concluded HUGSIM runs and of
`runs/bench/hugsim` (refilled since 10-09). Kept on request: every `datasets/` entry (`waymo_perception`, `womd`); kept by
earlier decisions: `runs/cosmos*` samples, `runs/hugsim-exam`, `runs/hugsim-wajepa` (the ply fallback of
`experiments/lowboard_diag/scripts/lbd_hugsim.py`). Each new HUGSIM bench run writes ply again (~27 MB x 2 per episode).

Live but regenerable (ask first):
`runs/op_adapt_H/{fixbank,bank,samples,bank2}` non-CARLA parts ~300G (H line, last trained 2026-10-03, no live reader; `runs/`
and `onnx/` stay for the bench `adapt_h` family), `runs/op_img_cmd/ft/{bank,runs}` 48G, `runs/op_adapt_L` 56G,
`runs/op_lb` side caches (`lb_hq_*`, `hq`, `g_*`, `lb_guardneg`, `lb_h1train`, `lb_imgtrain`) 42G, `runs/op_parity/runs/smoke-vt-*`
24G, `processed/op_adapt/nusc` 41G, `datasets/comma1M` 21G, stale engines in `runs/op_interp/trt_cache` and
`models/openpilot/trt_cache` (118G in total, prune by model name), `runs/{vlm_thin,vlm_cmp,opctrl,opctrl_long}` 31G.
Paused CARLA / Bench2Drive material deliberately kept on 10-10 (`third_party/carla`, the leaderboard-model envs,
`runs/op_route_ft`, `runs/op_parity/cache/b2d_v2`) is about 110G. Re-scan before deleting anything.
