# Versions (official / Tokyo / box) and the render frame-time issue upstream (2026-10-22)

**Question.** (1) Do the official AlpaSim E2E challenge evaluation, our Tokyo reads and our box reads run the same AlpaSim
commit, presets, renderer image, scene artifacts and Harmonizer weights? (2) Is the frame-time effect of
[../../alpasim_human/results/render_lag.md](../../alpasim_human/results/render_lag.md) (the runtime sends sim time as
`frame_start_us` / `frame_end_us`, so an ego that lags or leads the log car on the path gets a degraded background) known or
fixed upstream? Read-only: `gh` on NVlabs/alpasim, nvcr.io registry API, Hugging Face API, the challenge Space, ssh to Tokyo and
the box (no jobs touched).

## 1. Versions

| Component | Official evaluation (source) | Tokyo | GPU box |
|---|---|---|---|
| AlpaSim commit | `0bb4c4b` ("Current server is using 0bb4c4bf...", challenge Space). Branch `e2e_challenge` HEAD is now `e121e59` (10-20) | `/data/third_party/alpasim` detached at `0bb4c4bfe109`, clean | `~/data/third_party/alpasim` detached at `0bb4c4bfe109`, clean (one stray untracked file `'/`) |
| Scorer | 0bb4c4b, but organisers said on 10-19 they will recompute the board with the #198 fix (`1b024f0`, PR #215, merged 10-20) "this week" | 0bb4c4b (without #215) | 0bb4c4b (without #215) |
| PAI preset | `+e2e_challenge=ec2`: topology `8gpu_32rollouts`, `render_bundling: BATCH_RENDER_RGB`, `nre_max_workers 1`, cache 2, unharmonised renderer command, video off. README: the runner "supplies ... any production-only renderer image/entrypoint overrides for official evaluation" | `+e2e_challenge=dev` (1gpu, `render_bundling: NONE`, `nre_max_workers 4`, cache 5, video on), `--enable-harmonizer` dropped by `pai_run.sh`, CONC 4 | same `dev` preset via `run_native.py`, harmonizer dropped by default; `ALPASIM_FAST=1` default (non-driver cameras at 64 px) |
| PAI renderer image | `nvcr.io/nvidia/nre/nre-ga:26.04` is the preset default (`competition_ec2.yaml`, `NRE_IMAGE` overridable); the production image actually used is not published | `nre-ga:26.04`, manifest `sha256:97f43e7130c5...dea284`, config `a6837f387712...`, created 2026-05-26 | rootfs unpacked from the DaoCloud mirror of `nre-ga:26.04`: saved manifest sha256 `97f43e7130c5...dea284`, config `a6837f387712...`: identical to Tokyo |
| nvcr.io today | `26.04` = `26.04.01` = `26` = `latest` = `97f43e71...` (registry API, 10-22). `26.04.00` = `1c3a8384...` (older), `26.02` = `dbb6be50...`. No newer NRE than ours | | |
| Physics / controller / runtime | run from the trusted `alpasim-base` image (`ALPASIM_IMAGE`), built by the organisers; digest not published | `alpasim-base:0.89.0` built locally 2026-10-09 from 0bb4c4b (`bbac5754f385`); package versions float with build date (`exclude-newer = "3 days"`, no `uv.lock`: issue #207, open) | no Docker: shared uv venv from the same checkout, torch 2.9.1 (docs/alpasim.md "Running it on our box") |
| PAI scenes | "an internal set of NuRec-compatible scenes similar to those available in the public NuRec dataset" (Contestant Guide); revision not stated | public curated val split: `sim_scenes.csv` (916 scenes, `26.01`, NRE `26.1.112`) + `sim_scenes_2604.csv` (1448, `26.04`, NRE `26.4.96`); our 40-scene list is 12 x 26.01 + 28 x 26.04 | same files (first 10 sha256-equal to Tokyo) |
| nuPlan assets | `OpenDriveLab/AlpasimChallenge2026_nuplan_track` (dataset last modified 2026-08-26, after the #156 cache fix) | n/a | fetched 2026-10-08/09, after the last upstream change |
| Harmonizer | off on the public board ("disabled for initial runs"); **on for PAI final runs**, off for nuPlan final runs (challenge Space) | `harmonizer_nontemporal.pt` sha256 `ece8e2da...` | same file, same sha256 |
| Harmonizer upstream | HF `nvidia/Harmonizer` (`DiffusionHarmonizer` redirects) main `20ca33d`, 2026-07-21: `harmonizer_nontemporal.pt` LFS oid `ece8e2da...` = ours; the repo also holds `diffusion_harmonizer.pkl` (5.0 GB), which file production loads is not stated | | |

**Verdict.** Same code, same renderer bits, same Harmonizer weights on all three, as far as anything is published. The known
differences: (a) the official scorer is about to move to `1b024f0` (lateral-corridor fix, #198/#215), our checkouts are at 0bb4c4b;
(b) preset `dev` vs `ec2` (bundling, worker count, topology, video; should not change pixels, not measured); (c) the organisers may
override the renderer image/entrypoint in production, unpublished; (d) the private PAI scenes are internal, not the public
26.01/26.04 files; (e) PAI final runs turn the Harmonizer on, our default reads have it off; (f) floating Python packages in
`alpasim-base` (issue #207). Box-only: `ALPASIM_FAST=1`.

## 2. Frame-time issue upstream

- **Code unchanged everywhere.** `construct_rgb_render_request` sends `frame_start_us=trigger.time_range_us.start`,
  `frame_end_us=...stop` (sim time) at 0bb4c4b (l. 302-303), `e2e_challenge` HEAD `e121e59` (same lines) and `main` `affc2ea`
  (l. 332-333). 0bb4c4b even asserts the request time equals the trigger (`_validate_rgb_metadata_matches_trigger`). No commit,
  branch or open PR changes which timestamp is sent; no "static background time" option exists in the wizard or runtime config.
- **Renderer flags.** The NRE image ships its Python source (`/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/nre/`,
  so not closed source as render_lag.md says). `nre/grpc/grpc_server_config.py` has no time-related option (artifact glob, host/port,
  renderer backend, harmonizer, ray chunk, hood dir, caches, workers, `enable_editing_actors`, metrics dir). `serve.py`
  `build_ray_bundle` passes the request time into the camera `PoseRange`, and the Gaussian feature collector has a scene-wide
  ("Holistic") time embedding with Fourier temporal modulation of the SH features (`nre/models/gaussians/collect/collector.py`):
  the background's appearance is a function of the requested time, consistent with the probe. Mechanism read from code, not traced
  per scene.
- **Not reported as such.** No issue, PR or discussion in NVlabs/alpasim (all issues/PRs searched for blur, artifact, timestamp,
  frame_start_us, lag, behind, background, NRE, nurec; 4 discussions) names the time-vs-pose effect. HF Space and NuRec
  dataset discussion pages returned 403 (not read); the NVIDIA NuRec forum was not read.
- **Related, acknowledged generally.** [#115](https://github.com/NVlabs/alpasim/issues/115) "Rendered RGB not good": mwatson-nvidia,
  "Nurec fails when one diverges too far from the original recording pose"; the remedy is the evaluation cutoff
  (`max_dist_to_gt_trajectory`, 4 m). [#192](https://github.com/NVlabs/alpasim/issues/192): "once ego diverges from the recorded path,
  when is a rollout still a valid policy evaluation? This is an unsolved problem. We typically use the 4m tunnel as a proxy";
  [#193](https://github.com/NVlabs/alpasim/issues/193) closed wont-fix.
- **The 4 m tunnel does not see this.** `dist_to_gt_trajectory` is `full_gt_linestring.distance(ego centroid)`
  (`src/eval/src/eval/scorers/ground_truth.py`, 0bb4c4b), the distance to the whole recorded path, not to the time-synced log pose
  (that is `dist_to_gt_location`, unused by the cutoff; #193 states the same). The cutoff (`aggregation/main.py` l. 155-159) therefore
  keeps an ego that is seconds behind or ahead on the path inside the evaluated window while its images degrade. Lag is scored on
  degraded frames; that is a gap in the organisers' stated remedy, not a documented intent.
- Unrelated fixes with "timestamp" or "ghost" in them ([#111](https://github.com/NVlabs/alpasim/pull/111) /
  [#160](https://github.com/NVlabs/alpasim/pull/160), [#156](https://github.com/NVlabs/alpasim/issues/156)) concern nuPlan MTGS stale actors.

## Recommendations (not applied)

1. No upgrade available: the renderer is the newest NRE tag and the deployed AlpaSim commit is the one we run. Optionally cherry-pick
   `1b024f0` into a local eval copy (scorer only) once the board is recomputed, so local scores track the board.
2. Treat lag/lead as an image-quality risk the evaluation does not filter: a policy slower than the log is evaluated on degraded
   frames. Model-side: train or augment on time-shifted renders (frame time = sim time, pose behind on the path), or prefer
   keeping up with the log speed. A runtime change (frame time = log time nearest the ego pose, actors at sim time) is not allowed
   for submissions; locally it would be a diagnostic only.
3. For PAI final-run parity, run candidate reads also with `HARMONIZER=1` (weights already identical to upstream).
4. If the user wants it raised upstream: an issue with the render_lag probe (lag_time vs lag_actors PSNR) and the
   `dist_to_gt_trajectory` vs `dist_to_gt_location` point; posting is the user's action.
