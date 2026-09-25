# HUGSIM patches

Minimal changes to third-party HUGSIM code so it builds and runs on our Blackwell box (sm_120).
`scripts/hugsim/install.sh` applies every top-level `*.patch` here to `$DATA_DIR/third_party/HUGSIM`
(hyzhou404/HUGSIM @ 62c690d), idempotently. Details and evidence: [docs/hugsim.md](../../docs/hugsim.md).

| Patch | Why |
|---|---|
| `0001-closed-loop-model-path-and-agent.patch` | `closed_loop.py`: (a) the exported scenes' `cfg.yaml` carries the authors' absolute `model_path` (`/nas/users/hyzhou/...`), which `cfg.update` copies over ours; point it at `<model_base>/<scene>`. (b) Accept any `--ad <name>` whose launcher is `<name>_path` in the base config (our pipe-protocol agents), not only uniad/vad/ltf. |
| `0002-simple-knn-cfloat.patch` | `simple_knn.cu` uses `FLT_MAX` without `<cfloat>`; nvcc 12.8 / gcc 11 no longer pull it in transitively. |
| `0003-release-compat.patch` | The released data is newer than the code (same three defects as documented by WA-JEPA's HUGSIM harness): scenario `plan_list` assets are named `<id>/postprocess/shadow.pth` but 3DRealCar ships `<id>/{gs.pth,wlh.json}` (strip the suffix, as upstream's own `export_multiple_scenes.py` does); 15 PandaSet `medium_02` scenarios pass `max_t` to `ConstantPlanner`, which takes no arguments (accept and ignore); `IDM.update` indexes an empty arc-length tensor when one waypoint is left (duplicate it). Behaviour is unchanged wherever upstream did not crash. |
| `optional/lqr-heading-fix.patch` | NOT applied by default. Upstream PR #57 (open, unmerged): `traj2control` gives the iLQR tracker a reference heading of about pi/2 minus the true one. Apply it only for the paired "fixed controller" numbers: `git -C $DATA_DIR/third_party/HUGSIM apply patches/hugsim/optional/lqr-heading-fix.patch`. |
| `optional/ideal-tracker.patch` | NOT applied by default; only in the private `ideal` tree of `scripts/hugsim/zs_run.py`, the reference for controller acceptance (todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md). `closed_loop.py` skips `traj2control` and moves the ego exactly along the plan for one step (quadratic through the origin and the first two waypoints) by presetting heading, speed and steer of the unchanged env's bicycle step. Only `closed_loop.py` changes: the env class itself is always imported from the editable install, i.e. `$DATA_DIR/third_party/HUGSIM`, whichever tree runs. |

No source patch was needed for the other extensions: the HUGSIM_splat gsplat fork, tiny-cuda-nn and trajdata
build unchanged against torch 2.8.0+cu128 with `TORCH_CUDA_ARCH_LIST=12.0` / `TCNN_CUDA_ARCHITECTURES=120`.
The version bumps (torch 2.4.1+cu118 -> 2.8.0+cu128, Python 3.11 -> 3.12, open3d 0.18 -> 0.19) live in
`install.sh`, not in a patch to `pixi.toml`.
