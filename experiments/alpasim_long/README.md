# alpasim_long: full-length nuPlan rollouts in local AlpaSim

status: live
decisions: 262
index: raw nuPlan logs + vehicle-only rebuild give 18-decision scenes (was 10)
key: experiments/alpasim_long/plans/2026-10-12-long-prereg.md, experiments/alpasim_long/scripts/al_chain.sh, experiments/alpasim_long/scripts/build_cache.py, experiments/alpasim_long/scripts/nuplan_zip.py, experiments/alpasim_long/scripts/fetch_logs.sh, experiments/alpasim_long/scripts/long_read.py, experiments/onpol_nuplan/results/long/smoke.md, experiments/onpol_nuplan/results/long/span.csv

**Question.** Can local AlpaSim nuPlan rollouts run longer than the shipped 10 decisions, how long, at what cost, and does the candidate
ordering (YR10m10-F, P2H10S-F, SSELFnoL-ON) hold on the 460 held scenes as scenes lengthen?

**Conclusion.** The 10-decision limit was the organisers' prebuilt trajdata cache, not AlpaSim: rebuilt from the raw nuPlan test logs
(136 logs, 7.7 GB fetched by range reads out of the public 96 GB archive) with a vehicle-only agent filter, the cache is identical to the
shipped one at the shipped window on all 1491 public scenes (G0), rollouts on it are bit-identical (G1), and at `num_timesteps_after=160`
1477 / 1491 scenes run 18 decisions (G2 on 10 train scenes: 10 / 10, 0 driver errors). The 18-decision limit is the MTGS reconstructions'
camera coverage, not the data. The held read is below (decision 262).

**Next.** The held 460 read at 18 decisions (running, see "The held read"); then closed-loop training at 18 decisions on the train
scenes and an 18-decision selection set.

**Read more.** [plans/2026-10-12-long-prereg.md](plans/2026-10-12-long-prereg.md); the earlier dead end:
[../onpol_nuplan/results/long/smoke.md](../onpol_nuplan/results/long/smoke.md).

## What limits the length

| Bound | Where | Length it allows |
|:--|:--|:--|
| Prebuilt trajdata cache (`datasets/alpasim_nuplan/nuplan_test`, 111 frames at 20 Hz per scene) | organisers' download | 10 decisions on every scene (the old limit; the window override is inert while it exists) |
| Raw nuPlan logs (rebuilt cache, this topic) | `$DATA_DIR/datasets/nuplan_raw/test`, 136 logs | minutes; 1477 / 1491 scenes reach 18 decisions at `after=160`, 14 stop earlier because the log ends |
| MTGS reconstruction's camera coverage (`assets/<scene>/video_scene_dict.pkl`) | renderer assets | held: 457 / 460 reach 18 decisions, 332 / 460 reach 20 (token + 9.0 s), 234 / 460 reach 22 |
| `n_sim_steps: 200` | shipped base.yaml | 200 decisions (the runtime clips it to the scene) |

So **18 decisions (token + 8.0 s, `num_timesteps_after=160`) is the maximum length the reconstructions support on almost every scene**;
beyond it the renderer draws views outside the reconstruction for a growing share of scenes (not tried). Private suite length: no
shipped config, doc or CLI says (alpasim 0bb4c4b and origin/main 2023534, alpasim-cli on the Tokyo box; the private presets are not
shipped), so "the private scenes are longer" stays a hypothesis.

**Why the previous attempt died** (job 1012-002841-ddb1): `+e2e_challenge_nuplan=full` is only the all-scenes preset; the job also set
`cache_location=<new dir>`, so trajdata tried to rebuild from raw logs that were not on the box and from a devkit that is not installed,
and the renderer exited before it listened.

**How the rebuild works** (`scripts/build_cache.py`, AlpaSim's venv unedited):
- raw logs: `scripts/nuplan_zip.py` reads only the 136 members of `nuplan-v1.1_test.zip` (S3 serves byte ranges; 7.7 GB compressed,
  13.2 GB on disk, ~1 h through Clash). Assumption: covered by the nuPlan licence the project already works under.
- devkit: only for nuPlan's log -> split table: PyPI `nuplan-devkit==2.0.0 --no-deps` in `$DATA_DIR/envs/nuplan_devkit_overlay` plus
  the splitter yaml from GitHub (the wheel omits it); maps are the shipped prebuilt files, symlinked (`require_map_cache=False`).
- **vehicle-only agents:** the shipped cache has only nuPlan category `vehicle`; trajdata's shipped query also takes bicycles and
  pedestrians (119 against 32 agents in one scene). The organisers filtered outside the shipped code; the builder drops non-vehicles at
  the query. With that, every vehicle row matched to 0.0.
- only the 1491 central tokens that name public scene ids are built (the configs list 12 146); about 12 min per window at 32 workers.
- outputs `$DATA_DIR/datasets/alpasim_nuplan_long/after{80,160}` (525 / 738 MB). Use them with
  `runtime.scene_provider.trajdata.cache_location=$DATA_DIR/datasets/alpasim_nuplan_long/after160
  runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=160` appended to `c0b_chain.OVERRIDES`.

## Gates ([plans/2026-10-12-long-prereg.md](plans/2026-10-12-long-prereg.md))

| Gate | Requirement | Measured | |
|:--|:--|:--|:--|
| G0 cache | `after80` rebuild = shipped cache on the 1491 public scenes | 1491 / 1491 exact (length, every agent row, agent list, traffic lights, split) [results/cache/g0_compare_summary.json](results/cache/g0_compare_summary.json) | pass |
| G1 rollouts | smoke10, YR10m10-F-s0, shipped against `after80`: no zero flip, scores within 0.01, plans <= 0.03 m on 99 % | 0 flips, 0.0 score difference, 100 / 100 plans identical (0.0 m) | pass |
| G2 long smoke | `after160`: 10 / 10 rollouts, 0 driver errors, 18 decisions each | 10 / 10, 0 errors, 18 decisions on all 10 | pass |

Decisions 0..9 of the long run are not served the same inputs as the 10-decision run (median plan difference 0.04 m, max 0.51 m at
decision 7): the route generator reads the recorded path of the whole window, as expected in the prereg; reported, not gated.

## Smoke: 10 train scenes, YR10m10-F-s0, 10 against 18 decisions

[results/smoke/](results/smoke/) (`smoke_table.md`, `smoke.json`, `per_scene.csv`). Train scenes, so not a read.

| scene | score 10 | score 18 | zero kind at 18 | GT dist 10 | GT dist 18 | GT dist 18, k <= 10 | at-fault 10 / 18 | logged path m 10 / 18 |
|:--|--:|--:|:--|--:|--:|--:|:--|:--|
| `0083_00485-34f30283d3bd53ec` | 1.000 | 1.000 |  | 0.11 | 0.15 | 0.11 | 0 / 0 | 43 / 60 |
| `1690_02183-30f09e001cf55013` | 1.000 | 0.000 | offroad | 2.73 | 2.95 | 2.70 | 0 / 1 | 23 / 37 |
| `4059_04203-6225b347244658c1` | 1.000 | 0.000 | offroad | 6.04 | 3.08 | 2.59 | 0 / 1 | 7 / 29 |
| `3385_04952-e61699cda2f15e01` | 1.000 | 1.000 |  | 0.13 | 1.67 | 0.14 | 0 / 0 | 27 / 37 |
| `1116_01336-0ba54149d1575f95` | 1.000 | 1.000 |  | 0.33 | 0.37 | 0.32 | 0 / 0 | 23 / 56 |
| `1099_01351-9af711dfa6eb5952` | 1.000 | 0.890 |  | 0.10 | 0.35 | 0.10 | 0 / 0 | 15 / 41 |
| `1749_03230-89e02236312d5038` | 1.000 | 0.890 |  | 3.72 | 3.73 | 3.73 | 0 / 0 | 28 / 61 |
| `0438_00833-887e4d57da835b12` | 0.980 | 0.770 |  | 0.21 | 0.35 | 0.21 | 0 / 0 | 10 / 28 |
| `0964_01689-51259bf89422548a` | 1.000 | 0.795 |  | 0.27 | 0.51 | 0.26 | 0 / 0 | 36 / 70 |
| `1245_02064-1c0c94dbcd5359fc` | 0.000 | 0.000 | offroad | 0.65 | 0.65 | 0.65 | 1 / 1 | 17 / 32 |
| **mean** | **0.898** | **0.635** | 3 zeros | 1.43 | 1.38 | 1.08 | 1 / 3 | 23 / 45 |

Two of the three zeros at 18 are new, both offroad after decision 10 (at k <= 10 of the same rollout both were inside the corridor with
no event). Five scenes drop from 1.0 to 0.77-0.89 on progress (the logged path doubles, 23 -> 45 m). GT distance changes little on these
10 because offroad truncates the series. Cost: runtime wall 10.6 s against 6.4-6.9 s per scene (10 scenes, start-up dominated), runtime
RSS 8.4 GiB, renderer VRAM 11.7 GiB, disk 46 against 27 MB per scene before pruning.

<!-- files:begin -->
<!-- files:end -->
