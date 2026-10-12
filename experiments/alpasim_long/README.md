# alpasim_long: full-length nuPlan rollouts in local AlpaSim

status: live
decisions: 262
index: 18-decision scenes (was 10); held 460: SSELFnoL-ON 0.821, P2H10S-F 0.787, YR10m10-F 0.776

**Question.** Can local AlpaSim nuPlan rollouts run longer than the shipped 10 decisions, how long, at what cost, and does the candidate
ordering (YR10m10-F, P2H10S-F, SSELFnoL-ON) hold on the 460 held scenes as scenes lengthen?

**Conclusion.** The 10-decision limit was the organisers' prebuilt trajdata cache, not AlpaSim: rebuilt from the raw nuPlan test logs
(136 logs, 7.7 GB fetched by range reads out of the public 96 GB archive) with a vehicle-only agent filter, the cache is identical to the
shipped one at the shipped window on all 1491 public scenes (G0), rollouts on it are bit-identical (G1), and at `num_timesteps_after=160`
1477 / 1491 scenes run 18 decisions (G2 on 10 train scenes: 10 / 10, 0 driver errors). The 18-decision limit is the MTGS reconstructions'
camera coverage, not the data. On the 460 held scenes (2 seeds) the GT-distance ordering holds from 10 to 18 decisions (SSELFnoL-ON <
YR10m10-F < P2H10S-F), SSELFnoL-ON stays first by score and separates (+0.044 [+0.018, +0.068] over YR10m10-F), and YR10m10-F drops to
last on at-fault zeros (42), tied with P2H10S-F; YR10m10-F-s0 at 18 decisions (0.771 / 0.474 km / 1.49 m) is near its private result
(0.80 / 0.529 km / 1.733 m) (decision 262).

**Next.** Amendment 2 (APY10m10-AB-s0 at 18 and 10 decisions against its private 0.640) is queued in the pool (jobs
1012-031731-cb61 / 1012-031732-3cd6, read-out 1012-032433-78f7 -> `$DATA_DIR/runs/alpasim_long/held/read_apy/report.md`; not read yet); then closed-loop
training at 18 decisions on the train scenes and an 18-decision selection set.

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

## The held read: 460 held scenes, 10 against 18 decisions

[results/held/report.md](results/held/report.md) (all columns; `report.json`, `per_scene.csv`). Six checkpoints at 18 decisions (`after160`),
YR10m10-F s0 / s1 new at 10 decisions; P2H10S-F and SSELFnoL-ON at 10 are onpol_nuplan's stored held reads (same driver path). 454 of
460 scenes start a session at both lengths (6 route fold-back scenes never do, as before); 446 / 460 run all 18 decisions, 14 end at 15-17
because their log ends.

| Recipe (2 seeds) | score 10 | score 18 | zeros 10 / 18 | at-fault zeros 10 / 18 | corridor zeros 10 / 18 | at-fault km 10 / 18 | GT distance 10 / 18 m | GT distance, first 10 decisions of the 18 run | drift slope k 4..10 / 4..18 m per decision |
|:--|--:|--:|:--|:--|:--|:--|:--|--:|:--|
| `SSELFnoL-ON` | **0.939** | **0.821** | 20.5 / 61.5 | 9.5 / 27.5 | 5 / 27 | 1.38 / 0.78 | **0.92** / **1.41** | 0.67 | 0.131 / 0.116 |
| `YR10m10-F` | 0.930 | 0.776 | 24.5 / 80.5 | 15 / 42 | 3.5 / 34 | 0.85 / 0.49 | 0.93 / 1.45 | 0.68 | 0.133 / 0.134 |
| `P2H10S-F` | 0.923 | 0.787 | 28 / 75.5 | 13.5 / 31.5 | 8.5 / 38 | 0.95 / 0.66 | 1.01 / 1.58 | 0.74 | 0.148 / 0.150 |

Paired recipe differences (seed mean per scene, log-clustered bootstrap over 39 logs, 10 000 draws):

| | score 10 | score 18 | GT distance 10 | GT distance 18 |
|:--|:--|:--|:--|:--|
| SSELFnoL-ON - YR10m10-F | +0.009 [-0.002, +0.023] | **+0.044 [+0.018, +0.068]** | -0.015 [-0.079, +0.042] | -0.038 [-0.153, +0.081] |
| SSELFnoL-ON - P2H10S-F | **+0.016 [+0.006, +0.027]** | **+0.034 [+0.017, +0.049]** | **-0.092 [-0.144, -0.043]** | **-0.164 [-0.224, -0.098]** |
| YR10m10-F - P2H10S-F | +0.007 [-0.006, +0.019] | -0.010 [-0.030, +0.010] | **-0.077 [-0.140, -0.016]** | **-0.126 [-0.229, -0.026]** |

**Does the ordering hold?** By GT distance, yes: SSELFnoL-ON < YR10m10-F < P2H10S-F at both lengths, and every gap widens. By score,
the top holds and separates: SSELFnoL-ON is first at both, and its lead over YR10m10-F goes from +0.009 (CI includes 0) to +0.044 (excludes
0). The other two swap places (YR10m10-F second at 10, last at 18), but that difference includes 0 at both lengths. What moves YR10m10-F
down is at-fault zeros: 15 at 10 decisions, 42 at 18, the most of the three (P2H10S-F 31.5, SSELFnoL-ON 27.5), at the fewest km per
at-fault event (0.49).

Reading the 10-decision metrics against this:
- **Zeros roughly triple** from 10 to 18 decisions for every recipe (20.5-28 to 61.5-80.5), corridor zeros most of all (3.5-8.5 to 27-38).
  The 10-decision suite leaves most failures unseen.
- **The 10-decision GT distance carries an end-of-path term.** In 21-23 scenes per checkpoint the logged car barely moves inside the
  short window (logged path median 7 m) while the ego drives about 12.5 m: it passes the end of the logged path, the distance reaches
  4-5 m and the scene still scores 1.0. In the long window the logged path continues and the same scenes stay at a median 1.6-1.8 m. So
  the long run's first 10 decisions read 0.66-0.75 m against 0.90-1.01 m in the 10-decision runs.
- **Drift slope** over k = 4..18 is the same as over 4..10 for YR10m10-F and P2H10S-F (0.133 / 0.134, 0.148 / 0.150); SSELFnoL-ON's falls
  (0.131 -> 0.116). Ordering by slope: SSELFnoL-ON < YR10m10-F < P2H10S-F at 18, the same as by GT distance.
- **Against submission #2:** YR10m10-F-s0 at 18 decisions reads 0.771 / at-fault 0.474 km / GT 1.49 m. Privately it scored 0.80 /
  0.529 km / 1.733 m; at 10 decisions it reads 0.928 / 0.787 km / 0.957 m. The 18-decision read lands near the private numbers on all
  three metrics; the 10-decision read does not. Amendment 2 tests this on submission #1's checkpoint (below).

Cost: 8 pool jobs, 34-37 min wall per 18-decision job (4.6 s per scene and stack) against 23 min at 10 decisions (1.5x); runtime RSS 33-34
GiB in both; disk 44 MB per scene and job before pruning (121 GB for the six 18-decision jobs), 17 GB after (rollout.asl kept for
zero-score rollouts only).
About 4.5 card-hours booked for the eight jobs (6 cards, 02:34-03:14 CST); the lane in total about 5 card-hours.

<!-- files:begin -->
<!-- files:end -->
