# ALPASIM-LONG pre-registration (2026-10-12, 03:30 JST, written before any long rollout)

Lane ALPASIM-LONG, main box. Asked by the main session after submission #2 (YR10m10-F-s0: private 0.80, GT distance 1.73 m against
0.94 m on the public scenes). The user decided that closed-loop training at full scene length on the public scenes is the main direction
and a private-like local selection set the other; this lane is the prerequisite for both: full-length nuPlan rollouts locally.

## What is known before this lane (no new measurement)

- `+e2e_challenge_nuplan=full` is the shipped preset that selects all navtest scenes (`src/wizard/configs/e2e_challenge_nuplan/full.yaml`),
  not a length setting. Every read since c0b passes it (`c0b_chain.OVERRIDES`).
- The failed job 1012-002841-ddb1 (arm `c160new` of onpol_nuplan's long smoke) died because `cache_location=<new dir>` makes trajdata
  rebuild the cache from raw nuPlan logs: 12 146 `Database file ... not found` and then `No module named 'nuplan'` in the renderer, which
  exits before it listens. Scene length is set by the prebuilt trajdata cache (111 frames at 20 Hz = token - 1.5 s .. token + 4.0 s,
  10 decisions); the window override is inert while that cache exists (onpol_nuplan results/long/smoke.md).
- Private suite length: no shipped config, doc or CLI states it (searched: alpasim 0bb4c4b and origin/main 2023534 on the box,
  alpasim-cli on the Tokyo box; same commits). The only statement is base.yaml's "All public and private NuPlan challenge presets inherit
  this shared setting" (about the route generator) and offroad.py's note that the private maps have long boundary segments. The private
  presets themselves are not shipped. That the private scenes are longer is a hypothesis (aud1_private_gap.md argued the opposite).
- What the scenes would carry (onpol_nuplan results/long/span.csv, all 1491): the nuPlan log continues for minutes; the MTGS
  reconstruction's own camera frames end at a median token + 10.2 s; held 457 / 460 reach the 18-decision mark (token + 7.52 s), 439
  reach token + 8.0 s, 332 token + 9.0 s, 234 token + 10.0 s.

## New facts this lane found before registering

- The raw nuPlan v1.1 test-split logs are in the public archive `nuplan-v1.1_test.zip` (95.9 GB, 1350 sqlite logs; the archive serves byte
  ranges). The 136 logs behind the 1491 public scenes are 7.7 GB compressed (13.2 GB raw), the 39 behind the held 460 are 2.1 GB.
  `scripts/nuplan_zip.py` reads only those members. Assumption (the user is asleep): fetching these logs is covered by the same nuPlan
  licence the project already works under for NAVSIM and the AlpaSim nuPlan data (non-commercial research); they stay on the box data disk.
- trajdata needs the devkit only for nuPlan's log -> split table (`create_splits_logs`) and to rebuild maps. The PyPI wheel
  `nuplan-devkit==2.0.0` installed `--no-deps` into `$DATA_DIR/envs/nuplan_devkit_overlay` (AlpaSim's venv untouched) plus the splitter yaml
  from github.com/motional/nuplan-devkit master (the wheel omits it) gives the table (test: 1349 logs, all 136 needed among them).
  Maps are not rebuilt: the shipped prebuilt map files are symlinked into the new cache (`build_cache.py`, `require_map_cache=False`).

## Chosen length

`num_timesteps_after = 160` (token + 8.0 s, 19 x 0.5 s of rig trajectory, **18 decisions**), the longest window that the reconstructions
cover on at least 95 % of the held scenes (439 / 460 to the last frame, 457 / 460 to the last decision). 20 decisions would leave 28 % of the
held scenes rendered beyond their reconstruction. `n_sim_steps` stays 200 (the runtime clips it to the scene). Everything else as the
lane's reads (`run.sh sh30`, `c0b_chain.OVERRIDES`); the only added overrides are
`runtime.scene_provider.trajdata.cache_location=<rebuilt cache>` and `...dataset.extra_params.num_timesteps_after=<N>`.

## Gates (stop at the first failure and write down why)

- **G0 cache fidelity.** Rebuild at `after = 80` from the raw logs (`cache_a80`). Against the shipped cache, on all 12 146 scenes of the
  shipped scene list: same names, locations and splits, same length, agent table at dt 0.1 identical (max abs difference 0 on every
  numeric column, same agent ids per step), same agent list, traffic-light table byte-identical. Pass: all 1491 public scenes exact
  (any other central token that differs is reported). A difference means the raw logs or the builder are not what the organisers used,
  and nothing below is run.
- **G1 rollout fidelity.** Smoke10 (`onpol_nuplan/results/long/smoke10_train.txt`, 10 train scenes, no held scene), `YR10m10-F-s0`: shipped
  cache against `cache_a80`: 0 zero flips, every scene score within 0.01, plan points within 0.03 m on >= 99 % of scene x decision pairs.
- **G2 long smoke.** Same 10 scenes and checkpoint on `cache_a160`: 10 / 10 rollouts end, 0 driver errors, every scene has 18 decisions
  (`Clipping n_sim_steps` line and the driver's drive calls). Reported, not gated: per-scene score, GT distance, at-fault events,
  zeros, decisions per scene against the 10-decision arm; runtime wall, RSS, VRAM, disk per scene (the cost model for the held read).
- **Go for the held read** if G0-G2 pass and the measured cost of the read (below) is under 24 card-hours.

## The read (only if the gates pass)

Held 460 scenes (`onpol_nuplan/results/split/held.txt`), at 18 decisions: `YR10m10-F` s0 s1, `P2H10S-F` s0 s1, `SSELFnoL-ON` s0 s1, one
pool job each (`cl fanout`). The 10-decision side: the stored onpol_nuplan held reads of `P2H10S-F` s0 s1 and `SSELFnoL-ON` s0 s1 (same
driver path), and two new 10-decision held jobs for `YR10m10-F` s0 s1 (no held read of it exists on the onpol driver path).

Per tag and length: mean scene score; zeros by kind (collision at fault, offroad, left corridor); at-fault events; mean GT distance
(`dist_to_gt_trajectory`, the leaderboard's GT distance); drift slope = OLS slope of the lateral distance to the logged path over decision
index k = 4..K (K = 10 / 18; as in aud1_drift_by_decision.md, from metrics.parquet); for the long read also the same metrics cut at
k <= 10. Recipe value = seed mean; paired differences with a log-clustered bootstrap (10 000 draws) over the held scenes.

**Question: does the candidate ordering hold as scenes lengthen?** Ordering = the rank of the three recipes by seed-mean scene score, and
separately by GT distance. "Holds" if the rank is the same at 10 and 18 decisions for both; any swap is reported with its paired CI. No
line on absolute values: longer scenes lower every score by construction (more steps to leave the 4 m corridor, progress over a longer
logged path). Descriptive: does the 18-decision read move the public GT distance toward the private 1.73 m?

## Not claimed

The 18-decision suite is not the private suite. Actors are replayed from the log (no reaction), as at 10 decisions. Decisions 0..9 of a
long run are not served the same route as in a 10-decision run (the route generator reads the recorded path of the whole window).

## Amendment 1 (before the full G0; written after a partial G0 on the first 44 logs)

A first rebuild at `after = 80` (613 scenes of the 44 logs then on disk) matched the shipped cache exactly in scene length, ego rows,
every vehicle row (0.0 difference on all columns) and the traffic-light table content, but had bicycles and pedestrians in addition:
the shipped cache holds only nuPlan category `vehicle` (AgentType 1) while trajdata's shipped query takes vehicle, bicycle and pedestrian
(same query since the MTGS commit ffa70ea). The organisers therefore built the cache with a vehicle-only filter that is not in the shipped
code (the track README: "Traffic on this track consists of vehicles only"). `build_cache.py` now drops non-vehicle detections at the
query (`--all-types` restores trajdata's own behaviour). G0 is unchanged and is read on that build; traffic lights are compared by
content (the feather bytes differ by writer version, the tables are equal). Building only the 1491 scene-id tokens instead of all 12 146
central tokens was decided before any comparison (cost: about 30 scenes per minute at 32 workers).

Note (test build with the filter, 896 scenes, logs still arriving): 491 scenes exact, 405 not written: trajdata's workers re-read the
central-token scene list and logs that arrived during the build shifted the indices. The real builds run only after all 136 logs are on
disk (al_chain.sh `cache` refuses otherwise).

## Amendment 2 (03:25 JST, after the held read, before running it): is 18 decisions private-like? (exploratory)

The read's YR10m10-F-s0 at 18 decisions (0.771, at-fault 0.474 km, GT 1.49 m) is close to its private submission (0.80, 0.529 km,
1.733 m) and far from its 10-decision values (0.928, 0.787 km, 0.957 m). Submission #1, APY10m10-AB-s0, scored privately 0.640,
0.132 km, 2.29 m. Two more pool jobs, held 460, `run.sh ap2` with `AP2_TAG=APY10m10-AB-s0` (its read path in ot3 / cf1), at 18 and at 10
decisions. Reading, descriptive only (two private points cannot validate a suite): does 18 decisions reproduce the private ordering
(YR10m10-F-s0 above APY10m10-AB-s0 by about 0.16 in score, with lower GT distance and more km per at-fault event), which 10 decisions
reverses on the public scenes (0.925 against 0.935)?

## Amendment 3 (2026-10-12 11:40 CST, before any run below): does the 10 -> 18 drop hit other driver families? (user request)

Reference drivers on the 460 held scenes at 18 decisions (`after160`), unchanged driver paths, same base overrides as their stored runs:
WA-JEPA (`run.sh wajepa`, `WAJ_AMP=0`, decision 188's driver; a reference arm only, never part of the method), garage `model_0014`
(`run.sh garage`), SimScale LTF sample (`run.sh ltf`), SimScale DiffusionDrive sample (`run.sh dd`). 10-decision side: the stored runs on
all 1491 public scenes, cut to the held 460 (WA-JEPA: c0c manifest; LTF / DD / garage: alpasim_proxy anchors/*.csv run dirs); no new
10-decision runs (all four cover the 460). WA-JEPA is split into 4 chunks of 115 scenes (its driver took 51 min for 234 scenes at 10
decisions); the others are one job each. Estimate: 7 pool jobs, about 40 min each, about 4.5 card-hours, about 45 min wall on the 6 free
cards.

Reported per driver, as for the held read: score 10 -> 18, zeros by kind, at-fault zeros, at-fault km, GT distance, first-10 GT distance,
drift slope k 4..10 / 4..18; next to the private leaderboard values (alpasim_proxy results/leaderboard_nuplan_20261012.json, the values
alpasim_proxy's README uses: LTF 0.738 / 0.270 km / 2.27 m, DD 0.772 / 0.267 / 2.16, garage 0014 0.848 / 0.228 / 0.718; WA-JEPA only
team-lucifer's `wajepa-s2-20260904` 0.762 / 0.626 / 1.85, not known to be our configuration) and with APY10m10-AB-s0 (amendment 2) and
YR10m10-F-s0. Question, descriptive: does each family's 18-decision local value land near its private value the way YR10m10-F-s0's did,
and does the 10 -> 18 drop differ between families? No line; 1 run per driver, the private side is 1000 unknown rollouts.
