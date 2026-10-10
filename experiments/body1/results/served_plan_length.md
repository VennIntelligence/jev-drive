# The plan served at decision 0 in AlpaSim is 9 % shorter than the scored plan: why

2026-10-11. Diagnosis only: existing rollout logs, existing caches, CPU forwards. No training, no closed loop, no pool job. Lane handed over
before the write-up was reviewed; see "Unfinished" at the end. Opened by [navtest_warp.md](navtest_warp.md), last section.

**Answer.** Not a frame-protocol mismatch on the nuPlan track. Two things, both measured: (1) the comparison in `prog_cl.py` is 1.5 s off: an
AlpaSim scene `<log>-<token>` starts 1.5 s before the navtest token, so decision 0 is not the token, decision 3 is; (2) decision 0 is a cold
start (one keyframe, 8 slots back-warped from it), an input state no NAVSIM-trained checkpoint was trained on, and its plan is a decelerating
one. From decision 3 on the driver's input is the benchmark's warp protocol. A real protocol mismatch does exist on the **PAI track** (real 10 Hz
frames in all slots, fed to warp-trained checkpoints), measured offline below.

## Measured

Base runs `P2H10-F-s0` / `-s1`, `$DATA_DIR/runs/alpasim/tr1/a/runs/P2H10-F-s{0,1}-chunk{0,1,2}`, 700 scenes each, `driver-logs/drive.jsonl`.
arc = 4 s arc length of the 8 served poses; ratios are pooled sums.

**1. Decision 0 is the token minus 1.5 s.** Ego speed fed at decision 0 against the logged speed of the token's four history keys
(`cache/lb_navtest/tab.npz`, `vel`), mean absolute difference: 0.012 m/s at -1.5 s, 0.212 at -1.0, 0.437 at -0.5, 0.665 at the token. Distance
driven from decision 0 to decision 3 against the logged distance -1.5 s -> 0: 0.14 m mean absolute difference (against the logged 0 -> +1.5 s:
1.05 m). Drive times: `now` = 17 000, 517 000, 1 517 000 us at k = 0, 1, 3. Already stated in
[../../alpasim/results/sh30_smoke.md](../../alpasim/results/sh30_smoke.md) ("NAVSIM cross-check") and used by
[m1_preturn_shift.md](../../alpasim/results/m1_preturn_shift.md); `prog_cl.py` (`load`, `base_arc0` against `ol_arc_base`) compares k = 0
with the token.

**2. Keyframes per decision: 1, 2, 3, then 4** (k = 0, 1, 2, >= 3; every scene). Code: `sh30_driver.py` `drive` collects the keyframes the
session has; `sh30_core.py` `fill_history` runs the oldest state backwards at constant velocity and yaw rate, `lattice` / `lattice_gpu` fill
the slots older than the oldest keyframe with that keyframe re-projected (`cold = "backwarp"`). With one keyframe all 8 image pairs are one
image under a road-plane homography.

**3. Served plan length by decision** (seed 0; seed 1 within 0.004):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| arc / (4 s x fed speed) | 0.911 | 0.976 | 0.983 | 0.988 | 0.992 | 0.993 | 0.994 | 0.991 | 0.987 | 0.982 |
| arc / scored plan at the token | 0.911 | 0.952 | 0.952 | 0.943 | 0.934 | 0.925 | 0.919 | 0.908 | 0.896 | 0.881 |

Open loop on the same 700 tokens (`runs/alpasim/ap2/offline/ot3-a/poses.npz`, `P2H10_*`, real frames, m newest keyframes; `P2H10_nav` equals
the bench plan, arc ratio 1.0000, 0.005 m mean): arc / (4 v0) = 0.913 (m = 1), 0.968 (m = 2), 0.995 (m = 3), 0.995 (m = 4), log 1.004;
m = 1 over the full plan 0.916 (all navtest 0.942). The decision-0 plan decelerates: distance per 0.5 s step over 0.5 x fed speed runs 1.02,
0.99, 0.96, 0.93, 0.90, 0.86, 0.83, 0.80; at decision 3 it is flat (0.987 .. 0.993). By start speed (decision 0 against the scored plan):
< 1 m/s 0.61, 1-3 0.65, 3-6 0.89, 6-10 0.97, > 10 0.94.

**4. Direct test, 200 scenes, CPU** (`sh30_core.Core("P2H10-F-s0", dev="cpu")`, fp32): the real CAM_F0 frame of the token's -1.5 s key, alone,
through the serving core with the simulator's ego state and command of decision 0, against the plan the driver served on the rendered frame:
arc ratio served / real-frame 1.005, r 0.995, mean pose distance 0.43 m (median 0.34). With the log's own state instead: 1.006, 0.49 m. Against
the token's four-keyframe plan (what `prog_cl.py` compared): 0.920, r 0.925, 2.22 m. So rendered frames and simulator ego state add nothing to
the length; cold start plus the 1.5 s offset is all of it. On the token itself, newest keyframe only against four keyframes: 0.931.

**5. Warm decisions are the benchmark's protocol.** Same code path (`op_interp.synth_cpu` `warp`, pairs (slot - 0.2 s, slot), first pair from a
zero image, `parity_adapter.ego_features`); `sh30_check.py` parity (tokens 0.08 % relative, poses 0.045 m max) and LAT1's bit-identical GPU warp
are the earlier identity checks. The table's 0.988 to 0.994 at k >= 3 against 0.995 open loop agrees. The row "arc / scored plan" keeps falling
after k = 3 because the simulator ego is slower than the log, not because plans are short for their speed: fed speed over logged speed 0.973 at
k = 3, 0.956, 0.940, 0.923, 0.907, 0.891, 0.877 at k = 9 (log speed after the token from central differences of the logged future).

**6. The match with the GIMM-frame plans (1.010) is a coincidence of size.** Warp-trained `P2H10-F-s0` on GIMM frames plans 0.892 of the log
at the token; the cold-start plan is 0.91 of the scored plan at another time, for another reason.

**7. PAI track: a real mismatch, every warm decision.** `pai_core.py` `slots` puts a rendered 10 Hz frame in every slot (listed as a
substitution in [pai_smoke.md](../../alpasim/results/pai_smoke.md), not measured there). Offline, 600 tokens of `lb_hq_navtestX` (identical ego
table), `P2H10-F-s0` policy on cached tokens, CPU fp32 (warp forward against the bench predictions: 0.066 m max):

| front frames | arc / log | ADE vs log (m) |
|:--|--:|--:|
| warp (trained, scored) | 0.994 | 0.61 |
| real 10 Hz lattice (`@real`, what PAI feeds) | 0.895 | 1.35 |
| GIMM | 0.885 | 1.52 |

real / warp by speed: < 1 m/s 0.96, 1-3 0.98, 3-6 0.90, 6-10 0.88, > 10 0.89. In PAI rollout logs on the box
(`runs/alpasim/fix1/paibox/runs/a_*`, 23 236 inferences, `poses_model`): arc / (4 v) 0.949 with 8 real slots (0.941 above 3 m/s), 0.887 at one
real slot; the nuPlan track reads 0.99 warm. Other camera and scenes, so only the offline table isolates the frame source.

## Scope

- nuPlan track, every SH30-family checkpoint trained on NAVSIM rows (`SH30`, `P2H10*`, `OT30`, `YR10m10`, `P2-F`, the body1 arms): decisions
  0 to 2 of 10. Same cold start in both arms of a paired comparison. `P2-F-s0` and `YR10m10-F-s0` read 0.913 / 0.917 at k = 0 in `ot3/a`.
- Checkpoints trained with the cold-start rule (`AP2*-AB`, `APY10m10-AB`): `AP2H10-AB` reads 0.980 at k = 0 and 1.00 to 1.02 later; fed speed
  over logged speed at k = 9 is 0.952 / 0.956 against 0.879 for `P2H10-F` on the same scenes (its rows also differ in ego definitions).
- body1 pages: the decision-0 "alignment" ratio (1.010, then 0.909) compares different instants and should read k = 3 (0.943, r 0.964, 1.5 m
  mean pose distance, closed-loop state included). Groups "at the navtest token" (`prog_cl.py`, `shape_cl_report.py`) describe the scene 1.5 s
  after its start, not its start. No score changes.
- PAI: every number from a nuPlan-trained checkpoint served through `pai_driver.py` (PAI1, COL1, FIX1, decisions 226, 235, 237, 241).
- HUGSIM, navhard: bench paths, not this driver. HUGSIM's frame path was not traced here.

## Not verified (hypotheses)

- That the cold start costs score. Indirect only: by k = 3 `P2H10-F` is at 0.950 of the logged speed against 0.9635 for `AP2H10-AB`; OT3 counted
  106.5 slow scenes against 54.5. The later speed loss (0.95 -> 0.88) comes with warm plans about 1 % short of fed speed, not with the cold start.
- That feeding PAI the warp lattice would raise PAI scores. The offline table says plan length and ADE recover on navtest frames; PAI's zeros
  are mostly longitudinal in the other direction (too fast into slow traffic).
- Why a single back-warped keyframe yields a decelerating plan (no real inter-frame content change in any pair is the candidate).

## Fix (described, not applied)

- nuPlan: none needed for protocol. To remove the cold start: serve a checkpoint trained with it (`ap2_prep.py --bw` rows, as `AP2H10-AB`).
- `prog_cl.py`: compare `D[s][3]` with the token's plan, or drop the check.
- PAI, minimal: in `pai_core.slots` take the rendered frames at t0 - 1.5 / 1.0 / 0.5 / 0 s as keyframes and build the slots with
  `sh30_core.lattice_gpu` as the nuPlan driver does (cold start unchanged), behind a switch; or train the PAI checkpoint on real-frame slots.

## Unfinished

- No second reader checked the numbers. Scratch scripts are not in the repo: Mac `/tmp/spl/{a..g}.py`, box `/tmp/spl_{a..g}.py`, per-scene
  table of the direct test `/tmp/spl_d_200.csv` (box). Run: `cd /tmp && PYTHONDONTWRITEBYTECODE=1 $DATA_DIR/envs/op-train/bin/python
  /tmp/spl_<x>.py` (`d`: `200 12` = scenes, threads; `g`: `600` = tokens). a-c, e, f: rollout logs by decision; d: direct test; g: PAI.
- Direct test at a warm decision on rendered frames (needs `SH30_DUMP` frames of a `P2H10` run; only `SH30-F-s0` dumps exist under
  `runs/alpasim/sh30_full48_c8/*/driver-logs/dump`). Item 5 rests on logs and earlier identity checks.
- PAI: only base-family arm `a` logs were read, on one checkpoint; PAI-trained checkpoints and their training frame source were not checked; no
  A/B on PAI frames (no PAI dumps on the box).
- HUGSIM adapter serving path; seed 1 for the open-loop table (offline plans exist for seed 0 only).
- Pool jobs submitted: none. Nothing left in the box checkout.
