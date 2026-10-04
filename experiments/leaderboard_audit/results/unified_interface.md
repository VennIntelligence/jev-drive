# Unified openpilot interface: small-step checks (HUGSIM warm-up and clock, B2D camera height, nored, NAVSIM / WOD virtual camera)

2026-10-05. Spec and per-board deviations: [docs/openpilot-interface.md](../../../docs/openpilot-interface.md); code
`jevdrive/openpilot/interface.py` (tests `tests/test_openpilot_interface.py`). Ledger: `tmp/2026-10-05-op-config-ledger.md` sections 13 and 16.
Shipped Cinque everywhere, one run per cell. Per-scene / per-route tables: [unified_interface/hugsim_runs.md](unified_interface/hugsim_runs.md),
[unified_interface/b2d_runs.md](unified_interface/b2d_runs.md). Lanes: `scripts/unified_hugsim_chain.sh` (box `$DATA_DIR/runs/unified/hugsim`),
`scripts/unified_b2d_lane.py` (`$DATA_DIR/runs/unified/b2d/arms/<arm>-s<seed>`). Every run dir has `interface.json`. No pre-registration; the
decision rules below were set before the runs in the lane docstrings and the task brief (default to the spec value unless the small set shows harm).

## 1. HUGSIM: lateral path with and without the static warm-up, dilate vs hold clock

11 scenes (`scripts/unified_hugsim_small.txt`): 6 exam spinners (one per dataset incl. Waymo), decision 118's two newly stuck runs (3000_3200-medium,
0418-hard), completers 0930-hard / 0051-easy, crash 034-hard-01. Arms: `exam` (iLQR tracks the plan, 5 s static warm-up: every number before
2026-10-05), `d118` (openpilot lateral path: action curvature -> clip_curvature -> 0.25 s delay, 5 s static warm-up), `cold` (same, no static
warm-up; was named `spec` when run, now preset `spec_cold`), `hold` (cold + hold clock, 0.2 s delay).

| arm | spins | stuck (max_steps) | complete | mean HD | v at +1 / +2 / +3 s (median) |
|---|---|---|---|---|---|
| exam | 6 | 1 | 6 | 0.539 | 1.47 / 2.37 / 3.29 |
| d118 | 0 | 6 | 3 | 0.544 | 1.47 / 2.40 / 2.64 |
| cold | 0 | 6 | 2 | 0.358 | 1.45 / 0.98 / 0.48 |
| hold | 0 | 8 | 0 | 0.049 | 0.30 / 0.09 / 0.03 |

- Noise: `exam` reproduces the four base reruns of decision 119 on 10 of 11 scenes (040-easy varies between base reruns too); `d118` reproduces decision
  118's run on all 11. HUGSIM is close to deterministic, so single runs are informative.
- Cold start (cold - d118, per scene): -0.94 (0013-medium), -0.92 (0051-easy), -0.75 (034-hard-01), -0.14 (0930-hard), +0.73 (0418-hard), the other six
  within +-0.03; mean -0.185 [-0.474, +0.092]. The model plans ~9 m/s at step 0 on 0051-easy, drops to 1.7 m/s at step 1 and the car decays from
  1.7 to 1.2 m/s: without the warm-up the car does not launch (reading: the cold context sees a slow ego and plans to stay slow).
- Hold clock (hold - cold): mean -0.309 [-0.500, -0.127]; -0.84 / -0.71 / -0.57 / -0.74 / -0.28 on 040-easy / 0418-hard / 053-medium / 8440_8640 /
  3000_3200; the car decays to ~0 by step 14, 8 of 11 never get going.
- Zero spins in cold / hold are not a fix: the car barely moves.

Decision for the spec: HUGSIM `spec` = decision 118's arm (lateral path, 5 s static warm-up kept as a declared deviation, dilate clock). Its 64-scene
number already exists (decision 118: spins 0, mean HD 0.286 vs 0.278, non-spin HD -0.009 [-0.058, +0.042]); because the runs are deterministic, the
planned 64-scene multi-repeat would repeat that number and was not run. HUGSIM headlines from now on: `spec` (= d118) with the exam (iLQR) reading next to it.

## 2. B2D: camera height 1.433 m (windshield top) vs 1.22 m (front bumper line, x 3.8 m)

6 val routes (forced turns 28180, 24944; choice-turn routes 27297, 9196, 6999, 34183; 7 turns), seed 2. `legacy` = preset `drive` (raw action curvature,
1.433 m); `s143` / `s122` = preset `spec` (action -> clip -> 0.2 s delay) at each height; `nz` = zones off (`"zones": false, "div_m": 1e9`, the action head
steers everywhere).

| arm | mean DS | mean RC | completed | collisions | red light | turns right (of 7) | mean abs action curvature in turns (1/m) |
|---|---|---|---|---|---|---|---|
| legacy | 57.6 | 83.9 | 4 | 3 | 4 | 5 | - |
| s143 | 55.1 | 100 | 6 | 4 | 4 | 7 | 0.026 |
| s122 | 67.6 | 93.5 | 5 | 4 | 2 | 6 | 0.029 |
| s143nz | 30.4 | 51.4 | 0 | 3 | 3 | 0 | 0.007 |
| s122nz | 36.7 | 90.8 | 4 | 10 | 3 | 5 | 0.043 |

- Paired DS (6 routes): s122 - s143 +12.5 (per route +30, +58, 0, -25, +12, 0); s122nz - s143nz +6.3 (-30, +75, -7, -7, +14, -7); s143 - legacy -2.5
  (RC +16.1: clip + delay mainly protects completion). One red light is 10-30 DS on a route; these are directions.
- With the zones on the heights do not differ on turns (7/7 vs 6/7) or turn curvature. With the zones off, 1.22 m asks for ~6x the curvature
  (per-turn max 0.87 vs 0.13) and takes 5 of 7 turns against 0 of 7 (the 1.433 m car mostly never starts the turn, mean 0.76 m/s), at 10 collisions vs 3.
  Consistent with decision 108's scale law (lane ratio 0.851 = 1.22 / 1.433): at 1.433 m the world looks farther and the turn later.
- 1.22 m at the bumper line works mechanically (smoke 28180 DS 100; lane probabilities 0.5-0.97 while moving; no hood in view).

Decision for the spec: B2D `spec` mounts the cameras at (3.8, 0, 1.22) (`interface.B2D_SPEC_MOUNT`); the camera sitting ~2 m ahead of a windshield
camera is a declared trick. 1.433 m stays in the legacy `drive` preset. Privileged / vmerge arms (`lib/b2d_privileged_geometry.py`,
`lib/vm3_perception.py`) still assume x 1.779 m and keep the `drive` preset. op_route_cmd's CARLA pairs (1.22 m, x 1.519 m, one pinhole) now match the
height, not x or the wide construction (note in `experiments/op_route_cmd/results/carla_pairs.md`).

Open, worth one more run: the zones-off turn result (5/7 at 1.22 m vs 0/7 at 1.433 m) is the first sign that part of decision 121's "action head goes
straight" (2/38, at 1.433 m) is the rig; it needs the 38 choice turns and the 12 forced turns at 1.22 m before anyone quotes it.

## 3. `nored` (ground-truth light holding the stop latch): which results used it and how much it moved

Used by: decision 74 (op-drive dev and tuning `drive` / `dlon`, resume policy R1), decision 81 (every op_adapt_L B2D arm: drive, dnod, dtz, lmain, lkd,
ltz, lnoint, ldw10, lmain1/2 and the pace-matched dbaseslow controls), decision 82 (the privileged ceiling units of `b2d_privileged_chain.py`), decision
102 (op_img_cmd closed-loop smoke: `drive` and the skyNA / skySA arms). Not used: decisions 84, 105, 107 (`drive` 67.82), 118.5, 121, 122.
In the logs (`rb` = resume withheld) it acted only on routes 9196 and 24944 (and 27297 / 27870 in a few arms), 13-1137 plan ticks per route.

Rerun of decision 102's `drive` arm with `"resume": "timer"` (4 routes x seeds 0, 1; old = `$DATA_DIR/runs/op_img_cmd/cl/arms/drive-s*`):

| set | n | DS nored | DS timer | RC diff | red lights nored / timer |
|---|---|---|---|---|---|
| all | 8 | 60.1 | 58.0 (-2.1) | +5.2 | 3 / 6 |
| runs where nored acted (9196 s0, s1; 24944 s1) | 3 | 50.3 | 44.8 (-5.6) | +14.0 | 0 / 3 |
| runs where it never acted | 5 | 66.0 | 66.0 | 0 | 3 / 3 |

The five untouched runs reproduce to the decimal, so the whole difference is the resume: without the ground-truth light the affected runs take back
exactly the red lights nored had avoided. Decision 81's paired arm differences used nored on both sides; its absolute DS and red-light counts carry it.
`nored` is now refused by the agent (`interface.FORBIDDEN`); live lanes use timer.

## 4. NAVSIM / WOD virtual camera (cheap open-loop subsets; no official scoring)

| board | subset | 1.22 m | 1.30 m | 1.40 / 1.52 m | source |
|---|---|---|---|---|---|
| NAVSIM | navtrain, same CPU-warp pipeline vs true height | -1.97 [-3.08, -0.85] PDMS | -0.09 | +0.72 [-0.14, +1.60] (1.40) | decision 104.9 (existing) |
| WOD-E2E | 479 rater frames, RFS vs the real 1.81 m rig | **-0.492 [-0.691, -0.303]** (lon bias 0.86 -> 4.77 m) | - | +0.043 [-0.086, +0.175] (1.52) | this note (Cinque h1.22 new; `model_smoke/lib/wod_openpilot_rigs.py`) |

On both open-loop boards the as-on-car height costs score (progress / speed overshoot against logged human trajectories: the corrected scale makes
the model plan faster than the logged driver), and no intermediate height clearly helps. Full official scoring is not warranted. NAVSIM and WOD keep the
true camera height, declared as an LB deviation.

## Claims

- [E] HUGSIM: without the static warm-up the openpilot lateral-path car does not launch (HD -0.185 mean, 3 scenes -0.75 to -0.94); the hold clock is worse
  (-0.309 [-0.500, -0.127]). Spec = decision 118's arm.
- [E] HUGSIM runs are reproducible to within one scene in 11 across reruns; repeats do not narrow decision 118's CI.
- [E, direction only, n = 6, one seed] B2D 1.22 m at the bumper line is not worse than 1.433 m; with the zones off it turns (5/7 vs 0/7).
- [E] nored inflated decision 102's drive arm by 2.1 DS (5.6 on the runs where it acted, all three red lights hidden).
- [E] WOD h1.22 -0.49 RFS; NAVSIM h1.22 -1.97: the virtual camera is not adopted on the open-loop boards.
