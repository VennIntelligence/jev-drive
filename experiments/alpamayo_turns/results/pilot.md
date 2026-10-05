# alpamayo_turns pilot: Alpamayo 1.5 on 10 real sharp turns, with and without its cross cameras

2026-10-05. Pre-registration: [../plans/2026-10-05-prereg.md](../plans/2026-10-05-prereg.md). Runner:
[../scripts/turns.py](../scripts/turns.py), figures: [../scripts/turns_figs.py](../scripts/turns_figs.py). Tables:
[per_case.csv](per_case.csv) (one row per case x t0 x arm, metrics averaged over 6 samples), [effects.md](effects.md)
(every arm and paired contrast, 95 % bootstrap CI over cases, B = 10000).

## Setup

- **Data.** PhysicalAI-AV @ `33f9bf4`, **test** split only (12 random test chunks, 1180 clips, egomotion scanned). This
  is Alpamayo's own rig: cross-left / front-wide / cross-right 120 deg f-theta, front-tele 30 deg, 1920x1080, resized by the
  processor to about 576x320. No camera mapping at all; arm A is exactly as shipped. 168 clips have a turn under the
  rule (heading change 60-135 deg within 6.4 s, median 2-10 m/s, peak curvature >= 1/20 m^-1, not yet turning in the
  1.5 s before t0); rng(0) picked 5 left + 5 right, frozen as `physical_ai_av/alpamayo-turns-pilot@v1`
  ([cases.json](cases.json)): heading change 64-129 deg, minimum radius 5-19 m, median speed 2.7-5.9 m/s.
- **t0.** Primary = earliest t0 with >= 90 % of the turn inside the 6.4 s horizon (car at the turn entry); secondary =
  primary + 1.5 s (in the turn).
- **Model.** Alpamayo 1.5-10B @ `7aba829`, NVlabs/alpamayo1.5 @ `36aeb4c`, official loader / message / sampler,
  6 samples per (case, t0, arm), top-p 0.98, T 0.6, 10 flow steps, SDPA; same seed across arms (paired). Cameras are
  removed through the loader's own `camera_features` (variable camera count is a documented 1.5 feature); the model
  then sees only the remaining cameras with their correct names. Nav text (`"Turn left in 26m"`, the training format)
  is an oracle built from the logged path: direction + distance to the point where half the heading change is reached.
- **openpilot (OP).** Shipped Cinque, the clip's front-wide f-theta reprojected into the road (31 deg) and wide
  (58.7 deg) model frames with the clip's calibration (as `zeroshot_openloop/archive/pai_openpilot.py`), 5 s warm-up at
  20 Hz from zero state, no desire, no route; plan heading at the same 64 times. Camera extrinsic z 1.26-1.60 m.
  TensorRT and CUDA backends agree to 0.4 deg / 0.24 m.
- **Metrics.** Primary A_H = heading gain over time, sum(psi_pred psi_log) / sum(psi_log^2) over 0.1-6.4 s (1 = logged,
  < 1 under-turn or late). lag50 = time to reach half of the logged heading change, pred minus log (+ = late; 6.5 s if
  never). Post-hoc (added after the readout): A_S = the same gain with heading as a function of distance travelled
  (path shape independent of speed / waiting), and prog = predicted / logged path length.

## Results (primary t0, n = 10 turns)

| arm | cameras | nav | A_H | A_S (post-hoc) | lag50 (s) | ADE 6.4 s (m) |
|---|---|---|---:|---:|---:|---:|
| A | 4 (as shipped) | - | 0.66 [0.38, 0.91] | 0.71 | +0.72 | 3.13 |
| An | 4 | oracle | 0.76 [0.55, 0.95] | 0.85 | +0.51 | 2.60 |
| B | front-wide + tele | - | 0.55 [0.31, 0.78] | 0.41 | +0.89 | 3.31 |
| Bn | front-wide + tele | oracle | 0.75 [0.53, 0.95] | 0.82 | +0.58 | 2.89 |
| B1 | front-wide only | - | 0.09 [0.01, 0.20] | 0.13 | +1.96 | 5.58 |
| OP | openpilot Cinque (front-wide reprojected) | - | 0.49 [0.14, 0.84] | 0.48 | +0.81 | 3.59 |
| *post-hoc diagnostics* | | | | | | |
| C | cross-left + front-wide + cross-right (no tele) | - | 0.27 [0.10, 0.46] | 0.32 | +1.58 | 4.79 |
| Tblk | 4, tele frames black | - | 0.59 [0.33, 0.83] | 0.66 | +0.80 | 3.44 |
| B1t | front-wide + black tele | - | 0.33 [0.16, 0.51] | 0.37 | +1.44 | 4.42 |
| B1n | front-wide only | oracle | 0.13 [0.03, 0.26] | 0.20 | +1.92 | 5.42 |

Paired contrasts, Delta A_H (95 % CI over cases):

| contrast | primary t0 | secondary t0 (+1.5 s) | reading |
|---|---|---|---|
| **An - Bn** (pre-registered primary) | **+0.011 [-0.145, +0.141]** | +0.001 [-0.071, +0.061] | cross cameras do not matter when the route is known |
| A - B | +0.102 [-0.085, +0.287] | +0.010 [-0.023, +0.043] | without nav: small, not resolved; lag50 -0.18 s at secondary t0 [-0.32, -0.04] |
| An - A | +0.107 [-0.020, +0.290] | | the oracle route helps about as much as the cross cameras without it |
| A - C | +0.386 [+0.196, +0.613] | +0.125 [+0.062, +0.202] | removing the **tele slot** hurts a lot |
| A - Tblk | +0.065 [-0.136, +0.283] | -0.004 [-0.031, +0.022] | ... but blanking the tele *content* barely does |
| Tblk - C | +0.322 [+0.135, +0.542] | | so it is the 4th camera slot, not what the tele sees |
| B - B1t / B1t - B1 | +0.222 [+0.035, +0.441] / +0.238 [+0.096, +0.405] | | with front-wide alone, both tele content and the slot count |
| A - OP | +0.165 [-0.091, +0.477] | +0.167 [-0.046, +0.439] | Alpamayo ahead of openpilot, CI crosses 0 |
| An - OP | +0.272 [-0.019, +0.581]; A_S +0.370 [+0.052, +0.702] | +0.218 [-0.008, +0.491] | with the route, ahead on path shape |

**Pre-registered verdict.** An - Bn = +0.011 with CI [-0.145, +0.141]: |Delta| < 0.05 and CI inside +-0.15, so
**cross cameras do not matter** on these turns; early stop, no extension to n = 30. Alpamayo's ceiling: An A_H 0.76 is
just under the 0.80 bar for "handles these turns" (A_S 0.85 post-hoc).

## What drives the numbers

- **The low A_H values are mostly not about vision.** Two failure types account for most of them (see
  `figs/cases_bev.png`): (1) `72e5cf27` (stop sign) and `292d7c4d` (yield): the model stops or waits while the logged driver
  goes, so heading-over-time is ~0 in every arm, openpilot included. (2) `cf3fc128`: without a route it goes straight
  through a green light (A -0.05) and turns with the route (An 0.79, Bn 0.83). That is route ambiguity, which nav fixes
  and cross cameras do not.
- **Camera-count fragility (post-hoc).** Removing a camera slot changes behaviour much more than removing its content:
  black tele frames in place of the real ones cost 0.07, dropping the tele slot costs 0.39, front-wide alone collapses
  to 0.09 (6 of 10 turns essentially straight), and an oracle route does not rescue it (B1n 0.13). The 1.5 README
  supports fewer cameras but warns accuracy may drop; here the drop depends on which subset, and the 2-camera
  front-wide + tele set behaves like the full rig while the 3-camera set without tele does not. So "Alpamayo with only
  a front camera cannot turn" would be a wrong reading: it is an input-format effect of this checkpoint.
- **openpilot.** On the same front-wide view openpilot turns about as often as Alpamayo without nav (A_H 0.49,
  turn rate 0.50 vs 0.72) but fails the same stop / ambiguity cases plus `45baa092` and `5f3f6e30`, where it goes
  straight (no route input). Its path length is right (prog 0.98); on the stop-sign case it does not turn either.

## Figures

- `figs/turn_<arm>.png` (A, An, B, Bn, B1, B1n, B1t, C, Tblk, OP): the showcase turn (`d966165b`, right 78 deg at 3.5 m/s,
  the pilot case with the median logged heading change, rule fixed before outputs). Top: the images that arm actually
  fed (last of the 4 frames per camera; openpilot: its road and wide luma frames). Bottom: 6 sampled paths vs the logged
  path, and heading over time. Look at turn_A vs turn_B (cross cameras off: same turn) and turn_B1 (front-wide only:
  two samples straight, the rest turn late and short, although the front-wide image shows the crossing).
- `figs/cases_bev.png`: all 10 turns, every sample of A, C, B, B1, An, Bn against the logged path. Look at the two
  stop / yield cases and `cf3fc128` (straight without nav).
- `figs/summary.png`: A_H and lag50 per case and arm, paired Delta A_H with CIs, and the post-hoc A_S.

## Limits

n = 10 turns from 12 test chunks, one t0 pair each; A_H mixes longitudinal timing with turning (A_S added post-hoc for
that); oracle nav; whether Alpamayo saw the test split in training is not stated by NVIDIA; Alpamayo's front-wide is
already 120 deg, so "no cross cameras" here still means a 120 deg forward view, twice openpilot's 58.7 deg wide; open
loop only; the C / Tblk / B1t / B1n arms are post-hoc.
