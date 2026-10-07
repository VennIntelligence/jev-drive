# Ego-history probe: does the planned speed before a sharp turn ride on the logged ego history? (P2H vs WA-JEPA, navtest)

Written 2026-10-07. Status: **token rule and verdict rule fixed below before any plan was computed**; results are appended under "Result".
Hypothesis under test (four_dirs.md, direction 1 / open issues): P2H slows before a sharp turn because the logged ego history (velocity,
acceleration, pose history) already shows the human slowing, not because of vision; in HUGSIM nobody slowed first, so it enters corners at
a median 9.8 m/s (WA-JEPA 2.6-2.8 m/s). Code: `scripts/ehp.py`; inference only, no training.

## Design

Per token, only the ego inputs are edited (command and images unchanged), the plan is recomputed and compared with the unmodified plan of
the same token. Current speed v = |velocity at t0|; dt = 0.5 s (the 4 history poses sit at -1.5, -1.0, -0.5, 0 s).

| variant | edit |
|---|---|
| `orig` | none |
| `cv` (primary) | constant-velocity history: vx = v, vy = 0, ax = ay = 0, past poses (x, y, yaw) = (-v * dt * k, 0, 0), k = 3, 2, 1 |
| `cvlong` | longitudinal part only: vx = v, ax = 0, past-pose x = -v * dt * k; y, yaw, vy, ay untouched |
| `acc0` | ax = ay = 0 only (channel ablation) |
| `pose` | past-pose x = -v * dt * k only (channel ablation) |

Models: P2H10-F-s0 and P2H10-F-s1 (op_parity, `pp_train.PModel` on the pp_prep token cache, W frames, fp16, batch 128, as
`jevdrive.bench.navsim.parity_plans`; the two seeds are averaged per token), plan -> poses (0.5 .. 4 s, rear axle) through op_interp's `base`
adapter. WA-JEPA: the released checkpoint through `top10_t2/wajepa_run.py` (fp32, its NAVSIM path, requests built from our navtest index as
in navhard.md / full.md; its flow noise is seeded per call, so orig and edited plans of a token share the noise). The same edit is applied to
the WA-JEPA request (`ego_status` vx, vy, ax, ay and `history_trajectory`).

Equivalence checks before reading anything: the `orig` P2H plans of all navtest tokens reproduce the stored plan file of the same checkpoint,
and `orig` ego features rebuilt from the edit code equal the cached `ego` input.

## Token rule (fixed from logs and inputs only)

Per token: v0 = |velocity|, yaw4 = logged heading change over the next 4 s (deg), hist_yaw = heading of the -1.5 s pose in the t0 frame (deg),
dv_hist = |velocity at -1.5 s| - |velocity at t0| (m/s, positive = the human has slowed), cmd = navtest driving command of t0. Base filter for every
set: v0 >= 3 m/s and |hist_yaw| < 10 deg (the ego is not yet in the turn).

- **T (turn approach)**: base, cmd left / right, |yaw4| > 45 deg. These are the tokens at and after the command flip while the sharp turn is
  still ahead (the HUGSIM situation: command just turned on, turn imminent).
- **P (pre-flip)**: base, cmd straight, 0.5 - 3.0 s before a straight -> left / right switch (adjacent 0.5 s tokens of one log) whose command
  run contains a token with |yaw4| > 45 deg. (The command flips late: in navtest the turn is already inside the 4 s horizon at the switch,
  four_dirs.md, so tokens 1-4 s before the flip have yaw4 mostly below 45 deg.)
- **CT / CP (matched straight controls)**: base, cmd straight, |yaw4| < 5 deg, not in P; each T (then each P) token gets one control from a
  different log, nearest in (v0 / 1 m/s, dv_hist / 0.5 m/s) Euclidean distance, no replacement, caliper 2.0; T tokens first. Matching on dv_hist
  makes the controls carry the same logged slowing as the turn tokens, so a T - CT difference is turn context (command, vision) and not the
  history. Unmatched tokens are dropped from both sides.

Sets and sizes (written before plans, from `ehp_sets.csv`): T 368 tokens / 78 logs (median v0 5.0 m/s, median |yaw4| 62 deg), CT 367 / 91;
P 129 / 27 (median v0 8.2 m/s), CP 129 / 60. Token list: [ehp_sets.csv](ehp_sets.csv).

## Readouts and verdict rule (fixed)

From the 8 planned poses: path length L(t) from the origin; **D2 = L(2 s), D4 = L(4 s)** (m), **v13 = (L(3 s) - L(1 s)) / 2** (m/s). Effect of
a variant = variant - orig, per token (P2H: mean of the two seeds). The constant-speed reference is v0 * t. 95% CIs: cluster bootstrap over
logs (B 10 000, seed 0); contrasts T - CT and P2H - WA-JEPA resample logs of each side independently.

**Primary: effect of `cv` on D4, set T.** Verdict on the hypothesis, in order:
1. **Supported** for P2H if (a) the effect on T is > +0.5 m with the CI above 0, (b) the T - CT contrast of the effect is above 0 with the CI above 0,
   and (c) the WA-JEPA effect on T is smaller than P2H's, P2H - WA-JEPA contrast CI above 0.
2. **Not supported** if the P2H effect on T has its CI containing 0 or below +0.5 m, or the T - CT contrast CI contains 0 or is negative
   (the history effect is not specific to turns).
3. Otherwise **partly / shared** (e.g. WA-JEPA responds as much as P2H: the mechanism is real but not P2H-specific).
Secondary reads, no verdict weight: set P, D2, v13, the channel ablations, `cvlong`, and the dose-response of the `cv` D4 effect on dv_hist
inside T and CT.
