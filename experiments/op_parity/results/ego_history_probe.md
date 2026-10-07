# Ego-history probe: P2H's planned speed does ride on the ego acceleration input, but not specifically before sharp turns (hypothesis not supported)

Written 2026-10-07. The token rule and verdict rule below were fixed and committed before any plan was computed; results are under "Result".
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

## Result

Run: P2H10-F-s0 / s1 on all 993 selected tokens, WA-JEPA 2 979 plans (993 tokens x orig / cv / cvlong; ~16 min on three cards; the
`acc0` / `pose` ablations were run for P2H only, WA-JEPA is ~1 plan/s per card). Checks: rebuilt `orig` ego features equal the cache (max abs
0); P2H `orig` poses equal the stored plan files of both checkpoints bit for bit (max abs 0.0 m, [ehp_equivalence.json](ehp_equivalence.json));
WA-JEPA `orig` D4 vs its stored plans on the same tokens: mean abs 0.16 m, r = 0.9996 (987 tokens, mean +0.11 m).

### Context: where the plans sit (all tokens of each set)

D4 / (v0 x 4 s), ratio of sums, cluster CI over logs ([ehp_ratio.csv](ehp_ratio.csv)); 1.00 = constant speed.

| set | n tokens / logs | median v0 | median dv_hist | logged | P2H orig | P2H cv | WA orig | WA cv |
|---|---|---|---|---|---|---|---|---|
| T turn approach | 368 / 78 | 5.0 | 0.18 | 0.959 | 0.950 | 0.956 | 0.970 | 0.971 |
| CT matched straight | 367 / 91 | 5.1 | 0.21 | 0.858 | 0.864 | 0.843 | 0.853 | 0.837 |
| P pre-flip | 129 / 27 | 8.2 | 0.48 | 0.808 | 0.804 | 0.869 | 0.810 | 0.841 |
| CP matched straight | 129 / 60 | 8.1 | 0.49 | 0.778 | 0.774 | 0.825 | 0.771 | 0.806 |

At the flip the human has barely slowed yet (T logged 0.96 of constant speed), and both models plan what the log does.

### Primary: effect of the constant-velocity history (`cv`) on planned 4 s distance D4, metres, mean [95% CI]

| set | n | P2H (2 seeds) | WA-JEPA | P2H - WA-JEPA (paired, same tokens) |
|---|---|---|---|---|
| **T turn approach** | 368 | **+0.13 [-0.35, +0.56]** | +0.04 [-0.23, +0.28] | +0.09 [-0.13, +0.30] |
| CT matched straight | 367 | -0.45 [-0.99, +0.06] | -0.34 [-0.70, +0.01] | -0.11 [-0.32, +0.09] |
| P pre-flip | 129 | +2.07 [+1.51, +2.61] | +0.97 [+0.73, +1.20] | +1.09 [+0.74, +1.44] |
| CP matched straight | 129 | +1.62 [+1.32, +1.91] | +1.14 [+0.93, +1.34] | +0.48 [+0.35, +0.63] |
| **T - CT** | | +0.58 [-0.13, +1.26] | +0.38 [-0.06, +0.80] | |
| P - CP | | +0.45 [-0.19, +1.07] | -0.16 [-0.48, +0.15] | |

The two P2H seeds agree (T +0.11 / +0.15, P +2.08 / +2.05). D2 and v13 move the same way ([ehp_effects.csv](ehp_effects.csv), [ehp_contrasts.csv](ehp_contrasts.csv)):
T v13 +0.05 [-0.08, +0.16] m/s (P2H), 0.00 [-0.07, +0.07] (WA-JEPA).

**Verdict (rule above): not supported.** The P2H effect on T is +0.13 m (CI contains 0, below the +0.5 m bar), and the T - CT contrast contains 0.
Removing the history from turn-approach tokens does not make P2H plan faster, because the history there carries no slowing.

### What the history does carry

- **The channel is the t0 acceleration.** `acc0` (ax = ay = 0, velocity and poses untouched) reproduces `cv`: P2H D4 effect T +0.08, P +1.97, CP +1.64
  (cv +0.13 / +2.07 / +1.62); `pose` (past-pose spacing only) changes nothing (|effect| <= 0.03 m). So P2H reads the
  history through ax (WA-JEPA: `cvlong` = `cv`, the acceleration cannot be separated there, not run).
- **Dose-response, generic across contexts.** Within T the `cv` D4 effect by logged slowing dv_hist (|v(-1.5 s)| - |v(0)|): <= 0 m/s -1.97 m (n 157),
  0-1 +0.89 (119), > 1 +2.72 (92); linear slope 1.70 [1.54, 1.85] m per m/s. The matched straights have the same shape (-3.65 / +1.15 / +2.47, slope
  2.43 [2.23, 2.59]); WA-JEPA's slope is 1.00 [0.92, 1.08] (T) and 1.64 [1.49, 1.77] (CT): **P2H is 1.4-2.1 x as sensitive to the history as WA-JEPA, in
  every context** (T / CT slope ratio 1.7 / 1.5; D4 effect on P / CP 2.1 / 1.4) ([ehp_dose.csv](ehp_dose.csv)). Turn context does not amplify it.
- **Both signs.** Where the history shows acceleration (dv_hist <= 0) the cv history lowers the plan by 2 m (T): P2H extrapolates its own acceleration.
  In closed loop HUGSIM feeds ax = the sim's acceleration (`lib/parity_hugsim.py`), so this is a positive-feedback candidate for the fast entries (not tested here).

### Fast approaches and history-slowed tokens (post hoc cuts, applied to every set alike)

v0 > 6 m/s (the four_dirs.md cut; n T 113 / CT 118 / P 117 / CP 117), D4 effect of `cv`:

| set | P2H | WA-JEPA | P2H - WA-JEPA |
|---|---|---|---|
| T | +1.45 [+0.86, +2.01] | +0.85 [+0.52, +1.13] | +0.60 [+0.29, +0.92] |
| CT | +1.10 [+0.55, +1.58] | +0.75 [+0.38, +1.08] | +0.35 [+0.13, +0.55] |
| T - CT | +0.35 [-0.42, +1.12] | +0.10 [-0.37, +0.56] | |

On these fast T tokens the planned D4 / (v0 x 4 s) is 0.846 (P2H) / 0.860 (WA-JEPA), logged 0.855; with the cv history 0.898 / 0.891. So the share of the
slowdown that rides on the history is **P2H 34%** ((0.898 - 0.846) / (1 - 0.846)) and **WA-JEPA 22%**; with a constant-speed history **both still plan a
~10% slowdown (0.898 vs 0.891)**, which the history does not explain. History-slowed tokens (dv_hist > 1 m/s; T 92 / CT 89): P2H +2.72 [+2.42, +2.98] / +2.47 [+2.25, +2.74],
WA-JEPA +1.52 / +1.61; T - CT +0.25 [-0.15, +0.59] (P2H).

## Reading

1. The hypothesis as stated is **not supported on navtest**: the history substitution moves neither model's plan on sharp-turn approach tokens (T +0.13 m P2H, +0.04 m
   WA-JEPA), and what effect exists (P2H on fast approaches, +1.45 m, 34% of the slowdown) is the same size on matched straights (T - CT +0.35 [-0.42, +1.12]).
   P2H's planned speed follows the logged acceleration, but not specifically before turns.
2. P2H is 1.4-2.1 x as history-sensitive as WA-JEPA in every context, entirely through ax. That is real, but it predicts P2H and WA-JEPA behave alike where the history is
   quiet: with a constant-velocity history P2H plans 0.956 / 0.898 (all / fast T) of constant speed against WA-JEPA 0.971 / 0.891, no P2H-specific deficit.
3. navtest cannot reproduce the HUGSIM difference (WA-JEPA 2.6-2.8 m/s into corners that P2H enters at 10-13 m/s): on navtest neither model plans a slowdown beyond what the
   log shows, in T both sit at the logged level. Whatever slows WA-JEPA 26 m before the HUGSIM turn is not visible in these 368 tokens, so this probe neither confirms nor
   excludes it. It does remove "P2H's junction slowing rides on the ego history" as the explanation of the navtest slowing; the HUGSIM half stays untested (it needs the
   closed-loop ax feed: a HUGSIM probe with the ego acceleration forced to 0, or the ego-history dropout the four_dirs.md text proposed, would test it).

## Caveats

- 368 / 129 tokens from 78 / 27 logs (T / P): the T contrasts have CIs of +-0.7 m; strata (v0 > 6: 113 tokens; dv_hist > 1: 92) are post hoc and thinner.
- The cv history is a combination the models never saw with these scenes (no deceleration while approaching a junction); the effect measures sensitivity to the input, not what a
  trained-without-history model would do. WA-JEPA's flow sampler is stochastic: edits and `orig` share the per-call seed, orig reproduces its stored plans to 0.16 m (mean abs D4).
- P2H plans through the frozen-Cinque port on W frames (the F recipe); poses go through op_interp `base` (lever arm), WA-JEPA's are its own rear-axle poses.
- No decision entries; no training.

## Reproduce

`ehp.py select` (CPU) -> `ehp.py p2h` (pool job, ~1 min) -> `ehp_wa.sh shards ...` (pool, 3 cards x 4 shards) + `merge` -> `ehp.py report`; outputs
[ehp_sets.csv](ehp_sets.csv), [ehp_context.csv](ehp_context.csv), [ehp_effects.csv](ehp_effects.csv) (strata all / v0>6 / dv_hist>1, variants, D2 / D4 / v13),
[ehp_contrasts.csv](ehp_contrasts.csv), [ehp_ratio.csv](ehp_ratio.csv), [ehp_dose.csv](ehp_dose.csv), [ehp_closure.csv](ehp_closure.csv) (ratio of sums, unstable where the
gap to constant speed is near 0: T), [ehp_equivalence.json](ehp_equivalence.json). Scripts: `scripts/ehp.py`, `scripts/ehp_wa.sh`.
