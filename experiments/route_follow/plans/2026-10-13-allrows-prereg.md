# Pre-registration: first training round under the new split (nuPlan all 1491 = training, PAI = exam), 2026-10-13

Context: [decision 286](../../../research/decisions/286.md) (RSG-L18 is the only standing candidate), the all-1491 pool
([ol_all.sh](../../onpol_long/scripts/ol_all.sh): `on1_L18RSGall`, 25 311 rows = RSG-L18-s0's rollouts on the 1031 onpol-train + 460 onpol-held
scenes at 18 decisions, pool gate passed: route ok 0.9999, route0 <= 5 cm 0.996), the slowness ratchet
([steer-long.md](../../../research/next-round/steer-long.md) 4.4 / 5 (2): the plan follows its own speed history; feeding the log's speed history
restores 83 %), and the PAI exam split ([pai_exam/README.md](../../alpasim/results/pai_exam/README.md), provisional v1: hard 131 / control 75).
Written, committed and pushed before any training is submitted. Amendments are appended, never edited in place.

## Arms (2 seeds each; every arm `--init RB0-F-s0`, S-only, no `--on-lon`)

Common recipe = RSG-L18's (d286): `rm_train.py train --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1
--label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_L18RSGall --on-k 13 --on-split alpasim_nuplan/onpol-all
--on-mode self --seed {0,1}`. The only change against RSG-L18 is the pool (`on1_L18RSG`, 17 474 rows, onpol-train only -> `on1_L18RSGall`).

| Tag | Extra | Question |
|:--|:--|:--|
| `RSGall-L18-s{0,1}` | none | (A) do +45 % rows (the held 460 scenes' rollouts) help? |
| `RSGallV-L18-s{0,1}` | `--ego-drop-speed 0.5` | (B) training-side fix of the slowness ratchet |

`--ego-drop-speed P` (new in `rm_train.py`, this round): on a share P of the non-anchor rows of every batch (pool rows included; own rng stream
[seed, 0, 71]) the ego row's speed history is zeroed: vx, vy, ax, ay and the x / y of the 4 history poses (ego_features columns 4-9, 11, 12,
14, 15, 17, 18). `present`, the command and the history yaws stay (the path heading survives, everything that carries the ego's own speed goes).
on_train's `--ego-drop` zeroes the whole EGO_STATE (yaws included) and stays refused in rm_train. Training only: the served ego row is unchanged.
Smoke (60 steps, smoke schedule = the recipe with --steps 60, seed 0; done before this file was committed): runs, step-0 dev ADE equals the
init's (0.5276), G-rely read true; but dev ADE after 60 steps is **0.671 with the drop vs 0.530 without** (control smoke): zeroing the speed history
on half the rows shocks a model that relies on it. Whether 2500 steps recover it is what the dev ADE gate below reads; no change to the arm.

Reference: `RSG-L18-s{0,1}` (stored reads; seed mean = `RSG-L18`). Everything else (YS / YS2G / YR10m10-F-s0) is context only.

## Offline (training run, dev rows): reported, gate on the arm

- **G-rely** (d282 / d286 line): junction rows, memory masked, ΔADE >= 0.01 m and junction − straight CI lower bound > 0 (`rely.json` read = true).
- **dev ADE** (memory on, final): not above RSG-L18's 0.528 / 0.527 + 0.01 m. An arm that fails either on both seeds is not read further for a
  claim (its reads still run and are reported).

## Line P: public 1491 at 10 decisions (the d264 Amendment 5 / guard.sh method)

All four checkpoints on all 1491 public scenes at 10 decisions (`ol_chain.sh readlist`, 10 chunks `lists/all_n10`), `ol_confirm.py` against the
stored 10-09 `YR10m10-F-s0` rollouts, seed means, log-clustered bootstrap B = 10 000; RSG-L18-s0 / s1 = the stored guard reads
(`runs/route_follow/guard/read10/`). Pairs: `RSGall-L18 − RSG-L18`, `RSGallV-L18 − RSG-L18`, `RSGallV-L18 − RSGall-L18`.

An arm (seed mean) passes P against RSG-L18 iff **mean score difference CI lower bound > −0.01** and **zeros difference (point) <= 0**; the zeros,
at-fault zeros and GT distance CIs are reported beside it. Both are needed for the arm to be called "not worse than RSG on nuPlan".

## Line D (arm B only): plan 2 s deficit

`rf_deficit.py` (rf_follow's plan 2 s deficit without the route regeneration: plan arc to 2 s − fed speed x 2 s, decisions at >= 1 m/s up to the
first zero-score event) on the same public reads. **B passes D iff `RSGallV-L18 − RSG-L18` (paired over scenes, mean of per-rollout means,
seed means) has CI lower bound > 0** (B's plan is less short) **and B's pooled decision median is above RSG-L18's**. Arm A is reported the same way.
(For scale: RSG on one 149-scene chunk, median −0.41 m, rollout mean −0.085 m; d286 held 18: median −0.26 m.)

## Line X: PAI exam (read by the coordinator on Tokyo)

Selection (before any PAI read): the two checkpoints of the four with the fewest public-1491 zeros (ties: higher mean score) are copied with their
sidecars to Tokyo `/data/runs/op_parity/runs/<tag>/`. Their PAI exam reads (same serving as RSG-L18-s0's stored `r281/rsg` read: weight 20,
`JEV_VCONT=1.0 JEV_LEAD=1`) are paired per scene with RSG-L18-s0's stored read:

- **hard 131**: zeros fewer (point) **and** mean score difference CI over scenes (scene bootstrap, B = 10 000) lower bound > 0.
- **control 75**: no regression: mean score difference CI lower bound > −0.02 and zeros not more than RSG-L18-s0's + 2.

Known bias: the hard list is "zero in any reference", and RSG-L18-s0 is one of the references, so all 70 of its stored zeros are in hard while
a new checkpoint's hard zeros are not selected on its own read (regression to the mean favours the new arm on hard). A hard-list win therefore
counts only together with a non-regressing control list; if the coordinator re-reads RSG-L18-s0 on hard (a second read), that pairing replaces
the stored one for the hard line.

## Reading order and claims

1. Training gates (G-rely, dev ADE) -> 2. line P -> 3. line D -> 4. selection and copy -> 5. line X (coordinator).
A candidate replaces RSG-L18 only if it passes G-rely, P and X (hard and control); B's ratchet claim needs D in addition. Passing P alone means
"the new pool / drop costs nothing on nuPlan", not "better".
Results: `experiments/route_follow/results/allrows/` (confirm.md / .json, deficit.md / .json, train gates); chain:
`experiments/route_follow/scripts/rf_allrows.sh`.
