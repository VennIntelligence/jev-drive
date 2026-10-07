# Turn training (pilot): re-weighting turns, anchor off on turns and a late-lateral weight do not close the sharp-turn gap (closure 0.01-0.02 of the > 20 deg gap)

Written 2026-10-07. Pre-registration [../plans/2026-10-06-turn-train-prereg.md](../plans/2026-10-06-turn-train-prereg.md) (written before any T arm was scored).
Chain `scripts/pp_turn_chain.sh`, report `scripts/pp_turn_report.py`; full stratified table [turn-train_navtest.md](turn-train_navtest.md) (+ CSV), gate
`$DATA_DIR/runs/op_parity/turn/gate.json`. Precondition: the first-line verdict of [turn-gain.md](../../op_probe/results/turn-gain.md) is "shrinkage (mild)", so
the lane continues and T3 runs.

## Result

navtest, 12 146 tokens, W frames, seed mean of 2, 95% cluster bootstrap over the 136 logs (B 2000). H = P2 + hinge at pilot scale (`HP-F`), the reference
of every arm; WA-JEPA 91.71.

| arm | what changes vs H | all tokens: arm - H | guard (>= -0.3) | > 20 deg pooled (n 3154): arm - H | closure of the H -> WA gap (9.21) | > 45 deg (n 1517): arm - H | DAC fail pp, all: arm - H |
|---|---|---|---|---|---|---|---|
| T1 | turn-balanced sampling | -0.24 [-0.49, -0.01] | pass | +0.13 [-0.52, +0.76] | 0.01 | **+0.97 [+0.13, +1.72]** | +0.29 [+0.05, +0.55] |
| T2 | T1 + no anchor on > 20 deg tokens | -0.30 [-0.58, -0.03] | fail (-0.303) | +0.15 [-0.74, +1.00] | 0.02 | +0.75 [-0.37, +1.80] | +0.30 [+0.02, +0.60] |
| T3 | T2 + x2 weight on y / yaw after 2 s | -0.32 [-0.58, -0.04] | fail | +0.15 [-0.71, +0.94] | 0.02 | +0.49 [-0.56, +1.43] | +0.36 [+0.09, +0.64] |

Reference: H 87.79 all, 78.63 on > 20 deg, 76.92 on > 45 deg (WA-JEPA 87.84 / 87.41); P2 without hinge (F) 87.39.

- **Gate**: no arm passes guard + closure >= 0.5, so by the pre-registration HUGSIM is not run (the declared `spec_plan_smooth` addition to the HUGSIM step,
  committed in 5fb357eb, therefore never executed).
- **One-line verdict (pre-registered rule on the > 20 deg bucket)**: formally "partial" (0 < closure < 0.5), in substance **no**: closure is 0.01-0.02 and
  every > 20 deg CI contains 0. Training distribution, anchor and late-horizon weighting do not close the turning gap at pilot scale.
- **Where the small gain is**: T1 lifts the > 45 deg bucket by +0.97 (closure 0.09 of that bucket's 10.5 gap) without changing its DAC failure rate
  (-0.03 pp [-0.86, +0.91]); the gain is in the non-DAC terms (progress / heading), not in the drivable-area failures that make the gap. The price is paid on
  straight and gentle tokens (straight -0.22, curve 8-20 deg -0.75 [-1.34, -0.11], DAC failures +0.15 to +0.62 pp): the re-weighted sampler moves effective
  data from where the model was already good.
- Collisions do not rise on turns: NC / TTC failures on > 20 deg tokens T1 -0.17 pp [-0.39, +0.03], T2 -0.10, T3 -0.16 vs H (3.03%); all tokens
  +0.09 to +0.15 pp ([four_dirs/turn_train_collisions.csv](four_dirs/turn_train_collisions.csv), `scripts/fd_entry.py turncoll`).
- Turning off the anchor (T2) and the late lateral weight (T3) add nothing over T1 on > 45 deg and cost more DAC failures: the anchor to shipped is not
  what holds the turns back.

## Reading

The sharp-turn DAC gap to WA-JEPA (> 45 deg: 10.8% vs 3.2% failing tokens) does not respond to how often turns are seen or how hard their late poses are
weighted. That fits decision 147 (about 2/3 of P2's DAC failures are decided at the frozen comma vision features) and turn-gain.md (the turning gain is
0.94, mostly dispersion, not shrinkage): the remaining turning failures are not a training-distribution problem of the plan head. See
[four_dirs.md](four_dirs.md) for the direction-level synthesis.

## Deviations and caveats

- The chain trained all 8 checkpoints on 2026-10-06 21:08-21:20 and stopped at the navtest stage when the evaluation was migrated to `jevdrive.bench`;
  it was resumed unchanged on 2026-10-07 10:39 (training stages skipped as finished, same checkpoints). No T arm was scored before the resume.
- Pilot scale (17 k navtrain tokens, 3 000 steps, 2 seeds); a full-scale effect can differ. T2 misses the guard by 0.003.
- Declared deviation (moot): `spec_plan_smooth` was added next to `exam` / `spec` for the HUGSIM step.
