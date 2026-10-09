# op_parity off-track ladder: does the shipped openpilot plan recover from an offset, and did adaptation remove it?

Written 2026-10-09, lane OT2 piece A, decision 209. A measurement, no registered line. Code: `scripts/ot_ladder.py`, `scripts/ot_ladder_chain.sh`
(and stage `ladder` of `experiments/alpasim/scripts/ot2_b_chain.sh`). Full tables: [ot_rows/ladder/ladder_ladder_ot1.md](ot_rows/ladder/ladder_ladder_ot1.md) (+-0.5 m rows),
[ot_rows/ladder/ladder_ladder_ot2.md](ot_rows/ladder/ladder_ladder_ot2.md) (+-1.5 m rows); `ladder_ladder.md` is the first run of the +-0.5 m rows (same numbers, fewer models).

## Answer

**No. The shipped plan does not return to the path after a lateral offset in this probe, at either size, so there was no recovery for open-loop
adaptation to erode. The framing "adaptation erodes the industrial policy's closed-loop recovery, off-track rows restore it" is dropped.**
What is measured instead: adaptation on logged states adds a small response (0.24-0.33 of a 0.5 m offset within 4 s, 0.07-0.11 of a 1.5 m one); off-track
rows are what create a real one (0.73-1.00 and 0.37-0.56).

Response to the lateral offset at 4 s (1 = the plan is back on the logged path, 0 = offset ignored, < 0 = moves further away), 1 137 held-out rows, 30 logs, 95 % CI by log:

| model | inputs | +-0.5 m rows (mean abs dy 0.26 m, dpsi 1.0 deg) | +-1.5 m rows (0.77 m, 2.5 deg) | yaw offset at 4 s, +-0.5 m / +-1.5 m rows |
|:--|:--|:--|:--|:--|
| P0: shipped Cinque | vision tokens only | **-0.03 [-0.21, +0.11]** | **-0.15 [-0.28, -0.05]** | 0.32 [0.15, 0.44] / 0.28 [0.19, 0.35] |
| P2-F-s0 / s1 (thin, no hinge) | NAVSIM | 0.24 [0.13, 0.37] / 0.23 | 0.07 [-0.00, 0.14] / 0.09 | 0.31 / 0.21 |
| RMH10-F-s0 / s1 | NAVSIM | 0.25 [0.12, 0.38] / 0.24 | 0.08 / 0.09 | 0.32 / 0.22 |
| P2H10-F-s0 / s1 (lambda 10) | NAVSIM | 0.27 [0.15, 0.40] / 0.26 | 0.09 [0.01, 0.16] / 0.10 | 0.35 / 0.23 |
| SH30-F-s0 / s1 (lambda 30) | NAVSIM | 0.33 [0.20, 0.46] / 0.32 | 0.10 [0.03, 0.17] / 0.11 | 0.39 / 0.26 |
| AP2-AB-s0 / s1 | AlpaSim | 0.28 [0.14, 0.39] / 0.27 | 0.10 [0.03, 0.18] / 0.10 | 0.34 / 0.27 |
| OT30-F-s0 / s1 (10 % +-0.5 m rows) | NAVSIM | 0.77 [0.66, 0.90] / 0.79 | 0.38 [0.31, 0.44] / 0.39 | 0.71 / 0.47 |
| APO-a05m10-s0 / s1 (AP2 + 10 %) | AlpaSim | 0.73 [0.63, 0.84] / 0.73 | 0.37 [0.30, 0.43] / 0.38 | 0.68 / 0.48 |
| APO-a05m25-s0 / s1 (AP2 + 25 %) | AlpaSim | 1.00 [0.90, 1.10] / 0.99 | 0.56 [0.50, 0.61] / 0.56 | 0.78 / 0.54 |

Paired differences at 4 s (same bootstrap draws), +-0.5 m / +-1.5 m rows: P2 - P0 +0.27 [+0.13, +0.45] / +0.23 [+0.14, +0.33]; P2H10 - P0 +0.30 [+0.16, +0.47] / +0.24 [+0.16, +0.35];
SH30 - P0 +0.35 [+0.21, +0.53] / +0.25 [+0.17, +0.36]; SH30 - P2H10 +0.05 [+0.03, +0.08] / +0.01 [+0.00, +0.03]; OT30 - SH30 +0.44 [+0.39, +0.51] / +0.28 [+0.25, +0.30];
APO-a05m25 - APO-a05m10 +0.27 [+0.21, +0.34] / +0.19 [+0.17, +0.22].

Other things the tables show:

- Shipped at longer horizons (its plan runs to 10 s): on the +-1.5 m rows the lateral response keeps falling, -0.37 [-0.63, -0.15] at 6 s, -0.92 [-1.55, -0.37] at 10 s: the plan continues
  the drift direction of the history instead of returning. Its response to the yaw offset is positive and grows (0.28 at 4 s, 0.60 at 10 s). At 1 s the yaw response of every model
  without off-track rows is negative (-0.10 to -0.16): the history-yaw extrapolation of decisions 92 / 96 / 98, seen here again; only off-track rows turn it positive.
- No dead zone that a larger offset gets out of: shipped is -0.03 in the small half and -0.16 [-0.29, -0.06] in the large half of the +-1.5 m rows.
- Models without off-track rows respond less, relatively, to the larger offset (0.24-0.33 -> 0.07-0.11); models trained on +-0.5 m rows keep about half of their response at +-1.5 m.
- AP2 under its own input standard and under NAVSIM-standard ego features reads the same (0.28 vs 0.30). The AlpaSim route command is the same at both poses for every row.
- The lambda-10 checkpoint P2H10, which is the better AlpaSim driver (decision 205: 0.948 vs SH30 0.914), has the same weak response as P2 and slightly less than SH30. Its closed-loop
  advantage is not a recovery ability.
- The first probe of decision 198 (256 rows of shards 2-4) read SH30 0.21 and OT30 0.64; on 1 137 rows of 12 shards they are 0.33 and 0.77. Same ordering and gap; the CIs here are about +-0.13.

## Why the comparison is fair, and where it is not

- Rows: held-out tokens (`navsim/op-parity-full-dev`) of the off-track caches `ot1` (decision 198) and `ot2` (this lane): real navtrain frames re-projected to a statically drifted pose,
  paired with the same token at the logged pose. The response is the least-squares coefficient of the plan's lateral shift between the two on the shift a full return asks for.
- Every model reads the same frozen vision tokens of the same frames, and the plan is exported to the rear axle by the same function. P0 is the shipped weights through the parity port
  (no ego input, no adapter): the path every benchmark number of the shipped model in this repo was read through. Adapted models get their own ego features, the history poses re-expressed in the perturbed frame.
- The response is relative to the model's own plan at the logged pose, so it does not assume the model targets the logged path: a model that centres in its lane also shows a positive coefficient.
- Not covered: this is the plan, read open loop on 2 Hz keys warped to the model's frame rate, not a comma device at 20 Hz with its own lateral controller. The one closed-loop read of the shipped
  model's own lateral path that exists agrees: in the re-projection engine it keeps 0.93 [0.87, 1.00] of a 0.5 m offset after 2 s (decision 132). The plane re-projection distorts objects above
  the road, more so at 1.5 m (decision 141); a model can only respond to what the warp shows, and the models trained on such rows have learned to read it, which the others have not.
  Cold-start decisions (m < 4) were not probed. One shipped model (Cinque).
- Cost: three ladder jobs and one 89-row gate, about 0.3 job-hours. The first ladder job declared 14 GB of VRAM and used 30 GB (later ones declare 32).
