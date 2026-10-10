# P2H10S-F on navhard and HUGSIM 64 (against P2H10-F)

Measurement only (2026-10-10), no training. All runs through `python -m jevdrive.bench` on the GPU pool. Tables: [other_boards/](other_boards/).
Unit of difference: navhard = per-group EPDMS, cluster bootstrap over stage-1 logs; HUGSIM = per-scenario HD, bootstrap over scenarios (B 10000, `jevdrive.stats`); seeds averaged per unit first.

## navhard two-stage (protocol G / `@gimm`, 225 groups, 4 seeds each)

New runs: `P2H10S-F-s0..3@gimm`, `P2H10-F-s2,s3@gimm`. Base s0, s1 are the stored op_parity runs (31.40 / 32.28, decision 170 / hinge_navhard_arms.md, same protocol).

| arm | combined | stage 1 | stage 2 | per seed |
|:--|--:|--:|--:|:--|
| P2H10S-F (4 seeds) | 33.46 | 73.42 | 46.03 | 33.27 / 34.84 / 32.51 / 33.20 |
| P2H10-F (4 seeds) | 31.54 | 74.33 | 43.11 | 31.40 / 32.28 / 31.26 / 31.25 |
| P2H10S-F (s0, s1) | 34.06 | 74.16 | 46.38 | 33.27 / 34.84 |
| P2H10-F (s0, s1) | 31.84 | 74.73 | 43.04 | 31.40 / 32.28 |

| difference | combined | stage 1 | stage 2 |
|:--|:--|:--|:--|
| S - base, 4 seeds | +1.91 [+0.13, +3.74] | -0.91 [-2.43, +0.82] | +2.92 [+1.02, +4.85] |
| S - base, s0+s1 | +2.22 [+0.25, +4.36] | -0.57 [-2.21, +1.32] | +3.33 [+1.27, +5.48] |

Per-seed (same seed) combined differences: +1.87 / +2.56 / +1.25 / +1.95. Base seed spread: combined 31.25 .. 32.28 (range 1.03); base 2-seed mean vs 4-seed mean -0.29 [-0.74, +0.16]. The gain is entirely stage 2; stage 1 is flat to slightly down.

## HUGSIM 64, preset `spec` (HD, 2 seeds s0, s1)

New: `P2H10S-F-s0,s1` (`*_spec`). Base: stored `P2H10-F-s0,s1_spec-rr1` (and independent repeat `-rr2`, identical to 3 decimals: repeat noise is ~0.000). Base seeds s2, s3 not run (scope cut).

| arm | HD | per seed | complete | fg coll | bg coll | off route | spin | stuck | launch stall |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| P2H10S-F | 0.403 | 0.411 / 0.396 | 21.5 | 27.0 | 11.0 | 3.5 | 1.0 | 0 | 1.5 |
| P2H10-F (rr1) | 0.395 | 0.388 / 0.402 | 21 | 27 | 11 | 3 | 2 | 0 | 1 |
| P2H10-F (rr2) | 0.395 | 0.388 / 0.403 | 21 | 27 | 11 | 3 | 2 | 0 | 1 |

S - base: +0.008 [-0.027, +0.050]. Base seed spread: 0.388 .. 0.402 (0.014). Counts are seed means; collisions = fg + bg. Nothing separates the arms on HUGSIM (HD difference is inside the base seed spread; 1 spin against 2, otherwise equal counts).

## Skipped
- WOD-E2E val zero-shot RFS: no `jevdrive.bench` command (decision 217 ran a lane script, `wod1_recipes_chain.sh`); dropped by the user's scope cut.
- Seeds 2, 3 on HUGSIM (both families): scope cut. navhard s2, s3 were already running and finished.
- The `-F` checkpoints were valid inputs on both boards; no non-F variant needed.
