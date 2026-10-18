# ACC + GAP on the nuPlan track (stop3), JTN-L18-s0 and RSG-L18-s0 (10-18)

Question: does the PAI serve recipe that fixed JTN's weaknesses (decision 359) help on nuPlan on top of the deploy hold `JEV_HOLD=stop3`?

**Answer: no. Both arms lose against the deploy candidate (RSG + stop3) in the arena, and both exceed the held-460 zero guard.**

## Composition and config

`JEV_ACC` / `JEV_GAP` compose with `JEV_HOLD=stop3` on the sh30 serve path: `sh30_driver` applies `serve_fix` (ACC hook, nuPlan handoff `ACC_T_HAND_NUPLAN` = 0.5 s) before `make_cached_plan`,
which `sh30_hold` wraps, so the hold sees the ACC-served poses; its latch time `T_HAND` is the same 0.5 s. The served version id of a live job confirms it:
`jev-RSG-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-holdstop3`. Serve env: `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` (GAP needs LEAD; VCONT as in the PAI probe serve),
plus `JEV_HOLD=stop3` for the arena reads (sh30hold driver). Held 460 @18 (6 chunks) was read WITHOUT the hold (sh30 driver), like the stored RSG-L18-s0 held read
(34 zeros, guard 39), so it is the ACC + GAP effect alone. Code 9edf98ec, `scripts/nuacc.sh` / `nuacc.py`, 48 read jobs at priority 14 (RSG arm) / 13 (JTN arm), one seed each.
Reference = ARENA-HOLD 2's stored RSG + stop3 read (arena held 2 228 windows, mean 0.9438, zero rate 0.029).
Earlier record: decision 329 read ACC (no GAP, no hold) on nuPlan held 460 for RSG and failed (-0.036, zeros 43 vs 26). No earlier nuPlan read with GAP.

## Arena held 2 228 @18 (stop3), paired vs RSG + stop3, log-clustered 95 % CI (jevdrive.stats)

| arm | mean | paired vs RSG+stop3 | zero rate (RSG+stop3 0.029) | zero-rate diff |
|---|--:|--:|--:|--:|
| RSG + stop3 + ACC + GAP (arm 2) | 0.9260 | **-0.0179 [-0.0265, -0.0101]** | 0.045 | +0.0162 [+0.0082, +0.0248] |
| JTN + stop3 + ACC + GAP (arm 1) | 0.9246 | **-0.0192 [-0.0292, -0.0068]** | 0.044 | +0.0148 [+0.0024, +0.0251] |
| arm 1 minus arm 2 | | -0.0013 [-0.0096, +0.0096] | 97 vs 100 zeros | -0.0013 [-0.0120, +0.0069] |

Stored references: RSG + stop3 0.9438, JTN + stop3 0.943 (tie, decision 353). Under ACC + GAP JTN no longer gains anything over RSG.

## By stratum (contribution to the paired mean; new / fixed zeros vs RSG + stop3)

| stratum (windows) | RSG-accgap contribution | new / fixed | JTN-accgap contribution | new / fixed | JTN-accgap vs RSG-accgap (paired) |
|---|--:|--:|--:|--:|--:|
| static (878) | +0.0000 | 0 / 0 | +0.0054 | 0 / 12 | +0.014 [+0.001, +0.037] (0 / 12) |
| launch (200) | -0.0061 | 4 / 0 | -0.0083 | 3 / 0 | -0.024 [-0.051, -0.000] (2 / 3) |
| braking (372) | **-0.0070** | 18 / 0 | **-0.0093** | 24 / 1 | -0.014 [-0.041, +0.008] (8 / 3) |
| turn (213) | -0.0053 | 15 / 1 | -0.0061 | 17 / 1 | -0.009 [-0.047, +0.029] (7 / 5) |
| cruise (116) | +0.0006 | 0 / 0 | +0.0006 | 0 / 0 | +0.000 |
| other (449) | -0.0001 | 3 / 3 | -0.0014 | 7 / 4 | -0.007 [-0.018, +0.003] (5 / 2) |

The loss is in braking, turn and launch windows (new zeros in launch + braking + turn: 37 for RSG, 44 for JTN; fixed: 1 and 2). Static windows are untouched for RSG (stop3 already serves them).

## Zero and partial types (2 228 common windows)

| read | AF-front | AF-side | offroad | corridor | rear-hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG + stop3 (reference) | 12 | 1 | 43 | 8 | 23 | 0 | 31 | 185 |
| RSG + stop3 + ACC + GAP | 16 | 0 | 59 | 25 | 30 | 8 | 36 | 94 |
| JTN + stop3 + ACC + GAP | 6 | 1 | 65 | 25 | 28 | 16 | 37 | 89 |

Braking windows only (AF-front / offroad / corridor; slow): reference 0 / 7 / 5; 42. RSG-accgap 1 / 8 / 21; 7. JTN-accgap 2 / 13 / 20; 8.
ACC + GAP does what it did on PAI to speed (slow partials 185 -> ~90, rear-hits not reduced) but trades it for corridor and offroad zeros (8 -> 25, 43 -> 59 / 65).
JTN's AF-front drops 12 -> 6 (RSG's rises 12 -> 16), the one place the PAI effect shows; it does not pay for the offroad / corridor loss.

## Held 460 @18 (ACC + GAP, no hold) vs stored RSG-L18-s0 (zeros 34, mean 0.8645; guard 39)

| read | zeros | mean | paired vs RSG-L18-s0 | progress_rel | front / lateral / rear coll | offroad | creep / no-launch |
|---|--:|--:|--:|--:|--:|--:|--:|
| RSG + ACC + GAP | **53** | 0.8472 | -0.0184 [-0.0554, +0.0117] | 0.869 | 7 / 0 / 23 | 22 | 21 / 0 |
| JTN + ACC + GAP | **52** | 0.8392 | -0.0264 [-0.0571, -0.0007] | 0.854 | 5 / 0 / 27 | 27 | 26 / 3 |
| JTN minus RSG (both ACC + GAP) | 52 vs 53 | | -0.0080 [-0.0235, +0.0093] | | | | |

Both fail the zero guard (39) by 13-14 zeros. Consistent with decision 329 (ACC alone on RSG: 43 zeros).

## Oddities

- Held 460 was read without the hold by design (matched to the stored references); a stop3 held 460 under ACC + GAP was not run.
- One seed per arm; the arena paired CIs are clustered by log, as in the JTB readout.
- The arena report's "RSG same serve" column is the stored RSG + stop3 read, not arm 2; arm 1 vs arm 2 is the separate paired table above (`nuacc.py pair`).
- The first foreground-blocking wait command died on an ssh drop (exit 255) although the jobs finished normally (summary 16:17 CST).
