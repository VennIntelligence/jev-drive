# S-DROP: full-scale drop-one ablation of `P2H10S` (three arms, two seeds)

2026-10-10, lane S-DROP. Registered in [plans/2026-10-10-sdrop-prereg.md](../plans/2026-10-10-sdrop-prereg.md) (pushed 65b15fae before any job).
An ablation only: no new loss term, no closed loop, no arm is promoted or served. Tables: [sdrop/](sdrop/) (`summary.md`, `contrasts.csv`,
`arc.csv`, `widening.csv`, `bench/`, the readers' own files `g3_sdrop_*`, `ol_sdrop_*`, `d_*_vs_*`).

`P2H10S` = `P2H10` + (A) agent hinge on the own plan, (B) hinge-only off-track rows `ot1` / `yr1` / `bd4`, (C) road hinge of those rows on the
road-and-lane raster; shape-only hinge gradient. Arms (seeds 0 and 1, 10 000 steps, all 12 shards, the trainer and flags of `P2H10S-F` minus one):

| Arm | Flag against `P2H10S-F` | What is left |
|:--|:--|:--|
| noA | `--agent-lam 0` | hinge-only rows with the road hinge only |
| noB | `--ho ""` | agent hinge on logged imitation rows only (C lives on B's rows and goes with them; 128 normal rows per batch) |
| noC | `--road-lam 0` | hinge-only rows with the agent hinge only, agent hinge on imitation rows |

All differences are arm minus reference with the two seeds averaged per unit and logs resampled (B 10 000); base = `P2H10-F-s{0,1}`,
S = `P2H10S-F-s{0,1}` (stored checkpoints, not retrained).

## Answers

**Question 1: which ingredient brings the navhard gain.** The hinge-only rows (B). Without them the stage-2 gain is gone
(noB - S: -3.53 [-5.48, -1.64]; noB - base: -0.20 [-1.09, +0.66]; registered line "carries" met). Neither hinge alone is the carrier: with
the rows kept, dropping the agent hinge keeps 65 % of the stage-2 gain and dropping the road hinge keeps 54 % (both "not needed" by the
registered line: arm - base >= half of the gain, interval excluding 0), and each drop also loses a part against the full recipe with an
interval excluding 0. The gain needs the off-track rows and is split about evenly between the two hinges on them.

| navhard, G frame, 225 groups, 76 logs | combined | stage 1 | stage 2 | per seed (combined) |
|:--|--:|--:|--:|:--|
| `P2H10-F` (base) | 31.84 | 74.73 | 43.04 | 31.40 / 32.28 |
| `P2H10S-F` (S) | 34.06 | 74.16 | 46.38 | 33.27 / 34.84 |
| noA | 32.78 | 73.51 | 45.21 | 32.61 / 32.95 |
| noB | 31.46 | 74.74 | 42.84 | 31.44 / 31.49 |
| noC | 32.84 | 73.64 | 44.84 | 32.54 / 33.14 |

| Difference | combined | stage 1 | stage 2 | share of the stage-2 gain kept | registered verdict |
|:--|:--|:--|:--|--:|:--|
| S - base | +2.22 [+0.25, +4.36] | -0.57 [-2.21, +1.32] | +3.33 [+1.27, +5.48] | 1 | |
| noA - S | -1.28 [-2.23, -0.46] | -0.65 [-1.45, +0.08] | -1.16 [-2.18, -0.24] | | |
| noA - base | +0.94 [-0.86, +2.77] | -1.22 [-2.99, +0.60] | +2.17 [+0.25, +4.01] | 0.65 | A not needed |
| noB - S | -2.59 [-4.59, -0.73] | +0.58 [-1.34, +2.22] | **-3.53 [-5.48, -1.64]** | | **B carries** |
| noB - base | -0.37 [-1.19, +0.44] | +0.01 [-0.75, +0.78] | -0.20 [-1.09, +0.66] | -0.06 | |
| noC - S | -1.22 [-2.75, +0.37] | -0.52 [-2.35, +1.10] | -1.53 [-2.98, -0.05] | | |
| noC - base | +1.00 [-0.40, +2.42] | -1.09 [-2.00, -0.25] | +1.80 [+0.12, +3.46] | 0.54 | C not needed |

What to read: stage 2 of noB sits on the base; noA and noC each sit a little over half way. On the combined score the two half arms
(32.78, 32.84) are above the base's four-seed range (31.25 to 32.28) but their intervals against the base include 0; only the full recipe
separates on the combined score. Stage 1 is lower than the base for every arm that has hinge-only rows (noC -1.09 [-2.00, -0.25], the only
one whose interval excludes 0).

**Question 2: does the widening at turns over 45 deg come only from the road hinge on hinge-only rows.** No; the registered claim fails.
noA is wide as expected, noB is not wide, but **noC is wide**: with no road hinge anywhere, the agent hinge on the hinge-only rows alone
puts 55 / 54 navtest turn tokens more than 2 m outside the logged path against 45 / 46 for the base (excess +10 / +8, tolerance +5).
The widening belongs to the hinge-only rows; the road hinge is the larger part, not the only one.

| navtest tokens over 45 deg (1 517), own plan | W2, seed 0 / 1 | excess over base | verdict | mean signed lateral at 4 s, arm - base (m, + = outside) | arm - S |
|:--|:--|:--|:--|:--|:--|
| base | 45 / 46 | | | | |
| S | 58 / 65 | +13 / +19 | | +0.224 [+0.186, +0.267] | |
| noA (road hinge on the rows) | 54 / 63 | +9 / +17 | wide | +0.168 [+0.130, +0.211] | -0.055 [-0.064, -0.047] |
| noB (no rows) | 45 / 47 | 0 / +1 | not wide | +0.032 [+0.024, +0.040] | -0.192 [-0.237, -0.152] |
| noC (agent hinge on the rows) | 55 / 54 | +10 / +8 | **wide** | +0.123 [+0.110, +0.139] | -0.100 [-0.138, -0.065] |

Hold on-log turn rows (1 173): W2 base 40 / 38, S 48 / 52, noA 44 / 51, noB 42 / 40, noC 47 / 53; lateral shift against the base
+0.067 (S), +0.047 (noA), +0.010 (noB), +0.056 (noC) m, every interval excluding 0. By side on navtest (W2, seed 0 / 1): right turns base
17 / 17, S 23 / 26, noA 19 / 27, noB 16 / 17, noC 21 / 23; left turns 28 / 29, 35 / 39, 35 / 36, 29 / 30, 34 / 31. The lateral shifts of
noA and noC against the base add up to more than S's (0.168 + 0.123 against 0.224 m): the two hinges push the same plans the same way.
noC's seed 1 clears the tolerance by 3 tokens; the continuous measure does not depend on that margin.

## Which ingredient carries which effect

Common rule of the pre-registration (an effect E = S - base with an interval excluding 0; "carries" = arm - S gives back at least half
with an interval excluding 0; "not needed" = arm - base keeps at least half with an interval excluding 0). Share kept = (arm - base) / E.

| Effect of `P2H10S` over the base | E [95 % CI] | noA (drop the agent hinge) | noB (drop the rows) | noC (drop the road hinge) |
|:--|:--|:--|:--|:--|
| navhard stage 2 | +3.33 [+1.27, +5.48] | not needed (0.65) | carries (-0.06) | not needed (0.54) |
| navhard combined | +2.22 [+0.25, +4.36] | carries (0.42) | carries (-0.17) | unresolved (0.45) |
| navtest EPDMS, 12 146 tokens | +0.38 [+0.20, +0.57] | not needed (1.03) | carries (0.11) | **carries (0.18)** |
| navtest EPDMS, > 45 deg (1 517) | +0.81 [+0.10, +1.53] | not needed (1.08) | unresolved (0.38) | **carries (0.04)** |
| navtest DAC failure %, all | -0.35 [-0.53, -0.19] | not needed (1.11) | carries (0.01) | carries (0.19) |
| navtest inside-cut %, > 45 deg | -0.73 [-1.39, -0.19] | not needed (1.09) | carries (0.14) | unresolved (0.36) |
| navtest own-plan boundary rate | -0.0065 [-0.0103, -0.0033] | not needed (0.86) | carries (0.15) | carries (0.34) |
| hold own-plan agent rate, pooled (30 083) | -0.0104 [-0.0124, -0.0085] | not needed (0.63) | carries (0.02) | not needed (0.67) |
| hold own-plan agent rate, off-track (19 157) | -0.0160 [-0.0190, -0.0130] | not needed (0.63) | carries (0.01) | not needed (0.68) |
| hold own-plan boundary rate, pooled | -0.0167 [-0.0190, -0.0145] | not needed (0.94) | carries (0.01) | carries (0.39) |
| hold own-plan boundary rate, on-log (10 926) | -0.0018 [-0.0029, -0.0009] | not needed (0.92) | carries (-0.03) | carries (0.31) |
| hold own-plan boundary rate, off-track | -0.0252 [-0.0286, -0.0221] | not needed (0.95) | carries (0.01) | carries (0.40) |
| lateral at 4 s, navtest > 45 deg (the widening) | +0.224 m [+0.186, +0.267] | not needed (0.75) | carries (0.14) | not needed (0.55) |

Effects of S whose own interval includes 0 on these two seeds get no verdict: navtest own-plan agent rate (-0.0002 [-0.0014, +0.0010]),
hold on-log agent rate, cannot-make-the-turn at > 45 deg (+0.16 [-0.41, +0.78] pp; noA +0.20, noB -0.10, noC +0.36 against the base, all
intervals including 0), navhard stage 1.

Read by ingredient:
- **B, the hinge-only off-track rows, is the precondition of everything.** noB (the agent hinge on logged rows only) stays at the base on every
  read, with three small shifts whose intervals exclude 0 (> 45 deg navtest EPDMS +0.30 [+0.09, +0.51], navtest boundary rate -0.0010
  [-0.0021, -0.00004], lateral at 4 s +0.03 m): navhard 31.46 against 31.84, navtest 88.72 against 88.67 (+0.04 [-0.03, +0.12]), hold agent rate 0.0268 against 0.0270, boundary
  0.0336 against 0.0338, W2 45 / 47 against 45 / 46. The agent hinge on logged rows does nothing measurable at full scale (pilot: -5.3 % /
  -2.9 %, decision 232 point 3).
- **C, the road hinge on those rows, carries the navtest gain and the road half.** Without it navtest is the base's (88.74, +0.07 [-0.06,
  +0.21]; > 45 deg +0.03), with it alone (noA) the whole navtest gain is there (89.06, +0.39 [+0.22, +0.57]; > 45 deg +0.87 [+0.15,
  +1.65]; inside cuts at > 45 deg -0.79 [-1.46, -0.26] pp). The hold boundary rate falls 47 % with it alone (S: 49 %) and 19 % without it.
- **A on the hinge-only rows carries a share of the object clearance and of navhard, nothing on navtest.** Hold agent rate: S -38.7 %,
  noA -24.5 %, noC -25.8 %, noB -0.7 %; on `bd4` states 0.042 (S), 0.053 (noA), 0.055 (noC), 0.080 (base and noB). Each hinge alone removes
  about two thirds of the object contacts the pair removes, so they overlap: bringing an off-track plan back onto the road also clears
  objects (as the pilot drop-one showed), and steering round objects also keeps some plans on the road (boundary -19 % with no road term).
- **No arm keeps the navhard gain without the widening.** The two arms that keep half of stage 2 are the two wide ones.

## Other reads

navtest (`jevdrive.bench`, 12 146 tokens, 136 logs):

| Arm | EPDMS | per seed | NC | DAC | EP | TTC | arm - base | arm - S |
|:--|--:|:--|--:|--:|--:|--:|:--|:--|
| base | 88.67 | 88.58 / 88.77 | 98.58 | 96.17 | 87.16 | 97.96 | | |
| S | 89.05 | 89.01 / 89.09 | 98.64 | 96.52 | 87.20 | 97.97 | +0.38 [+0.20, +0.57] | |
| noA | 89.06 | 89.06 / 89.06 | 98.62 | 96.55 | 87.20 | 98.01 | +0.39 [+0.22, +0.57] | +0.01 [-0.09, +0.11] |
| noB | 88.72 | 88.64 / 88.80 | 98.60 | 96.18 | 87.16 | 97.97 | +0.04 [-0.03, +0.12] | -0.33 [-0.52, -0.16] |
| noC | 88.74 | 88.72 / 88.76 | 98.58 | 96.24 | 87.18 | 97.99 | +0.07 [-0.06, +0.21] | -0.31 [-0.46, -0.17] |

Turn-oracle replay at > 45 deg (1 517 tokens; %, two-seed mean; arm - base):

| Arm | DAC failure | inside cut | cannot make the turn |
|:--|:--|:--|:--|
| base | 10.19 | 4.78 | 2.60 |
| S | 9.33, -0.86 [-1.66, -0.07] | 4.05, -0.73 [-1.39, -0.19] | 2.77, +0.16 [-0.41, +0.78] |
| noA | 9.20, -0.99 [-1.85, -0.19] | 3.99, -0.79 [-1.46, -0.26] | 2.80, +0.20 [-0.30, +0.80] |
| noB | 9.89, -0.30 [-0.59, -0.03] | 4.68, -0.10 [-0.29, +0.06] | 2.50, -0.10 [-0.36, +0.13] |
| noC | 10.15, -0.03 [-0.49, +0.43] | 4.52, -0.26 [-0.59, +0.04] | 2.97, +0.36 [-0.07, +0.82] |

4 s arc-length ratio against the base (two-seed mean; per seed in `sdrop/arc.csv`): no arm misses a line of Amendment 6 on either seed.
navtest pooled S 0.9990, noA 0.9997, noB 1.0024, noC 0.9969 (seed 1: 0.9958); open 0.9997 / 1.0005 / 1.0025 / 0.9971; lead 0.9964 /
0.9966 / 1.0020 / 0.9969; hold pooled 1.0005 / 1.0003 / 1.0003 / 1.0001; `bd4` 0.9990 / 0.9982 / 1.0007 / 0.9987. The small shortening
behind a lead (0.3 to 0.4 %) is present in both arms with hinge-only rows and absent without them.

Own-plan contact rates on the hold logs (30 083 states, 121 logs; rate, relative change against the base):

| States | rate | base | S | noA | noB | noC |
|:--|:--|--:|:--|:--|:--|:--|
| pooled | agent | 0.0270 | 0.0165, -38.7 % | 0.0204, -24.5 % | 0.0268, -0.7 % | 0.0200, -25.8 % |
| pooled | boundary | 0.0338 | 0.0171, -49.3 % | 0.0181, -46.6 % | 0.0336, -0.6 % | 0.0273, -19.4 % |
| on-log (10 926) | agent | 0.0063 | 0.0057 | 0.0058 | 0.0062 | 0.0062 |
| on-log | boundary | 0.0076 | 0.0058, -23.6 % | 0.0059, -21.8 % | 0.0076 | 0.0070, -7.3 % (interval includes 0) |
| off-track (19 157) | agent | 0.0388 | 0.0227, -41.3 % | 0.0287, -25.9 % | 0.0385, -0.6 % | 0.0279, -28.0 % |
| off-track | boundary | 0.0488 | 0.0236, -51.6 % | 0.0250, -48.7 % | 0.0484, -0.7 % | 0.0388, -20.4 % |
| `bd4` (4 425) | agent | 0.0803 | 0.0423 | 0.0530 | 0.0800 | 0.0553 |
| > 45 deg (3 954) | agent | 0.0319 | 0.0209 | 0.0226 | 0.0314 | 0.0295 (-0.0024 [-0.0056, +0.0005]) |
| > 45 deg | boundary | 0.0876 | 0.0512 | 0.0527 | 0.0857 | 0.0794 |

navtest on-log tokens: boundary rate base 0.0365, S 0.0300, noA 0.0309, noB 0.0355, noC 0.0343; agent rate 0.0120 / 0.0118 / 0.0122 / 0.0123 /
0.0119 (no arm moves it, as decision 232 found for S).

## Checks, cost, deviations, limits

Checks. (1) Trainer identity: 60 steps with the reference flags at today's checkout against the stored `B43A7-IDENT-S1`: 42 scalars and
every weight bit for bit (`sdrop/ident_sdrop.json`). (2) Config identity of the six runs against the stored `P2H10S-F-s<k>`: only the tag,
the arm's dropped flag(s) and two switches added after the stored run and left off (`route_band` 0, `route_lam`); nothing unexpected
(`sdrop/config_identity.json`). (3) Smokes: each arm logs its own terms (`sdrop/smoke.json`). (4) Training sanity: `dev_ade` 0.539 to 0.545 m,
`dev_drift_off` 0.041 to 0.044, hinge-only rows per batch 13 / 0 / 13. (5) The report's per-seed contact rates equal `bd4_g3.py`'s on all 24
numbers (0.0 states apart); the stored W2 of the base and of S (45 / 46, 58 / 65) and the stored navhard and navtest numbers of S are
reproduced.

Cost. About 5 card-hours of job time against the estimate of 5.5 (at most 8): six full runs 3.8 (noA and noB 46 min each, noC 23 min; the
cards were shared with another lane's 12 jobs that arrived as the chain started, and the full runs waited about 30 min for room), the
rest smokes, 12 `bd4_g3.py` jobs, two dumps and the bench's plan stages. Wall 18:43 to 20:26 box time. All GPU and CPU-reader jobs through
the pool, owner `body1-sdrop`; 0.5 GB new on the box (six checkpoints, two plan dumps `runs/body1/prog/ol_sdrop_*`).

Deviations. (1) "Minus C" is `--road-lam 0`, not Amendment 4 item 11's raster swap, and "minus B" removes C with it (fixed in the
pre-registration). (2) The chain stopped twice on its own shell errors after the reads had been produced (a CPU job declared with 0 GB VRAM,
an unset optional argument of the report helper); it was resumed unchanged apart from those two lines, no read was repeated and no line
touched. (3) `bench report`, `bd4_g3d.py` and the two check commands ran directly in the chain (CPU, seconds), as in the earlier arms.
(4) No figure: the tables above are the result.

Limits. Two seeds per arm; on navhard the "not needed" verdicts of noA and noC rest on lower bounds of +0.25 and +0.12, and on the combined
score neither half arm separates from the base. The hold logs have now been read seven times by this lane's variants, navtest by the same
arms, and navhard a second time (four checkpoints of the recipe family before, six more here). Everything except the two bench boards is
open loop on the student's own plan; W2 has not been tested as a predictor of closed-loop corridor zeros (decision 234). noB also changes
the batch (128 normal rows). The drop-ones are of the shape-only recipe; the pilot drop-ones of decision 232 were of the full-gradient one.
The A x C interaction is read from two drop-ones, not from a 2 x 2 with a "rows with no hinge" cell (such rows would carry no loss term).
