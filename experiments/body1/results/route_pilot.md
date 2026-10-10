# BODY1 arm 4.3, Amendment 7 (`P2H10R`, route hinge): stopped at the pilot gate

2026-10-10. The **fourth trained variant of the loss arm**, reopened by the user's decision for one variant aimed at the wide turns of decision 233,
designed after reading all 700 x 4 development scenes and after the look of section 1 on hold logs and navtest. Registered in
[Amendment 7](../plans/2026-10-10-body1-prereg.md) (pushed 74f5e670 before any code of the variant ran; selection pushed 81f2e633 before the hold-log
read). Hold logs have now been read at pilot scale by four variants of this lane. **Nothing follows it without a new decision by the user.**

**Correction, 2026-10-11 (navtest own-plan reads re-read on warp frames).** Until e7747c0b the own-plan reader (`bd4_g3.plans`) fed the navtest
tokens to these warp-trained checkpoints as GIMM frames; every navtest number of this page was off-protocol (plans 1.74 m from the bench's on average).
They are re-read on warp frames ([navtest_warp.md](navtest_warp.md), tables [navtest_warp/route/](navtest_warp/route/)) and replaced below, each with
the old number beside it. Hold-log and validation-part reads (band selection, the contact, ADE, slope and hold arc lines, (C-b), the hold rows of every
table) were always on warp frames and are not affected. **The gate verdict stands**: (C-a) is missed by more (138 against 110; old 67 against 54), (C-b)
is unchanged. Three statements changed: the size of the shift on navtest (about 0.1 m, not 0.2 to 0.3 m), its sign convention's absolute level (every
checkpoint is outside the logged path at 4 s, not inside), and the navtest arc ratio of the variant (0.9993, not 0.9915: no second line to clear).
The files in `route/` named `ol_*_navtest*` and the navtest lines of `route/pilot_gate.md` are kept as the superseded record.

**Result.** The pilot gate is **not met**: the new corridor line fails on both of its discriminating halves. With the selected band of 2.5 m
(`P2H10R-Pb25-s0` against the switch-off pilot `P2H10-P-s0`), on navtest turn tokens over 45 deg the own plan is more than 2 m outside the logged path in
138 tokens against 110 (line: at most +5; `P2H10S-P-s0`: 141; old read 67 against 54 and 75), and on on-log hold turn rows in 12 rows against 10 (line: at most +1; `P2H10S-P-s0`: 15).
Every line of Amendment 6's pilot gate is met (agent contact -40.2 %, boundary -33.8 %, dev ADE, slope, arc ratios), and the tube does what a tube
does: plans leaving the 4 m tube fall from 25 to 19 navtest tokens (old read 47 to 31) and from 27 to 10 hold rows. But the outward shift of the plan
at turns is a shift of about 0.1 m in the mean on navtest (old read: 0.2 to 0.3 m) with most of its mass inside 2.5 m, and the hinge does not touch it:
mean lateral at 4 s on navtest turn tokens +0.43 m against +0.42 m for `P2H10S-P-s0` and +0.31 m for the switch-off pilot (+ = outside; old read -0.29,
-0.36, -0.48 m). The tighter band of 1.5 m was not eligible (dev ADE over the limit).
By item 3 the variant ends: no full run, no G3, no closed loop, no navtest bench, no PAI read. The servable stays `P2H10-F`.

## 1. The open-loop look (before the amendment; `P2H10S-F` and the base, four seeds; no training)

`scripts/route_ol.py` on the plan dumps of `prog_ol.py`, `lib/route.py`: distance of the 8 own-plan poses (0.5 to 4 s) to the logged path of the same
state (polyline through the logged t0 pose and the 8 logged future poses, a ray at each end), signed by the side of the logged turn. Turn rows =
|logged 4 s heading change| > 45 deg. Tables `results/route/ol_a6_{full,s23,pilot}_{hold,navtest}*.csv`.

| Turn rows > 45 deg | n | `P2H10S-F` s0 / s1 / s2 / s3 | `P2H10-F` s0 / s1 / s2 / s3 |
|:--|--:|:--|:--|
| navtest (warp frames): W2 = plan more than 2 m outside the logged path within 4 s, tokens | 1 517 | 116 / 118 / 119 / 126 | 84 / 79 / 87 / 91 |
| navtest (warp frames): plan leaves the 4 m tube, either side, tokens | 1 517 | 18 / 17 / 18 / 22 | 16 / 16 / 18 / 19 |
| navtest (warp frames): mean lateral at 4 s, + = outside (m) | 1 517 | +0.38 / +0.43 / +0.40 / +0.43 | +0.29 / +0.27 / +0.30 / +0.31 |
| superseded, off-protocol frames: W2 | 1 517 | 58 / 65 / 59 / 65 | 45 / 46 / 47 / 49 |
| superseded: leaves the 4 m tube | 1 517 | 34 / 23 / 29 / 35 | 33 / 37 / 35 / 36 |
| superseded: mean lateral at 4 s (m) | 1 517 | -0.36 / -0.31 / -0.35 / -0.35 | -0.54 / -0.58 / -0.57 / -0.56 |
| hold logs, on-log states: W2, rows | 1 173 | 48 / 52 / 51 / 50 | 40 / 38 / 43 / 40 |
| hold logs, on-log states: leaves the 4 m tube, rows | 1 173 | 7 / 6 / 7 / 6 | 7 / 7 / 7 / 7 |
| hold logs, on-log + `ot1` + `yr1` + `bd4`: W2, rows | 3 954 | 350 / 367 / 357 / 374 | 389 / 381 / 400 / 406 |
| hold logs, the same: leaves the 4 m tube, rows | 3 954 | 64 / 59 / 64 / 59 | 127 / 122 / 126 / 125 |
| hold logs, the same: mean lateral at 4 s, + = outside (m) | 3 954 | +0.32 / +0.36 / +0.36 / +0.37 | +0.16 / +0.14 / +0.16 / +0.18 |

- The own plan of `P2H10S` lies further outside at turns in every seed and every state family: paired difference of the mean lateral at 4 s, seeds 0 / 1,
  navtest +0.09 [+0.05, +0.14] / +0.15 [+0.12, +0.19] m on warp frames (seeds 2 / 3: +0.10 [+0.07, +0.13] / +0.12 [+0.08, +0.17]; old read
  +0.18 [+0.14, +0.23] / +0.27 [+0.23, +0.31]); hold on-log +0.04 / +0.09 m, `ot1` +0.14 / +0.19 m, `yr1` +0.14 / +0.21 m, `bd4` +0.31 / +0.39 m
  (`bd4` right turns +0.57 / +0.65 m); all intervals exclude 0.
- On logged states the wide tail grows with it: W2 +32 / +39 / +32 / +35 navtest tokens (old read +13 / +19 / +12 / +16) and +8 / +14 / +8 / +10 hold
  rows over the base of the same seed; the four base seeds differ among themselves by at most 12 navtest tokens (old read: 4) and 5 rows. The +5
  tolerance of the corridor line was derived from the old spread of 4; on warp frames it is inside the base's own seed spread, and the excesses read
  here (+28 at pilot scale, +32 to +39 at full scale) are outside it by a wide margin either way.
- The unsigned 4 m tube, the line the main session had proposed, does not separate `P2H10S` from the base: within 3 tokens of the base on navtest in
  every seed (old read: below the base in three of four seeds), and at half the base's count on hold states, because the base, never trained on `bd4` states, wanders to both sides there.
- Pilot-scale ablations of Amendment 6: "A on on-log rows only" leaves the plan where the base has it (W2 115 against 110 tokens, lateral +0.003 m
  [-0.008, +0.013]; old read 55 against 54), "B + C without A" moves it out (141 against 110, +0.117 m [+0.078, +0.155]; old read 71 against 54):
  the shift comes from the hinge-only rows, where the road hinge pushes the path off the inside edge and nothing names the
  route.

## 2. The change and its checks

One term added to `P2H10S` (`lib/loss43.py`, `bd4_train.py --route-band B`): on the imitation rows of the train logs and on the hinge-only rows,
`mean_k relu(d_k - B)`, `d_k` the distance of plan pose k to the row's logged path, shape-only gradient, the existing weights (lambda 10, x 3 on hinge-only
rows). Checks, all before a pilot: the edited code with the term off against the code before the edit on the `P2H10S` recipe, 60 steps, 42 scalars and every
weight bit for bit (`results/route/ident_a7_shape.json`); every switch off against the unedited `pp_train.py`, 18 scalars and every weight bit for bit
(`ident_a7_pp.json`); the distance on synthetic rows: torch against numpy 5e-9 m, gradient a unit vector beyond the band and exactly 0 inside it,
along-heading component 1e-16 through `shape_only` (`route_check.json`).

## 3. Band selection on the validation part of the train logs (no hold-log read)

| Validation part, shards s2 + s3 | `P2H10R-Pb15-s0` (B = 1.5 m) | `P2H10R-Pb25-s0` (B = 2.5 m) | `P2H10S-P-s0` | `P2H10-P-s0` |
|:--|--:|--:|--:|--:|
| dev ADE at step 3 000 (m; limit 0.6013) | **0.6051** | 0.5949 | 0.5882 | 0.5913 |
| `dev_drift_off` (limit 0.30) | 0.059 | 0.052 | | |
| own-plan agent-contact rate, relative fall (>= 30 %) | 31.2 % | 32.6 % | | |
| own-plan boundary rate, relative fall (>= 25 %) | 37.5 % | 36.3 % | | |
| route hinge non-zero, imitation rows, steps 2 701 to 3 000 (<= 5 %) | 3.16 % | 0.68 % | | |
| route hinge non-zero, hinge-only rows (<= 20 %) | 9.67 % | 3.44 % | | |
| eligible | no | yes | | |
| W2, turn rows > 45 deg, four families (n = 490) | 49 | 55 | 65 | 66 |
| W2, on-log turn rows | 10 | 10 | 11 | 10 |
| mean lateral at 4 s, + = outside (m) | +0.37 | +0.44 | +0.44 | +0.26 |

B = 2.5 m is selected as the only eligible candidate. The term is a tube as registered (zero on 99.3 % of imitation rows and 96.6 % of hinge-only rows).

## 4. Pilot gate, one read (`P2H10R-Pb25-s0` against `P2H10-P-s0`; hold states of shards s2 + s3, 5 062 states; navtest tokens)

| Line | value | line | verdict | controls: `P2H10S-P-s0` (positive) / "A on on-log rows only" (negative) |
|:--|:--|:--|:-:|:--|
| own-plan agent-contact rate, hold | 0.0156 against 0.0261: -40.2 %, [-0.0142, -0.0071] | >= 30 %, interval excluding 0 | met | -37.1 % / -5.3 % |
| own-plan boundary rate, hold (NAVSIM raster) | 0.0182 against 0.0275: -33.8 %, [-0.0142, -0.0044] | >= 25 %, interval excluding 0 | met | -33.8 % / -2.9 % |
| dev ADE at step 3 000 (m) | 0.5949 against 0.5913 | <= + 0.01 | met | 0.5882 |
| continuation slope `alpha_05` | 0.851 against 1.079 | <= + 0.05 | met | 0.914 |
| 4 s arc ratio, hold pooled | 1.0001 [0.9991, 1.0013] | >= 0.995 | met | 0.9989 / 0.9977 |
| 4 s arc ratio, hold open states | 1.0006 [0.9994, 1.0019] | >= 0.995 | met | 1.0004 / 0.9983 |
| **(C-a) W2, navtest turn tokens > 45 deg (1 517), warp frames** (old read: 67 against 54; controls 75 against 54 / 55 against 54) | **138 against 110** | <= reference + 5 | **not met** | 141 against 110 (not met) / 115 against 110 (met, at the line) |
| **(C-b) W2, on-log hold turn rows (207)** | **12 against 10** | <= reference + 1 | **not met** | 15 against 10 (not met) / 9 against 10 (met) |
| (C-c) leaves the 4 m tube, navtest turn tokens, warp frames (old read: 31 against 47; controls 48 / 44 against 47) | 19 against 25 | <= reference + 5 | met | 26 against 25 (met) / 24 against 25 (met) |
| (C-c) leaves the 4 m tube, hold turn rows, four families (686) | 10 against 27 | <= reference + 1 | met | 25 against 27 (met) / 28 against 27 (met) |

The controls behave as registered: `P2H10S-P-s0` fails (C-a) and (C-b), the ablation without the hinge-only rows passes them, (C-c) does not separate.
The variant removes 3 of `P2H10S`'s 31 excess tokens on (C-a) (old read: 8 of 21) and 3 of its 5 excess rows on (C-b); the line needs 26 and 4.
On warp frames the negative control passes (C-a) exactly at the line (+5).

## 5. The > 45 deg bucket, by side and state family (own plan, counts of turn rows)

| Set | n | checkpoint | mean lateral at 4 s (m, + = outside) | more than 1 / 2 / 3 / 4 m outside | more than 2 m inside | leaves the 4 m tube |
|:--|--:|:--|--:|:--|--:|--:|
| navtest, all turns (warp frames) | 1 517 | switch-off | +0.308 | 356 / 110 / 37 / 24 | 37 | 25 |
| | | `P2H10S-P-s0` | +0.424 | 419 / 141 / 47 / 26 | 39 | 26 |
| | | `P2H10R-Pb25-s0` | +0.432 | 441 / 138 / 44 / 19 | 35 | 19 |
| navtest, right turns (warp frames) | 599 | switch-off | +0.236 | 130 / 40 / 7 / 4 | 12 | 5 |
| | | `P2H10S-P-s0` | +0.480 | 170 / 53 / 15 / 7 | 8 | 7 |
| | | `P2H10R-Pb25-s0` | +0.394 | 170 / 48 / 12 / 4 | 15 | 4 |
| navtest, left turns (warp frames) | 918 | switch-off | +0.355 | 226 / 70 / 30 / 20 | 25 | 20 |
| | | `P2H10S-P-s0` | +0.387 | 249 / 88 / 32 / 19 | 31 | 19 |
| | | `P2H10R-Pb25-s0` | +0.456 | 271 / 90 / 32 / 15 | 20 | 15 |
| superseded (off-protocol frames): navtest, all turns | 1 517 | switch-off | -0.475 | 158 / 54 / 27 / 18 | 178 | 47 |
| | | `P2H10S-P-s0` | -0.356 | 206 / 75 / 37 / 25 | 168 | 48 |
| | | `P2H10R-Pb25-s0` | -0.292 | 227 / 67 / 32 / 21 | 173 | 31 |
| superseded: navtest, right turns | 599 | switch-off | -0.270 | 75 / 19 / 7 / 4 | 47 | 6 |
| | | `P2H10S-P-s0` | -0.054 | 102 / 30 / 11 / 8 | 29 | 8 |
| | | `P2H10R-Pb25-s0` | -0.064 | 104 / 25 / 7 / 5 | 36 | 5 |
| superseded: navtest, left turns | 918 | switch-off | -0.609 | 83 / 35 / 20 / 14 | 131 | 41 |
| | | `P2H10S-P-s0` | -0.554 | 104 / 45 / 26 / 17 | 139 | 40 |
| | | `P2H10R-Pb25-s0` | -0.440 | 123 / 42 / 25 / 16 | 137 | 26 |
| hold, on-log | 207 | switch-off | +0.238 | 39 / 10 / 4 / 2 | 3 | 2 |
| | | `P2H10S-P-s0` | +0.352 | 47 / 15 / 5 / 4 | 6 | 4 |
| | | `P2H10R-Pb25-s0` | +0.345 | 46 / 12 / 5 / 2 | 3 | 2 |
| hold, off-track (`ot1` + `yr1` + `bd4`) | 479 | switch-off | +0.023 | 143 / 70 / 27 / 11 | 64 | 25 |
| | | `P2H10S-P-s0` | +0.209 | 150 / 62 / 22 / 11 | 37 | 21 |
| | | `P2H10R-Pb25-s0` | +0.259 | 144 / 53 / 13 / 6 | 27 | 8 |

What to read: the hinge acts where it is non-zero. Beyond 3 m it removes plans on both sides (off-track 22 -> 13 and 11 -> 6 outside, tube exits 21 -> 8;
navtest tube exits 26 -> 19, right turns at 12 / 4 tokens beyond 3 / 4 m against 15 / 7 for `P2H10S-P-s0` and 7 / 4 for the switch-off pilot; old
read 48 -> 31 and 7 / 5 against 7 / 4). Between 1 and 2.5 m, where the outward shift of `P2H10S` lives, nothing pulls: plans more than 1 m outside are
441 navtest tokens against 419 for `P2H10S` and 356 for the switch-off pilot (old read 227 / 206 / 158), and the mean stays where `P2H10S` has it
(+0.432 against +0.424 m). The first version's "the mean moves further out, partly because the far inside excursions that offset it are trimmed" is
not re-established: on warp frames there are few inside excursions to trim (35 to 39 tokens more than 2 m inside for all three; the 168 to 178 of the
old read were the off-protocol frames). Contact rates on hold turn rows: agent 0.0190 against 0.0379, boundary 0.0364
against 0.0583 (the body lesson is intact at turns).

Reported, not a pilot line: the 4 s arc ratio on navtest (warp frames) is 0.9993 [0.9987, 1.0000] pooled, 1.0000 on open and 0.9970 on lead states
(`P2H10S-P-s0`: 0.9995 / 1.0005 / 0.9964). The first version read 0.9915 [0.9904, 0.9926] / 0.9925 / 0.9872 (`P2H10S-P-s0` 0.9945 / 0.9948) and
concluded that a full run would have had a second line (G3 (e), 0.995) to clear: **not re-established**, the pilot is above that line on warp frames.

## 6. What it means

By Amendment 7 item 5: "missed on the corridor line = a tube on navtrain rows does not reach the wide tail of unseen turns". More precisely than that
sentence: the tube reaches the far tail (beyond its band) and not the shift. The wide turns of decision 233 are an outward bias of one to four
decimetres per plan (about 0.1 m on logged navtest turns on warp frames, 0.31 to 0.39 m on `bd4` hold states; that the closed loop accumulates it over
the turn is the reading, W2 was never tested against closed-loop corridor zeros); a dead band wide enough to stay a tube (2.5 m) is wider than the bias, and the band that starts to
bite on it (1.5 m: W2 49 of 490 on the validation part against 55, mean lateral +0.37 against +0.44 m) already costs imitation accuracy beyond the
registered limit (dev ADE +0.014 m), i.e. it stops being a tube and starts acting as a lateral target, the regime decisions 212 / 213 warn about. On this
row set the inside-edge lesson and the route are not separable by a hinge on distance to the log.

## 7. Costs, deviations, limits

Costs: about 0.5 card-h of the 10 (three 60-step identity runs, two pilots sharing card 2 for 16 min, readers), 25 min of box wall from the first run
(12:03) to the gate (12:28); 0.1 GB on the box (pilot checkpoints `P2H10R-Pb15-s0`, `P2H10R-Pb25-s0`, plan dumps `runs/body1/prog/ol_a7_*`); 320 GB free.
All jobs ran directly on the lane's held card 2 through `scripts/route_run.sh` / `route_chain.sh` (cores 176 to 207, memory margin over 200 GiB at every
start); nothing was submitted to the pool.
Deviations: (1) The corridor line is W2 (2 m, outside), not the unsigned 4 m tube of the main session's proposal: the look showed the latter does not fail
`P2H10S`; it is kept as guard (C-c). Fixed in the amendment before any code ran. (2) The line's tolerance is in rows / tokens (+5 at full scale, +1 on the
207 / 686 pilot rows), derived from the base's seed spread on W2, stated in the amendment. (3) Two band candidates and a selection on the validation part
instead of one fixed band. (4) The read of navtest tokens at pilot scale is part of the gate (as `P2H10S-P-s0` had been read for the arc lines).
(5) `results/route/ol_a6_pilot_*.json` were regenerated with counts for the gate reader after the amendment (same dumps, same numbers).
Limits: one seed, pilot scale (2 of 12 shards, 3 000 steps); (C-b) rests on 207 rows (12 against 10); the gate is open loop, and whether W2 predicts
closed-loop corridor zeros was never tested, since no `P2H10R` checkpoint ran in a simulator; the band candidates were two, chosen after the look;
hold logs read at pilot scale for the fourth time by this lane; the look itself read hold logs and navtest with `P2H10S` at full scale.
Found on the way, not this lane's: the box's system disk (`/`, 30 GB) is 100 % full; `/tmp` holds 26 GB, among them about 5 460 leaked temporary directories
(many of 157 MB, dated over several days up to today) and a 4 GB `torchinductor` cache. Nothing of it was deleted by this lane.
