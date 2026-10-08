# op_parity geo-e2e: can true geometry be made to be read through the adapter memory channel, and what is it then worth?

Written 2026-10-09. Follow-up of the geo-oracle ([geo_oracle.md](geo_oracle.md), decision 197). Pre-registration:
[plans/2026-10-09-geo-e2e-prereg.md](../plans/2026-10-09-geo-e2e-prereg.md) (committed and pushed before any score was read; pre-launch note with the smoke
result and the measured VRAM also before any score). Code: `scripts/geo_e2e.py`, `scripts/geo_e2e_chain.sh`, `pp_train.py --mem-e2e / --mem-init`,
`geo_oracle.py tok --weights`. Tables: [geo_e2e/](geo_e2e/) (`tables.md` every stratum and contrast, `arms.csv`, `verdict.json`, `tok_b_init.json`). Decision 200.

The true drivable SDF, the true agent boxes and (arm JP) the logged future path are **privileged inputs**: every memory arm here is an oracle probe,
not a method, and none of them enters a reportable inference path. No WA-JEPA weights or features were used. Pilot scale, 2 seeds per arm.

## Answer

1. **Registered verdict (judged arm JB, tokenizer trained jointly from a random init): negative, channel not read, set-up not shown to absorb.**
   JB - H0 navtest EPDMS **-0.12 [-0.29, +0.04]** (line: < +0.3). JB - JX (shuffled) +0.01 [-0.14, +0.16]; memory masked at test -0.06 [-0.16, +0.03].
   The positive control JP (same set-up, logged-path field) fails the registered "absorbs" rule: its CI is above 0 (+0.63 [+0.40, +0.83]) but the dev ADE
   rule misses (2-seed mean 0.507 m vs the 0.504 m threshold) because **only one of its two seeds learned to read the input at all** (seed 0: dev ADE 0.628 m,
   EPDMS +0.01; seed 1: 0.387 m, +1.24). So on the pre-registered table JB alone leaves the ceiling unmeasured: a randomly initialised tokenizer behind the
   zero-initialised channel is an optimisation lottery in 3 000 steps even when its input leaks the label.
2. **Registered variant JW (the same joint training started from the pre-trained geo-oracle tokenizer) meets the channel-read rule and measures the
   ceiling: +0.13 [-0.04, +0.29] over H0.** JW - JX +0.26 [+0.09, +0.42]; memory masked at test -0.28 [-0.44, -0.12]; dev ADE with mismatched tokens
   0.66 / 0.69 m vs 0.62 / 0.63 m. On the registered lines this is "negative" with the channel read, i.e. the task book's "what is missing is not
   geometry": the whole CI is below +0.3, against WA-Cf's +0.95 [+0.47, +1.37] (decision 160, lambda-10 baseline, not rerun).
3. **Joint training adds nothing over the frozen tokenizer, and decision 197's "not read" was partly one-seed power.** With seed 1 scored, the frozen
   arm GB is +0.11 [-0.10, +0.31] over H0 and +0.20 [+0.01, +0.38] over its shuffled control GX; JW - GB +0.02 [-0.14, +0.19].
4. **What the geometry buys when it is read is straight-road progress, nothing on turns.** JW: EP +0.19 [+0.11, +0.28], EPDMS < 5 deg +0.30
   [+0.10, +0.51]; 20-45 deg -0.13 [-0.74, +0.44], > 45 deg -0.46 [-1.09, +0.15]; > 20 deg DAC failures +0.16 pp [-0.31, +0.69]; NC + TTC failures
   +0.03 pp [-0.05, +0.12]; > 45 deg inside cuts +0.20 pp, cannot-make-turn -0.20 pp (both inside the noise).

| read-out (navtest, 2 seeds, x 100; diff vs H0, 95% CI over logs) | H0 no memory | **JB joint, random init (judged)** | JX joint, shuffled | JW joint, warm start | JP path field (label leak) | GB frozen (d197) | GX frozen shuffled (d197) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **EPDMS, all 12 146** | 88.48 | **88.37 (-0.12 [-0.29, +0.04])** | 88.36 (-0.12 [-0.26, +0.01]) | 88.62 (+0.13 [-0.04, +0.29]) | 89.11 (+0.63 [+0.40, +0.83]) | 88.59 (+0.11 [-0.10, +0.31]) | 88.39 (-0.09 [-0.21, +0.03]) |
| per seed (s0 / s1) | 88.46 / 88.51 | 88.36 / 88.37 | 88.35 / 88.37 | 88.65 / 88.59 | 88.47 / 89.75 | 88.55 / 88.63 | 88.37 / 88.42 |
| EPDMS, < 5 deg (6 400) | 92.71 | -0.03 [-0.24, +0.18] | -0.17 [-0.31, -0.04] | +0.30 [+0.10, +0.51] | +0.51 [+0.25, +0.76] | +0.28 [+0.06, +0.51] | -0.05 [-0.18, +0.08] |
| EPDMS, 5-20 deg (2 592) | 88.57 | +0.06 [-0.24, +0.36] | +0.14 [-0.15, +0.46] | +0.24 [-0.07, +0.58] | +0.67 [+0.31, +1.06] | +0.16 [-0.21, +0.54] | -0.07 [-0.38, +0.22] |
| EPDMS, 20-45 deg (1 637) | 81.48 | -0.40 [-0.97, +0.11] | -0.05 [-0.50, +0.38] | -0.13 [-0.74, +0.44] | +0.82 [+0.27, +1.38] | -0.28 [-0.82, +0.22] | -0.13 [-0.51, +0.23] |
| EPDMS, > 45 deg (1 517) | 78.04 | -0.46 [-0.95, +0.03] | -0.45 [-0.87, +0.00] | -0.46 [-1.09, +0.15] | +0.84 [+0.23, +1.44] | -0.27 [-0.95, +0.40] | -0.25 [-0.71, +0.27] |
| DAC fail %, all | 3.66 | -0.06 [-0.19, +0.09] | -0.04 [-0.15, +0.07] | -0.13 [-0.30, +0.05] | -0.19 [-0.36, -0.04] | -0.18 [-0.38, +0.02] | +0.02 [-0.09, +0.13] |
| DAC fail %, > 20 deg (3 154) | 8.53 | +0.14 [-0.25, +0.57] | -0.03 [-0.34, +0.26] | +0.16 [-0.31, +0.69] | -0.40 [-0.92, +0.10] | +0.10 [-0.41, +0.61] | -0.06 [-0.42, +0.27] |
| DAC fail %, > 45 deg | 9.76 | +0.16 [-0.34, +0.70] | +0.07 [-0.37, +0.47] | +0.46 [-0.17, +1.18] | -0.33 [-1.02, +0.35] | +0.30 [-0.50, +1.17] | -0.10 [-0.58, +0.38] |
| inside-cut %, > 45 deg | 4.12 | +0.03 [-0.31, +0.39] | -0.03 [-0.28, +0.22] | +0.20 [-0.21, +0.61] | +0.49 [+0.00, +1.02] | +0.40 [+0.07, +0.75] | -0.10 [-0.41, +0.20] |
| cannot-make-turn %, > 45 deg | 4.19 | -0.20 [-0.71, +0.32] | +0.07 [-0.22, +0.37] | -0.20 [-0.84, +0.42] | -0.96 [-1.45, -0.41] | -0.13 [-0.76, +0.53] | -0.10 [-0.39, +0.20] |
| NC + TTC fail %, all | 2.37 | +0.12 [+0.01, +0.24] | +0.09 [+0.00, +0.18] | +0.03 [-0.05, +0.12] | -0.48 [-0.67, -0.28] | +0.09 [-0.02, +0.23] | +0.01 [-0.07, +0.10] |
| NC (all) | 98.45 | -0.11 [-0.19, -0.03] | -0.08 [-0.17, -0.01] | -0.01 [-0.10, +0.07] | +0.41 [+0.23, +0.57] | -0.07 [-0.17, +0.01] | -0.02 [-0.10, +0.06] |
| DAC (all) | 96.34 | +0.06 [-0.09, +0.19] | +0.04 [-0.07, +0.15] | +0.13 [-0.05, +0.30] | +0.19 [+0.04, +0.36] | +0.18 [-0.02, +0.38] | -0.02 [-0.13, +0.09] |
| EP (all) | 87.00 | -0.02 [-0.06, +0.02] | -0.04 [-0.08, +0.00] | +0.19 [+0.11, +0.28] | +0.16 [+0.05, +0.26] | +0.14 [+0.06, +0.23] | -0.01 [-0.05, +0.03] |
| TTC (all) | 97.79 | -0.09 [-0.19, -0.01] | -0.07 [-0.15, +0.00] | -0.05 [-0.14, +0.04] | +0.45 [+0.27, +0.61] | -0.09 [-0.22, +0.03] | -0.04 [-0.12, +0.04] |
| LK (all) | 97.18 | +0.02 [-0.08, +0.12] | +0.04 [-0.04, +0.13] | +0.11 [-0.01, +0.23] | +0.26 [+0.15, +0.37] | +0.05 [-0.06, +0.16] | +0.00 [-0.08, +0.08] |
| DDC / TLC / HC (all) | 99.38 / 99.70 / 98.32 | -0.01 / +0.00 / -0.00 | -0.02 / -0.00 / -0.00 | -0.08 [-0.15, -0.01] / +0.04 [+0.01, +0.08] / -0.00 | +0.04 / +0.04 / -0.01 | -0.04 / +0.04 / -0.00 | -0.03 / +0.01 / +0.00 |
| dev ADE to the log, s0 / s1 (m) | 0.632 / 0.629 | 0.627 / 0.628 | 0.626 / 0.628 | 0.624 / 0.632 | 0.628 / **0.387** | 0.628 / 0.631 | 0.627 / 0.636 |

How to read it: the first column is the value, the others are differences to H0 (the first row also shows the value). JP is not a geometry arm; its
2-seed mean averages one seed that read the path and one that did not. Reference, not rerun: WA-Cf tokens on the same channel +0.95 [+0.47, +1.37].

Registered lines (copied from O1): combined arm navtest >= +0.7 and CI low > +0.3 = geometry is enough; < +0.3 = negative, not geometry; between =
partial. Channel read iff the arm - JX CI is above 0 **and** the memory-masked change CI is below 0. Set-up absorbs iff JP - H0 CI above 0 and JP dev
ADE <= 0.8 x H0's.

## Channel-read evidence

| contrast (EPDMS, all; 2 seeds) | JB (random init) | JW (warm start) | GB (frozen, d197) |
|:--|:--|:--|:--|
| arm - shuffled control | +0.01 [-0.14, +0.16] (per seed +0.01 / +0.00) | **+0.26 [+0.09, +0.42]** (+0.30 / +0.22) | +0.20 [+0.01, +0.38] (vs GX) |
| memory masked at test - arm | -0.06 [-0.16, +0.03] | **-0.28 [-0.44, -0.12]** | seed 0 only: -0.15 [-0.36, +0.05] |
| rule met | no | **yes** | first half only (masked read of seed 1 not scored) |
| dev ADE on / masked / mismatched tokens, s0 (m) | 0.627 / 0.630 / 0.636 | 0.624 / 0.636 / 0.661 | - |
| the same, s1 | 0.628 / 0.632 / 0.636 | 0.632 / 0.643 / 0.689 | - |
| tokenizer grad norm at step 200 / 1 000 / 3 000 (s0) | 0.01 / 0.03 / 0.34 | 0.11 / 0.28 / 0.71 | - |

Where the read shows (JW): masking its memory lowers EP by 0.21 [-0.30, -0.11], raises all-token DAC failures by 0.21 pp [+0.03, +0.37], and lowers
EPDMS on < 5 deg by 0.37 [-0.56, -0.21] and on 5-20 deg by 0.78 [-1.16, -0.47]; on > 20 deg nothing moves (+0.27 [-0.24, +0.81] / +0.39 [-0.32,
+1.06] on the two turn buckets, i.e. masking is if anything harmless there). NC + TTC failures do not respond (+0.06 pp [-0.03, +0.15]).
JB, by contrast, sits on its own shuffled control in every row; both lose 0.12 to H0, with NC + TTC failures +0.12 pp [+0.01, +0.24] and +0.09 pp
[+0.00, +0.18]: a jointly trained random tokenizer is a small nuisance input, with or without matching content.

JP (the set-up's positive control). Seed 1: dev ADE 0.387 m with the path field, 0.660 m masked, 2.73 m with another row's path; navtest 89.75
(+1.24). Seed 0: 0.628 / 0.631 / 0.631 m, 88.47 (+0.01), tokenizer grad norm 0.006-0.016 at steps 200-2 000 (seed 1: 0.07 -> 0.88 -> 1.56 at 200 / 1 000 / 2 000). In the 2-seed
mean the path field moves exactly what geometry does not: NC + TTC failures -0.48 pp [-0.67, -0.28], > 45 deg cannot-make-turn -0.96 pp [-1.45,
-0.41], > 45 deg EPDMS +0.84 [+0.23, +1.44], 20-45 deg +0.82 [+0.27, +1.38].

## Reading

1. Geometry can be made to be read (JW, and with two seeds the frozen GB too), and once read it is worth about +0.1, with a 95% upper bound of +0.29
   (JW) / +0.31 (GB). That is the ceiling decision 197 could not give, for this channel at pilot scale. It does not reach the "partial" band.
2. The gain is progress on straight and gently curving road (EP, < 5 deg); no turn bucket and no collision sub-score moves. The gap to WA-JEPA is in
   turns (three quarters of it) and in NC, so a branch that outputs geometry does not address it.
3. Decision 197's conjecture gets direct support: what this channel turns into score is information about the path to drive, not the drivable
   surface. One seed that read a path-only field (no timing, no speed) gained +1.24 and moved NC, TTC and cannot-make-turn. This is a label leak and one
   seed; it says what kind of content pays, not how much a real branch would get.
4. End-to-end training of a randomly initialised branch through the zero-initialised channel is unreliable at this scale: 1 of 2 seeds picked up a
   label-leaking input, 0 of 2 picked up geometry. A null from such an arm is not evidence about its input. The warm start removed the lottery.
5. For plan.md 2.1: (a) "if the oracle shows explicit geometry is enough, try the lightest form that outputs a compact geometry map" is closed, it is
   not enough; (b) SDF / occupancy stay useful only as auxiliary supervision, and this pilot gives no evidence that tokens shaped by them carry
   what the policy needs; the branch's tokens have to be plan-predictive (the evaluator's candidate sub-scores, future-path style targets) rather than
   scene-descriptive; (c) the branch must be pre-trained with its own head before it is attached, or the pilot read is a lottery; (d) the "true
   geometry oracle" bar of the main figure is about +0.1, which is itself a result: the future-latent encoder's +0.95 is not geometry.

## Limits, and what would overturn this

- The verdict on the registered judged arm (JB) is "unmeasured"; the ceiling rests on JW, a pre-registered variant (the prereg says to report it against
  the lines if it alone meets the rule), and on the frozen reference GB. Evidence strength: medium.
- "Read" is not "fully exploited": a different tokenizer, a wider channel, full data or a longer schedule could extract more. Pilot scale (25 k
  tokens, 3 000 steps), one tokenizer design, one learning rate, 2 seeds.
- The baseline already uses the same SDF as a training label (lambda-30 hinge); the number is the increment of geometry as an input on top of that.
- Agents are the 16 nearest boxes at t0 with velocity; no future occupancy. A future-occupancy oracle is a prediction oracle and was not run.
- JP: one of two seeds; its 2-seed means mix a read and an unread run. The absorbs rule failed by 0.003 m on the mean; the substantive fact is the seed split.
- GB / GX seed 1 were trained under decision 197 and left unscored by its gate; scoring them here was registered as a reference row and does not
  change decision 197's registered verdict, only its reading (status line updated).
- Would overturn "geometry is worth < +0.3": a geometry memory arm (any tokenizer, this recipe) at full data or with a longer schedule whose gain over
  its own no-memory baseline has a CI above +0.3, with the shuffled control separated. Would test reading 3 without a label leak: a branch trained to
  predict the path field from the front view, fed through the same channel.

## Setup

- **Recipe (all arms, = decision 197).** `pp_train --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`, `navsim/op-parity-s234` (25 415
  train / 408 dev), 3 000 steps x 64, warmup 100, memory masked on 25% of rows; row order, anchor rows and masked rows identical across arms per seed.
  H0 = `GH0-F-s0 / s1` and GB / GX = `GOB / GOX-F-s0 / s1`, decision 197's checkpoints (not retrained).
- **Joint tokenizer.** `geo_oracle.build_net` conv stack (1.1 M parameters, 32 x 512 tokens) computed every step from the batch rows' rasters (true SDF
  + agent distance / vx / vy; JX: the raster of another log through geo_x's fixed permutation; JP: distance to the logged future polyline, clip 6 m /
  3), its own optimizer group at the adapter's lr 3e-4 with the same schedule, gradient clipped on its own at 1.0. JW starts from a tokenizer
  re-trained with decision 197's recipe (`tok_b_init.json`: dev ADE true / shuffled 1.27 / 4.57 m, as before). After training the same job writes
  the navtest bank `mem/ge_<tag>/lb_navtest.npy` (fp16; dev ADE with fp16-rounded tokens equals the fp32 read to 4 decimals) that `jevdrive.bench` reads.
- **Read-outs.** navtest through `jevdrive.bench` (v2 devkit, 12 146 tokens), per-token 2-seed means, log-cluster paired bootstrap (B 4 000). Inside cut
  / cannot-make-turn from the four_dirs replay (`turn_oracle.py replay`).

## Verified / not verified

- Verified: all 8 trainings pass `pp_full_check.py train`; replay DAC equals bench DAC on every failing token of the 18 reads (406-460 each, 0
  disagreements); JX's permutation has no row from its own log (asserted in training and at export); prereg pushed before the first score.
- Not verified: variance beyond 2 seeds; the masked read of GB seed 1; whether JP seed 0 would have taken off with more steps.
- Not done: navhard (no labels on its synthetic frames), HUGSIM, full data, any training after the first read (registered stop).

## Deviations from the pre-registration

None in lines, arms or read-outs. The report prints "ceiling unmeasured" for the judged arm exactly as the registered table says; the JW reading is
the registered variant clause.

## Cost

GPU about 1.7 card-hours of the 4 budgeted: warm-start tokenizer 10 min, 8 trainings of 6.5-10 min on shared cards (peak 18-20 GB, booked 26), three
30-step smokes, 15 navtest plan exports. CPU: 15 navtest scorings and one replay, about 35 min of wall time.
