# op_parity geo-oracle: what is a perfect geometry branch worth through the adapter memory channel?

Written 2026-10-09. Pre-registration: [plans/2026-10-09-geo-oracle-prereg.md](../plans/2026-10-09-geo-oracle-prereg.md) (committed and pushed before any
score was read). Code: `scripts/geo_oracle.py` (tokenizer, banks, gate, report), `scripts/geo_oracle_chain.sh`, four `--mem` kinds in `scripts/pp_train.py`.
Tables: [geo_oracle/](geo_oracle/) (`tables.md` every stratum and contrast, `arms.csv`, `verdict.json`, `gate-s0.json`, `tok_{s,a,b}.json`). Decision 197.
The true drivable SDF and true agent boxes are **privileged inputs**: every memory arm here is an oracle probe, not a method and not a reportable driver.
No WA-JEPA weights or features were used. Pilot scale, **seed 0 only** (the pre-registered seed-0 gate stopped the lane).

## Answer

**Registered verdict: negative (GB - H0 navtest EPDMS +0.10 [-0.14, +0.34], line < +0.3), with the pre-registered qualifier "the channel was not read".
This run does not measure the ceiling of geometry.** The tokens demonstrably carry the geometry; the policy demonstrably did not use them for its
plan (dev ADE unchanged, shuffled control and memory-off reads inside the noise on EPDMS). What it shows is that a frozen, plan-supervised geometry
tokenizer on the decision-160 channel does not move the SH30 pilot in 3 000 steps, the same state decision 166 reached with raw SDF rasters.

| read-out (navtest, seed 0, x 100; diff vs H0, 95% CI over logs) | H0 no memory | GS true SDF | GA true agents | GB both | GX shuffled GB |
|:--|:--|:--|:--|:--|:--|
| **EPDMS, all 12 146** | 88.46 | 88.64 (+0.18 [-0.10, +0.46]) | 88.52 (+0.07 [-0.14, +0.24]) | **88.55 (+0.10 [-0.14, +0.34])** | 88.37 (-0.09 [-0.25, +0.08]) |
| EPDMS, < 5 deg (6 400) | 92.73 | 92.85 (+0.12) | 92.81 (+0.08) | 93.00 (+0.27 [+0.03, +0.53]) | 92.62 (-0.10) |
| EPDMS, 5-20 deg (2 592) | 88.70 | 88.97 (+0.27) | 88.63 (-0.08) | 88.79 (+0.09) | 88.67 (-0.04) |
| EPDMS, 20-45 deg (1 637) | 81.10 | 81.69 (+0.59 [-0.42, +1.70]) | 81.68 (+0.57 [-0.34, +1.43]) | 81.02 (-0.09 [-0.87, +0.64]) | 81.24 (+0.13) |
| EPDMS, > 45 deg (1 517) | 77.94 | 77.81 (-0.13) | 77.65 (-0.28) | 77.52 (-0.42 [-1.26, +0.42]) | 77.61 (-0.33) |
| DAC fail %, all | 3.68 | 3.32 (-0.36 [-0.60, -0.12]) | 3.57 (-0.11) | 3.49 (-0.19 [-0.41, +0.03]) | 3.68 (+0.00) |
| DAC fail %, > 20 deg (3 154) | 8.72 | 8.28 (-0.44 [-1.18, +0.25]) | 8.28 (-0.44 [-0.95, +0.07]) | 8.69 (-0.03 [-0.59, +0.51]) | 8.56 (-0.16) |
| DAC fail %, > 45 deg | 9.82 | 9.95 (+0.13 [-0.97, +1.23]) | 9.76 (-0.07) | 10.09 (+0.26 [-0.68, +1.27]) | 9.76 (-0.07) |
| inside-cut %, > 45 deg | 3.96 | 4.75 (+0.79 [+0.00, +1.54]) | 3.76 (-0.20) | 4.55 (+0.59 [+0.00, +1.22]) | 4.02 (+0.07) |
| cannot-make-turn %, > 45 deg | 4.28 | 3.30 (-0.99 [-2.09, +0.07]) | 4.48 (+0.20) | 4.15 (-0.13 [-0.93, +0.66]) | 4.28 (+0.00) |
| NC + TTC fail %, all | 2.36 | 2.67 (+0.30 [+0.15, +0.48]) | 2.33 (-0.03 [-0.13, +0.06]) | 2.49 (+0.13 [-0.02, +0.30]) | 2.38 (+0.02) |
| NC / DAC / EP / TTC / LK (sub-scores, all) | 98.49 / 96.32 / 86.97 / 97.78 / 97.13 | 98.27 / 96.68 / 87.22 / 97.53 / 97.27 | 98.49 / 96.43 / 86.99 / 97.80 / 97.17 | 98.37 / 96.51 / 87.14 / 97.68 / 97.23 | 98.44 / 96.32 / 86.98 / 97.74 / 97.13 |
| dev ADE to the log (m; decision 160: H0 0.62, WA-Cf memory 0.42) | 0.632 | 0.619 | 0.636 | 0.628 | 0.627 |

Reference row, not rerun: decision 160's MW (WA-Cf tokens, same channel and subset, P2H lambda-10 baseline) +0.95 [+0.47, +1.37].

Registered lines (copied): combined arm navtest >= +0.7 and CI low > +0.3 = geometry is enough; < +0.3 = not geometry, negative; between = partial.
Seed-0 gate (`geo_oracle/gate-s0.json`): GB - H0 < +0.3 and CI high (+0.34) < +0.7 -> clear negative, seed 1 not scored.

## Do the tokens carry the geometry? Yes (shown)

The tokenizers' throw-away thin head, on the 408 dev tokens (logs never seen by the tokenizer), true tokens vs the same tokens permuted across logs
(`geo_oracle/tok_*.json`; pre-registered check: ratio <= 0.9):

| tokenizer | dev ADE true / shuffled (m) | raw-footprint off-road, true / shuffled | ego box overlaps an agent box, true / shuffled | ADE fit rows / pilot-train rows / dev |
|:--|:--|:--|:--|:--|
| S (drivable SDF) | 1.02 / 2.34 (0.43) | 0.5% / 30.6% | 1.7% / 14.5% | 1.02 / 1.11 / 1.02 |
| A (agents) | 1.26 / 2.55 (0.49) | 9.3% / 18.9% | 1.5% / 10.5% | 0.93 / 1.10 / 1.26 |
| B (both) | 1.27 / 4.57 (0.28) | 0.2% / 32.6% | 0.7% / 20.1% | 0.64 / 0.94 / 1.27 |

A plan decoded from the tokens stays on the road and out of the boxes; with mismatched tokens it does not. Banks: label coverage 1.0, RMS spread
across the three shards and navtest < 3%, the shuffled bank has no row from its own log. Caveat that matters below: the head's dev ADE (1.0-1.3 m) is
**worse** than the policy's own (0.63 m), and tokenizer B over-fits its fit rows (0.64 -> 0.94 on pilot-train rows of the same logs -> 1.27 on unseen logs).

## Is the channel read? Not on the registered rule; only faintly on sub-scores

Rule (pre-registered): GB - GX or GB memory-off - GB on EPDMS with a CI excluding 0. Neither does:

| contrast (EPDMS, all) | value |
|:--|:--|
| GB - GX (shuffled control) | +0.18 [-0.06, +0.42] |
| GB memory masked at test - GB | -0.15 [-0.36, +0.05] |
| GS masked - GS | -0.16 [-0.41, +0.10] |
| GA masked - GA | -0.07 [-0.19, +0.06] |
| dev ADE, any memory arm vs H0 | 0.62-0.64 vs 0.63 (WA-Cf in decision 160: 0.42 vs 0.62) |

Below the registered rule there is a small, real use of the SDF tokens (not registered as evidence, reported as seen): masking GS's memory raises
all-token DAC failures by +0.37 pp [+0.15, +0.60], lowers NC + TTC failures by 0.30 pp [-0.47, -0.15], and moves > 45 deg cannot-make-turn back from
3.30% to 4.28% (+0.99 [+0.11, +1.85]) and inside cuts from 4.75% to 3.76%; GS vs GX: DAC fail -0.36 [-0.58, -0.15], NC + TTC +0.29 [+0.12, +0.48].
So with the SDF tokens the policy drives a little further and wider (EP +0.25 [+0.15, +0.34]): fewer off-road and cannot-make-turn failures, more
collisions and inside cuts, net zero on EPDMS. The agent tokens (GA) change nothing measurable: NC + TTC -0.03 [-0.13, +0.06], masked read identical.

## Reading

1. The registered line is met on the negative side, but under the pre-registered qualifier it must not be read as "the ceiling of geometry is below
   +0.3". The lane did not get the policy to read the tokens, so the ceiling is unmeasured.
2. The likely reason, not tested here: the tokens offer the imitation loss nothing. A head on them predicts the log worse (1.0-1.3 m) than the policy
   already does from vision (0.63 m), while WA-Cf tokens improved it to 0.42 m. Only the hinge has a reason to read them, and that is what the faint
   sub-score effect looks like (DAC down, progress up). Decision 160's gain may therefore come from what WA-Cf knows about the expert's path, not
   from geometry; this run cannot separate the two.
3. For plan.md's light branch: "predict SDF / occupancy from the front view and feed it as frozen tokens" is not supported by this pilot in either
   direction. Three attempts now (decision 166 raw raster, 166 stage B with lambda 30, this tokenizer) show that frozen geometry tokens on this
   channel at pilot scale are not picked up.
4. > 45 deg: nothing moves on DAC; the SDF tokens trade cannot-make-turn (-0.99) for inside cuts (+0.79), the same trade decision 166 saw with 9 000 steps.

## Limits, and what would overturn this

- One seed (gate stop; seed-1 checkpoints of all five arms were trained early to keep the cards busy and are **not scored**), pilot scale (25 k tokens,
  3 000 steps), one tokenizer design and seed, agents = the 16 nearest boxes at t0 with velocity, no future occupancy.
- The tokenizer was trained on log-overlapping (token-disjoint) navtrain rows; its tokens are more plan-predictive on pilot-train rows than on unseen
  logs (B: 0.94 vs 1.27 m), a train / test mismatch of the memory.
- The baseline already carries the lambda-30 hinge, which uses the same SDF as a training label; decision 160's +0.95 was measured over the lambda-10 baseline.
- Would overturn "not read": the same tokens with the tokenizer trained end to end inside the adapter (gradients from the policy's losses), or a
  longer / full-data run, giving GB - GX with a CI above 0. Would establish a real ceiling: a memory arm whose masked read drops by >= 0.3 EPDMS and
  whose gain over H0 is then compared with the registered lines. Would support reading 2: WA-Cf tokens with the expert-path information removed
  (e.g. regressed on geometry only) losing most of the +0.95.

## Setup

- **Recipe (all arms).** `pp_train --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`, split `navsim/op-parity-s234` (25 415 train / 408
  dev), 3 000 steps x 64, warmup 100, seed 0; memory arms `--mem geo_s | geo_a | geo_b | geo_x`, memory masked on 25% of rows from its own rng stream
  (same rows and anchors as H0). H0 = `GH0-F-s0`, retrained with the current code (the older same-recipe `SHP-F` runs were not reused).
- **Tokens.** 32 x 512 fp16 per token, the WA-Cf layout. Input rasters on the decision-148 grid (x -8..56 m, y +-24 m, 0.5 m): drivable SDF clipped
  to +-6 m; agent channels = signed distance to the nearest of the 16 nearest boxes at t0 plus that box's velocity (from the t0 / t0 + 0.5 s
  annotations) within 2 m. Tokenizer: 4 stride-2 convs + 1, 1.1 M parameters, pooled to 8 x 4; trained once (6 000 steps x 256) with a thin plan head
  on imitation + the drivable hinge (S, B) + the decision-158 agent hinge (A, B) on `navsim/op-parity-geotok-train@v1:905892337c03` (76 084 tokens
  of shards 0, 1, 5-11; token-disjoint from the pilot, log-disjoint from dev), then frozen. GX = the GB bank deranged across logs inside each data
  dir, navtest included.
- **Read-outs.** navtest through `jevdrive.bench` (v2 devkit, 12 146 tokens); log-cluster paired bootstrap (`jevdrive.stats.paired`, B 4 000). Turn
  strata by logged 4 s heading change (3 154 / 1 517 tokens > 20 / > 45 deg). Inside cut / cannot-make-turn from four_dirs' instrumented replay
  (`turn_oracle.py replay`), decision 166's definitions.

## Verified / not verified

- Verified: replay DAC equals bench DAC on every failing token of all 8 reads (403-451 each, 0 disagreements, a side for all); all trainings pass
  `pp_full_check.py train`; tokenizer checks pass; pre-registration and the seed-1 queueing note were pushed before the first score existed.
- Not verified: seed variance; that the tokenizer's fp16 bank (RMS 0.02-0.04, read through the adapter's LayerNorm) loses nothing; H0 against the
  older `SHP-F-s0` read (not compared).
- Not done: seed-1 scoring, navhard (no SDF / agent labels on its synthetic frames), HUGSIM, full data, any extra read after the gate.

## Deviations from the pre-registration

None in lines, gate or read-outs. Process: the seed-1 trainings were queued before the seed-0 read at the coordinator's request (declared in the
pre-registration before any score; training only, not scored after the gate stopped).

## Cost

GPU about 1.3 card-hours of a 5 card-hour budget: 3 tokenizers (1.4-2.6 min each plus preflights), 10 trainings of 4-7 min (5 of them the unscored
seed 1), 8 navtest plan exports. CPU: 8 navtest scorings, one replay.
