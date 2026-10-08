# NC / TTC failure taxonomy of SH30 and the same-path longitudinal scaling oracle (overnight N1)

Written 2026-10-09. Driver SH30 (`SH30-F-s0` / `s1`, decision 170), navtest 12 146 tokens (W frames) and navhard stage 1 (450 tokens, G frames).
Pre-registration [plans/2026-10-09-nc-taxonomy-prereg.md](../plans/2026-10-09-nc-taxonomy-prereg.md), committed before any per-token read. No model
was run: stored bench plans and per-token scores, decision 153's instrumented devkit replay for the collision events, and
`python -m jevdrive.bench score-poses --traffic non_reactive` for the scaled plans. WA-JEPA enters only through its stored per-token scores.
Code `scripts/nc_tax.py`, `scripts/nc_tax_chain.sh`. Every table (all classes, both sets, navhard) is in [nc_taxonomy/tables.md](nc_taxonomy/tables.md);
per-token rows in `nc_taxonomy/navtest_fail_tokens.csv`. This is not a go / no-go item.

## Rules (as registered)

- **Sets**: NC failure = NC < 1; TTC-only = TTC < 1 with NC = 1. Counts are per seed; `n` is the seed mean. Shares pool both seeds; 95% CIs are
  cluster bootstraps over logs (`jevdrive.stats`, B 10 000).
- **Class**: decision 153's `ctype_class` on the first at-fault collision (NC set) or first TTC event (TTC-only), merged to the five classes of the task
  book; **D side contact** = nuPlan lateral collision, or a stopped vehicle whose centroid at the event is beside the ego body (dx < 4.05 m, |dy| >= 1.0 m).
- **Buckets**: bench strata, |logged 4 s heading change| and t0 ego speed.
- **Oracle**: same path, arc length x a (`turn_ceiling.speed`); main family a in {0.7, 0.8, 0.9, 1.0, 1.1}, extended adds {0.5, 0.6}.
  Recovered = some a != 1 passes the set's criterion; clean = that a also raises the token's no-EC EPDMS; a* = best-scoring recovering main scale.

Checks: the replay's 8 sub-scores equal the bench archive on every failing token (max |diff| 1e-16 navtest, 0 navhard; 0 rows dropped); the
a = 1.0 rows of both scoring runs equal the archive on all 12 146 tokens (identity gate, 0 differing rows).

## 1. What fails (navtest)

| set | n (s0 / s1) | rate | WA-JEPA passes the token | event at ego < 3 m/s | plan > 1.1 x logged arc |
|:--|--:|--:|--:|--:|--:|
| NC failure | 176.5 (180 / 173) | 1.45% | 76% | 42% | 63% |
| TTC-only | 87.5 (88 / 87) | 0.72% | 63% | 18% | 27% |

WA-JEPA's own counts on the same tokens: 75 NC failures, 67 TTC-only.

**NC failures by class** (n = seed mean; SH30-specific = SH30 fails and WA-JEPA passes):

| class | n | share % [95% CI] | WA-JEPA passes % | SH30-specific n | SH30-specific share of all NC failures % | event < 3 m/s % | plan > 1.1 x log % | recovered, main % [95% CI] | recovered, extended % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A stopped / slow vehicle ahead** | **93.5** | **53 [43, 62]** | 79 [69, 90] | **73.5** | **42 [34, 50]** | 51 | 73 | 77 [65, 89] | 95 |
| - A1 stopped vehicle ahead | 58.5 | 33 | 81 | 47.5 | | 49 | 57 | 68 | 92 |
| - A2 lead vehicle (moving) | 35 | 20 | 74 | 26 | | 54 | 100 | 93 | 100 |
| B cut-in | 23 | 13 [6, 22] | 61 [35, 87] | 14 | 8 [3, 14] | 74 | 94 | 91 [82, 100] | 100 |
| C crossing at junction | 13.5 | 8 [0, 17] | 59 [47, 80] | 8 | 5 [0, 9] | 44 | 74 | 59 [35, 100] | 93 |
| D side contact | 16.5 | 9 [5, 15] | 97 [89, 100] | 16 | 9 [5, 15] | 0 | 6 | 73 [56, 90] | 91 |
| E other (static 14.5, oncoming 8.5, VRU 7) | 30 | 17 [11, 24] | 77 [55, 96] | 23 | 13 [8, 20] | 10 | 32 | 60 [41, 79] | 97 |

**NC failures, class x turn bucket** (seed-mean counts; bucket failure rate in the last row):

| class | < 5 deg | 5-20 | 20-45 | > 45 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 23 | 10 | 10.5 | 15 | 58.5 |
| A2 lead vehicle (moving) | 33 | 2 | 0 | 0 | 35 |
| B cut-in | 11.5 | 10.5 | 1 | 0 | 23 |
| C crossing / turn conflict | 8.5 | 2 | 2 | 1 | 13.5 |
| D side contact | 6.5 | 3 | 1 | 6 | 16.5 |
| E static object / VRU / oncoming | 16.5 | 5 | 4 | 4.5 | 30 |
| all | 99 | 32.5 | 18.5 | 26.5 | 176.5 |
| NC failure rate of the bucket | 1.55% | 1.25% | 1.13% | 1.75% | 1.45% |

**NC failures, class x t0 ego speed band** (m/s):

| class | < 2 | 2-5 | 5-10 | >= 10 | all |
|:--|--:|--:|--:|--:|--:|
| A1 stopped vehicle ahead | 19 | 28.5 | 11 | 0 | 58.5 |
| A2 lead vehicle (moving) | 8.5 | 7.5 | 19 | 0 | 35 |
| B cut-in | 2 | 15 | 6 | 0 | 23 |
| C crossing / turn conflict | 2 | 8.5 | 3 | 0 | 13.5 |
| D side contact | 0 | 5 | 8.5 | 3 | 16.5 |
| E other | 3 | 11.5 | 15.5 | 0 | 30 |
| all | 34.5 | 76 | 63 | 3 | 176.5 |
| NC failure rate of the band | 1.36% | 2.20% | 1.22% | 0.31% | |

How to read it:

- More than half of the NC failures are a vehicle ahead in the ego lane (A), and four fifths of those are SH30-specific. The class is not a turning
  problem: 59% of A is on < 5 deg tokens, and the NC failure rate is flat across turn buckets (1.1-1.8%).
- It is a speed problem. 73% of A plans cover more than 1.1 x the logged 4 s arc (the board base rate for P2H was 13%, decision 153), all 35 moving-lead
  cases do, and 51% of the A events happen with the ego below 3 m/s: the plan keeps rolling or pulls away while the logged car holds. The largest
  single cells are a moving lead at 5-10 m/s on straight tokens (17), a stopped vehicle at t0 speed < 2 m/s on straight tokens (12.5, WA-JEPA passes
  all of them) and a stopped vehicle at 2-5 m/s on > 45 deg tokens (12).
- The 2-5 m/s band has the highest NC failure rate (2.20%), and 87% of the events of the < 2 m/s band happen below 3 m/s: the low-speed signature of
  the two AlpaSim at-fault collisions is also the navtest one.
- **Side contact while turning** is small: D is 16.5 tokens, 7 of them on >= 20 deg tokens (4% of the NC failures). It is almost entirely
  SH30-specific (97%) and is a path problem (6% over-speed, 0% below 3 m/s).
- Against decision 153 (P2H, NC or TTC together 266, WA-JEPA failing 30% of them): SH30 has 264 with the same picture, so the strong hinge did not
  move this block; "stopped vehicle ahead" by 153's own rule is 40% of the NC failures here (36% of P2H's D3).

**TTC-only** (87.5): spread differently. E other is 48% (static object 21, oncoming 11.5, VRU 9.5), A 27% (20 of 23.5 stopped), B 12%, C 7%, D 6%;
77% of TTC-only failures are on >= 5 deg tokens with a peak rate at 20-45 deg (1.62%); only 27% are over-speed. WA-JEPA passes 63%, and only 29% of
the cut-in and 22% of the oncoming cases. Full tables in `nc_taxonomy/tables.md`.

## 2. Same-path longitudinal scaling oracle (navtest, non-reactive)

| family | NC recovered % [95% CI] | NC clean % | TTC-only recovered % | TTC-only clean % |
|:--|--:|--:|--:|--:|
| 0.9-1.1 | 39 [33, 47] | 35 | 47 [37, 58] | 39 |
| 0.8-1.1 | 61 [53, 70] | 58 | 61 [51, 72] | 52 |
| **0.7-1.1 (main)** | **74 [66, 83]** | **70 [61, 79]** | **73 [62, 84]** | **62 [49, 74]** |
| 0.5-1.1 (extended) | 95 [93, 98] | 93 | 83 [71, 93] | 75 |
| slower only (0.7-0.9) | 73 [64, 82] | 69 | 67 [56, 79] | 55 |
| faster only (1.1) | 3 [1, 6] | 3 | 13 [7, 19] | 11 |

- **Recovery**: slowing the same path by at most 30% clears 74% of the NC failures (70% without trading another gate), 95% with up to 50%.
  By class (main family): moving lead 93%, cut-in 91%, side contact 73%, stopped vehicle 68% (92% extended), static object 69%, crossing 59%,
  oncoming 59%, VRU 43%. The stopped-vehicle cases at t0 speed < 2 m/s on straight tokens recover only 44% in the main family: there the plan should
  not move at all, and a 0.7 scale of a launch still reaches the vehicle.
- **EP cost on the recovered tokens** (a* = best-scoring recovering scale): EP -3.0 points on the 131 NC tokens (a* = 0.9 on 57, 0.8 on 39, 0.7 on
  31.5, 1.1 on 3.5), -3.7 on the 64 TTC-only tokens; no-EC EPDMS on them +88 / +32 points.
- **Board-level upper bound** (a* only on SH30's failing tokens with a clean recovery, 178.5 tokens per seed; everything else unchanged):
  NC +0.98, TTC +1.27, **EP -0.05**, **no-EC EPDMS +1.13** (s0 +1.10 / s1 +1.15); extended family NC +1.30, EP -0.09, no-EC EPDMS +1.48.
  This is privileged (the scale is picked with the simulator's score) and non-reactive.
- **Uniform scaling has no free setting** (all 12 146 tokens, change against a = 1.0, seed means):

| a | NC | TTC | DAC | EP | no-EC EPDMS | NC failures (new / fixed) | TTC-only failures |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.5 | +0.25 | -2.66 | -0.05 | -23.70 | -9.77 | 142 (127.5 / 162) | 485.5 |
| 0.6 | +0.50 | -1.20 | +0.26 | -18.27 | -6.31 | 111.5 (86 / 151) | 317 |
| 0.7 | +0.68 | +0.05 | +0.52 | -13.16 | -3.54 | 92 (43 / 127.5) | 171 |
| 0.8 | +0.69 | +0.65 | +0.65 | -8.39 | -1.44 | 91.5 (18.5 / 103.5) | 85.5 |
| 0.9 | +0.48 | +0.56 | +0.46 | -3.97 | -0.33 | 116 (5 / 65.5) | 70 |
| 1.0 | 0 | 0 | 0 | 0 | 0 | 176.5 | 87.5 |
| 1.1 | -0.76 | -0.88 | -0.96 | +3.39 | -0.98 | 272 (100.5 / 5) | 112 |

  A blanket 0.9 buys NC +0.48 for EP -3.97 and is net negative (-0.33); a blanket 0.8 halves the NC failures and costs 1.44 EPDMS. The value is
  entirely in knowing where to slow down: the directed oracle costs 0.05 EP for the same NC. Slowing everything also creates new failures (43 new NC
  failures at 0.7; TTC-only failures double at 0.7 and grow 5.5x at 0.5), because the logged traffic does not react to a slower ego.

## 3. navhard stage 1 (450 tokens, counts only)

SH30 fails NC on 13 / 14 tokens (s0 / s1) and TTC-only on 1 / 1; WA-JEPA on 9 and 4. NC by class (seed mean): stopped vehicle ahead 6 (WA-JEPA passes
58%), oncoming 4 (100%), VRU 2 (0%), cut-in 1, side contact 0.5; by turn bucket 5.5 / 2 / 3 / 3; 9 of 13.5 at 5-10 m/s. Too few to conclude anything
beyond "stopped vehicle ahead is again the largest class".

## 4. Which class the next NC fix should target

By the registered rule (largest SH30-specific count): **A, a stopped or slow vehicle ahead in the ego lane, approached too fast at low speed on
mostly straight tokens**: 73.5 of the 176.5 NC failures are SH30-specific there (42% [34, 50]; the next class is 13%), and 77% [65, 89] of the
class is recovered by slowing the unchanged path by at most 30%. The fix is a longitudinal one (when to slow and when not to launch), not a path
one; turning-time side contact is 4% of the NC failures.

## Limits

- Class labels are heuristics on the scorer's collision state; the side flag uses the object centroid, not the contact point. For TTC-only events the
  position is the simulated ego state at the event step, not the constant-velocity projection that triggered it.
- Non-reactive traffic: other vehicles replay the log, so recovery by slowing is optimistic for closed loop (no rear-end is charged), and new
  failures of a blanket slowdown are pessimistic for the same reason.
- The oracle picks the scale per token with the simulator's score: an upper bound, not a method. No-EC EPDMS throughout (EC is undefined for scaled plans).
- The scaled plan goes through the devkit's LQR tracker, so the path is the same only at plan level; "clean" recovery accounts for the gate flips.
- The over-speed base rate (13%) is P2H's from decision 153; SH30's own base rate was not recomputed.
- navhard: stage 1 only, 14 failing tokens per seed; stage 2 not done.
- Decisions 158 / 179 found this block hard to fix on frozen features (agent hinge, own lead read-out); this page says where the failures are and
  what a perfect longitudinal correction is worth, not that it is learnable.

## Files

- Tables: `nc_taxonomy/tables.md`; `nc_taxonomy/navtest_{NC,TTC-only}_{class,subclass,fine}.csv`, `navtest_class_turn_speed.csv`,
  `navtest_fail_tokens.csv`, `oracle_{recovery,ep_cost_tokens,nc_by_class,ttc_by_class,board_upper}.csv`, `uniform_scaling.csv`,
  `navhard_s1_*.csv`, `summary.json`.
- Box: `$DATA_DIR/runs/op_parity/nc_tax/` (replay parquet + states, `poses.npz`, `score_fail.csv`, `score_all.csv`, gates, chain log).
