# op_parity nt-cache: v2 metric cache of navtrain and simulator labels for SH30's turn plans

Written 2026-10-08. Code: `scripts/nt_cache.py` (cache builder), `scripts/nt_labels.py` (plans, candidates, scoring, table). Operating notes and
resume commands: [docs/bench.md](../../../docs/bench.md#v2-navtrain-metric-cache-and-turn-labels-nt-cache). CPU only for the cache; one short
GPU pass (about 20 card-minutes) for SH30's navtrain plans. No model is trained.

## Answer

1. **The v1 navtrain cache cannot serve the v2 scorer.** A v1 `metric_cache.pkl` has 7 fields (`centerline`, `drivable_area_map`, `ego_state`,
   `file_path`, `observation`, `route_lane_ids`, `trajectory`); the v2 devkit's `MetricCache` has 16 and the scorer reads the missing ones
   (`human_trajectory`, `past_human_trajectory`, `current_ / future_tracked_objects`, `map_parameters`, `scene_type`, `timepoint`, ...). The build was needed.
2. **The v2 cache of the 28 323 navtrain turn tokens (|dyaw| >= 20 deg) is built:** `$DATA_DIR/runs/navsim/metric_cache/v2_navtrain/`, 28 323 / 28 323 tokens,
   no missing, 11.17 GB, 394 KB per token, 3.20 core-s per token (90 577 core-s = 25.2 core-h; about 30 min wall on 4 pool jobs x 12 cores).
   Same devkit and settings as v2_navtest: control gate below.
3. **The rest of navtrain is not built (stop rule).** Measured on 4 497 rest tokens (3 shards, 437-476 KB per token): 456 KB per token, 2.83 core-s per token.
   The rest (74 965 tokens) extrapolates to 34 GB, the whole navtrain cache to 45 GB, above the 40 GB limit; the disk has 491 GB free of which 458 GB are
   reserved for AlpaSim. 5 480 rest tokens (4 497 + 983 from an earlier shard plan, 35 in both, so 948 outside the finished shards) are in the cache as a by-product.
   Finishing the rest would add about 32 GB and about 57 core-h (about 70 min on 4 x 12 cores).
4. **Labels** (SH30-F-s0 / s1 on the 28 323 turn tokens, `score-poses --traffic non_reactive`, turn-ceiling's path): per-token sub-scores of the identity
   plan and of speed x 0.8 / x 0.6, `labels/labels_turn.csv.gz` (keyed by token). Base rates below.

## Base rates (28 323 navtrain turn tokens, seed 0 / seed 1)

Sub-score failure = 0 (DAC and NC are 0 / 1 or 0 / 0.5 / 1 multipliers). "repairs" = share of the tokens where the identity plan fails and the variant passes;
"breaks" = share of the tokens where the identity plan passes and the variant fails. EPDMS without EC, x 100.

| | identity | speed x 0.8 | speed x 0.6 |
|:--|:--|:--|:--|
| DAC fails (identity) | 4.48% / 4.55% | | |
| DAC repaired / broken | | 49.1% / 51.2%; broken 1.45% / 1.41% | 70.1% / 71.7%; broken 3.73% / 3.63% |
| NC = 0 fails (identity) | 1.15% / 1.16% (NC < 1: 1.40% / 1.44%) | | |
| NC repaired / broken | | 66.7% / 66.8%; broken 0.14% | 93.6% / 93.6%; broken 0.73% / 0.75% |
| DAC or NC fails (identity) | 5.51% / 5.58% | repaired 52.1% / 53.9% | repaired 73.9% / 75.3% |
| mean EPDMS, change vs identity | 88.70 / 88.65 | -0.76 / -0.61 | -6.12 / -5.95 |

Best of {identity, x 0.8, x 0.6} per token (privileged): +4.89 / +5.00 EPDMS over the identity. By bucket (seed 0 / 1): 20-45 deg (17 550 tokens)
DAC fail 3.67% / 3.74%, best-of-3 +4.50 / +4.63; > 45 deg (10 773) DAC fail 5.80% / 5.87%, best-of-3 +5.52 / +5.60. All 5 groups x 2 seeds x all columns:
[nt_cache/base_rates.md](nt_cache/base_rates.md) (copy of `$DATA_DIR/runs/op_parity/nt_cache/labels/base_rates.csv`).

**These plans are in-sample.** SH30 was trained on `navsim/op-parity-full-train` (101 499 of the 103 288 navtrain tokens); 27 892 of the 28 323 turn tokens are in it.
The 431 held-out turn tokens (the log-disjoint dev rows) have DAC fail 5.34% / 5.80% and EPDMS 86.6 / 86.2, against 4.47% / 4.53% and 88.7 / 88.7 in-sample and
7.31% / 83.7 on navtest > 20 deg (decision 178); with 431 tokens the held-out interval is about +-2 points, so in-sample and held-out are not separated, but the
navtest level is not reproduced on navtrain. A gate trained on these labels sees fewer and different failures than navtest has; cross-fitted or held-out SH30
plans would be needed for a faithful label (not done: no second SH30 trained on a navtrain fold exists).

## Cost and gates (staged launch, 2026-10-08, shared box)

| step | tokens | wall / cores | core-s per token | KB per token |
|:--|--:|:--|--:|--:|
| control: 10 navtest tokens rebuilt with `nt_cache.py`, compared with v2_navtest | 10 | 20 s / 12 | | 193 (1 928 064 vs 1 927 904 B; the 160 B is the `file_path` string) |
| stage 1: 1 turn shard | 657 | 163 s / 12 | 2.98 | 532 |
| stage 10: 10 turn shards spread over the date-sorted logs | 6 335 | 1 592 s summed / 12 each | 3.02 | 392 |
| turn, all 45 shards | 28 323 | 7 548 s summed / 12 each, 4 jobs | 3.20 | 394 |
| rest, first plan (5 logs, one with 549 tokens) | 983 | 716 s / 12 | 8.74 | 602 |
| rest, balanced plan, 3 shards | 4 497 | 1 060 s summed / 12 each | 2.83 | 456 |

- **Control gate (same devkit and settings as v2_navtest):** 10 random v2_navtest tokens rebuilt with the same command line (only `train_test_split=navtest`);
  15 fields x 10 tokens re-pickled and compared, 0 differ (`file_path` excluded).
- **Checklist after stage 1 and stage 10:** 100% of the tokens cached, 0 missing, empty staging area, cached log of every token = the log of the v1 navtrain tree,
  per-token size and cost stable across shards of the same plan except the city effect (144-591 KB per token between the turn shards of the first plan; the balanced rest shards, which mix many logs, are 437-476).
- **Turn-ceiling's estimate** (1.6 h turn / 5.7 h all, by extrapolation) was on the pessimistic side: measured about 30 min wall for the turn subset at 48 cores
  (25 core-h); the byte size was not estimated there. Nothing here contradicts decision 178.
- **Devkit parallelism is per log.** A log is one serial unit. Turn logs are small (31 tokens on average), so whole logs are fine; rest logs hold up to 549 tokens, so
  the rest plan cuts logs into pieces of at most 64 tokens, deals them largest-first to shards of about 1 500 tokens and never puts two pieces of one log in one shard
  (first rest plan: 8.7 core-s per token because most cores idled behind the 549-token log; balanced plan: 2.8).
- **Scoring path on the new cache:** `score-poses --mcache v2_navtrain` (new option, default unchanged, run identity of old runs unchanged,
  `tests.test_bench.TestPoses` passes). 28 323 tokens x 6 keys scored in 370 s on 5 pool jobs x 12 cores (0.088 core-s per key). The identity plans' median ADE to the logged future
  is 0.52 m (both seeds): the exported navtrain poses are sane.

## Verified / not verified

- Verified: v1 cache lacks the v2 fields (unpickled both); all 45 turn shards complete, `nt_cache.py verify` finds every cached token's `metric_cache.pkl` in the final cache
  with the right log; control gate; score.csv has 28 323 x 6 rows with no duplicates; the bucket counts equal decision 178's (17 550 / 10 773).
- Not verified: the navtrain cache against an independent navtrain v2 score (none exists; the guarantee is the navtest control plus identical code path); per-token
  agreement of the SH30 navtrain plans with an archived export (none exists; checked only by ADE); the held-out base rates (431 tokens); behaviour when the
  pool dispatcher is down at resume (the worker has a stand-alone mode, described in docs/bench.md, which was not exercised); the labels for tokens outside the turn subset.
