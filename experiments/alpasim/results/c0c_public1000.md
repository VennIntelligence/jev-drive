# C0c: five drivers on 1000 landed public AlpaSim scenes (2026-10-09, batch b1; decision 208, amends 201)

Extends [c0b_public700.md](c0b_public700.md) with the shards that landed after it: part006, part010, part011 (300 new scenes, 35 nuPlan logs in total). Same drivers, setup, and
pre-registered OT30 line as C0b; scored scenes are reused (the simulator is deterministic, 24 of 24 overlap scenes reproduced exactly in C0b), only the 300 new scenes ran.
Nothing was submitted to AlpaSim. Full generated read-out: [c0c_report_1000.md](c0c_report_1000.md); per-scene table [c0c_per_scene_1000.csv](c0c_per_scene_1000.csv) / [.json](c0c_per_scene_1000.json);
numbers [c0c_stats_1000.json](c0c_stats_1000.json). Code: `scripts/c0c_chain.py` (at most three pool jobs at a time, 20-minute stall watchdog), `scripts/c0b_report.py`.
Shard labels of the new scenes are inferred the same way as in C0b (contiguous 100-scene blocks); the labels of the first 700 are unchanged.

## Result (1000 scenes, 35 logs)

| driver | mean scene score [95% CI] | score 0 | at-fault collision | offroad | left corridor | at-fault events |
|:--|:--|--:|--:|--:|--:|--:|
| SH30-F-s0 | 0.8952 [0.8699, 0.9189] | 89 | 21 | 27 | 31 | 48 |
| AP2-AB-s0 | 0.9005 [0.8762, 0.9232] | 92 | 25 | 23 | 34 | 48 |
| OT30-F-s0 | 0.9066 [0.8851, 0.9272] | 78 | 14 | 22 | 32 | 36 |
| OT30-F-s1 | 0.9136 [0.8887, 0.9370] | 73 | 15 | 18 | 30 | 33 |
| WA-JEPA (**reference**, fp32) | 0.8844 [0.8472, 0.9198] | 108 | 4 | 55 | 39 | 59 |

Ten scenes (parts 006, 010, 011) score 0 for every driver with no decision made: the scene route fails AlpaSim's sanity check ("route folds back on itself"); they are in all denominators and do not affect any difference.

## OT30 line (registered in plans/2026-10-09-ot30-closedloop-prereg.md, read as registered)

- OT30 mean of two seeds minus SH30: **+0.0149 [+0.0018, +0.0274]**; line: >= +0.010 and lower bound > 0 -> met (700 scenes: +0.0156 [-0.0011, +0.0326], not met).
- At-fault collision zeros 14.5 against 21 -> met. Verdict by the line: candidate. Seeds alone: s0 +0.0113 [-0.0030, +0.0241], s1 +0.0184 [+0.0045, +0.0321].
- Offroad + left-corridor zeros: SH30 58, OT30 54 / 48.

## Other readings

- WA-JEPA (reference) 0.8844 stays below every one of our drivers on the mean but not separable (OT30 mean minus WA-JEPA +0.0257 [-0.0062, +0.0582]); its part001 lead is still a shard effect (+0.1037 [+0.0628, +0.1455] against the other shards, our drivers within +-0.03).
- Best-of-k oracle (upper bound): SH30 + AP2 0.9330 (+0.0325 [+0.0233, +0.0423] over the best single), all four 0.9551 (+0.0415); two OT30 seeds alone +0.0140.
- Hardest shards for all of ours: part006 (0.84-0.89), part007, part010, part011 (0.83-0.88).

## Data note

The download of the remaining shards had a second fault besides the empty tree-API reply: two tmux windows fetched the same shards into the same range files, and a curl that returned an error page
was appended as data, so eight tarballs failed sha256 (all eight were refetched; part010 was the first to pass after the fix). Fixed in `scripts/fetch_data.sh` (one flock per shard, ranges appended only on HTTP 206, retried tree API).
