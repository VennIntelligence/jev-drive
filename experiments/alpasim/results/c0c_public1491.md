# C0c: five drivers on all 1491 public AlpaSim scenes (15 shards, 2026-10-09; decision 208, amends 201)

Extends [c0b_public700.md](c0b_public700.md) to every shard of the public nuPlan-track asset set (part001-part015; part015 has 91 scenes, 1491 scenes from 44 nuPlan logs). Same drivers, setup and
pre-registered OT30 line as C0b; scored scenes are reused (the simulator is deterministic, 24 of 24 overlap scenes reproduced exactly in C0b), only the 791 new scenes ran, in three batches
(b1: part006, 010, 011; b2: part004, 012; b3: part013, 014, 015). Nothing was submitted to AlpaSim. Full generated read-out: [c0c_report_1491.md](c0c_report_1491.md); per-scene table
[c0c_per_scene_1491.csv](c0c_per_scene_1491.csv) / [.json](c0c_per_scene_1491.json); numbers [c0c_stats_1491.json](c0c_stats_1491.json). Code: `scripts/c0c_chain.py` (at most three pool jobs at a time,
20-minute stall watchdog; it never fired in these runs), `scripts/c0b_report.py`. Shard labels of the new scenes are inferred as in C0b (contiguous 100-scene blocks); labels of the first 700 are unchanged.

## Result (1491 scenes, 44 logs)

| driver | mean scene score [95% CI] | score 0 | at-fault collision | offroad | left corridor | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 0.8883 [0.8677, 0.9092] | 139 | 32 | 44 | 51 | 76 | 0.922 |
| AP2-AB-s0 | 0.9038 [0.8835, 0.9232] | 132 | 32 | 36 | 52 | 68 | 0.967 |
| OT30-F-s0 | 0.8978 [0.8793, 0.9183] | 125 | 19 | 38 | 56 | 57 | 0.926 |
| OT30-F-s1 | 0.9127 [0.8947, 0.9312] | 106 | 19 | 29 | 46 | 48 | 0.932 |
| WA-JEPA (**reference**, fp32) | 0.8947 [0.8660, 0.9218] | 146 | 5 | 70 | 59 | 75 | 0.926 |

Twelve scenes (parts 006, 010, 011, 013-015) score 0 for every driver with no decision made: the scene route fails AlpaSim's sanity check ("route folds back on itself"). They are in all denominators and do not affect any difference.

## OT30 line (registered in plans/2026-10-09-ot30-closedloop-prereg.md, read as registered)

- OT30 mean of two seeds minus SH30: **+0.0170 [+0.0032, +0.0294]**; line: >= +0.010 and lower bound > 0 -> met. At-fault collision zeros 19 against 32 -> met. Verdict by the line: candidate.
- Seeds alone: s0 +0.0095 [-0.0057, +0.0229] (below the +0.010 on its own), s1 +0.0244 [+0.0102, +0.0377]. The seed gap (0.0149) is as large as the mean effect.
- The 700-scene reading was +0.0156 [-0.0011, +0.0326] (not met). Path of the mean difference and its lower bound as scenes were added: 700: +0.0156 / -0.0011; 1000: +0.0149 / +0.0018; 1200: +0.0199 / +0.0084; 1491: +0.0170 / +0.0032.
- Gain is in collisions (at-fault events 76 -> 57 / 48) and now also offroad + left-corridor zeros (SH30 95, OT30 94 / 75), mostly from s1.

## Other readings

- AP2 minus SH30 is now separable: +0.0156 [+0.0012, +0.0310] (700 scenes: not); collision zeros equal (32 / 32), the gain is in slow rollouts (120 against 249 with 0 < score < 1; progress 0.967 against 0.922).
- WA-JEPA (reference) 0.8947 sits between SH30 and AP2 and is not separable from any of ours (OT30 mean minus WA-JEPA +0.0105 [-0.0134, +0.0372]); its part001 lead is a shard effect (+0.0890 [+0.0561, +0.1217] against the other shards; ours within +-0.04). Collisions 5 against our 19-32, offroad 70.
- Best-of-k oracle (upper bound): SH30 + AP2 0.9354 (+0.0316 [+0.0244, +0.0390] over the best single), all four 0.9580 (+0.0454 [+0.0335, +0.0575]); the two OT30 seeds alone +0.0129 (seed-noise level).
- Per shard, part004 and part012 favour OT30 / AP2 and WA-JEPA respectively (part012: WA-JEPA 0.975, AP2 0.949, SH30 0.877); part006, 007, 010, 011, 015 are the hard ones for ours (0.83-0.88).

## Data note

The shard download had two faults besides the empty tree-API reply that killed it: two tmux windows fetched the same shards into the same range files, and a curl that returned an error page was appended as data
(size stays right, bytes wrong), so eight tarballs failed sha256; all eight were deleted and refetched and passed. Fixed in `scripts/fetch_data.sh` (one flock per shard, ranges appended only on HTTP 206, retried tree API, refetch once on mismatch).
The mirror gave 10-16 MB/s in total whatever the connection count.
