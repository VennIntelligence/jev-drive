# P4 self-branch search: box read (2026-10-14)

Prereg: [2026-10-14-p4-self-branch-prereg.md](../../plans/2026-10-14-p4-self-branch-prereg.md) (with Amendments 1-2). Smoke: [smoke.md](smoke.md).
Instrument and runner: `rf_p4.py`, `rf_p4.sh`, driver mode `ONP_CELLS`. Code served: [code.json](code.json) (commit 652d48b7, clean worktree).
Run: GPU box, 27 scene jobs through the pool, 693 rollouts (27 phase A + 666 phase B), 0 scene jobs failed, 0 retries.
Files: [read.json](read.json) (per rollout), [table.md](table.md) (per scene x D x cell), [summary.json](summary.json), [secondary.md](secondary.md)
(the numbers below, from [rf_p4_summary.py](../../scripts/rf_p4_summary.py)), [timing.md](timing.md), [p4_run_log.txt](p4_run_log.txt), rows: [pool_self/](pool_self/).
Videos (2026-10-14): camera re-runs of identity + best completing cell per scene (plus the first batch's wider selection behind "more"), each checked bit-identical to the stored rollout (`rf_p4.sh vfanout` / `vcollect`, box `$DATA_DIR/runs/route_follow/p4/video`); watch on the Tokyo desktop: `~/Desktop/p4_videos.html` ([p4_videos.py](../../scripts/p4_videos.py), human log-following rollouts from `/data/runs/route_night/pait/pass`).
Box vs Tokyo is not bit-identical (from k 0), so these rollouts are compared to nothing from Tokyo's r281 reference. Determinism check: for all 648
self rollouts the plans before h equal phase A's identity at the same k (0 of 648 differ).

## Headline (the registered line)

**Turns with a completing branch cell at >= 2 distinct handovers with realized entry speed <= class v_e + 1 m/s: 10 / 27 = 37 %. INCONCLUSIVE**
(pass >= 17 / 27 = 60 %, fail <= 9 / 27 = 35 %). 10 / 27 on self handovers alone (the 2 src=log distances add nothing to the line).
The 10 turns: 8a365ac2, a4be61d6, e5e47fd2, 232d240e, 268949d2, 37d660b1, 96da4f1a, b40f4729, dc1966f4, ea468790.

| count | turns / 27 |
|:--|--:|
| identity (the policy alone) completes | **9** |
| >= 1 completing branch cell at some distance | **20** |
| >= 2 distinct handovers with a completing cell | 11 |
| the line (the above, entry speed <= v_e + 1) | **10** |
| >= 1 completing cell that passes the full shape gate | **13** |
| >= 2 distinct handovers with a gate-passing completing cell | **7** |
| max reachable for the line (scenes with >= 2 distinct handovers) | 23 (4 scenes have one distinct handover) |

The line sits 1 turn above the fail threshold; 4 of the 10 counted turns are not identity completions (8a365ac2, e5e47fd2, 232d240e, 37d660b1),
and e5e47fd2's cells (g1.25-ve08 at D40 / D30) fail the shape gate (its entry speed is 13.8 m/s on the identity). Nothing here is a claim about
rows training the turn; the per-scene reading is below.

## (a) Identity (the policy alone) completes 9 / 27

96eb4700, a4be61d6, c6c01d25, 268949d2, 77fdd75c, 96da4f1a, b40f4729, dc1966f4, ea468790. Of these, 3 have one distinct handover
(96eb4700, c6c01d25, 77fdd75c) and cannot reach the line. The other 18 turns fail the identity; a completing branch there is a new completion.
Turns that the identity fails and a branch completes (see [table.md](table.md)): 8a365ac2, 232d240e, 37d660b1, 403d558a, ddc3e8df, e5e47fd2, 1e8e939e (log fallback), 213dfdac (D0 only), 98694f91 (D0 only),
a174faed (D10 / D0), f9e361aa (D20): 11 turns.

## (b) Where the identity rollout first fails

Full per-scene table in [secondary.md](secondary.md) (b). `end rel entry` = the ego's route station at the zero event minus the strict entry.

- The identity fails **before the turn entry in only 2 of 27** scenes (1e8e939e collision, 9 m before the entry; 3232fdb7 corridor, 4 m before),
  and these are exactly the 2 scenes whose D0 handover is unreached and taken from the log (src=log). The other 16 failing identities zero
  **after** the entry: inside or after the turn (corridor 8, offroad 5, collision 2, end of run without completing 1).
  So 16 of the 18 identity failures are turn failures, and 25 of 27 turns give the branch search a state at the entry.
- Four failures are early and fast, not turn failures in the shape sense: e5e47fd2, 403d558a, eaba3f6a and f9e361aa enter the turn at 11-14 m/s
  (identity realized entry speed 13.8, 11.8, 12.5, 11.4 m/s against v_e 4-5) and leave the corridor within 3-5 m of the entry.

## (c) Completions per cell, per profile, per D

Distinct rollouts (a distance that shares its handover decision with a larger D is run once, the table counts it under the largest D).
Done = P2's completion rule; gate = completing and passing the full shape gate.

| cell | rollouts | done | done + gate |
|:--|--:|--:|--:|
| g1-raw (identity at an unreached distance, src=log) | 2 | 1 | 1 |
| g1-ve | 83 | 8 | 6 |
| g1-ve08 | 83 | 8 | 6 |
| g1.25-raw | 83 | 22 | 14 |
| g1.25-ve | 83 | 23 | 13 |
| g1.25-ve08 | 83 | 20 | 9 |
| g1.5-raw | 83 | 10 | 3 |
| g1.5-ve | 83 | 6 | 4 |
| g1.5-ve08 | 83 | 8 | 5 |

| profile | rollouts | done | done + gate | rear-ended | crawl approach | crawl inside |
|:--|--:|--:|--:|--:|--:|--:|
| raw | 168 | 33 | 18 | 6 | 55 | 11 |
| ve | 249 | 37 | 23 | 6 | 82 | 21 |
| ve08 | 249 | 36 | 20 | 24 | 82 | 25 |

| gain | rollouts | done | done + gate |
|:--|--:|--:|--:|
| g1 | 168 | 17 | 13 |
| g1.25 | 249 | 65 | 36 |
| g1.5 | 249 | 24 | 12 |

Per D (done / rollouts): D40 50 / 216, D30 4 / 80, D20 3 / 80, D10 5 / 104, D0 44 / 186 (turns with >= 1 completing cell: D40 14, D30 2, D20 2, D10 2, D0 13).
D40 and D0 carry the completions; D30 / D20 / D10 are distinct handovers in only 10 / 10 / 13 of the 27 scenes (see (e)) and complete in 2 / 2 / 2 turns.
Per D x profile: [secondary.md](secondary.md) (c).

What the three profiles say against each other: the profiles differ little in completions (raw 33, ve 37, ve08 36 of 168 / 249 / 249, i.e. 20 %, 15 %, 14 % per rollout; per
distinct cell raw is the highest rate because it is the cheapest). The gain is the lever: g1.25 completes 65 of 249 (26 %), g1 17 of 168 (10 %), g1.5 24 of 249 (10 %).
ve08 is the worst for rear-ending (24 rear-ended rollouts against 6 and 6): the slower branch is hit by the log's following car, which does not react. The profile binds
(s_k < 0.99 at some decision) in 450 of 498 ve / ve08 rollouts, so the near-equal rates are not a no-op.
Median realized entry speed of the completing branches 3.76 m/s (the class v_e of those turns: median 4.6, the human median over the 27 turns 4.01).

## (d) Fallback, rear-ended, crawl, shape gate

- **src=log**: 2 of the 83 distinct distances (1e8e939e D0 and 3232fdb7 D0, the two scenes whose identity fails before the entry), 18 rollouts, 8 complete (1e8e939e: all 8 cells, all pass the shape gate; 3232fdb7: none).
- **Rear-ended** (collision_rear): 36 rollouts (raw 6, ve 6, ve08 24), 17 of them counted done (the log's following car does not react).
- **Crawl (min speed between h and the entry < 0.5 v_e)**: 219 of 666 branch rollouts (D0 starts at the entry: 0); among the 106 completing ones 32.
  **Crawl inside [entry, exit]**: 57 of 666; among the completing 6.
- **Shape gate** (class p95 kappa and dkappa, a_lat <= 4.42, entry speed <= v_e + 1): of the 106 completing branch cells **61 pass** (of these 53 on self handovers, 8 on log).
  Failing parts among the 106 (overlapping): kmax 34, dk 26, alat 9, v_entry 10. 96 of the 106 have an entry speed <= v_e + 1.
- Scenes that failed to run: **none** (27 / 27 jobs DONE, no ERROR, no retry).

## (e) Collapsed handovers

The clip starts within 40 m of the entry on many scenes, so several distances land on the same handover decision (17, the end of force-GT).
Distinct handover decisions per scene: 1 distinct: 4 scenes (96eb4700, c6c01d25, 77fdd75c, ddc3e8df), 2 distinct: 10, 3 distinct: 3, 5 distinct: 10; 83 in total.
So 14 scenes have only one or two distinct handovers; the 4 with one cannot reach the line (max reachable 23 / 27); the 10 with two can reach it
only if both completing handovers are D40 (h = 17) and D0. Handover table per scene: [secondary.md](secondary.md) (e).

## Rows: on1_SELF (builder run, counting only; no merge, no training)

`rf_selfpool.py build --backend box --roots <P4 run> --tag on1_SELF --served <run>/code` as the pool job `selfpool` (--vram 6 --cpu 16 --ram 40), box
work dir `$DATA_DIR/runs/route_follow/selfpool_SELF`, pool `$DATA_DIR/runs/op_parity/cache/on1_SELF` (144 MB). Wall about 9 min. Outputs: [pool_self/](pool_self/)
(`coverage_SELF.md`, `pool_gates_SELF.json`, `branches.json`, `failing.json`). The serving tree was the frozen worktree of the run (G-in then checks the replay).

**4 528 rows from 67 branches over 14 turns** (115 completing rollouts selected: 106 branch cells + 9 identity; G-shape passes 67; 578 failing rollouts over 27 scenes in failing.json).

| gate | value | pass |
|:--|:--|:--|
| G-in (replayed inputs = served) | dumps 2 300 / 2 300, drive.jsonl records 22 875 / 22 875 | yes |
| G-route (regenerated first route point vs route0 <= 0.05 m on >= 99 %) | 94.0 % of 4 528 rows, max 0.275 m | **no, by the letter** (affects only the route-hinge label; the memory input is the served one) |
| G-rmem | max abs diff 1.4e-4 | yes |
| G-road (demo crosses a road edge <= 1 %) | 0 % (the served plan 0.27 %) | yes |
| G-speed (4 s chord vs the controller trace within 10 % on >= 99 %) | 100 %, median ratio 1.000 | yes |
| G-r1 (pool tokens / ego bit-identical) | front 256 / 256, ego yes, feat fp16 max 9.8e-4 | yes |
| G-shape (decision 293, realized track) | 67 / 115 branches pass; fails kmax 35, dk 27, alat 10, v_entry 12 (overlapping); 3 317 rows dropped | - |
| G-crawl (approach rows with min(v0, future speed before the entry) < 0.5 v_e) | 2 003 rows dropped | - |

| coverage | rows |
|:--|:--|
| phase | approach 1 191, inside 1 981, exit 1 356 |
| D | D40 2 112, D0 1 375, Did (identity) 388, D10 330, D20 199, D30 124 |
| cell | g1.25-ve 978, g1.25-raw 920, g1.25-ve08 720, g1-ve 437, g1-raw 409, g1-ve08 339, g1.5-ve08 318, g1.5-ve 229, g1.5-raw 178 |
| src | own 4 269, log 259 |
| push (served plan transformed at the decision) | on 2 827, off 1 701 |
| topo | curve_single 2 396, straight_exit 1 371, no_straight_exit 761 |
| part | train 2 470, hard_train 2 058 |
| per scene (rows) | ea468790 587, a4be61d6 514, c6c01d25 489, 8a365ac2 457, 232d240e 428, 96da4f1a 352, 37d660b1 330, 268949d2 304, ddc3e8df 281, 1e8e939e 259, dc1966f4 230, f9e361aa 199, 403d558a 60, b40f4729 38 |
| branches with rows per scene | 268949d2 9, 1e8e939e 8, 8a365ac2 7, a4be61d6 7, ea468790 7, c6c01d25 6, 232d240e 5, 96da4f1a 4, dc1966f4 4, 37d660b1 3, ddc3e8df 3, f9e361aa 2, 403d558a 1, b40f4729 1 |

- **Only 14 of the 27 turns give rows**; 13 give none: 213dfdac, 77fdd75c, 96eb4700, 98694f91, a174faed, e5e47fd2 (completing cells all fail the shape gate), and 7 with no completing cell at all.
- **Identity rollouts supply 388 rows** (9 branches, the policy alone); the non-identity branches supply 4 140. **2 014 rows (7 scenes) come from scenes whose
  identity does not complete**: 1e8e939e 259, 232d240e 428, 37d660b1 330, 403d558a 60, 8a365ac2 457, ddc3e8df 281, f9e361aa 199.
- The pool has no per-scene balance: the top scene (ea468790) has 13 % of the rows, the top 4 scenes 44 %; the training cap `--on-weight log=<scene>:w` / `rm_train.py --on-cap-scene` applies.
- Duplicates: 9 branches fall in 4 groups with an identical realized track (the same state reached at D40 and D0, or a profile that never bound), 137 rows are extra copies. G-shape does not remove them.
- The turn-level gate count of the read (13 turns with >= 1 gate-passing completing branch cell, 7 at >= 2 handovers) counts non-identity cells; the pool has 14 turns because
  b40f4729 contributes only its identity rollout (38 rows; its 7 branch cells fail the gate).

## Run notes

- Stages 1 and 10 went through `rf_p4.sh fanout` with a local `PYTHONPATH` shim for the then-missing `pool.fanout_fill` (the CLI static check; the submitter process only, nothing in the repo
  changed, no `--no-check`); stage `all` ran after the fix f987a4d5 without it. The first stage-10 jobs finished before `all` was launched; the checklist was run on 5 finished scenes
  (cells = rows, no duplicate or missing tag, reach h rising as D falls, S2 transform-on at exactly h in 16 / 16 cells of each) while the other 5 were running.
- Memory: working set 163-207 GiB at 9 jobs at once, margin 57-194 GiB; one sample at 8 GiB margin at 13:51 box time while 17 jobs started together; no SIGKILL.
- The row-builder work dir (11 GB) and the pool `on1_SELF` (144 MB) are on the box; nothing was merged or trained.
