# Turning school P3: branch ceiling from the policy's own plan (2026-10-14)

Prereg: [../../plans/2026-10-14-p3-branch-prereg.md](../../plans/2026-10-14-p3-branch-prereg.md). The setup is P2's: the ego follows the human
log to the handover decision k (d10 / d0: 10 m before / at the junction entry), then RSG-L18-s0 drives (gains 20, VCONT 1.0 LEAD 1). New in
P3: for 3.0 s from k (30 decisions) the served plan is transformed before the MPC sees it (`onp_log_driver.py` `ONP_BRANCH`, `branch_poses`).
Curvature gain g multiplies the plan's heading change at the same arc length. Speed scale s re-samples the same path at s times the arc
length. After the window the raw policy drives alone. 12 hard PAI turn scenes, 8 transforms x 2 handovers = 192 rollouts on Tokyo card 1,
code 1b52e69d, 03:56-07:15 JST; nothing dropped.

Files: `read.json` (per rollout, rf_branch.py read --win 30), `table.md` (rf_branch.py p3, includes window-end table), `plan.json` (P2's,
reused), `s1.txt` / `s2.txt` (smoke gates), `log.txt` (run), `read_w15.json` (amendment 1, 1.5 s window).

**Result: ceiling 10 / 12. That passes the prereg line (>= 8), but the evidence is weaker than the number suggests (see "How much to trust
the 10").** Identity alone gives 4 / 12, as in P2. Six more scenes complete under some push: 302c5c99, e37aa159, 3ce13eed, 94c1d722, a174faed
and f9e361aa. ce161092 and e5e47fd2 complete under no cell.

## Smoke gates

- S1: identity cells (d0@g1s1, d10@g1s1) against P2's stored d0 / d10. All 24 scenes have every served plan bit-identical, with identical
  tracks and scores (`s1.txt`).
- S2: f9e361aa d0@g1.5s1. The transform is on at k 69..98 (exactly 30 decisions), and the raw plan is served again at k 99. Plans before
  k 69 equal the identity run's. Raw vs served poses at k 69 (x, y, yaw): pose 8 is (16.32, -14.65, -1.35) raw and (8.63, -17.00, -2.02)
  served (`s2.txt`).

## Outcome per scene x handover x transform

`done` = turn completed (P2's rule). Otherwise the zero type: corr = corridor, offr = offroad, coll = collision. rear = rear-ended by the
log's car behind before the target; end = rollout ended or not on the route at the target. Transform columns are g<gain>s<scale>; g1s1 is
identity.

### handover d10

| scene | side | g1s1 | g1.25s1 | g1.5s1 | g2s1 | g1s0.8 | g1s0.6 | g1.5s0.8 | g1.5s0.6 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 302c5c99 | L | corr | corr | corr | corr | corr | rear | corr | rear |
| 382a7247 | L | **done** | corr | corr | corr | offr | coll | corr | coll |
| ce161092 | L | offr | end | corr | corr | rear | rear | rear | rear |
| e37aa159 | L | corr | **done** | corr | corr | rear | rear | rear | rear |
| e5e47fd2 | L | corr | corr | corr | offr | rear | rear | rear | rear |
| fd01e9ab | L | corr | coll | corr | offr | corr | corr | corr | corr |
| 2f227f2c | R | **done** | **done** | offr | offr | **done** | **done** | **done** | **done** |
| 3ce13eed | R | corr | corr | corr | coll | corr | offr | corr | offr |
| 94c1d722 | R | offr | offr | offr | offr | offr | offr | end | offr |
| a174faed | R | corr | corr | **done** | offr | rear | rear | rear | rear |
| eaba3f6a | R | **done** | offr | offr | offr | **done** | **done** | **done** | **done** |
| f9e361aa | R | corr | offr | offr | corr | **done** | rear | coll | rear |

- d10@g1s1: completed 3 / 12, zeros 10, mean score 0.167
- d10@g1.25s1: completed 2 / 12, zeros 9, mean score 0.250
- d10@g1.5s1: completed 1 / 12, zeros 11, mean score 0.083
- d10@g2s1: completed 0 / 12, zeros 12, mean score 0.000
- d10@g1s0.8: completed 3 / 12, zeros 5, mean score 0.371
- d10@g1s0.6: completed 2 / 12, zeros 4, mean score 0.321
- d10@g1.5s0.8: completed 2 / 12, zeros 5, mean score 0.332
- d10@g1.5s0.6: completed 2 / 12, zeros 4, mean score 0.321

### handover d0

| scene | side | g1s1 | g1.25s1 | g1.5s1 | g2s1 | g1s0.8 | g1s0.6 | g1.5s0.8 | g1.5s0.6 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 302c5c99 | L | corr | corr | **done** | offr | corr | rear | **done** | offr |
| 382a7247 | L | **done** | corr | corr | corr | **done** | offr | corr | corr |
| ce161092 | L | offr | coll | coll | coll | rear | rear | coll | rear |
| e37aa159 | L | corr | **done** | corr | corr | end | rear | corr | rear |
| e5e47fd2 | L | offr | offr | offr | offr | offr | rear | offr | offr |
| fd01e9ab | L | **done** | corr | offr | offr | **done** | rear | offr | **done** |
| 2f227f2c | R | **done** | **done** | **done** | offr | corr | end | offr | **done** |
| 3ce13eed | R | coll | **done** | offr | offr | end | end | offr | offr |
| 94c1d722 | R | offr | **done** | offr | offr | offr | offr | offr | offr |
| a174faed | R | corr | **done** | offr | offr | corr | rear | offr | offr |
| eaba3f6a | R | **done** | offr | offr | offr | coll | **done** | offr | offr |
| f9e361aa | R | corr | offr | offr | offr | corr | coll | offr | offr |

- d0@g1s1: completed 4 / 12, zeros 8, mean score 0.273
- d0@g1.25s1: completed 5 / 12, zeros 7, mean score 0.395
- d0@g1.5s1: completed 2 / 12, zeros 10, mean score 0.167
- d0@g2s1: completed 0 / 12, zeros 12, mean score 0.000
- d0@g1s0.8: completed 2 / 12, zeros 8, mean score 0.206
- d0@g1s0.6: completed 1 / 12, zeros 3, mean score 0.379
- d0@g1.5s0.8: completed 1 / 12, zeros 11, mean score 0.083
- d0@g1.5s0.6: completed 2 / 12, zeros 8, mean score 0.182

ceiling (>= 1 completing cell over 2 handovers x 8 transforms): 10 / 12; identity only: 4 / 12

| scene | simplest completing cell(s) | needed |
|:--|:--|:--|
| 382a7247, fd01e9ab, 2f227f2c, eaba3f6a | identity | nothing (P2's 4) |
| e37aa159 | d10@g1.25s1, d0@g1.25s1 | curvature |
| a174faed | d0@g1.25s1, d10@g1.5s1 | curvature |
| 3ce13eed | d0@g1.25s1 | curvature |
| 94c1d722 | d0@g1.25s1 (score 0.73) | curvature |
| 302c5c99 (183 deg U-turn) | d0@g1.5s1, d0@g1.5s0.8 | curvature |
| f9e361aa | d10@g1s0.8 | speed |
| ce161092, e5e47fd2 | none | - |

## Secondary

- **Mostly curvature, and mild.** Five of the six new scenes complete with a curvature push alone (g 1.25 or 1.5). The sixth (f9e361aa) only
  completes with the speed cut. g 2.0 completes nothing (0 / 24, all zeros, mostly offroad). Among single cells, d0@g1.25s1 completes the
  most scenes (5 / 12), but not the same ones as identity. It adds 3ce13eed, 94c1d722, a174faed and e37aa159 (keeping 2f227f2c), and it
  loses 382a7247, fd01e9ab and eaba3f6a, which identity completes. No single transform works across the board: the right push is
  scene-specific.
- **The speed cut is effectively a stop.** At the window end the fed speed is 0.1-1.8 m/s in every s < 1 cell (`table.md`, w_v), against
  5-9 m/s with s = 1. The 0.8 cut seems to compound. The car slows, the fed speed history (decision 285's ratchet) makes the next plan slower
  still, and the cut applies again. Most s < 1 rollouts then end "rear" (the log's car runs into the stopped ego: 5-7 of 12 per d10 cell).
  Speed completions mostly score low (382a7247 d0@g1s0.8 0.00; 2f227f2c d10 s-cells 0.34-0.41). They are completions by the rule, not good
  drives.
- **The policy finishes the turn after the push.** In all 25 non-identity completing cells the target is reached after the window has
  ended: 0.6-10.3 s later, median 6.4 s (the speed cells account for the long delays). So the raw policy drives the rest of the turn itself.
  At the window end the completing curvature cells are 0.2-4.8 m from the route (per-cell means 1.5-4.3 m). The failing ones in the same
  cells average 2.5-6.3 m. Speed differs little between the two groups.
- **1.5 s window (Amendment 1).** The 9 new-completion cells were re-run with the push cut to 1.5 s. 2 of 9 stay done: a174faed
  d0@g1.25s1 and 302c5c99 d0@g1.5s1. e37aa159 d0@g1.25s1 drives past the target with score 1.0 but is > 2.5 m off the route there, so by the
  rule it is not done. The other 6 fail: corridor 3, collision 2, offroad 1. Half the push is not enough. The policy holds the turn only when
  pushed for most of the curve.

## How much to trust the 10

- The ceiling takes the best of 16 cells per scene (14 of them new). One seed, one checkpoint, deterministic simulator (S1 is bit-exact), so
  each cell is one draw of a chaotic closed loop. Three of the six new scenes complete in exactly one cell (3ce13eed, 94c1d722, f9e361aa), and
  their neighbours fail: d10 vs d0, g1.25 vs g1.5. Requiring >= 2 completing cells per scene gives 4 + 3 = 7 / 12, which is in the
  inconclusive band. The 1.5 s window keeps 2 of 9 cells.
- The pass line is met, but "the policy's own plan, mildly bent, contains the turn" holds robustly for about 3 of the 8 P2 failures (e37aa159,
  a174faed, 302c5c99). For the rest it holds only as single lucky cells. Two scenes (ce161092: offroad / rear, e5e47fd2: offroad / corridor,
  30-40 m short) have no completing cell.
- The handover is still from the human log at the entry (route C). A self-demonstration line would need the push to be found from the
  policy's own states and be scene-specific (no fixed g works). That is a search over branches per scene, not a fixed transform.
- 12 scenes, one seed: the ceiling is a gate, not a rate.

Tokyo: `/data/runs/route_night/p3/` (arms/, s2/, w15/, read.json, table.md, log.txt).
