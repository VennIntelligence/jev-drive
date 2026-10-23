# S-SPLIT: do the S rows' hinges or their distill_ho carry NTV0's PAI gain? (round 9, 2026-10-24)

Pre-registration (Chinese): [../../plans/2026-10-24-s-split.md](../../plans/2026-10-24-s-split.md). Lane `scripts/ssp.sh`, helpers `scripts/ssp.py`
(reuses `scripts/pcl.py`). Box dir `$DATA_DIR/runs/round9/s_split/` (reads, scorer_v2 rescoring). Decision 464. Seed 0 only, as registered.

**Answer. The hinges carry the gain; distill_ho is not needed.** Both scorers land on the same line.
- **NH** (S rows still drawn, all their hinges off, distill_ho kept) falls to the no-S level: 0.716 on the new scorer, which keeps
  19 % of the NTV0NS -> P0 gap. That is -0.087 [-0.129, -0.046] vs P0 and +0.020 [-0.007, +0.050] vs NTV0NS.
- **ND** (distill_ho off on the S rows, hinges kept) stays at P0: 0.819 on the new scorer, +0.017 [-0.015, +0.049] vs P0.

So this gain supports the paper claim "body schema = consequence hinge". A clean NTV1 can drop distill_ho on the S rows.

## What the S rows carry (checked from code and P0's stored argv)

P0 = PCL0-L18-s0 = NTV0-L18-s0's argv with `--on-wfile JT00S-rb0`. The argv has no `--ho`, so Losses43's hinge-only rows are exactly the 32
pool `self` rows of each batch. On those rows the loss adds:
- `agent_ho`, scaled by ho_w x agent_lam (`--agent-on-only`: no offline row has an agent label);
- `road_ho`, scaled by ho_w x road_lam;
- the nr1_lat terms `on/lat_{onc,soft,side}_ho`, scaled by ho_w (through `--on-lon-all`);
- `distill_ho` (non-plan heads to the shipped teacher), scaled by lam_d 30. This one is not scaled by ho_w.

The S rows carry no imitation term and no navtrain drivable hinge.

## Switches

| arm | tag | token diff vs P0's stored argv | removed |
|:--|:--|:--|:--|
| NH | SSNH-L18-s0 | `--ho-w 3 -> 0` | every S-row hinge (agent, road, lateral). Existing switch: ho_w multiplies exactly these terms |
| ND | SSND-L18-s0 | `+ --ho-distill 0` | distill_ho on the S rows |

`--ho-distill W` is new in `on_train.py` (commit 815ffff3, default 1 = off, no operation added). It adds (W - 1) x lam_d x distill_ho, so at
W = 0 the term's gradient is exactly zero; the value is still logged. rm_train inherits the flag through `recipe_args`; rm_train.py itself
was not changed.

## Gates (all passed)

| gate | result |
|:--|:--|
| flag off: P0's argv on HEAD 7f5ffbed vs the same argv on the pre-flag commit 4cf07256 (git worktree), steps <= 50 | 149 / 149 scalars bit-identical ([g0_flagoff.json](g0_flagoff.json)). HEAD vs the stored P0 run: also 149 / 149 ([g0_head_vs_stored.json](g0_head_vs_stored.json)), so the known drift of decision 456 does not affect P0 |
| 60-step smokes with `--diag-every 20` | both pass: batch 83 / 13 / 32 mixed. NH: grad/pool, grad/pool_hinge and grad/lat are exactly 0 at every diag step. ND: grad/pool is 2.8-5.3. distill_ho and the hinge values are logged in both ([gcheck_NH.json](gcheck_NH.json), [gcheck_ND.json](gcheck_ND.json)) |
| final navtrain dev ADE <= 0.566 | NH 0.5326, ND 0.5352 (P0 0.5366) |

## PAI 281, ACC + GAP, Harmonizer OFF (P0's serve and chunk lists)

278 scenes are common to six reads, scored by both scorers with no driver RPC error. Full tables: [verdict.md](verdict.md),
[pai281.md](pai281.md).

| read | new | old | zeros new / old | offroad | drift + side | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|--:|
| NH SSNH-L18-s0 | 0.716 | 0.651 | 67 / 85 | 23 | 11 + 2 | 21 | 8 |
| ND SSND-L18-s0 | 0.819 | 0.747 | 37 / 57 | 13 | 7 + 1 | 10 | 4 |
| P0 PCL0-L18-s0 | 0.803 | 0.734 | 42 / 61 | 14 | 7 + 0 | 16 | 2 |
| NTV0NS-L18-s0 | 0.696 | 0.624 | 73 / 93 | 21 | 13 + 2 | 20 | 14 |
| NTV0-L18-s0 | 0.814 | 0.743 | 39 / 59 | 12 | 6 + 0 | 15 | 3 |

Registered lines, recomputed on these scenes:
- **new scorer:** P0 0.803, NTV0NS 0.696. An arm >= 0.783 means the removed part is not needed; an arm <= 0.726 means it carries the gain.
  NH is at **carries the gain**, ND at **not needed**.
- **old scorer:** P0 0.734, NTV0NS 0.624, lines 0.714 / 0.654. Same result for both arms.

Paired differences, new scorer (old in brackets):

| pair | new | old |
|:--|--:|--:|
| NH - P0 | -0.087 [-0.129, -0.046] | -0.083 [-0.126, -0.042] |
| NH - NTV0NS | +0.020 [-0.007, +0.050] | +0.027 [-0.002, +0.058] |
| ND - P0 | +0.017 [-0.015, +0.049] | +0.013 [-0.019, +0.047] |
| ND - NTV0NS | +0.123 [+0.086, +0.164] | +0.123 [+0.086, +0.164] |
| NH - ND | -0.103 [-0.146, -0.064] | -0.096 [-0.140, -0.055] |

Zero classes, count difference with 95 % CI (old-scorer classes):

| arm - ref | zeros | offroad | AF-front | route-turn | zero-other (drift) |
|:--|--:|--:|--:|--:|--:|
| NH - P0 | +24 [+12, +36] | +9 [+1, +17] | +6 [+1, +12] | +5 [-2, +13] | +4 [-1, +9] |
| NH - NTV0NS | -8 [-17, +1] | +2 [-3, +8] | -6 [-11, -2] | +1 [-4, +6] | -2 [-7, +3] |
| ND - P0 | -4 [-14, +5] | -1 [-7, +5] | +2 [-2, +6] | -6 [-12, -1] | +0 [-3, +3] |
| ND - NTV0NS | -36 [-48, -25] | -8 [-15, -1] | -10 [-17, -4] | -10 [-17, -4] | -6 [-12, -1] |

What to look at in this table:
- Without the hinges, the offroad and AF-front zeros come back.
- NH keeps a little of the AF-front gain over NTV0NS (-6). This is the only sign that the S rows do something without their hinges.
- ND loses nothing against P0 and has fewer route-turn zeros than P0.

The training logs agree (means over the last 1 000 steps):
- Without the hinges, NH's own S-row hinge values rise: lat_onc 0.019 vs 0.007, road 0.009 vs 0.006.
- ND's distill_ho roughly doubles (0.0016 vs 0.0009), and that does not hurt the read.

## Cost

7.2 GPU job-hours: training 1.08, reads 6.05, gates and smokes 0.09. Wall time 1.2 h.
