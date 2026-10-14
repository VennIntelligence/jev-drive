# P4 training round 1: RSGSELF-L18 on self-branch rows (2026-10-14)

Prereg: [Amendments 3-4](../../../plans/2026-10-14-p4-self-branch-prereg.md) of the P4 self-branch prereg. The round ran by the user's
decision even though the coverage verdict was inconclusive (10 / 27). It was fixed before any merge, training or read.

**Verdict: refuted by the registered rule.**

| line | verdict |
|:--|:--|
| L1a | **fail** |
| L1b | inconclusive |
| L2 | inconclusive, leaning worse |
| L3 | **fail** |
| L4 | pass |

The arm completes no more of its own training turns than the control. On the public set it pays about +11 zeros, mostly offroad: the same
pattern as RBSG2 and RSGPAIT.

## Setup

- **Rows (pool `on1_SELF4R`, [pool/](pool/))**:
  - Selection: completing P4 cells that pass G-shape, gain 1.25, handover D40 / D0, **profile raw only**. The longitudinal part is taken
    out; the trainer has no lateral-only mask (Amendment 3).
  - Size: **753 rows, 12 branches, 8 turns** (Amendment 4: 10 rows dropped for G-route).
  - Sources and handovers: src own 731 / log 22; D40 476 / D0 277.
  - Phases: approach 157, inside 370, exit 226.
  - Distance: no row farther than 20 m before the entry, because G-crawl removed them all. **The round tests in-junction and exit
    demonstrations only.**
  - Gates: G-in, G-route (100 % after the filter, max 0.049 m), G-rmem, G-road, G-speed, G-r1 all pass.
  - Gate history: the first build failed G-route by the letter (98.69 %; 10 rows of 8a365ac2 D0 exit at 0.050-0.056 m). The round stopped,
    the user chose the row filter over a waiver, and the pool was re-packed with `--drop-route0`.
- **Arms** (box, `rf_self_train.sh`, `O=$DATA_DIR/runs/route_follow/selftrain/round1`):
  - **RSGSELF-L18-s0 / s1**: RSG-L18's recipe plus on1_SELF4R as `both` rows in `--on-mode mixed`.
    - W 9.92 = 30.0 % of the pool draws. CAP 0.20; it binds on 232d240e (0.208) and ea468790 (0.223).
    - Mixcheck pass: self 0.0, both 0.0 ([nuplan/draws_RSGSELF.json](nuplan/draws_RSGSELF.json)).
  - **RSGCTL-L18-s0**: RSG-L18's recipe retrained now. **Bit-identical to the stored RSG-L18-s0**: 109 net and 97 adapter tensors, max diff 0
    ([nuplan/ident_ctl.json](nuplan/ident_ctl.json)). So the control's nuPlan reads are the stored RSG-L18 reads (Amendment 3).
- **Training gates** (staged: s0 was checked before anything else was launched):

| arm | final dev ADE (line <= 0.538) | junction reliance, masked - on (line >= +0.0125, CI lower > 0) | loss |
|:--|:--|:--|:--|
| RSGSELF-L18-s0 | 0.5299 | +0.0225 [+0.0092, +0.0370] | finite; SELF rows' on/ade 1.11 -> 0.79 m |
| RSGSELF-L18-s1 | 0.5289 | +0.0255 [CI lower +0.0095] | finite |
| RSGCTL-L18-s0 | 0.5284 (= RSG-L18-s0) | +0.0250 | = RSG-L18-s0 |

## Lines

**L1a: identity completion on the 27 P4 train turns** (PAI, P4 instrument phase A only, `rf_p4.sh fanout` with `TAG=<ckpt> P4_ONLY=none`;
P2's rule). Files: [l1a.md](l1a.md), per-scene reads in [pai/](pai/).
- The control reproduces the stored RSG-L18-s0 phase A of the P4 run **scene for scene** (9 / 27, the same zero kinds, mean 0.3467): the
  instrument did not drift.
- Line: pass if both seeds >= control + 3; fail if the seed mean <= control. Seed mean 8.5 <= 9: **fail**.

| label | completed / 27 | on the 8 row turns | on the other 19 | zeros | mean score |
|:--|--:|--:|--:|--:|--:|
| RSG-L18-s0 (P4 run, stored) | 9 | 4 | 5 | 17 | 0.3467 |
| RSGCTL-L18-s0 (control) | 9 | 4 | 5 | 17 | 0.3467 |
| RSGSELF-L18-s0 | 8 | 3 | 5 | 18 | 0.3116 |
| RSGSELF-L18-s1 | 9 | 4 | 5 | 18 | 0.2997 |

Per-scene changes against the control (both seeds unless noted):
- Row turns: **gained** 232d240e, ddc3e8df (s1 only); **lost** 268949d2 (now a collision) and ea468790 (now a corridor zero).
  ea468790 is the row turn with the most rows (168, capped at 0.20).
- Other turns: **gained** 37d660b1, 98694f91 (s1 only); **lost** 96da4f1a (offroad), dc1966f4 (s1 only, offroad), and a4d2f7cf (s1 only,
  non-zero -> corridor zero).

The turns move in both directions, and the count does not.

**L1b: held 460 nuPlan turns (24, rf_follow)** ([nuplan/l1b.md](nuplan/l1b.md), [nuplan/follow.md](nuplan/follow.md)).
- Seed mean 0.583 vs RSG-L18 0.625: **-4.2 pp [-23.1, +10.7]**. **Inconclusive.**
- Right turns: s0 37.5 % / s1 25.0 % vs 62.5 / 50.0 (8 turns).
- Junction lateral median: 0.70 / 0.71 m vs 0.52 / 0.54. The p90 is 2.62 / 2.71 vs 1.90 / 2.07.

**L2: held 460 zeros and mean.** The tolerance is 0 zeros and 0.002 mean: the stored seed spread, and the control is bit-identical.

| held 460 (held.py, all 460) | s0 | s1 | seed mean |
|:--|--:|--:|--:|
| RSG-L18 mean / zeros | 0.8656 / 34 | 0.8676 / 34 | 0.8666 / 34 |
| RSGSELF-L18 mean / zeros | 0.8606 / 37 | 0.8587 / 38 | 0.8597 / 37.5 |

Paired on the 452 common scenes (ol_confirm, [nuplan/held18_confirm.md](nuplan/held18_confirm.md)), RSGSELF-L18 - RSG-L18:
- mean -0.0070 [-0.0166, +0.0032];
- zeros +3.5 [-1.2, +7.9];
- at-fault zeros +4.0 [-0.0, +7.8].

Worse beyond the tolerance with CIs that include 0: **inconclusive** by the registered rule (leaning worse).

**L3: public 1491 guard** (1 479 common scenes, [nuplan/guard_confirm.md](nuplan/guard_confirm.md)). **Fail** on every part.

| | RSG-L18 (seed mean) | RSGSELF-L18 (seed mean) | line | |
|:--|--:|--:|:--|:--|
| (a) at-fault zeros vs YR10m10-F-s0 | -11.5 [-22.1, -1.4] | **-1.0** [-13.1, +11.1] | <= -1.4 | fail |
| (a) mean score vs YR10m10-F-s0 | +0.0158 [+0.0085, +0.0235] | **+0.0070** [-0.0016, +0.0156] | >= +0.0085 | fail |
| (b) at-fault zeros | 38.5 | **49** | <= 43.5 | fail |
| (b) zeros | 45 | **56** | <= 50 | fail |

- Paired RSGSELF-L18 - RSG-L18: zeros **+11.0 [+2.8, +20.3]**, at-fault zeros +10.5 [+2.5, +19.5], mean -0.0088 [-0.0151, -0.0031].
  Both seeds give +11 zeros.
- By kind (collision / offroad / corridor): 9.5 / **39.5** / 7 vs 8 / 30.5 / 7.5. The extra zeros are offroad.
- At-fault distance: 0.80 vs 1.03 km per incident.

**L4: GT lateral** (line: not worse by > 0.05 m). **Pass.**
- Held 460 `latgt` seed mean: 0.859 vs 0.873 m (-0.014).
- Public GT distance: +0.001 [-0.019, +0.021] m.
- The registered anti-lateral-ratchet guard ([nuplan/lat.md](nuplan/lat.md)) also passes: mean |dlat2| +0.0055 [-0.0060, +0.0170] m
  (line +0.02).

## Reading

- Refutation clauses met:
  - L1a fails: the checkpoints trained on these rows do not complete more of the turns the rows came from.
  - L3 fails: the rows cost public nuPlan driving, offroad first.
- What the rows carry: 753 rows from the policy's own states, inside the junction and in the exit, at the plan's own speed with a 1.25x
  lateral push. What came out is a wider junction line: junction lateral to the route 0.70 vs 0.52 m median, fewer completed right turns.
  The offroad cost is the RBSG2 / RSGPAIT pattern, at 30 % of the draws.
- The in-sample result is a reshuffle, not a gain. 2 row turns are gained and 2 lost (s0). The largest row turn (ea468790) is lost in both
  seeds.
- The round says nothing about the approach (slowing in before the entry). G-crawl left no rows there; that gap stays open.

## Run

- Box, all GPU work through the pool, chain `rf_self_train.sh` (`stage1`, `rest`, `report-submit`).
- PAI reads: `rf_p4.sh fanout <root> all` with `TAG`, `P4_ONLY=none`, `P4_VRAM=10`. The queued PAI jobs were retargeted to the measured
  8.6 GB RSS / 6.1 GB VRAM (12 / 8 declared).
- About 125 pool jobs, **15.2 GPU job-hours, about 2.7 card-hours** (booked VRAM share). First job to last: 1.74 h wall (17:20-19:05 JST).
  The box was shared with another agent's P4 camera re-runs (p4v), which held most of the booked host memory for the first ~40 min.
- Served code: s0's PAI read at 528a75a4; s1's and the control's at 3cd4b22b. The driver was the same: the control reproduces the stored
  P4 phase A.
- The report job ran without the `O` env (the pool does not pass it). Its outputs were written to `selftrain/` and moved into
  `selftrain/round1/`.
- Box paths:
  - checkpoints `$DATA_DIR/runs/op_parity/runs/{RSGSELF-L18-s0,RSGSELF-L18-s1,RSGCTL-L18-s0}`;
  - merged pool `on1_RSGSELF`;
  - held reads `runs/route_mem/read/RSGSELF-L18-s*/`, public `runs/route_follow/guard/read10/RSGSELF-L18-s*/`;
  - PAI `runs/route_follow/p4/round1/pai_*`;
  - report `runs/route_follow/selftrain/round1/`.
- Line scripts: [rf_round1.py](../../../scripts/rf_round1.py) (`l1a`, `l1b`).

## Files

- [pool/](pool/): gates, coverage, branches, failing.
- [l1a.md](l1a.md), [l1a.json](l1a.json): L1a table and per-scene matrix.
- [pai/](pai/): PAI identity reads and tables per checkpoint.
- [nuplan/](nuplan/): `summary.md` (ladder table), `held_raw.txt`, `held18_confirm.md`, `guard_confirm.md`, `follow.md`, `l1b.md`, `lat.md`,
  `ident_ctl.json`, W / draws, job ids.
- Figure `../../../figs/self_bev_SELF4R.png` (6 rows of the filtered 753-row pool, re-drawn by the `--drop-route0` pack).
  - Red is the realized raw branch, green is the policy's own plan.
  - Look at the spacing: red and green are close, so there is no imposed slowing.
  - The exception is 232d240e, whose plan is short (the ratchet) while the car kept its speed.
