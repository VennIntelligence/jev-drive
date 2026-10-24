# D3 on nuPlan (round 9, 2026-10-24)

**Question.** D3 = `NT4D3-F-s0` (decisions 473 / 475) is the clean PAI-trained model. In the S-NU lane, its own onpol-train roll scored 0.854,
against 0.788 for RB0-F on the same 1 031 scenes. How does it read on the nuPlan exams?

**Readout.** The readout is decision 444's, unchanged:
- arena held split, native serve (no hold, no stop3; decision 371), 18 chunks (`jt.sh aread NT4D3-F-s0 off <c>`);
- held 460 @18 (`rm_chain.sh read NT4D3-F-s0 <i> 6`).

It is paired against the stored reads of B2 (`STANDFMXH-L18-s1`, the nuPlan candidate, decision 391) and NTV0-L18-s0. Decision 444's R8K-L18-s2 /
R6DAG-L18-s1 reads are included as context; their rows reproduce [p1.md](../../../nuplan_dag/results/p1.md) exactly. Lane:
[d3nu.sh](../../scripts/d3nu.sh) (`launch`, `sum`), box `runs/round9/d3_nuplan/`. One read of one checkpoint.

## Arena held, native (2 228 windows)

| arm | ref | mean arm / ref | paired [CI] | zeros arm / ref |
|:--|:--|--:|--:|--:|
| D3 | B2 | 0.7879 / 0.9358 | -0.1479 [-0.2056, -0.0942] | 456 / 101 |
| D3 | NTV0 | 0.7879 / 0.7699 | **+0.0180 [+0.0028, +0.0361]** | 456 / 494 |
| R8K-s2 (d444) | NTV0 | 0.7626 / 0.7699 | -0.0074 [-0.0200, +0.0042] | 512 / 494 |

Frozen board-mix PCS ([pcs.txt](pcs.txt); Mac `pcs_local.py score`, 2 106 windows):

| read | PCS |
|:--|--:|
| D3 | 1116 |
| NTV0 | 1104 |
| R6DAG-s1 | 1100 |
| R8K-s2 | 1099 |
| B2 | 1398 |

The stored B2 and NTV0 PCS values reproduce the earlier ones (1398 / 1104).

Zero classes (D3 / B2 / NTV0):

| class | D3 | B2 | NTV0 |
|:--|--:|--:|--:|
| corridor | 359 | 43 | 396 |
| offroad | 59 | 53 | 61 |
| AF-front + corridor | 20 | 0 | 18 |
| AF-front | 16 | 2 | 19 |

Strata ([strata.md](strata.md)), mean and zeros for D3 against B2:

| stratum | D3 | B2 |
|:--|--:|--:|
| static (878) | 0.584, 365 zeros | 0.973, 24 zeros |
| launch (200) | 0.940 | 0.767 |
| braking | 0.911 | 0.920 |
| turn | 0.873 | 0.896 |

The whole gap to B2 is NTV0's static-window roll-out. D3 cuts it a little: corridor zeros are 359 against NTV0's 396. D3 launches better than B2.

## Held 460 @18

| arm | ref | mean arm / ref | paired [CI] | zeros arm / ref |
|:--|:--|--:|--:|--:|
| D3 | B2 | 0.8650 / 0.8527 | +0.0123 [-0.0207, +0.0465] | 43 / 41 |
| D3 | NTV0 | 0.8650 / 0.8876 | **-0.0225 [-0.0415, -0.0016]** | 43 / 32 |

Zero classes, D3 against NTV0:

| class | D3 | NTV0 |
|:--|--:|--:|
| offroad | 22 | 15 |
| AF-front | 5 | 4 |
| AF-side | 2 | 1 |
| corridor | 6 | 4 |
| route-fail | 8 | 8 |

These are the same extra offroad zeros as decision 444's PAI checkpoints (22 / 21). Partial-score loss is even with NTV0: slow 87 windows / 13.2 points
against 82 / 13.2.

## Reading

- D3 is the best PAI-trained read on nuPlan so far. On the arena it is the first one that beats NTV0 with a CI above 0 (+0.018). R8K-s2 and
  R6DAG-s1 were at -0.007 / -0.003.
- On held 460 it loses to NTV0 (-0.023, CI just below 0), through offroad zeros.
- It is not a nuPlan candidate. It sits 0.148 below B2 on the arena and 282 PCS below B2, because it does not hold still in static windows.

Files:
- [arena.md](arena.md) / [arena.json](arena.json): arena tables with the lateral-zero counts;
- [held460.md](held460.md) / [held460.json](held460.json): held 460 tables;
- [arena_windows.csv.gz](arena_windows.csv.gz): per-window scores of the five reads;
- [strata.md](strata.md), [pcs.txt](pcs.txt).

The failure statistics and the video page are in `research/d3-failures/` ([d3f.py](../../scripts/d3f.py)).
