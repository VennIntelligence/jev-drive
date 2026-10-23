# DIAG-READS: P1L with the Harmonizer ON, INTP75-NR re-read (2026-10-24)

Plan (Chinese): [../../plans/2026-10-24-diag-reads.md](../../plans/2026-10-24-diag-reads.md). Lane: [../../scripts/diag.sh](../../scripts/diag.sh), report
[../../scripts/diag.py](../../scripts/diag.py). Full tables: [diag.md](diag.md), numbers [diag.json](diag.json). Box: `$DATA_DIR/runs/round9/diag_reads/`.
Descriptive; no registered line. PAI 281 ACC + GAP, new scorer primary (old in brackets).

**A. Harmonizer ON** (279 common scenes, no RPC errors; renderer logs show the weights loaded in 8 / 8 chunks). P1L's serve and chunk lists are
the stored OFF read's, plus `HARMONIZER=1`.

| read | mean | zeros | route-turn | offroad |
|:--|--:|--:|--:|--:|
| P1L OFF / ON | 0.813 (0.735) / 0.804 (0.733) | 39 (61) / 41 (61) | 10 / 6 | 14 / 18 |
| NTV0-s0 OFF / ON | 0.815 / 0.809 | 39 / 40 | 15 / 9 | 12 / 17 |
| R8K-s2 OFF / ON | 0.873 / 0.866 | 28 / 27 | 2 / 2 | 14 / 15 |

ON - OFF: P1L -0.009 [-0.040, +0.023], NTV0-s0 -0.006 [-0.040, +0.028], R8K-s2 -0.007 [-0.037, +0.023]. P1L ON vs NTV0-s0 ON -0.005
[-0.043, +0.033]; vs R8K-s2 ON -0.062 [-0.103, -0.021]. The three models move alike under the Harmonizer: about -0.007, route-turn zeros down
(P1L 10 to 6, NTV0 15 to 9), offroad zeros up (P1L 14 to 18, NTV0 12 to 17); R8K-s2 shows almost no zero-class shift.

**B. INTP75-NR re-read** (same checkpoint, serve and chunk lists, Harmonizer OFF; 278 common scenes): all 278 scenes score identically to the first
read (mean |diff| 0.000, both scorers). Negative flips on NTV0-s0's 202 full-score scenes: 8 in both reads (the same 8 scenes), R8K-s2 14.
So with identical chunk lists the closed loop reproduced bit for bit here: the first-read count of 8 carries no re-read noise, and this re-read does
not give a nonzero same-checkpoint floor. (Noise across different chunk lists or seeds is not measured.)

**Cost.** 13.10 GPU job-hours, all in the 16 read chunks.
