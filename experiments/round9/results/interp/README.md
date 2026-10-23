# INTERP: NTV0-L18-s0 <-> R8K-L18-s2 weight interpolation (2026-10-24)

Plan (Chinese): [../../plans/2026-10-24-interp.md](../../plans/2026-10-24-interp.md). Code: [../../scripts/interp.py](../../scripts/interp.py)
(merge / check / report), [../../scripts/interp.sh](../../scripts/interp.sh) (lane). Box: `$DATA_DIR/runs/round9/interp/`, checkpoints
`$DATA_DIR/runs/op_parity/runs/INTP50-NR`, `INTP75-NR`. Decision 463. This is a descriptive baseline (BCWI-style, research/method-search/notes.md
A-1) for frontier anchoring. It is not a candidate.

**What was done.** theta = (1 - a) NTV0-L18-s0 + a R8K-L18-s2 for a = 0.5 / 0.75, over every tensor in `ckpt-final.pt`: `model.net` (109 trained
policy tensors, 16.2 M) and `model.parity` (adapter + route-memory branch, 97 tensors, 3.16 M). Both sources have the same key sets, shapes and fp32
dtypes, and every tensor differs between them. R8K has no extra module: its twin token is a training-data swap, not a parameter. The frozen encoder
is not in the checkpoints: 468 tensors (365 M) are loaded from the shipped Cinque and are bit-equal across every loaded model. `adapter.pt` holds the
merged parity. Non-tensors and `route_mem.json` are R8K's (head config equal to NTV0's), plus an `interp` record. Check
([check.json](check.json), 18 / 18 checks pass): the serve's `sh30_core.Core` loads every point. On 4 seeded random forwards, a = 0 equals NTV0
and a = 1 equals R8K bit for bit; an NTV0 reload equals NTV0. Reads: PAI 281 ACC + GAP, Harmonizer OFF, with R8K-L18-s2's stored serve
(identical to NTV0's: no twin or JEV_CMD env) and its chunk lists.

**Result** (277 common scenes, no RPC errors; new scorer, old in brackets; full report [interp.md](interp.md), numbers [interp.json](interp.json)):

| read | mean | vs NTV0-s0 | vs R8K-s2 | zeros | neg. flips on NTV0's full scenes | R8K's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| NTV0-s0 | 0.814 (0.745) | | | 39 (58) | (201 full) | |
| R8K-s2 | 0.872 (0.807) | +0.058 [+0.016, +0.099] | | 28 (46) | 14 (15) | 40 / 40 |
| INTP50-NR | 0.836 (0.778) | +0.022 [-0.010, +0.053] | -0.036 [-0.074, +0.002] | 36 (52) | 9 (6) | 17 / 40 |
| INTP75-NR | 0.879 (0.814) | +0.065 [+0.028, +0.102] | +0.007 [-0.017, +0.032] | 26 (44) | 8 (8) | 33 / 40 |

- **INTP75 keeps R8K's gain and cuts its negative flips from 14 to 8.** Of those 8, 6 are R8K's own and 2 are new; 6.7 points are lost there
  against R8K's 10.5. It keeps 33 of R8K's 40 fixes. Its mean equals R8K's (the CI includes 0). By old-scorer zero class it has fewer offroad
  zeros (10 vs 14) and the same route-turn zeros (3 vs 2).
- **INTP50 drops too much of the gain.** It keeps only 17 of 40 fixes and sits between the two sources. Its offroad zeros rise to 18, above both
  ends (12 / 14).
- **Caveat.** There is one read per point and one R8K seed. Closed loop is not bitwise deterministic, so some flips are re-read noise, and this
  lane has no identical-checkpoint baseline. With that, "14 -> 8" is suggestive, not established. Frontier anchoring (A-2) now has to beat
  INTP75: about 8 negative flips at R8K's mean, with no training.

**Cost.** 6.1 GPU job-hours, almost all in the 16 read chunks (each books 22 GB; 3-4 share a card). The merge, check, scoring and report are minutes.
