# PAI2: PAI closed loop on the GPU box (2026-10-10)

Question: can the PAI-track evaluation run on the GPU box through the pool, and does it agree with the containerised stack on Tokyo?
Answer: yes. The renderer image is unpacked (no Docker, no root) and started by `run_native.py`; on the 40 scenes Tokyo has scored, P2H10-F-s0
agrees on 37 of 40 scene classes and on 39 of 40 scores to 0.01. Setup and the pool command: [docs/alpasim.md](../../../docs/alpasim.md), "PAI on the GPU box".

## Parity with Tokyo (P2H10-F-s0, `pai_driver.py` as shipped, unharmonised renderer, CONC 4, same scene lists as Tokyo's runs)

| Scenes | Tokyo mean | Box mean | Same class | Same score (+-0.01) | Class changes |
|---|--:|--:|--:|--:|---|
| first 10 (`a10`, table_10.md) | 0.1538 | 0.1539 | 10/10 | 10/10 | none; box run twice: both identical to Tokyo (8 zeros: 4 collision, 2 offroad, 2 left corridor) |
| `b1` | 0.2117 | 0.1364 | 9/10 | 9/10 | b988494a 0.752 -> 0 (left corridor) |
| `b2a` | 0.0160 | 0.0160 | 10/10 | 10/10 | none |
| `b2b` | 0.2960 | 0.2956 | 8/10 | 10/10 | 145bf9cc 1.000 -> 0.995 (a pass that is 0.5 % short, score equal to 0.01); 8f3902f9 stays a zero but changes kind (collision_at_fault -> left_corridor_laterally) |
| **all 40** | **0.1694** | **0.1505** | **37/40** | **39/40** | 3 scenes: b988494a (real flip), 8f3902f9 (zero stays zero, kind changes), 145bf9cc (1.000 -> 0.995) |

Per-scene table: [pai/parity_40.md](pai/parity_40.md). The two box runs of the first 10 scenes (same code, same box) are identical
([pai/repeat_a10.md](pai/repeat_a10.md)); box-vs-Tokyo differences are the cm-level numerics of decision 215, amplified by closed loop in one scene
with a marginal score (0.752 on Tokyo). Zeros: Tokyo 30, box 31 of 40. The mean over the 40 differs by 0.019, which is the flipped scene (0.752 / 40).

## Baseline on the box (P2H10-F-s0 and P2H10-F-s1, current driver, no serving switches)

Chunks (one scene list = one launch, reuse them to pair): `a10`, `b1`, `b2a`, `b2b` (Tokyo's lists of the 40), `e1`, `e2`
([pai/chunks/](pai/chunks/), run dirs `$DATA_DIR/runs/alpasim/pai2/base/<chunk>_<s0|s1>`, summaries in [pai/box/](pai/box/)).

40 scenes (`pai_scenes_40.tsv`), with Tokyo's s0 for comparison and the organisers' reference rows on the same scenes:



Extension: 20 further scenes of `pai_scenes_ext120.tsv` (stage 1 order, `e1`, `e2`; same rule as the 40, over the other 401 curated-val scenes):



All 60 scenes, per seed and pooled:



Per-scene tables: [pai/baseline_40.md](pai/baseline_40.md), [pai/baseline_ext20.md](pai/baseline_ext20.md), [pai/baseline_60.md](pai/baseline_60.md).
Seed difference on the 40: s1 0.2600 against s0 0.1505 on the same scenes, so one seed's PAI mean moves by about 0.1 on 40 scenes;
compare arms on the same chunks and seeds only.

## Cost per stack (CONC 4, 10 scenes per launch, pool jobs declared 24 GB VRAM, 6 cores, 40 GB RAM)

| | Value |
|---|---|
| VRAM | renderer 16.2-21.3 GiB, driver 1.9, physics 1.3-1.8: 20-25 GiB per stack (declare 24 GB; 3 stacks fit a 84 GB card) |
| Host RAM (sum of process RSS peaks) | 23-28 GiB per stack (runtime 11, renderer 12, driver 2.5, physics 1, controller 0.8) |
| Cores | 2.5-2.8 on average over the run (2 010 core-seconds in 737 s); the box was at load 30-50 from other lanes |
| Wall | 64-78 s per scene at CONC 4 (Tokyo 3090: 57-62 s), 10 scenes in 643-801 s of runtime + about 25 s start-up; one scene alone 219 s |
| Output | about 0.45 GB per scene |

Per-run table: [pai/stats.md](pai/stats.md). Up to 6 stacks were requested for this lane; about 4 ran at once (two chunks x two seeds), no worker was killed.

## Issues

- Scenes beyond the 60: stopped by the free-disk rule (400 GB free) at 115 GB of scenes, with 67 of 160 scenes fetched; `e3`..`e6` are listed but
  not fetched. The 150 GB cap was not reached.
- `run_native.py` changed: `--sub OLD=NEW` (command rewrites), `--rewrite-configs`, and a scored zero (`failure_reason` with metrics) no longer
  counts as a failed rollout (before, a run whose rollouts all scored zero ended with rc 1). nuPlan behaviour is otherwise unchanged.
