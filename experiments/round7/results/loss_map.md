# LOSS-MAP: where R6DAG still loses points on PAI 281 (ACC + GAP), what is recoverable, and what to work on next (2026-10-21)

Diagnosis, no training, no new closed-loop reads. Subject: R6DAG-L18 s0 / s1 (DAgger + NuRec-map lateral teacher, decision 434) under the
submission serve (ACC + GAP 0.7,2.7, g10, ALPASIM_FAST=1); reference NTV0-L18-s0 + ACC + GAP (`p1021a`, board PCS 2154). 1 point = 1 scene
(partials count 1 - score), 281 scenes, no driver RPC error in either DAG read. Human-readable page with the clips:
[research/round7/loss_diag/index.html](../../../research/round7/loss_diag/index.html) (decision 436).

Inputs: pai_fix facts (`$DATA_DIR/runs/pai_fix/facts/<tag>-281ag.csv`, `fclass` / `primary` / `stratum`) of every stored 281 ACC + GAP read
(R6DAG s0 / s1, NTV0, R2C, R2J, R3MV s0 / s1, R5LC s0 / s1, R5A, R6FIX s0 / s1); the ruler rows of decision 434 (`$DATA_DIR/tmp/pcs_r6d/rows.csv.gz`,
45 subjects incl. native / ACC-only / g5 / g20 / holdlead reads and third-party garage / go-straight) and the paimix 2PL fit
(`tmp/pcs_r6d/out/fits/paimix.npz`: item discrimination a, difficulty b; band = the 13 pool reads of our family). Code:
[scripts/loss_map.py](../scripts/loss_map.py) (`pull`, `table`); per-scene table [loss_map.csv](loss_map.csv), class summary [loss_map.json](loss_map.json).

## Definitions

- Class of a lost scene (score < 0.98): pai_fix `fclass`, refined: offroad by `primary` (turn inside = `off_junc_inside` + `off_curve_inside`,
  turn outside = `off_junc_outside`, junction straight, straight), rear-hit by stratum (launch / moving), `near1` = 0.98 <= score < 1
  (no class in the 281 report, but a non-1 score on a high-a item is what the zoib ruler charges).
- Ours solves = some stored 281 read of our checkpoints under ACC + GAP scores >= 0.98 (21 reads: NTV0 s0 / s1, R2C, R2J, R3MV s0 / s1,
  R5LC s0 / s1, R6FIX s0 / s1, R5A, JTN s0 / s1, JTXS, NTVX, NTVXS, STANDFMX s0 / s1, STANDFMXH, BNC2K, RSG; plus the other R6DAG seed).
  Recoverable points = sum over the class's scenes of (best of those reads - DAG score). Never solved = no read of ours under any serve.
- Ruler weight: imperfect scenes on a >= 1 items (the items every band read aces); `band pass` = share of the 13 band reads scoring 1 there.
  The band is our own family (NTV0 / JTN / RSG / STAND): high a means "our family solves it", not "strong teams solve it".
- Board conversion (decision 426): 1 local scene point ~ 3 board PCS by the mean (0.6 x the local mean gap), without the 2PL amplification.

## Loss map

R6DAG s0 0.7767 (lost 62.8), s1 0.8165 (lost 51.6). Lost scenes: s0 82, s1 70, both seeds 67 (s1's losses are nearly a subset of s0's).
Best of ours per scene (oracle, the other DAG seed included): 0.879 for both seeds (recoverable 28.7 / 17.5 points); 35 lost scenes no read
of ours ever solved under any serve.

| rank | class | s0 scenes / zeros / pts | s1 scenes / zeros / pts | both / either | NTV0 solves (s0 / s1) | ours ACC+GAP solves (s0 / s1), recoverable pts | never solved, any serve | ruler: imperfect on a >= 1 (s0 / s1); band pass | lever | cost | est. gain (pts, mean of seeds) |
|--:|---|--:|--:|--:|--:|---|--:|---|---|---|---|
| 1 | offroad: turn, cutting the inside | 10 / 10 / 10.0 | 7 / 7 / 7.0 | 7 / 10 | 4 / 3 | 9 / 6, 9.0 / 6.0 | 0 / 0 | 2 / 0; 0.35 | DAgger map teacher iteration 2 (running: student = R6DAG, curve-entry / over-speed label speed min(student, human)); iteration 3 if needed | iteration 2 paid; iteration 3 ~1 h rows + ~10 GPU job-h | 7.5 x 0.5 = 3.8 |
| 2 | partial: slow / unforced slow-down (+ near-1 0.98-0.99) | 6+3 / 0 / 0.95 | 6+2 / 0 / 0.79 | 4+1 / 8+4 | 5 / 4 | 6 / 5, 0.64 / 0.49 | 2 / 2 | 4 / 5 (+ d90fe123 in row 6); 0.37 / 1.0 | F2 row hygiene (FIX fixed bbc53aa8 on both seeds) + a longitudinal "no unforced slow-down" label on the DAgger rows (no lead, human holds speed: arc length max(student, human)) | ~1-2 h code, rides on the iteration-3 training | 0.6 x 0.5 = 0.3 (mean); ruler: s1's 6 imperfect a >= 1 items are all here (+225 if fixed, d434) |
| 3 | drift / lane change out of the corridor | 5 / 5 / 5.0 | 4 / 4 / 4.0 | 4 / 5 | 2 / 1 | 4 / 3, 4.0 / 3.0 | 1 / 1 | 0 / 0; 0.29 | command rule from route heading change (d432) + DAgger lateral teacher | ~2-3 h code + closed-loop smoke; training rides on iteration 3 | 3.5 x 0.3 = 1.1 |
| 4 | offroad: straight / junction straight / turn outside | 10 / 10 / 10.0 | 5 / 5 / 5.0 | 4 / 11 | 2 / 0 | 5 / 1, 5.8 / 1.9 | 3 / 3 | 2 / 0; 0.0-0.04 | DAgger lateral teacher (as 1) | as 1 | 3.9 x 0.3 = 1.2 |
| 5 | stop overshoot (passes the human's rest point, no lead) | 20 / 20 / 20.0 | 19 / 19 / 19.0 | 19 / 20 | 1 / 0 | 1 / 0, 1.0 / 0.0 | 15 / 15 | 1 / 0; 0.0 | stop-line longitudinal teacher on DAgger rows (arc length min(student, distance to the human's rest point)); ACC + GAP sends gentle decelerations at cruise speed (d406) | ~3-4 h code and gates + ~10 GPU job-h; read native and ACC + GAP | 19.5 x 0.05-0.2 = 1-4 (undemonstrated) |
| 6 | partial: rear-ended while moving / stopping | 13 / 0 / 4.4 | 13 / 0 / 4.2 | 11 / 15 | 3 / 2 | 6 / 6, 2.0 / 1.9 | 5 / 5 | 1 / 1; 0.23 | none beyond 2 (non-reactive replay followers) | - | 2.0 x 0.1 = 0.2 |
| 7 | partial: slow launch, rear-ended | 10 / 0 / 4.4 | 8 / 0 / 3.6 | 8 / 10 | 3 / 1 | 5 / 3, 2.3 / 1.4 | 5 / 5 | 0 / 0; 0.09 | none (launch line closed, d419-d428; launches 35 / 37) | - | 1.9 x 0.1 = 0.2 |
| 8 | at-fault collision: front / oncoming / stationary; side / cyclist | 4+2 / 6 / 6.0 | 3+3 / 6 / 6.0 | 3 / 9 (distinct scenes) | 3 / 1 | 3 / 2, 3.9 / 2.9 | 2 / 2 | 0 / 0; 0.03-0.13 | none ready (collision hinge on PAI rows unverified) | ~4 h + ~10 GPU job-h | 3.4 x 0.15 = 0.5 |
| 9 | route turn missed | 2 / 2 / 2.0 | 2 / 2 / 2.0 | 2 / 2 | 0 / 0 | 0 / 0, 0 / 0 | 2 / 2 | 0 / 0; 0.0 | done (NTV0 15 -> 2) | - | 0 |

Per-class split of rows 4 and 8 (s0 / s1 zeros): straight 4 / 2 (both 1), junction straight 3 / 2 (both 2), turn outside 3 / 1 (both 1);
AF-front 4 / 3 (both 2), side 2 / 3 (both 0). PCS value per pai_fix class if fixed (decision 434, `pcs_dag.md`, s0 / s1): rear-hit +5 / +21,
slow +79 / +45, overshoot +7 / +8, route-turn 0 / +1, offroad +16 / +12, other zeros +4 / +6; every imperfect a >= 1 item -> 1: +249 / +225.

## Reading

1. **Lateral is still the largest recoverable block, and it is demonstrated.** Turn-inside offroad: 7 scenes fail on both seeds, every one is
   solved by some read of ours (NTV0 solves 3-4); the s0 - s1 gap (offroad 20 vs 12) is mostly this class. It is what DAgger iteration 2 targets.
2. **The ruler gap of s1 to NTV0 (-155) sits on six partial scenes, not on zeros.** s1 has no zero on any a >= 1 item; its six imperfect
   high-a items are bbc53aa8 0.74 (highway slow-down), d90fe123 0.65 (rear-ended after slowing), a1b166cc 0.90, 94c1d722 0.978, 2a0efada 0.99,
   37d660b1 0.99. NTV0 and R2C score 1.0 on all six; every PAI-human-row recipe (R2J, R3MV, R5LC, R5A, R6FIX, R6DAG) loses some of them
   (2a0efada 0.81-0.95, d90fe123 0.0-0.75) -> the slow-down is inherited from the PAI human rows, and R6DAG's labels keep the student's speed.
   R6FIX (F2: no-lead decelerating rows dropped) scores 1.0 on bbc53aa8 on both seeds. Worth little in the mean (about 1 point), a lot on the
   ruler, with the caveat that the band is our family.
3. **Stop overshoot is the biggest block (19-20 points, 0.07 of the mean) and the least reachable.** 19 of 20 fail on both seeds; no read of
   ours under ACC + GAP solves any of the 19; four are solved only under other serves: 444fca69 and 47db2eb6 by native VC + LEAD reads
   (NTV0 native, JTN native, RSG-g10 / g20; garage too), 9b6c68b4 by RSG with ACC but no GAP, 68c3be44 by NTV0 + ACC + GAP + keywarp (0.99).
   Mostly consistent with decision 406 (ACC sends the plan's gentle deceleration at cruise speed); 15 are solved by nothing of ours. Item a is about 0.2 (the whole family fails them), so the ruler
   values a fix at +7 / +8. Every past stop fix paid in launches or rear-hits (decisions 400, 402, 412).
4. **Rear-hit partials (8.0-8.8 points) are mostly not ours to fix**: rear-ended at about the human's speed by non-reactive replay followers;
   the best read of ours recovers only about 2 points per kind.
5. Collisions (6 zeros per seed) are mostly seed noise (3 of 9 distinct scenes fail on both seeds); route-turn is done (2 left, never solved).

## Recommendation

Concentrate on (1) lateral turn-inside / straight offroad through the DAgger iterations (already running; judge iteration 2 by these 7
both-seed scenes and the s0 offroad count), (2) the unforced slow-down partials: add F2 and a longitudinal keep-speed label to the next
DAgger row build (cheap, protects the ruler), and (3) the command rule from route heading change for the high-speed drifts, folded into the
same arm. Do not open a stop-line arm now: it is the largest pool but nothing of ours solves it under the submission serve, and the history
says it costs launches.

## Clips

See the page; clip list, sources and re-read checks in `research/round7/loss_diag/manifest.json`. Videos: `$DATA_DIR/runs/round7/loss_videos/`
(box) and `research/round7/loss_diag/media/` (Mac, gitignored). Code [scripts/loss_videos.py](../scripts/loss_videos.py) (`asl` offline
render of stored zero rollouts on CPU, `launch` low-priority pool re-reads for partials and contrasts, `collect`, `page`),
[scripts/loss_page.py](../scripts/loss_page.py).
