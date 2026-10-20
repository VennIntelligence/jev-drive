# pai_fix: targeted PAI exam for stop and turn fixes

status: live
decisions: none
index: PAI-FIX exam: 118 scenes, 0.44 card-h (val160 part 0.28); NTV0 AG TARGET 0.486, STOP saturated under ACC

**Question.** Can a small PAI closed-loop exam tell, in well under 0.5 card-hours a read, whether a candidate checkpoint fixes NTV0-L18-s0 +
ACC + GAP's two main PAI losses: driving past the human's stopping point (d381: 23 zeros) and entering route turns too fast and under-turning
(11 straight-at-turn plus junction / curve offroad)?

**Conclusion.** The exam is 118 scenes: val160 TARGET 60 (STOP 12, STOP-L 11, TURN 37) + GUARD 12, and a separate 281-derived column T281
(46 of A0's zeros). A full read costs 1.32 stack-hours = 0.44 card-hours at 3 stacks a card (0.32 by VRAM share), about 22 min wall; the
val160 part alone 0.28 card-hours. NTV0-L18-s0 + ACC + GAP scores TARGET 0.486 (29 zeros: 15 overshoot, 7 route-turn, 2 offroad, 5 AF-front),
seed 1 0.462 (paired -0.024 [-0.089, +0.035]). On the 281-derived version, TARGET ranks the 15 stored ACC + GAP checkpoints like the full read
(Spearman +0.87, 91 % of pairs) — **but under ACC + GAP the STOP subset is saturated**: every stored checkpoint has the same stop zeros (val160 11 / 12
for 6 checkpoints; 281 17 / 22 for 15), while the native serve fixes some (val160 1, T281 3). So stop fixes must be read natively (VC + LEAD)
as well as under ACC + GAP.

**Next.** Screen the ~3 station-anchored variants with `pai_fix.sh read <tag> vl` and `pai_fix.sh read <tag> ag`, paired against the NTV0-L18-s0
rows here. Then send 1-2 to the full PAI 281 read.

**Read more.** [plans/2026-10-20-pai-fix-prereg.md](plans/2026-10-20-pai-fix-prereg.md) (selection rules, metrics, read plan; written
before any exam read), [results/calib/calib.md](results/calib/calib.md) (stored-read calibration), [results/reference.md](results/reference.md).

## Exam definition (`scenes/exam.csv`, prereg [plans/2026-10-20-pai-fix-prereg.md](plans/2026-10-20-pai-fix-prereg.md))

The rules use only the human's log facts, never a model's score; the one exception is the guard's "passed" filter. Code: `scripts/pai_fix.py select`.
The facts come from the rf_tax extract (`pai_gap.human` / `rest_cue`, the d381 code) and the turn angle from `exam_nurec.csv` (6.4 s window).

| subset | scenes | source | rule | NTV0-s0 AG zeros |
|---|--:|---|---|--:|
| STOP | 12 | val160 | moving at the handoff (>= 1 m/s), ends still, no logged vehicle within 15 m of the rest point | 11 (10 overshoot) |
| STOP-L | 11 | val160 | the same with a vehicle there (stopping behind a lead; doubles as a guard) | 5 (5 overshoot) |
| TURN | 37 | val160 | neither, turns >= 25 deg within 6.4 s at a median > 1 m/s | 13 (7 route-turn, 2 offroad, 4 AF-front) |
| G-LAUNCH | 6 | val160 | below 1 m/s at the handoff, log moves > 5 m; all 6 family reads >= 0.98 | 0 |
| G-LEAD | 6 | val160 | moving, not ending still, a vehicle in the path within 30 m on >= 30 % of steps; family >= 0.98 | 0 |
| T281 (separate column) | 46 | test281 | A0's zeros: 23 cor_route_end + 23 turn / junction / curve | 45 (1 RPC error) |

- **Why val160.** It is the curated validation set minus PAI 281, and every scene is on the box. Its RSG-L18-s0 rollouts (P160) are in none of
  the NTV0 / RB0-F / RSG pools (d404); a closed 1 000-step pilot (d367) trained on them.
- **What is excluded.** Eight val160 scenes share a drive (log session clock within 600 s and a drivable start distance, pav.py's rule) or a road
  (track points within 10 m) with one of the 854 curated-train scenes on the box. The 854 are the d402 pool of 570 plus 284 more that the
  downloader was still fetching on 10-20. Four of the 8 links go to the d402 pool. All 8 were dropped: [scenes/drives_val160.csv](scenes/drives_val160.csv),
  [scenes/onbox_train_20261020.txt](scenes/onbox_train_20261020.txt).
- **For training lanes.** No exam scene is linked to an on-box curated-train scene. The 216 curated-train scenes that are linked to an exam scene
  are listed in [scenes/train_exclude.csv](scenes/train_exclude.csv). Training must exclude them along with [scenes/exam_scenes.txt](scenes/exam_scenes.txt).
- **The guard's "passed" filter.** It is set by the RSG / RB0-F family, because NTV0 had no val160 read; NTV0 then scored 1.0 on all 12 under
  ACC + GAP.
- **Chunks.** The lists are fixed: v000-v002 for val160 (24 each), t000-t001 for T281 (23 each), and s000 as a 1-scene smoke. The simulator
  repeats a scene only within an identical list.

## How to read a checkpoint

```bash
# box, repo root; chunk jobs via pai_box.sh (unchanged) in the pool, then a CPU tax + facts job; subsets "v t" (default) or "v"
bash experiments/pai_fix/scripts/pai_fix.sh smoke <TAG> ag                 # optional 1-scene check
bash experiments/pai_fix/scripts/pai_fix.sh read <TAG> ag [v|"v t"]        # ACC + GAP (submission serve) -> $DATA_DIR/runs/pai_fix/facts/<TAG>-ag.csv
bash experiments/pai_fix/scripts/pai_fix.sh read <TAG> vl [v|"v t"]        # native VC + LEAD
.venv/bin/python experiments/pai_fix/scripts/pai_fix.py report --ref "NTV0-s0 AG" --out <md> \
  --read "NTV0-s0 AG=experiments/pai_fix/results/facts/NTV0-L18-s0-ag.csv" "<TAG> AG=$DATA_DIR/runs/pai_fix/facts/<TAG>-ag.csv"
.venv/bin/python experiments/pai_fix/scripts/pai_fix.py cost --read $DATA_DIR/runs/pai_fix/reads/<TAG>-ag
```

## Cost (measured 10-20, pool events)

| read | scenes | stack-h | card-h (3 a card) | card-h (VRAM share) | s per scene per stack |
|---|--:|--:|--:|--:|--:|
| NTV0-L18-s0 AG, v + t | 118 | 1.32 | 0.44 | 0.32 | 40.4 |
| NTV0-L18-s0 VL, v + t | 118 | 1.27 | 0.42 | 0.31 | 38.7 |
| NTV0-L18-s1 AG, v | 72 | 0.84 | 0.28 | 0.20 | 41.8 |
| 1-scene smoke | 1 | 0.04 | 0.01 | 0.01 | 133 |

The five chunks ran in parallel: about 22 min wall from submit to facts. For comparison, a full PAI 281 read is 8 chunks of about 22 min, roughly
3 stack-hours or 1 card-hour.

## Calibration on stored reads (no compute; [results/calib/calib.md](results/calib/calib.md))

The same rules on test281 give STOP 22, STOP-L 23 and TURN 66. Over the 270 scenes that all 21 stored reads score without an RPC error:

- **Ranking.** Over the 15 full-281 ACC + GAP checkpoints (RSG; JTN s0/s1; NTV0 s0/s1; NTVX; NTVXS; BNC2K; JTXS; B s0/s1; B2 s0/s1; CPAIC1K;
  CPAIP1K), the 281-derived TARGET agrees with the full 281 mean:

  | subset | Spearman | Kendall | pairs with \|full diff\| >= 0.01 ordered the same |
  |---|--:|--:|--:|
  | TARGET | +0.87 | +0.73 | 0.91 (67 / 74) |
  | TURN | +0.83 | | 0.88 |
  | STOP-L | +0.42 | | |
  | STOP | +0.32 | | 0.22 |

  - STOP is uninformative for ranking because it is the same in every checkpoint: 17 / 22 zeros in 13 of the 15, and 18 in the other two.
  - TARGET also predicts the rest of 281 (the complement) at Spearman +0.51.
  - The T281 column does not rank checkpoints (-0.32). It counts fixes; it does not rank.
- **Noise floor.**
  - The A0 rerun (R0) is identical on every scene.
  - Seed pairs on TARGET: NTV0 +0.033 [+0.004, +0.070], JTN +0.018, B -0.036, B2 -0.002. These come with 3-9 zero flips and 9-20 changed
    scenes, almost all in TURN. STOP and STOP-L move by at most one scene (one JTN STOP flip).
  - The new exam chunks reproduce the stored reads exactly on the 281 scenes: NTV0-s0 AG 45 / 45 against A0, NTV0-s0 VL 46 / 46 against N0.
    Chunk composition did not change a score.
- **Chance level of T281 fixes.** Of A0's 43 clean T281 zeros, NTV0-s1 scores 0 and the other checkpoints 0-5 (median 2), almost all turn
  scenes. Overshoot fixes are 0-1 for every ACC + GAP read and 2-3 for native reads.

## Reference rows (NTV0-L18; [results/reference.md](results/reference.md), per scene [results/ref_scores.csv](results/ref_scores.csv))

| read | TARGET mean | zeros (overshoot / route-turn / offroad / AF-front / side) | rear-hit / slow partials | launches | GUARD mean | T281 scored > 0 |
|---|--:|---|---|--:|--:|--:|
| s0 AG | 0.486 | 29 (15 / 7 / 2 / 5 / 0) | 4 / 2 | 6 / 9 (TURN), 6 / 6 (G) | 1.000 | 0 / 45 |
| s1 AG | 0.462 | 30 (16 / 6 / 3 / 5 / 0) | 5 / 2 | 6 / 9, 6 / 6 | 1.000 | (not read) |
| s0 VL | 0.497 | 26 (12 / 4 / 5 / 4 / 1) | 8 / 9 | 6 / 9, 5 / 6 | 0.802 | 8 / 46 |

- **Seed noise on val160 TARGET (s1 - s0, AG).** -0.024 [-0.089, +0.035]: 2 new zeros, 1 fixed, 9 scenes changed, all in TURN. STOP and STOP-L
  are identical.
- **Stored family on the same 60 scenes, ACC + GAP, own chunks.** RSG s0 / s1 score 0.426 / 0.437 and RB0-F s0 / s1 0.402 / 0.420, so NTV0's
  margin on the exam matches its PAI 281 margin. STOP is the identical 11 zeros in all six checkpoints.

## Native (VC + LEAD) read alongside ACC + GAP: yes, for the stop question

- **The mechanism.** ACC's set speed is max(the speed at the handoff, the plan's own 2-4 s speed). The model's profile is used only when its
  previous plan asks to stop or brake: a 2 s arc under 2 m, or a deceleration of at least 1 m/s^2 (serve_fix.py `Acc`).
  - A plan that slows gently toward a line is therefore served at the handoff speed or above.
  - This is consistent with the saturation: every ACC + GAP checkpoint has the same stop zeros. Native reads fix 1 / 12 (val160) and 3 / 23
    (T281) of them; 281 STOP scores 0.29-0.34 natively against 0.227 for every checkpoint under ACC + GAP.
  - A station-anchored stop variant could be partly masked by ACC. Its gain must show natively first.
- **Turns.** ACC's curve limit already removes route-turn zeros (val160 AG 7 against VL 4 route-turn, but AG 2 against VL 5 offroad). The
  serve partly does the work a turn variant is meant to do, so the native read is the cleaner model readout here too.
- **Costs that are the serve's, not the model's.** The native GUARD shows rear-hit (3 + 1) and slow partials that ACC + GAP removes (GUARD 0.80
  against 1.00). In a native read, judge GUARD only against NTV0-s0 VL, never against AG.
- **Proposal.** Read each variant both ways: vl for "did the model fix it", ag for "does it survive the submission serve". Two reads cost 0.86
  card-hours with T281 and 0.56 without.

## Caveats

- **Small STOP and TURN cells.** STOP has 12 val160 scenes (11 NTV0 zeros) and STOP-L 11. A variant that fixes fewer than about 2-3 stop
  scenes cannot be told from nothing on val160 alone. T281 adds 23 stop zeros with a seed-chance level of 0, but it was chosen on A0's failures
  (selection-biased). Keep it as its own column, never in the main score.
- **TURN noise.** TURN is noisy across seeds: 9 of 37 scenes change, ±0.04 in the mean.
- **GUARD.** It is small (12 scenes) and was chosen as passing by the RSG / RB0 family. It is an alarm, not a regression suite. Breakage
  elsewhere is checked on the full PAI 281 read of the 1-2 picks.
- **Ranking calibration.** It is on 281, not val160. On val160 only the six family reads plus NTV0 s0 / s1 are available, and they order as on
  281 (NTV0 > RSG > RB0-F).
- **Same-drive rule.** It uses NuRec session clocks and positions only. The PAI-AV drive groups were not used (as in d402).
- **Seeds and stack.** One seed per checkpoint. The local NuRec renderer stack, not the private board.

<!-- files:begin -->
## Files

- `README.md` (README.md): targeted PAI exam for stop and turn …
- `2026-10-20-pai-fix-prereg.md` (plans): PAI-FIX 预登记：定向 PAI 小考（停车越点 + …
- `exam.csv` (scenes): 
- `train_exclude.csv` (scenes): 
- `pai_fix.sh` (scripts): a small targeted PAI closed-loop exam
- `pai_fix.py` (scripts): the targeted PAI exam
- `calib.md` (results): PAI-FIX calibration on stored PAI 281 …

[results/](results/) 14 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
