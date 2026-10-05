# Stage B: frame protocol x input arm, trained: the pre-registered rule picks W (CPU warp); native 2 Hz (N) loses 4.0 EPDMS with ego inputs

Written 2026-10-06. Design and rule: [../plans/2026-10-06-parity-prereg.md](../plans/2026-10-06-parity-prereg.md), addendum 1 (written before any pilot
score was read; the user approved Stage B as written). Chain: `scripts/pp_stageB_submit.sh`; report `scripts/pp_stageB_report.py` ->
[stageB_cells.md](stageB_cells.md) (every protocol x arm x seed, all subscores), [stageB_effects.md](stageB_effects.md) (contrasts),
[stageB_decision.json](stageB_decision.json) (rule and guards). Stage A (no training): [stageA.md](stageA.md).

Setup: train protocol N (4 slots at the 2 Hz keys, 0.5 s pairs, no synthesis), W (8 slots at 0.2 s, CPU ego-motion warp lattice frames), G (8 slots,
GIMM) x arm P1 (inputs zeroed) / P2 (+ ego, pose history, command) x seed 0 / 1; the pilot recipe (4 874 navtrain tokens, 600 steps, batch 64),
same rows, same init. The anchor teacher is shipped Cinque on the G frames for every protocol (identical targets across protocols). Test (i) the
matched protocol on all 12 146 navtest tokens, (ii) the 1 499 real-frame tokens (W / G models on real 0.2 s frames, N models on the real keys).
v2 EPDMS, devkit navsim main @ 0a380a9; contrasts per token on seed means, bootstrap over 136 navtest logs.

## Numbers (matched protocol, full navtest, seed mean)

| protocol | P0 shipped | P1 | P2 | P2 - P1 [95% CI] | P2 EP / DAC | P2 ADE vs log (m) | P2 speed ratio |
|:--|--:|--:|--:|:--|:--|--:|--:|
| G (GIMM) | 81.11 | 81.77 | **86.63** | +4.86 [+4.18, +5.52] | 86.8 / 95.0 | 0.85 | 1.00 |
| W (warp) | 80.51 | 81.58 | **85.73** | +4.15 [+3.63, +4.67] | 86.4 / 94.8 | 0.85 | 1.00 |
| N (native 2 Hz) | 48.80 | 79.84 | **82.63** | +2.78 [+2.19, +3.38] | 85.8 / 92.8 | 1.21 | 1.00 |

Seed spread (s1 - s0) is <= 0.17 in every P2 cell and 0.07-0.48 for P1.

| contrast | mean [95% CI] |
|:--|:--|
| P2: G - W | **+0.90 [+0.47, +1.31]** |
| P2: G - N | +4.00 [+3.12, +4.93] |
| P1: G - W | +0.19 [-0.30, +0.69] |
| P1: G - N | +1.92 [+1.17, +2.73] |
| interaction (P2 - P1)_N - (P2 - P1)_G | -2.08 [-2.63, -1.52] |
| interaction (P2 - P1)_W - (P2 - P1)_G | -0.71 [-1.23, -0.20] |
| test on real frames - matched test, P2: G / W / N | -0.47 [-1.48, +0.56] / +0.73 [-0.52, +1.95] / +0.78 [+0.57, +0.99] |
| test on real frames - matched test, P1: G / W / N | +3.28 [+2.01, +4.56] / +3.22 [+2.25, +4.13] / +2.12 [+1.81, +2.45] |

## Pre-registered decision

Rule (on P2, matched protocol): N if G - N <= 1.0, else W if G - W <= 1.0, else G; guards: plan speed ratio within 5 % of G, no EP / DAC trade.

| protocol | G - X | speed ratio (G 1.001) | EP / DAC vs G | guards | verdict |
|:--|--:|--:|:--|:--|:--|
| N | 4.00 | 0.995 | -0.94 / -2.18 (both down, no trade) | pass | **fails** (4.0 > 1.0) |
| W | 0.90 | 0.996 | -0.39 / -0.21 (no trade) | pass | **chosen** |

**Full run uses W** (CPU warp, train and test). The margin is thin: G - W's mean is 0.90 but its CI reaches 1.31.

## What this says

- **Explicit ego state does not replace the image-pair motion cue; the opposite holds.** The interaction is negative: the inputs gain less under
  N (+2.78) than under G (+4.86), so P2 widens the protocol gap (G - N 1.9 on P1, 4.0 on P2). After training, P1 under N recovers almost all
  of shipped's 32-point N loss (48.8 -> 79.8), so the 0.5 s pair is mostly re-learnable scale shift (Stage A), but a residual stays, and with
  the ego inputs it is larger: P2-N's plans keep the log speed (ratio 1.00) yet are less accurate (ADE 1.21 vs 0.85 m, DAC -2.2, NC -1.0).
  [I] The 4-slot, 0.5 s queue carries less scene dynamics (other agents, lane geometry over time) than 8 slots at 0.2 s; speed alone is not it.
- **No scorer artifact in the choice.** All P2 arms have plan speed ratio 1.00 (the ego input pins the speed), and the protocol losses come
  with EP and DAC both down, not traded.
- **What P2 learned does not depend on test-time synthesis.** Tested on real frames, P2-G changes by -0.47 (CI includes 0) and P2-W by +0.73:
  with ego inputs the model is insensitive to the frame source. P1 gains ~3 points from real frames in every protocol (its speed comes from
  the images).
- Strict-parity framing: W uses only the 2 Hz keys and the logged ego track (no learned interpolator, no information WA-JEPA lacks beyond what
  its own ego history gives); G adds a learned interpolator's frames. W costs 0.90 EPDMS against G.

## Full-run estimate under W (not launched)

- **Data.** Full navtrain: 103 288 tokens (WA-JEPA's stage 2 set). op_lb has navtrain keys only for 5 400 tokens; full keys stored would be
  ~150 GB (box has 266 GB free), so prep renders the 4 CAM_F0 keys on the fly (needs a small pp_prep change), warps 6 lattice frames on CPU,
  and stores only tokens: front 256 KB + side 288 KB per token = **~56 GB** on disk.
- **Cache.** Measured: front (warp) 650-700 pairs/s per job (CPU-bound, 18 workers), side 46-63 tokens/s (22 workers). 103 k tokens x 8 pairs,
  sharded over 3 cards with ~20 cores each: front ~25-30 min, side ~15-25 min; **~1.5 GPU·h, ~1 h wall**.
- **Teacher.** The pilot anchored to shipped on GIMM frames; full navtrain has no GIMM frames (63 GPU·h to make). Proposal: teacher = shipped
  on the W frames (Stage A: W8 - R8 -2.8 for shipped). This is a recipe change; in the pilot the W arms' anchor loss stayed high (cons 0.58 vs
  0.02 under G) because input and teacher protocols differed, which this removes.
- **Training.** P2 tokens fit on one card (front 27 GB fp16 + model; P3 + side 30 GB). Pilot throughput: P2 ~10 it/s, P3 ~15 it/s at batch 64.
  At ~8 epochs (pilot ratio) = 13 k steps: ~25 min per run; at batch 256 / 30 k steps ~1-1.5 h. Arms P1 / P2 / P3 x 2 seeds: **~3-6 GPU·h**.
- **Readouts.** navtest plans minutes; devkit scoring ~10 min per model (CPU, 22 threads); HUGSIM 64 under `exam` and `spec` for P0-P3:
  **~1-2 GPU·h**.
- **Total: ~6-9 GPU·h, ~4-5 h wall** on the 3 shared cards: over the 6 GPU·h line, so it waits for main.

## Caveats

- Pilot scale for every cell (4 874 tokens, 600 steps); two seeds, seed spread small (<= 0.48).
- The anchor teacher is the G-protocol shipped output for every protocol (identical targets, by design); it favours G slightly, which is in
  the direction that makes the W choice conservative.
- N's queue length (4 slots) is fixed by NAVSIM's 1.5 s history; a longer N queue is not testable on NAVSIM.
