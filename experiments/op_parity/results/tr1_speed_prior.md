# TR1: training the ego-only speed prior out of the nuPlan adapter. No arm keeps the nuPlan-board gain and removes the prior; the two halves of the prior separate cleanly

Written 2026-10-10. Pre-registration [plans/2026-10-10-tr1-prereg.md](../plans/2026-10-10-tr1-prereg.md) (pushed before any TR1 score; amendments 1 - 3: a WOD
in-domain track was added, trimmed to one arm and then dropped on lane DIAG1's nil training-free ceiling on WLG; nothing was trained in the WOD recipe).
Code `scripts/pp_train.py` (TR1 block), `lib/parity_adapter.py` (`ego_transform`, `lead_features`), `scripts/tr1.py`, `scripts/tr1_chain.sh`. Tables
[tr1/](tr1/), figure `figs/tr1/stage0_frontier.png`. Decision 221. Follows decision 218 (DIAG1). No WA-JEPA weights or features; logged futures only as
labels; the base model's own lead outputs are inputs; nothing submitted; no on-policy training.

Five arms were compared at pilot scale against the reference; one went to full scale under a registered exception. Read every number below with that count.

## Arms (P2H10 recipe otherwise unchanged)

| arm | change |
|:--|:--|
| B0 | none (reference at the arm's scale) |
| NA | fed ax zeroed inside the adapter (training and serving) |
| NAH | NA + history poses re-spaced along their own path at the current speed |
| LD | NAH + one adapter memory token of the base model's lead outputs (3 lead probabilities, distance, speed, acceleration, fed speed), warm-started with its own head |
| LG | LD + imitation target = the logged path at the **base plan's arc lengths** on rows fed >= 0.5 m/s where the base model reports a lead (p > 0.5): 42% of imitation rows |
| PO | LD + that target on every row fed >= 0.5 m/s (91% of imitation rows); the launch keeps the logged target |

Plumbing: the modified trainer with every option off reproduces the unmodified one (20 steps: losses, dev ADE and all saved weights identical, max |d| 0.0);
the probe file's plans equal the bench navtest plans (max 0.0 m). Lead-token pre-training (own head, target = logged minus base-plan arc length): dev
Huber 0.805 against 1.160 with the speed only and 1.298 with no input (pilot rows; full rows 0.848 / 1.180 / 1.321), R^2 at 4 s 0.47 against 0.13: the
base model's lead outputs say where its timing departs from the log.

## Stage 0: pilot scale (navsim/op-parity-s234, 3 000 steps x 64, seed 0)

navtest probes with `diag1.py`'s definitions on 12 146 tokens ([tr1/probe_s0.md](tr1/probe_s0.md)); navtest EPDMS through `jevdrive.bench`, CI over logs
([tr1/bench-s0/navtest_paired.md](tr1/bench-s0/navtest_paired.md)); WOD val through the decision 155 harness ([tr1/wod-s0/](tr1/wod-s0/contrasts.md)). Base model (P0)
on the same frames: onset share 0.822 (741 frames), launch behind a stopped lead 0.33 (388 frames); the log 0.565 / 0.90.

| arm | brake onset share [95% CI] (x base) | slope: ax / ax + history (m/s per m/s^2) | observed slope on fed ax | launch behind a stopped lead | navtest EPDMS (- B0 [95% CI]) | WOD val RFS (- B0 [95% CI]) | onset / slope / navtest line |
|:--|:--|:--|:--|:--|:--|:--|:--|
| B0 | 0.637 [0.593, 0.688] (0.78) | 0.326 / 0.318 | 0.315 | 0.99 | 87.93 | 7.885 | reference |
| NA | 0.688 [0.647, 0.736] (0.84) | 0 / -0.003 | 0.272 | 0.99 | 87.34 (-0.59 [-0.82, -0.35]) | 7.458 (-0.426 [-0.593, -0.265]) | no / yes / no |
| NAH | 0.691 [0.650, 0.737] (0.84) | 0 / 0 | 0.271 | 0.99 | 87.44 (-0.49 [-0.75, -0.22]) | 7.453 (-0.432 [-0.600, -0.268]) | no / yes / yes |
| LD | 0.682 [0.637, 0.728] (0.83) | 0 / 0 | 0.279 | 0.98 | 87.54 (-0.39 [-0.67, -0.12]) | 7.456 (-0.429 [-0.589, -0.273]) | no / yes / yes |
| LG | 0.831 [0.797, 0.862] (1.01) | 0 / 0 | 0.184 | 0.99 | 86.07 (-1.86 [-2.28, -1.45]) | 7.787 (-0.098 [-0.242, +0.044]) | yes / yes / no |
| PO | 0.833 [0.798, 0.863] (1.01) | 0 / 0 | 0.009 | 0.99 | 84.83 (-3.10 [-3.58, -2.64]) | 7.948 (+0.063 [-0.093, +0.211]) | yes / yes / no |

Shipped on WOD val 8.005; full-scale P2H10 7.708. Figure `figs/tr1/stage0_frontier.png`: look at the left panel, where no point is in the upper-right
region (onset restored and navtest kept): the three input arms stay left of the onset line, the two target arms fall below the navtest line; the right
panel shows that only the target arms move toward shipped on WOD, in the order of their navtest cost.

**Registered gate: no arm passes all three lines.** Reading by mechanism:

- **Removing the acceleration input costs 0.5 on navtest after retraining, not the 3.3 of zeroing it at serving** (NA -0.59, NAH -0.49: NC -0.2, TTC -0.35,
  EC -1.2, EP 0). The injected-acceleration slope is 0 by construction. But the observed slope of (arm - base) first-segment speed on the acceleration the
  ego actually has stays at 0.27 of B0's 0.31: the retrained plan pathway reads the ego's acceleration from the frames (the synthesised slots are warped
  along the ego track, as for yaw in decision 205) once the adapter stops supplying it. The continuation moved from the ego input to the image path; the
  registered slope probe cannot see it. Brake onset rises only from 0.78 to 0.84 of the base model's share.
- **The pose history adds nothing** (NAH = NA within noise on every read), as DIAG1's `posecv` switch said.
- **Lead outputs as an adapter input do not make the longitudinal change scene-conditional** (LD = NAH: onset 0.83, launch behind a stopped lead 0.98, navtest
  +0.10 [not separable]), although the same numbers predict the log-minus-base arc residual in the token's own pre-training. Consistent with decision 204:
  a read-out of the same frozen features fed back as memory is not used.
- **Brake onset is restored only by changing the target** (LG, PO: 1.01 x the base model; "base slows, arm does not" on closing-lead frames 5.8% -> 0.4% / 0.2%).
  The cost is EP (-2.3 / -1.7) and above all EC (-9.0 / -13.6): plans that follow the base model's timing on some frames and the log's on others are less
  consistent from frame to frame.
- **The launch is untouched by every arm** (0.98 - 0.99 behind a stopped lead against the base model's 0.33): no arm trained it away, and the lead token did
  not condition it. navtrain standstill rows are launches by selection (decision 218), so the label does not contain the hold.
- **WOD zero-shot gets worse, not better, without the acceleration input** (-0.43 for NA / NAH / LD, ADE@3s +0.25 to +0.28 m): the speed prior is still there
  and now rests on speed alone. Only PO, which never learns nuPlan timing while moving, is at B0's level or above (7.948; against shipped -0.057
  [-0.218, +0.104], moving frames +0.101 [-0.009, +0.222]).

Registered exception (no arm passes; one arm passes the onset and slope lines with navtest >= B0 - 2.0): LG is the only arm inside it (PO is at -3.10) and
went to stage 1 as a transfer-only arm that cannot receive the verdict.

## Stage 1: LG at full scale (navsim/op-parity-full, 10 000 steps x 128, seeds 0 / 1; tags `T1LG-F-s0`, `T1LG-F-s1`)

| read | LG | P2H10-F | difference [95% CI] | line |
|:--|:--|:--|:--|:--|
| navtest EPDMS ([tr1/bench-s1/](tr1/bench-s1/navtest_paired.md)) | 86.82 (86.81 / 86.84) | 88.67 | -1.85 [-2.30, -1.44] (EP -2.27, EC -9.04, NC -0.16, TTC -0.11) | >= -0.3: not met |
| navhard ([tr1/bench-s1/](tr1/bench-s1/navhard_paired.md)) | 33.28 | 31.84 | +1.44 [-1.21, +4.06] (stage 1 -2.51 [-4.92, -0.08], stage 2 +2.72 [-0.40, +5.72]) | reported |
| brake onset share x base ([tr1/probe_s1.md](tr1/probe_s1.md)) | 1.00 / 1.00 (0.825 / 0.819) | 0.78 / 0.77 | | >= 0.9: met |
| slope ax / ax + history | 0 / 0.001 | 0.31 / 0.30 | | <= 0.3 x: met |
| observed slope on fed ax | 0.176 | 0.288 | | reported |
| launch behind a stopped lead | 0.99 / 0.98 | 0.99 / 0.98 | | reported |
| WOD val RFS zero-shot ([tr1/wod-s1/](tr1/wod-s1/contrasts.md)) | 7.610 (7.635 / 7.585) | 7.708 | -0.098 [-0.247, +0.045]; against shipped -0.395 [-0.601, -0.204]; WLG 8.187 | >= 7.955: not met |
| WOD ADE@3s | | | -0.066 m [-0.117, -0.018] | reported |
| AlpaSim nuPlan public, 700 scenes ([tr1/alpasim_a.md](tr1/alpasim_a.md)) | 0.8915 (0.8885 / 0.8945) | 0.9481 (0.9468 / 0.9495, re-run at this checkout) | -0.0566, scenes [-0.0718, -0.0412], logs [-0.0804, -0.0315] | >= 0, lower bound > -0.005: not met |
| AlpaSim at-fault collision zeros / at-fault events / slow scenes | 5.5 / 17 / 190 | 7 / 17 / 108.5 | zero <-> non-zero 8 / 9 scenes | zeros lower: met |

- WOD strata of LG - P2H10: moving +0.015 [-0.161, +0.184], standstill -0.277 [-0.536, -0.023], launch frames -0.376 [-0.666, -0.079]. The moving
  frames, where the target changed, do not move; the standstill loss is not explained (same launch share; seeds -0.18 / -0.38).
- AlpaSim: the same picture as FIX1's lead limit at serving (fix1_700.md: (b) -0.086, slow 112 -> 271): at-fault collision zeros go down a little and the
  score is paid in progress (mean progress 0.86 against 0.94, slow scenes 190 against 108.5). The baseline re-run (recipe mean 0.9481) reproduces OT3's 0.9483.
- No arm passed, so no winner was trained in the AlpaSim input standard with yaw-rate rows.

## Verdict

**No arm "keeps the nuPlan gains and removes the exported prior".** Five arms compared; none passes the three stage-0 lines; the one transfer-only arm (LG)
fails navtest (-1.85), the WOD line and the AlpaSim nuPlan line, and the transfer evidence it was run for is nil (WOD -0.098, CI including 0). Nothing is
promoted on a secondary read (navhard +1.44 has a CI including 0).

What the lane does establish, as a capability boundary:
1. The acceleration input is not where the navtest gain lives once the adapter is retrained (-0.5, not -3.3): the plan pathway recovers the same
   continuation from the frames. Removing an ego input does not remove an ego-state behaviour while the frames carry the ego motion.
2. In this recipe "keep the nuPlan-board gain" and "inherit the base model's brake onset" are the same axis read from two ends: the arms line up
   B0 -> NA / NAH / LD -> LG -> PO with onset 0.78 -> 0.84 -> 1.01 -> 1.01 and navtest 0 -> -0.5 -> -1.9 -> -3.1. The nuPlan board scores the logged human's
   timing (EP, and EC through frame-to-frame consistency); the base model's timing is later to go and earlier to brake.
3. The base model's lead outputs are informative (R^2 0.47 on the log-minus-base arc residual) but giving them to the adapter changes nothing; they act
   only when they select the target.
4. The launch behind a stopped lead cannot be trained away from navtrain labels.

## Limits

- Open loop except the AlpaSim read; pilot scale, one seed, for five of the arms (seed noise of the reference recipe at pilot scale is about 0.1 - 0.3 EPDMS,
  decision 172); NAH and LD pass the navtest line by 0.01 and 0.11, NA misses it by 0.09: these three are not separable from each other.
- The registered slope gate was met by construction for every arm (the injected input is not read); the observational slope is the informative one and
  is correlational (on logs the acceleration also predicts the human's next second). A probe that re-synthesises the slots at another speed profile was
  not built.
- The frame-path reading of point 1 is an inference from the observed slope and the unchanged navtest NC / TTC, not an intervention on the frames.
- PO was not run at full scale or in AlpaSim (below the exception's floor); it is the arm closest to shipped on WOD.
- EC dominates the navtest cost of LG / PO; whether a smoother blend of the two timings (instead of a per-row switch at p = 0.5) keeps onset at a lower
  EC cost was not tested.
- AlpaSim: local rendering, 27 logs, all 1 491 public scenes already used for recipe selection (regression check only); the P2H10-F baseline is a re-run
  at the lane's checkout.
- PAI: not run. Lane PAI2 has a baseline chain on the box (`experiments/alpasim/scripts/pai2_chain.sh`) but no "PAI on the GPU box" section or
  `results/pai2_box.md` yet. Once it is runnable, the kept tags run as driver `sh30` with `SH30_TAG=T1LG-F-s0` (and `-s1`) through PAI2's chain with its
  chunk lists, paired with its P2H10 baseline; the nuPlan command used here was
  `TR1_STACKS=3 scripts/tmux_run.sh tr1-cl env TR1_STACKS=3 experiments/op_parity/scripts/tr1_chain.sh cl a T1LG=T1LG-F-s0+T1LG-F-s1`.

## Box notes

- Jobs were SIGKILLed 20+ times while the container sat at `memory.high` (docs/remote-box.md, "Host memory"): the six pilot WOD evals (B0's 14 starts), both
  full trainings twice (restarted from step 0 by the pool, `--tries 3`). No simulator stack was killed. Sampler excerpt of the first kill:
  `$DATA_DIR/runs/op_parity/tr1/boxwatch-wod-kill.tsv` (memory.current 549 of 552 GiB, pids 17 600 while six navtest scorings started). The WOD evals are
  resumable and were looped to completion; the closed loop ran with 3 stacks.
- Kept on the box: `T1LG-F-s0 / s1`, `T1B0-P-s0` (82 MB each), `$DATA_DIR/runs/alpasim/tr1` (2.8 GB), `$DATA_DIR/runs/op_parity/tr1` (0.15 GB). Deleted:
  the pilot checkpoints of NA, NAH, LD, LG, PO, all pilot ONNX exports, smokes, the reference worktree.
- Cost: about 7 card-hours (pilots 0.6, full trainings 2.5 including the restarts, probes / WOD / navhard 1.5, AlpaSim 2.4), about 4 h wall.
