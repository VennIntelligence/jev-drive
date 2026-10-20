# D3: is there outcome signal for closed-loop learning on PAI? (2026-10-21)

Diagnostic for the round 5 framework's stream S3 ([../plans/2026-10-21-round5-framework-draft.md](../plans/2026-10-21-round5-framework-draft.md)).
No training. Code: [../scripts/d3.sh](../scripts/d3.sh) (chain), [../scripts/d3.py](../scripts/d3.py) (select / check / report).
Generated tables: [d3/tables.md](d3/tables.md); per scene: [d3/d3_scores.csv](d3/d3_scores.csv); everything: [d3/summary.json](d3/summary.json).
Box root `$DATA_DIR/runs/round5/d3/` (14 GB, zero-score rollout.asl only).

## Short answer

**There is signal, but only one part of it is a class-level rule.**

1. **42 % of scenes are informative** (best − worst ≥ 0.2 over the 4 rollouts): 63 of 149.
   - stop without lead 16 %, sharp turn 54 %, launch 66 %, other 33 %.
   - The unperturbed rollout is never the sole best in an informative scene. It ties for best in 22 of 63, and in 104 of 149 overall.
2. **One consistent winner: faster wins the partials.** Where A scores a partial (rear-hit 17, slow 13; 30 scenes, 17 of them launches),
   F (×1.15) gains on 11 and loses on none, mean +0.209. S (×0.85) loses on 16 and gains on none.
   - By class this is the launch result: F − A **+0.124 [+0.062, +0.198]**, 9 up / 0 down. Launch zeros 3 → 2. F takes 18.8 of 25 informative launches.
3. **The zeros have no class rule.** A has 43 zeros (corridor 26, offroad 12, AF 4). Some arm lifts 14 of them to ≥ 0.5 (S 7, L 10, F 2).
   - But the same arms break scenes A passes: S 4, L 8 (all AF-front / AF-side), F 6 of A's 76 full scores drop to 0.
   - In sharp turns the informative scenes split F 8.2 / L 5.2 / S 3.7 / A 3.0. Every arm's turn mean is within ±0.06 of A.
4. **Stops are not reachable by these perturbations.** 17 of 38 stop scenes are corridor zeros under A (the overrun, d404). No arm changes 14 of them.
   The ACC stop and brake guard serves the model's own profile, and the model's plan is what overruns.
5. **A class rule does not pay overall.** Picking each class's best-mean arm, leave-one-scene-out, gives −0.016 [−0.064, +0.029] against A:
   - launch +0.124 (always F);
   - turn −0.130;
   - stop −0.027;
   - other −0.035.

   The per-scene oracle is +0.137 [+0.092, +0.185]. So most of the variance is scene-specific: an outcome-selected row knows which arm won,
   a class prior does not.
6. **Rows: 63 informative scenes give 11 456 winner decisions** (10 Hz, after the 1.6 s handoff). 8 877 of them come after the winner leaves the
   worst rollout. 5 608 of those are from scenes A did not win. That is about 1 100 rows at the training rows' 2 Hz.
7. **Cost: 2.21 card-hours** for 600 rollouts, plus a 0.03 card-hour smoke. That is **1.49 card-hours per 100 scenes × 4 arms**,
   0.37 per 100 scenes per arm.
8. **Reads are deterministic.** A matches the stored d402 roll (same serve with frame dumps, different chunk lists) on 119 of 119 shared scenes.
   So the spread is closed-loop sensitivity to the serve, not read noise.

## Setup

**Scenes** ([d3_scenes.tsv](d3_scenes.tsv), fixed and committed before any roll, 60698555):
- 150 drawn from the PAI-ROWS eligible table (`experiments/pai_rows/results/scenes.tsv`, d410). That table already excludes test281, val160 and
  val160's drives.
- Non-held scenes only (crc ≥ 15), so the rows' held split stays clean.
- At most one scene per drive group (`drive_g` of nurec_select.csv): 150 drive groups, so the bootstrap is over independent drives.
- Order sha1(scene id). Classes: stop without lead 38, sharp turn 38, launch 38, other 36 (bend 18 + rest 18).
- 120 of the 150 are in the d402 pool (`physical_ai_av/curated-pilot@v1`); the 30 "rest" / new bend scenes are not.
- S failed on one sharp-turn scene with a driver RPC error (007fc5b6), so the tables use 149 scenes.

**Serve**: NTV0-L18-s0, `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`, steering weight 10, `ALPASIM_FAST=1`, 4 rollouts per stack.
This is the stored NTV0 + ACC + GAP read's serve (the PAI candidate, d366).

**Arms**: one rollout each per scene. Every knob already exists in the driver; no serve code was written.

| arm | env added to the serve | what it changes | active on (post-handoff decisions) |
|---|---|---|--:|
| A | none | — | — |
| F faster ×1.15 | `JEV_ACC_VSET=1.15 JEV_ACC_ALAT=3.174` | ACC set speed = max(handoff speed, 1.15 × the plan's 2–4 s speed); curve limit at 2.4 × 1.15² m/s², so the curve speed scales by 1.15 as well | 0.67 |
| S slower ×0.85 | `JEV_ACC_VSET=0.85 JEV_ACC_ALAT=1.734` | the same at 0.85 | 0.56 |
| L route-seeking | `JEV_TTS=1 JEV_TTS_W=scripts/d3_tts_route.json` | among the model's own 16 TTS candidates (lateral ±0.5 / ±1 m, curvature gain 0.8–1.4, arc 0.9 / 1.1; lib/tts.py), serve the one whose mean distance to the fed route is ≥ 0.25 m smaller than the plan's (weights: w_route 1, all else 0, margin 0.25) | 0.28 |

Caveats on the arms:
- **F / S are asymmetric by construction.** The set speed is floored at the handoff speed, so S slows the drive only where the plan is above it.
  That holds on 97 % of launch decisions but only 29–31 % in stops and "other".
  - When the plan brakes, the stop / brake guard serves the model's own profile and neither knob acts.
  - A "×0.85 of the served plan" perturbation would need new serve logic, which this diagnostic was not to write.
- **L is the only path (geometry) perturbation.** A fixed lateral offset toward the turn does not exist as a switch.
  JEV_TURN_GAIN needs JEV_TRACK off, which changes the serve.

**Chain** (one submit, PRIO 8, `--after` links):
1. Smoke: 1 launch scene × 4.
2. chk1:
   - the knobs reached the driver: at the first post-handoff decision, where the states are still identical, v_set F / A = 1.1501 and S / A = 0.8499;
   - L wrote `tts` records;
   - A was bit-identical to d402.
3. c000: 25 scenes × 4.
4. chk2: 25 / 25 scored per arm, projected 2.3 card-hours.
5. The rest: 5 chunks × 4 arms.
6. Report.

Wall time was 01:19–02:27 box time, sharing the pool with round 3b.

## 1. Informative share and winners by class

| class | n | informative | mean spread | A best (ties count) | A best of informative | oracle mean | A mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| stop without lead | 38 | 6 (16 %) | 0.146 | 34 | 3 | 0.616 | 0.549 |
| sharp turn | 37 | 20 (54 %) | 0.434 | 23 | 8 | 0.697 | 0.505 |
| launch | 38 | 25 (66 %) | 0.420 | 19 | 6 | 0.938 | 0.790 |
| other | 36 | 12 (33 %) | 0.288 | 28 | 5 | 0.910 | 0.767 |
| all | 149 | 63 (42 %) | 0.322 | 104 | 22 | 0.789 | 0.652 |

Wins among informative scenes. Ties are split; sole wins are in brackets.

| class | A unperturbed | F faster | S slower | L route-seeking |
|---|--:|--:|--:|--:|
| stop without lead | 1.2 (0) | 1.3 (1) | 2.7 (2) | 0.8 (0) |
| sharp turn | 3.0 (0) | 8.2 (6) | 3.7 (2) | 5.2 (3) |
| launch | 2.3 (0) | **18.8 (16)** | 0.3 (0) | 3.5 (2) |
| other | 1.7 (0) | 3.0 (2) | 2.2 (0) | 5.2 (4) |
| all | 8.2 (0) | 31.3 (25) | 8.8 (4) | 14.7 (9) |

Columns:
- arm − A: paired difference, bootstrap over drive groups.
- up / down: scenes where the arm beats or trails A by ≥ 0.2.

| class | F − A | S − A | L − A | oracle − A | F up / down | S up / down | L up / down | zeros A / F / S / L |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| stop without lead | −0.025 [−0.105, +0.054] | +0.012 [−0.067, +0.090] | −0.026 [−0.079, +0.002] | +0.066 [+0.001, +0.145] | 1 / 2 | 2 / 1 | 0 / 1 | 17 / 18 / 16 / 18 |
| sharp turn | −0.017 [−0.107, +0.063] | −0.062 [−0.195, +0.075] | −0.014 [−0.158, +0.130] | +0.192 [+0.085, +0.316] | 1 / 2 | 3 / 8 | 4 / 5 | 16 / 18 / 16 / 16 |
| launch | **+0.124 [+0.062, +0.198]** | **−0.196 [−0.281, −0.108]** | −0.050 [−0.149, +0.053] | +0.148 [+0.081, +0.226] | 9 / 0 | 2 / 18 | 3 / 8 | 3 / 2 / 2 / 4 |
| other | −0.024 [−0.118, +0.056] | +0.048 [−0.005, +0.128] | +0.028 [−0.113, +0.169] | +0.143 [+0.049, +0.254] | 2 / 2 | 2 / 0 | 5 / 3 | 7 / 9 / 5 / 6 |
| all | +0.015 [−0.029, +0.058] | −0.050 [−0.101, −0.001] | −0.016 [−0.073, +0.041] | +0.137 [+0.092, +0.185] | 13 / 6 | 9 / 27 | 12 / 17 | 43 / 47 / 39 / 44 |

## 2. The consistent part is the outcome type, not the class

Splitting by A's own outcome makes the structure plain. Each cell gives scenes where the arm is ≥ 0.2 above / ≥ 0.2 below A, then the mean difference.

| A's outcome | scenes | informative | F | S | L | classes |
|---|--:|--:|---|---|---|---|
| full score (≥ 0.98) | 76 | 22 | 0 / 6, −0.079 | 0 / 11, −0.100 | 0 / 9, −0.113 | other 25, stop 20, launch 18, turn 13 |
| partial (0 < s < 0.98) | 30 | 26 | **11 / 0, +0.209** | 0 / 16, −0.242 | 2 / 8, −0.101 | launch 17, turn 8, other 4, stop 1 |
| zero | 43 | 15 | 2 / 0, +0.047 | 9 / 0, +0.170 | 10 / 0, +0.215 | stop 17, turn 16, other 7, launch 3 |

- **Partials: faster wins, in every class that has them.**
  - Rear-hit (17): F +0.257, 8 up / 0 down. Slow (13): F +0.147, 3 up / 0 down.
  - S loses 7 and 9 of them.
  - This is decision 381's under-planning, and the PAI-PARTIALS finding that rear-hit / slow partials are the plan's own missing speed, now seen on
    train scenes. On test281 the same serve knob (VSET 1.1 / 1.2) fixed partials 26 → 11–13 but added 4–5 corridor / offroad zeros
    (pai_partials.md). Here F adds the same kind of zeros: 6 of A's full scores go to 0 (offroad 3, corridor 2, AF-side 1), all on moving scenes.
    In launches F has none.
- **Zeros: a slower or route-seeking rollout sometimes saves them, scene by scene.**
  - S lifts 7 of A's zeros to ≥ 0.5 and L lifts 10 (turn 4, other 4, launch 2).
  - The same arms break full scores: S 4, L 8. L's breaks are all at-fault contacts (AF-front 4, AF-side 4). The route-seeking candidate moves the
    car into another agent, as in decision 324's TTS on nuPlan.
  - Which zeros are saved is not predictable from the class. In sharp turns the leave-one-out class rule loses 0.130.
- **Full scores: perturbations only break.** Here the informative rows are "A was right": its own served plan beat the perturbation (22 scenes).
- **Stops:** 17 corridor zeros (the overrun) under A. Only 3 change under any arm (F 1, S 2). Speed-target perturbations cannot reach a failure the
  model's plan makes while the ACC guard serves that plan.

## 3. Rows the informative rollouts give

Counts are 10 Hz inference decisions after the handoff. "From divergence" starts at the first decision where the winner's ego is > 5 cm from the
worst rollout's ego. Before that point the states are identical and no outcome contrast exists.

| class | winner decisions | from divergence | winner not A | winner not A, from divergence |
|---|--:|--:|--:|--:|
| stop without lead | 1 091 | 1 015 | 545 | 495 |
| sharp turn | 3 638 | 3 084 | 2 183 | 1 928 |
| launch | 4 543 | 2 722 | 3 452 | 2 005 |
| other | 2 184 | 2 056 | 1 274 | 1 180 |
| all | 11 456 | 8 877 | 7 454 | 5 608 |

By A's outcome (winner decisions / from divergence):
- full score 4 002 / 3 269 (all A-winner rows);
- partial 4 728 / 3 346;
- zero 2 726 / 2 262.

Launches diverge late (median decision 74, 7.4 s, around the human's launch), so only 60 % of their decisions carry contrast. Elsewhere it is about 85 %.

Scale:
- One round over the 719 non-held eligible scenes at this rate would give about 300 informative scenes, about 55 k winner decisions and about
  27 k non-A from-divergence decisions (≈ 5.4 k at 2 Hz).
- That is a few percent of the PAI human rows (136 k, d410) and of the d402 S pool (84 k).
- Cost: 4 arms × 719 scenes ≈ 10.7 card-hours.
- Frames: the rollouts kept no frame dumps. A winner's rollout is re-rolled bit-identically with `PAI_DUMP` (shown above), about 0.37 card-hours per
  100 scenes. d402's dumped rolls ran about 38 s a scene per stack, this run about 42 s on a shared pool, so the dump overhead is small.

## 4. What this means for S3

- **S3 has signal worth building only in its outcome-selected form.** Use per-scene winners chosen by score, not a per-class arm.
  - The class prior earns nothing outside launches.
  - The oracle's +0.137 is all scene-specific.
- **The one balanced, consistent signal is "plan faster where you end up rear-ended or slow".** That is the launch / partial class, and it is
  exactly what round 2–3 lost: R2J launches 33 → 21 / 37, d412 / d414. Human rows (S1) slow launches; F-winner rows push the other way.
  S3 is therefore the natural counterweight in the same recipe, not a separate line.
- **S3 does nothing for stops.** The stop overrun needs states and a perturbation of the plan itself (an arc scale of the model's plan inside the
  guard). That is new serve logic and was not tested here. Stops stay with S1 / S2.
- **The zero-fixing rows (S / L winners on lateral zeros) are real but small and mixed.** 15 scenes, about 2.3 k decisions.
  L's rollouts also show that the route-seeking candidates collide when they win nothing. Only the winners' rows would be used; loser rows are
  never targets.
- The sketch of how the rows enter the recipe is in [../plans/d3_awr_sketch.md](../plans/d3_awr_sketch.md) (Chinese working note).

## Caveats

- One rollout per arm and scene, one checkpoint (NTV0-L18-s0), one serve family.
- With bit-identical reads, "informative" measures sensitivity to the serve change, not read noise. Part of it is chaotic: perturbations that
  break full scores are the same phenomenon as the ones that save zeros.
- F / S move the ACC targets, not the model's plan. Only the set-speed part is expressible by a model under ACC: a faster plan raises
  v_set = max(v_hand, plan). The curve-limit part is not, because ACC applies its own curve limit.
- The class labels are the curation labels (nurec_select), not the rollout's own events.
- Scenes with frame dumps (d402) and without (this run) are bit-identical, so dumps do not perturb the drive.
