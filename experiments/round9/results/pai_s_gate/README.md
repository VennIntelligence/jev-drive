# PAI-S-GATE: S hinges on the policy's own PAI states, CPU gate (2026-10-22)

The question: can "PAI-state S rows" go into the next DAgger training as part of one joint arm? These are S consequence hinges on the
DAgger states: the road_pai raster plus the replay-agent boxes. Pre-registered before any number:
[../../plans/2026-10-22-pai-s-gate.md](../../plans/2026-10-22-pai-s-gate.md). Code: [../../scripts/pai_s_gate.py](../../scripts/pai_s_gate.py)
and the chain [../../scripts/pai_s_gate.sh](../../scripts/pai_s_gate.sh). Box outputs (3.0 GB, mostly rasters): `$DATA_DIR/runs/round9/pai_s_gate/`.
Here: [gate.json](gate.json) (every number below), [g1_offroad_events.csv](g1_offroad_events.csv) (per offroad scene) and
[road_build.json](road_build.json). CPU only: about 1.5 h wall and about 30 core-hours, mostly the two forwards (27 threads x 26 min each).

**Verdict: fail. Do not fold PAI-state S rows into the next DAgger arm.** G1 passes. G2 fails its label line, G3 fails both lines, G4
fails the speed and dose lines, and G5 fails narrowly on the agent term. The raster and the boxes are sound. What fails is learnability
and reach: only about 6 % of the candidate states carry a hinge that the model could learn. Those learnable-active rows hold only 16 % of
the hinge mass and are mostly below 13 m/s. At a dose of 16 rows they would give about 1 such row per batch. **Expected PAI-281 effect:
0 points per seed, plausible range -3 to +2 (-0.01 to +0.01 mean).** That is far below the two-seed detection limit of about 10 points.

## Material

- States: `on1_PAID`, DAgger iteration 1. The student was R5LC-L18-s0, on 718 train scenes: 110 981 rows, of which 90 300 are
  teacher-labelled candidates (v0 >= 2 m/s). Only these rows still have row files on disk, so only they have anchors for rasters.
  `on1_PAID2A` / `PAID2B` (iteration 2, R6DAG-s0 / s1 on their own states) have no row files any more. For them, agent and side are read
  from the stored GPU trainer-path forwards (every row). Road is not available there.
- Policy: R6DAG-L18-s1 (primary) and s0 (replicate). CPU forward through the trainer's own build with R6DAG's argv, on a seeded 30 % of
  PAID (33 294 rows, 27 042 candidates), about 26 min each.
- Terms at the trainer weights: agent 30 x hinge (margin 0.3), road 30 x road_pai hinge (margin 0.3), side 15 x nr1_lat side (buffer
  0.4). There is no onc / soft term because PAI has no lane-direction raster. The standalone agent / side terms used here are a slight
  superset of the trainer's (agent active 6.2 % vs 5.5 % on the forward sample; every trainer-active row is also active here; correlation
  0.997, mass ratio 1.035). The activity numbers below are therefore an upper bound.
- **Already in R6DAG:** the DAgger rows are `both` rows of a `mixed` pool with `--on-lon-all`. They already carry the agent and side
  hinges at ho_w 3. The only new term a PAI-S row would add is road_pai. Everything else would be more dose of terms that are already there.

## Gate lines

| line | registered threshold | measured (R6DAG-s1; s0) | result |
|:--|:--|:--|:-:|
| G1a human's log pose at t0 inside the student's raster | >= 97 % | 98.5 % (96.2 % with the 0.3 m margin; 99.7 % when the human is < 3 m away) | pass |
| G1b evaluator agreement: offroad zero scenes | >= 70 % read off or hinge-active | 71 / 71 scenes (100 %). The event decision itself is never in the pool (post-event rows are dropped), so this is read over the 2.5 s before. Per decision: served-plan hinge active 85.5 %, t0 off 4.1 %. 41 % of the scenes have t0 off somewhere in the window | pass |
| G1b control (scenes without an offroad event) | t0 off <= 2 % | 0.16 % (served-plan hinge active 9.6 %, the d418 false-trigger level) | pass |
| G1c t0 already off the surface (unlearnable) | <= 10 % of candidates | 0.19 % | pass |
| G2 boxes: coverage / t0 overlap without a collision | overlap <= 2 % | agent label 100 %, >= 1 valid box at t0 97.0 % (mean 22.5 boxes); overlap 0.03 % (2.8 % in the 30 collision scenes) | pass |
| G2 human log future also violates agent, of agent-active | <= 20 % | 1.8 % (corrected 10-22; this row first read 12.6 %; 12.7 %, which drew the human path from the student's position instead of the human's own; see ../hinge_audit/README.md) | pass |
| G2 teacher label also violates agent, of agent-active | <= 30 % | **76.4 %; 76.7 %** | **fail** |
| G3 learnable-active (LA) share of candidates | >= 10 % | **6.2 %; 6.2 %** (own states PAID2B / 2A, no road: 2.5 %; 2.6 %) | **fail** |
| G3 hinge mass on learnable rows | >= 50 % | **16.3 %; 15.5 %** (own states 25 %; 22 %) | **fail** |
| G4 LA rows above 13 m/s, of LA | >= 30 % | **23.3 %; 22.5 %** (own states 15.5 %; 12.5 %) | **fail** |
| G4 LA turns (> 30 deg) at 4-10 m/s, of LA | >= 5 % | 18.8 %; 17.7 % | pass |
| G4 LA rows per batch at a dose of 16 | >= 2 | **1.0; 1.0** (0.23 above 13 m/s; own states 0.39) | **fail** |
| G5 road step: 4 s arc / end speed change (median) | within +-2 % / +-0.3 m/s | +0.01 % / +0.0002 m/s | pass |
| G5 road step reduces the term / moves toward the teacher label | >= 90 % / >= 60 % | 99.6 % / 92.2 % (median cos 0.28) | pass |
| G5 agent step reduces the term / toward the label | >= 90 % / (reported) | **89.4 %** / 69.5 % (median term reduction only 20 %) | **fail (narrow)** |
| G5 road-active rows whose teacher label also violates (d420 pitfall) | reported | 64.6 %; the human's own log future also violates on 51.4 % | see below |

Activity by term on the candidates (R6DAG-s1 plan; PAID; [gate.json](gate.json) G3):

| term | active | LA | share of hinge mass | label also violates, of active |
|:--|--:|--:|--:|--:|
| road (road_pai) | 11.9 % | 4.1 % | 55 % | 65 % |
| agent | 4.2 % | 0.9 % | 41 % | 76 % |
| side | 3.1 % | 1.8 % | 3 % | 58 % |
| any | 17.0 % | 6.2 % | 100 % (16 % on learnable rows) | |

Reading.
1. **The raster is valid on the student's states (G1).** Placed in the student's raster, the human's own pose is inside 98.5 % of the
   time. Every evaluator offroad zero has an off-surface footprint or an active plan hinge in the 2.5 s before the event. Unrelated scenes
   read off-surface on 0.16 % of decisions. The boxes are aligned too (G2: 0.03 % t0 overlap outside collision scenes).
2. **Most of the hinge cannot be learned because of how the label is built, not because of the raster.** The DAgger teacher keeps the
   student's own arc lengths (d434). On 76 % of agent-active rows the recovery label overlaps the agent as well: a speed problem, which a
   shape-only hinge cannot fix. On 65 % of road-active rows the label is also within 0.3 m of the edge. On 51 % of them the human's own log
   path is too. These are narrow roads where the 0.3 m margin bites even for the human. A hinge there pushes away from the human's line,
   which is the d420 pattern (offroad 4 -> 7).
3. **Reach is no better than the nuPlan pool's where it matters.** Above 13 m/s the LA rows are 23 % on the R5LC states and 12-16 % on
   R6DAG's own states (agent / side only). The nuPlan S pool has about 1 %, so this is a gain, but it misses the 30 % line. The drift band
   (>= 17 m/s, straight, road or side active) is 7 % of LA, about 0.07 rows per batch at a dose of 16. Turns are well covered (19 % of LA;
   25 % in the turn-inside band). That is the one class the DAgger map teacher already targets with labels.
4. **G5: the gradient itself is safe.** The road step is purely lateral by construction: arc and end speed are unchanged, the term is
   cut by a median 84 %, and 92 % of steps point toward the teacher label. The step does not teach slowing. The risk sits in the rows
   (point 2), not in the step direction. The agent step reduces its term on only 89 % of rows and by a median of only 20 %. That is the
   shape-only ceiling on lead-overlap rows again (smix_value Q2).

## Expected PAI-281 effect

From smix_value Q3, the S-addressable block is about 19 points per seed. PAI states would add drift (4.5) to that. The new term is road
only, because agent and side are already on these rows in R6DAG. At 16 rows a batch it gives about 0.65 learnable-active road rows per
batch, and 0.07 in the drift band. The measured ceiling of the whole S schedule is about 5 % hinge movement on its active rows, and S-MIX's
x4.7 dose of learnable-active rows added 0 to it. The PAI S line is flat twice (d367 -0.010, d402 -0.007), and road_pai on PAI human rows
raised offroad (d420). Estimate: **0 points per seed, range -3 to +2.** The downside is the d420 kerb pattern on the 51 % of road-active
rows where the human is also within the margin.

## What would change this

If a later iteration of the PAI S idea is wanted, the two blockers are measurable on CPU with this script:
- a teacher whose longitudinal label also clears the lead. Then "label also violates agent" falls from 76 % and the agent rows become
  learnable.
- a road margin that the human's own path respects. For example, use per-row margin = min(0.3, the human's own clearance) instead of 0.3.
  Then the road-active rows the human also violates (51 %) stop pulling away from the human line.

Neither is in the current DAgger lane, and neither is worth GPU before this gate passes on them.
