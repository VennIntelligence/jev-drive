# INERTIA: does the plan follow the ego's speed or the stop / curve cue? (2026-10-20)

This is a diagnosis. Nothing was trained and nothing new was rolled out.
- Prereg: [../../plans/2026-10-20-inertia-prereg.md](../../plans/2026-10-20-inertia-prereg.md), committed at b8535a2e before any output.
  Supplement 1 (cad68d46) adds the receding-stop read. Supplement 2 (1910c39a) was written after the smoke run and before the full counterfactual
  run. It amends gate G0 to 2 fp16 steps, fixes a window bug in supplement 1, and adds an exploratory read.
- Code: `scripts/inertia.py` (`recede`, `cf`, `report`) and `scripts/inertia.sh` (pool jobs).
- Box output: `$DATA_DIR/runs/defect_fix/inertia/`.
- Compute: one GPU job of 4.5 min plus two CPU jobs. One seed throughout.
- Files: [report.md](report.md) (all cells), [report.json](report.json), [recede.csv](recede.csv) / [recede.json](recede.json) (per scene,
  with per-decision series), and [recede_prefix.csv](recede_prefix.csv) (supplement 1 before the window fix).

**Answer: H-INERTIA, for all three models.** The plan's length and speed follow the input speed almost as much on a stop approach or a
turn as on open road. Each +20 % of input speed moves the plan's stop or 4 s end point about 4 m further. That is 0.89x the response
with no stop ahead, and 0.92x the open-road response for turns. Turn path shape is not affected; only the speed is. About 90 % of the
effect comes through the frames, and only 5-9 % through NTV0's ego-input row. RB0-F-s0 behaves the same, so this comes from the base model.

In A0's stored decisions on the 23 overrun scenes, the plan never contains a stop on the approach. Its 4 s end point moves forward
with the ego (16 of 20 scenes; "between" in 4, held in place in none). Only 47db2eb6 holds its plan end at the stop point and then lets go.
In the 10 scenes where A0 does stop at the human's point, the planned stop is fixed in the world (5 fixed, 5 not judged, 0 receding).

**Stage 1 (NTV0's own states on the d381 scenes, and the human states H) confirms it.** The re-roll reproduces A0 in 54 of 54 scenes.
- Own states: stop ratio 0.94, turn ratio 0.98, open-road copy 0.80. On the 20 overrun scenes the plan's stop end lands a median 10 m past
  the human's stop point, and moves +6.8 m per +20 %.
- Human states: stop ratio 0.76, turn 0.89, copy 0.83. Inertia here too, but the served plan ends near the human's stop (+0.7 m), because
  the frames show the human slowing.
- Frames carry 91-94 % of the effect again.

**Supplement 4: the copying is already in shipped Cinque (P0).** P0 has no ego input. Its frame channel alone matches RB0-F-s0's total
response (stop: -0.14 m [-0.38, +0.12]) and exceeds RB0's frame channel (+0.67 m [+0.41, +0.97]). The base training moved part of the
copying into the ego row; it did not create or enlarge it.

## Data and what changed (read before the numbers)

**The states are not the d381 scenes.**
- The tokens of NTV0's own states on PAI 281 (the A0 rerun) were deleted in the 10-19 cleanup. The H read never dumped frames.
- So the counterfactual runs on P160: RSG-L18-s0's own rollouts on the 160 PAI practice scenes, under the A0 serve.
- P160 stores the rendered 10 Hz frames, the fed row and the logged route memory. These rows are in none of the three models' training pools.
- Rows: 2 225 decisions in 129 scenes, taken at 2 Hz, with the prereg filters. By window: STOP 187 (28 scenes), of which STOP-line is 52
  (6 scenes, underpowered); NS 212 (32); TURN 176 (22); OPEN 1 650 (111).

**Arms.**
- R_f: the fed speed history scaled by f (vx vy, ax, the three past poses re-spaced; d375's rule).
- Z: the speed entries zeroed.
- W_f: the slots taken from the rendered frames at t0 + f x SLOT_T, with f = 0.5 / 1.5 (exact at 10 Hz).
- B = W + R.

**Gate G0** (supplement 2's form): RSG's re-encoded forward against its logged plan. 99.6 % of rows are within 2 fp16 steps; p99 is 0.062 m
and max 0.096 m. It passes. W/B could be used on 94 % of rows.

**Kinematic reference.**
- If the plan extrapolates the input speed at a fixed deceleration, the stop distance goes as v², so +20 % speed gives +44 %.
- A pure copy of a 4 s arc gives +20 %, about 4.8 m at the 24 m arcs seen here.
- A cue-locked plan gives 0.

## Stop / turn / open road (NTV0-L18-s0 primary; CIs are 95 % scene-cluster bootstrap)

Response to input speed. ΔS is the plan's stop station (or its 4 s arc when the plan has no stop), linearised to each +20 % of speed.
G is d375's 4 s-arc gain: 1 means the plan continues the input speed exactly.

| model | window | ΔS row only (R) | ΔS frames only (W) | ΔS both (B) | G_B | G_B / G_ref | plan has a stop (F) |
|---|---|--:|--:|--:|--:|--:|--:|
| **NTV0-L18-s0** | STOP (8-40 m, no lead) | +0.30 [+0.24, +0.37] | +3.83 [+2.92, +4.88] | **+4.12 [+3.17, +5.20]** | 0.72 | **0.89** (vs NS) | 0.03 |
| NTV0-L18-s0 | STOP-line (6 scenes) | +0.34 | +3.46 | +3.75 [+2.36, +6.69] | 0.74 | | 0.00 |
| NTV0-L18-s0 | NS (no stop within 8 s) | +0.33 | +6.03 | +6.44 | 0.80 | | 0.00 |
| NTV0-L18-s0 | TURN (>= 30° within 40 m) | +0.29 | +3.43 | +3.75 | 0.73 | **0.92** (vs OPEN) | 0.00 |
| NTV0-L18-s0 | OPEN | +0.49 | +9.52 | +10.01 | 0.79 | | 0.00 |
| RB0-F-s0 | STOP | +0.89 [+0.63, +1.16] | +3.37 | +4.18 [+3.18, +5.30] | 0.73 | 0.88 | 0.04 |
| RB0-F-s0 | TURN | +1.06 | +2.96 | +3.94 | 0.77 | 0.94 | 0.01 |
| RSG-L18-s0 | STOP | +0.91 | +3.37 | +4.21 | 0.73 | 0.89 | 0.04 |
| RSG-L18-s0 | TURN | +1.07 | +2.92 | +3.92 | 0.76 | 0.94 | 0.00 |

**Other quantities (NTV0).**
- Δ(share of plans with a stop) per +20 %: 0.000 (R) / +0.008 (B) on STOP. The plan almost never contains a stop on the approach (3 %),
  at any input scale from 0.6 to 1.4. It does not lose a stop at higher speed because it never has one.
- With the input as served, the plan's stop / 4 s end sits a median 0.5 m short of the human's stop point (2.0 m short on STOP-line).
  But it moves 4 m per +20 % of input speed.
- Turn path shape is locked in space: the heading at 20 m of arc moves -0.51° per +20 % (B; RB0 -0.78, RSG -0.66). Line: -2°.
- The turn-window speed is not locked: v24 slope 0.68 (B), against 0.78 on OPEN.
- OPEN v24 slope is 0.78 (B), a copy. The plan's v24 minus the human's at the same station is +0.27 m/s on these RSG states.
- Z (speed entries zeroed): the STOP plan shortens by 2.8 m (RB0 / RSG 4.6 m), and the share of plans with a stop is 0.04.

**Registered verdicts.**

| | STOP | TURN | OPEN | row share of the effect (R / B at ±50 %) |
|---|---|---|---|---|
| NTV0-L18-s0 | **inertia** (ΔS_B +4.1 > 3 m; aux 0.89 >= 0.7) | **inertia** (ratio 0.92 >= 0.7; Δhead20 -0.5° not <= -2°) | **copy** (0.78) | STOP 0.08, TURN 0.09, OPEN 0.05 |
| RB0-F-s0 | inertia (0.88) | inertia (0.94) | copy | STOP 0.19, TURN 0.26, OPEN 0.12 |
| RSG-L18-s0 | inertia (0.89) | inertia (0.94) | copy | STOP 0.20, TURN 0.26, OPEN 0.12 |

- **Base:** RB0's verdicts equal NTV0's, and |ΔS_B difference| = 0.06 m < 1 m. By the rule, the effect is **already in the base**.
- **Channels:** the JT / NTV stage cut the row channel to about a third (G_R 0.16 to 0.05). The frame channel rose (G_W 0.59 to 0.67), and the
  total stays 0.72-0.73. This is the same split as d375 on nuPlan (row 0.065, frames 0.66).

## Supplement 1: does A0's planned stop recede? (stored 10 Hz plans, the 23 overrun scenes)

**Data.**
- 20 of the 23 scenes have an approach. In 095cf563 and 4e4a2a51 no decision has the human's stop within 60 m. 3fe2bac8 drops out with the window fix:
  its stop point is at the start of its log, about 0.2 m in, so it has no approach.
- Controls. The first six (now `control_v1`) were chosen at scene level: the human ends still, no lead at the rest point, A0 scores 1.0.
  Corrected 10-20 (supplement 3): they were not stops by the model.
- The proper controls are 10 scenes where A0 scores 1.0, the human stops, there is no lead, and A0 rests for >= 2 s within [-8, +4] m of
  the human's stop point (`scenes_s1.csv` stop_ctrl).

**Registered read.**
- Overrun scenes: a plan with a stop point (< 0.5 m/s) appears in fewer than 10 % of approach decisions in **20 / 20**. So the class is
  **"never appears"**: 0 receding, 0 fixed, 0 let-go.
- **Proper stop controls (10): 5 fixed (slopes -0.36 to +0.25), 5 not judged, 0 receding, 0 never.** When A0 stops at the human's point,
  its planned stop stays in place in the world. Their approach speed is a median 3.7 m/s, against 10.6 m/s in the overrun scenes.
- `control_v1` (6): 6 never.
- Before the window fix, the read was 19 / 21 "never" and 2 not judged, in [recede_prefix.csv](recede_prefix.csv). The bug: past the log end,
  the ego's station is clamped at H.

**Exploratory read (supplement 2): the plan's 4 s end point E = s_e + A4 against the ego's station.**
- **16 / 20 overrun scenes: the end point moves with the ego** (slope >= 0.7).
- **4 / 20 are "between"** (127c4f85, 47db2eb6, 62cfe9d3, fc9fde87; slope 0.43-0.66).
- **0 / 20 are held in place.**
- Proper stop controls: 6 pinned, 1 between, 1 recedes (slope 0.71, at the line), 2 too short to judge. `control_v1`: 6 / 6 recede.

Median over overrun decisions, binned by the ego's distance to H:

| distance to H (m) | 40-60 | 30-40 | 20-30 | 10-20 | 5-10 | 0-5 |
|---|--:|--:|--:|--:|--:|--:|
| ego speed (m/s) | 10.8 | 10.6 | 10.5 | 10.1 | 9.6 | 8.9 |
| E - H (m) | -10.9 | +0.6 | +8.5 | +21.5 | +27.9 | +30.0 |
| plan speed at 3.5-4 s (m/s) | 7.7 | 7.0 | 5.6 | 8.3 | 8.1 | 8.0 |

From 30 m out, the 4 s plan already reaches past the human's stop. The overshoot grows as the ego closes in, and the plan's terminal
speed rises again inside 20 m.

**47db2eb6 is the only scene that holds and then lets go** (plan end within ±3 m of H at terminal speed < 2 m/s for >= 1 s, then > 5 m past H):
- About 28-12 m out, the end sits at H + 0..3 m, and terminal speed falls 5.3 to 0.9 m/s.
- From about 11 m out it lets go: E - H goes +8 to +22 m, and terminal speed climbs back to 6 m/s.
- fc9fde87 dips to 1.7 m/s about 20 m out and then lets go the same way. It is not counted, because it holds for less than 1 s.
- The other scenes never hold. Their minimum terminal speed is 3-14 m/s, except 127c4f85, 62cfe9d3 and 9b6c68b4 at 1.5-1.9 m/s.

**Correction (10-20, supplement 3).** This paragraph first said that "the controls are not 'the model stops' cases" and that "A0 has no clean
no-lead stop control". The first half holds for the six scene-level controls (`control_v1`): their ego reaches H (the log end) at 5-9 m/s, and
they score 1.0 by the scorer's end-of-route handling. The second half was wrong. My scene-level filter missed the 10 scenes above, in which
A0 does stop at the human's point.
- Per-scene series (decision k, s_e, stop station, E, terminal speed, H): [recede.json](recede.json) `series`.

## Reading

**Stop.** The 4 s plan on a stop approach is about the human's stop distance long at served speed, but it is not anchored to the stop.
It stretches with the speed the frames show, at 0.89 of its no-stop response. The plan's terminal speed stays above 0.5 m/s, so the stop is
reached only by replanning again and again.
- On the human's states (d381 H), the frames show a decelerating human, and the plan stops near the human.
- On its own states, the ego arrives 1.41x faster. The frames encode that speed, and the plan stretches past the stop point. That is the
  receding end point of supplement 1.
- d386's finding fits this: the stop cue is readable in the tokens, but the head weights speed continuation over it.

**Turn.** The path shape is spatial, but the speed through the turn is copied from the input (0.92 of open road). Under-turning at PAI turns
is then a speed problem: entry too fast for the controller and MPC. This is consistent with d403.

**Which channel.** The ego-input row carries 5-9 % for NTV0 and 12-26 % for RB0 / RSG. The frames carry the rest.

## Implication for training (proposal, not registered)

1. **Ego-row perturbation alone will not move this.** Speed noise, dropping the row speed, and vhist-type changes act on 5-9 % of NTV0's
   response, as d320 / d343 / d375 already showed for slowness.
2. **The augmentation must change the frame-encoded speed while keeping the label's stop point fixed.**
   - Time-stretch the slot frames at f ∈ [0.6, 1.5]. This is exact on PAI row files (10 Hz renders; P160 / PPAI and any own-state rows).
     On nuPlan, warp along a scaled track (d375's synthesizer).
   - Scale the row by the same f.
   - Rebuild the label as the human's path, with a feasible speed profile from v x f that still stops at the human's world stop station
     (and slows to the human's speed at the same curve stations).
   - This manufactures the "arriving too fast" states that the pool lacks. That is why STOPPT's one-sided term could not fire (d386:
     0.4-1 % of rows).
   - A cheap offline check first: on P160 STOP rows, the trained head's ΔS_B per +20 % should fall from about 4 m toward below 1 m, while
     G on OPEN stays near 0.8, so free-road speed copying is kept.
3. **Turns:** the same stretch, with curve-station speed targets. The path-shape term is not needed (Δhead20 is already spatial).

## Limits and uncertainty

- **P160 is RSG's states on practice scenes.** There, STOP-line has only 6 scenes (underpowered), and TURN 22 (20 with W/B).
- **Stage 1 covers NTV0's own states and the human states on the d381 scenes.** Own-state TURN has 14 scenes (underpowered), and the group
  splits are descriptive.
- **The frame stretch is out of distribution.** Pair spacings of 0.1 / 0.3 s were never trained. Stretching also speeds up or slows down
  other agents and signals: there are none in STOP / OPEN by the windows, but lights still change. Its size matches d375's in-distribution
  warp on nuPlan (frame gain 0.66 there, 0.67 here), which supports it, but this is still a single open-loop decision.
- **The STOP line ΔS > 3 m partly follows mechanically from censoring.** 97 % of plans have no stop, so S is the 4 s arc. The registered
  auxiliary ratio (0.89) is the more informative number, and it gives the same verdict.
- Supplement 1's first controls were wrong. The proper ones are above. The overrun "never appears" result rests on the < 0.5 m/s stop
  definition. The end-point read is exploratory and was added after the registered result.

## Stage 1: NTV0's own states and the human states on the d381 scenes (supplement 3)

**What ran.**
- The main session approved Stage 1 through the coordinator.
- 54 scenes (`scenes_s1.csv`): 23 overrun, 11 straight-at-turn, 10 stop controls, 10 turn controls (A0 1.0, stratum turn, >= 45°).
- Own read: A0's exact configuration (VCONT + LEAD + ACC + GAP 0.7,2.7, steering weight 10, CONC 4, ALPASIM_FAST=1) plus
  `PAI_DUMP=100000 KEEP_ASL=1`, at 14 scenes per chunk.
- H read: `pai_gap.sh`'s log-replay configuration on the same chunk lists, with the same dumps.
- Then `onp_rows`, then `inertia.py cf` unchanged, with `--gate-tag NTV0-L18-s0`.
- Compute: 8 read stacks of about 10.5 min each (1.4 stack-hours, about 0.5 card-hours), plus two 4-min cf jobs.
- Cleanup: the rollout.asl files and frame dumps were deleted after the row files were built. Kept: row files OWN 3.0 GB and HUM 2.9 GB in
  `$DATA_DIR/runs/defect_fix/inertia/s1/rows/` until this result is accepted; the read dirs are now 73 / 80 MB.

**Gates.**
- Parity on 47db2eb6 alone: score equal, and the plan is bit-identical over all 199 decisions (max difference 0.0 m).
- Full own read: 54 / 54 scenes have the same score as A0 and a plan within 0.15 m over the first 50 decisions (`parity_own.json`).
  So the overrun and turn groups are A0's own failures.
- G0: own rows 99.8 % within 2 fp16 steps (568 rows), H rows 99.6 % (791 rows). Both pass.

**Registered cells (NTV0-L18-s0; all 54 scenes pooled by window).** Files: [report_s1_own.md](report_s1_own.md), [report_s1_hum.md](report_s1_hum.md).

| states | window | rows (scenes) | ΔS_R | ΔS_W | ΔS_B per +20 % | G_B | ratio | plan stop / end - human stop (F, median) | verdict |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| own | STOP | 167 (28) | +0.29 | +5.18 | **+5.50 [+4.48, +6.58]** | 0.78 | **0.94** | **+3.5 m** | inertia |
| own | STOP-line | 79 (12) | +0.33 | +4.72 | +5.06 | 0.75 | | +2.8 m | |
| own | TURN | 100 (14, underpowered) | +0.36 | +4.55 | +4.87 | 0.78 | 0.98 | | inertia (underpowered) |
| own | OPEN | 161 (27) | +0.32 | +6.81 | +7.20 | 0.80 | slope 0.78 | | copy |
| H | STOP | 198 (28) | +0.21 | +3.70 | **+3.89 [+3.37, +4.45]** | 0.65 | **0.76** | **+0.6 m** | inertia |
| H | STOP-line | 85 (12) | +0.22 | +3.74 | +3.97 | 0.66 | | -0.2 m | |
| H | TURN | 144 (18) | +0.33 | +3.38 | +3.68 | 0.75 | 0.89 | | inertia |
| H | OPEN | 272 (28) | +0.39 | +5.71 | +6.18 | 0.84 | slope 0.81 | | copy |

- Row share for NTV0: 6-7 % on own states, 6-9 % on H. RB0 / RSG: 14-25 %.
- The verdicts are the same for RB0-F-s0 and RSG-L18-s0 on both state sets.
- Plan v24 minus the human's at the same station: own STOP +1.99 m/s, H STOP +0.54 m/s.

**By scene group (descriptive, not registered; NTV0, B channel, STOP / TURN windows with W/B usable).** File: `extra_p0_groups.json`.

| states | group | STOP rows (scenes) | ΔS_B per +20 % | G_B | plan end - human stop | ego speed |
|---|---|--:|--:|--:|--:|--:|
| own | overrun | 109 (20) | **+6.8 [+5.7, +8.0]** | **0.83** | **+10.1 m** | 10.3 m/s |
| own | stop control | 46 (8) | +2.5 [+1.7, +3.1] | 0.55 | -0.4 m | 5.4 m/s |
| H | overrun scenes | 137 (20) | +4.3 [+3.9, +5.0] | 0.67 | +0.7 m | 8.1 m/s |
| H | stop control scenes | 49 (8) | +2.6 [+1.8, +3.4] | 0.57 | +0.3 m | 5.0 m/s |

| states | TURN group | rows (scenes) | ΔS_B | G_B |
|---|---|--:|--:|--:|
| own | turn fail | 43 (6) | +4.1 | 0.71 |
| own | turn control | 42 (5) | +5.3 | 0.83 |
| H | turn fail | 74 (10) | +3.1 | 0.73 |
| H | turn control | 49 (5) | +4.6 | 0.76 |

**Reading.**
- On its own overrun approaches, NTV0 at about 10 m/s plans to end about 10 m past the human's stop point, and moves that end +6.8 m per
  +20 % of speed, which is near pure continuation (G 0.83).
- On the same scenes from human states, the frames show the human's slower, braking motion. The plan is still speed-led (G 0.67), but
  lands at the human's point.
- Where A0 stops (the controls), it approaches at about 5 m/s, and the copying is weaker (G 0.55, still about 2.5 m per +20 %).
- So the overruns are the high-speed, high-copy end of one speed-following plan. They are not a different mechanism.
- Turn failures do not copy more than well-taken turns (G 0.71 against 0.83 on own states; 5-6 scenes each). The under-turn is not explained
  by a larger speed copy at the turn. It goes with the ego's higher entry speed, which d381 / d403 already showed.

## Supplement 4: shipped Cinque P0 against RB0-F-s0 (frame channel, P160 rows)

**What ran.**
- `cf --tags P0,RB0-F-s0,RSG-L18-s0`. P0's own network encoded the frames, and G0 matched the earlier run exactly (99.6 %, the same
  p99 / max), so the encoders are the same.
- Frame stretch at the registered 0.5 / 1.5x, linearised to each +20 %. The coordinator asked for 0.6-1.4x, but only 0.5 / 1.5 are exact
  at 10 Hz.
- File: [report_p160_p0.md](report_p160_p0.md); paired differences in `extra_p0_groups.json`.

| window | P0 ΔS_W | P0 G_W | RB0 ΔS_W / ΔS_B | RB0 G_W / G_B | P0 W - RB0 W (ΔS) | P0 W - RB0 B (ΔS) | P0 W - RB0 B (G) |
|---|--:|--:|--:|--:|--:|--:|--:|
| STOP | +4.04 | 0.70 | +3.37 / +4.18 | 0.59 / 0.73 | +0.67 [+0.41, +0.97] | -0.14 [-0.38, +0.12] | -0.02 [-0.07, +0.02] |
| NS | +6.46 | 0.81 | +5.26 / +6.65 | 0.66 / 0.83 | +1.20 [+0.90, +1.49] | -0.19 [-0.40, +0.03] | -0.02 [-0.05, +0.00] |
| TURN | +3.83 | 0.75 | +2.96 / +3.94 | 0.58 / 0.77 | +0.87 [+0.70, +1.04] | -0.10 [-0.31, +0.11] | -0.02 [-0.06, +0.02] |
| OPEN | +10.86 | 0.86 | +8.98 / +10.26 | 0.71 / 0.82 | +1.89 [+1.68, +2.09] | +0.60 [+0.41, +0.78] | +0.05 [+0.03, +0.06] |

**Registered verdicts for P0:** STOP inertia (ratio 0.87), TURN inertia (0.87, Δhead20 -0.04°), OPEN copy (0.84).

**Reading against supplement 4's three cases.**
- The registered W-against-W difference is not in any of the three cases as written: P0 is significantly larger than RB0, not smaller.
- The nearest reading is **"already in shipped Cinque"**. P0 copies speed by the registered rules, and RB0's total equals P0's frame-only
  response (the CI includes 0 on STOP / NS / TURN).
- RB0's training moved about 20 % of the copying from the frames into the ego row. It neither created it nor enlarged it. On open road,
  P0 copies slightly more (+0.05 G).
- So the 2 Hz keyframe + warp training is not the origin.
- The lineage gap (P0 native AF-front 16 against NTV0's 43, d405) therefore does not come from more speed copying in NTV0's plan. Per d405 it
  more likely comes from NTV0's plan not following leads. This probe did not test leads: STOP / OPEN exclude them.
- P0's served plan also sits further short of the human's stop at served speed (-2.9 m against RB0 -0.8 m on P160 STOP).

**Effect on the training proposal.** The proposal stands, and is stronger. The speed copy is a property of the pretrained openpilot plan.
It is not something our base training added that could be removed. It has to be trained out on purpose, with frame-stretch states and
stop-preserving labels.
