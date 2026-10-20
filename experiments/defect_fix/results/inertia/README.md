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
- Six controls: the human ends still, there is no lead at the rest point, and A0 scores 1.0.

**Registered read.**
- Plans with a stop point (< 0.5 m/s) appear in fewer than 10 % of approach decisions in **20 / 20 overrun scenes and 6 / 6 controls**.
- So the class is **"never appears"** everywhere: 0 receding, 0 fixed, 0 let-go.
- Before the window fix, the read was 19 / 21 "never" and 2 not judged, in [recede_prefix.csv](recede_prefix.csv). The bug: past the log end,
  the ego's station is clamped at H.

**Exploratory read (supplement 2): the plan's 4 s end point E = s_e + A4 against the ego's station.**
- **16 / 20 overrun scenes: the end point moves with the ego** (slope >= 0.7).
- **4 / 20 are "between"** (127c4f85, 47db2eb6, 62cfe9d3, fc9fde87; slope 0.43-0.66).
- **0 / 20 are held in place.**

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

**The controls are not "the model stops" cases.**
- In all 6 controls the ego reaches H (the log end) at 5-9 m/s, and the plan end recedes there too (6 / 6).
- Their score of 1.0 comes from the scorer's end-of-route handling (probably time or log end), not from braking.
- So A0 has no clean no-lead stop control.
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

- **Not NTV0's states on the 23 / 11 failure scenes.** These are RSG's states on practice scenes. The cue strengths and the speed mix may
  differ. STOP-line has only 6 scenes (underpowered). TURN has 22 scenes, and its W/B arms 20.
- **The frame stretch is out of distribution.** Pair spacings of 0.1 / 0.3 s were never trained. Stretching also speeds up or slows down
  other agents and signals: there are none in STOP / OPEN by the windows, but lights still change. Its size matches d375's in-distribution
  warp on nuPlan (frame gain 0.66 there, 0.67 here), which supports it, but this is still a single open-loop decision.
- **The STOP line ΔS > 3 m partly follows mechanically from censoring.** 97 % of plans have no stop, so S is the 4 s arc. The registered
  auxiliary ratio (0.89) is the more informative number, and it gives the same verdict.
- **Supplement 1's controls are invalid as controls** (see above). Its "never appears" result rests on the < 0.5 m/s stop definition. The
  end-point read is exploratory and was added after the registered result.

## Next (needs the user's word)

**Stage 1.** Re-roll A0 (NTV0-L18-s0 + ACC + GAP, `KEEP_ASL=1 PAI_DUMP=100000`, `pai_box.sh` unchanged), then `onp_rows` to get row
files in P160's format. Then run `inertia.py cf` with `ROWS` pointed at them.
- Precedent, 10-19 A0R: 8 chunks took about 70 min wall (about 2 card-hours at 4 reads per card); rows took 6 min (24 CPU) and the cache 6 min.
  It needed about 26 GB transient disk before cleanup.
- A list of only the 23 + 11 + controls scenes is about 2 chunks: about 0.5 card-hours and about 7 GB.
- The H states (d381) need the same dump on `pai_gap.sh`'s H read.
