# STOP-GAP: rest gap to the lead and creep at standstill under ACC + GAP (PAI 281, 2026-10-22)

The question came from watching rollouts. Behind a stopped lead, our car seems to stop much closer than the human did, and once stopped it
keeps creeping forward. Is this costing points, and is a serve change (`JEV_GAP` stop distance) or a model fix warranted?

This is a diagnosis only, on CPU from stored reads: no rollouts, no training. The reads are the five stored PAI 281 reads under the
submission serve ACC + GAP 0.7,2.7 with Harmonizer off: R6DAG-L18 s0 / s1 / s2, R8K-L18-s2 and NTV0-L18-s0 (`p1021a`). For each read the
code uses its rf_tax pkl (ego at 0.1 s, the replayed logged traffic at 0.5 s), its driver `drive.jsonl` (the served lead reading and ACC
state of every decision) and its pai_fix facts. Code: [scripts/stop_gap.py](../../scripts/stop_gap.py). `scan` runs on the box in about
1 min and writes to `$DATA_DIR/runs/round9/stop_gap/`; `report` writes [tables.md](tables.md) (every number below),
[rest_gap_scenes.csv](rest_gap_scenes.csv), [counterfactual.csv](counterfactual.csv) and [summary.json](summary.json).

## Definitions

- **Standstill onset:** the first time after the handoff that the ego stays below 0.2 m/s for at least 1 s.
- **Lead and true gap:** the lead is the nearest vehicle in the path within 30 m along the heading line. The true gap runs from our front
  bumper to the lead's rear. This is the same geometry pai_gap uses for the human's `rest_lead_gap`.
- **Reading:** `fix.lead.d`, the model's lead distance minus the camera-to-front offset, as the lead MPC consumes it.
- **Creep:** the ego's path length from the onset until the lead has moved more than 1 m, or until the event or the end of the rollout.
  Each 0.1 s of motion is charged to the served term that was binding at the latest decision:
  - `lead`: the lead MPC cap.
  - `model`: the stop guard passes the model's own profile through.
  - `ref0`: the served 2 s arc is below 0.1 m, so the reference does not move and any motion comes from the controller.
  - `set` / `ramp`: the stop guard has released.

## Findings

1. **The human's own rest gap is smaller than 6 m.** The 55-scene figure (median 6.0 m) counts every scene with a vehicle within 15 m at
   the end of the log. In 28 of those 55 the human is still moving at that point. In the 27 scenes where the human really ends still
   behind a lead, its gap is p10 / p50 / p90 = 2.1 / 4.05 / 6.1 m.
2. **We do stop closer, by about 1 m.** We have 89 scene-reads (21 scenes) where the human ends still behind a lead and we stop behind one
   too. Our true gap at the onset is 4.1 m (median). Our minimum before the lead moves is 3.8 m (p10 2.2), against the human's 4.5 m in
   the same scenes. The paired difference is -1.2 m (p10 -3.2, p90 +2.6). We rest about 1 m ahead of the human's rest station (median
   +0.97 m).
   - Counting only stops from motion (59 scene-reads), the difference is -1.0 m, and only 3 of 59 come within 2 m of the lead (1 within
     1 m).
   - The clearest cases (human / ours): 2bbf1efc 6.1 / 2.6 m, 7d16a9a9 6.2 / 2.9 m, 5fcb2ba7 4.05 / 2.0 m.
3. **The lead head cannot see distance at standstill.** While stopped (10 411 decisions), the reading is almost flat at 2.7-3.9 m for any
   true gap from 0 to 6 m (median by true-gap bin 3.0 / 2.7 / 2.9 / 3.4 / 3.8 / 3.9 m). It reads 1.5-2.5 m long below 2 m and 0.5-1.5 m
   short at 4-6 m.
   - A reading of 2.7 m means a true 2.6 m on average, but anywhere from 1.3 to 3.7 m (p10-p90).
   - It also reads the stopped lead as moving away (median +0.10 m/s, p90 +0.45).
   - Decision 330 found the opposite bias for moving leads, which read ~1 m short. Stop distance 2.7 m therefore does not mean "2.7 m" at
     rest.
4. **The creep is real but small, and the lead MPC allows it.**
   - Path length before the lead moves: median 0.34 m for stops from motion (91 episodes, p90 2.5 m) and 0.46 m standing at the handoff
     or in a queue (107 episodes, p90 1.0 m). More than 0.5 m in 38 % / 47 % of episodes.
   - The gap closes by a median 0.24 m.
   - Of the creep metres, 76 % (stops from motion) and 94 % (queue) are charged to the lead MPC cap. The stop guard's model pass-through
     takes 13 % / 2 %, and controller motion on a reference that does not move takes 11 % / 4 %. The stop guard never released inside a
     creep window.
   - The mechanism: the model itself plans a creep at standstill (its own 2 s arc is 0.54 m median). The lead cap trims that only to what
     a reading of about 2.7-3.3 m, with a lead read as moving away at about 0.1 m/s, allows, which is 0.20 m served per 2 s and an ego
     speed of 0.08 m/s.
   - Stop lines without a lead: only 2 stops from motion, each creeping about 0.3 m.
5. **The losses are not where the clips suggest.**
   - **AF-front zeros (19 over 5 reads):**
     - 1 is a creep-after-standstill contact: R8K-s2 dbfd6c72, a pedestrian contact at 0.04 m/s after creeping 0.21 m with the lead not
       moved.
     - 1 is a slow approach with a close-range over-read: NTV0 2c0ef536, at 0.85 m/s, reading +1.0 m long.
     - 5 are oncoming, 7 have the partner off the heading line (cut-in / lateral: 3ce13eed on every read, 52bd89ce, 6e6cbe67), and 5
       have no reading or a long reading at speed (0feb3afa at 28 m/s, 09528162 at a turn, 68e53160).
     - None of the 5 R6DAG-family zeros comes from creep, and only 1 of the 19 overall.
   - **Stop-overshoot zeros (`cor_route_end`, 102 over 5 reads, 25 scenes):** none came to a standstill first (0 / 102). The ego passes
     the human's rest point at a median 7.4 m/s (p10 3.8, p90 16), ~4 m past it. This is decision 406's no-stop problem, not creep.
   - **Rear-hit partials:** across the 76 stop-from-motion rests (human ends still), none is rear-ended, wherever the ego rests. That
     includes 15 scene-reads that rest behind the human's rest point with a logged follower within 15 m. The 16 rear-hit partials in
     human-ends-still scenes are 15 launch scenes where the ego lags 6-11 m (cbb22323, d641893f, f18c6340). They have nothing to do with
     the rest gap.
   - A closer rest is therefore free today: it keeps us ahead of the replay follower.

## Counterfactual (offline)

The model of what each arm would do:

- **Larger stop distance X:** because of finding 3, X is translated into a true rest gap GX through the standstill reading-to-truth table:
  X = 4 gives about 4.5 m, 5 about 5.5 m and 6 about 6.8 m, each ±0.7 m. Every stop from motion behind a lead moves back by
  max(0, GX - our gap).
- **New rear-hit risk:** counted when a logged follower exists and our new station is more than (follower gap - 0.5 m) behind the human's
  rest point. On the current reads this rule predicts 23 of 27 observed rear-hits, with 7 false alarms over 114 rests.
- **Cost of a new rear-hit:** the mean rear-hit partial, 0.384.

| arm | zeros plausibly fixed (5 reads) | new rear-hit risks (5 reads, low / p50 / high GX) | PAI-281 points per seed |
|---|---|---|---|
| stop distance 4 m | 1 (dbfd6c72) | 0 / 0 / 0 | +0.2 (range 0 to +0.2) |
| stop distance 5 m | 2 (+ 2c0ef536, NTV0 only) | 0 / 2 / 10 (a309e228, d5f732f6) | +0.25 (range -0.4 to +0.4) |
| stop distance 6 m | 2 | 10 / 10 / 15 | -0.4 (range -0.75 to -0.4) |
| no creep at standstill until the lead moves | 1 (dbfd6c72) | 0 | +0.2 (range 0 to +0.2) |

**Uncertainty.** Every gain rests on one or two events out of five reads. A single event is +0.2 per seed, below the seed-to-seed noise of
a full 281 read (decision 434: R6DAG s0 vs s1 differ by 0.04 of the mean, about 11 points).

**What the table leaves out: a larger X also changes driving while moving.** The desired distance is v^2/5 + 0.7 v + X at every speed, so
X = 4-5 adds 1.3-2.3 m to the following gap that decision 330 fitted to the human's. That costs progress, queue partials and "rear-ended
while stopping" partials (13 scenes, 4.3 points per seed in the loss map). Nothing offline can price it, and it is more likely to be
negative than the +0.2 is to be positive.

## Recommendation

1. **Serve parameter: leave `JEV_GAP=0.7,2.7` alone.** Stopping about 1 m closer than the human costs nothing on PAI 281 today: 0 rest
   rear-hits, 0 overshoots caused by creep, 1 near-contact in 59 stops from motion. At best a larger stop distance buys about 0.2 points
   per seed from one scene, pays an unmeasured following cost while moving, and starts adding rear-hits from 5 m on. The user's
   observation is correct as geometry, but it does not show up as score.
2. **Creep: if anything, a serve-side standstill hold, not a gap change.**
   - The rule: when the ego is below 0.3 m/s, a lead is present, its reading is under 4 m and its read speed is under 0.5 m/s, serve
     speed 0.
   - It removes the about 0.4 m creep and the dbfd6c72 contact (about +0.2 per seed) with no rest-position rear-hit risk.
   - Its risk is launch latency if the lead's speed reading is late, and the launch line is sensitive (decisions 419-428). Worth building
     only to fold into a candidate that is being read anyway.
3. **Model side:** the creep is the model's own plan at standstill (0.54 m in 2 s behind a stopped lead). A zero-arc label at standstill
   behind a stationary lead on the DAgger rows is the principled fix. It is cheap enough to ride on the next row build, but it is not
   worth an arm on its own.
4. **Paired targeted read: not worth it on its own.** If it is bundled with another read, use:
   - Scenes: the 21 rest-behind-lead scenes of [rest_gap_scenes.csv](rest_gap_scenes.csv), plus dbfd6c72, 2c0ef536, a309e228 and
     d5f732f6, the 13 rear-move partial scenes of the loss map, and pai_fix's G-LAUNCH and G-LEAD guards.
   - Arms: (a) ACC + GAP 0.7,2.7, (b) GAP 0.7,4.0, (c) 0.7,2.7 plus the standstill hold.
   - Seeds: R6DAG-L18 s1 and s2.
   - Gate: no new rear-hit or launch loss on the guards.
5. **The largest stop-related block is unaffected by all of this.** That is the overshoot zeros, which never stop: 19-20 per seed, at
   cruise-like speed. They remain the stop-line teacher question of the loss map (row 5).
