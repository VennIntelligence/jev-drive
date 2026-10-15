# SPIN v9: handoff guard on the PAI control path (serve side, no training)

Prereg [plans/2026-10-15-spin-v9-prereg.md](../../plans/2026-10-15-spin-v9-prereg.md) (Amendments 1-3 written before the full read). Code:
`experiments/alpasim/lib/track_ref.py` (`JEV_TRACK=v9`), `pai_driver.py` (passes the time since the session's first call and whether the
lead limit cuts); scripts `sv9_diag.py`, `sv9_check.py`, `sv9_read.py`, `sv9.sh`. Tokyo runs `/data/runs/route_night/spin_v9/`. PAI, gains
20, `JEV_VCONT 1, JEV_LEAD 1`, paired against the stored identities (APP1X `v2pai/APP1X-L18-s0`, RSG `r281/rsg`).

**Status (2026-10-15 16:50 JST): v9a takes the spins out on both checkpoints (APP1X 29 -> 1, RSG 26 -> 1) and passes lines 1 and 3, but
fails line 2 by the letter on both. Under the prereg it is not adopted; the decision on line 2 is the user's.** No v9b (its trigger, > 2
spins left on a spin set, did not fire). The v8 diagnostic shows plain v8 already does the same on APP1X (29 -> 1).

## Mechanism (Amendment 1; [diag_handoff.md](diag_handoff.md), [diag_replay.md](diag_replay.md))

- v8 was never served in the APP1X or r281/rsg reads (`JEV_TRACK 0` in driver.log). APP1X's 29 spins tell us nothing about v8; they come from
  the plain control path.
- All 29 start 0.1-0.2 s after the 1.7 s handoff, at 22-36 m/s. The first steering command is 0.69-1.05 rad (median 0.78). The plan at k 17
  is nearly straight (|y| at 2 s has a median of 0.44 m).
- The cause is longitudinal. The fed (log) speed is 7.3-11.5 m/s above the model plan's speed over 0.5-1 s (median 8.4). The gap is 2.6-6.2
  m/s in APP1X rollouts at >= 15 m/s that do not spin, and 0.6-3.5 m/s for RSG. JEV_VCONT's 1 s blend turns the gap into a deceleration
  demand of 7-12 m/s^2. That is at the MPC's -9 m/s^2 bound, so the MPC loses the distance by steering (d285's mechanism, at gaps RSG
  almost never produces).
- Replaying v8 on the logged plans shows its re-timing would cap the demand at a 7.4 m/s drop and a 10.3 m deficit at 2 s. Non-spinning
  rollouts carried up to 11 m, so v8 is at that edge. v9a halves the cap (3.7 m/s drop, 5.1 m deficit).
- v9a = v8 + the guard. For the first 2.7 s of the session at v0 >= 5 m/s, the deceleration bound is -2.5 instead of -5 m/s^2. The
  exception is when the lead limit cuts the served speed. The lateral side is v8's gate + PP.

## Read ([v9a_app1x.md](v9a_app1x.md), [v9a_rsg.md](v9a_rsg.md); per scene in each)

| checkpoint | set | spins (identity) | zeros (identity) | new / recovered | zeros diff [95 %] | mean (identity) | mean diff [95 %] | line |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| APP1X-L18-s0 | spin set 29 | **1** (29) | 3 (29) | 0 / 26 | -26 [-29, -22] | 0.399 (0.000) | +0.399 [+0.334, +0.461] | L1 pass |
| | controls 33 | 0 (0) | 7 (6) | 2 / 1 | +1 [-2, **+4**] | 0.442 (0.453) | **-0.011** [-0.110, +0.073] | L2 **fail** |
| | all 46 controls | 0 (13) | 7 (16) | 2 / 11 | -9 [-16, -3] | 0.455 (0.371) | +0.084 [-0.003, +0.164] | reported |
| | union 90 | 1 (38) | 21 (45) | 3 / 27 | -24 [-34, -15] | 0.458 (0.343) | +0.115 [+0.053, +0.175] | L3 pass (0 new early offroad; identity 20) |
| RSG-L18-s0 | spin set 38 | **1** (26) | 17 (16) | 3 / 2 | +1 [-3, +5] | 0.433 (0.444) | -0.011 [-0.090, +0.058] | L1 pass |
| | controls 46 | 0 (0) | 7 (6) | 2 / 1 | +1 [-2, **+5**] | 0.589 (0.567) | +0.022 [-0.043, +0.086] | L2 **fail** |
| | union 90 | 1 (26) | 24 (22) | 5 / 3 | +2 [-3, +8] | 0.537 (0.529) | +0.008 [-0.039, +0.052] | L3 pass |

Per case:
- The spins left are the two known kinds. **2fe500f7** (APP1X, 2.0 s) is a third regime (Amendment 3): the model's plan reverses at 24 m/s,
  with x at 2 s from -16 to -42 m. The gate's full reference then follows a path turned 180 deg, and the reference jumps when the plan flips
  back. Such plans are rare: in the identity reads, 25 of 27 878 APP1X decisions at >= 5 m/s and 1 of 33 242 RSG decisions. **d8bfaaf2**
  (RSG, 18.3 s) is d291's simulator pose fault.
- RSG controls are d291's v8 outcome scene for scene: 268949d2 and 09528162 are new zeros (v8 gate at low speed, curv re-timing) and
  1bfad87c recovers, with the same scores as the v8 read. The guard never acts there; 268949d2 is below 5 m/s throughout the window. The
  prereg predicted this L2 failure from the arithmetic: +1 net reads [-2, +5].
- RSG v9a against the d291 v8 read on the same 84 scenes ([v9a_vs_v8_rsg.md](v9a_vs_v8_rsg.md)): 48 / 84 identical scores. Most of the
  rest gain +0.01 to +0.04 in progress, from keeping speed in the first second. Two scenes become zeros (26004ca8 corridor at 6.9 s,
  bbc53aa8 collision at 12.2 s; both in the RSG spin set), for a mean of -0.005.
- APP1X controls (33): the new zeros are 268949d2 (no guarded call: v8's own low-speed gate, as for RSG) and f8ee31cb (brake control,
  at-fault lateral collision at 5.2 s; the guard's -2.5 bound was active at k 25-27, 6 m/s, and v8 alone was not run on this scene).
  09528162 recovers. The mean is -0.011, 0.001 past the line.

## v8 diagnostic (APP1X, 29 spin scenes; [v8diag_app1x.md](v8diag_app1x.md), [v9a_vs_v8_app1x.md](v9a_vs_v8_app1x.md))

**v8 alone (`JEV_TRACK=gate`) already removes the new regime.** It leaves 1 of 29 spins (the same 2fe500f7), with zeros 2 vs 29 and mean
+0.387 [+0.330, +0.442] against the identity. Against v8, v9a reads spins 1 vs 1, zeros 3 vs 2 and mean +0.013 [-0.016, +0.030]. The extra
zero is bbc53aa8, a collision at 6.1 s; the same scene also becomes a collision zero for RSG under v9a (12.2 s), where v8 scored 0.576.
The guard keeps more speed through the first second. That buys progress but no spins, and costs bbc53aa8 on both checkpoints.

## Verdict

- The user's premise, read against the logs: the earlier fix did not fail on APP1X. It was never switched on (v8 was not adopted, so the
  d303 read ran the plain path). Served, v8 takes APP1X's handoff spins from 29 to 1. The new regime is decision 285's longitudinal
  mechanism at larger gaps, which v8's bounded re-timing already covers.
- v9a (v8 + the handoff guard) passes the spin line on both checkpoints (1 each: a reversing plan and the d291 pose fault) and the
  early-offroad line. It fails the registered control line on both: zeros +1 net, bootstrap upper bound +4 / +5 against +2, and APP1X's
  mean -0.011 against -0.01. **Not adopted under the prereg.** The RSG control cost is v8's own and identical scene for scene (268949d2,
  09528162), so the same line would fail v8 too.
- The guard is not needed for spins. Its only measured effects are +0.01-0.04 progress on fast starts and the bbc53aa8 collision on both
  checkpoints. If a serve default is wanted, the evidence favours plain v8 (`JEV_TRACK=gate`) over v9a, under a relaxed line 2.
  That choice is the user's.
- Two open items for a fix of record:
  - (a) The low-speed control zeros of the v8 gate (268949d2, opened at 3-4 m/s accelerating out of a turn) and of curv re-timing
    (09528162). These decide line 2, and nothing in v9 touches them.
  - (b) The reverse guard of Amendment 3 for reversing plans (2fe500f7). Not run.
- Tokyo: cards 0 and 1 were used 14:25-16:42 JST. Both were free of this lane's jobs at 16:42 (card 1 was then taken by the rule-expert
  lane).
