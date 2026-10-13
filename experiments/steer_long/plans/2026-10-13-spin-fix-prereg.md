# SPIN-FIX pre-registration (2026-10-13, written before the first full read)

Goal: the ego never spins on PAI, by changing only the driver's control path (the trajectory handed to AlpaSim's MPC), never the model.
The user's line: a spin is a mechanical fault and must go to zero, not be made rarer by a weight (`rel_front_steering_angle_weight` 20 is a threshold shift).

## Spin detector (scripts/sf_detect.py, controller CSV rows; fixed before any fix rollout)

- handoff: |vx wz| > 6 m/s^2 anywhere in (1.7, 3.2] s (steer-long amendment 4, gains pilot rule)
- fast: after 1.7 s, vx >= 5 m/s and |vx wz| >= 6 m/s^2 for >= 0.3 s, or sideslip |vy| / vx >= 0.15 at vx >= 5 m/s
- slow: after 1.7 s, vx < 5 m/s and |steering state| >= 0.7 rad (full lock 0.785) for >= 1.0 s
- spin = any of the three. On the stored reads: RSG-L18-s0 (gains 20) 26 / 281 rollouts; YS-L18-s0 / s1 (gains 5) 40 / 37.

## Sets (results/spin_fix/)

- `spin_set.txt` 38 scenes = every scene with a spin in the stored RSG-L18-s0 PAI 281 read (gains 20, 26) + the 14 handoff-spin scenes of the
  gains pilot (pilot_g1, any of ybase / ys0 / ys1 at gains 5); 2 overlap.
- `ctrl_fast.txt` 20 scenes: pilot_g2 (handoff >= 20 m/s, no handoff spin), clean in the stored RSG read (score > 0, no spin), crc32 order.
- `ctrl_slow.txt` 16 scenes: low-speed turns (heading change >= 45 deg in 4 s with v < 5 m/s, pilot_g3 rule; 2 more with v < 8 m/s), clean in the
  stored RSG read. Only 16 qualify (the brief asked for 20).
- Baseline for both = the stored RSG-L18-s0 read `/data/runs/route_night/r281/rsg` (same checkpoint, `JEV_VCONT=1.0 JEV_LEAD=1`, gains 20; a rerun
  of an unchanged configuration is identical per scene, docs/tokyo-box.md).

## Arms (Tokyo card 0, RSG-L18-s0, VCONT 1.0, LEAD 1, gains 20 unless stated; driver experiments/alpasim/lib/track_ref.py)

- `time`: JEV_TRACK=time (re-time the plan's path from the ego speed, deceleration of the reference bounded by AMIN = -5 m/s^2).
- `full`: JEV_TRACK=full (time + path-tracking reference with curvature <= min(0.2, ALAT 4 / v^2), curvature rate <= min(JLAT 5 / v^2, 0.1 /s)).
- `full-g5`: full at the default gains 5 (is the fix alone enough).
- Iterations of the configuration (JEV_TRACK_CFG) after a failing read are allowed; each one is listed in results/spin_fix.md with its
  numbers. A 2-scene smoke precedes every full read.

## Lines

1. Spin set: spins = 0 (all kinds, 38 scenes, one seed).
2. Controls (36 scenes): spins 0, zeros not above the baseline (0), mean score not below baseline - 0.01.
3. Adopted switch set = an arm that passes 1 and 2 on its own full reads. If full-g5 also passes 1 and 2, the recommendation drops the gains knob.
4. Reported, not gated: zeros and mean on the spin set against the baseline; the nuPlan smoke5 check (switch off: plans identical; on: no spin, no new zero).

## Amendment 1 (coordinator, before any read; 2026-10-13)

The -5 m/s^2 bound on the reference's deceleration can weaken emergency braking (a plan that says "stop now" gets a reference that still rolls).
- The bound is a parameter: `JEV_TRACK_CFG=amin=-7` etc. The spin set is read with amin -5 (default) and -7.
- Every read reports at-fault collisions, front collisions (the ego into something ahead) and hit-from-behind separately (sf_read.py).
- Brake control `ctrl_brake.txt`, 10 scenes: stored RSG rollouts without a spin and outside the other sets, with the ego's acceleration <= -4
  m/s^2 for >= 0.5 s, the 10 with the lowest minimum (-8.2 .. -5.2). Asked for <= -6 for 0.5 s: only 1 rollout of 281 qualifies (hard
  braking at speed in the stored read almost always came with a spin, so the spin set is itself the main brake test). Line for it as for the
  other controls (spins 0, zeros and at-fault collisions not above the baseline, mean not below baseline - 0.01).
- Reported, not gated: zeros and mean score on the spin set against the stored RSG read (16 zeros, 0.4437).

## Amendment 2 (v7, 2026-10-13 17:40 JST, written before any v7 closed-loop read)

The user's requirement: spins are a mechanical fault and go to 0 without costing turns or nuPlan. One serving setting must pass all four:
1. spin set (38, gains 20): spins = 0;
2. controls (46 = fast 20 + slow 16 + brake 10, gains 20) against the stored RSG read: spins 0, zeros not above the baseline (+0), mean not
   below baseline - 0.01;
3. nuPlan held 452 @18 against the stored RSG-L18-s0 read: mean within 0.01, zeros within +3 (`ol_confirm.py`);
4. nuPlan public 1491 @10 guard: mean not below the stored read - 0.02.

Per-case look before choosing (offline, `scripts/sf_mid.py` / `sf_gate.py` on the stored `time` read, results/spin_fix/): in the 21 spins
that `time` leaves, the served plan's own path asks for a lateral acceleration of 4-21 m/s^2 (median 7) at the speed the car can still have
there, and the curvature speed limit does not change that (the curv rebuild of the same plans gives the same peak: the turn starts inside
the braking distance). These plans are the model swerving while it brakes (3ffad395 at 17.4 s: 2.3 m to the side within 3.7 m at 10 m/s);
no reference that keeps to the plan's path there is drivable. Hence two arms:

- `curv` (`JEV_TRACK=curv`): re-timing (AMIN -5, AMAX 3, JLON 10 from the measured acceleration) + VCURV on the plan's own polyline, no
  path tracking, no curvature clipping (the brief's candidate v7).
- `gate` (`JEV_TRACK=gate`): curv, and on any call where a path segment the curv reference reaches within T_GATE 2 s has a curvature above
  GATE x the full mode's cap at the speed it is passed (max(v_ref, v0 - 2 t), segments passed below GATE_V 2 m/s not checked), that call
  serves the v6 full reference instead. Threshold from the offline screen on stored plans: GATE 1.0 opens on 7.5 % of nuPlan held rollouts
  (1.0 % of calls), 1.25 on 1.1 % (0.1 %), 1.5 on none; on PAI, 21 / 21 spin rollouts of the `time` read open before the onset (20) or
  never spin under v6 (1e8e939e, the deceleration step the jerk limit removes), 1 / 20 fast controls (one call in the force-GT phase),
  3 / 16 slow and 3 / 10 brake controls open at 1.0 (2 and 1 at 1.5).

Order: nuPlan held @18 for curv and gate 1.0 on the box pool, plus smoke5 switch off (parent commit vs this one, same card) and gate on.
PAI (card 0 when free): gate on the 38, then on the 46. If the 46 fail at the threshold read first only on scenes where the gate opened,
the threshold moves to 1.5 and both PAI sets are read again (the nuPlan read is valid for every threshold at or above the one read). curv on
the 38 as an ablation if card time is left; weight 5 last. The public guard is read for the setting that passes 1-3.

## Amendment 3 (v8, 2026-10-13 22:40 JST, written before any v8 read)

The v7 `gate` PAI read (84 scenes, Tokyo card 1, `/data/runs/route_night/r281/spin_read`): spin set 2 spins (60cc35db, d8bfaaf2) vs 26
stored, zeros 15 vs 16, mean 0.4516 vs 0.4437; controls 0 spins, zeros 7 vs 6 (new: 268949d2, 09528162; recovered: 1bfad87c), mean 0.5823 vs
0.5668, one new at-fault front collision (68e53160, a brake control that was already a zero). Lines 1 and 2 fail. Per case
(results/spin_fix.md section 7, `spin_fix/v8_traces.txt`):
- 60cc35db: the gate opened 15.0-16.1 s on a braking swerve and closed at 16.2 s while the plan still put its 0.5 s point 1.5 m to the side
  (first chord 23 deg off the car). Path takes that chord as its origin heading (YAW_TOL), so the per-segment check saw no curvature; curv
  served the kink and the MPC answered 0.62 then 0.88 rad at 8.4 m/s.
- d8bfaaf2: not the control path. AlpaSim's physics teleports the ego 2.1 m onto a pose with 40-70 deg of roll / pitch at a fixed place
  (same spot in the stored RSG read, `time`, v4, v5); the driver's anchor yaw from the full quaternion is then 0.8 rad off the controller's
  planar yaw, the straight reference is handed over rotated, the MPC commands -1.25 rad. 1e8e939e (`time` read, stored RSG) is the same
  fault: the "deceleration step" attributed to it and to d8bfaaf2 under v5 was wrong.
- Controls: 268949d2 is the gate (opened at 3.3-4.1 m/s accelerating out of a right turn, ratio 1.0-1.19, full capped the turn at KDYN,
  corridor). 09528162 is curv's re-timing (the trajectory leaves the stored one from 2 s; the gate opened only at 19.5 s). 68e53160 is the
  gate bounding an undrivable evasive swerve (ratio 5-10 at 6.5 m/s): front collision instead of the stored read's corridor exit, zero
  both ways.

v8 (`JEV_TRACK=gate`, commit a1fbf3fd), three changes, each only where v7 was wrong:
1. PP: the gate also checks the arc from the car onto each plan node reached by T_GATE (2 |y| / d^2 against the cap), when v0 >= 5 m/s.
2. GATE_LO 1.5 instead of GATE 1.0 while v0 < 5 m/s and a0 >= 0.5 m/s^2 (accelerating at low speed); slowing into a turn keeps 1.0.
3. Anchor guard (any JEV_TRACK mode): on a pose with |(qx, qy)| > 0.3 the anchor yaw is the controller's planar yaw 2 atan2(qz, qw).
Off, time, curv, full: unchanged (3000 random plans, identical arrays; gate with pp=0, gate_lo=1 = v7).

Offline screen on v7's own closed loop (`sf_screen.py`; a rollout whose every gate decision agrees is the same closed loop): spin set
3 / 38 change (60cc35db PP opens at 4.7 s and at 16.2-16.3 s; d8bfaaf2 PP at 11.5 s + guard at 18.3 s; b40f4729 one low-speed opening
closed); controls 1 / 46 (268949d2); nuPlan held RSG 17 / 452, RSGboth 17 / 452 (mostly low-speed accelerating openings closed). The tilt
guard fires on no nuPlan held rollout of RSG / RSGboth.

RSGboth (coordinator, 22:30): v7 gate costs RSGboth-L18-s0 on held -0.024, zeros +8. Of that, -0.019 and 8 of 11 new zeros are in the
425 rollouts where the gate never opened (it opens on 0.96 % of calls, 35 rollouts), and the served reference matches the plan's 8 points
within 3 cm at 1-2 s on 99 % of calls. So the cost is not the gate firing; it is either curv's re-timing or closed-loop perturbation noise.
A plain RSGboth rerun on the current tree is read alongside as the noise control. A plain-by-default variant (re-time only on a
deceleration / curvature trigger) was screened on the stored PAI read and rejected: even triggering on any deceleration demand leaves 3 / 26
spins without the trigger in the second before the onset (A -2 m/s^2: 8 / 26), so it cannot be expected to reach 0.

Lines for v8 (amendment 2's four, line 3 now for both policies):
1. spin set 38: spins 0; 2. controls 46: spins 0, zeros not above the stored RSG read, mean not below it - 0.01;
3. nuPlan held 452 @18, RSG-L18-s0 and RSGboth-L18-s0 each against its plain read: mean within 0.01, zeros within +3;
4. public 1479 @10 (RSG-L18-s0): mean not below the stored read - 0.02. Also smoke5 switch off: plans identical to the same tree with the
   v7 files (card 1 for both).
d8bfaaf2 after the guard is still on a tilted pose for the rest of the rollout (the simulator keeps the car at 40-70 deg); if it is still
counted a spin there, it is reported as a simulator fault with its trace, not hidden: the detector is not changed.

Addendum to amendment 3 (23:30 JST, after the first v8 nuPlan read, before any v8 PAI read): change 2 (GATE_LO) is withdrawn. The held 452
read of a1fbf3fd (RSG-L18-s0) differs from v7 on 3 scenes, and 2 of them went 1 -> 0 exactly where GATE_LO closed a v7 opening at
3.2-4.5 m/s. The same openings that cost PAI's 268949d2 save nuPlan scenes, so no threshold choice in this range is a fix. v8 = changes 1
and 3 (commit 7d001d79); on v7's closed loop it changes 2 / 38 spin-set rollouts (the 2 spins), 0 / 46 controls, 1 / 452 RSG and 2 / 452
RSGboth held rollouts. Line 2 is then expected to fail as for v7 (268949d2, 09528162); reported, not tuned away.
