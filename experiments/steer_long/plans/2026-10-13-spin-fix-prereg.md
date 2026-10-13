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
