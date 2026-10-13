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
