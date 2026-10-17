# ARENA-HOLD supplement 2: what leash3 still costs on the held windows, and the stop3 estimate

Held windows 2228 (route-fail dropped). leash3 against RSG: +331.1 summed (+0.1486). Serve-time observables only: the driver's measured speed per call, the model's plan (2 s arc), the checkpoint's lead outputs (lead variant's log, ego held at 0 m).

## (a) launch windows

| latched class | windows | RSG | leash3 | leash3 - RSG (sum) |
|---|--:|--:|--:|--:|
| free | 1224 | 0.928 | 0.928 | -0.0 |
| launch, human < 5 m | 65 | 0.969 | 0.985 | +1.0 |
| launch, human >= 5 m | 82 | 0.899 | 0.497 | -32.9 |
| static, moving end | 50 | 0.940 | 0.960 | +1.0 |
| static, still end | 807 | 0.534 | 0.983 | +362.0 |

Separability, launch with the human >= 5 m (82, all of the cost) against still-end static (807), per window over the calls from 1 s: AUC plan 2 s arc, max (leash3) 0.76, plan 2 s arc, max (ego held) 0.85, lead speed, max 0.77, lead probability, median (low = go) 0.59, lead distance, median 0.62.

Release rules (release a latched session at the first call from 1 s where lead probability > p, lead speed > v and the plan's 2 s arc >= arc; 75 rules): the released window is credited with RSG's score, so the net is an in-sample upper bound (it ignores the late release and picks the best of the grid):

| p | v | arc | released | launch >= 5 m | still-end static | net (sum) |
|--:|--:|--:|--:|--:|--:|--:|
| 0.8 | 1.0 | 0.8 | 40 | 25 | 4 | +6.8 |
| 0.8 | 1.5 | 0.8 | 24 | 19 | 3 | +5.5 |
| 0.8 | 1.0 | 0.6 | 58 | 32 | 8 | +5.0 |
| 0.8 | 1.0 | 0.0 | 62 | 32 | 10 | +5.0 |
| 0.8 | 2.0 | 0.6 | 15 | 12 | 2 | +3.3 |

## (b) front collisions

AF-front zeros leash3 12, full 1; the 11 that leash3 adds: strata {'static': 11}, 3 logs (9 windows from one), every one latched with a stopped lead at a logged gap 2.36-2.88 m that RSG also hits (RSG AF-front in 11); ego travel at the hit 2.52-2.95 m. The checkpoint's lead head puts that lead at a median 20.8 m (probability 0.63): no serve-time signal of the gap. A shorter leash for every latch:

| leash (m) | AF-front avoided | launch progress lost (sum) | net (sum) |
|--:|--:|--:|--:|
| 2.0 | 11 | -10.6 | +0.4 |
| 2.3 | 11 | -7.5 | +3.5 |
| 2.5 | 11 | -5.3 | +5.7 |

## (c) corridor zeros the latch leaves

34 corridor zeros under leash3, all in unlatched sessions and the same windows under full / lead (RSG's own). Measured speed at the handoff >= 0.3 m/s in every one; the human brakes and stops inside the window in most:

| stratum | windows | handoff speed (m/s), median | deceleration over the 0-0.5 s calls (m/s^2), median | human still for the last (s), median | human arc after the handoff (m), median | RSG travel (m), median |
|---|--:|--:|--:|--:|--:|--:|
| braking | 16 | 2.16 | 1.34 | 1.8 | 2.56 | 8.65 |
| other | 9 | 1.21 | 0.97 | 2.5 | 1.02 | 6.42 |
| static | 6 | 0.48 | 0.86 | 3.9 | 0.30 | 5.05 |
| turn | 3 | 3.37 | 0.34 | 0.0 | 14.16 | 14.92 |

A higher static latch threshold (every newly latched session leashed 3 m from the handoff point; same estimate as below):

- below 0.5 m/s: 22 more latched ({'launch': 13, 'static': 9}), net -1.2
- below 0.8 m/s: 56 more latched ({'launch': 28, 'static': 20, 'other': 7, 'turn': 1}), net -6.9
- below 1.0 m/s: 79 more latched ({'launch': 39, 'static': 21, 'other': 17, 'turn': 2}), net -11.2

## stop3: leash3 + a stop leash (lib/static_hold.py, JEV_HOLD=stop3)

A session free at the handoff whose measured speed fell by more than 0.5 m/s^2 between the 0 s and the 0.5 s calls, with the constant-deceleration stop within 3 s, latches with the leash v^2 / (-2a) + 3 m around the handoff point. Replayed on leash3's logged speeds (the module itself): 142 sessions latch ({'braking': 73, 'other': 38, 'static': 18, 'turn': 9, 'launch': 2, 'cruise': 2}), 68 of them cut (RSG went past the leash). Corridor zeros fixed 25 of 34, new zeros 0; still-end cut windows: leash end past the human's rest point 0.44-2.90 m (median 2.62; a zero needs 4).

**Estimate against leash3: +18.5 summed, +0.0083 per held window** (gain +24.7, loss -6.3); by stratum braking +11.0, cruise -0.4, launch +0.0, other +3.6, static +6.0, turn -1.8.

Sensitivity (the same estimate, rule thresholds varied):

| a_stop (m/s^2) | t_stop (s) | margin (m) | latched | net (sum) |
|--:|--:|--:|--:|--:|
| 0.3 | 2.0 | 2.0 | 106 | +14.1 |
| 0.3 | 2.0 | 3.0 | 106 | +16.3 |
| 0.3 | 3.0 | 2.0 | 151 | +16.3 |
| 0.3 | 3.0 | 3.0 | 151 | +19.0 |
| 0.3 | 4.0 | 2.0 | 210 | +13.5 |
| 0.3 | 4.0 | 3.0 | 210 | +17.2 |
| 0.5 | 2.0 | 2.0 | 100 | +14.8 |
| 0.5 | 2.0 | 3.0 | 100 | +16.8 |
| 0.5 | 3.0 | 2.0 | 142 | +15.9 |
| 0.5 | 3.0 | 3.0 | 142 | +18.5 |
| 0.5 | 4.0 | 2.0 | 197 | +13.7 |
| 0.5 | 4.0 | 3.0 | 197 | +17.2 |
| 1.0 | 2.0 | 2.0 | 76 | +9.4 |
| 1.0 | 2.0 | 3.0 | 76 | +11.0 |
| 1.0 | 3.0 | 2.0 | 108 | +8.6 |
| 1.0 | 3.0 | 3.0 | 108 | +10.7 |
| 1.0 | 4.0 | 2.0 | 152 | +7.2 |
| 1.0 | 4.0 | 3.0 | 152 | +10.1 |
