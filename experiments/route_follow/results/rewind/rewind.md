# RECOVERY-EXPERT REWIND: can the rule expert recover RSG-L18-s0's own PAI turn failures? (no training)

Prereg [plans/2026-10-16-recovery-rewind-prereg.md](../../plans/2026-10-16-recovery-rewind-prereg.md) (Amendment 1 after the smoke, Amendment 2
packing only). Code: `experiments/onpol_pai/lib/onp_log_driver.py` (`rew` cells), [scripts/rf_rewind.py](../../scripts/rf_rewind.py),
[scripts/rf_rewind.sh](../../scripts/rf_rewind.sh); served tree 5a8beed6 ([code.json](code.json)). Box, PAI 36 strict turns, gains 20,
`JEV_VCONT=1.0 JEV_LEAD=1`, **path `JEV_TRACK 0`** (the expert's only path; pairs with r281/rsg, not with gate reads). Expert XP-L12-E46.
Raw table: [table_raw.md](table_raw.md); per unit: [rewind.csv](rewind.csv); every rollout: [read.json](read.json); box run
`$DATA_DIR/runs/route_follow/rewind/run` (smoke: `.../rewind/smoke`).

**How the rewind works.** No restart or seeding. The same scene is opened again on the box, where sessions are bit-identical. RSG-L18-s0
drives it exactly as in the failing identity rollout up to decision h = k_fail - 10 Delta. From h on, the driver serves the rule expert's
plan. The car's pose, speed, controller state and the other cars are the policy's own at that moment. Check: in all 80 expert sessions, the
served plans and the controller track before h are bit-identical to the identity's, and v(h) is the identity's to 0 m/s. Smoke idB (a second
identity session) reproduces the identity. In the first second after h the expert brakes toward its 4.6 m/s entry speed at most 3 m/s².
There is no jump to 0 and no jump to the log speed, so the velocity is carried.

**Units.** On the box, the identity fails 27 of 36 strict turns: 22 `event`, where t_fail is the first zero event, and 5 `off`, where
t_fail is the first time it is more than 2.5 m off the route after entry - 20 m. 5 identity rollouts complete and 4 are short. Box vs Tokyo
r281/rsg outcome: 33 / 36 equal. At Delta 3 s, 403d558a falls inside the forced-log window (h < 17), so n = 26.

| Delta | n | expert completes | policy (identity) | paired [95 % CI] | v(h + 3 s) >= 2 m/s | at-fault zero | d293 shape gate, post-h (of completed) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 s | 27 | **23 (85.2 %)** | 0 | +85.2 [+70.4, +96.3] pp | 27 / 27 | 5 | 7 / 23 |
| 2 s | 27 | **22 (81.5 %)** | 0 | +81.5 [+66.7, +96.3] pp | 26 / 27 | 3 | **15 / 22** |
| 3 s | 26 | **22 (84.6 %)** | 0 | +84.6 [+69.2, +96.2] pp | 24 / 26 | 1 | **16 / 22** |

Read: the policy column is 0 by construction. The units are the identity's failures, and handing the rewound state back to the policy is
the identity rollout itself. The paired line therefore adds nothing beyond the expert's own rate (this was stated in the prereg). Bootstrap
is over scenes. The at-fault column includes zero events after the turn was completed: 268949d2 and 403d558a complete and are hit later.

**Lines.** RECOVERS at every Delta: >= 0.6 and >= +0.3 paired. The creep check passes at every Delta: 100 / 96 / 92 % keep >= 2 m/s 3 s
after the handover. This is not d290's ratchet, because the expert's speed comes from its own profile, not from the fed history.
- Rear-ended: 0 / 1 / 1 rollouts are truncated by rear contact (`short_rear`). Rear contact after h, including after completion, occurs in
  4 / 3 / 2 rollouts.
- Expert failures, Delta 1 / 2 / 3: collision 3 / 2 / 1, offroad 0 / 0 / 1, short 1 / 3 / 2.
- 3232fdb7 is short at every Delta. Its t_fail is 17.6 s into the 20 s clip, so the clip ends 2.4 s after the zero time. That is too
  little road to reach exit + 10 m.

**Reading.**
- The expert can recover the policy's own failure states. At Delta 2-3 s, most recovered paths are smooth enough to be label candidates:
  15-16 of 22 pass the d293 gate on the post-handover track.
- At Delta 1 s the expert still completes 23 of 27, but only 7 pass the gate. The failing component is mostly lateral acceleration
  (11 of 23 fail it; peaks up to 8 m/s²): it yanks the car back.
- Two caveats on what "recovery" means here:
  - (a) At Delta 2-3 s the handover is often still before the entry: median |lat| at h is 0.7-0.8 m, and h lies before the entry in
    14 of 27 units at Delta 2 s. Those units are closer to "the expert takes the approach from the policy's speed and position" than to
    pulling back from a deviation.
  - (b) The expert from the scene's start already completes 24 of these 27 (29 / 36 overall). The rewind result is largely the expert
    being good at these turns. The rewind adds a174faed (Delta 1-2 s) and 1e8e939e (Delta 2-3 s), which the expert fails from the start;
    e37aa159 completes at Delta 1 s only.
- So recovery demonstrations exist. The candidate label set is the gated Delta 2-3 s paths: about 15-16 turns, once each. The deep-deviation
  cases (Delta 1 s, |lat| up to 3.4 m) are recovered too, but with hard pulls.

[figs/rewind_bev.png](../../figs/rewind_bev.png) shows three scenes in bird's-eye view. Dots are the handovers at Delta 1 / 2 / 3 s, the
grey x is the policy's zero, and the dashed line is the human log.
- ddc3e8df: the policy leaves the corridor at the corner; every rewound expert completes, handed over 1.4-3.4 m off the route.
- 1e8e939e: the expert from the start leaves the road on the S-bend before the turn (black x). Rewound at 2-3 s, it carries the policy's
  line through.
- 3232fdb7: every rewind runs out of clip (short).
