# ARENA-HOLD 2: stop3 = leash3 + a stop leash, read on the held arena windows (2026-10-18)

Prereg: arena supplement 2 in [plans/2026-10-17-arena-prereg.md](../../plans/2026-10-17-arena-prereg.md), written before the read (7b709626).
Diagnosis and offline estimate: [diag.md](diag.md) (`arena_hold.py diag`). Full readout: [hold2_report.md](hold2_report.md) (`arena_hold.sh chain2`,
box `$DATA_DIR/runs/alpasim_proxy/arena/hold/report2/`). Code: `JEV_HOLD=stop3` in `experiments/alpasim/lib/static_hold.py`; checks
`experiments/alpasim/scripts/hold_check.py` (item 4: identical to leash3 on sessions latched below 0.3 m/s).

## What leash3 still cost (diag.md)

- **(a) Launch, -32.9 points.** All of it is in the 82 latched launch windows where the human drives >= 5 m (leash3 0.497, RSG 0.899).
  At serve time they do not separate from the 807 still-end static windows: AUC 0.59-0.85 for the plan arc and the lead outputs. The best
  of 75 release rules is +6.8 points, and that is an in-sample upper bound. Released static windows are nearly all RSG zeros. Not pursued.
- **(b) 11 added AF-front zeros.** All are latched static windows in 3 logs, 9 of them from one log. Each has a stopped lead at 2.36-2.88 m
  that RSG also hits. The model's lead head puts that lead at about 21 m, so no serve-time gap signal exists. Not pursued.
- **(c) 34 corridor zeros in unlatched sessions** (braking 16, other 9, static 6, turn 3). The speed at the handoff is 0.32-4.5 m/s. In most
  of them the human brakes to a stop inside the window and the model overshoots the stop by 4-5 m. The constant-deceleration stop distance
  from the 0 s / 0.5 s measured speeds matches the human's stop. Raising the latch threshold is net negative.

## The variant

A session that is free at the handoff latches when two conditions hold: a = (v_handoff - v_prev) / dt < -0.5 m/s^2, and v / |a| <= 3 s.
Its leash is v^2 / (2|a|) + 3 m around the handoff point. Offline estimate: +18.5 points, **+0.0083** per held window against leash3.

## Read (RSG-L18-s0, 2 228 held windows, 18 route-fail dropped as before; log-clustered 95 % CI)

| | mean | paired vs RSG | paired vs leash3 | zero rate | corridor zeros |
|---|--:|--:|--:|--:|--:|
| RSG (stored) | 0.786 | - | - | 0.206 | 387 |
| leash3 | 0.934 | +0.149 [+0.093, +0.208] | - | 0.040 | 33 |
| **stop3** | **0.944** | **+0.158 [+0.101, +0.218]** | **+0.0095 [+0.0048, +0.0146]** | 0.029 | 8 |

Contributions to the +0.0095 against leash3, by stratum:

| stratum | contribution |
|---|--:|
| braking | +0.0049 |
| static | +0.0027 |
| other | +0.0025 |
| turn | -0.0005 |
| cruise | -0.0001 |
| launch | 0 |

- Gain +25.7 and loss -4.6 against leash3; **0 new zeros, 26 fixed** (25 corridor zeros, and 1 offroad zero also gone).
- 142 sessions stop-latched, the same count the offline replay predicted. The read landed above its estimate (+0.0095 against +0.0083).
- Smoke gate: [gate2 PASS] 12 windows, 0 latch errors, 3 static and 6 stop latches, nothing past its leash.

**Decision line (prereg):** the point estimate is >= +0.004 and new zeros < fixed zeros, so it is met. **Recommendation: JEV_HOLD=stop3 for the
submission image, replacing leash3 (rebuild needed).** One model, one seed, held arena windows only. The remaining levers are launch (-35)
and offroad (43), and neither has a serve-time signal here.
