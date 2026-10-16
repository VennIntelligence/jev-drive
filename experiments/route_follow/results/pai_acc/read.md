# PAI ACC read (JEV_ACC=1) against box g10: main-risk report

Prereg: [plans/2026-10-16-pai-acc-prereg.md](../../plans/2026-10-16-pai-acc-prereg.md) (section 4; Addendum 2 there is the Chinese version of this note).
Reads: box `$DATA_DIR/runs/route_follow/pai_g10/{g10,acc}` (RSG-L18-s0, steering weight 10, v8 gate, same 8 chunk lists).
Reports: [report.md](report.md) (the registered report job, `rf_pai281.py` at a78c8cba) and [report_logwindow.md](report_logwindow.md)
(the same with the log-window turn proxy added in b9fbd62f). Per-scene extraction: `scripts/accread/` (extract / logs / ser on the box,
ana / ana2 on the Mac).

## Registered lines (paired vs g10, scene bootstrap 10 000)

| line | value | threshold | result |
|:--|:--|:--|:--|
| 1 all scenes (279 scored by both) | 0.6785 vs 0.5733, **+0.105 [+0.066, +0.145]** | CI low > -0.01, point > 0 | pass |
| 2 exam (71 scored by both) | +0.045 [-0.045, +0.131] | point >= -0.01 | pass |
| 3 zeros | 76 vs 76 (281, direct count); 74 vs 74 (279 common) | <= g10 + 2 | pass |
| 4 handoff spins | 0 vs 0 | <= g10 + 1 | pass |

Each read has one rollout lost to an RPC error ("Expected all tensors to be on the same device"): g10 on 13d646af, acc on
054b5901. Both count as zeros in the direct count. The common set therefore has 279 scenes, and exam has 71 (054b5901 is an exam scene;
it scores 0 in every read). 13d646af is a normal corridor zero in acc, so it is outside the paired set.

**Report-bug check.** The first table gives ACC the same zero classes as g10 (74 / 9 collision / 16 offroad / 49 corridor). A direct
count from results-summary.json confirms them. They match by coincidence: 15 new zeros (2 collision, 4 offroad, 9 corridor), 15
fixed zeros (3 / 3 / 9), and 8 scenes where both labels score zero but in different classes. No fix was needed. A real defect in the
turn proxy turned up instead (see below).

Descriptive: share of all 55 495 ACC decisions won by each term: set 0.47, lead 0.31, ramp 0.09, model 0.07, curve 0.06. The
guard is on for 0.19 of them (brake 0.10, stop 0.09). Rear-ended 40 vs 75. Plan 2 s deficit: exam -0.11 vs -0.72 m, control
-0.15 vs -0.49 m. stop17 (16 scored): 0.0625 vs 0, zeros 15 vs 16. The fixed one is cbd87234; it is the only stop17 scene that changes.

## The 15 new zeros

Event = the first at-fault contact / offroad / corridor flag in metrics.parquet. v = ego speed there. human = the log's speed at the
same station. won / guard = ACC terms in the 3 s before the event.

| scene | list | class | event | ACC v | g10 at same station | human v | ACC terms | group |
|:--|:--|:--|:--|--:|:--|--:|:--|:--|
| 444fca69 | control | corridor | 12.4 s, log end (105 m) | 8.4 | never reached (max 99 m), score 1.0 | 0.0 (stops at log end) | curve 17, set 14; no guard | **stop** |
| 47db2eb6 | control | corridor | 14.5 s, log end (118 m), STOP line 118 m | 6.2 | never reached (max 94 m), 1.0 | 0.0 | model 17 (brake guard), curve 14 | **stop** |
| 9b6c68b4 | exam | corridor | 13.9 s, log end (93 m), STOP line 93 m | 6.4 | never reached (max 83 m), 1.0 | 0.1 | curve 31 (v_curve 6.1); no guard | **stop** |
| 0e05cb0c | - | corridor | 15.0 s, 486 m | 32.5 | never reached (rear-ended, 0.47) | 32.5 | set / lead | high-speed drift |
| 32be03e4 | - | corridor | 13.1 s, 500 m | 36.2 | never reached (0.44) | 35.5 | set / ramp | high-speed drift |
| 4cb05d65 | exam | corridor | 13.2 s, 509 m | 38.5 | never reached (0.70) | 38.0 | set 31 | high-speed drift |
| 26004ca8 | exam | corridor | 6.2 s, 150 m | 20.0 | 13.7 (0.32, front collision 6.1 s) | 27.3 | curve 22 (v_curve 13.1), ramp 9 | faster in a bend |
| 596af0c7 | - | offroad | 10.8 s, 238 m | 19.9 | 8.5 (0.42, front collision 9.2 s) | 26.5 | curve 22, ramp 8 | faster in a bend |
| a68ad579 | exam | offroad | 7.6 s, 111 m | 9.6 | never reached (0.43) | 14.0 | curve 20, ramp 8 | faster in a bend |
| 4f7cb072 | exam | offroad | 2.8 s, 64 m | 20.7 | 19.8 (0.40) | 23.4 | lead 16, ramp 9 | near-identical state |
| 382a7247 | exam | collision (lateral) | 19.5 s, 149 m | 8.9 | 9.0 (1.0, rear-ended) | 9.8 | lead 27 | near-identical state |
| 2c0ef536 | - | collision (front) | 19.4 s, log end (43 m), STOP line 43 m | 3.8 | 3.2 (1.0) | 4.5 | lead 31 | near-identical state |
| b661da67 | - | offroad | 18.3 s, log end (104 m) | 4.7 | 4.7 (1.0) | 6.3 | lead 31 | near-identical state |
| abd45a30 | exam | corridor | 13.4 s, 131 m (left turn at 125-146 m) | 4.3 | 3.0 (1.0) | 3.5 | model 13 (brake), ramp 10 | turn entry |
| e0f6ebe1 | exam | corridor | 16.1 s, 91 m (right turn exit) | 7.0 | 8.3 (1.0) | 3.2 | ramp 22, set 8 | turn exit |

**Stop class ("the human stopped here, the car did not"): 3 new zeros: 444fca69, 47db2eb6, 9b6c68b4.** This meets the prereg's
>= 3 mark, which section 4 requires us to state. All three are route-end scenes:
- The human stops at the log end: at a STOP line in 47db2eb6 and 9b6c68b4, at a turn entry in 444fca69.
- ACC arrives at 6-9 m/s, with the curve limit or the cruise set speed governing.
- The model's stop guard never fires. Only 47db2eb6 has a late brake guard.
- g10 does not stop either. It is just too slow to reach the stop point within the 20 s (it gets 6-24 m short). Its 1.0 comes from
  slowness, not from reading the stop.

These are the same mechanism as d315's 17 `cor_route_end` zeros. None of them is in stop17, because RSG at gains 20 also stayed short of
these points. Two of the three are control scenes. They are the only control zeros (2 vs 0).

The other 12 new zeros break down as follows:
- **3 high-speed lateral drifts.** The car holds the human's 32-38 m/s on a near-straight road and leaves the corridor laterally at
  13-15 s. g10 never got that far because it was slow or rear-ended.
- **3 faster in a bend.** The curve limit wins, but ACC is still above g10, and below the human.
- **4 flips from a near-identical state.** Speed differs by < 1 m/s from g10 at the same station.
- **2 turn cases** (below).

## The 15 fixed zeros (g10 zero, ACC scores)

| scene | list | g10 class, event | g10 v / human v there | ACC there | why ACC scores |
|:--|:--|:--|:--|:--|:--|
| 229791a5 | - | collision (lateral), 10.5 s | 4.4 / 5.2 | 7.3, set | faster: through before the conflict |
| 68e53160 | - | corridor, 11.8 s (turn) | 6.3 / 6.5 | 9.0, set | faster |
| 8a143cb2 | - | corridor, 5.5 s | 6.0 / 4.9 | 7.0, ramp | faster |
| 9f393c8a | exam | collision (lateral), 4.9 s | 3.8 / 6.1 | 5.7, set | faster; 0.31 (rear-ended later) |
| a1b166cc | exam | corridor, 10.2 s (turn) | 4.4 / 5.6 | 7.1, set | faster |
| c8641276 | - | collision (front), 14.6 s | 5.1 / 9.6 | 10.2, set | faster |
| fda39805 | - | corridor, 15.8 s (turn) | 3.3 / 3.6 | 4.5, set | faster |
| a174faed | - | corridor, 11.0 s (turn) | 5.3 / 4.1 | 4.5 | rear-ended at 11.9 s (0.51) before its own failure |
| ce161092 | exam | offroad, 14.7 s | 7.9 / 6.3 | never reached | rear-ended at 11.3 s (0.48) |
| bd97de34 | exam | corridor, 15.5 s (85 deg turn) | 12.6 / 5.7 | never reached | rear-ended at 8.3 s (0.49) |
| 0a228e44 | - | corridor, 7.8 s (-56 deg turn) | **17.0 / 6.3** | 6.0, curve (v_curve 4.9) | curve limit slows the too-fast g10 |
| a2c5ae46 | exam | corridor, 9.6 s | **15.9 / 12.5** | 4.6, ramp / curve | slower |
| cbd87234 | exam, stop17 | corridor at log end, 10.1 s | **14.2 / 4.3** | 4.8, curve (v_curve 4.4) | curve limit at the route end |
| 1bfad87c | - | offroad at log end, 18.0 s | **16.4 / 12.3** | 13.4, curve | slower at the route end |
| e1c6ae9f | - | offroad at log end, 14.8 s | **16.1 / 10.0** | 9.3, ramp | slower at the route end |

The fixed zeros fall into three groups:
- **7 slow-induced zeros**, fixed by cruising at the hand-off speed.
- **5 where g10 was faster than the human**, which the curve limit or the route-end curve corrected.
- **3 where ACC was rear-ended first.** Its own failure point was never reached, so these are not really fixed.

## Turn-proxy drop (whole trace: ACC 5 / 9 left, 1 / 3 right vs g10 9 / 9, 3 / 3)

4 of the 6 "missed" turns are artefacts of the proxy:
- In 3ffad395, 4e8c8f9e, 51400458 and 9d9052bd the log turns only 1-27 deg.
- g10's 67-81 deg come from driving past the log end. The score no longer looks there (g10 score 1.0; it leaves the corridor after
  progress 1.0). ACC also scores 1.0 on all four.
- `rf_pai281.py` integrated yaw over the whole 20 s controller trace. That contradicts its own docstring.
- The fix (b9fbd62f) adds log-window columns: yaw only until the controller arc reaches the log length. The whole-trace columns are kept.

On the log window there are 8 turn scenes: **ACC 4 / 5 left, 2 / 3 right vs g10 5 / 5, 3 / 3.** The two real losses:
- **abd45a30 (left, 92 deg; new exam zero).**
  - ACC reaches the turn earlier and faster: 6.2 m/s at 124 m vs g10 3.9 m/s, human 3.5 m/s.
  - Its path goes wide to the right (-5.3 m at 132 m), and it leaves the corridor before the turn.
  - The curve limit did engage (v_curve_min 5.3 m/s, route covered). Its 2.4 m/s² target is above what the human drove here.
  - The model plan at the same station turns less: 4 s yaw 49 vs 58 deg. That is d310's speed-dependent plan curvature.
- **e0f6ebe1 (right, -86 deg; new exam zero).**
  - The turn itself is completed (-87 deg), at a lower speed than g10 (4.5 vs 7.7 m/s at 80 m).
  - At the exit it drifts left while ramping back toward v_set: +2.8 / +4.2 m at 88 / 96 m. g10 drifts too (+1.4 / +2.9 m) but stays inside.
  - Marginal: a turn-exit drift during re-acceleration.

None of the drop is a scene ending before the turn.

## Summary

Lines 1-4 pass. The main risk materialised at the prereg's mark: **3 new stop-class zeros** (route-end stops at STOP lines and a turn
entry, reached at 6-9 m/s with no stop guard). They are offset by 1 fixed stop17 zero (cbd87234, through the curve limit). The real
turn cost is 2 scenes, not 6: one fast entry, one exit drift. ACC and holdlead (JEV_VHIST) were never read together. Submission is
the user's call.
