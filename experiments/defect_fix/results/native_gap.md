# NATIVE-GAP: what separates the native arena held reads from ~0.93 (2026-10-19)

Diagnosis, no training. Ruler of decision 371: the arena held split (2 228 windows), native serve (no hold). Reads: RSG-L18-s0 (the arena's
stored read), NTV0-L18-s0, JTN-L18-s0, JTN-L18-s1 (`jt.sh aread ... off`). NTV0-L18-s1 has no native arena read and is not used.
Code: `scripts/native_gap.py` (box: `extract`, `cflist`, `cf`), `scripts/native_gap.sh` (pool chain), `scripts/native_gap_report.py`
(Mac: every table below; raw output [native_gap_tables.md](native_gap_tables.md)). Box outputs: `$DATA_DIR/runs/defect_fix/native_gap/`.

**Short answer.** JTN-L18-s1 needs 87 more points (0.891 to 0.930). 114 of its 243 lost points are static windows where the human stands
at a light with no lead and the car rolls out of the 4 m disc. In 85 of them the human stands at a red light's line. The same model stands
almost perfectly behind a close lead (5 zeros in 477) and on empty static scenes (1 in 38). So the model holds a stop when a vehicle gives a
reason. It does not hold one for a red light. Whether a window fails is mostly set at decision 0, under force-GT, on the human's own standing
input: a plan 4 s arc below 2.5 m almost never fails, and one above 3 m fails 19-91 % of the time. The closed loop then amplifies it (creep).
Staying at a red line with no lead would be worth +72 net, after paying 13 points on the never-rendered green launches. Fixing JTN's late
launch back to RSG's level is worth +14. Those two reach 0.9296. The speed ratchet is real: the ego-to-human speed ratio drops 0.99 to 0.79
(RSG) and 0.99 to 0.89 (JTN1) over 3.5 s, and 90-102 % of that drop is in the plan, not in execution. But the arena scores it at most 13
points, because the score saturates at 0.8 progress in a 5 s window. Fed the human's speed history on the same frames, the plan recovers only
26-44 % of its shortfall (d285 on PAI: 83 %). The rest of the trend comes through the frames, which are synthesised along the ego's own slow
track.

## Setup and gate

- Per decision (10 per window, 0.5 s apart; k = 0, 1 are under force-GT and not executed): fed speed `|ego[4:6]| x 10`, the served plan,
  the ego anchor; per window the human's logged track and the logged traffic from the window's data source (the loader arena_resid uses),
  and the scorer's per-step metrics. A decision is valid before the first kept-step event (any collision, offroad, d >= 4 m), as the scorer
  truncates.
- Human speed "at the same station" = the human's speed when its track passed the ego anchor's projection (first passage).
- Gate: the four reads' means on the 2 228 common windows reproduce the stored readouts (RSG 0.7857, NTV0 0.7699, JTN0 0.8744, JTN1 0.8909;
  jtb/JTN-L18-s1/arena.md 0.7857 / 0.8909). At k = 0 the ego sits on the human track (lateral 0.000 m at p99), and the fed speed equals the
  human's to 0.05 m/s (median).

## A. The speed ratchet

### A1. Ego / human speed at the same station, natively

Median over moving decisions (k >= 2, valid, the human >= 2 m/s at that station):

| k (0.5 s steps) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | 0.989 | 0.968 | 0.944 | 0.916 | 0.893 | 0.875 | 0.858 | 0.831 |
| NTV0-L18-s0 | 0.987 | 0.966 | 0.949 | 0.933 | 0.914 | 0.902 | 0.892 | 0.880 |
| JTN-L18-s0 | 0.987 | 0.966 | 0.950 | 0.936 | 0.923 | 0.914 | 0.901 | 0.891 |
| JTN-L18-s1 | 0.987 | 0.968 | 0.954 | 0.943 | 0.934 | 0.922 | 0.918 | 0.912 |

On windows valid to the end, the per-window ratio at k = 2 / k = 9 is: RSG 0.989 / 0.791, NTV0 0.987 / 0.853, JTN0 0.987 / 0.870,
JTN1 0.987 / 0.894. The drop has no plateau inside the 5 s window. It is steepest on the free road and weaker behind a lead or in turns.
Median ratio by decision stratum: free 0.930 / 0.945 / 0.948 / 0.956 (RSG / NTV0 / JTN0 / JTN1), lead 0.951-0.958, turn 0.967-0.984. This
is the trend-following signature of d315: no scene trigger is needed.
The JT stage roughly halves the ratchet: the per-step log drop is -0.045 for RSG and -0.029 for JTN1. NTV0 sits between them (-0.035).
After a stop the ratchet runs the other way. Once moving again, the ego is at 0.89-0.93 of the human 1-2 s after its stop and at 1.07-1.12
after 3 s, because it arrives at each station later than the human did. Decisions where the ego stands still at a station the human passed
at >= 2 m/s (late launch) number 32 / 18 / 27 / 18.

### A2. Plan-short against execution lag

For each step k to k+1 (k = 2..8, both valid, the human >= 2 m/s at both stations, the ego >= 1 m/s), the change of log(ego / human speed at
the same station) splits exactly into two terms:

- the **plan term**, log(vp(0.5 s) / v) - log(vh[k+1] / vh[k]): the plan's own speed at 0.5 s against the current speed, net of the human's
  own change;
- the **execution term**, log(v[k+1] / vp(0.5 s)): the executed speed against the plan.

| model | steps | per-step drop (log) | plan term | execution term | plan share | plan term m/s | execution m/s | executed behind plan at 0.5 / 1 / 2 s (m) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG | 7 193 | -0.0447 | -0.0457 | +0.0010 | 1.02 | -0.175 | +0.006 | -0.02 / +0.00 / +0.17 |
| NTV0 | 7 183 | -0.0350 | -0.0346 | -0.0005 | 0.99 | -0.129 | -0.001 | -0.01 / +0.01 / +0.12 |
| JTN0 | 7 070 | -0.0364 | -0.0336 | -0.0027 | 0.92 | -0.110 | -0.009 | -0.00 / +0.02 / +0.10 |
| JTN1 | 7 056 | -0.0292 | -0.0264 | -0.0029 | 0.90 | -0.083 | -0.010 | -0.00 / +0.02 / +0.07 |

By stratum, the plan share is 0.86-0.96 on the free road (where most of the loss is), 1.11-1.14 behind a lead, and 0.92-1.19 in turns (JTN1's
turns lose nothing). After a stop the ratio rises, by +0.04 to +0.05 per step, about half of it plan and half execution (the controller
accelerating past the plan's 0.5 s speed). On the losing steps alone (63-73 % of all steps), the plan carries 99 % of the loss for RSG, 97 %
for NTV0 and 93 % for JTN0 / JTN1.

**Reading.** The ratchet is the model's: each plan asks for about 2-4 % less speed than the car has (the human's own change already netted
out), and the controller delivers what is asked to within 2 cm at 1 s and 7-17 cm at 2 s. Decision 364's 0.84 m plan-to-execution gap at
4 s is not execution lag. It is the next plans being shorter again, the same plan-short counted over eight replans. JTN / NTV0 differ from
RSG only in the size of the plan term (0.6-0.75 x RSG's), not in where the loss sits.

### A3. Open-loop counterfactual: the same frames, the human's speed history

Pool chain `native_gap.sh cf` (priority 12): 270 held moving windows (non-static strata, human path >= 10 m; 45 per cache shard, seed 0)
re-read natively with `SH30_DUMP` for RSG and JTN1, then each dumped decision replayed on the same frames twice: **F** as fed, and **H**
with the ego row's speed history (vx, vy, ax and the three past poses, re-spaced along the ego's own history polyline, as d285's variant B
and vhist.respace) rewritten to the human's speed at the station of each history key. Gate: F against the dumped plan, max 0.09 / 0.13 m,
p99 0.06 m over the 8 poses (RSG / JTN1; first try without the logged route memory failed the gate at 2.0 m p99 and is discarded).
Decisions k >= 2 with the human >= 2 m/s at the ego's station; plan speed = plan 2 s arc / 2 s, against the human's speed there:

| model | decisions | n | ego / human | plan F / human | plan H / human | recovery (H - F) / (1 - F) | gain (vp_H - vp_F) / (vh - v) |
|---|---|--:|--:|--:|--:|--:|--:|
| RSG | all | 2 046 | 0.930 | 0.863 | 0.923 | 0.44 | 0.48 |
| RSG | ego < 0.9 x human | 877 | 0.751 | 0.614 | 0.748 | 0.35 | 0.46 |
| RSG | ego < 0.7 x human | 354 | 0.545 | 0.402 | 0.581 | 0.30 | 0.37 |
| JTN1 | all | 2 033 | 0.951 | 0.912 | 0.938 | 0.29 | 0.35 |
| JTN1 | ego < 0.9 x human | 729 | 0.768 | 0.616 | 0.725 | 0.29 | 0.35 |
| JTN1 | ego < 0.7 x human | 266 | 0.527 | 0.355 | 0.524 | 0.26 | 0.30 |

(Plan speed at 0.5 s gives the same picture: RSG 0.889 to 0.927, JTN1 0.925 to 0.943.)

**Reading.** The trend-follower claim holds, but here only about a third of it runs through the numeric history. With the human's history on
the same frames, the plan moves 0.35-0.48 of the way from the ego's speed toward the human's (gain). It recovers 26-44 % of its shortfall,
against d285's 83 % on PAI. The plan also asks less than the car's current speed even when fed the human's history (plan H / human
0.92-0.94 against ego / human 0.93-0.95). The rest of the trend is in the frames. Six of the eight image slots are synthesised along the
ego's own (slow) track, and the two real keyframes are taken where the slow ego actually was. So a history-only fix (d320 S1-S3, d343
noise, JEV_VHIST hold) can remove at most about 40 % of the plan-short. That matches those lines closing without moving the slowness. JTN
follows its own history less than RSG does (gain 0.35 against 0.48), which is part of why its ratchet is smaller.

## B. Static windows natively (878 held windows, the human moves < 1 m)

Cue = the arena set's lead at the window start (wait_set's strip rule, within 30 m; split at 8 m), then the census light labels
(red_light_readers.py's join): "red light at the line" = the human stands at a red light's stop line (red_stop, class C; the front bumper
within about 1 m of the line, median -0.8 / -1.3 m for JTN1's zeros / successes); "light of unknown state" = a light line ahead with no state in the log (median 119 m away), effectively no
visible cue; "no light". Plan arc = the plan's 4 s arc (m, median); roll k = first k >= 2 with the ego above 0.2 m/s.

| cue | windows | RSG zeros | JTN0 zeros | JTN1 zeros | RSG arc k0 / k2 / max | JTN1 arc k0 / k2 / max | JTN1 rolled | JTN1 disp at k9 (m) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| lead <= 8 m | 477 | 54 | 4 | 5 | 3.36 / 3.75 / 3.64 | 1.76 / 1.88 / 2.18 | 0.23 | 0.06 |
| lead 8-30 m | 49 | 26 | 13 | 12 | 4.09 / 4.90 / 14.4 | 2.28 / 2.36 / 4.41 | 0.59 | 0.62 |
| red light at the line, no lead | 239 | 228 | 111 | 87 | 5.62 / 7.49 / 26.6 | 3.23 / 3.63 / 14.9 | 0.62 | 1.56 |
| light of unknown state, no lead | 75 | 50 | 42 | 29 | 5.01 / 6.94 / 18.4 | 2.97 / 3.39 / 15.6 | 0.73 | 2.42 |
| no light, no lead | 38 | 27 | 2 | 1 | 4.56 / 5.47 / 11.5 | 2.46 / 2.50 / 2.84 | 0.16 | 0.00 |
| all | 878 | 385 | 172 | 134 | 3.97 / 4.64 / 7.56 | 2.14 / 2.27 / 2.67 | 0.40 | 0.13 |

(NTV0 is RSG-like: 403 zeros, 228 at red lines, 67 behind a close lead.) Every JTN1 static zero is also an RSG zero (134 shared, 251 RSG-only).

**When the roll starts.** JTN1's zeros start rolling at k = 2 (26), 3 (10), 4 (54), 5 (42), 6 (1): mostly 1.0-1.5 s after the handoff.
RSG's start at k = 3 (314 of 385), the first executed step after it. **Creep amplification** (JTN1 zeros, medians by k = 0..9): plan arc
3.49, 3.52, 4.36, 5.74, 7.99, 10.9, 13.5, 16.6, 18.9, 20.9 m; speed 0 at k = 2, then 0.10, 0.28, 0.61, 1.09, 1.76, 2.52, 3.29 m/s. The
successes stay at 1.97-2.23 m and 0.00-0.01 m/s throughout. RSG's zeros: 5.15 to 24.5 m. Its successes: 3.34 m, falling to 2.3-3.0.

**Decision 0 decides most of it.** At k = 0 the ego still stands on the human's pose under force-GT, so this plan is the model's open-loop
reading of a standing scene. Zero rate by that plan's 4 s arc:

| plan arc at k = 0 (m) | <= 1.5 | 1.5-2 | 2-2.5 | 2.5-3 | 3-4 | 4-6 | > 6 |
|---|--:|--:|--:|--:|--:|--:|--:|
| RSG | 0.00 | 0.00 | 0.00 | 0.03 | 0.23 | 0.71 | 0.83 |
| JTN1 | 0.00 | 0.02 | 0.02 | 0.33 | 0.45 | 0.35 | 0.83 |
| windows (4 models) | 270 | 544 | 398 | 393 | 797 | 860 | 250 |

AUC of the k = 0 arc for the window's zero: JTN1 0.86, RSG 0.89 (k = 2: 0.91 / 0.94; k = 3: 0.95 / 0.96). Below about 2.5 m the controller
does not move the car, and the window is won. Above 3 m the car creeps, and the creep feeds the next plan.

**JTN1's successes against its failures.** Plan arc at k = 0: 1.97 against 3.49 m. Speed at k = 5: 0.00 against 0.61 m/s. The share that
rolls at all: 0.29 against 0.99. The successes are the scenes where the decision-0 plan is short, mainly a close lead. Ranked by the
median k = 0 arc, the cue types order the same way for every model: close lead (JTN1 1.8 m, RSG 3.4) < far lead (2.3, 4.1) <
no light (2.5, 4.6) < unknown-state light (3.0, 5.0) < red line (3.2, 5.6). **The red light lengthens the standstill plan rather than shortening it.** The model reads an
open junction ahead of a stop line as "go", and reads the lamp little (d315: the light accounts for about 0.025 of the braking).

**Reading.** JTN's large native gain over RSG (+251 static windows fixed, 0 broken) comes from the d363 standstill offset. Its decision-0 plan
is 1.6-2.4 m shorter than RSG's in every cue class, which pushes most scenes below the creep threshold. It does not come from reading the reason to
stand. Where the scene says "go" more strongly (a light ahead, no lead), the offset is not enough.

## C. Points to 0.93 (JTN-L18-s1, 2 228 windows)

0.93 needs 2 072 points. JTN1 has 1 985.0: a gap of **87.1** (0.039). The four reads lose 477 / 513 / 280 / 243 points (RSG / NTV0 / JTN0 /
JTN1). The best of the four per window would score 0.9127. Classes, first match: human-track offroad, then static stratum, then zero types,
then partials. Best-of-4 = the points still lost on JTN1's windows by the best of the four reads (what no existing model recovers).

| loss class | RSG | JTN1 | JTN1 windows | best-of-4 | fix kind |
|---|--:|--:|--:|--:|---|
| static: red light at the line, no lead | 226 | **85** | 85 | 84 | model signal: hold at a red line; input is in the frame (lamp rendered, constant) |
| static: light of unknown state, no lead | 50 | 29 | 29 | 29 | mostly no visible cue: not fixable by reading |
| human-track offroad (ceiling) | 26 | 23 | 23 | 23 | not fixable (map / scorer) |
| offroad: non-turn strata | 10 | 21 | 21 | 6 | model: JTN's lateral offset (of JTN1's 21 offroad zeros where RSG has none, 7 are in log veh-52, as d336) |
| launch partials (late / slow launch) | 3.6 | 17.7 | 47 | 2.6 | model signal + input: d363 standstill offset; lead motion onset invisible for 0.5 s at 2 Hz |
| corridor: braking (the human stops after the handoff) | 16 | 13 | 13 | 12 | model signal: arrival stop (d336), no run-time cue (d354) |
| offroad: turn stratum | 9 | 13 | 13 | 6 | model: turns (closed serve line) |
| static: lead 8-30 m | 26 | 12 | 12 | 12 | scorer: closing up to a far lead is legal driving, scored 0 by the 4 m disc |
| slow partials (other / turn / braking / cruise) | 14.1 | 13.2 | 116 | 8.3 | model: the ratchet (A); capped by the 0.8 progress saturation |
| corridor: other / launch / turn | 12 | 6 | 6 | 5 | mixed |
| static: lead <= 8 m / no light | 80 | 5 | 5 | 4 | solved by JTN |
| AF-front / AF-side / rear-hit partials | 4.8 | 5.1 | 11 | 2.7 | small |
| **all** | **477** | **243** | 381 | 195 | |

**The red-light trade.** The render's lamp never switches (wait_cap census), so a model that holds at a red line cannot see the green when
the human launches. Its value is the stand-still read (`starter`) on those windows:

- stands at red, no lead (239 windows): JTN1 152.0 points, stand-still 237.0 (+85.0);
- launches at green, no lead (40): JTN1 31.3, stand-still 18.3 (-13.0);
- launches at green behind a lead (34): JTN1 30.8, stand-still 20.3. A model that follows the lead's launch need not lose these.

Net +72.0 points (+0.032) for "stay when standing at a red line with no lead". The arena held split has 6 red-standing windows for each
green launch without a lead.

**Arithmetic to 0.93 (+87):**

| step | points | cumulative mean |
|---|--:|--:|
| JTN-L18-s1 native | | 0.8909 |
| hold at a red line with no lead (net of the green launches) | +72.0 | 0.9233 |
| launch on time behind a lead (late-launch partials to RSG's level, d363) | +14.1 | 0.9296 |
| JTN's extra non-turn offroad to RSG's level (lateral offset) | +11 | 0.9345 |
| far-lead / unknown-light static zeros (no cue; scorer disc) | up to +41 | (not counted) |
| the ratchet removed entirely (all slow partials) | up to +13 | (not counted) |
| ceiling (23 human-track offroad windows lost by any policy) | | 0.9897 |

The target is reachable only through the first two rows. Both are standstill decisions, and d363 showed that in JTN they trade against each
other through one low-speed offset (wait 0.94 vs on-time launch). Getting both needs two readable cues: the red line (in the frame) and the
lead starting to move (weak at 2 Hz). The ratchet, offroad and turns together can add about 0.01. They cannot close the gap without the
standstill pair.

## What each fix implies

- **Red line, no lead (85 points; +72 net).** Model-side signal. At standstill the lamp and line are in the frame and the policy does not use
  them. Lamp and stop line make it plan longer, not shorter. The training needs a consequence for crossing a red line from a standing
  start: logged light state in the S rows, or trainval standing rows at red lines kept apart from the general standstill offset. The
  offset helps here but costs the launch (d363). First check, cheap: a probe of the frozen tokens for "red lamp at the line" on these 239
  windows' k = 0 frames, as d362 did for PAI. If the tokens do not carry it, it is an input problem (resolution / crop of the lamp), not a
  training problem.
- **Late launch behind a lead (14 points).** Model signal plus input: the trainval rows' offset (d363), and lead motion visible only in 2 of
  8 slots. A launch cue the policy can read (the lead's displacement between the two real keyframes) is the precondition for keeping the wait
  and dropping the delay.
- **Speed ratchet (at most 13 points here).** In the plan (A2); a trend follower on its own speed. It matters more on PAI's long rollouts
  (d285 / d315) than on the 5 s arena windows. Input-side fixes (history) were tried in d320 / d343 and did not move it.
- **Offroad: turns and the JTN lateral offset (34 points, about 12 shown fixable by another read).** Model, closed serve line.
- **Not fixable by the policy:** 23 human-track offroad windows (ceiling 0.990); about 29 unknown-light and 12 far-lead static zeros, where no
  visible cue asks for standing. The far-lead ones are a scorer artefact of the 4 m disc.

## Limits

- One seed per model except JTN (two seeds, consistent). The cue labels come from the log (light state, lead strip), not from the
  render. "Red light at the line" is the census label. Whether the lamp is legible in the frame is not measured here (the probe above).
- Human speed at a station uses first passage. Where the human creeps through a station, the ratio is noisy, so medians are reported.
- The stand-still values for the red-light trade come from the `starter` read, which never moves. A real "hold" model might also roll a
  little on the green-launch windows.
