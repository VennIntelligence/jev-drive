# ONPOL-NUPLAN clip read: is ON-IMIT's extra at-fault offroad imitation at the visited states, or the imitation target overshooting logged stops?

Lane R-NIGHT, 2026-10-12 night (CPU, Mac mini). Descriptive read of the 46 clips indexed in [clips.md](clips.md) (box copies under
`$DATA_DIR/runs/onpol_nuplan/clips`, frames extracted to `tmp/onpol_clips/` on the Mac, never in the repo). The numbers it starts from:
ON-IMIT GT distance 0.677 m (lowest of every arm; control P2H10S-F 1.009 m), at-fault zeros 20 against 12.75 (4 control seeds), offroad
18 against 12 ([on_report.md](on_report.md), decision 259 items 5 and 7). The imitation target is Amendment 2 of the
[prereg](../../plans/2026-10-11-onpol-prereg.md): lateral critically damped return to the logged path, speed starting at the ego's own
speed and blending linearly into the log's over 2 s; the on-policy rows sit on average 0.92 m (median) past the logged stop point when
the log is stopped.

## Pre-registration (written before any frame was looked at)

Two readings and what tells them apart in a clip (layout of col1_clip.py: bird's-eye view with blue ego trail, green served plan, dashed
logged path, grey road area; speed panel ego against log; gap and lead-prob panels):

- **Reading A, imitation at the visited states.** The zeros are lateral: the ego sits beside or behind the logged path, the plan pulls it
  back onto the dashed line (or the training on return-to-log rows makes it cut toward the line), and the ego clips the road edge, an
  island or a corner while the speed panel shows ego and log at about the same speed. The log is not stopping at the moment of the zero.
  Expected signature: offroad at a turn or lane bend, ego lateral offset to the dashed line visible before the zero, ego speed within
  about 1 m/s of the log's at the zero.
- **Reading B, the modified target overshooting logged stop points.** The zeros are longitudinal: the log decelerates toward a stop (log
  speed falls under about 1 m/s inside the 5 s window, or is falling steeply) while the ego, whose target speed started at its own speed
  and only reaches the log's after 2 s, keeps rolling and passes the point where the log stops: past an intersection stop line, a lane
  end, a kerb or into a stopped actor. Expected signature: ego speed clearly above the log's (more than about 1 m/s) in the last second
  before the zero, the zero located ahead of where the dashed path ends or turns, no lateral offset before it.
- **Neither.** The zero looks like the control's own failure class on the same road (same corner cut, same island), or an actor or map
  artefact unrelated to the label: then the extra offroads are not explained by either reading and the note says so.

Per clip the reader records: scene, tag, zero kind, time of the zero (first frame the ego box is red or outside the grey road area),
ego and log speed at that time and whether the log is stopping, lateral offset to the dashed path before the zero, road geometry
(straight, turn, intersection), any actor involved, and the reading it supports (A / B / neither). Decision rule for the answer: B is
supported if at least 4 of the 6 ON-IMIT offroad clips (3 broken scenes x 2 seeds) show the B signature; A if at least 4 show the A
signature; otherwise mixed or neither. The fixed scenes (control clips only: the arm's passing rollouts were pruned) say what ON-IMIT
avoided; the SSELFnoL-ON fixed scenes are read the same way to see whether the S-only arm avoids the same control failures by a different
route. Per-scene numbers (score / zero class / at-fault events / progress per arm and seed) come from on_per_scene.csv and are quoted
where they bear on the reading; they were not looked at scene by scene before this section was written beyond the aggregate above.

Known limit of the material: a broken scene has only the arm's clips, a fixed scene only the control's (ot2_loop keeps rollout.asl for
failed rollouts only), so "control counterpart" of a broken ON-IMIT scene means the control's per-scene numbers and, where another arm
also failed there, that arm's clip. Scenes are 5 s (10 decisions); a stop the log reaches after the window cannot be seen.


## Clips

How the clips were read. The renderer marks only collisions (ego box and title turn orange, orange line in the bottom panels at the
first at-fault contact; the "red" of the layout note renders orange). Offroad is not marked: its time is the first frame in which the
ego box crosses the grey mapped road area, located with a pixel check (background pixels touching the ego box, per frame) and then
confirmed on 2x crops of every frame around it. The grey area is the mapped road polygon, not necessarily the simulator's drivable
area, so a zero time is good to about one frame and a grazing contact can be invisible. The dashed box is the logged ego at the same t.
Corridor zeros have no single frame; the time given is where the ego leaves the logged path. Speeds are read off the axes, one decimal.

Full scene ids (8-char suffix -> id): 42c7526f `2021.05.25.17.54.41_veh-35_02723_02902-42c7526fc6845005`, 8227648b
`2021.09.16.17.40.09_veh-45_02539_02745-8227648b1da95adb`, 9e522849 `2021.10.06.08.16.17_veh-52_01590_01725-9e522849163c53b8`, b87c15c9
`2021.06.28.16.29.11_veh-38_03263_03766-b87c15c9f6cf54eb`, dd162898 `2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4`, 078bc102
`2021.10.06.08.16.17_veh-52_01590_01725-078bc1027dde5d1a`, 77155a60 `2021.05.25.17.54.41_veh-35_01654_01850-77155a60b2ae5e75`, 365c0937
`2021.09.16.15.47.30_veh-45_01199_01391-365c0937de9d5885`, 3b4031de `2021.10.06.08.16.17_veh-52_01949_02501-3b4031def0f45d96`, b958e696
`2021.10.06.08.16.17_veh-52_01949_02501-b958e6967993560f`, e31131de `2021.09.16.19.49.00_veh-42_00990_01609-e31131deed6656ea`, 3502b309
`2021.10.06.08.16.17_veh-52_00922_01296-3502b30911d75ed8`, 8c9e3fd9 `2021.10.06.08.16.17_veh-52_00922_01296-8c9e3fd944c051e3`, f4495a91
`2021.10.06.08.16.17_veh-52_01590_01725-f4495a91b9fb505d`. 9e522849, f4495a91 and 078bc102 are three windows of the same log segment.

### 1. ON-IMIT broken, offroad (the six clips the decision rule counts)

| scene | tag | zero | t zero | ego / log m/s at zero, log stopping? | lateral offset before zero | geometry | actor | reading | what the clip shows |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| 42c7526f | SIMIT-ON-s0 | offroad | 5.0 s | 3.8 / 4.5, no (log accelerating 3.7 -> 4.5) | on the log to 4.0 s, then 0.3-0.5 m left of it, yawed left | S-bend, left part, 4 m/s | none (car following behind) | A | Tracks the log, then the served plan swings hard left off the mapped road toward the command-L exit and the front-left corner grazes the left edge in the last frame. |
| 42c7526f | SIMIT-ON-s1 | offroad | 5.0 s | 4.0 / 4.5, no | same, slightly larger | same | none | A | Same as s0; the front-left corner is clearly over the left edge at 5.0 s. |
| 8227648b | SIMIT-ON-s0 | offroad | 4.0 s | 3.9 / 2.7; log yields to 0.6 m/s at 2.2 s and restarts | none before 3.0 s; the ego then turns in ahead of the log | right turn at an intersection | none | B | The log nearly stops before the turn; the ego only drops to 1.9 m/s, gets 1-1.5 car lengths ahead, turns in early and tight, and its right rear quarter sweeps over the inside corner. |
| 8227648b | SIMIT-ON-s1 | offroad | 4.25 s | 3.8 / 2.8; same logged yield | same | same | none | B | Same as s0, a quarter second later. |
| 9e522849 | SIMIT-ON-s0 | offroad | 2.25 s, held to the end | 7.4 / 7.2, no (7.3 -> 6.7) | 0.2-0.3 m left of the log; the log itself runs along the mapped left edge | gentle left bend on a narrow mapped strip (left-hand traffic) | none | neither | The ego sits a few decimetres outboard of a logged path that already grazes the mapped edge; same edge and same failure class as the control's own offroad on f4495a91 (same log segment). |
| 9e522849 | SIMIT-ON-s1 | offroad | 2.25 s, held | 7.4 / 7.2, no | same | same | none | neither | Identical to s0. |

9e522849, progress 0.62 against 1.05: the clip does not show a slow ego. The ego is faster than the log from 1 s on (7.8 against 6.7 m/s
at 5 s) and ahead of the logged box at 5 s. The low progress is consistent with progress being counted only up to the first at-fault
event, which here comes at about 2.25 s (inference from the clip, not checked in the scorer; the control's f4495a91 offroad on the same
road, also early, has progress 0.41-0.50).

### 2. ON-IMIT broken, other kinds

| scene | tag | zero | t zero | ego / log m/s at zero, log stopping? | lateral offset before zero | geometry | actor | reading | what the clip shows |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| b87c15c9 | SIMIT-ON-s0 | collision | 4.5-5.0 s (gap 0 at 4.5, contact line at 5.0) | 2.8 / 1.3, yes (6.5 -> 1.2) | about 1 m left of the log, which drifts right (command R late) | straight multi-lane arterial | slowing car ahead in the ego's lane | B | The log brakes behind the same car while easing right; the ego brakes later and less and rear-ends it. The model's lead distance stays at 5-7 m while the simulator gap closes to 0. |
| b87c15c9 | SIMIT-ON-s1 | collision | 4.5 s | 4.4 / 2.0, yes | same | same | same | B | Same, harder: the ego holds 4.4 m/s to the contact. |
| b87c15c9 | SBOTH-ON-s0 | collision | 4.5 s | 3.2 / 2.0, yes | same | same | same | B | The same rear-end as ON-IMIT. |
| b87c15c9 | SBOTH-ON-s1 | collision | 4.0 s | 4.4 / 2.6, yes | same | same | same | B | The same rear-end, half a second earlier. |
| dd162898 | SIMIT-ON-s0 | collision | 4.2-5.0 s (gap 0 at 4.2, contact line at 5.0) | 0.9 / 0.2, yes (log stops at 4.7 s) | none; ego slightly ahead of the log box | straight queue before an intersection | stopped car ahead | B | Crawling queue: the ego speeds up to 1.2 m/s while the log holds 0.8 and then stops, and it rolls into the car ahead. The gap is under the nominal 1 m/s, but the log is at standstill. The model's lead distance is flat at 4 m. |
| dd162898 | SIMIT-ON-s1 | collision | about 4.6 s | 1.0 / 0.1, yes | none | same | same | B | Same as s0. |
| 078bc102 | SIMIT-ON-s0 | corridor | from about 2.5 s | 3.6 / 2.5, no (log 2.4 min, then 3.0) | none before; the ego goes straight where the log turns | right turn (-101 deg) at a junction, command R | none | neither | Missed turn: the ego drives straight on and accelerates to 5.1 m/s. |
| 078bc102 | SIMIT-ON-s1 | corridor | from about 2.5 s | 3.6 / 2.5, no | same | same | none | neither | Same as s0. |
| 078bc102 | SBOTH-ON-s0 / s1 | corridor | from about 2.5 s | 3.6 / 2.5, no | same | same | none | neither | The same missed turn as ON-IMIT (5.1 and 5.5 m/s at the end). |
| 078bc102 | SSELF-ON-s0 / s1 | offroad | 4.5 s | 4.4 / 3.0 and 4.0 / 2.9, no | outside the logged arc from about 3.5 s | same right turn | none | neither | ON-SELF does take the turn, but wide and fast, and leaves over the outer edge of the exit road. |

### 3. ON-IMIT fixed scenes (control clips)

| scene | tag | zero | t zero | ego / log m/s at zero, log stopping? | lateral offset before zero | geometry | actor | reading | what the clip shows |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| 77155a60 | P2H10S-F-s0 / s1 | corridor | from about 2.5 s | 4.0 / 5.2, no (log 5.2 -> 6.5) | none; goes straight where the log turns | right turn at an intersection, command R | none | neither | Missed right turn: the control goes straight and slows to 2.5 m/s. ON-IMIT passes (progress 0.93-0.95). |
| 365c0937 | P2H10S-F-s0 / s1 | offroad | 5.0 s | 5.6 / 4.7 and 6.0 / 4.6, no (starting from rest) | 0.5-1 m outside the logged arc, ahead of the log | left turn from a near-standstill at an intersection | none | B-like (faster than log, swings wide) | The control accelerates harder than the log, swings wide and crosses the far edge of the crossing road. ON-IMIT passes with progress 0.61 / 0.65, i.e. by driving much slower. |
| 3b4031de | P2H10S-F-s0 / s1 | offroad | 4.75 s | 4.1 / 2.8, no (starting from rest) | none; ahead of the log by about a car length | pull-out into a T junction, right turn | none | B-like | The control pulls out faster than the log, is still pointing straight when it reaches the far kerb of the T, and crosses it. ON-IMIT passes with progress 1.06-1.09 against the control's 1.21. |
| b958e696 | P2H10S-F-s0 / s1 | corridor | from about 3.0 s | 3.3 / 3.5, no (log 4.5 at 5 s) | the ego turns less than the log | right turn at a junction | none | neither | Under-turned right turn plus braking (2.3 m/s at 5 s). ON-IMIT passes (progress 0.93-1.00). |

### 4. SSELFnoL-ON fixed scenes not in group 3 (control clips)

| scene | tag | zero | t zero | ego / log m/s at zero, log stopping? | lateral offset before zero | geometry | actor | reading | what the clip shows |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| e31131de | P2H10S-F-s0 / s1 | corridor | from about 2.5 s | 3.7 / 4.3, no | 2-3 m outside the logged arc by 4 s | right turn at an intersection | none | neither | Wide right turn, slower than the log. On this scene ON-IMIT turns the corridor zero into an offroad (both seeds). |
| 3502b309 | P2H10S-F-s0 / s1 | offroad | 5.0 s (s1 4.75) | 3.3 / 3.5, no | about 1 m outside the logged arc, under-rotated | right turn into a side road | none | A-like (control's own) | The control follows the arc but under-rotates on exit and crosses the far edge. ON-IMIT also goes offroad here. |
| 8c9e3fd9 | P2H10S-F-s0 / s1 | corridor | from about 3.0 s | 3.5 / 3.6 (s1 accelerates to 4.9 against 3.7) | none; goes straight where the log turns | right turn at a fork | none | neither | Missed right turn. ON-IMIT also fails here, as offroad. |
| f4495a91 | P2H10S-F-s0 / s1 | offroad | from about 0.25-1 s, held | 8.1 / 8.2 at onset, then braking to 2.8 against 6.0 | 0.2-0.3 m left of a log that runs on the mapped left edge | narrow mapped strip, slight left bend | none | neither (map edge) | The 9e522849 situation on the same log segment: the ego sits on the mapped left edge from the start, and the control brakes hard. ON-IMIT also goes offroad here (progress 0.52). |

Optional, ON-SELF / ON-BOTH broken (s0 frame 20 only): 4faa4706 SSELF, left bend, under-turns to the outside, faster than the log
(4.4 against 3.7), corridor. 20728d3e SSELF, veers right toward a fork branch away from the logged left path and brakes (2.6 against
4.0), offroad. 3a3297d2 SSELF and SBOTH, right-curving road, the ego cuts inside the logged arc by about 1 m after a speed dip, offroad.
ffe92084 SSELF, the log brakes 9 -> 4 m/s into a right turn, the ego goes straight at 9 m/s, corridor. e072351f SBOTH, the log brakes
5.5 -> 2.9 into a right turn, the ego goes straight at about 5 m/s, corridor.

## Answer

Of the six ON-IMIT offroad clips, 2 show the A signature (42c7526f s0/s1: lateral graze at an S-bend, speed within 0.7 m/s of a log
that is accelerating), 2 show the B signature (8227648b s0/s1: the log yields to 0.6 m/s before a right turn, the ego only slows to
1.9 m/s, runs 1.2 m/s faster than the log, gets ahead of it and clips the inside corner) and 2 are neither (9e522849 s0/s1: a few
decimetres of drift onto a mapped edge that the logged path already grazes, the same failure class as the control's own offroad on
f4495a91, which is on the same log segment). By the pre-registered rule (4 of 6 needed) the answer is mixed: neither reading explains
the extra offroads. Counting 9e522849 as A, which its literal signature allows, would give 4 A; it is held as neither because the
pre-registration names exactly this case. The secondary groups lean toward B on the longitudinal axis: all four ON-IMIT collisions
(b87c15c9, dd162898) are rear-ends into an actor that the log brakes for (log down to 1.2 m/s and to standstill), with the ego 0.7-2.4
m/s faster at contact, and ON-BOTH repeats b87c15c9 exactly. 078bc102 is a missed turn, which is neither reading. Two things cut
across both readings. First, in no A-signature clip does the plan pull the ego back onto the dashed line, which is the mechanism
reading A names; the offsets are sub-metre and point away from the log. Second, in both collision scenes the model's lead distance
stays flat at 4-7 m while the simulator gap closes to 0, so the late braking may be perception, not the speed target. Limits: six
counted clips on three scenes, two seeds each that behave nearly identically (effectively three observations); offroad times come from
the mapped road polygon, not the simulator's drivable area; and with no control clip on a broken scene, the counterpart is the control's
per-scene numbers (it passes all three offroad scenes in 4 of 4 seeds).

## SSELFnoL-ON fixed scenes

The control fails these six scenes in three ways: four are turn-following failures at right turns (8c9e3fd9 missed outright, b958e696
under-turned, e31131de taken wide, 3502b309 under-rotated on exit; the first three end as corridor zeros, the last offroad); one is a
wide left turn taken faster than the log from rest (365c0937); and one sits on the mapped left edge of the narrow strip on log segment
01590_01725 (f4495a91). The overlap with what ON-IMIT fixed is partial. ON-IMIT also fixes the turn-following class (77155a60,
b958e696) and the faster-than-log pull-outs (365c0937, 3b4031de), but on 365c0937 it does so by driving at 0.6 of the log's progress.
On e31131de, 8c9e3fd9, 3502b309 and f4495a91 ON-IMIT still fails; on the first two it turns the control's corridor zero into an
offroad. So SSELFnoL-ON fixes the same failure class (turn following at intersections) more broadly, and in addition fixes the
narrow-strip edge that ON-IMIT shares with the control. Its progress there is 0.70-1.13 against the control's 0.41-1.19, so it is not buying these passes by crawling the way ON-IMIT does on 365c0937.
