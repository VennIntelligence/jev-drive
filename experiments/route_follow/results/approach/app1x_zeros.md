# APP1X new held zeros: which channel of the approach loss (decision 303 Amendment 5)

2026-10-16. Descriptive, CPU only. The stored held 452 / 18-decision reads of APP1X-L18-s0 (approach course v1, W 1), APP2W05-L18-s0 (v2, W 0.5,
the replication) and APP0C-L18-s0 (the W 0 control) are compared with RSG-L18-s0 scene by scene. Code: `scripts/rf_appzeros.py` (extract on the box
with AlpaSim's venv in 40 s on 32 cores; read / summary / onset / plan01 / fig run anywhere). Rollout pickle on the box:
`$DATA_DIR/runs/route_follow/approach/zeros/rollouts.pkl`.

Files here: [app1x_zeros.csv](app1x_zeros.csv) (one row per new-zero scene and label), [app1x_zeros.json](app1x_zeros.json) (distributions),
[app1x_zeros_onset.json](app1x_zeros_onset.json) (by station from the turn entry), [app1x_zeros_plan01.json](app1x_zeros_plan01.json)
(plans on identical inputs), figure [../../figs/app1x_zeros.png](../../figs/app1x_zeros.png).

## Definitions

- A new zero is a scene where the label scores 0 and RSG-L18-s0 does not. APP1X has 52 new zeros and 8 fixed ones, net +44. APP2 has 50 new and 6 fixed.
  APP0C has 4 new and 6 fixed.
- Zero step: the first decision with offroad, a front or lateral collision, or a GT distance of at least 4 m (rf_follow's k_z). The kind is the summary's
  failure reason.
- Route: AlpaSim's MAP route regenerated at every logged anchor (`rf_follow.regen_full`). All labels share one polyline per scene: it is a prefix of the
  longest one in all 452 scenes.
- Route turns: every turn of the route of at least 45 deg (find_turn's rule at the approach rows' threshold). A scene falls into one class:
  - one of the 24 held turns (rf_follow);
  - a route turn: an entry within [s0 - 10, s_zero + 60] m;
  - a bend: a net heading change of 20 to 45 deg over that stretch;
  - straight.
- Phase: where the zero station lies against the nearest entry:
  - before: from entry - 40 m to the entry;
  - inside: from the entry to exit + 10 m;
  - after: past exit + 10 m;
  - far: more than 40 m before an entry.
- Lateral offset and heading error carry a + sign when they point toward the inside of the nearest turn. With no turn nearby the sign follows the local
  curvature, and on a straight road + means left.
- Channels: the last 3 s before the zero (decisions kz - 5 .. kz). The zero label is compared with RSG at the same station, with RSG's track interpolated
  linearly in station.
- Opened: a turn entry of at least 45 deg lies within [st - 10, st + 60] m of some decision up to the zero, so the approach window was open there.
- Plan: the served plan's 1.0 / 1.5 / 2.0 s poses. The plan's lateral offset is measured to the route, and its shift is the plan's offset minus the ego's.
- Identical inputs: decisions 0 and 1 have the same state in every label, because the rollouts only split from decision 2. The plans there differ only
  because the policies differ.

## 1. Where

| label | new zeros | held turn / route turn / bend / straight | phase before / inside / after / far / none | zero station - entry (median) | fail offroad / corridor / collision |
|:--|--:|:--|:--|:--|:--|
| APP1X | 52 | 7 / 37 / 6 / 2 | 11 / 30 / 2 / 3 / 6 | inside +15.0 m, before -4.3 m | 29 / 18 / 5 |
| APP2W05 | 50 | 5 / 35 / 4 / 6 | 16 / 21 / 2 / 1 / 10 | inside +16.9 m, before -4.5 m | 21 / 19 / 10 |
| APP0C | 4 | 1 / 3 / 0 / 0 | 2 / 2 / 0 / 0 / 0 | | 3 / 1 / 0 |

- The new zeros sit at turns. 44 of APP1X's 52 are in scenes where a turn of at least 45 deg opened the approach window.
- 30 of the 52 happen inside the turn, a median 15 m past the entry: the car is already turning.
- Only 7 of the 52 are on the 24 held turns. Most lie on route turns that rf_follow does not count, because the log itself does not finish them within
  the 18 decisions.

## 2. Which channel (last 3 s, against RSG at the same station)

| label | speed ratio q25 / med / q75 | lateral A - RSG (m, + inside) | heading A - RSG (deg, + inside) | lateral at the zero (A) |
|:--|:--|:--|:--|:--|
| APP1X | 0.88 / **0.99** / 1.06 | -1.59 / **-0.75** / +0.09 | -11.1 / **-4.8** / +0.4 | -2.73 / -0.52 / +0.11 |
| APP2W05 | 0.98 / **1.04** / 1.23 | -1.80 / **-0.59** / +0.13 | -10.8 / **-4.1** / +2.9 | -2.86 / -0.86 / +0.11 |
| APP0C | 0.93 / 1.03 / 1.05 | -1.64 / -0.81 / -0.03 | -7.4 / +1.1 / +2.8 | |

Every zero by its side at the zero step (lateral offset to the route beyond 0.5 m):

| label | inside / on the line / outside | outside: corridor / offroad / collision |
|:--|:--|:--|
| RSG-L18-s0 | 13 / 8 / 5 (26) | 3 / 2 / 0 |
| APP1X | 19 / 19 / 32 (70) | 20 / 10 / 2 |
| APP2W05 | 19 / 15 / 36 (70) | 23 / 7 / 6 |
| APP0C | 9 / 8 / 7 (24) | 3 / 4 / 0 |

- APP1X has 44 more zeros than RSG. They split as +27 outside, +11 on the line and +6 inside; APP2 splits its +44 as +31 / +7 / +6.
- The outside zeros of APP1X (n 27 of its new ones) share one signature:
  - speed ×1.05 against RSG;
  - lateral -1.57 m;
  - heading -10.1 deg;
  - a median 16 m past the entry;
  - mostly lateral corridor exits.
  The car drives the turn at RSG's speed and turns too little: it under-turns and goes wide.
- The 11 inside new zeros look like RSG's own zero mode: inside offroads near the entry, with a median speed ×0.91, lateral +0.60 m and heading +5.8 deg.
  The control's noise alone produces 4 new zeros.
- Per-scene mode (speed / line) for APP1X:

  | mode | scenes |
  |:--|--:|
  | same speed / outside | 15 |
  | slower / outside | 8 |
  | faster / outside | 8 |
  | same speed / on the line | 6 |
  | same speed / inside | 5 |
  | slower / on the line | 3 |
  | slower / inside | 3 |
  | faster / on the line | 2 |
  | faster / inside | 1 |

  Only 11 of the 52 are faster.

Onset by station from the entry, over the new-zero scenes (A - RSG at the same station, median):

| bin (m from entry) | APP1X n (scenes) | lateral | heading | plan 1-2 s shift | speed ratio | APP2 lateral / heading / plan shift |
|:--|:--|--:|--:|--:|--:|:--|
| -20 .. -10 | 102 (19) | +0.00 | +0.0 | +0.05 | 0.99 | +0.00 / +0.0 / +0.02 |
| -10 .. 0 | 228 (39) | +0.00 | +0.0 | +0.00 | 1.00 | +0.00 / +0.0 / +0.00 |
| 0 .. 10 | 157 (36) | -0.22 | -4.2 | **-0.60** | 1.03 | -0.23 / -4.6 / -0.65 |
| 10 .. 20 | 111 (28) | -1.33 | -8.7 | -0.62 | 0.97 | -1.79 / -8.7 / -0.76 |

On every other turn scene, APP1X against RSG in the 0 .. 10 m bin shows lateral +0.06, heading +0.3 deg and plan shift -0.01. Most held reads start
about 5 m before the entry, so the pre-entry bins are mostly decisions 0 and 1, which are the same state in both labels. The tracks part in the first
10 m after the entry. There the plan points outward first, by -0.60 m against RSG's plan; the heading follows at -4.2 deg, and the lateral offset
follows at -0.22 m and then -1.33 m. Speed stays equal throughout (0.97 to 1.03).

## 3. Does the plan show the pull (identical inputs, decisions 0 and 1, A minus RSG)

| bin (ego station - entry) | n | APP1X plan 1-2 s shift | APP1X 4 s pose lateral | APP1X 4 s yaw (deg) | APP1X plan arc 4 s (m) | APP2 4 s yaw | APP0C 4 s yaw | RSG's own 4 s yaw |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| -40 .. -20 | 46 | +0.010 | +0.00 | -0.45 | -0.42 | -0.11 | -0.04 | +0.5 |
| -20 .. -10 | 59 | +0.019 | -0.09 (mean -0.36) | -1.42 (mean -3.05) | -0.16 | -0.03 | +0.00 | +9.5 |
| -10 .. 0 | 168 | +0.019 | -0.12 (mean -0.19) | -1.01 (mean -1.47) | +0.00 | -0.55 | -0.34 | +37.7 |
| no turn within 60 m | 578 | -0.002 | +0.01 | +0.11 | -0.57 (mean -0.82) | +0.00 | +0.06 | +0.3 |

All values are medians unless marked.

- On the same input in the last 20 m before an entry, APP1X's plan:
  - leans 0.02 m inside at 1-2 s, which is the lateral pull toward the entry state and is tiny;
  - turns 1.0 to 1.4 deg less by 4 s (means 1.5 to 3.1 deg), against RSG's 9.5 to 37.7 deg, and ends 0.1 to 0.4 m less inside.
- APP2's 4 s deficit is smaller (-0.55 deg), and APP0C's is -0.34 deg.
- The speed channel does not move near the entry: the 4 s plan arc changes by +0.00 m in the last 10 m.
- On straight roads APP1X's plan is 0.57 m shorter over 4 s (about -0.14 m/s). L3 is not active there, and APP2 does not show it.

## 4. Sanity split: no turn within 60 m

- Of the 452 scenes, 240 never open the approach window.
- APP1X:
  - 8 new zeros there against 2 fixed, net +6, compared with net +38 in the 212 opened scenes;
  - 6 of the 8 are 20 to 45 deg bends below the 45 deg row threshold, 3 of them lateral corridor exits outside at a heading of -11 to -14 deg: the same
    under-turn on a bend shallow enough to fall in neither the turn rows nor the straight negatives;
  - 3 are slow at-fault collisions (×0.32 to ×0.40).
- APP2: 10 new zeros there. 6 of them are at-fault collisions on straight roads at ×1.14 to ×1.55 of RSG's speed. APP2 added a symmetric speed term on
  the straight negatives, so this is the speed channel failing in the faster direction, and it is small.

## Reading

Most of the harm is lateral and is not speed. The zeros come from under-turning after the entry at unchanged speed, not from cutting inside. On
identical inputs the trained plan turns about 1 deg less by 4 s just before the entry. In closed loop this compounds within 10 m of the entry into a
-0.6 m outward plan, -4 to -9 deg of heading and lateral corridor or offroad exits on the outside.

The entry-pose and 1-2 s blend terms (L1 / L2, the shape terms) are the only parts of the loss that act on the plan's lateral shape and heading.
The read cannot separate L1 from L2. L3 moves poses only along their own heading, and near the entry it left the plan arc and the speed at the zeros
unchanged.

The +0.17 m inside offset at the entry (Amendment 4) is real, but it is not where the cars fail. They fail 15 m later, on the outside. This is the late-wide
direction of d299 / d308 / d313 (the plan widens by itself), amplified.

Figure: [../../figs/app1x_zeros.png](../../figs/app1x_zeros.png). Each panel shows the route (grey), RSG's track (blue), APP1X's track up to the zero
(orange) with the black cross at the zero, and both served plans 1 s before the zero (dashed).

| panel | what it shows |
|:--|:--|
| 600c570e | Lateral corridor exit 17 m into a left turn: the same speed as RSG, a heading 11 deg short |
| dc09590a | Held turn: offroad on the outside |
| 44c051e3 | Held turn: one of the minority inside offroads, with the plan cutting the corner |
| e21f5d25 | A bend below 45 deg, so no approach row applies: the same outward under-turn, which is the leak |
