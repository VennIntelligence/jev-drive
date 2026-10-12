# road_probe: city road geometry in frozen Cinque tokens

status: concluded
decisions: 275 (inputs 223, 243, 268, 272)
index: frozen eyes do not see city roads: R3 zeros 1.26 m (line > 1.0); straight 0.25 m, turns 0.88 m

**Question.** Is the road geometry of nuPlan city scenes (ego-lane boundaries, road edges, lane heading at 10 / 20 / 40 m), especially at
junctions and bends, present in the frozen openpilot (Cinque) vision tokens our AlpaSim driver already computes, although the shipped
lane-line / road-edge heads do not decode it (decision 272, cited as 269 in the brief)? A capability probe that decides whether a learned road head is worth building.

**Conclusion.** By the registered line on the 98 road-exit zero windows the verdict is **"frozen eyes do not see city roads"**: a probe trained on
the frozen tokens (decision 243's head, 69 419 on-policy rows, 1020 train scenes) misses the ego-lane boundaries at 20 m by a window median of
1.26 m [1.03, 2.02] (line > 1.0), with both sides within 0.5 m in 0.11 of the windows (45 of 98 windows have a defined route lane at 20 m). On
ordinary held driving the tokens do carry geometry the shipped heads do not output: straight 0.25 m (shipped 0.60, 0.39 with the 0.689 squeeze
undone; p90 1.18 m, so decision 272's "usable" line is still not met), turns 0.88 m against 3.94 m shipped and 2.01 m for an ego + command
blind floor. The error grows in and just before junctions (1.1-1.7 m) and more data still helps on turns (-0.10 m per doubling of log units)
but not reliably at the zeros (-0.26 [-0.54, +0.01]) (decision 275).

**Read more.** [plans/2026-10-12-road-probe-prereg.md](plans/2026-10-12-road-probe-prereg.md) (Chinese, pushed before any read, with
pre-read clarifications in section 7); [results/rp_report.md](results/rp_report.md) (gates, R1-R3 tables, verdict, learning curve, junction
distance, oncoming-lane AUC, post-read exploratory section); tables `results/*.csv`, `rp.json`. Box `$DATA_DIR/runs/road_probe/`: held
rollouts with frame dumps `roll/c*/run`, tokens `tok/{held,zero}/`, labels `labels/`, predictions `pred/`, read `read/20261012-175622`.
Pipeline: `rp_chain.sh submit-roll` (3 pool jobs) -> `rp.py tok` / `replay` (GPU) -> `rp.py labels --group train|held|zero` (CPU)
-> `rp.py train --arm P|blind|lin [--frac F] [--seed S]` -> `rp.py read` -> `rp_report.py picks`, `rp.py bev`, `rp_report.py report` (Mac).
Cost: about 1.2 card-hours (held rollouts 3 x 12 min, tokens / replay 3 min, 7 probe runs 4-7 min each).

Figures:
- [figs/rp_errors.png](figs/rp_errors.png): (a, b) ego-lane boundary error against distance ahead on held straight and turn decisions for the
  probe (blue), the linear floor (green), the ego + command blind floor (pink) and the shipped heads raw (grey) and with the 0.689 squeeze undone
  (orange): on straight roads the probe is the only one well under 0.3 m at 10-20 m; on turns everything is worse and the shipped heads leave the
  road (4-5 m at 20-40 m). (c) the same at 20 m by scene type with the registered 0.3 / 1.0 m lines dotted: the R3 zeros are the hardest class for
  every predictor, and the probe sits above the 1.0 m line there. (d) learning curve: straight and turn errors fall with more log units, the R3
  windows fall from 1.9 to 1.3 m but flatten between 14 and 27 units.
- [figs/rp_junction.png](figs/rp_junction.png): turn-decision error at 20 m against the route distance to the junction: the probe is at 0.4-0.5 m
  while the junction is still 20-60 m away and at 1.1-1.7 m inside it and in the last 20 m before it; the shipped heads are 4-5 m there.
- [figs/rp_bev.png](figs/rp_bev.png): four zero windows (the event decision; picked at the 10 / 40 / 70 / 95 % quantiles of the probe's window
  error among scenes zeroed by one driver) and two held turns (median and 90th percentile), ego frame, right positive. Map lane edges grey,
  road-area boundary black, logged route green, served plan red; probe ego-lane boundaries blue dots and road edges light-blue squares at 10 / 20 / 40 m;
  map labels black / grey crosses; shipped ego lines grey dashed and with the squeeze undone orange dotted. Look at how the shipped lines run
  straight on through every junction, and at the probe: it bends with the road on the held turns but in the zero windows places a lane of the
  right width at the wrong lateral position (top row).

<!-- files:begin -->
## Files

- `2026-10-12-road-probe-prereg.md` (plans): ROAD-PROBE 预登记：nuPlan …
- `rp_report.md` (results): is city road geometry in the frozen …
- `rp.py` (scripts): is the road geometry of nuPlan city …
- `rp_report.py` (scripts): ROAD-PROBE report, Mac .venv
- `rp_chain.sh` (scripts): Every GPU stage is a pool job
- `r12.csv` (results): 
- `r3_windows.csv` (results): 
- `rp.json` (results): 
- `rp_errors.png` (figs): 
- `rp_bev.png` (figs): 
- `README.md` (experiments/eyes_vs_head): openpilot's own heads at our zeros

[results/](results/) 8 result files · [figs/](figs/) 6 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->
