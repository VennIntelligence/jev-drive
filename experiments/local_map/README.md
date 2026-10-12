# local_map: carried probe and route at the zeros

status: concluded
decisions: 279 (inputs 272, 275)
index: route received earlier: 0.35 m at 20 m on 28 / 45 zero windows (line < 0.5); carried probe 2-4 m

**Question.** Decision 275's frozen-token road probe misses the ego lane by 1.26 m at 20 m in the 98 road-exit zero windows. Would a
rule-based local map have saved it: carry what was seen (the probe's lane boundaries) or given (the route waypoints, which covered the
junction when it was 40-80 m away) 1-4 s earlier into the zero window's frame with the egomotion the driver receives, and compare with the map?

**Conclusion.** By the registered line the verdict is "local map works", but only through the route and only where it reaches: the newest
received route that covers the 20 m line puts the lane centre at a window median of 0.35 m [0.11, 3.65] on 28 of the 45 windows with a
defined map lane (line < 0.5 m on >= 23); carrying the probe's own earlier estimate makes it worse, not better (1.26 m now, 2.0 / 3.0 / 2.4 / 3.9 m
from 1 / 2 / 3 / 4 s earlier; held turns 0-20 m before a junction 1.38 -> 2.2-5.9 m), because the probe reaches 20 m from earlier only
through its weak 40 m point. Fixed-d routes are mostly out of reach (the car covers a median 14 m in 4 s, the route starts ~40 m ahead): d = 8 gives
0.20 m but on 19 windows only, and the registered first-crossing rule picks up a route's far leg when the route starts beyond the line (the
30-50 m tail); a label-free start-behind fix (exploratory) gives 0.23 m [0.10, 0.44] on 25 windows and 0.58 m on pre-junction held turns.
Still 0.42 of window decisions have no received route covering 20 m at all (decision 279).

**Read more.** [plans/2026-10-12-local-map-prereg.md](plans/2026-10-12-local-map-prereg.md) (Chinese, pushed before any read; section 7
amends G-carry before the read); [results/lm_report.md](results/lm_report.md) (gates, Z / H tables, verdict, plan check, exploratory
section); tables `results/z_windows.csv`, `h_turns.csv`, `plan_check.csv`. Box `$DATA_DIR/runs/local_map/read/20261012-183650`
(est_{zero,held}.parquet, bev.pkl). Pipeline: `lm.py read` (box, AlpaSim venv, CPU, 25 s on the whole box) -> copy the read dir ->
`lm_report.py --read DIR` (Mac). Inputs are decision 275's probe predictions, labels and token tables (`$DATA_DIR/runs/road_probe/`) and
decision 272's stored driver messages (`$DATA_DIR/runs/eyes_vs_head/msgs/`); held routes are regenerated with AlpaSim's RouteGenerator
(G-route: 0.999 of decisions within 0.05 m of the logged first waypoint).

Figures:
- [figs/lm_errors.png](figs/lm_errors.png): (a) zero-window 20 m error (log scale, window bootstrap CI) against how many decisions earlier the
  estimate was made, for the carried probe (blue), the carried route centre (orange) and route centre + probe width (green); squares = newest
  covering route, hollow = the exploratory start-behind rule; dotted 0.5 / 1.0 m lines. (b) how many windows each rule covers (dotted: the
  registered 23). (c) the same error on held turns 0-20 m before a junction. Look at the blue line rising with d (memory of the probe hurts) and
  the orange line dropping to 0.2-0.3 m only from d = 6-8, where coverage is lowest.
- [figs/lm_bev.png](figs/lm_bev.png): four zero windows at the 10 / 40 / 70 / 95 % quantiles of the L3 newest-route window error, each at the
  last window decision with a defined map lane; ego frame, right positive. Map lane edges grey, road boundary black, human log green, served
  plan red, newest covering route orange, probe now light-blue dotted, probe carried from 2 s earlier blue, map lane boundaries at 20 m as crosses.
  Look at the fourth panel: at k = 0 the only route starts 40 m ahead
  and the 20 m line meets its far leg (the 38 m error is the first-crossing rule, not the route). In panels 1 and 3 the route bends away from the
  human log beyond the junction; it agrees with the map lane at 20 m.

<!-- files:begin -->
## Files

- `2026-10-12-local-map-prereg.md` (plans): LOCAL-MAP 预登记：把 2–4 s …
- `lm_report.md` (results): would a carried local map have saved …
- `lm.py` (scripts): would a rule-based local map have saved …
- `lm_report.py` (scripts): LOCAL-MAP report, Mac .venv
- `z_windows.csv` (results): 
- `h_turns.csv` (results): 
- `plan_check.csv` (results): 
- `gates.json` (results): 
- `lm_errors.png` (figs): 
- `lm_bev.png` (figs): 
- `README.md` (experiments/road_probe): city road geometry in frozen Cinque …

[results/](results/) 7 result files · [figs/](figs/) 4 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
