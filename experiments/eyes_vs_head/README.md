# eyes_vs_head: openpilot's own heads at our zeros

status: concluded
decisions: 272
index: eyes wrong at 98 / 98 zeros (lane probs < 0.5); eyes + rules not supported

**Question.** When our driver scores zero on the stored 18-decision held AlpaSim nuPlan rollouts, did the frozen openpilot model's own
perception heads (lane lines, road edges, lead) read the scene right while the plan still left the road ("eyes right, head wrong"), or were
the heads wrong on the reconstructed images ("eyes wrong")? This decides whether a zero-shot "openpilot eyes + rule planner" driver is worth building.

**Conclusion.** Eyes wrong, by every registered line: on unselected straight driving (200 public scenes) both ego lane lines are present
(prob > 0.5) on only 0.240 of decisions (line 0.80), with a median error at 20 m of 0.468 m (line 0.30, a 0.7 lateral scale: lines sit
~0.35-0.40 m inside the true lane on each side); on turns presence is 0.003. All 98 corridor + offroad zeros of YS-L18-s1 and YR10m10-F-s0
are class (b) eyes wrong, 0 eyes right / head wrong (line (a) >= 0.60); with the probability dropped and the scale undone (exploratory)
it is still 1 (a) / 84 (b). The lead head reported the struck object in 0 / 51 at-fault collisions: the "same-direction" hits are side
contacts with adjacent-lane cars 0.5-1.5 m to the side, which an in-path lead head does not cover (decision 272).

**Read more.** [plans/2026-10-12-eyes-vs-head-prereg.md](plans/2026-10-12-eyes-vs-head-prereg.md) (Chinese, pushed before any read);
[results/ev_report.md](results/ev_report.md) (gates, R1-R4 tables, lines, verdicts, the per-event table); per-event classes
`results/events_P0*.csv`, collisions `results/collisions_lead.csv`. Prior on real nuPlan frames: decision 104
([skill_pack roadedge report](../skill_pack/results/roadedge/report.md): lane width 0.69 of the map, both lines present on 0.22 of navtest tokens).
Box: `$DATA_DIR/runs/eyes_vs_head/` (msgs, replay pickles, read run `read/20261012-165552`).

Figures:
- [figs/ev_errors.png](figs/ev_errors.png): (a) the smaller of the two ego-line probabilities: almost all mass below 0.5, on turns and in the
  2 s before a zero near 0; (b) ego-line error at 20 m regardless of probability: straight driving peaks at 0.3-0.5 m, turns and the pre-zero
  window are flat out to 2 m; (c) signed error on straight driving: the left line sits right of the true boundary and the right line left of
  it, i.e. a compressed lane (the camera-height scale of decision 104), not a random error.
- [figs/ev_bev.png](figs/ev_bev.png): four zeros in the ego frame (map lane edges grey, road-area boundary black, logged path green, P0 lane
  lines blue, dashed when prob < 0.5, P0 road edges orange, served plan red). Look at how the heads draw a straight road through junctions and
  bends while the map turns (both (b) panels), and at the one geometry-only (a): the lines are right but at prob 0.01.

<!-- files:begin -->
## Files

- `2026-10-12-eyes-vs-head-prereg.md` (plans): EYES-VS-HEAD 预登记：零分时 openpilot …
- `ev_report.md` (results): did openpilot's own perception heads …
- `events_P0.csv` (results): 
- `collisions_lead.csv` (results): 
- `ev.py` (scripts): when the driver scores zero on the …
- `ev_report.py` (scripts): the registered tables, lines and …
- `ev_errors.png` (figs): 
- `ev_bev.png` (figs): 
- `README.md` (experiments/zeros_mech): mechanisms behind the S-family zeros
- `report.md` (experiments/skill_pack/results/roadedge): Road-edge width diagnosis

[results/](results/) 7 result files · [figs/](figs/) 4 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
