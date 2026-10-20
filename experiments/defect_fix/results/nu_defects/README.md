# NU-DEFECTS: does B2-s1 have the PAI plan defects, and what costs it the most? (2026-10-21, overnight nuPlan track)

This is a diagnosis. Nothing was trained, nothing was rolled out, and no GPU was used.
- Code: `scripts/nu_defects.py` (stored-plan gates, Mac or box) and `scripts/nu_defects_price.py` (frozen PCS of a window set, Mac).
- Inputs: the stored native arena held reads of B2 `STANDFMXH-L18-s1`, its s0 twin, JTN-L18-s0 / s1 and RSG-L18-s0. That is the per-decision
  table of the stand_b extract (`$DATA_DIR/runs/defect_fix/stand_b/ext_full_H_s1`, copied to `tmp/nu_defects/`, 35 MB), plus the loss classes of
  decision 393 ([easy_items_B2_s1.json](../../../alpasim_proxy/results/easy_items_B2_s1.json),
  `easy_items/misses_STANDFMXH-s1.csv.gz`).
- Files: [tables.md](tables.md), [report.json](report.json), and [stop_windows.csv.gz](stop_windows.csv.gz) (per stop-approach window).
- Run:
  - `.venv/bin/python experiments/defect_fix/scripts/nu_defects.py --src tmp/nu_defects/ext_full_H_s1 --out experiments/defect_fix/results/nu_defects`
  - `tmp/pcs_local/.venv/bin/python experiments/defect_fix/scripts/nu_defects_price.py`

**Why the gates are in stored-plan form.** inertia.py and lead_break.py are counterfactual forwards on the decision's frames. The arena
reads did not dump frames, so here they become the same quantities measured on the stored plans. The cost of each defect is bounded by
its outcome windows anyway, and that bound decides whether a fix is worth the card-hours.

## Answer

**B2-s1 has both PAI defects in kind, but on nuPlan neither costs much.**

**1. Inertia.** On stop approaches the plan copies speed as much as on open road.
- The window set: the human comes to rest inside the window with no lead (142 windows, 541 decisions). The elasticity of the 4 s arc to
  the ego's speed is 1.00 there, against 1.07 on open road (ratio 0.94). d404's PAI ratio was 0.89. This is observational and not a
  counterfactual. JTN and RSG are the same (0.95-1.04).
- The failures are the PAI overrun pattern. On B2-s1's 21 failing stop approaches, the plan contains a stop in 14 % of the decisions,
  against 62 % on the 118 passing ones. The plan's end lands a median +6.1 m past the human's stop, against +0.2 m on the passing ones.
- But nuPlan's 5.5 s windows start from the human's state. Every stop approach is within 20 m of the stop, at a median ego speed of 2.3 m/s.
  PAI's overruns are at about 10 m/s and 30 m out, and that regime does not occur here.
- Price: the 18 zero windows among the frozen items are worth **+18 [+6, +52] board-mix / +36 [+13, +100] g15 PCS** if every one is set
  to 1. Only 4 of them are easy items; 10 are stratum b and 4 stratum c. They are d395's a1 windows (moving-corridor and straight-offroad
  zeros past the human's stop), and STAND-R's rolling rows already cover part of them, at a launch cost (d399 / d400 / d411).

**2. Lead-break.** It does not cost on nuPlan.
- B2-s1 has 2 AF-front windows out of 2 228 (JTN-s1 4, RSG 10), and none among the frozen items. It also has 9 rear-hit windows.
- Only 24 windows have a brake-needed decision. Under the constant-speed lead approximation, clear rates are low for every model (B2-s1
  0.09, JTN-s1 0.04, RSG 0.03), and only 1 of those windows ends at zero. The approximation is too crude to compare with PAI's 0.18 / 0.63,
  but the outcome count settles it.
- When following a lead, B2's plan barely brakes (a02 -0.03 against JTN -0.24). That is FREE-M's mark, and it is harmless here.

## Where B2-s1's points go (frozen 2 106 items, single-class fix to 1, d393's classes)

B2-s1 is at frozen PCS 1398 / 1526, mean 0.936, with 96 zeros and 86 partials. Garage is at 1525, and the local all-1 ceiling is 2251.

| cause | windows | zeros / partials | mean points lost | easy windows | PCS if fixed, board-mix / g15 |
|---|--:|--:|--:|--:|--:|
| partial: launch late / not launching | 47 | 0 / 47 | 28.0 | 5 | **+49 / +66** (easy 5 alone +28 / +30) |
| zero: static, red line, no lead | 12 | 12 / 0 | 12.0 | 9 | +30 / +56 |
| zero: turn window, offroad / corridor | 14 | 14 / 0 | 14.0 | 8 | +28 / +53 |
| zero: corridor, moving non-turn | 19 | 19 / 0 | 19.0 | 7 | +25 / +51 |
| zero: offroad, non-turn | 18 | 18 / 0 | 18.0 | 5 | +24 / +50 |
| partial: rear-ended | 8 | 0 / 8 | 6.4 | 1 | +22 / +20 |
| unsolvable: human's own track offroad | 24 | 23 / 1 | 23.4 | 0 | +6 / +15 |
| partial: free-road ratchet / slow | 11 | 0 / 11 | 1.4 | 3 | +5 / +6 |
| zero: static, other cue | 9 | 9 / 0 | 9.0 | 1 | +4 / +8 |
| stop / lock, near-miss, follow, other slow, AF-side | 20 | 1 / 19 | 2.6 | 8 | each <= +3 / +4 |
| *cross-cut: stop-approach overrun (inertia), zeros* | *18* | *18 / 0* | *18.0* | *4* | *+18 / +36* |
| *cross-cut: lateral group (moving corridor + turn + straight offroad), easy only* | *20* | | | *20* | *+80 / +178* |

- The PCS sits in the 47 easy misses (31 zeros and 16 partials, 37 mean points). The other 135 misses lose 97 mean points, but they are
  items garage also fails (a around 0.1) and are worth little.
- The launch class is the largest single class. 38 of its 47 windows are stratum b (median a 0.08); 27 have a lead at about 4 m and 38 a light.
- The lateral zeros are mostly overspeed in tight right turns and past stop points (d395). The red-line class is the creep from standing (d393).

## Verdict on a bet tonight: none

No lever that is data or one simple signal is aimed at a class worth the card-hours.
- **Each class fixed in full is worth +25-30 board-mix**, launch apart. A realistic partial fix is +10-15. The seed spread between B2-s0 and
  B2-s1 is 29 PCS (1369 / 1398), so a 1 000-3 000-step pilot cannot resolve a gain of that size.
- **The two PAI data levers have no cheap nuPlan form.**
  - Human-state imitation rows: nuPlan already trains on human trainval rows. The render-domain version is the MTGS human-rows bet, which
    the user postponed.
  - Time-dilated stretch rows: they need frames at stretched times. MTGS renders are postponed. The 2 Hz real keyframes are exact only at
    f = 2, with an 8 s label. Their target here, the stop-approach overrun, is worth +18 / +36 at full fix.
  - A stretch label also crosses a curve at f x speed, so it does not teach slowing for tight turns. On PAI, J's turn ratio did not move (0.942).
- **The simple signals for the big classes are already closed.**
  - Red line: STAND-R / R2 / FR fix it but cost launches at every dose (d399, d400, d411).
  - Turn and corridor overspeed: the TURN-CAP and TIGHT-IN gates fail, because they fire on passing turns (d396, d397, d401). The plan does
    not read curvature, and that habit is shared by every model (d403).
  - Onset flattening: no nuPlan effect (d411).
  - Best-of-N: failed (d379).
- **Launch** is the largest class, but it is mostly low-discrimination items. Every lever toward it trades against the red-line and static
  classes (d365, d378, d399-400). d411 also shows that a plain continuation of B2 already loses launches.

GPU used: 0 card-hours. B2-s1 stays the nuPlan candidate unchanged.
