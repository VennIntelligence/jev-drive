# LON-ANCHOR: station-anchored two-sided arc term (2026-10-20)

- Prereg: [../../plans/2026-10-20-lon-anchor-prereg.md](../../plans/2026-10-20-lon-anchor-prereg.md) (commits 0331347b, 9bafbe89, before any output).
- Code: `lib/nr1_anchor.py` (label rule + term), `rm_train --on-anchor W` through `lib/nr1_lat.py` (default 0: nothing loaded),
  `scripts/lon_anchor.py` / `lon_anchor.sh` (labels, gate, draw, dose calibration, inertia read-out).
- Box output: `$DATA_DIR/runs/defect_fix/lon_anchor/`; labels `$DATA_DIR/runs/op_parity/cache/on1_JTCP/lon_anchor.npz`.

## Status: stopped at G1 (no training)

**Labels** ([labels.json](labels.json)). 171 054 S rows in on1_JTCP. 143 144 are labelled: nuPlan 80 850 of 87 012, PAI 62 294 of 84 042.
Lead skips: 16 655 (nuPlan 6 091, PAI 10 564), plus 164 PAI rows where the human's lead could not be measured. Undefined labels: 8 273 log_end
(almost all PAI) and 2 818 no_reach.

**G1: FAIL as registered** ([gate.md](gate.md)). The line is 1 %. 6 656 of 143 144 rows (4.65 %) are active, meaning max |d| > 0.5 m
when the human's station-matched path is used as the plan. nuPlan is 4.82 % and PAI 4.43 %.

Diagnosis (read after the gate, same labels):
- The failing rows are rows where the ego is off the human's line. In 99.9 % of active rows d_j is the same at all 8 poses, and active rows
  have median |e0| 2.5 m. The share is 22 % for |e0| > 0.5 m and 0.1 % for |e0| <= 0.5 m.
- The constant is the length of the first chord, from the ego's origin onto the human's line, minus s_t,1. So the registered construction
  puts the plan on the human's line in the first 0.5 s, and that lateral jump counts as along-track arc.
- Remove that first-chord excess and 8 rows are active (0.01 %). The remainder is the 0.1 m storage resolution of `gpath`.
- So the label and the trainer's term agree. The registered G1 plan is not a plan that starts at the ego.
- What this says about the term itself (not gated): about 20 % of S rows have |e0| > 0.5 m. On those rows any plan that rejoins the human's
  line gets extra arc from the lateral motion, which the along-path target does not count.

**Reported statistics (not gated), on the stored plans.**
- 96 % of labelled rows are active.
- nuPlan: plans are mostly shorter than the anchored target (75 % too slow, d4 median -3.1 m). On these rows the two-sided term mainly
  pushes faster, which is the JT failure direction (d352 / d353).
- PAI stop_nolead: plans are longer than the target (d4 median +0.9 m, 90 % +12.3 m).
- Clock conflicts. 1 591 rows have a target that ends standing while the human was moving at the same clock time.
- 4 120 rows go the other way: the target moves >= 2 m while the human stood at the same clock time. 3 203 of these are PAI rows: launch 931,
  stop_nolead 1 262, turn_sharp 1 010. This is the standstill ambiguity: at a stand, a few cm of station maps to a time after the human's launch.

GPU: 0 card-hours (CPU labels 1 min, gate 1 min).
