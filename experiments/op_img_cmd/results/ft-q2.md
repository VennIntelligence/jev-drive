# Q2: does a light fine-tune make Cinque follow a route drawn in the image? (navtrain, 2026-10-03)

Pre-registration and deviations: [../plans/2026-10-04-img-cmd-ft-prereg.md](../plans/2026-10-04-img-cmd-ft-prereg.md). Full tables: [ft/](ft/)
(`<model>_vs_O_<pool>.{md,csv}`, `_guards.json`). Scripts: `scripts/img_ft_{data,bank,train,eval}.py`, `scripts/img_ft_chain.sh`.

**Setup.** op_adapt_l's PyTorch port of Cinque (L3's recipe, imported): stage 1-3 frozen, stage 4 + off-policy plan pathway trained (126.7 M
parameters). Training pool: 1 500 junction frames (decision-93 pair inventory) + 300 lane-keeping frames from 375 navtrain logs that hold no eval
token (splits `navsim/img-ft-{train,dev}`, dev = 10% of logs), on op_lb's frame protocol (4 keys + 6 GIMM frames, data `lb_imgtrain`).
Batch of 48: 24 overlay rows of the two trained families (band, barrier) for every exit class, target = that branch's route with the logged speed profile;
16 no-overlay rows distilled to the original; 8 band-on-own-lane rows pinned to the original's plan. Eval: the Q1 set (385 junction frames / 247 logs,
293 lane-keeping frames), same overlays and metrics as `img_report.py`, at 4 s, paired cluster bootstrap by log. The port on the 5 Hz 10-frame lattice
reproduces the TensorRT zero-shot run: `none` 4 s point median 1.7 cm (p90 4.7 cm) apart, n = 678; the family table is the same to 0.01
([ft/port-O_zeroshot_effects.md](ft/port-O_zeroshot_effects.md)).

**Models.** `ft-s0` (lr 3e-5, 2 000 steps, the pre-registered first run) underfit: dev uptake band 0.06 and barrier 0.21. The pre-registered iteration rule
then gave `ft2-s0` (lr 1e-4, 3 000 steps), the primary model. `ft3-s0` = `ft2` with the distillation weights ×4. It was added after the `ft2` eval was read,
to see whether the drift cost can be removed; it is exploratory. Training takes ~9 min per model on one card (5.7 it/s, 12.4 GB).

## Primary: ft2 vs the original (eval, 385 junction frames, tau = 4 s)

Uptake = fraction of a full branch switch; correct = fixed-set correct (chance ~0.5); d = ft2 − O, 95% paired cluster bootstrap by log.

| family | role | uptake O | uptake ft2 | d uptake | correct O | correct ft2 | Delta ft2 − O (m) | verdict (prereg) |
|:--|:--|--:|:--|:--|--:|:--|:--|:--|
| band | trained | 0.01 | 0.54 [0.46, 0.63] | +0.53 [+0.45, +0.62] | 0.52 | 0.81 [0.76, 0.86] | +4.04 [+3.28, +4.83] | learned |
| barrier | trained | 0.12 | 0.64 [0.57, 0.71] | +0.52 [+0.45, +0.59] | 0.59 | 0.91 [0.87, 0.94] | +3.96 [+3.35, +4.59] | learned |
| lines | held out, paint | 0.00 | 0.09 | +0.09 [+0.06, +0.13] | 0.52 | 0.59 | +0.63 [+0.36, +0.94] | generalizes (small) |
| arrow_road | held out, paint | 0.00 | 0.04 | +0.04 [+0.02, +0.06] | 0.51 | 0.54 | +0.19 [+0.07, +0.31] | generalizes (small) |
| sign | held out, paint | −0.00 | −0.00 | +0.00 [−0.00, +0.00] | 0.51 | 0.52 | +0.00 [−0.00, +0.01] | no |
| cones | held out, block | 0.05 | 0.21 | +0.16 [+0.11, +0.21] | 0.52 | 0.66 | +1.07 [+0.78, +1.38] | generalizes |
| wall | held out, block | 0.07 | 0.13 | +0.05 [+0.02, +0.09] | 0.54 | 0.64 | +0.34 [+0.15, +0.55] | generalizes (small) |
| fill_grey | held out, fill | 0.00 | −0.01 | −0.01 [−0.04, +0.01] | 0.50 | 0.52 | −0.14 [−0.25, −0.04] | no |
| fill_grass | held out, fill | 0.11 | 0.06 | −0.05 [−0.09, −0.02] | 0.59 | 0.56 | −0.29 [−0.49, −0.08] | worse than O |
| combo | band + fill_grass + barrier (contains trained) | 0.14 | 0.81 | +0.67 [+0.58, +0.77] | 0.62 | 0.95 | +5.85 [+5.02, +6.67] | — |

The uptake gain holds in every speed bin (band: moving 0.62, low 0.43; barrier: moving 0.69, low 0.57). It is spread over all three
commanded classes: toward L / S / R +2.3 / +1.8 / +2.1 m (band) and +3.0 / +1.8 / +2.6 m (barrier).

**Guards (prereg).**

| guard | line | ft2 | pass |
|:--|:--|:--|:--|
| `none` plan drift vs O, median (0-5 s mean L2) | ≤ 0.10 m | junction 0.29 m, lane-keeping 0.18 m | **no** |
| lane-keeping `none` lateral error at 3 s, ft2 − O | ≤ +0.10 m | −0.002 [−0.013, +0.009] m | yes |
| overlay-induced lateral error on lane-keeping frames | ≤ +0.10 m | band +0.01, lines / arrow_road / sign ≤ +0.00 m | yes |
| no-overlay junction plan moves toward the taken branch, ft2 − O | CI contains 0 or abs ≤ 0.20 m | −0.07 [−0.26, +0.08] m | yes |

**Side effects (not pre-registered as guards).**
- The overlays also act as a go signal. At 4 s, O's plan falls ~3.0 m short of the logged position on `none` (mean), and ft2's own `none` falls 2.7 m short.
  With the overlay for the taken branch, ft2's plan lands on the logged distance (band −0.3 m, barrier −0.3 m). The training targets carry the
  logged speed, so the fine-tune also removed the blocks' braking (barrier dx vs `none`: O −2.4 m, ft2 +2.3 m).
- `band_all` (paint on every branch, no route information): ft2's 4 s point moves sideways more than O's (|dy| median 0.77 m vs 0.17 m) but picks no side on
  average (mean +0.06 m). Its plan is also 3.8 m longer than ft2's own `none`.

## Secondary arms

| model | band uptake (eval) | barrier uptake (eval) | correct band / barrier | held-out gains with CI > 0 | `none` drift junction / lane-keeping |
|:--|:--|:--|:--|:--|:--|
| ft-s0 (lr 3e-5, 2 000 steps) | 0.10 (+0.09 [+0.07, +0.12]) | 0.21 (+0.09 [+0.04, +0.13]) | 0.54 / 0.64 | lines +0.04, fill_grass +0.04 | 0.44 / 0.21 m |
| ft2-s0 (primary) | 0.54 (+0.53) | 0.64 (+0.52) | 0.81 / 0.91 | lines, arrow_road, cones, wall | 0.29 / 0.18 m |
| ft3-s0 (×4 distillation, exploratory) | 0.34 (+0.33 [+0.27, +0.39]) | 0.58 (+0.46 [+0.39, +0.54]) | 0.69 / 0.86 | lines, arrow_road, cones, wall | 0.19 / 0.14 m |

ft3 missed its dev line (dev drift 0.21 m, band dev uptake 0.25). Stronger distillation trades band uptake for drift and does not reach the 0.10 m guard.

## Reading

- **Q2 = yes for the trained drawings.** On held-out logs, band (paint = route) and barrier (block the other branches) both pass the pre-registered
  "learned" line: uptake 0.54 and 0.64, fixed-set correct 0.81 and 0.91, up from 0.01 / 0.12 and 0.52 / 0.59. That is about 4 m of branch-directed
  movement at 4 s. Band's uptake CI lower bound is 0.46, so the point estimate clears 0.5 but the CI does not.
- **Generalization is weak.** By the pre-registered count, both groups have a held-out family with a gain CI > 0 (paint: lines, arrow_road; block: cones, wall),
  so the mechanical verdict is "reads the route". But the gains are 1/5 to 1/10 of the trained ones (0.04-0.16 uptake). The largest transfer is to the drawings
  that look most like a trained one (cones ~ barrier, lines ~ band edges). The sign does not move at all, and fill_grass gets worse than zero-shot. The fair
  summary: the model mostly learned the two drawings it saw, with a little transfer to similar-looking cues. It did not learn a general "read the route" skill.
- **Cost.** The unperturbed plan drifts 0.29 m (median) from the original at junctions, which fails the 0.10 m guard. Lane keeping is untouched, and
  without an overlay the model does not guess the branch. ×4 distillation lowers the drift only to 0.19 m and costs a third of band's uptake.

## Limits

1 seed per arm; nav only (no WOD / CARLA rows, no CARLA check); overlays are not occluded by real vehicles; road-plane assumption; the eval frames are
mostly near the stop line at low speed (64 stop / 169 low / 151 moving); the training pool is faster (median 4.3 m/s). ft3 was chosen after the ft2 eval
was read.
