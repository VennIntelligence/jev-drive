# BODY1: navtest own-plan reads re-read on warp frames (correction of 2026-10-11)

**What was wrong.** `scripts/bd4_g3.py: plans()` is the one function through which this lane computes a checkpoint's own plan on a cache
directory (`bd4_g3.py` itself and `prog_ol.py states`, whose dumps `$DATA_DIR/runs/body1/prog/ol_<name>.parquet` / `_plans.npy` feed
`route_ol.py`, `route_gate.py`, `shape_gate.py`, `sdrop_report.py`, `prog_cl.py`, `shape_cl_report.py`). Until e7747c0b it opened every board
directory (`lb_*`) on GIMM frames whatever the checkpoint. All `P2H10*` checkpoints of this lane are trained on warp frames, so their navtest
own plans were computed on frames they were never trained on: 1.74 m mean difference to the plans `jevdrive.bench` scores, ADE to the log
1.60 m against 0.63 m. Hold-log, validation-part and `bd4` states were always opened on warp frames; `jevdrive.bench` (EPDMS, navhard, the
turn-oracle replay) and the closed-loop runs never went through this function.

**The re-read.** No training, no new experiment. Same readers, plans from the bench's archived plan files where they exist (`--bench-plans`),
a CPU forward on the warp cache otherwise; chains `scripts/navtest_warp_chain.sh`, `navtest_warp_chain2.sh` (e7747c0b, b26f4342), both DONE
2026-10-10 23:16 / 23:53 box time, CPU jobs of the pool. Tables: [navtest_warp/](navtest_warp/) (`sdrop/`, `route/`, `shape/`, `loss/`,
`prog/`, `prog_cl/`, `shape_cl/`, `sh30s/`, `check/`); the dumps stay on the box (`runs/body1/prog/ol_*_w.*`). The tables of the first
read stay where they were (`results/{sdrop,route,shape,loss,prog,sh30s}/`), as the superseded record.

**Check of the fixed reader** (`check/`): four checkpoints forwarded on warp frames on CPU against the bench's archived plans of the same
checkpoints: W2 at > 45 deg 141 / 97 / 111 / 83 against 141 / 97 / 110 / 84, mean lateral at 4 s within 0.0013 m. Independent of this lane,
`experiments/corridor` (decision 245) computed W2 of `P2H10S-P-s0` from the bench's plans as 141: the same number.

All intervals are 95 %, logs resampled (`jevdrive.stats.paired(groups=log)`), navtest 12 146 tokens / 136 logs, > 45 deg 1 517 tokens.
W2 = tokens whose own plan is more than 2 m outside the logged path within 4 s. Verdicts: **re-established** (same conclusion; the number
may have moved), **changed** (the conclusion or a registered verdict is different), **not re-established**.

## Decision 244 (S-DROP), [sdrop.md](sdrop.md)

| Claim | old (off-protocol) | warp frames | verdict |
|:--|:--|:--|:--|
| W2, seed 0 / 1: base | 45 / 46 | 84 / 79 | changed number |
| W2: S (`P2H10S-F`) | 58 / 65 (+13 / +19) | 116 / 118 (+32 / +39) | re-established (wide) |
| W2: noA | 54 / 63 (+9 / +17) | 111 / 114 (+27 / +35) | re-established (wide) |
| W2: noB | 45 / 47 (0 / +1) | 85 / 82 (+1 / +3) | re-established (not wide) |
| W2: noC | 55 / 54 (+10 / +8) | 97 / 98 (+13 / +19) | re-established (wide) |
| Question 2, "only the road hinge on hinge-only rows" | fails | fails | re-established |
| lateral at 4 s, > 45 deg, S - base (m) | +0.224 [+0.186, +0.267] | +0.123 [+0.085, +0.163] | re-established, about half the size |
| noA - base | +0.168 [+0.130, +0.211] | +0.102 [+0.062, +0.145] | re-established, smaller |
| noB - base | +0.032 [+0.024, +0.040] | +0.012 [+0.007, +0.018] | re-established, smaller |
| noC - base | +0.123 [+0.110, +0.139] | +0.059 [+0.044, +0.074] | re-established, about half |
| share of the shift kept: noA / noB / noC | 0.75 / 0.14 / 0.55 | 0.83 / 0.10 / 0.48 | noC's common-rule label "not needed" -> "carries" (on the 50 % boundary in both reads) |
| the two half arms add up to more than S | 0.168 + 0.123 > 0.224 | 0.102 + 0.059 > 0.123 | re-established |
| W2 share, S - base | +1.1 pp [+0.4, +1.7] | +2.3 pp [+1.1, +3.8] | re-established, larger |
| absolute mean lateral at 4 s, base / S (m, + = outside) | -0.54, -0.58 / -0.36, -0.31 | +0.29, +0.27 / +0.38, +0.43 | changed (sign) |
| navtest own-plan boundary rate, S - base | -0.0065 [-0.0103, -0.0033] (0.0365 -> 0.0300) | -0.0029 [-0.0046, -0.0013] (0.0172 -> 0.0143) | re-established; the level is half |
| the same, noB - base | -0.0010 [-0.0021, -0.00004] | -0.0003 [-0.0009, +0.0003] | not re-established |
| the same, noC - base | -0.0022 [-0.0041, -0.0005] | -0.0005 [-0.0014, +0.0004] | not re-established |
| navtest own-plan agent rate, S - base | -0.0002 [-0.0014, +0.0010] | -0.0006 [-0.0014, +0.0001] | re-established (no arm moves it) |
| navtest arc ratio pooled, S / noA / noB / noC | 0.9990 / 0.9997 / 1.0024 / 0.9969 | 1.0012 / 1.0011 / 1.0001 / 1.0008 | re-established (no line missed) |
| "shortening behind a lead of 0.3 to 0.4 % in the arms with rows" (navtest lead) | 0.9964 / 0.9966 / 1.0020 / 0.9969 | 1.0006 / 1.0003 / 0.9999 / 1.0006 | not re-established |
| navhard, navtest EPDMS, replay, every hold read | | | not affected |

## Decision 234 (`P2H10R`, pilot gate), [route_pilot.md](route_pilot.md)

| Claim | old (off-protocol) | warp frames | verdict |
|:--|:--|:--|:--|
| (C-a) W2, `P2H10R-Pb25-s0` against the switch-off pilot, line <= +5 | 67 against 54, not met | 138 against 110, not met | re-established (missed by +28, was +13) |
| positive control `P2H10S-P-s0` | 75 against 54 | 141 against 110 | re-established (fails) |
| negative control "A on on-log rows only" | 55 against 54 | 115 against 110 | re-established (passes, exactly at the line) |
| pilot drop-one "B + C without A" | 71 against 54 | 141 against 110 | re-established |
| (C-c) leaves the 4 m tube, navtest | 31 against 47, met | 19 against 25, met | re-established |
| W2 at full scale, `P2H10S-F` s0-3 against `P2H10-F` s0-3 | 58 / 65 / 59 / 65 against 45 / 46 / 47 / 49 | 116 / 118 / 119 / 126 against 84 / 79 / 87 / 91 | re-established |
| lateral shift of `P2H10S-F`, seeds 0 / 1 (m) | +0.18 [+0.14, +0.23] / +0.27 [+0.23, +0.31] | +0.09 [+0.05, +0.14] / +0.15 [+0.12, +0.19] | changed: "0.2 to 0.4 m per plan" is about 0.1 m on navtest (hold `bd4` +0.31 to +0.39 m not affected) |
| base seed spread of W2 on navtest (basis of the +5 tolerance) | at most 4 tokens | at most 12 tokens | changed |
| variant's lateral shift against the switch-off pilot (m) | +0.184 [+0.138, +0.235] | +0.124 [+0.075, +0.176] | re-established, smaller |
| mean lateral at 4 s, variant / `P2H10S-P` / switch-off (m) | -0.29 / -0.36 / -0.48 | +0.43 / +0.42 / +0.31 | changed (sign) |
| navtest arc ratio of the variant ("a second line to clear at full scale") | 0.9915 [0.9904, 0.9926] | 0.9993 [0.9987, 1.0000] | not re-established |
| band selection, contact / ADE / slope / hold arc lines, (C-b) | | | not affected |

## Decision 232 (`P2H10S`, G3), [shape_pilot.md](shape_pilot.md)

| Claim | old (off-protocol) | warp frames | verdict |
|:--|:--|:--|:--|
| **G3 (b), seed 0: navtest own-plan agent positives, arm against base** | **149 against 148, +0.00008 [-0.00111, +0.00136], not met** | **135 against 140, -0.00041 [-0.00127, +0.00041], met** | **changed: the registered miss is not re-established** |
| G3 (b), seed 1 | 138 against 144, met | 132 against 142, -0.00082 [-0.00183, +0.00009], met | re-established |
| G3 (b), seeds 2 / 3 | 143 against 151, 147 against 151, met | 134 against 141, 130 against 137, met | re-established |
| G3 (b), boundary rate, seeds 0 / 1 | -16.6 % / -19.1 % | -14.9 % / -18.6 % (0.0146 against 0.0171, 0.0141 against 0.0173) | re-established |
| G3 (e), navtest arc pooled, seeds 0 / 1 | 0.9996 / 0.9984 | 1.0017 [1.0012, 1.0021] / 1.0008 [1.0004, 1.0012] | re-established (met) |
| the previous recipe (`P2H10B-F`) on the same line | 0.9905 / 0.9905 | 0.9985 [0.9977, 0.9993] / 0.9980 [0.9972, 0.9988] | changed: above the 0.995 line; shorter than the base by 0.15 to 0.2 %, not 1 % |
| navtest lead, `P2H10S-F` against `P2H10B-F` | 0.997 / 0.996 against 0.982 / 0.984 | 1.0007 / 1.0004 against 0.9931 / 0.9931 | re-established (direction), smaller |
| G3 (e), seed 3, navtest open | 0.9949 [0.9935, 0.9963], not met | 1.0001 [0.9993, 1.0008], met | changed |
| "the navtest agent rate does not fall without timing" | 149 / 138 / 143 / 147 against 148 / 144 / 151 / 151 | 135 / 132 / 134 / 130 against 140 / 142 / 141 / 137 (-4 to -7 %, no single interval excludes 0) | re-established |
| hold-log rates (a), slope (c), EPDMS (d), pilot gate, code checks, base seed spread in closed loop | | | not affected |

With (b) met on seed 0, every line of G3 is met on both registered seeds on the corrected read. The arm was stopped on 2026-10-10 on a
number that came from the reader, not from the checkpoint. This page does not reopen the arm: the registered closed-loop read was never
made, a descriptive four-seed closed loop exists (decision 233), and whether the registered read is now owed is for the main session.

## Decisions 229, 230, 236 (the same reader)

| Claim | old (off-protocol) | warp frames | verdict |
|:--|:--|:--|:--|
| 229, G3 (b) of `P2H10B-F`, agent positives, seeds 0 / 1 | 139 against 148, 132 against 144 (intervals including 0) | 115 against 140, -0.0021 [-0.0031, -0.0010]; 119 against 142, -0.0019 [-0.0031, -0.0007] | re-established (met); the fall is 16 to 18 % with intervals excluding 0, not 6 to 8 % |
| 229, G3 (b) boundary rate | -14.8 % / -18.4 % | -13.9 % / -13.8 % | re-established |
| 230, open-loop navtest arc ratio of `P2H10B-F` | 0.9905 / 0.9905 | 0.9985 / 0.9980 | changed (see 232) |
| 230, by proximity group (seed 0 / 1): lead; open | 0.9822 / 0.9836; 0.9931 / 0.9927 | 0.9931 / 0.9931; 0.9998 / 0.9995 | changed: the open-loop shortening is behind a lead only |
| 230, share of the open-loop shortening in lead states | 34 % / 31 % | 83 % / 63 % | changed |
| 230, closed-loop progress loss by group of the base plan (lead share; lead / open difference) | 46 %; -0.0296 [-0.0395, -0.0206] / -0.0099 | 49 %; -0.0314 [-0.0407, -0.0238] / -0.0105 [-0.0171, -0.0040] | re-established (grouping moved 2 to 20 scenes per group) |
| 230, closed-loop arc ratios, gradient split, zero-flip accounting | | | not affected (served plans, trainer) |
| 233, closed-loop progress by group (`shape_cl/report.md`): lead / open | +0.0115 [+0.0038, +0.0203] / +0.0078 [+0.0032, +0.0126] | +0.0094 [+0.0021, +0.0170] / +0.0093 [+0.0046, +0.0144] | re-established |
| 236, navhard and HUGSIM | | | not affected (bench) |

## Result pages corrected in place

[sdrop.md](sdrop.md), [route_pilot.md](route_pilot.md), [shape_pilot.md](shape_pilot.md) (G3 (b), the navtest arc rows of (e) and of the pilot
table, seeds 2 and 3), [loss_g3.md](loss_g3.md) (G3 (b) of Amendment 5), [progress_diagnosis.md](progress_diagnosis.md) (open-loop navtest arcs,
the group tables), [shape_closed_loop.md](shape_closed_loop.md) (lead / open split), the lane README and its `experiments/INDEX.md` line, the
Chinese page `research/body1/index.html`, decisions 229, 230, 232, 233, 234, 244 and their lines in `research/decisions.md`. Each replaced
passage names the first read's number. Figures redrawn the same night from the warp-frame tables (`shape_fig.py --res`, `prog_cl.py figs --warp --warp-cl`): `figs/shape/shape_gate.png`
(navtest points), `figs/prog/ol_arc.png` (left panel) and `figs/prog/cl_progress.png` (group colours); the old files are superseded in git history. Identity of the fixed reader over all
12 146 tokens (8 poses): forward on warp frames against the bench's archived plans 0.004 m mean, 0.028 m at the 99th percentile, 0.07 m at most
(four checkpoints); the first read's dumps against the same archived plans 1.34 to 1.45 m mean, 6 m at the 99th percentile.

## Open after the re-read: the plan served at decision 0 in AlpaSim

`prog_cl.py` checks its placement of closed-loop traces by comparing the 4 s arc of the plan served at decision 0 of each AlpaSim scene with the
open-loop plan at the scene's navtest token (`cl_summary.json`, `k0_arc_vs_openloop`, the base runs of the 1 400 (seed, scene) pairs). Against the first read's
off-protocol plans the ratio was 1.010 (r = 0.90); against the scored plans it is **0.909** (r = 0.92). So the plan the simulator serves first is
about 9 % shorter than the plan `jevdrive.bench` scores at the same token, and about as long as the GIMM-frame plan (whose arc was 0.892 of the
logged one; on warp frames the base is 0.997). Two readings, not separated here: the first decision in the simulator differs from the token for
its own reasons (rendered frames, history at start, the driver's serving path), or the closed-loop driver gives these warp-trained checkpoints
frames closer to GIMM than to warp. The second would concern every AlpaSim number of the `P2H10*` family, not this lane only. A check that
separates them: forward the first frames the simulator rendered through the checkpoint on both front protocols and compare with the served plan.

## Other consumers checked

| Consumer | reads | affected |
|:--|:--|:--|
| `bd4_g3.py --set hold / val`, `bd4_select.py`, `bd4_pilot_read.py` | hold / validation states | no |
| `bd4_g3d.py`, `sdrop_report.py` bench and replay parts, `jevdrive.bench` | bench plans | no |
| `experiments/corridor/scripts/head1_pilot_report.py --widening` (decision 245, W2 99 / 120 against 141 / 150) | bench plans on the checkpoint's own frames | no; its base count 141 equals the warp re-read |
| `experiments/corridor/scripts/head1_aux.py` | `lb_navtest` on warp frames, explicit | no |
| `prog_cl.py`, `shape_cl_report.py` | closed-loop logs, grouped by the base plan at the navtest token | grouping only; re-read in `prog_cl/`, `shape_cl/` |
