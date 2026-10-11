# onpol_long: closed-loop training at 18 decisions, round 1

status: live
decisions: 264 (inputs 259, 262)
index: S + corridor rows at 18 decisions: not adopted; P offroad zeros 79 vs 24, Y at-fault 80 vs 79.5

**Question.** At 18 decisions (token + 8 s) in local AlpaSim nuPlan, do on-policy rows of the base's own rollouts on the 1031 train scenes,
carrying only consequence hinges on the model's own plan (S: swept footprint against vehicles, road-and-lane raster; K: 1.5 m dead band to
the logged path; C: lane-centre offset if it passes its label gate), give fewer zeros than the same continued training without rows
(ON-NONE) on the 460 held scenes, with at-fault zeros and GT distance not worse?

**Conclusion.** Neither family's arm is adopted (decision 264). The lane-centre hinge C was dropped before training (the logged driving
itself violates it on 46 % of the rows), so the arm is S + corridor hinge K. On the P2H10S-F base it is strongly negative (0.590 against
ON-NONE 0.771, at-fault zeros 97.5 against 35.5, offroad zeros 79 against 24, corridor zeros 22 against 47, drives 25 % less distance);
on the YR10m10-F base it has fewer zeros (96 against 120, -24 [-47.9, -2.5], all corridor) and a lower GT distance (-0.50 m) but fails the
hard gate by half a scene (at-fault zeros 80 against 79.5) and ties on mean score. The same continued training without rows (ON-NONE) is
itself worse than the base on both families (-0.028 / -0.092).

**Next.** Amendment 3: the S-only arm on the same 18-decision rows (separates K); PAI transfer read on the Tokyo box (Amendment 1, running).

**Read more.** [plans/2026-10-12-ol1-prereg.md](plans/2026-10-12-ol1-prereg.md); the plan: [research/next-round/closed-loop.md](../../research/next-round/closed-loop.md);
the row and trainer code it reuses unedited: [../onpol_nuplan/README.md](../onpol_nuplan/README.md); the 18-decision driver path:
[../alpasim_long/README.md](../alpasim_long/README.md).

## Held 460 at 18 decisions (2026-10-12 morning; 452 scenes after 8 route-fold scenes, two training seeds from the s0 base)

[results/read/report_P.md](results/read/report_P.md), [results/read/report_Y.md](results/read/report_Y.md) (per-scene csv and json beside them).

| family | | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | drift slope (m / decision) |
|:--|:--|--:|--:|:--|--:|--:|--:|
| P | arm `PSKC-L18` (S + K) | 0.590 | 118.0 | 18.5 / 79.0 / 22.0 | 97.5 | 1.292 | 0.093 |
| P | ON-NONE `PNONE-L18` | 0.771 | 81.0 | 11.5 / 24.0 / 47.0 | 35.5 | 1.674 | 0.177 |
| P | base `P2H10S-F-s0` | 0.799 | 68 | 8 / 24 / 39 | 32 | 1.588 | 0.150 |
| Y | arm `YSKC-L18` (S + K) | 0.694 | 96.0 | 18.0 / 62.0 / 17.0 | 80.0 | 1.243 | 0.081 |
| Y | ON-NONE `YNONE-L18` | 0.693 | 120.0 | 23.5 / 56.0 / 51.5 | 79.5 | 1.746 | 0.191 |
| Y | base `YR10m10-F-s0` | 0.785 | 74 | 10 / 33 / 35 | 43 | 1.495 | 0.142 |

| a - b (log-clustered 95 % CI) | mean score | zeros | at-fault zeros | GT distance (m) | fixed / broken (null max) |
|:--|:--|:--|:--|:--|:--|
| P arm - ON-NONE | -0.181 [-0.217, -0.137] | +37.0 [+16.0, +54.5] | +62.0 [+38.7, +86.7] | -0.383 [-0.590, -0.190] | 24 / 51 (1) |
| P ON-NONE - base | -0.028 [-0.050, -0.011] | +13.0 [+4.8, +22.9] | +3.5 [-3.2, +11.6] | +0.087 [+0.012, +0.163] | 11 / 18 |
| Y arm - ON-NONE | +0.001 [-0.044, +0.052] | -24.0 [-47.9, -2.5] | +0.5 [-21.5, +21.4] | -0.503 [-0.660, -0.358] | 41 / 33 (8) |
| Y ON-NONE - base | -0.092 [-0.127, -0.060] | +46.0 [+30.8, +62.6] | +36.5 [+21.4, +53.1] | +0.252 [+0.116, +0.392] | 15 / 38 |

Registered adoption line (fewer zeros, at-fault zeros not above, GT distance not larger, all against ON-NONE): P fails the first two, Y
fails the hard gate (80 against 79.5). CL-1 as written: not met on either. Descriptive: the arms drive less (P 35 m against 46 m, Y 40 against
44) and their extra at-fault events start early (first at-fault step at decision <= 10: P 52 / 65 against 15 / 14). On the arms' own held
rollouts the corridor hinge K fires on the served plan about as often as for ON-NONE and the base (P 0.30 against 0.26 / 0.25, Y 0.27
against 0.30 / 0.24; [results/read/mech_P.md](results/read/mech_P.md), [mech_Y.md](results/read/mech_Y.md)): the fit to the pool rows
(K's training value 0.22 -> 0.03) did not carry to held states. Dev ADE after 10 000 steps: arms 0.545-0.548 against ON-NONE 0.524-0.529.

Pools and gates ([results/gates/](results/gates/)): `on1_L18Y` / `on1_L18P` 17 306 / 17 325 rows from 18 352 decisions each; R1 0.9991
(one ulp), interpreted control 100 / 100, wrong seed 0.016; R2 after Amendment 2 (one train rollout without time alignment dropped) actors
within 0.5 m 0.99998, ego <= 0.018 m. K fires on 21.9 % / 21.7 % of the served plans and 0.6 % of the logged futures; C on 58 % / 61 % and
46 % (dropped). ![K and C on pool rows](figs/bev_kc_L18Y.png) `figs/bev_kc_L18{Y,P}.png`: rows where K or C is largest and random rows;
check that the logged path (dashed), its 1.5 m band and the lane-centre band (grey) sit on the served plan's road. Frame dumps: `cur`
removed after the tokens (35.1 -> 5.6 GB per family), JPEG / plan / ego kept.

## Chain

```bash
bash experiments/onpol_long/scripts/ol_chain.sh submit-roll <base tag> <after ids>        # 6 rollout jobs (cl fanout), 18 decisions, frame dump
bash experiments/onpol_long/scripts/ol_chain.sh submit-rest <base tag> Y|P <rollout ids> <after ids for ON-NONE>
#   geo (on_rows.py geo + ol_labels.py build) -> tok (on_rows.py tok + ol_labels.py gate) -> strip (drop dump frames) ;
#   tok -> train <F>SKC-L18-s{0,1} -> read1 ; train <F>NONE-L18-s{0,1} -> read1 ; all reads -> report (ol_report.py)
```

State on the box: `$DATA_DIR/runs/onpol_long/{roll,rows,read,report,pool}`, job ids in `jobs.txt`; pools `runs/op_parity/cache/on1_L18{Y,P}`,
rows `runs/onpol_nuplan/rows/L18{Y,P}` (on_rows.py's own locations).

<!-- files:begin -->
## Files

- `2026-10-12-ol1-prereg.md` (plans): ONPOL-LONG 第 1 轮：18 决策闭环 on-policy 行 + …
- `ol_chain.sh` (scripts): on-policy rows at 18 decisions, hinge …
- `ol_labels.py` (scripts): ONPOL-LONG labels K and C on an …
- `ol_kc.py` (lib): ONPOL-LONG on-policy terms K and C on …
- `ol_report.py` (scripts): ONPOL-LONG round 1 read-out
- `on_rows.py` (experiments/onpol_nuplan/scripts): finished on-policy rollouts -> one …
- `on_train.py` (experiments/onpol_nuplan/scripts): with the S flags, plus a start from an …
- `long_read.py` (experiments/alpasim_long/scripts): per-scene and per-decision metrics of …
- `closed-loop.md` (research/next-round): 闭环训练：全长、全部公开 scene、用模拟器的后果信号监督（计划稿）

[results/](results/) 14 result files · [figs/](figs/) 2 figures · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 5 entry points
<!-- files:end -->
