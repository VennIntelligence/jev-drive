# onpol_long: closed-loop training at 18 decisions, round 1

status: live
decisions: 264 (inputs 259, 262)
index: S-only rows at 18 decisions on YR10m10-F: 0.838 vs base 0.785, zeros 44 vs 74; S + corridor not adopted

**Question.** At 18 decisions (token + 8 s) in local AlpaSim nuPlan, do on-policy rows of the base's own rollouts on the 1031 train scenes,
carrying only consequence hinges on the model's own plan (S: swept footprint against vehicles, road-and-lane raster; K: 1.5 m dead band to
the logged path; C: lane-centre offset if it passes its label gate), give fewer zeros than the same continued training without rows
(ON-NONE) on the 460 held scenes, with at-fault zeros and GT distance not worse?

**Conclusion.** On the YR10m10-F-s0 base, on-policy rows at 18 decisions carrying only the S hinges (`YS-L18`, Amendment 3) are
adopted: on 452 held scenes 0.838 against the base's 0.785 (+0.053 [+0.024, +0.082]), zeros 44 against 74, at-fault zeros 23.5 against 43,
GT distance 1.36 against 1.50 m, both seeds alike (decision 264). Adding the corridor hinge K (the registered round-1 arm `SKC`) is not
adopted on either base (P2H10S-F strongly negative: offroad zeros 79 against 24; YR10m10-F fails the at-fault gate). On P2H10S-F the
S-only arm is flat against the base (-0.0075). The same continued training without rows loses to the base on both (-0.092 Y, -0.028 P).

A second on-policy round (rows from `YS-L18`'s own rollouts, from `YS-L18-s1`) does not pass the registered round gate: it is no better
than its own no-row control and slightly worse than round 1 (zeros 50.5 against 43), so the loop stops at round 1 and "more rounds help"
is recorded as negative for this setting.

**Next.** PAI transfer read on the Tokyo box (Amendments 1, 4, 5, running).

**Read more.** [plans/2026-10-12-ol1-prereg.md](plans/2026-10-12-ol1-prereg.md); the plan: [research/next-round/closed-loop.md](../../research/next-round/closed-loop.md);
the row and trainer code it reuses unedited: [../onpol_nuplan/README.md](../onpol_nuplan/README.md); the 18-decision driver path:
[../alpasim_long/README.md](../alpasim_long/README.md).

## Round 2 (Amendment 5; rows from `YS-L18` s0 + s1 rollouts, from `YS-L18-s1`, full schedule)

[results/read/report_YS2_prev.md](results/read/report_YS2_prev.md) (against round 1, `YS-L18-s1`), [report_YS2_base.md](results/read/report_YS2_base.md)
(against the base). Pool `on1_L18YS`: 34 788 rows from 2042 rollouts, gates pass ([results/gates/gates_L18YS.md](results/gates/gates_L18YS.md)).

| | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |
|:--|--:|--:|:--|--:|--:|
| `YS2-L18` (S rows of round-1 rollouts) s0 / s1 | 0.833 (0.838 / 0.827) | 50.5 | 7.5 / 23.0 / 22.0 | 30.5 | 1.396 |
| `YS2NONE-L18` (same continuation, no rows) | 0.824 | 51.5 | 6.0 / 23.5 / 25.5 | 29.5 | 1.370 |
| round 1 `YS-L18-s1` (the start) | 0.841 | 43 | 9 / 15 / 21 | 24 | 1.346 |
| base `YR10m10-F-s0` | 0.785 | 74 | 10 / 33 / 35 | 43 | 1.495 |

- Round gate (closed-loop.md section 6, line 3; against `YS-L18-s1`): zeros 50.5 against 43, at-fault zeros 30.5 against 24, GT 1.396
  against 1.346 m: **not passed**. By the registered rule the loop stops at round 1, "more rounds help" is negative here, round 3 is not run.
- Against its own control: +0.009 [-0.016, +0.033], zeros -1.0 [-13.0, +11.1], at-fault +1.0: the second round's rows add nothing measurable;
  the continuation itself costs round 1 a little (`YS2NONE` - `YS-L18-s1` -0.017 [-0.043, +0.007]). Against the base `YS2` is still +0.048.
- `YS2-L18-s0`'s read ran on card 0 (Amendment 7).

## Round 1b: gentle continuation (Amendment 6; Y family, same rows, lr x 0.1, no warm-up restart, 2 500 steps)

[results/read/report_YSG_base.md](results/read/report_YSG_base.md) (against the base), [report_YSG_ys.md](results/read/report_YSG_ys.md) (against `YS-L18`, seed mean).

| | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |
|:--|--:|--:|:--|--:|--:|
| `YSG-L18` (S rows, gentle) s0 / s1 | 0.829 (0.829 / 0.828) | 51.5 | 8.5 / 22.0 / 23.5 | 30.5 | 1.300 |
| `YNONEG-L18` (no rows, gentle) | 0.774 | 78.0 | 9.0 / 37.0 / 35.5 | 46.0 | 1.454 |
| `YS-L18` (round 1, full schedule) | 0.838 | 44.0 | 9.0 / 14.5 / 22.0 | 23.5 | 1.357 |
| base `YR10m10-F-s0` | 0.785 | 74 | 10 / 33 / 35 | 43 | 1.495 |

- Control line (registered): `YNONEG` - base -0.0106 [-0.0266, +0.0053], zeros +4.0 [-3.8, +12.1]: the CI includes 0, **holds**. The gentle schedule
  stops the continuation from hurting the base (the full schedule cost -0.092). Dev ADE stays at the base's (0.539 -> 0.537-0.539).
- `YSG` against `YS-L18` (registered: preferred for round 3 only if not worse on any of four): mean score -0.009 [-0.030, +0.011], zeros +7.5
  [-2.4, +17.4], at-fault zeros +7.0 [-2.1, +15.8], GT distance -0.057 [-0.132, +0.024]: worse on three of four, **not preferred**; round 3
  keeps `YS-L18`'s schedule. Against the base `YSG` is still +0.044 [+0.019, +0.069], zeros -22.5, and against its own control +0.054.
- Reading: with the gentle schedule the rows' effect is about 80 % of `YS-L18`'s gain over the base (+0.044 against +0.053) with the
  continuation itself near zero, so most of `YS-L18`'s gain is the rows, not a lucky continuation; the full schedule's loss on ON-NONE
  (-0.092) did not carry into `YS-L18`.

## S-only arms (Amendment 3; same pools, same schedule, no K)

[results/read/report_YS.md](results/read/report_YS.md), [report_PS.md](results/read/report_PS.md).

| family | | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |
|:--|:--|--:|--:|:--|--:|--:|
| Y | `YS-L18` s0 / s1 | 0.838 (0.835 / 0.841) | 44.0 (45 / 43) | 9.0 / 14.5 / 22.0 | 23.5 | 1.357 |
| Y | ON-NONE `YNONE-L18` | 0.693 | 120.0 | 23.5 / 56.0 / 51.5 | 79.5 | 1.746 |
| Y | base `YR10m10-F-s0` | 0.785 | 74 | 10 / 33 / 35 | 43 | 1.495 |
| P | `PS-L18` s0 / s1 | 0.791 (0.793 / 0.790) | 66.0 | 15.0 / 21.5 / 31.0 | 36.0 | 1.430 |
| P | ON-NONE `PNONE-L18` | 0.771 | 81.0 | 11.5 / 24.0 / 47.0 | 35.5 | 1.674 |
| P | base `P2H10S-F-s0` | 0.799 | 68 | 8 / 24 / 39 | 32 | 1.588 |

| a - b (log-clustered 95 % CI) | mean score | zeros | at-fault zeros | GT distance (m) | fixed / broken (null max) |
|:--|:--|:--|:--|:--|:--|
| `YS-L18` - ON-NONE | +0.145 [+0.119, +0.173] | -76.0 [-89.9, -63.5] | -56.0 [-69.8, -43.1] | -0.389 [-0.509, -0.272] | 52 / 5 (8) |
| `YS-L18` - base | +0.053 [+0.024, +0.082] | -30.0 [-43.5, -16.7] | -19.5 [-28.8, -9.3] | -0.138 [-0.266, -0.014] | 38 / 10 |
| `PS-L18` - ON-NONE | +0.021 [-0.010, +0.053] | -15.0 [-30.2, -0.9] | +0.5 [-11.5, +10.9] | -0.245 [-0.370, -0.111] | 28 / 19 (1) |
| `PS-L18` - base | -0.008 [-0.044, +0.025] | -2.0 [-17.6, +14.5] | +4.0 [-4.8, +12.3] | -0.158 [-0.274, -0.020] | 25 / 25 |

Confirmation reads (Amendment 5; [results/read/confirm.md](results/read/confirm.md)), against `YR10m10-F-s0`:

| scene set | `YS-L18-s0` - base | `YS-L18-s1` - base | zeros base / s0 / s1 | at-fault zeros base / s0 / s1 |
|:--|:--|:--|:--|:--|
| all 1491 public at 10 decisions (1479 with a rollout; base = stored 2026-10-09 rollouts) | +0.0125 [+0.0033, +0.0226] | +0.0131 [+0.0036, +0.0227] | 68 / 47 / 48 | 50 / 30 / 36 |
| train 1031 at 18 decisions (the scenes that supplied the rows: guardrail only) | +0.0591 [+0.0381, +0.0819] | +0.0495 [+0.0298, +0.0714] | 168 / 99 / 114 | 102 / 58 / 63 |

The public-mean guardrail (not more than 0.02 below the base on all 1491 at 10 decisions) holds for both seeds: 0.945 against 0.932,
GT distance -0.04 m.

Registered line: `YS-L18` adopted (and CL-1 as written met); `PS-L18` not (at-fault zeros 36 against 35.5). So on these rows K is what
broke the round-1 arms: S alone on the same rows is the best checkpoint of the night on the YR base and neutral on the P2H10S base.

Why ON-NONE loses to the base (descriptive): the continuation repeats the base's whole schedule from its end point (fresh AdamW, lr 3e-5
backbone / 3e-4 adapter, 300 warm-up steps back to full rate, cosine over 10 000 steps): a second full training on the same data, not a
fine-tune. It lowers open-loop dev ADE (0.539 -> 0.524 Y, 0.545 -> 0.529 P) and loses closed loop at 18 decisions. In round 1 (decision
259, 10 decisions) the same continuation on P2H10S matched its control (-0.004); at 18 decisions P's loss sits in decisions 11-17 (first
at-fault step at k <= 10: ON-NONE 15 / 14 against the base 14 / 16) while Y's already shows early (35 / 35 against 21 / 16). What I would
change (not launched): continue at a tenth of the rate (3e-6 / 3e-5), no warm-up restart, 2 000-3 000 steps, so that the on-policy rows
move the plan without re-fitting the imitation data; read ON-NONE first to show it stays at the base.

## Round 1 as registered (S + K): held 460 at 18 decisions (2026-10-12 morning; 452 scenes after 8 route-fold scenes, two training seeds from the s0 base)

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

## Which card each simulator job ran on (Amendment 7)

Card 0 has ECC on and renders 1-2 grey levels differently from cards 1-5 (pipeline lane, `results/pipeline_profile.md`). Before Amendment 7
the pool placed jobs freely. Held reads on card 0: the base `YR10m10-F-s0` at 18 decisions (alpasim_long `al-h18`), `YS-L18-s0`,
`YNONE-L18-s0`, `YNONE-L18-s1`, `PSKC-L18-s1`; every other held read of this topic ran on cards 1-5 (`YS-L18-s1` card 1, `YSG-L18-s0 / -s1`
cards 2 / 3, `YNONEG` 5 / 2, `PS-L18` 2 / 3, `PNONE` 5 / 4, `YSKC` 1 / 3, `PSKC-L18-s0` 2). Rollout chunks on card 0: `YR10m10` c0,
`P2H10S` c4, `YS-L18-s0` c2, `YS-L18-s1` c4 / c5. From round 3 on, rollouts and reads use cards 1-5 only (`OL_SIM_GPUS`).

## Chain

```bash
bash experiments/onpol_long/scripts/ol_chain.sh submit-roll <base tag> <after ids>        # 6 rollout jobs (cl fanout), 18 decisions, frame dump
bash experiments/onpol_long/scripts/ol_chain.sh submit-rest <base tag> Y|P <rollout ids> <after ids for ON-NONE>
#   geo (on_rows.py geo + ol_labels.py build) -> tok (on_rows.py tok + ol_labels.py gate) -> strip (drop dump frames) ;
#   tok -> train <F>SKC-L18-s{0,1} -> read1 ; train <F>NONE-L18-s{0,1} -> read1 ; all reads -> report (ol_report.py)
```

Faster rounds (same programs, same per-scene results; [results/pipeline_profile.md](results/pipeline_profile.md)): `ol_fast.sh submit-round <init>
<arm>` queues one Y-recipe round with rollouts in 12 jobs and each held read in 6 (two simulator stacks per card), the K / C label build beside
the tokens instead of before them, and the arm's path one priority step above the controls: critical path ~45-50 min against 95-110 for
`ol_chain.sh` (estimate until the first ol_fast round is timed). `OL_DRY=1` prints the submissions.

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
