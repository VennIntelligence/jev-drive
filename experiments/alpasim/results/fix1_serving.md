# FIX1: serving switches for the AlpaSim drivers (speed-continuous serving, openpilot lead limit, base-model speed profile)

Written 2026-10-10. Lane FIX1, after lane COL1's collision study and with lane DIAG1's audit (decision 218) for switch (c).
Pre-registration with three amendments: [../plans/2026-10-09-fix1-prereg.md](../plans/2026-10-09-fix1-prereg.md) (each pushed before the
scores it covers). Code: `lib/serve_fix.py` (specification in its docstring), hooks in `lib/{sh30,ap2,pai}_driver.py`, chains
`scripts/fix1_chain.sh` (nuPlan, box), `scripts/fix1_pai_chain.sh` (Tokyo), `scripts/fix1_pai_box.sh` (PAI, box), reports
`scripts/fix1_report.py`, `fix1_pai_report.py`, `fix1_pai_box_report.py`, open-loop check `scripts/fix1_replay.py`. Tables: `fix1/`.
Serving-side only: no training, no simulator ground truth in the driver, nothing submitted.

## Conclusion

- **nuPlan track: every switch loses; the driver stays as it is.** On the 700 public scenes (P2H10-F-s0, paired with a base re-run at the
  same checkout) none of nine arms passes the registered lines. (a) is the only one near zero: -0.0066 [-0.027, +0.011].
- **PAI track: (a) + (b) is the best configuration measured**, and the gain is large against the baseline. BOX_SUMMARY
- The two tracks need different serving. One image with one default cannot serve both; the flags are per track.

## The switches

| Switch | What is served | Where the constants come from |
|---|---|---|
| (a) `JEV_VCONT=1.0` | the plan's path; the speed profile starts at the ego's speed and blends linearly into the plan's segment speeds over 1.0 s | AlpaSim's nonlinear MPC costs the reference only from horizon index 10 of 20 (0.1 s steps): positions 1.0-2.0 s ahead. A linear blend over that second is what a constant acceleration meets exactly at the first costed index |
| (b) `JEV_LEAD=1` | the profile above, limited pointwise by the speed solution of openpilot's lead MPC on the checkpoint's own `lead` / `lead_prob` outputs | openpilot ec95db3f: radard's probability filter and vision lead, `long_mpc.py`'s cost / horizon / limits, the planner's filter and min of candidates; solved by scipy SLSQP, 20 Hz ticks. Ours: the mapping to a position reference |
| (c) `JEV_BASE=1` / `2` | the adapter's path sampled at the arc lengths of the plan without the adapter bias (the shipped model's plan, decision 218). 1 = always; 2 = while moving, at standstill the adapter's launch unless a lead is inside openpilot's following distance | decision 218's split at 0.5 m/s; openpilot's d(v) |

How (b) differs from the settled mechanisms (decisions 140, 157, 167, 179, 203, `op_control_stack_long`, COL1's guard ceiling) is in
the pre-registration. All switches are off by default; with them off the drivers return the same plans as before (2 320 replayed
decisions, difference 0.0 m).

## nuPlan: the 700 public scenes (GPU box, native stack)

P2H10-F-s0, OT3's three chunk lists, 27 logs. Driver code 0763ea89 for base / a / b / ab and ee5bf897 for the (c) arms (exported trees
verified file by file against the commits; the box could not fetch from GitHub that night), box defaults (compiled model passes).
Full tables: [fix1/fix1_700_s1c.md](fix1/fix1_700_s1c.md), per scene `fix1/fix1_700_s1c_per_scene.csv`; c+a arms: [fix1/fix1_700_s1ca.md](fix1/fix1_700_s1ca.md).

| Arm | Mean | Difference to base [95% CI, logs] | Zeros | At-fault collision / offroad / corridor | Slow | zero -> non-zero / non-zero -> zero | Lines |
|:--|--:|:--|--:|:--|--:|:--|:--|
| base (this checkout) | 0.9468 | | 26 | 8 / 10 / 8 | 112 | | |
| OT3's run of the base | 0.9483 | +0.0016 | 25 | 7 / 10 / 8 | 110 | 1 / 0 | (run-to-run: 1 scene) |
| +a | 0.9402 | -0.0066 [-0.0272, +0.0112] | 32 | 8 / 14 / 10 | 68 | 5 / 11 | not kept |
| +b | 0.8609 | -0.0858 [-0.1130, -0.0568] | 24 | 7 / 10 / 7 | 271 | 4 / 2 | not kept |
| +a+b | 0.8756 | -0.0712 [-0.1044, -0.0376] | 28 | 6 / 13 / 9 | 207 | 8 / 10 | not kept |
| +c1 | 0.8146 | -0.1321 [-0.1665, -0.0997] | 34 | 9 / 14 / 11 | 245 | 11 / 19 | not kept |
| +c2 | 0.8495 | -0.0973 [-0.1252, -0.0692] | 34 | 9 / 14 / 11 | 233 | 11 / 19 | not kept |
| +c1+b | 0.7946 | -0.1521 [-0.1895, -0.1165] | 38 | 13 / 14 / 11 | 299 | 10 / 22 | not kept |
| +c2+b | 0.8268 | -0.1199 [-0.1521, -0.0863] | 37 | 13 / 14 / 10 | 288 | 10 / 21 | not kept |
| +c1+a (post hoc) | 0.8403 | -0.1064 [-0.1429, -0.0712] | 32 | 10 / 9 / 13 | 185 | 9 / 15 | not kept |
| +c2+a (post hoc) | 0.8703 | -0.0764 [-0.1086, -0.0439] | 32 | 10 / 9 / 13 | 174 | 9 / 15 | not kept |

Lines (kept = at-fault-collision zeros drop, difference >= 0, log-clustered lower bound > -0.005, slow <= 1.1 x base): no arm kept, so by
the registered stop rule the other 791 scenes and APY10m10-AB-s0 were not run.

What the arms did:

1. **(a) trades slow scenes for zeros.** Slow scenes fall from 112 to 68, but 11 scenes turn to zero (4 offroad, 4 corridor, 3 collisions)
   against 5 repaired; on the 663 scenes non-zero in both the means differ by +0.0007. On nuPlan the plan's first segment is only 1-2 %
   below the ego's speed above 2 m/s (open-loop replay, `fix1/replay.md`), so the base driver brakes a little at every decision; without
   that the car is faster and fails laterally more often.
2. **(b) does what it was built for and costs progress everywhere else.** Both stopped-lead collisions of the base (39cbed62, b5b8d206)
   become score 1.0. The limit removes >= 0.5 m of the first 2 s in 304 of 700 scenes; those lose 0.199 on average, the other 396 change by
   +0.001. openpilot keeps 6 m + 1.45 s behind a lead; the logged nuPlan drivers follow closer and the score is progress against the log.
   The open-loop replay had predicted it (the limit acted in 98 of 200 control scenes).
3. **(c) loses more than (b).** The base model's speed profile does not start at the ego's speed here: its first segment is 0.84-0.96 x
   the ego's between 0.5 and 10 m/s (closed-loop records; 2 Hz decisions with six synthesised slots), and it holds at standstill where the
   log launches (c2 served the adapter's launch at 49 % of standstill decisions). With (a) on top the loss shrinks by about 0.02 and stays
   large. Decision 218's open-loop cost on navtest (-3.3 / -2.0 EPDMS) shows up as -0.13 / -0.10 scene score in the closed loop.

Latency on the box (8 concurrent rollouts, `drive` median / p99): base 17.8 / 47 ms; (b) arms 33-35 / 78-94 ms (the lead MPC: 16 ms
median for 10 ticks); (c) arms 32.6 / 73 ms (one more interpreted policy pass); c2+b 49 / 123 ms.

## PAI: 40 scenes on the Tokyo box (P2H10-F-s0, one rollout per scene; counts only)

Lists and chunking of COL1's baseline runs; +a is COL1's arm `v1` (the same function). Table with per-scene rows and the organisers'
reference subjects: [fix1/pai_tokyo40.md](fix1/pai_tokyo40.md). "Untouched" = the 20 scenes of `b2a` + `b2b`, whose results nobody had
read when (a), (b) and (c) were fixed.

| Arm | All 40: mean | zeros | at-fault collision | zero -> non-zero / non-zero -> zero | Untouched 20: mean | zeros | Seen 20: mean |
|:--|--:|--:|--:|:--|--:|--:|--:|
| base | 0.169 | 30 | 12 | | 0.156 | 15 | 0.183 |
| +a | 0.292 | 25 | 8 | 7 / 2 | 0.366 | 11 | 0.219 |
| +b | 0.383 | 20 | 2 | 11 / 1 | 0.402 | 9 | 0.365 |
| **+a+b** | **0.462** | 17 | 2 | 14 / 1 | **0.529** | 7 | 0.395 |
| +c1 | 0.323 | 14 | 2 | 19 / 3 | 0.365 | 6 | 0.281 |
| +c2 | 0.382 | 16 | 3 | 17 / 3 | 0.423 | 8 | 0.341 |
| +c1+b | 0.366 | 12 | 1 | 20 / 2 | 0.408 | 5 | 0.323 |
| +c2+b | 0.387 | 16 | 2 | 17 / 3 | 0.461 | 7 | 0.312 |
| alpamayo1 (reference, mean of 3) | 0.506 | | | | 0.702 | | |
| vavam-linear (reference) | 0.344 | | | | 0.408 | | |

- (c) removes the most zeros (30 -> 12-16) but leaves 16-20 slow scenes (base 6): with c1 the car stands at 29 % of the decisions
  (2 317 of 7 947; +a+b 8 %). (a)+(b) has more zeros and more scenes at score 1.
- No driver error in any arm. The serving step costs 2.0 ms median with (b) (2 ticks per decision), 0.04 ms with (c) plus its second
  policy pass (`drive` median 57 ms against 50 ms on the healthy card).
- Tokyo's card 0 was power-throttled that night (225 MHz, runs 4x slower); the queue put all but five runs on card 1. Timing columns of
  runs on card 0 are not usable; scores do not depend on the card.

BOX_SECTION

## Guardrails

- **Image.** An image built from main failed `docker/test.sh` before any switch: the drivers' default `SH30_COMPILE=1` cannot run in
  the image (triton's cache under the read-only `$HOME`, then no C compiler), the driver fell back with a logged traceback. The
  Dockerfile now sets `SH30_COMPILE=0` (the interpreted passes every image so far has served) and points the caches at `/tmp`. With that,
  on Tokyo card 1: `jev-alpasim:p2h10-f-s0-9c592640` (switches off) passes, `drive` p50 / p99 50.6 / 71.9 ms;
  `...-9c592640-vc1.0-lead` (built with `VCONT=1.0 LEAD=1`) passes, 62.2 / 80.8 ms, 1.6 GiB. Consequence: the box runs compiled passes,
  the image interpreted ones (fp16-level differences, lat1_frame_synthesis.md); the pre-registration's remark that compiled passes are
  the image's path was wrong.
- **Latency.** Above; (b) on the nuPlan cadence adds 12-16 ms per call, on PAI 2 ms. (c) adds one policy pass.
- **HUGSIM / jevdrive.bench.** Not applicable: the switches act in the AlpaSim drivers' export step only.

## Limits

One checkpoint (two seeds on the box PAI read), one rollout per scene and arm; local rendering; PAI numbers come from 40-60 public scenes
with a large seed spread and say nothing certain about the private suite. (b) uses the served checkpoint's lead outputs, not the shipped
weights'. The mapping of openpilot's acceleration limit to a position reference (pointwise minimum of speed profiles) is ours. (c)+(a) was
not run on PAI. The nuPlan arms ran with compiled passes, the image serves interpreted ones. No APY10m10-AB run: on nuPlan by the stop
rule, on PAI because `pai_driver.py` builds NAVSIM-standard ego features and a plain P2 model (`sh30_core.Core`), not the
AlpaSim-input-standard path of `ap2_core.py`.
