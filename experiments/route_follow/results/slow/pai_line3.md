# SLOW line 3: PAI 281 reads of RSGS1 / RSGS2 / RSGS3 (seed 0) vs the RSG-L18-s0 gate reread

Prereg: `plans/2026-10-15-slow-s1s2-prereg.md` line 3 (Amendment 1 order S1, S2, S3; Sall stopped at its pilot gate, Amendment 3). Serving = the reference
`v2pai/RSG-L18-s0-gate` exactly (decision 309: gate serve, gains 20, `JEV_VCONT=1.0 JEV_LEAD=1`, CHUNK 36, the reference's chunk lists, CONC 4), paired per scene
over all 281. Numbers from `rf_londiag.py pai` (raw output and json per arm in `pai_line3/`). The reference's own row (`RSG-L18-s0-gate` against itself: all
differences 0) is the new reference: slow-scene speed ratio 0.587, handoff 0.867, collision zeros 12, all zeros 71, rear-ended 108, `cor_route_end` 18, mean 0.5860
(the old `r281/rsg` read, serve path 0, gave 0.589 / 0.867 / 9 / 70 / 111 / 17).

| line (prereg) | ref | S1 RSGS1 | S2 RSGS2 | S3 RSGS3 |
|:--|--:|:--|:--|:--|
| slow-scene speed ratio median, 73 `low_slow_free` (>= 0.80) | 0.587 | 0.618 (+0.032) **fail** | 0.645 (+0.059; 72 of 73 scenes) **fail** | 0.649 (+0.063) **fail** |
| handoff k 0-2 v_p / v_f, v_log >= 16 m/s (>= 0.95) | 0.867 | 0.868 (+0.001) **fail** | 0.923 (+0.056) **fail** | 0.916 (+0.049) **fail** |
| collision zeros (<= ref + 2) | 12 | 11 (-1) pass | 10 (-2) pass | 10 (-2) pass |
| all zeros (<= ref + 3) | 71 | 72 (+1) pass | 77 (+6) **fail** | 77 (+6) **fail** |
| exam 72 mean-score diff, CI low (> -0.02) | | +0.0048 [-0.0499, +0.0579] **fail** | -0.0071 [-0.0610, +0.0433] **fail** | -0.0153 [-0.0767, +0.0448] **fail** |
| rear-ended (<= 0.7 x ref = 75.6) | 108 | 115 (+7) **fail** | 98 (-10) **fail** | 101 (-7) **fail** |
| `cor_route_end` (<= ref + 2) | 18 | 18 (0) pass | 19 (+1) pass | 19 (+1) pass |
| all-281 mean score | 0.5860 | 0.5718 (-0.0142) | 0.5987 (+0.0128) | 0.5925 (+0.0065) |
| **LINE 3** | | **FAIL** | **FAIL** | **FAIL** |

Reading (facts only; the decision is the main session's):
- No arm reaches the speed-ratio or handoff line. The PAITV data arms move the handoff bin (S2 0.923, S3 0.916, S1 0.868 = ref), the empty-road history drop (S1) does not;
  the data effect alone (S3) is almost the whole S2 effect, S2 - S3 = +0.007 on handoff and -0.004 on the slow ratio. The prereg's "only S2 passes the handoff line"
  did not happen (neither does), so "handoff-short = data gap" is partly supported (+0.05) but not sufficient (0.95 not reached).
- Speed ratio improves by 0.03-0.06 for all three, still far from 0.80.
- S2 and S3 add 6 zeros over the reference (77 vs 71, line +3), with fewer collision zeros (-2) and fewer rear-endings (-7 to -10); the exam-72 CI is wide (about +-0.06) in every arm.
- Together with lines 1-2 (`lines12.md`: S1, S2 fail both, S3 fails line 1) no arm is adopted by prereg line 4.

Run notes: RSGS1 ran 00:07-00:47 JST on cards 0 1 (8062 s). RSGS2 collided with the spin_v9 jlon read on card 1 (chunks c001-c007 failed twice within seconds,
OOM / port clash, logs under `v2pai/RSGS2-L18-s0-gate/failed/`); it was resumed in the same run dir (card 0 from 01:09 JST, card 1 joined at 02:44 when jlon finished),
so all 8 chunks are complete reads of the same checkpoint and the 281 scenes are paired as above. RSGS3 ran 04:21-06:37 JST on both cards (8196 s).
Line 3 extracted with the Tokyo queue `/data/runs/route_follow/slow/queue_night.sh` (rf_tax extract + decision slim + `rf_londiag.py pai` in a numpy/pandas venv,
`/data/venv_line3`, because the system python3 has no pandas).
