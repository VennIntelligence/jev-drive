# RED line (d): closed-loop PAI, the 24 test281 red-light scenes (2026-10-13, Tokyo card 0)

RRED-L18-s0 read with the serving of RSG-L18-s0's stored PAI 281 read (gains `rel_front_steering_angle_weight=20`, `JEV_VCONT=1.0 JEV_LEAD=1`,
code 19137cec, `/data/runs/route_night/r281/red_rred24`, one scene rerun in `red_rred24_retry` after a renderer fault, see below). Classified
with `rf_tax.py report` (ref = RSG's read, wait lines `pai_wl.pkl`); per-scene rows in `pai_red24_rred.csv` / `pai_red24_rsg.csv`.

| | mean | zeros | "drove past the human's stop" zeros (score 0 + `ran_stop_line`) | scenes with any stop line run |
|:--|--:|--:|--:|--:|
| RRED-L18-s0 | 0.470 | 10 | **6** | 12 |
| RSG-L18-s0 (reference) | 0.368 | 13 | **7** | 12 |

Paired difference RRED - RSG +0.1025, bootstrap 95 % CI [-0.058, +0.279]; better on 4 scenes, worse on 4, within 0.01 on 16.

**Reading.** The line "fewer than 7" is met by one scene, and that scene is not a red-light fix: the six ran-stop zeros of RRED are the same
six scenes as RSG's (127c4f85, 229791a5, 22fa5c92, 559d4ad6, 68e53160, a1b166cc); the seventh (cbd87234) is a `cor_route_end` zero of RSG
that RRED drives to 1.0 while still running the same stop line (`stopline_ran` 2 in both). Scenes with any stop line run are 12 of 24 for
both models. The score gain is corridor behaviour (RSG's two `cor_drift` and one `cor_route_end` zeros become 1.0), offset by one extra
`coll_veh_side` and a `low_stopped_lead` scene that RRED stalls in (ca7accbc, 0.008 against 0.774). On n = 24 this is descriptive: the
B hinge alone does not change what the policy does at the human's red-light stops in closed loop.

**Infra.** Scene 229791a5 failed in the first pass inside the NRE renderer (`ego_hood.overlay_on_image`: "Expected all tensors to be on the
same device, cuda:0 and cpu"), the first such failure in this lane's reads (rsg, rsgpai, rbsg2, rsgboth turn26: 0 failed rollouts). Rerun
alone on card 0 it scored 0 by at-fault collision, the same primary as RSG on that scene; the merged read uses the rerun.
