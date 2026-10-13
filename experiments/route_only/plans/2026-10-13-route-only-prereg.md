# ROUTE-ONLY pre-registration (2026-10-13)

Question (user, 2026-10-13): "openpilot controls only the longitudinal; turns and lane changes are taken entirely from the route
centreline carried as memory." What does a driver score when the served PATH is the AlpaSim route carried as memory and the SPEED
PROFILE is the model's own plan? Compared against the current candidate RSG-L18-s0 (decision 286). No training.

Scope change before any read (coordinator, 2026-10-13): Tokyo is full; the PAI 281 read is dropped (not run, no code copied to
Tokyo). The line is box-only: nuPlan held 460 / 18 decisions (both arms) + public 1491 / 10 decisions (R-own). PAI is a stated
follow-up (pai_driver.py is not hooked; the switch lives in sh30_driver.py only).

## Arms (same checkpoint RSG-L18-s0, same serving as its stored reads: every serving switch off; only the plan post-processing differs)

Switch `JEV_ROUTE` (experiments/alpasim/lib/route_path.py, default off; hooked into experiments/alpasim/lib/sh30_driver.py after the
JEV_VCONT / JEV_LEAD / BODY1 hooks, before JEV_TRACK). Route path = the ROUTE-MEM memory (experiments/route_mem/lib/rmem.py: every
route message of the episode carried into the current rear-axle frame by the ego poses and merged; the same code d282 proved
train / serve consistent), as the full merged polyline (not the 40-point window).

- **R-own** (`JEV_ROUTE=own`): path from the ego's current pose into the centreline, then along it; the 8 poses at 0.5 .. 4 s placed
  at the cumulative arc lengths of the model's own served plan; yaw = path tangent.
  - Blend (chosen here, before any read): cubic Hermite from (ego position, ego heading) to the route point at arc
    max(s_e, 0) + Lb with the route tangent, tangent lengths = the chord; Lb = clip(2 s x v_ego, 8 m, 30 m), so the join lies at or
    beyond the 2 s pose and the 1-2 s window the official MPC tracks (docs/alpasim.md) sees one smooth curve, no step.
  - Coverage: s_e = the ego's projection on the merged route (negative when the memory starts ahead of the ego: first decisions,
    whose only message starts ~40 m ahead). If the gap g = -s_e > 0.5 m, the uncovered part is the model plan's own path up to
    arc g and the blend starts from that point; poses with arc <= g (and any pose beyond the route's end) keep the model plan. If the
    plan does not reach g, the whole decision is the model plan. Logged per decision in drive.jsonl `route` (cov = poses on the
    route path 0..8, gap, s_e, ego-route distance, route length ahead, Lb, lateral change at 1 / 2 / 4 s) and `route_path`.
- **R-log** (`JEV_ROUTE=log`, **privileged ceiling, reference only**): the same path; arc lengths = the human log's travelled
  distance from t0 to t0 + 0.5 k s (AlpaSim recording ground truth, `send_recording_ground_truth=true`; past the log end at the
  log's last speed). It uses information no submission has; labelled privileged in every table; no verdict line.
- **Reference**: the stored RSG-L18-s0 reads: held `$DATA_DIR/runs/route_mem/read/RSG-L18-s0` (18 decisions), public
  `$DATA_DIR/runs/route_follow/guard/read10/RSG-L18-s0` (10 decisions). No re-run of the reference (budget); the smoke measures
  whether a switch-off re-run reproduces the stored plans.

## Reads

- Held 460 at 18 decisions (onpol_long held_n6 lists), both arms; paired on the 452 scenes with rollouts (39 logs) by
  `ol_confirm.py` (log-clustered bootstrap, B = 10 000): mean score, zeros, collision / offroad / corridor zeros, at-fault zeros,
  GT distance; plus progress (`progress_rel` mean, `dist_traveled_m` / `gt_dist_traveled_m`) from results-summary; turn completion
  of the 24 held turns by `rf_follow.py` (its nuPlan route regeneration exists, so the rf_pai_follow-style measure is available);
  zero taxonomy by `rf_tax.py` where it applies to nuPlan, else by failure reason.
- Public 1491 at 10 decisions (all_n10 lists), R-own only; paired against the stored RSG-L18-s0 guard read by `ol_confirm.py`.
- Coverage: share of decisions with the route path served (cov > 0), cov distribution, by decision index; gap distribution.

## Lines (judged once, on the first complete read)

Primary, held 452 paired R-own - RSG-L18-s0 (stored s0: 0.8809 / 26 zeros / 4-17-6 / at-fault 21 / GT 1.135):
- **H1 corridor**: R-own corridor zeros <= 2 (reference 6). Expected to pass by construction (the route is the log's lane).
- **H2 verdict**: *better* = mean-score CI lower > 0 and zeros CI upper < 0; *not worse* = mean-score CI lower >= -0.02 and
  at-fault zeros CI upper <= +5 (and not better); *worse* = otherwise.
- **H3 GT distance**: R-own - reference CI upper < 0.
- **H4 turn completion** (24 held turns, rf_follow): R-own >= 80 % (reference 66.7 %); reported with CI, not a gate.
- Progress: reported (expected |delta progress_rel| < 0.02, the speed profile is the model's own).

Guard, public 1491 / 10 decisions paired R-own - RSG-L18-s0 (stored s0: 0.9473 / 46 zeros / 8-31-8 / at-fault 39 / GT 0.792):
- **G1** mean-score difference >= -0.01 (point) and at-fault zeros difference CI lower <= 0 (no significant increase).

The user's hypothesis counts as supported if H2 is *better* or *not worse* with H1 and H3 passing, and G1 passes. If H2 is
*worse*, the zero taxonomy says which failure mode below carries it.

## Expected failure modes (written before any read)

1. Early decisions are not covered (memory starts ~40 m ahead): the first metres stay model-driven; zeros there remain.
2. The route is the log projected onto the lane centreline: a human swerve around a parked car or a lane offset disappears, so the
   route path can run into static obstacles the model would avoid (collisions); the route can snap to the neighbouring lane.
3. The blend cuts corners when Lb spans a junction entry (inner-kerb offroad, right turns most).
4. The model's speed profile was made for its own (straighter, slower-turning) path: on a sharp route turn there is no lateral
   speed cap, so the MPC may lag (wide exits); and a turn across oncoming traffic is taken at the model's speed with the model's
   yielding tied to its own path (collisions at fault).
5. Past the log end the route extends straight (driving past where the human stopped: corridor / progress effects).
6. 2 Hz re-blending from an off-centre pose can oscillate around the centreline (lateral jerk, corridor).

## Smoke (before the reads)

2 held turn scenes (e14711ef: left turn, RSG zero; 8c9e3fd9: right turn, completed), arms off / own / log at 18 decisions. Gates:
S1 switch off = byte-identical driver code path; plans compared with the stored RSG-L18-s0 drive.jsonl (reported: bitwise or the
re-run spread); S2 own / log run without driver errors, `route` logged on every decision, served arcs = plan arcs (own) / log arcs
(log) within 0.1 m on covered poses; S3 one BEV of a junction decision (model plan vs route path vs served poses) looked at.

Budget: box ~3 card-h (held 2 x ~0.4, public ~1.5, smoke 0.1). Tokyo: none.

## Smoke result and Amendment 1 (2026-10-13, before any held or public read)

Smoke (`$DATA_DIR/runs/route_only/smoke/{off,own,log}`, 2 held turn scenes x 18 decisions):
- S1 passes: switch off reproduces the stored RSG-L18-s0 served poses **bitwise on 36 / 36 decisions** (max |diff| 0.0), same
  scores and metrics. The stored reference read is therefore a valid paired reference (no re-run noise at the plan level).
- S2 passes: no driver errors; `route` logged on every decision; served arcs = plan arcs (own) within 0.004 m, = log arcs (log)
  within 0.022 m on covered poses.
- Coverage finding: in both scenes the memory starts 32-34 m ahead of the ego for the first 12-13 decisions (the ego is slow and
  AlpaSim never sends the near 40 m), and the junction lies inside that gap: 8c9e3fd9 has poses on the route only from decision 12
  on (cov 1-5), e14711ef has **no** covered pose in 18 decisions. Under the registered fallback R-own serves the model plan through
  these turns, so R-own reads "route as path where the memory already covers it", not "turns from the route".

Amendment 1 (added arm, registered arms and lines unchanged): **R-bridge** (`JEV_ROUTE=bridge`): as R-own, but when the memory
starts ahead of the ego (gap g > 0.5 m) the cubic Hermite blend runs from the ego pose / heading straight into the route at route
arc max(0, Lb - g) from its start (the route start whenever g >= Lb), tangent lengths = the chord; every pose is then on the
path (arc lengths still the plan's own). This is the most direct reading of "turns from the route" that the 40-80 m messages
allow: the junction inside the gap is crossed by a smooth curve from the ego to where the route resumes, not by the model.
Read: held 460 / 18 decisions only (public guard only if R-bridge passes H2 as better or not worse; budget). Lines: the same
H1-H4 as R-own, judged the same way; R-bridge vs R-own also reported paired. Expected failure modes in addition: the bridge
cuts across the junction (an inner-kerb or median offroad when the route resumes on the exit road ~30 m away at a right angle;
the Hermite curve does not know the lane geometry), and on straight roads with a lateral route offset the bridge is a slow
lane change.
