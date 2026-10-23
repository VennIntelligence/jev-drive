#!/usr/bin/env python3
"""ROUTE-MEM trainer (plans/2026-10-12-route-mem-prereg.md): onpol_nuplan's on_train.py `train` (its setup() imported unedited, its training
loop repeated here) with the route-memory branch of lib/rmem.py on the adapter. Changes against on_train.py cmd_train, nothing else:

  branch    after setup(): the P2 adapter becomes rmem.RouteParityAdapter (every --init tensor kept; the branch's output layer starts at 0, so
            step 0 is the init checkpoint: logged dev ADE at step 0 must equal the init's), the fed row = ego (20) ++ route features (200) from
            $DATA_DIR/runs/route_mem/feat/<data dir>.npz (scripts/rm_build.py: nav / yr / pool; rows without a rebuilt route get an empty memory)
  lr        the branch is a third parameter group at --rm-lr (default 3e-4 = the adapter's original lr_new, NOT scaled by --lr-scale) with its
            own linear warm-up (--rm-warmup 100) under the same cosine; the net and the old adapter keep on_train's groups and schedule
  drop      the route features are zeroed on a share --rm-drop (0.25, pp_train MEM_DROP) of the non-anchor rows of every batch, own rng stream
            [seed, 0, 89] (every row stream of on_train unchanged): "no memory" stays in distribution, which the reliance read needs
  after     ckpt-final.pt (model.state(): the branch is in `parity`) + route_mem.json (sidecar: sh30_core serves the branch iff it exists);
            the reliance read on the dev rows (memory on / masked, per-row ADE; junction rows = the route enters a junction within 40 m,
            straight rows = no junction within 60 m and < 5 deg route turn over 40 m) -> rely.json in the run dir and runs/route_mem/rely/<tag>.json

Route-night additions (research/next-round/route-night.md 4 C):
  --init    a checkpoint WITH the branch (sidecar route_mem.json beside ckpt-final.pt, e.g. RB-F-s0): on_train.setup runs without --init (its
            load_init demands the plain P2 key set), then the branch is attached and the whole state loaded (cmd_rely's --rm-from path); the
            dev ADE at step 0 is logged against the init sidecar's recorded final dev ADE (`dev_ade`). An init without sidecar: as before
            (on_train's load_init, then the branch at zero). No --init: on_train's fresh path (shipped Cinque weights + a fresh P2 adapter,
            joint_train's start), then the branch; dev metrics at step 0 are logged in every case.
  defaults  on_train's schedule (10 000 steps, warm-up 300, --lr-scale 1.0, eval every 1000) = the YR10m10-F recipe from the shipped
            weights and d264's full continuation schedule; the old ROUTE-MEM continuation passes --steps 2500 --warmup 1 --lr-scale 0.1.
  no pool   --agent-on-only without --on (RB0, RNONE) means no row carries the agent hinge: run as --agent-lam 0 (on_train.setup would
            refuse the flag without a pool), so the RB / RB0 / RBS / RNONE commands share one $REC.
  pool rows --rm-drop never zeroes a pool row's (--on) route input (the route hinge must not train a turn without its reason); the drop rng
            stream still draws one number per batch row. --on-lon / --on-lon-lam / --on-road / --on-road-pai(-w) go to on_train.setup unchanged.

Next-round addition (2026-10-13, experiments/route_follow/plans/2026-10-13-allrows-prereg.md):
  --ego-drop-speed P   the speed history of the fed ego row (EGO_SPEED: vx, vy, ax, ay and the x / y of the 4 history poses, i.e. everything
            that carries the ego's own speed; `present`, the command and the 4 history yaws stay) zeroed on a share P of the non-anchor rows of
            every batch (pool rows included), own rng stream [seed, 0, 71] (on_train --ego-drop's stream; every other stream unchanged).
            Training only: nothing is saved with the weights, the served row is unchanged. The training-side fix for the slowness ratchet
            (research/next-round/steer-long.md 4.4: the plan follows its own speed history). on_train's --ego-drop (the whole EGO_STATE,
            yaws included) stays refused here. 0 = off: no operation is added.

SLOW S1 / S2 addition (2026-10-15, experiments/route_follow/plans/2026-10-15-slow-s1s2-prereg.md):
  --ego-drop-open   with --ego-drop-speed P: the speed history is zeroed only on the drawn rows that are OPEN (the same draw from [seed, 0, 71],
            then masked): pool rows by on_lon's `p_open_design` (no A-eligible actor, no eligible stop line, no eligible turn peak within 60 m)
            from --open-pack (default on_lon.pack_path(<POOL>)), offline rows (normal and yaw-rate) by rf_lonol.py's `free` cell (no A-eligible
            actor <= 40 m, no stop line of any type <= 60 m, route turn over 60 m < 30 deg, speed >= 2 m/s) joined by token from --open-cells;
            a row without a label (no token match, a pool row absent from the pack, a demonstration pool's rows) is never dropped. Names of a
            merged pool (on:<prefix>:<name>, tab `pool`) are matched without the prefix. Logged: the open / labelled share of every row class
            in the store and per batch (non-anchor rows) and the dropped share, `drop/*` scalars, a summary block `ego_drop_open`. Off: the
            --ego-drop-speed path exactly as before (no operation added).
  --on-weight with weight 0   rows of weight 0 are never drawn; when they are the pool's last rows and every other weight is equal, the draw
            is the uniform draw over the rest (orng.choice(n_rest, k)), i.e. bit for bit the draw of the pool without them (gate of the
            merged on1_L18RB0 + on1_PAITV pool against S1).

Turning-school addition (2026-10-13, experiments/route_follow/plans/2026-10-13-turning-school-prereg.md):
  --on-weight KEY=VALUE:W,...   pool rows are drawn with probability proportional to the product of W over the matching terms (string keys of
            the pool's tab.npz, e.g. src=synthetic:0.5,topo=straight_exit:2; a row without the key or value has factor 1): the expected
            weight of a row class in the imitation / hinge sums, without touching the loss. Same orng stream [seed, 0, 53] (numpy's
            weighted choice). '' = off: the uniform draw, bit for bit as before.

  --on-cap-scene C [--on-cap-in KEY=VALUE]   per-scene draw cap (2026-10-14, turning school phase 2: a branch-search pool is dominated by the
            scenes with many completing branches): among the pool rows matching --on-cap-in (default: every pool row), the weights after
            --on-weight are rescaled per scene (tab `log`) so that no scene holds more than a share C of the subset's draw mass. Water-filling:
            scenes over C are set to C, the rest scaled up in proportion, repeated until none exceeds; the subset's total mass is kept, so a
            --on-weight W computed for the subset's share of the draws stays exact. Infeasible (scenes x C < 1) is an error. Same orng stream
            and weighted choice as --on-weight; 0 = off. `capcheck` is its known-answer test (CPU: 3 scenes 70 / 20 / 10 % capped at 0.4 ->
            40 / 40 / 20 % within 1e-6, the rest of the pool untouched; exit 2 otherwise); `draws --on POOL [--on-weight ..] [--on-cap-scene ..]`
            prints the draw shares the trainer would use (per `pool` tag, and per scene inside the cap subset before / after), no GPU.

  --on-mode mixed   on_train's per-row mode: pool rows tagged mode == "self" (S's on-policy rows) keep the `self` recipe, the rest (demonstration
            rows) are `both`; `mixcheck` is its known-answer test: on one setup, a batch whose pool rows are all `self` rows gives the
            `self` loss and gradient exactly, a batch of `both` rows the `both` ones (exit 2 otherwise).

Longitudinal safety (2026-10-13, experiments/route_follow/plans/2026-10-13-lon-prereg.md):
  --lon-lead W --lon-curve W --lon-stop W   the timing-only terms of onpol_nuplan lib/on_lonsafe.py on the pool rows (lead envelope, curve speed,
            stop-respect): any > 0 sets --on-lon on_lonsafe and --on-lon-lam 1 (the weights carry the scale: each term enters as W x ho_w x
            sum over the pool rows / n_imit); refused together with another --on-lon module. All 0 (default) = off: no operation is added.

Route v2 (2026-10-15, experiments/route_follow/plans/2026-10-15-route-v2-prereg.md); every flag off = the recipe above operation for operation:
  --rm-order        the ordered, progress-anchored memory layout (lib/rmem.py "Ordered layout": the 40 slots at the route arc they claim,
            + d_turn, trav; DIM_O = 203) from feat/<data>.o.npz (rm_build.py order); the head gets order=True (sidecar head `order`); an --init
            with the old layout is converted by rmem.order_state (every tensor kept, the scalar map at 0).
  --lane-lam L [--lane-dz 0.5] [--lane-neg 0.2] [--lane-shift 3.5] [--lane-over-lam L] [--lane-over-deg 10]
            near-field lateral hinge to the CARRIED route on every non-anchor row whose memory is fed (pool rows included): the row's memory
            polyline in the input layout (2 m slots, -6 .. +72 m) as target, on_route.near (shape-only lateral gradient, poses covered by a
            segment), relu(distance - dz), pose weights 0 / 2 / 2 / 2 / 1 / 1 / 1 / 1 at 0.5 .. 4 s (1.0 - 2.0 s weight 2); summed / non-anchor
            rows x L. Negatives: a share --lane-neg of the eligible rows (non-anchor, memory fed, a valid slot at >= 0 m, not a --lc-cf row), own rng
            stream [seed, 0, 97], get a FAKE input memory: the same route shifted sideways by +-shift m (same direction), or (half of the rows
            that qualify: NAVSIM rows whose route turns >= 15 deg within 40 m (against the +0 m slot's heading, or the ego's in the near gap)
            and whose token has no alternative-exit counterfactual in runs/route_follow/cf/rows, i.e. no junction choice) a straight route
            (from the +0 m slot along its heading, or from the ego along its own heading in the near gap); the hinge
            target stays the ORIGINAL route and the imitation target is unchanged. Over-turn guard: relu(s (psi_2s - h_route(a_2s)) - 10 deg)
            (radians) with h_route the target route's heading at the plan's own arc length at 2 s and s the side of the route's turn between 0
            and that arc (|turn| >= 5 deg, else 0): fires when the plan at 2 s points more than 10 deg further into the turn than the route;
            gradient through the plan heading only; summed / non-anchor rows x --lane-over-lam (default L).
  --lc-cf [--lc-k 8] [--lc-lam L] [--lc-dz 0.5]
            lane-change counterfactual pairs: the `lc` rows of runs/route_follow/cf/rows/<shard>.npz (with --rm-order the stay-lane memory
            from <shard>.o.npz, rf_cfpool build --order; rf_cfpool build: navtrain tokens whose
            matched lanes change within 4 s and end >= 2 m off the t0 lane; ok = the stay-lane route generated and >= 2 m off the real memory;
            train-split rows only). Every batch draws k tokens (own stream [seed, 0, 101]): the token's real row replaces one of the last k normal
            rows (its memory never dropped nor faked; imitation of the log, the lane hinge on its real memory) and a stay-lane row is appended
            (same camera tokens, ego row with the counterfactual command, memory = the stay-lane route; no imitation, no other term: the lane
            hinge to the stay-lane label polyline with --lc-dz and the over-turn guard, summed / non-anchor rows x --lc-lam (default
            --lane-lam, else 10)). --lc-adapter-only: that term's gradient (torch.autograd.grad of its scaled value) is added to the adapter
            and route-branch parameters only; the shared network gets none from the stay rows (Amendment 4).
  --rm-latfeat      (with --rm-order) the route's signed lateral offset at 10 / 20 / 30 / 40 m of route arc (the ordered slots' y), / 10 with
            validity bits, through a zero-initialised map added to the branch's ego token (rmem.lat_feats; computed inside the head from the
            fed block, so nothing is rebuilt and the served values are the trained ones; sidecar head `latfeat`).

Approach course (2026-10-15, experiments/route_follow/plans/2026-10-15-approach-course-prereg.md):
  --app-lam W       the approach terms of onpol_nuplan lib/on_approach.py (entry-state pose L1, 1 - 2 s blend L2 x 2, speed excess over the
            human L3) on the pool rows labelled in <POOL>/approach.npz (rf_apppool.py): sets --on-lon on_approach and --on-lon-lam 10 W, so
            each term enters as 10 W x ho_w x sum over the pool rows / n_imit (W = 1: the route hinge's per-row weight in RB-F, the S agent
            hinge's); refused together with another --on-lon module. 0 (default) = off: no operation is added.
  --app-v2          (with --app-lam) APPROACH COURSE v2 (plans/2026-10-15-approach-v2-prereg.md): on_approach.configure(True): no term in the
            first 1.5 s after the handoff, lateral dead zone 0.5 m to the human's entry offset clipped to <= 0.75 m inside the route, the speed
            term only with the ego inside the 26 m window, a symmetric speed term to the human on the straight negatives, the 2 s over-turn
            guard (10 deg); needs <POOL>/approach_v2.npz (rf_apppool.py v2). Off (default): on_approach exactly as v1.

RECOVERY-ENVELOPE (2026-10-16, experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md); off (default) = no operation added:
  --env-lam W       the asymmetric longitudinal speed envelope of onpol_nuplan lib/on_envelope.py (envU: speed above the row's upper bound,
            envL: below its lower bound; timing-only plan geometry) on the pool rows labelled in <POOL>/envelope.npz (rf_recov.py pack): sets
            --on-lon on_envelope and --on-lon-lam 10 W (the --app-lam scale); refused together with another --on-lon module.
  --env-lo-scale F  on_envelope's lower-bound term (envL) x F (Amendment 3: upper and lower bounds at their own weights); 1 = off.
  --on-lon-all      with --on-mode mixed: the --on-lon terms on every pool row of the batch (on_train LossesOn._lon_all), not only on the
            `both` rows (the RL arm: envelope on the S / envelope self rows, recovery rows as `both`).

STACK (2026-10-16, experiments/route_follow/plans/2026-10-16-stack-prereg.md); every flag off = the recipe above operation for operation:
  --rx [--rx-drop 0.25 --rx-lr 3e-4 --rx-warmup 100]   the reflex module of lib/reflex.py on the route branch (self-state appended to the fed row
            from $DATA_DIR/runs/route_follow/stack/self/<POOL>.npz for the pool rows, zeros + valid 0 elsewhere; zeroed on a share --rx-drop of
            the pool rows of every batch, own rng stream [seed, 0, 113]); its parameters are a fourth group (own lr / warm-up, not x --lr-scale)
            and take only the constraint side's gradient (lib/stack_train.py); sidecar key `reflex`. An --init / --rm-from with the module
            loads it; an init without it starts the module at its zero-output init.
  --dual / --dual-tau / --dual-level / --dual-eta, --cons PATH, --imit-drop, --pref W   lib/stack_train.py (its docstring).

AWR-ONPOL (2026-10-17, experiments/awr_onpol/plans/2026-10-17-awr-onpol-prereg.md; lib/awr_train.py); off (default) = no operation added:
  --awr-pack FILE [--awr-k 16 --awr-lam 1 --awr-mode filt --awr-keep 0.1 | --awr-mode awr --awr-beta B --awr-wmax 20] [--awr-kinds 1]
            consequence-weighted cloning: every step K rows of the --on pool, drawn from the outcome pack (aw_score.py: advantage A of the
            decision against near-identical peers) on its own rng stream [seed, 0, 167], get a second forward; loss = the imitation distance
            to the plan their own policy output there, summed / normal rows x lam (filt: rows with A > keep, uniform; awr: p ~ exp(A / B)).

DESIRE D1 / D2 (2026-10-17, WAIT-TRAIN prereg Amendment C; lib/desire_train.py); every flag off = the recipe above operation for operation:
  --desire-rows F   D1 row table (dt.py rows: names, side +1 left / -1 right, src nav | pool; the rule is lib/route_desire.py's): every non-anchor
            row of the table in a batch is fed the lane-change desire of its side (PModel.forward desire=, dc.py's queue), pulse age drawn per
            row from desire_train.AGES (own rng stream [seed, 0, 137]); the row's loss is its own recipe's (navtrain rows: imitation of the log;
            pool rows: their hinges / --on-lon). --d1-k K: K table rows of the navtrain train split replace the last K normal rows of every batch
            (own stream [seed, 0, 131]; --lc-cf's slots).
  --d2-lam L [--d2-frac 0.125 --d2-teacher DIR]   desire-delta distillation: the anchor rows and a share --d2-frac of the other normal rows
            without a D1 desire (own stream [seed, 0, 127]: side, age, membership) get a second forward with a random lane-change desire; loss =
            desire_train.d2_loss of (plan(x, d) - sg[plan(x, 0)]) against shipped Cinque's increment from DIR/<data>.npz (dt.py teacher), mean
            over those rows x L. 0 = off: no operation is added.

RETAIN / ANTI-COPY (2026-10-17, experiments/wait_cap/plans/2026-10-17-retain-anticopy-draft.md; decision 335 options iii / iv); every flag off
= the recipe above operation for operation:
  --kd-lam L [--kd-mode lon --kd-k 16 --kd-pool on1_WAIT --kd-groups AR:2,W:1 --kd-nav FILE --kd-nav-k 0]   P0 retention (lib/retain_train.py,
            decision 336's aim): every step K rows of the retention pool (rolling arrivals / stops / yields, standing waits; rows the stage never
            exercises) and K_nav navtrain approach rows fed the constant-velocity history get a second forward; loss = their plan's arc-length
            profile against shipped Cinque's (lon: the lateral path untouched; full: the anchor `cons` distance to the whole plan), summed over
            the 8 poses, mean over the rows x L; own rng stream [seed, 0, 151]. 0 = off: no operation is added.
  --ego-scale-speed P [--ego-scale-max M]   history-speed noise (the ego-history bottleneck of ChauffeurNet's past-motion perturbation, against
            the own-motion trend of decision 315): on a share P of the non-anchor, non-pool rows (the imitation rows), the fed speed history is
            rescaled by s ~ U[1 - M, 1 + M]: vx, vy x s, ax, ay x s^2, the history poses' x, y x s (the same path at s x the speed; yaws, the
            command and the route memory unchanged); the imitation target stays the log, so the plan's speed must come from the scene, not the
            history's scale. Own rng stream [seed, 0, 149]; training only. 0 = off: no operation is added.

SELF-ITER (2026-10-17, experiments/self_iter; lib/self_iter.py); off (default) = the recipe above operation for operation:
  --si-scan FILE [--si-mode top --si-top 0.1 --si-mult 3 | --si-mode rank --si-gamma G] [--si-key ade] [--si-k-s 13]   hard-example draw from a
            policy's per-row error scan (si.py scan): the nN normal rows are drawn with numpy's weighted choice on rm_train's own stream (weights
            ranked inside the navtrain train rows); with a merged pool holding new trainval imitation rows (`pool` t, mode both) the pool draw's p
            gives the S rows (`pool` s) the mass --si-k-s / --on-k and the new rows the rest (ranked inside them; `hold` 1 rows weight 0), and
            --rm-drop treats the new rows as normal rows. --si-mult 1: the uniform draw over the same rows.

ROUND7 joint arm (2026-10-22, experiments/round7/plans/2026-10-22-joint-arm.md); off (default) = nothing touched:
  --cmd-rule heading   the command (ego cols 1:4) of the pool rows of CMD_PARTS (PAI human, DAgger, S) := rmem.cmd_heading on the row's own fed
            route memory, the rule the PAI driver serves with JEV_CMD=heading; navtrain rows keep NAVSIM's command (cmd_rule).
NR1 (2026-10-17, experiments/defect_fix/plans/2026-10-17-nr1-prereg.md); every flag off = the recipe above operation for operation:
  --on-wfile FILE   per-row draw weights of the --on pool rows from FILE (npz: `names` = the pool's tab names in order, `w`; defect_fix df.py
            weights): numpy's weighted choice on the pool stream [seed, 0, 53], as --on-weight; refused with --on-weight / --on-cap-scene / --si-scan.
            ROUND3c (experiments/round3/plans/2026-10-21-round3c-prereg.md): a file that also holds `nav_names` / `nav_w` gives the nN normal
            (navtrain) rows per-row draw weights too (rows not in the file weigh 1; numpy's weighted choice on rm_train's own stream, as
            SELF-ITER's normal-row draw); a file without them changes nothing.
  --on-normal       the pool rows are imitation rows like the normal rows for --rm-drop (their memory dropped at the same share, the drop stream
            unchanged) and --ego-scale-speed (SELF-ITER's rule for its trainval rows): the base stage's trainval imitation pool (--on-mode imit).
  --lat-onc W --lat-soft W --lat-side W [--lat-file F --lat-depth 0 --lat-soft-depth 0 --lat-buf 0.4]   the hard / soft lateral terms of
            experiments/defect_fix/lib/nr1_lat.py on the pool rows (opposite-direction lanes by point at the plan's own heading, same-direction
            lanes off the ego's own chain, the agent hinge's side buffer; its docstring): any > 0 sets --on-lon nr1_lat and --on-lon-lam 1 (the
            weights carry the scale: each term enters as 10 W x ho_w x sum over the pool rows / n_imit); refused with another --on-lon module.

NR1HVR (2026-10-18, experiments/defect_fix/plans/2026-10-17-nr1-prereg.md supplement 8, draft); every flag off = the recipe above
operation for operation (identity: experiments/defect_fix/results/identity_nr1hvr*.json):
  --on-arc W        the two-sided longitudinal arc-length term of experiments/defect_fix/lib/nr1_lon.py on the pool rows (target: the arc of the
            row's own `fut`, the human's speed blended from the ego's over 2 s; rows where the logged human stands or stops in the horizon are
            excluded; gradient only along each pose's own heading), through nr1_lat as the --on-lon module: 10 W x ho_w x sum / n_imit.
  --on-hold-res R [--on-hold-n 2048]   the pool logs with sha1(log) % 10 == R are never drawn (this changes the training rows); at step 0 and
            every dev eval, N rows of them and N rows of the drawn logs (fixed, rng [seed, 0, 167], its own stream) are scored in eval mode on
            the pool terms (agent, road, the --on-lon terms; per-row mean and active share) and, with --on-arc, the 4 s arc bias against the
            target and the shares slower than 0.8x / faster than 1.2x it -> scalars pool_held/* and pool_fit/*.
  --on-free W       (NR1FREE, NR1 prereg supplement 15) the one-sided longitudinal consequence term of experiments/defect_fix/lib/nr1_free.py
            on the S rows: the plan's 4 s arc below 0.8 x the human's (the smaller of his logged and his reachable arc) with no logged object in
            the plan's extended sweep -> the per-step deficit, gradient only along each pose's heading; through nr1_lat: 10 W x ho_w x sum / n_imit.
            0 (default) = off: nothing loaded, no operation added.
  --on-stand W [--on-stand-margin M]   (NR1STAND, NR1 prereg supplement 17) the mirror term of nr1_free.py on the standing S rows (the visited
            state below 0.5 m/s, the human's logged 4 s arc below 0.5 m): the plan's arc beyond max(his two arcs) + M (1.0 m) per step, gradient
            only along each pose's heading; through nr1_lat: 10 W x ho_w x sum / n_imit. 0 (default) = off. With --diag: grad/lon_free and
            grad/lon_stand (monitoring) when those terms are on.
  --on-stand-roll V   (STAND-R, NR1 prereg supplement 20) STAND also on rolling S rows (0.5 <= ego t0 speed < V m/s, the human standing all
            4 s) with the cap max(b_j, max(g, 0) + M), g = the human's standing station minus the ego's (<pool>/stand_gap.npz); their part is
            logged as on/lon_standr. 0 (default) = off: nothing loaded, no operation added.
  --on-stand-launch H   (STAND-R2, NR1 prereg supplement 23) STAND / STAND-R rows dropped when the human launches (> 0.5 m or >= 1 m/s)
            within 4 + H s of the row, or the log ends before then (<pool>/stand_launch.npz). 0 (default) = off: nothing loaded.
  --on-free-clear CMIN,CTH   (NR1 prereg supplement 19) FREE's free check made stricter: the safety distance ahead of the bumper is
            max(2 m + 0.5 s x v, CMIN m, CTH s x v) (v = the reference speed of each 0.5 s step). 0,0 (default) = off.
  --on-stoppt W     (NR1STOPPT, NR1 prereg supplement 18) nr1_free.py's "do not pass the human's stop point" term on the approach S rows of
            <pool>/stoppt.npz (defect_fix scripts/stoppt.py labels): the plan's arc beyond max(rest + 1 m, v0^2 / 8) per step, gradient only along
            each pose's heading; through nr1_lat: 10 W x ho_w x sum / n_imit. 0 (default) = off. With --diag: grad/lon_stoppt.
  --on-turncap W    (TURN-CAP, NR1 prereg supplement 21) nr1_free.py's one-sided upper limit on moving S rows in sharp turns (ego t0 >= 2 m/s,
            the human's logged 4 s arc >= 4 m, his path's heading range over -10 .. +20 m around the ego's station >= 45 deg; labels
            <pool>/turn_cap.npz from defect_fix scripts/turn_cap.py label): the plan's arc beyond 1.05 x max(arc(fut), arc(fut_reg)) per step,
            gradient only along each pose's heading; through nr1_lat: 10 W x ho_w x sum / n_imit. 0 (default) = off. With --diag: grad/lon_turncap.
  --on-tightin W    (TIGHT-IN, NR1 prereg supplement 22) defect_fix lib/nr1_tight.py on moving S rows in tight turns (ego t0 >= 2 m/s, the
            human's logged 4 s arc >= 4 m, his path turning >= 15 deg over the next 20 m with max 1 m curvature >= 0.108 /m; labels
            <pool>/tight_in.npz from defect_fix scripts/tight_in.py label): the plan outside the human's line by > 0.2 m (across each pose's
            heading; on/lon_tightin) + TURN-CAP's arc cap (along the heading; on/lon_tightina); through nr1_lat: 10 W x ho_w x sum / n_imit
            each. 0 (default) = off. With --diag: grad/lon_tightin (both parts together).
  --flat-brk FILE   (SIMP-F, lon-anchor prereg supplement 3) navtrain training rows whose token is in FILE (lead_break.py's brake-needed
            rows) are fed a flat history on every draw: ax 0 and the 3 past poses re-spaced at the current speed (parity_adapter.cv_history);
            their label is kept. '' (default) = off: no operation added.
  --flat-p P        (ROUND2, experiments/round2/plans/2026-10-20-round2-prereg.md) with --flat-brk: each draw of a listed row is flattened
            with probability P (one uniform per batch row from its own stream [seed, 0, 179]). 1 (default) = every draw, as before (no stream,
            no operation added); 0 = off (the list is not loaded: bit-identical to no --flat-brk).
  --simp-region W / --simp-clear W   (SIMP, defect_fix plans/2026-10-20-lon-anchor-prereg.md supplement 2) defect_fix lib/nr1_simp.py on the
            S rows: the road hinge and lat_onc as one hinge on their union, the agent hinge and lat_side as one clearance hinge (lead objects
            may only shorten the plan); each replaces the trainer's own hinge on the S rows (nr1_lat.attach); through nr1_lat: 10 W x ho_w x
            sum / n_imit. 0 (default) = off: nothing loaded, attach does nothing.
  --on-anchor W     (LON-ANCHOR, defect_fix plans/2026-10-20-lon-anchor-prereg.md) defect_fix lib/nr1_anchor.py on the S rows of
            <pool>/lon_anchor.npz (scripts/lon_anchor.py labels): the plan's arc against the human's own travel started from the ego's station
            on his path (smooth L1 per pose, two-sided), gradient only along each pose's heading; rows with a lead closer than his are skipped;
            through nr1_lat: 10 W x ho_w x sum / n_imit. 0 (default) = off. With --diag: grad/lon_anchor.
  --on-arc-rows brake   (JTB / JTN, NR1 prereg supplement 10) the arc term's rows: also the rows where the human stops in the horizon; only
            the rows where he stays below 0.5 m/s all 4 s are excluded (nr1_lon.labels). Loads the labels even at --on-arc 0 (monitoring) and adds
            to the held / fitted slices the 4 s arc bias per row kind (cruise / brake / stop / launch / still) -> pool_*/arc4_bias_<kind>,
            pool_*/n_<kind>. Default "move" = the old rule, no operation added.
  --diag [--diag-every 100]   monitoring only, no effect on the update: act/* (share of rows with a nonzero term, pool rows and navtrain rows
            apart), grad/global (norm before the clip) and grad/clip (share of steps where the 1.0 clip fired) every step; every K steps the
            trunk-gradient norm (parameter groups base + adapter; the route branch is left out) of each weighted term: imit, nav_hinge (P2H10's
            drivable hinge on navtrain rows), pool_hinge (agent + road on pool rows), lat (the three lateral terms), lon (--on-arc), pool (the
            three pool groups together) and cos(imit, pool), through torch.autograd.grad(retain_graph=True) on the scaled terms before the
            step's backward (.grad untouched). With --compile it sets torch._functorch.config.donated_buffer = False (donated buffers refuse
            retain_graph; memory reuse only); the compiled smoke pair in results/identity_nr1hvr_smoke.json checks the update is unchanged.
  --diag-infl N [--diag-freeze]   (LAUNCH-DIAG, experiments/defect_fix/results/rtja/launch_diag.md; with --diag) monitoring only: at every
            --diag-every step, for fixed probe sets of N rows (the --probe-n trainval launch / standing rows; the fitted pool slice's launch /
            still / cruise S rows), the gradient of the mean plan 4 s arc against each loss term's gradient (the imitation term split by row
            group: navtrain, yaw-rate, trainval and each trainval draw group; the same rows without the history-speed noise; the S-row agent /
            road hinges, the trainval rows' hinges, the drivable hinge, the three lateral terms, the total): cos and d1k = the arc change per
            1 000 steps that term alone predicts under Adam's current preconditioner at each group's base lr -> scalars infl/<probe>/*.
            --diag-freeze: every group's lr is 0 each step (the weights stay at the init, Adam's moments accumulate). Off: no operation added.
  --lr-floor F      the cosine decays to F x base lr instead of 0 (lr = base x warm-up x (F + (1 - F) cos)); 0 (default) = the plain cosine.
            The warm-up length is the existing --warmup.

Speed (2026-10-17, docs/speed.md; validation experiments/cl_infra/plans/2026-10-17-compile-prereg.md):
  --compile   the main training forward of every step runs through torch.compile(model, dynamic=False) (inductor; pp_train --compile's
            pattern): ~2x the steps per second on the RSG-L18 recipe. Not bit-identical to eager (fused fp16 kernels round differently;
            pp_train: compiled runs repeat each other). The second forwards (--kd-lam, --awr-lam, --d2-lam), dev eval and the reliance read stay eager.
            Default ON for new lanes since 2026-10-17 (user decision; experiments/cl_infra/results/compile.md: dev / held / PAI within seed
            noise, 1.7x wall). Lanes started before keep eager through JEV_RM_EAGER=1 in their pool --env (nx / wt / si / aw / rm_chain),
            so no lane switches mid-way. --no-compile (or JEV_RM_EAGER=1): eager, no operation added, bit-identical to the old default.

ANCHOR (2026-10-17, experiments/route_mem/results/anchor_check.md; lib/rmem.py "Anchored layout"); off (default) = every file, head and term
as before:
  --rm-anchor       the anchored memory layout (rmem.memory_a: merge_a's fold-back fix, AlpaSim's trim tracked by the waypoint count, the
            car's driven chord) in the ordered slot format (DIM_O = 203, so the head is RouteHead(order=True)) from feat/<data>.a.npz
            (rm_build.py anchor) for every data dir of the store and, with --kd-lam, the retention pool's feat/<pool>.a.npz; sidecar head
            `order: true, anchor: true` (the drivers serve the layout from it, rmem.head_anchor); an --init with the old layout is converted
            as for --rm-order (rmem.order_state), an ordered init is taken as it is; refused with --lc-cf (no anchored lane-change rows).

WIN (2026-10-22, never trained: the 10-22 base retrain was dropped, experiments/round9/results/route_check/README.md; lib/rmem.py "Window layout"); off (default) = every file, head and
term as before:
  --rm-win          the window layout (the anchored layout's merge and arc, 40 slots at -6 .. L(v) m with L(v) = clip(5 v + 20, 40, 78) of the
            row's fed speed, the heading command): the head reads the first DIM_O numbers of feat/<data>.w.npz (rm_build.py win; round9 nb.py
            pai for the PAI pool parts) for every data dir of the store; the file's command (rmem.cmd_heading of its fixed-grid block) replaces
            the ego row's command on the pool rows of CMD_PARTS (AlpaSim-route rows, round7's --cmd-rule set; navtrain rows keep NAVSIM's);
            sidecar head `order, anchor, win (rmem.WIN), cmd: heading` (the drivers serve the layout and the command from it). An --init must
            carry the same `win` (no conversion); refused with --cmd-rule, --kd-lam, --lc-cf, --lane-lam.

  RB:  rm_train.py train --tag RB-F-s0 --steps 10000 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 \
       --seed 0 --on on1_CF --on-k 12 --on-mode self --on-lon on_route          (RB0: the same without the four --on flags)

  $DATA_DIR/envs/op-train/bin/python experiments/route_mem/scripts/rm_train.py train --tag RM-L18-s0 --init YS-L18-s1 --steps 2500 --warmup 1 \
      --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --on on1_L18YS --on-k 13 \
      --on-split alpasim_nuplan/onpol-train --on-mode self --seed 0
  ... rely --tag RM-L18-s0 [same recipe flags]     the reliance read alone, from a saved checkpoint
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/onpol_nuplan/scripts"), str(_R / "experiments/route_mem/lib")]
import argparse, hashlib, json, os, time  # noqa: E401,E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402

import on_train as OT  # noqa: E402  (sets the repo paths)
import rmem as RM  # noqa: E402

FEAT = None
HEAD = dict(d=128, layers=2, heads=4)
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]              # ego_features: vx, vy, ax, ay, history poses x / y (not present, cmd, yaw)
EGO_SCALE_POW = [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]                  # --ego-scale-speed: power of s per EGO_SPEED column (velocity, acceleration, pose)


def feats(cfg_data, S, dev, order=False, old=False, anchor=False, win=False, cmdout=None):
    """Route features (S.n, DIM | DIM_O with order) float32 on dev, and per Store row: junction distance / route turn (dev rows of the NAVSIM
    dirs, else nan), the source dir index. Asserts every file's names equal the Store tab's names of its dir. old: also the input layout's point
    block (S.n, DIM) f16 on dev and trav (S.n,) (route v2's targets / negatives; with order both come from the .o files). win: the window
    layout's file (.w; the route part = its first DIM_O numbers) and its command (S.n, 4) appended to the list cmdout."""
    import torch
    from jevdrive.common import data_dir
    root = data_dir() / "runs/route_mem/feat"
    F, JD, TU, src, okc, FO, TR = [], [], [], [], {}, [], []
    o = 0
    for j, d in enumerate(cfg_data):
        z = np.load(root / f"{d}.npz")
        n = len(z["names"])
        assert np.array_equal(z["names"], S.tab["names"][o:o + n]), f"{d}: feature rows differ from the Store's"
        f = z["f"].astype(np.float32)
        f[~z["ok"]] = 0
        if order:
            zo = np.load(root / f"{d}{'.w' if win else '.a' if anchor else '.o'}.npz")   # ANCHOR / WIN: the layout's file (rm_build.py anchor / win)
            assert np.array_equal(zo["names"], z["names"]) and (win or np.array_equal(zo["ok"], z["ok"])), f"{d}.o: rows differ"
            TR.append(zo["trav"].astype(np.float32))
            f = zo["f"].astype(np.float32)
            if win:                                                                     # WIN: route part | command (rmem.compose_w)
                assert f.shape[1] == RM.DIM_W and not (zo["ok"] & ~z["ok"]).any(), f"{d}.w: width / ok"
                cmdout.append(f[:, RM.DIM_O:].copy())
                f = f[:, :RM.DIM_O]
            f[~zo["ok"]] = 0
        elif old:
            TR.append(np.zeros(n, np.float32))
        if old:
            FO.append(f[:, :RM.DIM].astype(np.float16))             # the input layout's point block: hinge target, negatives
        F.append(f)
        JD.append(z["jdist"] if "jdist" in z.files else np.full(n, np.nan, np.float32))
        TU.append(z["turn40"] if "turn40" in z.files else np.full(n, np.nan, np.float32))
        src.append(np.full(n, j))
        okc[d] = float(z["ok"].mean())
        o += n
    assert o == S.n, (o, S.n)
    ex = (torch.as_tensor(np.concatenate(FO), device=dev), torch.as_tensor(np.concatenate(TR), device=dev)) if old else (None, None)
    if win:
        cmdout[:] = [np.concatenate(cmdout)]
        assert len(cmdout[0]) == S.n
    return (torch.as_tensor(np.concatenate(F), device=dev), np.concatenate(JD), np.concatenate(TU), np.concatenate(src), okc) + ex


def rows_ade(model, S, rows, W, mask, bs=256):
    """Per-row ADE of the 8 rear-axle poses to the log (rows with a future), memory on (mask False) or masked."""
    import torch
    import pp_train as T
    pi = torch.as_tensor(S.pi, device=S.ego.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=S.ego.device)
            ego = S.ego[r].clone()
            if mask:
                ego[:, RM.EGO_DIM:] = 0
            p = model(S.front[r], ego, S.tc[r], None, None, nv=None if S.nv is None else S.nv[r]).float()[:, pi].view(-1, 33, 15)
            x, y, _ = T.rear(p, S.cam_x[r], W)
            out.append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1).cpu().numpy())
    return np.concatenate(out)


def rely(model, S, dv_o, W, JD, TU, logs):
    """The registered reliance read (prereg 'Gates'): delta = ADE(memory masked) - ADE(on) per dev row; junction rows vs straight rows,
    log-clustered bootstrap (B 10000, rng 0) of mean delta (junction) - mean delta (straight)."""
    import torch
    model.eval()
    has = S.has_fut[torch.as_tensor(dv_o, device=S.ego.device)].cpu().numpy().astype(bool)
    r = dv_o[has]
    on, off = rows_ade(model, S, r, W, False), rows_ade(model, S, r, W, True)
    d, jd, tu, lg = off - on, JD[r], TU[r], np.asarray(logs)[r]
    J = np.isfinite(jd) & (jd <= 40)
    St = ~np.isfinite(jd) & np.isfinite(tu) & (tu < 5)
    ul = np.unique(lg)
    li = np.searchsorted(ul, lg)
    rng = np.random.default_rng(0)
    bj, bs = np.zeros(10000), np.zeros(10000)
    sj, nj = np.bincount(li[J], d[J], len(ul)), np.bincount(li[J], minlength=len(ul))
    ss, ns = np.bincount(li[St], d[St], len(ul)), np.bincount(li[St], minlength=len(ul))
    for b in range(10000):
        c = np.bincount(rng.integers(0, len(ul), len(ul)), minlength=len(ul))
        bj[b], bs[b] = (c * sj).sum() / max((c * nj).sum(), 1), (c * ss).sum() / max((c * ns).sum(), 1)
    diff = bj - bs
    res = dict(rows=int(len(r)), logs=int(len(ul)), ade_on=float(on.mean()), ade_masked=float(off.mean()), junction_rows=int(J.sum()), straight_rows=int(St.sum()),
               d_junction=float(d[J].mean()), d_straight=float(d[St].mean()), d_all=float(d.mean()),
               diff=float(d[J].mean() - d[St].mean()), diff_lo=float(np.quantile(diff, 0.025)), diff_hi=float(np.quantile(diff, 0.975)),
               ade_on_junction=float(on[J].mean()), ade_on_straight=float(on[St].mean()))
    res["read"] = bool(res["d_junction"] >= 0.01 and res["diff_lo"] > 0)
    return res


def head_of(a) -> dict:
    """The branch config: HEAD, + order=True with --rm-order (route v2); `rely` reads it from the checkpoint's sidecar."""
    if a.rm_from and RM.has_branch(a.rm_from):
        a.rm_order = RM.head_order(a.rm_from)
    if a.rm_from and RM.has_branch(a.rm_from):
        a.rm_latfeat = bool(json.loads((__import__("pp_train").proot("runs", a.rm_from) / RM.SIDECAR).read_text()).get("head", {}).get("latfeat", False))
    h = dict(HEAD, order=True) if getattr(a, "rm_order", False) else dict(HEAD)
    if a.rm_from and RM.has_branch(a.rm_from):                        # ANCHOR: the layout comes with the checkpoint (sidecar head `anchor`)
        a.rm_anchor = RM.head_anchor(a.rm_from)
    if getattr(a, "rm_anchor", False):                                # ANCHOR: the ordered slot format, anchored at the car (rmem.memory_a)
        h = dict(HEAD, order=True, anchor=True)
    if a.rm_from and RM.has_branch(a.rm_from):                        # WIN: the layout comes with the checkpoint (sidecar head `win`)
        a.rm_win = RM.head_win(a.rm_from) is not None
    if getattr(a, "rm_win", False):                                   # WIN: the anchored layout's slots at the speed-scaled window + the command
        h = dict(HEAD, order=True, anchor=True, win=dict(RM.WIN), cmd="heading")
    if getattr(a, "rm_latfeat", False):
        assert h.get("order"), "--rm-latfeat needs --rm-order"
        h["latfeat"] = True
    return h


def build(a, dev):
    """on_train.setup + the branch + the route features. -> (ns, route params, JD, TU, okc)."""
    import torch
    import pp_train as T
    init, a.init = a.init, ("" if a.rm_from or (a.init and RM.has_branch(a.init)) else a.init)   # setup's load_init: plain P2 checkpoints only
    ns = OT.setup(a, dev)
    a.init = init
    head = head_of(a)
    RM.attach(ns.model, **head)
    src = a.rm_from or (a.init if a.init and RM.has_branch(a.init) else "")
    ns.head, ns.converted = head, False
    rxc = None
    if src and RM.has_branch(src):
        rxc = json.loads((T.proot("runs", src) / RM.SIDECAR).read_text()).get("reflex")
    if rxc is not None or getattr(a, "rx", False):                    # STACK reflex module (lib/reflex.py)
        import reflex as RX
        a.rx = True
        RX.attach(ns.model, rx=(rxc or {}).get("rx"))
    ns.rx = bool(getattr(a, "rx", False))
    if src:                                                           # a checkpoint with the branch: the whole state, branch included
        ck = torch.load(T.proot("runs", src) / "ckpt-final.pt", map_location="cpu", weights_only=False)
        sd = json.loads((T.proot("runs", src) / RM.SIDECAR).read_text()) if RM.has_branch(src) else {}
        sh = sd.get("head", HEAD)
        assert {k: v for k, v in sh.items() if k not in ("order", "latfeat", "anchor", "win", "cmd")} == HEAD, f"{src}: branch {sh} != {HEAD}"
        assert sh.get("win") == head.get("win") and sh.get("cmd") == head.get("cmd"), f"{src}: window layout {sh.get('win')} != {head.get('win')}"   # WIN
        assert sh.get("order", False) <= head.get("order", False), f"{src}: an ordered-layout branch needs --rm-order"
        assert sh.get("anchor", False) <= head.get("anchor", False), f"{src}: an anchored-layout branch needs --rm-anchor"   # ANCHOR
        if (head.get("order") and not sh.get("order")) or (head.get("latfeat") and not sh.get("latfeat")):   # route v2: converted init
            ck["model"]["parity"] = RM.order_state(ck["model"]["parity"], head.get("latfeat", False))
            ns.converted = True
        if ns.rx and not any(k.startswith("rx.") for k in ck["model"]["parity"]):   # an init without the module: the module at its init
            ck["model"]["parity"] = dict(ck["model"]["parity"]) | {k: v.detach().cpu() for k, v in ns.model.adapter.state_dict().items() if k.startswith("rx.")}
        ns.model.load_state(ck["model"])
        st = ns.model.adapter.state_dict()
        assert set(st) == set(ck["model"]["parity"]), "adapter keys differ from the checkpoint's"
        da = max(float((v.detach().cpu().float() - ck["model"]["parity"][k].float()).abs().max()) for k, v in st.items())
        dn = max(float((ns.model.net.params[k].detach().cpu().float() - v.float()).abs().max()) for k, v in ck["model"]["net"].items())
        assert da == 0 and dn == 0, (da, dn)
        if not a.rm_from:
            ns.init = dict(init=src, with_branch=True, net_loaded=len(ck["model"]["net"]), adapter_loaded=len(st), max_abs_diff_after_load=0.0,
                           init_dev_ade=sd.get("dev_ade"), **({"converted_to_order": True} if ns.converted else {}))
    lane = getattr(a, "lane_lam", 0) > 0 or getattr(a, "lc_cf", False)
    cw = []
    F, JD, TU, src, okc, ns.f_old, ns.trav = feats(ns.cfg.data, ns.S, dev, order=head.get("order", False), old=lane, anchor=head.get("anchor", False),
                                                   win=head.get("win") is not None, cmdout=cw)
    ns.S.ego = torch.cat([ns.S.ego.float(), F], 1)
    ns.cmd_rec = cmd_rule(ns, a.cmd_rule) if getattr(a, "cmd_rule", "") else None   # round7 joint arm: off (default) = nothing touched
    if head.get("win") is not None:                                   # WIN: the file's heading command on the AlpaSim-route pool rows
        assert not getattr(a, "cmd_rule", ""), "--rm-win carries its own command: not with --cmd-rule"
        ns.cmd_rec = cmd_win(ns, cw[0])
    rp = list(ns.model.adapter.route.parameters())
    ns.rxp = list(ns.model.adapter.rx.parameters()) if ns.rx else []
    ids = {id(p) for p in rp + ns.rxp}
    ns.new = [p for p in ns.model.adapter.parameters() if id(p) not in ids]
    if ns.rx:
        ns.S.ego, ns.rx_rec = rx_feats(ns, dev)
    return ns, rp, JD, TU, okc


CMD_PARTS = ("ph", "ph05", "ph15", "pd", "rb0", "rtb1", "y", "ys")   # pool parts whose command came from AlpaSim's route (PAI human, DAgger, S)


def cmd_rule(ns, rule):
    """--cmd-rule heading (round7 joint arm, change 3, decision 432): the command of the fed row (ego cols 1:4) of every pool row of CMD_PARTS
    := rmem.cmd_heading on the row's own fed route memory (the stored float16 features, as the PAI driver's JEV_CMD=heading computes it at
    serve). navtrain rows (NAVSIM's own driving command) and the other pool parts keep theirs. In place on S.ego; -> record."""
    import torch
    assert rule == "heading", f"--cmd-rule {rule}: heading"
    assert not ns.head.get("order") and not ns.head.get("anchor"), "--cmd-rule: the old (projected) memory layout only"
    S = ns.S
    part = np.array([x.split(":")[1] for x in S.tab["names"][ns.on0:].astype(str)])
    r = torch.as_tensor(ns.on0 + np.flatnonzero(np.isin(part, CMD_PARTS)), device=S.ego.device)
    f = S.ego[r, RM.EGO_DIM:RM.EGO_DIM + RM.DIM].cpu().numpy()
    assert np.array_equal(f.astype(np.float16).astype(np.float32), f), "--cmd-rule: the fed memory is not the stored float16 features"
    old = S.ego[r, 1:4].cpu().numpy()
    new = RM.cmd_heading(f.astype(np.float16))[:, :3]
    S.ego[r, 1:4] = torch.as_tensor(new, device=S.ego.device, dtype=S.ego.dtype)
    pr = part[(r - ns.on0).cpu().numpy()]
    lab = lambda c: np.where(c.max(1) > 0.5, c.argmax(1), 3)  # noqa: E731
    lo, ln = lab(old), lab(new)
    rec = dict(rule=rule, rows=int(len(r)), changed=int((lo != ln).sum()), parts={})
    for p in CMD_PARTS:
        m = pr == p
        if m.any():
            rec["parts"][p] = dict(rows=int(m.sum()), old_lr=round(float(np.isin(lo[m], [0, 2]).mean()), 4), new_lr=round(float(np.isin(ln[m], [0, 2]).mean()), 4),
                                   old_unk=round(float((lo[m] == 3).mean()), 4), changed=round(float((lo[m] != ln[m]).mean()), 4))
    return rec


def cmd_win(ns, C):
    """--rm-win: the command of the fed row (ego cols 1:4) of every pool row of CMD_PARTS := the window file's command (rmem.cmd_heading of the
    row's fixed-grid block, as the drivers serve it); navtrain rows and the other pool parts keep theirs. In place on S.ego; -> record."""
    import torch
    S = ns.S
    rec = dict(rule="win-heading", rows=0, changed=0, parts={})
    if not ns.n_on:
        return rec
    part = np.array([x.split(":")[1] for x in S.tab["names"][ns.on0:].astype(str)])
    sel = ns.on0 + np.flatnonzero(np.isin(part, CMD_PARTS))
    r = torch.as_tensor(sel, device=S.ego.device)
    old = S.ego[r, 1:4].cpu().numpy()
    new = C[sel, :3].astype(np.float32)
    assert ((new == 0) | (new == 1)).all() and (new.sum(1) == 1).all(), "--rm-win: the command is a one-hot of L / S / R"
    S.ego[r, 1:4] = torch.as_tensor(new, device=S.ego.device, dtype=S.ego.dtype)
    pr = part[sel - ns.on0]
    lab = lambda c: np.where(c.max(1) > 0.5, c.argmax(1), 3)  # noqa: E731
    lo, ln = lab(old), lab(new)
    rec.update(rows=int(len(sel)), changed=int((lo != ln).sum()))
    for p in CMD_PARTS:
        m = pr == p
        if m.any():
            rec["parts"][p] = dict(rows=int(m.sum()), old_lr=round(float(np.isin(lo[m], [0, 2]).mean()), 4), new_lr=round(float(np.isin(ln[m], [0, 2]).mean()), 4),
                                   old_unk=round(float((lo[m] == 3).mean()), 4), changed=round(float((lo[m] != ln[m]).mean()), 4))
    return rec


def rx_feats(ns, dev):
    """The fed row ++ the self-state (reflex.SF): pool rows from runs/route_follow/stack/self/<POOL>.npz (names asserted equal), else zeros."""
    import torch
    import reflex as RX
    from jevdrive.common import data_dir
    S = ns.S
    X = torch.zeros(S.n, RX.SF, device=dev)
    rec = dict(pool=None)
    if ns.n_on and RX.OFF:                                            # JEV_REFLEX_OFF=1 (reads only): the module's switch-off, valid 0 everywhere
        rec = dict(pool=ns.cfg.data[-1], off=True)
    elif ns.n_on:
        pool = ns.cfg.data[-1]
        z = np.load(data_dir() / "runs/route_follow/stack/self" / f"{pool}.npz")
        assert np.array_equal(z["names"].astype(str), S.tab["names"][ns.on0:].astype(str)), f"self-state rows differ from {pool}'s"
        X[ns.on0:] = torch.as_tensor(z["f"].astype(np.float32), device=dev)
        rec = dict(pool=pool, rows=int(len(z["names"])), valid=float(z["f"][:, 0].mean()))
    return torch.cat([S.ego, X], 1), rec


# ---------------------------------------------------------------- route v2: lane hinge, negatives, lane-change pairs
LANE_W = (0.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0)                    # plan poses 0.5 .. 4 s: 1.0 - 2.0 s weight 2, 2.5 - 4 s weight 1
LC_GRID0, OLD_GRID0 = -10.0, float(RM.OFFS[0])                       # arc offset of the first point: cf label (rf_cfpool) / old memory window
CURVE_DEG, SIDE_DEG = 15.0, 5.0


def old_poly(f):
    """A point block (n, DIM) tensor (either layout) -> (n, N_PT, 2) row-frame points, NaN where invalid."""
    import torch
    P = f.float().view(-1, RM.N_PT, RM.NF)
    xy = P[..., 1:3] * 20.0
    return torch.where((P[..., :1] > 0.5), xy, torch.full_like(xy, float("nan")))


def seglen(P):
    import torch
    return torch.linalg.norm(P[:, 1:] - P[:, :-1], dim=-1)            # NaN where an end is NaN (on_route.near reads NaN as no segment)


def lane_hinge(xyp, P, dz):
    """Plan poses (n, 8, 3) with grad, target polylines (n, M, 2) NaN padded, dead zone -> (n,) weighted hinge (m), on_route's geometry."""
    import torch
    import on_route as OR_
    nr = OR_.near(xyp, torch.arange(len(P), device=P.device), dict(poly=P, seglen=seglen(P)))
    h = torch.relu(nr - dz) * (nr < 1e3)
    w = torch.as_tensor(LANE_W, device=h.device, dtype=h.dtype)
    return (h * w).sum(1) / w.sum()


def seg_heading(P, g0, a):
    """Heading (n,) of polylines P (n, M, 2) (2 m grid from arc g0) at arc a (n,): linear between the headings of the segments whose midpoints
    bracket a (one segment where the other is missing), and its validity."""
    import torch
    M = P.shape[1]
    d = P[:, 1:] - P[:, :-1]
    ok = torch.isfinite(d).all(-1) & (torch.linalg.norm(torch.nan_to_num(d), dim=-1) > 1e-3)
    h = torch.atan2(d[..., 1], d[..., 0]).nan_to_num()
    u = ((a - g0 - 1.0) / 2.0).clamp(0, M - 2)
    j = u.floor().long().clamp(0, M - 3)
    w = (u - j).clamp(0, 1)
    ar = torch.arange(len(P), device=P.device)
    h0, h1, o0, o1 = h[ar, j], h[ar, j + 1], ok[ar, j], ok[ar, j + 1]
    dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
    hh = torch.where(o0 & o1, h0 + w * dh, torch.where(o0, h0, h1))
    return hh, o0 | o1


def over_turn(xyp, P, g0, deg):
    """The over-turn guard (n,) radians: relu(side x (psi_2s - h_route(a_2s)) - deg); a_2s = the plan's arc to its 2 s pose (detached)."""
    import torch
    q = torch.cat([torch.zeros_like(xyp[:, :1, :2]), xyp[:, :4, :2]], 1).detach()
    a2 = torch.linalg.norm(q[:, 1:] - q[:, :-1], dim=-1).sum(1)
    h2, ok2 = seg_heading(P, g0, a2)
    h0, ok0 = seg_heading(P, g0, torch.zeros_like(a2))
    h0 = torch.where(ok0, h0, torch.zeros_like(h0))                   # the near gap: the ego's own heading
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    turn = wrap(h2 - h0)
    side = torch.sign(turn) * (turn.abs() >= np.radians(SIDE_DEG))
    return torch.relu(side * wrap(xyp[:, 3, 2] - h2) - np.radians(deg)) * ok2


def fake_shift(f, s):
    """Old-layout features (n, DIM) -> the same route shifted sideways by s (n,) m (left +), headings kept."""
    import torch
    P = f.float().view(-1, RM.N_PT, RM.NF).clone()
    v = P[..., 0] > 0.5
    x, y = P[..., 1] * 20.0 - s[:, None] * P[..., 4], P[..., 2] * 20.0 + s[:, None] * P[..., 3]
    P[..., 1] = torch.where(v, x / 20.0, torch.zeros_like(x))
    P[..., 2] = torch.where(v, (y / 20.0).clamp(-4, 4), torch.zeros_like(y))
    return P.view(-1, RM.DIM)


def fake_straight(f):
    """A point block (n, DIM) -> a straight route: from the +0 m slot along its heading when that slot is valid, else (the near gap) from the
    ego along its own heading; slots at offsets >= 0 keep their validity, move onto the line and take its heading; slots behind unchanged."""
    import torch
    P = f.float().view(-1, RM.N_PT, RM.NF).clone()
    v = P[..., 0] > 0.5
    o = torch.as_tensor(RM.OFFS, device=f.device, dtype=torch.float32)[None, RM.I0:]
    g = v[:, RM.I0:RM.I0 + 1]
    one, zero = torch.ones_like(P[:, RM.I0, 3:4]), torch.zeros_like(P[:, RM.I0, 3:4])
    c, sn = torch.where(g, P[:, RM.I0, 3:4], one), torch.where(g, P[:, RM.I0, 4:5], zero)
    x, y = torch.where(g, P[:, RM.I0, 1:2] * 20.0, zero) + o * c, torch.where(g, P[:, RM.I0, 2:3] * 20.0, zero) + o * sn
    vv = v[:, RM.I0:]
    P[:, RM.I0:, 1] = torch.where(vv, x / 20.0, torch.zeros_like(x))
    P[:, RM.I0:, 2] = torch.where(vv, (y / 20.0).clamp(-4, 4), torch.zeros_like(y))
    P[:, RM.I0:, 3] = torch.where(vv, c.expand_as(x), torch.zeros_like(x))
    P[:, RM.I0:, 4] = torch.where(vv, sn.expand_as(x), torch.zeros_like(x))
    return P.view(-1, RM.DIM)


def lane_prep(a, ns, dev):
    """-> namespace of the route v2 tables: straight-negative rows (S.n,) bool, the lane-change pairs (store row, stay-lane input, command,
    label polyline)."""
    import torch
    from types import SimpleNamespace
    from jevdrive.common import data_dir
    S, n = ns.S, ns.S.n
    fo = ns.f_old.float().view(n, RM.N_PT, RM.NF).cpu().numpy()
    v = fo[..., 0] > 0.5
    h = np.arctan2(fo[..., 4], fo[..., 3])
    ref = np.where(v[:, RM.I0], h[:, RM.I0], 0.0)[:, None]                 # the near gap: the ego's own heading
    dh = np.abs((h - ref + np.pi) % (2 * np.pi) - np.pi)
    sel = v & (np.arange(RM.N_PT) >= RM.I0) & (RM.OFFS <= 40.0)
    curved = sel.any(1) & (np.where(sel, dh, 0).max(1) >= np.radians(CURVE_DEG))
    rows_dir = data_dir() / "runs/route_follow/cf/rows"
    nav = [d for d in ns.cfg.data if d.startswith("navtrain_full.")]
    offs, o = {}, 0
    for d in ns.cfg.data:
        m = len(np.load(data_dir() / "runs/op_parity/cache" / d / "tab.npz")["names"])
        offs[d], o = o, o + m
    assert o == n, (o, n)
    junction = np.zeros(n, bool)
    is_nav = np.zeros(n, bool)
    is_nav[ns.tr_o] = True                                            # the normal (NAVSIM) training rows
    LCr, LCf, LCc, LCp = [], [], [], []
    tr_set = set(ns.tr_o.tolist())
    order = ns.head.get("order", False)
    for d in nav:
        z = np.load(rows_dir / f"{d}.npz")
        if order and getattr(a, "lc_cf", False):                      # the ordered stay-lane memory (rf_cfpool build --order)
            zo = np.load(rows_dir / f"{d}.o.npz")
            fo_map = {(t, c): f for t, c, f in zip(zo["token"].tolist(), zo["cls"].tolist(), zo["f"])}
        r = z["row"].astype(np.int64) + offs[d]
        assert (S.tab["names"][r] == z["token"]).all(), f"{d}: cf rows vs Store"
        junction[r[np.isin(z["cls"], ["j0", "j1"])]] = True
        if not getattr(a, "lc_cf", False):                            # the lane-change tables only with --lc-cf (each flag stands alone)
            continue
        m = (z["cls"] == "lc") & z["ok"] & np.isfinite(z["poly"]).all(-1).any(-1)
        m &= np.array([x in tr_set for x in r.tolist()])
        if not m.any():
            continue
        f = np.stack([fo_map[t, "lc"] for t in z["token"][m].tolist()]).astype(np.float32) if order else z["f"][m].astype(np.float32)
        LCr.append(r[m])
        LCf.append(f)
        oh = np.zeros((m.sum(), 3), np.float32)
        c = z["cmd"][m].astype(int)
        oh[np.flatnonzero(c < 3), c[c < 3]] = 1.0                      # parity_adapter.ego_features: cmd[..., :3], unknown -> all zero
        LCc.append(oh)
        LCp.append(z["poly"][m].astype(np.float32))
    straight = curved & ~junction & is_nav
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    L = SimpleNamespace(straight=t(straight), curved=int(curved.sum()), straight_rows=int(straight.sum()), junction_rows=int(junction.sum()), n_lc=0)
    if getattr(a, "lc_cf", False):
        assert LCr, "--lc-cf: no lane-change rows in runs/route_follow/cf/rows"
        L.lc_row, L.lc_f, L.lc_cmd, L.lc_poly = t(np.concatenate(LCr)), t(np.concatenate(LCf)), t(np.concatenate(LCc)), t(np.concatenate(LCp))
        L.n_lc = int(len(L.lc_row))
    return L


def cap_share(w, g, cap):
    """Weights w (n,) of rows in groups g (n,) -> weights with no group above a share `cap` of sum(w), sum kept (--on-cap-scene)."""
    w = np.asarray(w, np.float64)
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    p = np.bincount(inv, w, len(u)) / w.sum()
    assert len(u) * cap >= 1 - 1e-12, f"--on-cap-scene {cap}: infeasible with {len(u)} scenes (needs >= {int(np.ceil(1 / cap))})"
    q, fixed = p.copy(), np.zeros(len(u), bool)
    while ((q > cap + 1e-12) & ~fixed).any():
        fixed |= q > cap + 1e-12
        q[fixed] = cap
        q[~fixed] = p[~fixed] * (1 - cap * fixed.sum()) / p[~fixed].sum()
    return w * (q / p)[inv]


def group_shares(w, g):
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    return {k: float(v) for k, v in zip(u, np.bincount(inv, w, len(u)) / np.sum(w))}


def draw_weights(tb, on_weight="", cap=0.0, cap_in=""):
    """Pool tab -> (per-row draw weights or None for the plain uniform draw, record): --on-weight product, then --on-cap-scene."""
    if not on_weight and not cap:
        return None, {}
    n = len(tb["names"])
    w = np.ones(n)
    for term in on_weight.split(",") if on_weight else []:
        kv, f = term.rsplit(":", 1)
        k, v = kv.split("=", 1)
        if k in tb.files:
            w[tb[k].astype(str) == v] *= float(f)
    rec = {}
    if cap:
        k, v = cap_in.split("=", 1) if cap_in else ("", "")
        sel = tb[k].astype(str) == v if k else np.ones(n, bool)
        assert sel.any() and "log" in tb.files, f"--on-cap-in {cap_in}: no pool row / no `log` tag"
        g = tb["log"].astype(str)[sel]
        before = group_shares(w[sel], g)
        m0 = w[sel].sum()
        w[sel] = cap_share(w[sel], g, cap)
        assert abs(w[sel].sum() - m0) <= 1e-9 * m0
        rec = dict(cap=cap, cap_in=cap_in or "all", cap_rows=int(sel.sum()), scenes=len(before), before=before, after=group_shares(w[sel], g))
    assert (w >= 0).all() and w.sum() > 0, "--on-weight / --on-cap-scene: weights"
    if "pool" in tb.files:
        rec["pool_share"] = group_shares(w, tb["pool"])
    return w, rec


def cmd_capcheck(a):
    """Known-answer test of --on-cap-scene (CPU): pool of 100 `s` rows (scenes a / b / c: 70 / 20 / 10 rows) + 30 `r` rows, cap 0.4 on pool=s.
    uniform: shares 0.4 / 0.4 / 0.2 exactly; weighted (--on-weight topo=x:3,pool=s:2.5, unequal weights inside a scene): the shares equal an
    independent solution (q = min(cap, lam p), lam by bisection); both: the subset's mass kept, `r` rows untouched, every row of a scene scaled
    by one factor; an infeasible cap refused."""
    class T(dict):
        files = property(lambda self: list(self))
    rng = np.random.default_rng(0)
    log = np.repeat(["a", "b", "c"], [70, 20, 10])
    tb = T(names=np.arange(130).astype(str), log=np.r_[log, np.full(30, "r")], pool=np.repeat(["s", "r"], [100, 30]), topo=rng.choice(["x", "y"], 130))
    res = {}
    for name, ow in (("uniform", ""), ("weighted", "topo=x:3,pool=s:2.5")):
        w0 = draw_weights(tb, ow)[0] if ow else np.ones(130)
        p0 = group_shares(w0[:100], log)
        p = np.array([p0[k] for k in "abc"])
        lo, hi = 0.0, 1.0 / p.min()
        for _ in range(200):
            lam = (lo + hi) / 2
            lo, hi = (lam, hi) if np.minimum(0.4, lam * p).sum() < 1 else (lo, lam)
        want = dict(zip("abc", (0.4, 0.4, 0.2) if name == "uniform" else np.minimum(0.4, lam * p)))
        w, rec = draw_weights(tb, ow, 0.4, "pool=s")
        f = w[:100] / w0[:100]
        res[name] = dict(before=p0, after=rec["after"], want={k: float(v) for k, v in want.items()},
                         max_err=max(max(abs(rec["after"][k] - want[k]) for k in "abc"), abs(w[:100].sum() / w0[:100].sum() - 1),
                                     float(np.abs(w[100:] - w0[100:]).max())),
                         within_scene_spread=max(float(np.ptp(f[log == k])) for k in "abc"))
    try:
        cap_share(np.ones(10), np.repeat(["a", "b"], 5), 0.4)
        res["infeasible_refused"] = False
    except AssertionError:
        res["infeasible_refused"] = True
    ok = res["infeasible_refused"] and all(res[n]["max_err"] <= 1e-6 and res[n]["within_scene_spread"] <= 1e-9 for n in ("uniform", "weighted"))
    print(json.dumps(dict(res, pass_=ok), indent=1))
    raise SystemExit(0 if ok else 2)


def cmd_draws(a):
    """Draw shares the trainer would use on --on with --on-weight / --on-cap-scene (CPU)."""
    from jevdrive.common import data_dir
    tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
    w, rec = draw_weights(tb, a.on_weight, a.on_cap_scene, a.on_cap_in)
    print(json.dumps(dict(pool=a.on, rows=len(tb["names"]), on_weight=a.on_weight, **rec), indent=1))


def open_rows(a, ns, dev):
    """--ego-drop-open: per store row OPEN.open (bool, unlabelled rows False) and OPEN.cls (0 offline normal, 1 yaw-rate, 2 pool) on dev, and the
    store-level record (rows / labelled / open per class)."""
    import torch
    from types import SimpleNamespace
    from jevdrive.common import data_dir
    import on_lon
    S, on0, n = ns.S, ns.on0, ns.S.n
    names = S.tab["names"]
    cls = np.zeros(n, np.int8)
    cls[np.r_[ns.tr_t, ns.dv_t].astype(int)] = 1
    cls[on0:] = 2
    lab, opn = np.zeros(n, bool), np.zeros(n, bool)
    C = np.load(a.open_cells if a.open_cells.startswith("/") else data_dir() / a.open_cells)
    cn, cf = C["names"].astype(str), C["free"].astype(bool)
    assert len(set(cn.tolist())) == len(cn), "cells: tokens not unique"
    o = np.argsort(cn)
    q = np.searchsorted(cn[o], names[:on0].astype(str)).clip(0, len(cn) - 1)
    hit = cn[o][q] == names[:on0].astype(str)
    lab[:on0], opn[:on0] = hit, hit & cf[o][q]
    rec = dict(cells=str(a.open_cells), pack="")
    if on0 < n:
        from jevdrive.common import data_dir as dd_
        pdir = dd_() / "runs/op_parity/cache" / a.on
        tb = np.load(pdir / "tab.npz")
        pk = _pl.Path(a.open_pack) if a.open_pack else on_lon.pack_path(pdir)
        pk = pk if pk.is_absolute() else dd_() / pk
        z = np.load(pk)
        strip = lambda v: v[3:] if v.startswith("on:") else v  # noqa: E731
        pos = {strip(x): i for i, x in enumerate(z["names"].astype(str).tolist())}
        pn = tb["names"].astype(str).tolist()
        if "pool" in tb.files:                                        # merged pool: on:<prefix>:<name>
            pn = [x[len(f"on:{p}:"):] if x.startswith(f"on:{p}:") else strip(x) for x, p in zip(pn, tb["pool"].astype(str).tolist())]
        else:
            pn = [strip(x) for x in pn]
        j = np.array([pos.get(x, -1) for x in pn])
        po = z["p_open_design"].astype(bool)
        lab[on0:], opn[on0:] = j >= 0, (j >= 0) & po[np.maximum(j, 0)]
        rec["pack"] = str(pk)
    for c, k in enumerate(("off", "yr", "pool")):
        m = cls == c
        rec[k] = dict(rows=int(m.sum()), labelled=int((lab & m).sum()), open=int((opn & m).sum()), open_share=float(opn[m].mean()) if m.any() else None)
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return SimpleNamespace(open=t(opn), cls=t(cls.astype(np.int64))), rec


def drop_rec(acc, p):
    """Accumulated --ego-drop-open counts -> realised dropped share vs the expected p x open share, per class open shares."""
    v = acc.cpu().numpy()
    na, dr = v[0], v[1]
    op = v[3] + v[5] + v[7]
    out = dict(rows=int(na), dropped=int(dr), dropped_share=float(dr / max(na, 1)), open_share=float(op / max(na, 1)), expected=float(p * op / max(na, 1)))
    out["ratio_to_expected"] = float(out["dropped_share"] / out["expected"]) if out["expected"] > 0 else None
    for c, k in enumerate(("off", "yr", "pool")):
        out[f"open_share_{k}"] = float(v[3 + 2 * c] / v[2 + 2 * c]) if v[2 + 2 * c] > 0 else None
        out[f"rows_{k}"] = int(v[2 + 2 * c])
    return out


def cmd_train(a):
    import torch
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d / 'ckpt-final.pt'} exists: a tag is never overwritten"
    a.rm_from = ""
    assert not a.init or a.init.startswith(("RB-F-", "RB0-F-")) or a.allow_init, f"--init {a.init}: tonight only RB-F / RB0-F (route-night.md 3; --allow-init overrides)"
    ns, rp, JD, TU, okc = build(a, dev)
    cfg, S, model, base, new, pools, kk = ns.cfg, ns.S, ns.model, ns.base, ns.new, ns.pools, ns.kk
    nB, nN, k_ho, k_on, tr_o, dv_o = cfg.batch, ns.nN, ns.k_ho, ns.k_on, ns.tr_o, ns.dv_o
    k_yr, tr_t, dv_t = ns.k_yr, ns.tr_t, ns.dv_t
    LS = ns.losses(ns.mode)
    pp_lr = not (k_yr or k_ho or k_on or a.agent_lam > 0)
    lane_on, lc_on = a.lane_lam > 0, a.lc_cf                          # route v2 (docstring); both off: no operation below is added
    if lane_on or lc_on:
        LN = lane_prep(a, ns, dev)
        k_lc = a.lc_k if lc_on else 0
        assert k_lc < nN, "--lc-k: fewer than the normal rows"
        over_lam = a.lane_lam if a.lane_over_lam is None else a.lane_over_lam
        lc_lam = a.lc_lam if a.lc_lam is not None else (a.lane_lam if lane_on else 10.0)
        nrng = np.random.default_rng([cfg.seed, 0, 97])              # negatives
        lrng = np.random.default_rng([cfg.seed, 0, 101])             # lane-change pairs
        pi_t = torch.as_tensor(S.pi, device=dev)
        ap_lc = [p_ for p_ in new + rp if p_.requires_grad]          # --lc-adapter-only: the adapter (incl. the route branch)
    rng = np.random.default_rng([cfg.seed, 0])
    hrng = np.random.default_rng([cfg.seed, 0, 43])
    orng = np.random.default_rng([cfg.seed, 0, 53])
    mrng = np.random.default_rng([cfg.seed, 0, 89])                  # route-memory drop
    erng = np.random.default_rng([cfg.seed, 0, 71])                  # --ego-drop-speed's rows
    esp = torch.as_tensor(EGO_SPEED, device=dev)
    sha = hashlib.sha1()
    lb, ln = (cfg.lr, cfg.lr_new) if a.lr_scale == 1.0 else (cfg.lr * a.lr_scale, cfg.lr_new * a.lr_scale)
    groups = [{"params": base, "lr": lb, "base": lb, "wu": cfg.warmup}, {"params": new, "lr": ln, "base": ln, "wu": cfg.warmup},
              {"params": rp, "lr": a.rm_lr, "base": a.rm_lr, "wu": a.rm_warmup}]
    rxp = ns.rxp
    if ns.rx:                                                         # STACK reflex module: its own group
        assert not (lane_on or lc_on), "--rx with the route v2 lane terms: the fakes write the route block only"
        groups.append({"params": rxp, "lr": a.rx_lr, "base": a.rx_lr, "wu": a.rx_warmup})
        xrng = np.random.default_rng([cfg.seed, 0, 113])
        import reflex as RX
        nr_ = RM.EGO_DIM + model.adapter.route.dim
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    ST = None
    if a.dual or a.cons or a.imit_drop or a.pref > 0 or ns.rx:      # STACK terms (lib/stack_train.py)
        import stack_train
        ST = stack_train.Stack(a, ns, LS, dev)
    scaler = torch.amp.GradScaler()

    pw, caprec = None, {}
    if a.on_weight or a.on_cap_scene:                                 # --on-weight / --on-cap-scene: per-row draw probabilities of the pool rows
        from jevdrive.common import data_dir
        tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
        assert len(tb["names"]) == ns.n_on, "--on-weight: pool rows"
        w, caprec = draw_weights(tb, a.on_weight, a.on_cap_scene, a.on_cap_in)
        pw = w / w.sum()
    PRB = {}                                                          # JT: wait / launch probe rows (from the wfile's groups)
    if a.on_wfile:                                                    # NR1: per-row draw weights from a file (docstring)
        from jevdrive.common import data_dir
        assert k_on and not (a.on_weight or a.on_cap_scene or a.si_scan), "--on-wfile: needs --on; not with --on-weight / --on-cap-scene / --si-scan"
        tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
        wz = np.load(a.on_wfile if a.on_wfile.startswith("/") else data_dir() / a.on_wfile)
        assert np.array_equal(wz["names"].astype(str), tb["names"].astype(str)), "--on-wfile: rows != the pool's"
        w = wz["w"].astype(np.float64)
        assert (w >= 0).all() and w.sum() > 0, "--on-wfile: weights"
        pw = w / w.sum()
        caprec = dict(on_wfile=str(a.on_wfile), rows_w0=int((w == 0).sum()))
        if "grp" in wz.files:
            caprec["group_share"] = group_shares(pw, wz["grp"])
            if a.probe_n and {"stand_r", "launch"} <= set(wz["grp"].astype(str).tolist()):   # JT: fixed wait / launch probe rows (monitoring)
                prng_ = np.random.default_rng([cfg.seed, 0, 173])
                PRB = {g: np.sort(prng_.choice(np.flatnonzero(wz["grp"].astype(str) == g), min(a.probe_n, int((wz["grp"].astype(str) == g).sum())),
                                               replace=False)) for g in ("stand_r", "launch")}
    upre = 0                                                          # weight-0 rows at the end, equal weights elsewhere: the plain uniform draw
    if pw is not None and (pw == 0).any():
        m0 = int((pw > 0).sum())
        if (pw[:m0] > 0).all() and (pw[:m0] == pw[0]).all():
            upre = m0
    OPEN, orec = open_rows(a, ns, dev) if a.ego_drop_open else (None, {})
    nw, si_free, sirec = None, None, {}
    if a.si_scan:                                                     # SELF-ITER (docstring); off: no operation below is added
        import self_iter as SI
        nw, pw_si, si_free, sirec = SI.draws(a, ns, S, tr_o)
        if pw_si is not None:
            pw, upre = pw_si, 0
    if a.on_wfile and "nav_w" in wz.files:                           # ROUND3c: normal-row draw weights from the same file (docstring)
        nwm_ = dict(zip(wz["nav_names"].astype(str).tolist(), wz["nav_w"].astype(np.float64).tolist()))
        wv_ = np.array([nwm_.get(x, np.nan) for x in S.tab["names"][tr_o].astype(str).tolist()])
        hit_ = np.isfinite(wv_)
        assert hit_.mean() >= 0.99 and (wv_[hit_] >= 0).all(), f"--on-wfile nav_w: {hit_.mean():.4f} of the normal train rows listed"
        wv_ = np.where(hit_, wv_, 1.0)
        nw = wv_ / wv_.sum()
        up_ = wv_ > 1
        caprec |= dict(nav_rows=int(len(tr_o)), nav_listed=int(hit_.sum()), nav_up_rows=int(up_.sum()), nav_up_w=sorted(set(wv_[up_].tolist())),
                       nav_up_share=float(nw[up_].sum()), nav_up_per_batch=float(nw[up_].sum() * nN))
    onx, hrec = None, {}
    if a.on_hold_res >= 0 or a.on_hold_pct:                          # NR1HVR: held-out pool logs (docstring); off: no operation is added
        from jevdrive.common import data_dir as _dd
        tbh_ = np.load(_dd() / "runs/op_parity/cache" / a.on / "tab.npz")
        lg_ = tbh_["log"].astype(str)
        assert len(lg_) == ns.n_on, "--on-hold-res: pool rows"
        el_ = tbh_["mode"].astype(str) == "self" if ns.mode == "mixed" and "mode" in tbh_.files else np.ones(len(lg_), bool)   # JT: the S rows
        ul_ = sorted(set(lg_[el_]))
        if a.on_hold_pct:                                             # JT: crc32(log) % 100 < P (research/onpol-scaleup.md section 4)
            import zlib
            held_l = [g for g in ul_ if zlib.crc32(g.encode()) % 100 < a.on_hold_pct]
        else:
            held_l = [g for g in ul_ if int(hashlib.sha1(g.encode()).hexdigest(), 16) % 10 == a.on_hold_res]
        held_m = np.isin(lg_, held_l) & el_
        hwb_ = float(pw[held_m].sum()) if pw is not None else None   # JT: the wfile already gives them weight 0 (then the draw is unchanged)
        assert held_m.any() and not held_m.all(), "--on-hold-res: the residue holds no log or every log"
        if pw is None:
            onx = np.flatnonzero(~held_m)
        else:
            pw = np.where(held_m, 0.0, pw)
            pw, upre = pw / pw.sum(), 0
        hg_ = np.random.default_rng([cfg.seed, 0, 167])
        HR = np.sort(hg_.choice(np.flatnonzero(held_m), min(a.on_hold_n, int(held_m.sum())), replace=False))
        FR = np.sort(hg_.choice(np.flatnonzero(~held_m & el_), min(a.on_hold_n, int((~held_m & el_).sum())), replace=False))
        hrec = dict(res=a.on_hold_res, pct=a.on_hold_pct, logs=len(ul_), held_logs=held_l, held_rows=int(held_m.sum()), drawn_rows=int((~held_m).sum()),
                    scored_held=len(HR), scored_fit=len(FR), eligible_rows=int(el_.sum()), held_weight_before=hwb_)

    TW = None
    if a.twin:                                                        # HARM-MIX (round7): twin (Harmonizer ON) tokens of pool rows at p; off: nothing added
        import twin_tokens
        from jevdrive.common import data_dir as _dt
        assert k_on, "--twin: needs --on"
        TW = twin_tokens.Twin(a.twin, _dt() / "runs/op_parity/cache", np.load(_dt() / "runs/op_parity/cache" / a.on / "tab.npz"), ns.on0, S.n, cfg.seed)
        TW.rec["draw_mass_with_twin"] = TW.mass(w=pw) if pw is not None else TW.mass(rows=onx)
        TW.rec["twin_rows_per_batch"] = TW.rec["draw_mass_with_twin"] * k_on
    lc_rows = LN.lc_row.cpu().numpy() if lc_on else None
    des_on, d2_on = bool(a.desire_rows), a.d2_lam > 0                # DESIRE D1 / D2 (docstring); both off: no operation below is added
    k_d1 = a.d1_k if des_on else 0
    if des_on or d2_on:
        import desire_train as DT
        from jevdrive.common import data_dir as _dd
        assert not (lc_on or ns.rx or ST is not None), "--desire-rows / --d2-lam: not with --lc-cf / --rx / STACK terms"
        n_base = sum(len(np.load(_dd() / "runs/op_parity/cache" / d / "tab.npz")["names"]) for d in a.data)
        drec = dict(n_base=n_base)
        pi_t = torch.as_tensor(S.pi, device=dev)
    if des_on:
        dz = np.load(a.desire_rows if a.desire_rows.startswith("/") else _dd() / a.desire_rows)
        side_np, d1_nav = DT.rows_side(S.tab["names"], n_base, ns.on0, dz)
        d1_nav = d1_nav[np.isin(d1_nav, tr_o)]                         # the K slots: train-split rows only
        assert k_d1 < nN and (k_d1 == 0 or len(d1_nav) >= k_d1), "--d1-k: slots / table rows"
        SIDE = torch.as_tensor(side_np, device=dev)
        arng = np.random.default_rng([cfg.seed, 0, 137])              # pulse ages of the fed rows
        drec |= dict(table=str(a.desire_rows), rule=str(dz["rule"]) if "rule" in dz.files else None, k=k_d1, nav_train_rows=int(len(d1_nav)),
                     store_rows=int((side_np != 0).sum()), pool_rows=int((side_np[ns.on0:] != 0).sum()), left=int((side_np > 0).sum()),
                     right=int((side_np < 0).sum()))
    if k_d1:
        d1rng = np.random.default_rng([cfg.seed, 0, 131])
    if d2_on:
        d2rng = np.random.default_rng([cfg.seed, 0, 127])
        TD = torch.as_tensor(DT.load_teacher([_pl.Path(a.d2_teacher) / f"{d}.npz" for d in a.data], S.tab["names"][:n_base]), device=dev)
        side_d2 = side_np[:n_base] if des_on else np.zeros(n_base, np.int8)
        drec |= dict(d2_lam=a.d2_lam, d2_frac=a.d2_frac, d2_teacher=str(a.d2_teacher))
    kd_on = a.kd_lam > 0                                              # RETAIN (docstring); off: no operation below is added
    if kd_on:
        import retain_train as RTN
        from jevdrive.common import data_dir as _dd
        assert not (lc_on or ns.rx or ST is not None or (ns.head.get("order") and not ns.head.get("anchor"))), \
            "--kd-lam: not with --lc-cf / --rx / STACK terms / --rm-order (--rm-anchor: the pool's .a features)"
        nb_ = sum(len(np.load(_dd() / "runs/op_parity/cache" / d / "tab.npz")["names"]) for d in a.data)
        RS = RTN.RetainSet(a.kd_pool if a.kd_k else "", dev, a.kd_groups, S, a.kd_nav, nb_, tr_o, layout="a" if ns.head.get("anchor") else "")
        assert not RS.n or RS.ego.shape[1] == S.ego.shape[1], f"--kd-pool {a.kd_pool}: fed row {RS.ego.shape[1]} != the store's {S.ego.shape[1]}"
        assert RS.n or len(RS.nav), "--kd-lam: no retention row (--kd-k 0 and no --kd-nav)"
        krng = np.random.default_rng([cfg.seed, 0, 151])
        kdrec = RS.rec | dict(kd_lam=a.kd_lam, mode=a.kd_mode, k=a.kd_k, k_nav=a.kd_nav_k if len(RS.nav) else 0)
    FLAT, flrec = None, {}
    flp_ = getattr(a, "flat_p", 1.0)
    if getattr(a, "flat_brk", "") and flp_ > 0:                       # SIMP-F (lon-anchor supplement 3): brake-needed navtrain rows, flat history
        import parity_adapter as PA_
        tk_ = set(_pl.Path(a.flat_brk).read_text().split())
        nm_ = np.asarray(ns.names).astype(str)
        fi_ = np.asarray(tr_o)[np.isin(nm_[np.asarray(tr_o)], list(tk_))]
        FLAT = torch.zeros(S.n, dtype=torch.bool, device=dev)
        FLAT[torch.as_tensor(fi_, device=dev)] = True
        flrec = dict(file=a.flat_brk, tokens=len(tk_), train_rows=int(len(fi_)), train_rows_total=int(len(tr_o)))
        if flp_ < 1:                                                  # ROUND2 --flat-p: per-draw probability, own stream
            frng_ = np.random.default_rng([cfg.seed, 0, 179])
            flrec |= dict(p=flp_, stream=[cfg.seed, 0, 179])
        assert len(fi_), f"--flat-brk {a.flat_brk}: no training row matched"
    if a.ego_scale_speed > 0:                                         # ANTI-COPY history-speed noise (docstring)
        srng = np.random.default_rng([cfg.seed, 0, 149])
        spow = torch.as_tensor(EGO_SCALE_POW, device=dev, dtype=torch.float32)
    awr_on = bool(a.awr_pack)                                         # AWR-ONPOL (docstring); off: no operation below is added
    if awr_on:
        import awr_train as AWR
        assert k_on and not (lc_on or ns.rx or ST is not None), "--awr-pack: needs --on; not with --lc-cf / --rx / STACK terms"
        AW = AWR.AwrSet(a, S, ns.on0, dev)
        wrng = np.random.default_rng([cfg.seed, 0, AWR.STREAM])

    assert not a.on_start or (k_on and not (lc_on or k_d1 or a.diag_infl)), "--on-start: needs --on; not with --lc-cf / --d1-k / --diag-infl"
    ndraw = [0]                                                       # --on-start: batches drawn so far (draw() runs once per step, in order)

    def draw():
        ko = k_on if ndraw[0] >= a.on_start else 0                    # POOL-PILOT: before step T the pool slots are normal rows, orng untouched
        ndraw[0] += 1
        nN = ns.nN + k_on - ko
        r = rng.choice(tr_o, nN, replace=len(tr_o) < nN) if nw is None else rng.choice(tr_o, nN, replace=True, p=nw)
        if k_yr:
            r = np.concatenate([r, rng.choice(tr_t, k_yr, replace=False)])
        an = rng.random(nB - k_ho - ko) < cfg.d_frac
        an[nN:] = False
        if ko:
            i = (orng.choice(ns.n_on, k_on, replace=ns.n_on < k_on) if pw is None and onx is None
                 else onx[orng.choice(len(onx), k_on, replace=len(onx) < k_on)] if pw is None
                 else orng.choice(upre, k_on, replace=upre < k_on) if upre else orng.choice(ns.n_on, k_on, replace=True, p=pw))
            sha.update(i.astype("<i8").tobytes())
            r, an = np.concatenate([r, ns.on0 + i]), np.concatenate([an, np.zeros(k_on, bool)])
        if k_ho:
            r = np.concatenate([r] + [hrng.choice(p, q, replace=False) for p, q in zip(pools, kk)])
            an = np.concatenate([an, np.zeros(k_ho, bool)])
        if lc_on:                                                     # route v2 pairs: real rows replace the last k normal rows, stay rows appended
            j = lrng.choice(LN.n_lc, k_lc, replace=False)
            lr = lc_rows[j]
            r = r.copy()
            r[nN - k_lc:nN], an[nN - k_lc:nN] = lr, False
            return np.concatenate([r, lr]), an, sha.hexdigest()[:16], j
        if k_d1:                                                      # D1: K table rows replace the last K normal rows
            r, an = r.copy(), an.copy()
            r[nN - k_d1:nN], an[nN - k_d1:nN] = d1_nav[d1rng.choice(len(d1_nav), k_d1, replace=False)], False
        return r, an, sha.hexdigest()[:16]

    def fetch(dr):
        return dr, S.front[dr[0]] if TW is None else TW.swap(dr[0], dr[2], S.front[dr[0]])
    S.front.pin = a.prefetch > 1
    Sd = T.dev_store(S, ns.dv_rows, a.dev_card_gb)

    def dev_ev():
        ev = T.dev_eval(model.eval(), Sd, dv_o, LS.W)
        if len(dv_t):
            ev |= {"ot_" + q: v for q, v in T.dev_eval(model, S, dv_t, LS.W).items()}
        has = S.has_fut[torch.as_tensor(dv_o, device=dev)].cpu().numpy()
        ev["ade_nomem"] = float(rows_ade(model, Sd, dv_o[has], LS.W, True).mean())
        model.train()
        return ev

    def pool_ev():
        """--on-hold-res: the pool terms on the held / fitted slices, eval mode (docstring)."""
        from loss43 import shape_only
        pi_ = torch.as_tensor(S.pi, device=dev)
        arcp = LS.pack.get("arc") if isinstance(LS.pack, dict) else None
        model.eval()
        res = {}
        with torch.no_grad():
            for nm, rr in (("held", HR), ("fit", FR)) if hrec else ():
                acc, n_, d4, dk = {}, 0, [], []
                for j in range(0, len(rr), 128):
                    g = ns.on0 + rr[j:j + 128]
                    rt = torch.as_tensor(g, device=dev)
                    o = model(S.front[g], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()
                    x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[rt], LS.W)
                    xs, ys = shape_only(x, y, psi) if LS.shape else (x, y)
                    tv = {}
                    if LS.agent2 is not None:
                        ok, av = LS.on_aok[rt], torch.zeros(len(g), device=dev)
                        if ok.any():
                            av[ok] = LS.agent2.per_step(xs[ok], ys[ok], psi[ok], rt[ok]).mean(1)
                        tv["agent"] = av
                    if LS.road is not None:
                        ko, rv = LS.road.ok[rt], torch.zeros(len(g), device=dev)
                        if ko.any():
                            rv[ko] = torch.relu(LS.road.margin - LS.road.margins(xs[ko], ys[ko], psi[ko], rt[ko])).mean(1)
                        if getattr(LS.road, "w", None) is not None:    # --on-road-pai-w, as LossesOn weighs the training term
                            rv = rv * LS.road.w[rt]
                        tv["road"] = rv
                    if LS.lon is not None:
                        tv |= LS.lon.loss(torch.stack([x, y, psi], -1), rt - LS.on0, LS.pack)
                    for q, v in tv.items():
                        acc[q + "_mean"] = acc.get(q + "_mean", 0.0) + float(v.sum())
                        acc[q + "_pos"] = acc.get(q + "_pos", 0.0) + float((v > 0).sum())
                    if arcp is not None:
                        import nr1_lon
                        lab = arcp["ok"][rt - LS.on0]
                        sp = nr1_lon.arc(x, y)[:, -1]
                        d4.append(torch.stack([sp - arcp["s_h"][rt - LS.on0][:, -1], arcp["s_h"][rt - LS.on0][:, -1]], 1)[lab])
                        if a.on_arc_rows != "move":                   # JTB / JTN: the bias per row kind (labelled or not)
                            dk.append(torch.stack([sp - arcp["s_h"][rt - LS.on0][:, -1], arcp["kind"][rt - LS.on0].float()], 1))
                    n_ += len(g)
                res |= {f"pool_{nm}/{q}": v / n_ for q, v in acc.items()}
                if d4 and nm in ("held", "fit"):
                    D_ = torch.cat(d4)
                    sh = D_[:, 1].clamp_min(1e-3)
                    res |= {f"pool_{nm}/arc4_bias": float(D_[:, 0].mean()), f"pool_{nm}/slow08": float(((D_[:, 0] + sh) < 0.8 * sh).float().mean()),
                            f"pool_{nm}/fast12": float(((D_[:, 0] + sh) > 1.2 * sh).float().mean())}
                if dk:
                    import nr1_lon
                    K_ = torch.cat(dk)
                    for ki, kn in enumerate(nr1_lon.KINDS):
                        m_ = K_[:, 1] == ki
                        res |= {f"pool_{nm}/n_{kn}": float(m_.sum())} | ({f"pool_{nm}/arc4_bias_{kn}": float(K_[m_, 0].mean())} if m_.any() else {})
            for nm, rr in PRB.items():                                # JT probe: plan 4 s arc on the trainval standing / launch rows
                import nr1_lon
                a4, h4, ad = [], [], []
                for j in range(0, len(rr), 128):
                    g = ns.on0 + rr[j:j + 128]
                    rt = torch.as_tensor(g, device=dev)
                    o = model(S.front[g], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()
                    x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[rt], LS.W)
                    f = S.fut[rt].float()
                    a4.append(nr1_lon.arc(x, y)[:, -1]), h4.append(nr1_lon.arc(f[..., 0], f[..., 1])[:, -1])
                    ad.append(torch.hypot(x - f[..., 0], y - f[..., 1]).mean(1))
                A4, H4, AD = torch.cat(a4), torch.cat(h4), torch.cat(ad)
                res |= {f"probe_{nm}/arc4": float(A4.mean()), f"probe_{nm}/stay1m": float((A4 < 1.0).float().mean()),
                        f"probe_{nm}/human_arc4": float(H4.mean()), f"probe_{nm}/ade": float(AD.mean())}
        model.train()
        return res
    if a.diag:
        tp_ = [p_ for p_ in base + new if p_.requires_grad]
        ACT = {"agent_ho_pos": "act/pool_agent", "road_ho_pos": "act/pool_road", "on/lat_onc_pos": "act/pool_lat_onc", "on/lat_soft_pos": "act/pool_lat_soft",
               "on/lat_side_pos": "act/pool_lat_side", "on/lon_arc_pos": "act/pool_lon_arc", "imit/hinge_pos": "act/nav_drivable",
               "imit/agent_pos": "act/nav_agent", "imit/route_pos": "act/nav_route"}

        def diag_grads(Ls):
            """Per-term trunk-gradient norms on the scaled terms (the optimizer's .grad untouched)."""
            hw, ll = LS.ho_w, getattr(LS, "lon_lam", 1.0)
            g0 = lambda q: Ls[q] if q in Ls and torch.is_tensor(Ls[q]) and Ls[q].requires_grad else None  # noqa: E731
            parts = dict(imit=[(1.0, g0("imit"))], nav_hinge=[(cfg.hinge_lam, g0("hinge"))],
                         pool_hinge=[(hw * cfg.agent_lam, g0("agent_ho")), (hw * LS.road_lam, g0("road_ho"))],
                         lat=[(hw * ll, g0(f"on/lat_{q}_ho")) for q in ("onc", "soft", "side")], lon=[(hw * ll, g0("on/lon_arc_ho"))])
            for q in ("free", "stand", "standr", "stoppt", "turncap", "anchor"):   # NR1STAND dose (supplement 17) / STAND-R / NR1STOPPT / TURN-CAP / LON-ANCHOR: only when on
                if g0(f"on/lon_{q}_ho") is not None:
                    parts[f"lon_{q}"] = [(hw * ll, g0(f"on/lon_{q}_ho"))]
            if g0("on/lon_tightin_ho") is not None:                    # TIGHT-IN (supplement 22): both parts as one term
                parts["lon_tightin"] = [(hw * ll, g0("on/lon_tightin_ho")), (hw * ll, g0("on/lon_tightina_ho"))]
            for q in ("simp_region", "simp_clear"):                     # SIMP (lon-anchor supplement 2): only when on
                if g0(f"on/{q}_ho") is not None:
                    parts[q] = [(hw * ll, g0(f"on/{q}_ho"))]
            sc, out, G = float(scaler.get_scale()), {}, {}
            for k, lst in parts.items():
                v = [w * t for w, t in lst if t is not None]
                if not v:
                    continue
                gr = torch.autograd.grad(scaler.scale(sum(v)), tp_, retain_graph=True, allow_unused=True)
                G[k] = [None if q is None else q.float() / sc for q in gr]
                out[f"grad/{k}"] = float(torch.sqrt(sum((q * q).sum() for q in G[k] if q is not None)))
            pk = [k for k in ("pool_hinge", "lat", "lon") if k in G]
            if pk:
                P = [sum(G[k][i] for k in pk if G[k][i] is not None) if any(G[k][i] is not None for k in pk) else None for i in range(len(tp_))]
                out["grad/pool"] = float(torch.sqrt(sum((q * q).sum() for q in P if q is not None)))
                if "imit" in G:
                    dot = sum((u * w).sum() for u, w in zip(G["imit"], P) if u is not None and w is not None)
                    out["grad/cos_imit_pool"] = float(dot / max(out["grad/imit"] * out["grad/pool"], 1e-30))
            return out

    IPR, grp_t = {}, None
    if a.diag_infl:                                                   # LAUNCH-DIAG probe sets (fixed rows; docstring)
        import nr1_lon
        n_ = a.diag_infl
        IPR = {f"{g}_tv": PRB[g][:n_] for g in ("launch", "stand_r") if g in PRB}
        arcp_ = LS.pack.get("arc") if isinstance(LS.pack, dict) else None
        if arcp_ is not None and hrec:
            kd_ = arcp_["kind"][torch.as_tensor(FR, device=arcp_["kind"].device)].cpu().numpy()
            for kn in ("launch", "still", "cruise"):
                IPR[f"{kn}_S"] = FR[kd_ == nr1_lon.KINDS.index(kn)][:n_]
        if a.on_wfile and "grp" in wz.files:
            gn_ = np.unique(wz["grp"].astype(str))
            grp_t = (torch.as_tensor(np.searchsorted(gn_, wz["grp"].astype(str)), device=dev), list(gn_))
        print(f"--diag-infl probe sets: { {q: len(v) for q, v in IPR.items()} }", flush=True)

    def diag_infl(Ls, out, rows, anchor, total, front, ego_c, step):
        """LAUNCH-DIAG (monitoring only, .grad untouched): for each probe set, d(mean plan 4 s arc) per 1 000 steps predicted from each loss
        term alone under the optimizer's current Adam preconditioner at the groups' base lr: -1000 sum_i lr_i g_f,i g_k,i / (sqrt(v_i) + eps)."""
        import nr1_lon
        hw, ll = LS.ho_w, getattr(LS, "lon_lam", 1.0)
        g0 = lambda q: Ls[q] if q in Ls and torch.is_tensor(Ls[q]) and Ls[q].requires_grad else None  # noqa: E731
        pos = torch.arange(len(rows), device=dev)
        imit_m = ~anchor & S.has_fut[rows]
        h_ = LS.ho[rows] if getattr(LS, "ho", None) is not None else torch.zeros_like(anchor)
        imit_m = imit_m & ~h_
        n_im = imit_m.sum().clamp_min(1).float()
        grpm = dict(nav=pos < nN, yr=(pos >= nN) & (pos < nN + k_yr), tv=(pos >= nN + k_yr) & (rows >= ns.on0) & ~is_pool[rows])
        if grp_t is not None:                                         # trainval rows by draw group
            gi = torch.full_like(rows, -1)
            pm = rows >= ns.on0
            gi[pm] = grp_t[0][rows[pm] - ns.on0]
            for j, gname in enumerate(grp_t[1]):
                grpm[f"tv_{gname}"] = grpm["tv"] & (gi == j)
        terms, chk = {}, 0.0
        for gname, gm in grpm.items():                                # the imitation term restricted to a row group, weighted as in `imit`
            m = gm & imit_m
            if not m.any():
                continue
            _, Lg = LS(out[m], S, rows[m], anchor[m])
            if torch.is_tensor(Lg.get("imit")) and Lg["imit"].requires_grad:
                terms[f"imit_{gname}"] = Lg["imit"] * m.sum().float() / n_im
                if gname in ("nav", "yr", "tv"):
                    chk = chk + float(terms[f"imit_{gname}"])
        res = {"infl/imit_split_resid": float(Ls["imit"]) - chk if torch.is_tensor(Ls.get("imit")) else float("nan")}
        if ego_c is not None:                                         # the same rows without the history-speed noise
            oc = model(front, ego_c, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows]).float()
            _, Lc = LS(oc, S, rows, anchor)
            if torch.is_tensor(Lc.get("imit")):
                terms["imit_clean"] = Lc["imit"]
        for nm_, (w_, q) in dict(imit=(1.0, "imit"), agent_S=(hw * cfg.agent_lam, "agent_ho"), road_S=(hw * LS.road_lam, "road_ho"),
                                 agent_tv=(hw * cfg.agent_lam, "on/agent_ho"), road_tv=(hw * LS.road_lam, "on/road_ho"),
                                 nav_hinge=(cfg.hinge_lam, "hinge"), lat_onc=(hw * ll, "on/lat_onc_ho"), lat_soft=(hw * ll, "on/lat_soft_ho"),
                                 lat_side=(hw * ll, "on/lat_side_ho")).items():
            if g0(q) is not None:
                terms[nm_] = w_ * g0(q)
        if total.requires_grad:
            terms["total"] = total
        sc = float(scaler.get_scale())
        G = {}
        for k, t_ in terms.items():
            gr = torch.autograd.grad(scaler.scale(t_), tp_, retain_graph=True, allow_unused=True)
            G[k] = [None if q is None else q.float() / sc for q in gr]
        pre = []                                                      # lr_i / (sqrt(v_hat_i) + eps) per trunk tensor
        lr_of = {id(p_): (g["base"] if not a.diag_freeze else g["nom"], g.get("betas", (0.9, 0.999))[1], g.get("eps", 1e-8))
                 for g in opt.param_groups for p_ in g["params"]}
        for p_ in tp_:
            st_ = opt.state.get(p_, {})
            lr_, b2, eps_ = lr_of[id(p_)]
            if "exp_avg_sq" not in st_:
                pre.append(None)
                continue
            t_ = float(st_["step"])
            vh = st_["exp_avg_sq"].float() / (1 - b2 ** t_)
            pre.append(lr_ / (vh.sqrt() + eps_))
        pi_ = torch.as_tensor(S.pi, device=dev)
        model.eval()
        for pn, rr in IPR.items():
            g = torch.as_tensor(ns.on0 + rr, device=dev)
            o = model(S.front[ns.on0 + rr], S.ego[g], S.tc[g], None, None, nv=None if ns.nv is None else ns.nv[g]).float()
            x, y, _ = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[g], LS.W)
            f_ = nr1_lon.arc(x, y)[:, -1].mean()
            gf = torch.autograd.grad(f_, tp_, allow_unused=True)
            res[f"infl/{pn}/arc4"] = float(f_)
            nf = float(torch.sqrt(sum((q.float() ** 2).sum() for q in gf if q is not None)))
            for k, Gk in G.items():
                dot = sum((u.float() * w).sum() for u, w in zip(gf, Gk) if u is not None and w is not None)
                nk = float(torch.sqrt(sum((w * w).sum() for w in Gk if w is not None)))
                res[f"infl/{pn}/cos_{k}"] = float(dot) / max(nf * nk, 1e-30)
                if any(q is not None for q in pre):
                    d = sum((u.float() * w * q).sum() for u, w, q in zip(gf, Gk, pre) if u is not None and w is not None and q is not None)
                    res[f"infl/{pn}/d1k_{k}"] = -1000.0 * float(d)
        model.train()
        return res
    with Run("route_mem", f"train-{a.tag}", seed=cfg.seed, config=asdict(cfg) | {"onpol": vars(a), "route_mem": dict(head=ns.head, feat_ok=okc)}) as run:
        for x in ns.sp + ((ns.trl,) if k_ho or ns.agent2 is not None else ()) + ((splits.load(a.on_split),) if a.on and a.on_split else ()):
            run.use_split(x)
        nroute = sum(p.numel() for p in rp)
        run.info(f"{a.tag}: on_train recipe + route branch ({nroute / 1e6:.2f} M, lr {a.rm_lr}, warm-up {a.rm_warmup}, drop {a.rm_drop}); "
                 f"train {len(tr_o)} normal + {len(tr_t)} yaw-rate rows, dev {len(dv_o)} + {len(dv_t)}; per batch {nN} normal, {k_yr} yaw-rate, "
                 f"{k_on} on-policy (mode {ns.mode or 'off'}; {json.dumps(ns.onrec)}); feature rows ok {json.dumps(okc)}")
        if pw is not None:
            run.info(f"--on-weight {a.on_weight}: pool row draw probability min {pw.min() * ns.n_on:.3f} / max {pw.max() * ns.n_on:.3f} x uniform"
                     + (f"; weight 0 on the last {ns.n_on - upre} rows: uniform draw over the first {upre}" if upre else ""))
        if caprec:
            run.info(f"draw shares: {json.dumps(caprec)}")
            run.summary.update(draw_shares=caprec)
        if a.ego_drop_speed > 0:
            run.info(f"ego speed history (ego cols {EGO_SPEED}) zeroed on a share {a.ego_drop_speed} of the non-anchor rows, rng [seed, 0, 71]"
                     + (f"; OPEN rows only (--ego-drop-open): {json.dumps(orec)}" if a.ego_drop_open else ""))
            if a.ego_drop_open:
                run.summary.update(ego_drop_open=orec)
                dacc = torch.zeros(8, dtype=torch.float64, device=dev)    # non-anchor rows / dropped / per class (off, yr, pool): rows, open
        run.info(f"init {json.dumps(ns.init) if ns.init else 'none: shipped weights + fresh adapter (on_train / joint_train start)'}")
        if lane_on or lc_on:
            run.info(f"route v2: layout {'ordered' if ns.head.get('order') else 'old'}; lane {a.lane_lam} dz {a.lane_dz} neg {a.lane_neg} shift "
                     f"{a.lane_shift} over {over_lam} / {a.lane_over_deg} deg; straight-negative rows {LN.straight_rows} (curved {LN.curved}, junction "
                     f"tokens excluded {LN.junction_rows}); lc pairs {('k ' + str(k_lc) + ' of ' + str(LN.n_lc) + ', lam ' + str(lc_lam)) if lc_on else 'off'}")
        elif ns.head.get("order"):
            run.info("route v2: layout ordered, no lane terms")
        if des_on or d2_on:
            run.info(f"DESIRE D1 / D2: {json.dumps(drec)}")
            run.summary.update(desire=drec)
        if kd_on:
            run.info(f"RETAIN: P0 speed-profile distillation, {json.dumps(kdrec)}")
            run.summary.update(retain=kdrec)
        if FLAT is not None:
            run.info(f"SIMP-F flat history (ax 0, constant-speed past poses: parity_adapter.cv_history) on the brake-needed navtrain rows: {json.dumps(flrec)}")
            run.summary.update(flat_brk=flrec)
        if a.ego_scale_speed > 0:
            screc = dict(share=a.ego_scale_speed, max=a.ego_scale_max, cols=EGO_SPEED, pow=EGO_SCALE_POW, rows="non-anchor, non-pool")
            run.info(f"ANTI-COPY history-speed noise: {json.dumps(screc)}, rng [seed, 0, 149]")
            run.summary.update(ego_scale=screc)
        if awr_on:
            run.info(f"AWR-ONPOL: {json.dumps(AW.rec)}, rng [seed, 0, {AWR.STREAM}]")
            run.summary.update(awr=AW.rec)
        if hrec:
            run.info(f"NR1HVR held-out pool logs (never drawn): {json.dumps(hrec)}")
            run.summary.update(on_hold=hrec)
            pe = pool_ev()
            run.scalars(pe, 0)
            run.info("pool slices @ 0: " + ", ".join(f"{q} {v:.4f}" for q, v in pe.items()))
        if a.lr_floor or a.diag:
            run.info(f"lr floor {a.lr_floor}; diag {a.diag} (per-term gradients every {a.diag_every} steps)")
        if TW is not None:
            run.info(f"--twin (round7 HARM-MIX): {json.dumps(TW.rec)}")
        if ns.cmd_rec:
            run.info(f"--cmd-rule (round7 joint arm): {json.dumps(ns.cmd_rec)}")
            run.summary.update(cmd_rule=ns.cmd_rec)
        ev = dev_ev()
        run.scalars({f"dev/{q}": v for q, v in ev.items()}, 0)
        run.info("dev @ 0: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items())
                 + (f"; init's recorded dev ade {ns.init['init_dev_ade']:.4f}" if ns.init and ns.init.get("init_dev_ade") is not None else ""))
        run.summary.update({f"dev0_{q}": v for q, v in ev.items()} | {"init": ns.init})
        is_pool = torch.zeros(S.n, dtype=torch.bool, device=dev)
        is_pool[ns.on0:] = bool(k_on) and not a.on_normal             # --rm-drop never applies to pool rows (NR1 --on-normal: imitation rows)
        if k_on and a.on_normal and ns.mode == "mixed":               # JT: in a mixed pool only the non-`self` rows are imitation rows
            from jevdrive.common import data_dir as _dd
            md_ = np.load(_dd() / "runs/op_parity/cache" / a.on / "tab.npz")["mode"].astype(str)
            is_pool[ns.on0:] = torch.as_tensor(md_ == "self", device=dev)
            run.info(f"--on-normal, mixed pool: {int((md_ != 'self').sum())} imitation rows (memory drop, history noise), {int((md_ == 'self').sum())} S rows")
            if a.si_split:
                from jevdrive.data import splits as _sp
                run.use_split(_sp.load(a.si_split))
        if a.si_scan:                                                 # SELF-ITER: the new trainval rows are imitation rows (memory dropped as normal rows)
            is_pool[torch.as_tensor(si_free, device=dev)] = False
            run.info(f"SELF-ITER draw: {json.dumps(sirec)}")
            run.summary.update(self_iter=sirec)
            if len(si_free):
                from jevdrive.data import splits as _sp
                run.use_split(_sp.load(a.si_split))
        if a.compile and a.diag:                                      # --diag's retain_graph backward: compiled backwards must not donate buffers
            import torch._functorch.config as _fc
            _fc.donated_buffer = False
        fwd = torch.compile(model, dynamic=False) if a.compile else model    # the main training forward only; everything else eager
        pre = T.Prefetch(draw, fetch, cfg.steps, a.prefetch, a.fetch_workers)
        t0, hist, hsh = time.time(), [], ""
        for step in range(cfg.steps):
            dr, front = pre.get()
            r, an, hsh = dr[:3]
            rows, anchor = torch.as_tensor(r, device=dev), torch.as_tensor(an, device=dev)
            if lc_on:                                                 # the stay rows at the end: never anchor rows
                n_main = len(an)
                jl = torch.as_tensor(dr[3], device=dev)
                anchor = torch.cat([anchor, torch.zeros(k_lc, dtype=torch.bool, device=dev)])
            cosf = 0.5 * (1 + np.cos(np.pi * (step / cfg.steps))) if pp_lr else 0.5 * (1 + np.cos(np.pi * step / cfg.steps))
            if a.lr_floor:
                cosf = a.lr_floor + (1.0 - a.lr_floor) * cosf
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / g["wu"]) * cosf
                if a.diag_freeze:                                     # LAUNCH-DIAG: the weights stay at the init (Adam state still accumulates)
                    g["nom"], g["lr"] = g["base"], 0.0
            ego = S.ego[rows] * (~anchor)[:, None].float()
            if lc_on:
                keep = torch.as_tensor(mrng.random(n_main) >= a.rm_drop, device=dev) | is_pool[rows[:n_main]]
                keep[nN - k_lc:nN] = True                             # the pair's real rows keep their memory
                keep = torch.cat([keep, torch.ones(k_lc, dtype=torch.bool, device=dev)])
                ego[n_main:, 1:4] = LN.lc_cmd[jl]
                ego[n_main:, RM.EGO_DIM:] = LN.lc_f[jl]
            else:
                keep = torch.as_tensor(mrng.random(len(r)) >= a.rm_drop, device=dev) | is_pool[rows]
            ego[:, RM.EGO_DIM:] = ego[:, RM.EGO_DIM:] * keep[:, None].float()
            if ns.rx:                                                 # self-state dropped on a share of the pool rows
                xd = torch.as_tensor(xrng.random(len(r)) < a.rx_drop, device=dev) & is_pool[rows]
                ego[:, nr_:] = ego[:, nr_:] * (~xd)[:, None].float()
            n_neg = 0
            if lane_on and a.lane_neg > 0:                            # negatives: a fake input memory, the original route stays the target
                nm = n_main if lc_on else len(r)
                rm_ = rows[:nm]
                fo_ = ns.f_old[rm_]
                elig = (~anchor[:nm]) & keep[:nm] & (fo_.view(nm, RM.N_PT, RM.NF)[:, RM.I0:, 0] > 0.5).any(1)   # a valid slot ahead
                if lc_on:
                    elig[nN - k_lc:nN] = False
                u = torch.as_tensor(nrng.random((nm, 3)), device=dev, dtype=torch.float32)
                neg = elig & (u[:, 0] < a.lane_neg)
                n_neg = int(neg.sum())
                if n_neg:
                    st = neg & LN.straight[rm_] & (u[:, 1] < 0.5)
                    sh = neg & ~st
                    fk = fo_.float().clone()
                    if sh.any():
                        fk[sh] = fake_shift(fo_[sh], torch.where(u[sh, 2] < 0.5, -a.lane_shift, a.lane_shift))
                    if st.any():
                        fk[st] = fake_straight(fo_[st])
                    fin = RM.to_order_t(fk[neg], ns.trav[rm_[neg]]) if ns.head.get("order") else fk[neg]
                    idx = torch.nonzero(neg)[:, 0]
                    ego[idx, RM.EGO_DIM:] = fin.to(ego.dtype)
            if a.ego_drop_speed > 0:
                dm = torch.as_tensor(erng.random(len(r)) < a.ego_drop_speed, device=dev)
                if a.ego_drop_open:                                   # the same draw, masked to the open rows
                    op, cl = OPEN.open[rows], OPEN.cls[rows]
                    dm = dm & op
                    na = ~anchor[:len(r)]
                    bs_ = torch.stack([na.sum(), (dm & na).sum()] + [x for c in range(3) for x in ((na & (cl == c)).sum(), (na & op & (cl == c)).sum())]).double()
                    dacc += bs_
                ego[:, esp] = ego[:, esp] * (~dm)[:, None].float()
            dstep = a.diag and a.diag_infl and (step + 1) % a.diag_every == 0
            ego_c = ego.clone() if dstep and a.ego_scale_speed > 0 else None   # LAUNCH-DIAG: the rows before the history-speed noise
            if a.ego_scale_speed > 0:                                 # history-speed noise on the imitation rows (anchor and pool rows never)
                us = torch.as_tensor(srng.random((len(r), 2)), device=dev, dtype=torch.float32)
                sm = (us[:, 0] < a.ego_scale_speed) & ~anchor[:len(r)] & ~is_pool[rows[:len(r)]]
                sc = torch.where(sm, 1.0 + a.ego_scale_max * (2.0 * us[:, 1] - 1.0), torch.ones_like(us[:, 1]))
                ego[:len(r), esp] = ego[:len(r), esp] * sc[:, None].pow(spow[None])
                Ls_s = dict(scale_share=sm.float().mean(), scale_abs=((sc - 1).abs().sum() / sm.sum().clamp_min(1)))
            if FLAT is not None:                                      # SIMP-F: no deceleration in the fed history where braking must start
                fm_ = FLAT[rows] & ~anchor
                if flp_ < 1:                                          # ROUND2 --flat-p: this draw is flattened with probability P
                    fm_ = fm_ & torch.as_tensor(frng_.random(len(rows)) < flp_, device=dev)
                nfl_ = fm_.sum()
                if nfl_ > 0:
                    e_ = ego[fm_]
                    e_[:, 6] = 0.0
                    e_[:, 8:20] = PA_.cv_history(e_[:, :20])
                    ego[fm_] = e_
            dq = None
            if des_on:                                                # D1: the table rows' desire (anchor rows never)
                sd_ = SIDE[rows].long() * (~anchor).long()
                ag_ = torch.as_tensor(arng.integers(0, len(DT.AGES), len(r)), device=dev)
                if (sd_ != 0).any():
                    dq = DT.queue(sd_, ag_, dev)
            out = fwd(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows], desire=dq)
            if lc_on:
                total, Ls = LS(out[:n_main], S, rows[:n_main], anchor[:n_main])
            else:
                total, Ls = LS(out, S, rows, anchor)
            if des_on:
                Ls["d1/rows"] = (sd_ != 0).sum().float()
            if FLAT is not None:
                Ls["flat/rows"] = nfl_.float()
            if a.ego_scale_speed > 0:
                Ls["scale/share"], Ls["scale/abs"] = Ls_s["scale_share"], Ls_s["scale_abs"]
            if kd_on:                                                 # RETAIN: P0's speed profile on the retention rows (second forward)
                lk, lg_ = RS.loss(model, RS.batch(krng, a.kd_k, a.kd_nav_k, S), LS, a.kd_mode)
                Ls["kd"] = lk.mean()
                Ls |= lg_
                total = total + a.kd_lam * Ls["kd"]
            if awr_on:                                                # AWR-ONPOL: the drawn rows' own plans, second forward
                wr_, wt_ = AW.batch(wrng, a.awr_k)
                lw, lgw = AW.loss(model, S, wr_, wt_, LS, ns.nv)
                Ls["awr"] = lw.sum() / nN
                Ls |= lgw
                total = total + a.awr_lam * Ls["awr"]
            if d2_on:                                                 # D2: desire increment of the plan against shipped Cinque's
                u2, s2, a2 = d2rng.random(nN), d2rng.choice(np.array([1, -1]), nN), d2rng.integers(0, len(DT.AGES), nN)
                an_ = an[:nN]
                ii = np.flatnonzero(an_ | ((u2 < a.d2_frac) & (side_d2[r[:nN]] == 0)))
                if len(ii):
                    it = torch.as_tensor(ii, device=dev)
                    ri = rows[it]
                    q2 = DT.queue(torch.as_tensor(s2[ii], device=dev), torch.as_tensor(a2[ii], device=dev), dev)
                    o2 = model(front[it.to(front.device)], ego[it], S.tc[ri], None, None, nv=None if ns.nv is None else ns.nv[ri], desire=q2)
                    xd, yd, _ = T.rear(o2.float()[:, pi_t].view(-1, 33, 15), S.cam_x[ri], LS.W)
                    x0, y0, _ = T.rear(out.float()[it][:, pi_t].view(-1, 33, 15), S.cam_x[ri], LS.W)
                    td = TD[ri, torch.as_tensor(DT.arm_of(s2[ii], a2[ii]), device=dev)]
                    l2 = DT.d2_loss(torch.stack([xd, yd], -1), torch.stack([x0, y0], -1), td)
                    Ls["d2"] = l2.mean()
                    total = total + a.d2_lam * Ls["d2"]
                    sg = torch.as_tensor(s2[ii], device=dev, dtype=torch.float32)
                    Ls["d2/n"], Ls["d2/anchor"] = torch.tensor(float(len(ii))), torch.tensor(float(an_[ii].sum()))
                    Ls["d2/resp4"] = (sg * (yd - y0)[:, 7]).mean().detach()
                    Ls["d2/resp4_p0"] = (sg * td[:, 7, 1]).mean()
            if lane_on or lc_on:                                      # route v2 terms on the plan in each row's own frame
                x, y, psi = T.rear(out.float()[:, pi_t].view(-1, 33, 15), S.cam_x[rows], LS.W)
                xyp = torch.stack([x, y, psi], -1)
                nm = n_main if lc_on else len(r)
                nA = (~anchor[:nm]).sum().clamp_min(1).float()
                if lane_on:
                    m_ = ((~anchor[:nm]) & keep[:nm]).float()
                    Pt = old_poly(ns.f_old[rows[:nm]])
                    hl = lane_hinge(xyp[:nm], Pt, a.lane_dz) * m_
                    ov = over_turn(xyp[:nm], Pt, OLD_GRID0, a.lane_over_deg) * m_
                    Ls["lane"], Ls["lane_over"] = hl.sum() / nA, ov.sum() / nA
                    Ls["lane_act"], Ls["lane_neg_n"] = (hl > 0).float().mean().detach(), torch.tensor(float(n_neg))
                    total = total + a.lane_lam * Ls["lane"] + over_lam * Ls["lane_over"]
                if lc_on:
                    Pl = LN.lc_poly[jl]
                    hs = lane_hinge(xyp[n_main:], Pl, a.lc_dz)
                    os_ = over_turn(xyp[n_main:], Pl, LC_GRID0, a.lane_over_deg)
                    Ls["lc"], Ls["lc_over"] = hs.sum() / nA, os_.sum() / nA
                    Ls["lc_swing2"] = (xyp[n_main:, 3, 1] - xyp[nN - k_lc:nN, 3, 1]).abs().mean().detach()
                    lc_term = lc_lam * Ls["lc"] + over_lam * Ls["lc_over"]
                    if a.lc_adapter_only:                             # the stay rows' gradient reaches the adapter + route branch only
                        g_lc = torch.autograd.grad(scaler.scale(lc_term), ap_lc, retain_graph=True, allow_unused=True)
                    else:
                        total = total + lc_term
            if a.ego_drop_speed > 0 and a.ego_drop_open:              # logging only
                Ls["drop/share"] = bs_[1] / bs_[0].clamp_min(1)
                for c, q in enumerate(("off", "yr", "pool")):
                    if bs_[2 + 2 * c] > 0:
                        Ls[f"drop/open_{q}"] = bs_[3 + 2 * c] / bs_[2 + 2 * c]
                Ls["drop/open_all"] = (bs_[3] + bs_[5] + bs_[7]) / bs_[0].clamp_min(1)
            if ST is not None:
                total, total_C, Ls = ST.terms(out, S, rows, anchor, front, total, Ls, n_main if lc_on else None)
                if ns.rx:
                    Ls["rx/gate"] = model.adapter.rx.gate.detach().float().abs().sum()
                    lr_ = model.adapter.last_res
                    if lr_ is not None:
                        Ls["rx/res_abs"] = lr_.detach().abs().mean()
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {q: float(v) for q, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            if a.diag and (step + 1) % a.diag_every == 0:            # monitoring only: before the step's backward, .grad untouched
                Ls |= diag_grads(Ls)
                if a.diag_infl:
                    Ls |= diag_infl(Ls, out, rows, anchor, total, front, ego_c, step)
            if ns.rx:                                                 # the module learns from the constraint side only
                g_rx = (torch.autograd.grad(scaler.scale(total_C), rxp, retain_graph=True, allow_unused=True) if total_C.requires_grad
                        else [None] * len(rxp))
            scaler.scale(total).backward()
            if ns.rx:
                for p_, g_ in zip(rxp, g_rx):
                    p_.grad = g_ if g_ is not None else torch.zeros_like(p_)
            if lc_on and a.lc_adapter_only:
                for p_, g_ in zip(ap_lc, g_lc):
                    if g_ is not None:
                        p_.grad = g_ if p_.grad is None else p_.grad + g_
            scaler.unscale_(opt)
            gn_ = torch.nn.utils.clip_grad_norm_(base + new + rp + rxp, 1.0)
            scaler.step(opt)
            scaler.update()
            if a.diag:                                                # monitoring only (values read after the step)
                if torch.isfinite(gn_):
                    Ls["grad/global"], Ls["grad/clip"] = gn_.detach(), (gn_ > 1.0).float()
                Ls |= {ACT[q]: Ls[q] for q in ACT if q in Ls}
            if ST is not None:
                ST.update(Ls, step)
            if a.on_start:                                            # POOL-PILOT: pool rows in this batch (0 before T, k_on after)
                Ls["on/batch_pool_rows"] = float((r >= ns.on0).sum())
            hist.append({q: float(v) for q, v in Ls.items()})
            if (step + 1) % 25 == 0 or step + 1 == cfg.steps:
                keys = sorted({q for h in hist for q in h})
                mm = {q: float(np.mean([h[q] for h in hist if q in h])) for q in keys}
                hist = []
                el = time.time() - t0
                rn = float(model.adapter.route.out.weight.norm())
                run.scalars({f"loss/{q}": v for q, v in mm.items()} | {"throughput/steps_per_s": (step + 1) / el, "route/out_norm": rn}, step + 1)
                if a.ego_drop_open and a.ego_drop_speed > 0 and ((step + 1) % 500 == 0 or step + 1 == cfg.steps):
                    dd = drop_rec(dacc, a.ego_drop_speed)
                    run.info(f"ego-drop-open after {step + 1} steps: {json.dumps(dd)}")
                    run.summary.update(ego_drop_open_realised=dd)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in mm.items() if "/" not in q) + f"; route out |W| {rn:.4f}"
                             + (f"; on_hash {hsh}" if k_on else "") + f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if k_on and (step + 1 == 60 or (step + 1) % 1000 == 0 or step + 1 == cfg.steps):
                run.summary.update({f"on_hash_{step + 1}": hsh})
            if getattr(a, "save_at", "") and step + 1 in {int(x) for x in a.save_at.split(",")} and step + 1 < cfg.steps:   # snapshots (no RNG, no update)
                stag = a.tag.replace("-L18-", f"x{step + 1}-L18-") if "-L18-" in a.tag else f"{a.tag}x{step + 1}"
                sd = T.proot("runs", stag)
                assert not (sd / "ckpt-final.pt").exists(), f"{sd} exists"
                sd.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state(), "cfg": asdict(cfg), "onpol": vars(a), "route_mem": ns.head, "snapshot_of": a.tag, "step": step + 1}, sd / "ckpt-final.pt")
                torch.save(model.adapter.state_dict(), sd / "adapter.pt")
                RM.write_sidecar(stag, ns.head, dict(init=a.init, rm_lr=a.rm_lr, rm_warmup=a.rm_warmup, rm_drop=a.rm_drop, rm_drop_pool=False,
                                                     on=a.on, on_lon=a.on_lon, steps=step + 1, snapshot_of=a.tag))
                run.info(f"snapshot @ {step + 1}: {sd / 'ckpt-final.pt'}")
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = dev_ev()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
                if hrec or PRB:
                    pe = pool_ev()
                    run.scalars(pe, step + 1)
                    run.info(f"pool slices @ {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in pe.items()))
                    run.summary.update({f"pool_{q.replace('/', '_')}": v for q, v in pe.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "onpol": vars(a), "route_mem": ns.head}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        v2 = dict(lane_lam=a.lane_lam, lane_dz=a.lane_dz, lane_neg=a.lane_neg, lc_cf=a.lc_cf) if (lane_on or lc_on or ns.head.get("order")) else {}
        rx_side = dict(reflex=dict(sf=__import__("reflex").SF, rx=None, drop=a.rx_drop, lr=a.rx_lr)) if ns.rx else {}
        RM.write_sidecar(a.tag, ns.head, dict(init=a.init, rm_lr=a.rm_lr, rm_warmup=a.rm_warmup, rm_drop=a.rm_drop, rm_drop_pool=False,
                                              on=a.on, on_lon=a.on_lon, steps=cfg.steps, dev_ade=ev.get("ade"), **v2, **rx_side))
        if ST is not None:
            run.summary.update(stack=ST.summary(), **({"rx_feats": ns.rx_rec} if ns.rx else {}))
            run.info(f"stack: {json.dumps(ST.summary())}")
        if TW is not None:
            run.summary.update(twin=TW.summary())
            run.info(f"--twin realised: {json.dumps(TW.summary())}")
        rr = rely(model, Sd, dv_o, LS.W, JD, TU, S.tab["log"])
        run.info(f"reliance: {json.dumps(rr)}")
        (_pl.Path(run.dir) / "rely.json").write_text(json.dumps(rr, indent=1))
        from jevdrive.common import data_dir
        out = data_dir() / "runs/route_mem/rely"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.tag}.json").write_text(json.dumps(rr, indent=1))
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"), on_hash=hsh, rely=rr, route_params=nroute,
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


def cmd_mixcheck(a):
    """Known-answer test of --on-mode mixed (docstring): same normal rows + k pool rows of one kind, mixed vs the pure mode, per batch."""
    import torch
    dev = torch.device("cuda")
    a.rm_from, a.on_mode = "", "mixed"
    ns, rp, JD, TU, okc = build(a, dev)
    S, model, k = ns.S, ns.model, ns.k_on
    from jevdrive.common import data_dir
    md = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")["mode"].astype(str)
    rng = np.random.default_rng([a.seed, 0, 97])
    res = {}
    for kind, ref in (("self", "self"), ("both", "both")):
        pool = np.flatnonzero(md == kind if kind == "self" else md != "self")
        worst = 0.0
        for b in range(3):
            r = np.concatenate([rng.choice(ns.tr_o, ns.nN, replace=False), ns.on0 + rng.choice(pool, k, replace=False)])
            rows, an = torch.as_tensor(r, device=dev), torch.zeros(len(r), dtype=torch.bool, device=dev)
            with torch.no_grad():
                out = model(S.front[rows], S.ego[rows], S.tc[rows], None, None, nv=None)
            vals = []
            for m in ("mixed", ref):
                o = out.clone().requires_grad_(True)
                tot, _ = ns.losses(m)(o, S, rows, an)
                g = torch.autograd.grad(tot, o)[0]
                vals.append((float(tot), g))
            worst = max(worst, abs(vals[0][0] - vals[1][0]), float((vals[0][1] - vals[1][1]).abs().max()))
        res[kind] = worst
    ok = all(v <= 1e-6 for v in res.values())
    print(json.dumps(dict(pool=a.on, max_abs_diff=res, pass_=ok)))
    raise SystemExit(0 if ok else 2)


def cmd_rely(a):
    import torch
    import pp_train as T
    from jevdrive.common import data_dir
    dev = torch.device("cuda")
    a.rm_from = a.tag
    ns, rp, JD, TU, okc = build(a, dev)
    LS = ns.losses(ns.mode)
    Sd = T.dev_store(ns.S, ns.dv_rows, a.dev_card_gb)
    rr = rely(ns.model, Sd, ns.dv_o, LS.W, JD, TU, ns.S.tab["log"])
    print(json.dumps(rr, indent=1))
    out = data_dir() / "runs/route_mem/rely"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(rr, indent=1))


def cmd_zero(a):
    """--init TAG --tag OUT: TAG's weights + the branch at its zero init (the serving check: plans equal TAG's, the memory is logged)."""
    import torch
    import pp_train as T
    d = T.proot("runs", a.tag)
    assert not (d / "ckpt-final.pt").exists(), f"{d} exists"
    m = T.load_pmodel(a.init, torch.device("cuda"))
    RM.attach(m, **HEAD)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state(), "route_mem": HEAD, "zero_of": a.init}, d / "ckpt-final.pt")
    RM.write_sidecar(a.tag, HEAD, dict(init=a.init, zero=True))
    print("wrote", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "rely", "mixcheck"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--exp", default="route_mem")
        p.add_argument("--steps", type=int, default=10000)
        p.add_argument("--batch", type=int, default=128)
        p.add_argument("--warmup", type=int, default=300)
        p.add_argument("--lr-scale", type=float, default=1.0)
        p.add_argument("--eval-every", type=int, default=1000)
        p.add_argument("--allow-init", action="store_true", help="permit an --init other than tonight's RB-F / RB0-F")
        OT.recipe_args(p)
        p.add_argument("--on-k", type=int, default=13)
        p.add_argument("--on-start", type=int, default=0, help="POOL-PILOT: the pool rows enter the batches from step T (before: normal rows "
                       "in their slots, the pool stream untouched; no lr restart). 0 (default) = from the start, as before")
        p.add_argument("--on-mode", default="", choices=("",) + OT.ALL_MODES)
        p.add_argument("--on-split", default="")
        p.add_argument("--on-lon", default="")
        p.add_argument("--on-lon-lam", type=float, default=10.0)
        p.add_argument("--on-road", default="")
        p.add_argument("--on-road-pai", default="", help="round5 road_pai.py index: a further road raster source of the pool rows (on_train; default off)")
        p.add_argument("--on-road-pai-w", type=float, default=1.0, help="weight of the road hinge on the rows --on-road-pai labels")
        p.add_argument("--ego-drop", type=float, default=0.0)
        p.add_argument("--on-weight", default="", help="KEY=VALUE:W,... draw weights of the pool rows from tab.npz string keys")
        p.add_argument("--on-cap-scene", type=float, default=0.0, help="max share of one scene (tab `log`) in the cap subset's draws (0 = off)")
        p.add_argument("--on-cap-in", default="", help="KEY=VALUE: the pool rows the cap applies to (default: all pool rows)")
        p.add_argument("--ego-drop-speed", type=float, default=0.0, help="share of non-anchor rows with the ego speed history zeroed (training only)")
        p.add_argument("--ego-drop-open", action="store_true", help="--ego-drop-speed only on OPEN rows (SLOW S1; docstring)")
        p.add_argument("--open-cells", default="runs/route_follow/lon_diag/openloop/cells/cells.npz", help="rf_lonol cells (offline rows' `free`)")
        p.add_argument("--open-pack", default="", help="on_lon pack with p_open_design for the pool rows (default on_lon.pack_path(<POOL>))")
        for t in ("lead", "curve", "stop"):
            p.add_argument(f"--lon-{t}", type=float, default=0.0, help=f"weight of on_lonsafe's {t} term on the pool rows (0 = off)")
        p.add_argument("--prefetch", type=int, default=4)
        p.add_argument("--fetch-workers", type=int, default=2)
        p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=os.environ.get("JEV_RM_EAGER", "") in ("", "0"),
                       help="torch.compile of the main training forward (default on since 2026-10-17; --no-compile or JEV_RM_EAGER=1 = eager)")
        p.add_argument("--dev-card-gb", type=float, default=4.0)
        p.add_argument("--rm-lr", type=float, default=3e-4)
        p.add_argument("--rm-warmup", type=int, default=100)
        p.add_argument("--rm-drop", type=float, default=0.25)
        p.add_argument("--rm-order", action="store_true", help="route v2: the ordered memory layout (feat/<data>.o.npz)")
        p.add_argument("--lane-lam", type=float, default=0.0, help="route v2: lateral hinge to the carried route on every fed row (0 = off)")
        p.add_argument("--lane-dz", type=float, default=0.5)
        p.add_argument("--lane-neg", type=float, default=0.2, help="share of eligible rows with a fake (shifted / straight) input memory")
        p.add_argument("--lane-shift", type=float, default=3.5)
        p.add_argument("--lane-over-lam", type=float, default=None, help="over-turn guard weight (default --lane-lam)")
        p.add_argument("--lane-over-deg", type=float, default=10.0)
        p.add_argument("--lc-cf", action="store_true", help="route v2: lane-change counterfactual pairs")
        p.add_argument("--lc-k", type=int, default=8)
        p.add_argument("--lc-lam", type=float, default=None)
        p.add_argument("--lc-dz", type=float, default=0.5)
        p.add_argument("--lc-adapter-only", action="store_true", help="the stay-lane rows' gradient reaches the adapter + route branch only")
        p.add_argument("--rm-latfeat", action="store_true", help="route v2: the route's lateral offset at 10 / 20 / 30 / 40 m into the branch")
        p.add_argument("--rm-anchor", action="store_true", help="ANCHOR: the anchored memory layout (feat/<data>.a.npz, rmem.memory_a)")
        p.add_argument("--rm-win", action="store_true", help="WIN: the speed-scaled window layout + heading command (feat/<data>.w.npz)")
        p.add_argument("--app-lam", type=float, default=0.0, help="approach course: weight W of on_approach's terms (--on-lon-lam 10 W; 0 = off)")
        p.add_argument("--app-v2", action="store_true", help="approach course v2 terms (on_approach.configure; needs --app-lam)")
        p.add_argument("--env-lam", type=float, default=0.0, help="RECOVERY-ENVELOPE L: weight W of on_envelope's terms (--on-lon-lam 10 W; 0 = off)")
        p.add_argument("--env-lo-scale", type=float, default=1.0, help="RECOVERY-ENVELOPE: on_envelope's lower-bound term x this (1 = as --env-lam)")
        p.add_argument("--on-lon-all", action="store_true", help="--on-mode mixed: the --on-lon terms on every pool row (self rows too)")
        p.add_argument("--desire-rows", default="", help="DESIRE D1: row table npz (dt.py rows; '' = off)")
        p.add_argument("--d1-k", type=int, default=0, help="DESIRE D1: table navtrain rows replacing the last K normal rows of a batch")
        p.add_argument("--d2-lam", type=float, default=0.0, help="DESIRE D2: weight of the desire-delta distillation (0 = off)")
        p.add_argument("--d2-frac", type=float, default=0.125, help="DESIRE D2: share of the non-anchor normal rows (anchors always)")
        p.add_argument("--d2-teacher", default="", help="DESIRE D2: dir of dt.py teacher files <data>.npz")
        p.add_argument("--kd-lam", type=float, default=0.0, help="RETAIN: weight of the P0 speed-profile distillation (0 = off)")
        p.add_argument("--kd-mode", default="lon", choices=("lon", "full"), help="RETAIN: arc-length profile only (lon) or the whole plan")
        p.add_argument("--kd-k", type=int, default=16, help="RETAIN: retention pool rows per step (second forward)")
        p.add_argument("--kd-pool", default="on1_WAIT", help="RETAIN: retention pool (cache dir with tab `cell`, @warp teacher, route features)")
        p.add_argument("--kd-groups", default="AR:2,W:1", help="RETAIN: draw shares of the arrival / standing-wait / launch / stop+yield groups")
        p.add_argument("--kd-nav", default="", help="RETAIN: navtrain approach rows (nx.py approach npz; '' = none)")
        p.add_argument("--kd-nav-k", type=int, default=0, help="RETAIN: navtrain approach rows per step, fed the constant-velocity history")
        p.add_argument("--ego-scale-speed", type=float, default=0.0, help="ANTI-COPY: share of the imitation rows with the speed history rescaled (0 = off)")
        p.add_argument("--ego-scale-max", type=float, default=0.2, help="ANTI-COPY: s ~ U[1 - M, 1 + M]")
        p.add_argument("--cmd-rule", default="", choices=("", "heading"), help="round7 joint arm: pool rows' command from the route memory (cmd_rule)")
        p.add_argument("--twin", default="", help="round7 HARM-MIX: <twin store>:<p>, a drawn pool row with a twin takes the twin's tokens with "
                       "probability p (lib/twin_tokens.py; '' = off)")
        p.add_argument("--on-wfile", default="", help="NR1: per-row draw weights of the pool rows (npz names, w; '' = off)")
        p.add_argument("--on-normal", action="store_true", help="NR1: pool rows are imitation rows for --rm-drop / --ego-scale-speed")
        p.add_argument("--lat-onc", type=float, default=0.0, help="NR1: weight of the opposite-lane point term (nr1_lat; 0 = off)")
        p.add_argument("--lat-soft", type=float, default=0.0, help="NR1: weight of the same-direction other-lane point term (0 = off)")
        p.add_argument("--lat-side", type=float, default=0.0, help="NR1: weight of the agent hinge's side buffer term (0 = off)")
        p.add_argument("--lat-file", default="", help="NR1: lateral raster file of the pool (df_lat.py build)")
        p.add_argument("--lat-depth", type=float, default=0.0, help="NR1: allowed depth (m) into the opposite-lane region")
        p.add_argument("--lat-soft-depth", type=float, default=0.0, help="NR1: allowed depth (m) into the soft region")
        p.add_argument("--lat-buf", type=float, default=0.4, help="NR1: side buffer (m) of the side term")
        p.add_argument("--on-arc", type=float, default=0.0, help="NR1HVR: weight of the two-sided longitudinal arc term on the pool rows (0 = off)")
        p.add_argument("--on-arc-rows", default="move", choices=("move", "brake"), help="JTB: the arc term's rows (move = old rule; brake = "
                       "every row but the still-all-horizon ones) and the per-kind arc bias monitoring")
        p.add_argument("--on-free", type=float, default=0.0, help="NR1FREE: weight of the one-sided longitudinal consequence term on the S rows "
                       "(defect_fix lib/nr1_free.py; 0 = off)")
        p.add_argument("--on-free-frac", type=float, default=0.8, help="NR1FREE-M: the term's band (plan 4 s arc below FRAC x the reference fires)")
        p.add_argument("--on-free-mode", default="all", choices=("all", "move"), help="NR1FREE-M: rows that may fire (move = ego >= 3 m/s, the "
                       "human neither stopping nor braking; prereg supplement 16)")
        p.add_argument("--on-free-clear", default="0,0", help="NR1 supplement 19: FREE's stricter free check CMIN,CTH (no logged object within "
                       "max(CMIN m, CTH s x v) ahead of the bumper along the sweep; 0,0 = off)")
        p.add_argument("--on-stand", type=float, default=0.0, help="NR1STAND: weight of the standing-row mirror term (defect_fix lib/nr1_free.py; 0 = off)")
        p.add_argument("--on-stand-margin", type=float, default=1.0, help="NR1STAND: the cap's margin (m) over the standing human's 4 s arc")
        p.add_argument("--on-stand-roll", type=float, default=0.0, help="STAND-R (NR1 prereg supplement 20): STAND also on rolling rows with ego "
                       "speed in [0.5, V) m/s, cap aligned to the human's standing station (labels <pool>/stand_gap.npz; 0 = off)")
        p.add_argument("--on-stand-launch", type=float, default=0.0, help="STAND-R2 (NR1 prereg supplement 23): drop STAND / STAND-R rows whose "
                       "human launches within 4 + H s (labels <pool>/stand_launch.npz; 0 = off)")
        p.add_argument("--on-stoppt", type=float, default=0.0, help="NR1STOPPT: weight of the approach-row stop-point term (defect_fix lib/nr1_free.py; "
                       "labels <pool>/stoppt.npz; 0 = off)")
        p.add_argument("--on-turncap", type=float, default=0.0, help="TURN-CAP (NR1 prereg supplement 21): weight of the sharp-turn upper arc limit "
                       "(defect_fix lib/nr1_free.py; labels <pool>/turn_cap.npz; 0 = off)")
        p.add_argument("--on-tightin", type=float, default=0.0, help="TIGHT-IN (NR1 prereg supplement 22): weight of the tight-turn outward "
                       "lateral term + arc cap (defect_fix lib/nr1_tight.py; labels <pool>/tight_in.npz; 0 = off)")
        p.add_argument("--on-anchor", type=float, default=0.0, help="LON-ANCHOR: weight of the station-anchored two-sided arc term on the S rows "
                       "(defect_fix lib/nr1_anchor.py; labels <pool>/lon_anchor.npz; 0 = off)")
        p.add_argument("--save-at", default="", help="steps (comma list) at which a snapshot checkpoint is saved as tag <tag with x<step> before -L18->; "
                       "'' = off")
        p.add_argument("--flat-brk", default="", help="SIMP-F / SIMP-G: file of navtrain tokens (brake-needed / braking-onset rows) whose fed history is "
                       "flattened on training draws (ax 0, past poses at constant current speed; label kept); '' = off")
        p.add_argument("--flat-p", type=float, default=1.0, help="ROUND2: with --flat-brk, probability that a draw of a listed row is flattened "
                       "(own stream [seed, 0, 179]); 1 = every draw (as before), 0 = off")
        p.add_argument("--simp-region", type=float, default=0.0, help="SIMP: weight (x 10) of the merged road + oncoming-lane hinge on the S rows; "
                       "replaces the trainer's road hinge there (defect_fix lib/nr1_simp.py; needs --lat-file for the onc raster; 0 = off)")
        p.add_argument("--simp-clear", type=float, default=0.0, help="SIMP: weight (x 10) of the merged object-clearance hinge on the S rows (leads may "
                       "only shorten the plan); replaces the trainer's agent hinge there (defect_fix lib/nr1_simp.py; 0 = off)")
        p.add_argument("--on-hold-res", type=int, default=-1, help="NR1HVR: hold out the pool logs with sha1(log) %% 10 == R (-1 = off)")
        p.add_argument("--on-hold-n", type=int, default=2048, help="NR1HVR: rows of the held / fitted pool slices scored at every dev eval")
        p.add_argument("--on-hold-pct", type=int, default=0, help="JT: hold out the pool logs with crc32(log) %% 100 < P (0 = off; mixed pools: "
                       "the `self` rows only); as --on-hold-res otherwise")
        p.add_argument("--lat-host", action="store_true", help="JT: the lateral rasters stay a host memmap (a pool too big for the card)")
        p.add_argument("--probe-n", type=int, default=1024, help="JT: rows per wait / launch probe slice (--on-wfile groups stand_r / launch; 0 = off)")
        p.add_argument("--diag", action="store_true", help="monitoring only: activity rates, grad norm / clip rate, per-term trunk gradients")
        p.add_argument("--diag-every", type=int, default=100, help="--diag: steps between per-term gradient norms")
        p.add_argument("--diag-infl", type=int, default=0, help="LAUNCH-DIAG (with --diag): rows per probe set of the per-term arc influence (0 = off)")
        p.add_argument("--diag-freeze", action="store_true", help="LAUNCH-DIAG: every group's lr is 0 (the weights stay at the init; Adam "
                       "state accumulates; --diag-infl uses the base lr)")
        p.add_argument("--lr-floor", type=float, default=0.0, help="cosine floor as a share of each group's base lr (0 = the plain cosine)")
        p.add_argument("--si-scan", default="", help="SELF-ITER: per-row error scan npz (si.py scan; '' = off)")
        p.add_argument("--si-mode", default="top", choices=("top", "rank"), help="SELF-ITER: top-share oversampling or rank^gamma weights")
        p.add_argument("--si-top", type=float, default=0.1, help="SELF-ITER: top share of each set oversampled (--si-mode top)")
        p.add_argument("--si-mult", type=float, default=3.0, help="SELF-ITER: weight of the top share (1 = uniform over the same rows)")
        p.add_argument("--si-gamma", type=float, default=1.0, help="SELF-ITER: rank exponent (--si-mode rank)")
        p.add_argument("--si-key", default="ade", help="SELF-ITER: scan error that ranks the rows")
        p.add_argument("--si-k-s", type=int, default=13, help="SELF-ITER: expected S (pool `s`) rows per batch in the merged pool")
        p.add_argument("--si-split", default="navsim/selfiter_tv", help="SELF-ITER: registered split of the new trainval rows (recorded)")
        p.add_argument("--rx", action="store_true", help="STACK: the reflex module (lib/reflex.py)")
        p.add_argument("--rx-drop", type=float, default=0.25)
        p.add_argument("--rx-lr", type=float, default=3e-4)
        p.add_argument("--rx-warmup", type=int, default=100)
        p.add_argument("--dual", default="", help="STACK: constraints with Lagrangian multipliers, e.g. agent,road,envU,cons")
        p.add_argument("--dual-tau", default="", help="reference violation rates c=r,..; missing or 'auto' = 0.5 x the first 25 steps' mean")
        p.add_argument("--dual-level", type=float, default=1.0)
        p.add_argument("--dual-eta", type=float, default=0.02)
        p.add_argument("--cons", default="", help="STACK: frozen consequence head checkpoint (lib/conshead.py) as a hinge on every row")
        p.add_argument("--imit-drop", action="store_true", help="STACK: imitation dropout on rows with an active constraint")
        p.add_argument("--pref", type=float, default=0.0, help="STACK: one-sided minimal-correction preference weight (0 = off)")
        p.add_argument("--awr-pack", default="", help="AWR-ONPOL: outcome pack of the --on pool (aw_score.py build; '' = off)")
        p.add_argument("--awr-k", type=int, default=16, help="AWR-ONPOL: pool rows per step (second forward)")
        p.add_argument("--awr-lam", type=float, default=1.0, help="AWR-ONPOL: weight (1: each drawn row weighs one imitation row)")
        p.add_argument("--awr-mode", default="filt", choices=("filt", "awr"), help="AWR-ONPOL: filtered cloning (A > keep) or exp(A / beta) draws")
        p.add_argument("--awr-keep", type=float, default=0.1, help="AWR-ONPOL filt: keep rows with advantage A > this")
        p.add_argument("--awr-beta", type=float, default=0.1, help="AWR-ONPOL awr: temperature")
        p.add_argument("--awr-wmax", type=float, default=20.0, help="AWR-ONPOL awr: weight clip")
        p.add_argument("--awr-kinds", default="1", help="AWR-ONPOL: baseline kinds kept (1 near-identical peers, 2 per-(scene, time) fallback)")
    p = sub.add_parser("zero")
    p.add_argument("--init", required=True), p.add_argument("--tag", required=True)
    sub.add_parser("capcheck")
    p = sub.add_parser("draws")
    p.add_argument("--on", required=True), p.add_argument("--on-weight", default="")
    p.add_argument("--on-cap-scene", type=float, default=0.0), p.add_argument("--on-cap-in", default="")
    a = ap.parse_args()
    if a.cmd in ("zero", "capcheck", "draws"):
        dict(zero=cmd_zero, capcheck=cmd_capcheck, draws=cmd_draws)[a.cmd](a)
        raise SystemExit(0)
    assert not a.on_cap_scene or (a.on and 0 < a.on_cap_scene < 1), "--on-cap-scene: needs --on, a share in (0, 1)"
    assert a.ego_drop == 0, "ROUTE-MEM: no --ego-drop (--ego-drop-speed drops the speed history only)"
    assert 0 <= a.ego_drop_speed < 1, "--ego-drop-speed is a share in [0, 1)"
    assert not getattr(a, "ego_drop_open", False) or a.ego_drop_speed > 0, "--ego-drop-open: needs --ego-drop-speed"
    assert 0 <= getattr(a, "ego_scale_speed", 0.0) <= 1 and 0 < getattr(a, "ego_scale_max", 0.2) < 1, "--ego-scale-speed in [0, 1], --ego-scale-max in (0, 1)"
    if a.agent_on_only and not a.on:                                 # agent hinge on pool rows only, and no pool: no row carries it (RB0 / RNONE)
        print("rm_train: --agent-on-only without --on -> --agent-lam 0 (no row has an agent label)", flush=True)
        a.agent_lam, a.agent_on_only = 0.0, False
    if max(a.lon_lead, a.lon_curve, a.lon_stop) > 0:                 # on_lonsafe: the weights carry the scale
        assert a.on and a.on_lon in ("", "on_lonsafe"), "--lon-*: needs --on and no other --on-lon module"
        import on_lonsafe
        a.on_lon, a.on_lon_lam = "on_lonsafe", 1.0
        print(f"rm_train: --on-lon on_lonsafe, --on-lon-lam 1, weights {on_lonsafe.configure(a.lon_lead, a.lon_curve, a.lon_stop)}", flush=True)
    if a.app_lam > 0:                                                # approach course: on_approach as the --on-lon module
        assert a.on and a.on_lon in ("", "on_approach"), "--app-lam: needs --on and no other --on-lon module"
        a.on_lon, a.on_lon_lam = "on_approach", 10.0 * a.app_lam
        print(f"rm_train: --on-lon on_approach, --on-lon-lam {a.on_lon_lam:g}", flush=True)
    if getattr(a, "app_v2", False):
        assert a.app_lam > 0, "--app-v2: needs --app-lam"
        import on_approach
        print(f"rm_train: approach v2 {on_approach.configure(True)}", flush=True)
    if getattr(a, "env_lam", 0.0) > 0:                               # RECOVERY-ENVELOPE L: on_envelope as the --on-lon module
        assert a.on and a.on_lon in ("", "on_envelope") and a.app_lam == 0, "--env-lam: needs --on and no other --on-lon module"
        a.on_lon, a.on_lon_lam = "on_envelope", 10.0 * a.env_lam
        print(f"rm_train: --on-lon on_envelope, --on-lon-lam {a.on_lon_lam:g}", flush=True)
        if a.env_lo_scale != 1.0:
            import on_envelope
            print(f"rm_train: on_envelope {on_envelope.configure(a.env_lo_scale)}", flush=True)
    assert getattr(a, "on_arc_rows", "move") == "move" or max(a.lat_onc, a.lat_soft, a.lat_side, a.on_arc, getattr(a, "simp_region", 0.0),
                                                              getattr(a, "simp_clear", 0.0), getattr(a, "on_anchor", 0.0)) > 0, "--on-arc-rows: needs nr1_lat"
    if max(getattr(a, "lat_onc", 0.0), getattr(a, "lat_soft", 0.0), getattr(a, "lat_side", 0.0), getattr(a, "on_arc", 0.0), getattr(a, "on_free", 0.0), getattr(a, "on_stand", 0.0), getattr(a, "on_stoppt", 0.0), getattr(a, "on_turncap", 0.0), getattr(a, "on_tightin", 0.0), getattr(a, "on_anchor", 0.0), getattr(a, "simp_region", 0.0), getattr(a, "simp_clear", 0.0)) > 0:   # NR1: nr1_lat as the --on-lon module
        assert a.on and a.on_lon in ("", "nr1_lat") and a.on_mode in ("self", "both", "mixed"), "--lat-*: needs --on (self | both | mixed), no other --on-lon"
        assert (a.lat_onc == 0 and a.lat_soft == 0) or a.lat_file, "--lat-onc / --lat-soft: need --lat-file"
        _sys.path.insert(0, str(_R / "experiments/defect_fix/lib"))
        import nr1_lat
        a.on_lon, a.on_lon_lam = "nr1_lat", 1.0
        print(f"rm_train: --on-lon nr1_lat, --on-lon-lam 1, {nr1_lat.configure(a.lat_onc, a.lat_soft, a.lat_side, a.lat_depth, a.lat_soft_depth, a.lat_buf, a.lat_file, a.on_arc, getattr(a, 'lat_host', False), getattr(a, 'on_arc_rows', 'move'), free=getattr(a, 'on_free', 0.0), free_frac=getattr(a, 'on_free_frac', 0.8), free_mode=getattr(a, 'on_free_mode', 'all'), stand=getattr(a, 'on_stand', 0.0), stand_margin=getattr(a, 'on_stand_margin', 1.0), stand_roll=getattr(a, 'on_stand_roll', 0.0), stand_launch=getattr(a, 'on_stand_launch', 0.0), stoppt=getattr(a, 'on_stoppt', 0.0), free_clear=tuple(float(x) for x in getattr(a, 'on_free_clear', '0,0').split(',')), turncap=getattr(a, 'on_turncap', 0.0), tightin=getattr(a, 'on_tightin', 0.0), anchor=getattr(a, 'on_anchor', 0.0), simp_region=getattr(a, 'simp_region', 0.0), simp_clear=getattr(a, 'simp_clear', 0.0))}", flush=True)
    if getattr(a, "on_lon_all", False):
        assert a.on_lon and a.on_mode == "mixed", "--on-lon-all: needs an --on-lon module and --on-mode mixed"
    assert getattr(a, "on_hold_res", -1) < 0 or (a.on and 0 <= a.on_hold_res < 10), "--on-hold-res: needs --on, a residue 0..9"
    assert getattr(a, "on_hold_pct", 0) == 0 or (a.on and 0 < a.on_hold_pct < 100 and a.on_hold_res < 0), "--on-hold-pct: needs --on, 1..99, not with --on-hold-res"
    assert 0 <= getattr(a, "flat_p", 1.0) <= 1 and (getattr(a, "flat_p", 1.0) == 1 or getattr(a, "flat_brk", "")), "--flat-p in [0, 1], needs --flat-brk"
    assert 0 <= getattr(a, "lr_floor", 0.0) < 1 and getattr(a, "diag_every", 100) > 0, "--lr-floor in [0, 1), --diag-every > 0"
    assert not (getattr(a, "rm_anchor", False) and a.lc_cf), "--rm-anchor: no anchored --lc-cf rows (rf_cfpool builds the ordered layout)"   # ANCHOR
    assert not getattr(a, "rm_win", False) or not (a.lc_cf or getattr(a, "kd_lam", 0) > 0 or getattr(a, "lane_lam", 0) > 0 or getattr(a, "cmd_rule", "")), \
        "--rm-win: not with --lc-cf / --kd-lam / --lane-lam / --cmd-rule"   # WIN
    dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)
