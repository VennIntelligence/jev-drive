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
            stream still draws one number per batch row. --on-lon / --on-lon-lam / --on-road go to on_train.setup unchanged.

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
import argparse, hashlib, json, time  # noqa: E401,E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402

import on_train as OT  # noqa: E402  (sets the repo paths)
import rmem as RM  # noqa: E402

FEAT = None
HEAD = dict(d=128, layers=2, heads=4)
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]              # ego_features: vx, vy, ax, ay, history poses x / y (not present, cmd, yaw)


def feats(cfg_data, S, dev, order=False, old=False):
    """Route features (S.n, DIM | DIM_O with order) float32 on dev, and per Store row: junction distance / route turn (dev rows of the NAVSIM
    dirs, else nan), the source dir index. Asserts every file's names equal the Store tab's names of its dir. old: also the input layout's point
    block (S.n, DIM) f16 on dev and trav (S.n,) (route v2's targets / negatives; with order both come from the .o files)."""
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
            zo = np.load(root / f"{d}.o.npz")
            assert np.array_equal(zo["names"], z["names"]) and np.array_equal(zo["ok"], z["ok"]), f"{d}.o: rows differ"
            TR.append(zo["trav"].astype(np.float32))
            f = zo["f"].astype(np.float32)
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
        assert {k: v for k, v in sh.items() if k not in ("order", "latfeat")} == HEAD, f"{src}: branch {sh} != {HEAD}"
        assert sh.get("order", False) <= head.get("order", False), f"{src}: an ordered-layout branch needs --rm-order"
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
    F, JD, TU, src, okc, ns.f_old, ns.trav = feats(ns.cfg.data, ns.S, dev, order=head.get("order", False), old=lane)
    ns.S.ego = torch.cat([ns.S.ego.float(), F], 1)
    rp = list(ns.model.adapter.route.parameters())
    ns.rxp = list(ns.model.adapter.rx.parameters()) if ns.rx else []
    ids = {id(p) for p in rp + ns.rxp}
    ns.new = [p for p in ns.model.adapter.parameters() if id(p) not in ids]
    if ns.rx:
        ns.S.ego, ns.rx_rec = rx_feats(ns, dev)
    return ns, rp, JD, TU, okc


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
    upre = 0                                                          # weight-0 rows at the end, equal weights elsewhere: the plain uniform draw
    if pw is not None and (pw == 0).any():
        m0 = int((pw > 0).sum())
        if (pw[:m0] > 0).all() and (pw[:m0] == pw[0]).all():
            upre = m0
    OPEN, orec = open_rows(a, ns, dev) if a.ego_drop_open else (None, {})

    lc_rows = LN.lc_row.cpu().numpy() if lc_on else None

    def draw():
        r = rng.choice(tr_o, nN, replace=len(tr_o) < nN)
        if k_yr:
            r = np.concatenate([r, rng.choice(tr_t, k_yr, replace=False)])
        an = rng.random(nB - k_ho - k_on) < cfg.d_frac
        an[nN:] = False
        if k_on:
            i = (orng.choice(ns.n_on, k_on, replace=ns.n_on < k_on) if pw is None else orng.choice(upre, k_on, replace=upre < k_on) if upre
                 else orng.choice(ns.n_on, k_on, replace=True, p=pw))
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
        return r, an, sha.hexdigest()[:16]

    def fetch(dr):
        return dr, S.front[dr[0]]
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
        ev = dev_ev()
        run.scalars({f"dev/{q}": v for q, v in ev.items()}, 0)
        run.info("dev @ 0: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items())
                 + (f"; init's recorded dev ade {ns.init['init_dev_ade']:.4f}" if ns.init and ns.init.get("init_dev_ade") is not None else ""))
        run.summary.update({f"dev0_{q}": v for q, v in ev.items()} | {"init": ns.init})
        is_pool = torch.zeros(S.n, dtype=torch.bool, device=dev)
        is_pool[ns.on0:] = bool(k_on)                                 # --rm-drop never applies to pool rows
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
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / g["wu"]) * cosf
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
            out = model(front, ego, S.tc[rows], None, None, nv=None if ns.nv is None else ns.nv[rows])
            if lc_on:
                total, Ls = LS(out[:n_main], S, rows[:n_main], anchor[:n_main])
            else:
                total, Ls = LS(out, S, rows, anchor)
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
            torch.nn.utils.clip_grad_norm_(base + new + rp + rxp, 1.0)
            scaler.step(opt)
            scaler.update()
            if ST is not None:
                ST.update(Ls, step)
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
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                ev = dev_ev()
                run.scalars({f"dev/{q}": v for q, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{q} {v:.4f}" for q, v in ev.items()))
                run.summary.update({f"dev_{q}": v for q, v in ev.items()})
        torch.save({"model": model.state(), "cfg": asdict(cfg), "onpol": vars(a), "route_mem": ns.head}, d / "ckpt-final.pt")
        torch.save(model.adapter.state_dict(), d / "adapter.pt")
        v2 = dict(lane_lam=a.lane_lam, lane_dz=a.lane_dz, lane_neg=a.lane_neg, lc_cf=a.lc_cf) if (lane_on or lc_on or ns.head.get("order")) else {}
        rx_side = dict(reflex=dict(sf=__import__("reflex").SF, rx=None, drop=a.rx_drop, lr=a.rx_lr)) if ns.rx else {}
        RM.write_sidecar(a.tag, ns.head, dict(init=a.init, rm_lr=a.rm_lr, rm_warmup=a.rm_warmup, rm_drop=a.rm_drop, rm_drop_pool=False,
                                              on=a.on, on_lon=a.on_lon, steps=cfg.steps, dev_ade=ev.get("ade"), **v2, **rx_side))
        if ST is not None:
            run.summary.update(stack=ST.summary(), **({"rx_feats": ns.rx_rec} if ns.rx else {}))
            run.info(f"stack: {json.dumps(ST.summary())}")
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
        p.add_argument("--on-mode", default="", choices=("",) + OT.ALL_MODES)
        p.add_argument("--on-split", default="")
        p.add_argument("--on-lon", default="")
        p.add_argument("--on-lon-lam", type=float, default=10.0)
        p.add_argument("--on-road", default="")
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
        p.add_argument("--app-lam", type=float, default=0.0, help="approach course: weight W of on_approach's terms (--on-lon-lam 10 W; 0 = off)")
        p.add_argument("--app-v2", action="store_true", help="approach course v2 terms (on_approach.configure; needs --app-lam)")
        p.add_argument("--env-lam", type=float, default=0.0, help="RECOVERY-ENVELOPE L: weight W of on_envelope's terms (--on-lon-lam 10 W; 0 = off)")
        p.add_argument("--env-lo-scale", type=float, default=1.0, help="RECOVERY-ENVELOPE: on_envelope's lower-bound term x this (1 = as --env-lam)")
        p.add_argument("--on-lon-all", action="store_true", help="--on-mode mixed: the --on-lon terms on every pool row (self rows too)")
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
    if getattr(a, "on_lon_all", False):
        assert a.on_lon and a.on_mode == "mixed", "--on-lon-all: needs an --on-lon module and --on-mode mixed"
    dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)
