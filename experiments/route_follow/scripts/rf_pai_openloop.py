#!/usr/bin/env python3
"""Turning school selection ladder step 3 (prereg experiments/route_follow/plans/2026-10-13-turning-school-prereg.md, Amendment 2): an OFFLINE
PAI turn read. Checkpoints are run open loop on log-following PAI rows (the human's states, frozen-encoder tokens, the driver's route memory)
around strict route turns and scored against the human. Box, envs/op-train, one GPU through the pool (CPU works, slowly).

Rows: a log-pass cache in the onp_cache layout (tok.npy, tab.npz: tix, ego_in, cam_t, lht, imit, log_fut, k, t0_us, usable, ...) and its
rf_pait.py `route` output (per scene: f (route memory), anchor, poly, turns, topo), the scenes of --scenes minus --exclude (default: the exam's
hard_exam list, never used for selection). Turns, windows and the distance to the entry d (route stations) are rf_pait pack's.
Per turn and checkpoint (sign s = the turn's direction, left +):
  speed_d      plan speed (chord arc of the 0.5 / 1.0 s poses over 1 s) at the row nearest d = 40, 20, 0 m before the entry (within 4 m),
               against the human's (the same on `imit`, the human-timed demo)
  lat2         rows with d in [-10, 20] m and |needed| >= 0.5 m: s x plan y at 2 s divided by s x human y at 2 s (needed); median per turn
  side4        rows with d in [-10, 20] m and |human y at 4 s| >= 1 m: plan y at 4 s on the turn's side (share)
--served-tag: the checkpoint that drove the rows (P1: RSG-L18-s0) is also compared with its own served plan (poses_model), a check that
the read's inputs (tokens, ego, regenerated route memory) reproduce what the driver computed.
Each turn's row of the read says whether its scene is in a training pool part (--train-scenes: the pool's scenes), so held-out and in-sample
turns are reported apart. -> <out>/<name>.json (per tag: per turn, medians; held-out / in-sample) and <out>/<name>.md.

  rf_pai_openloop.py --cache <log-pass cache> --route <rf_pait route dir> --scenes <list> --train-scenes <list> \
      --tags RSG-L18-s0 RSGPAIT-L18-s0 RPAIT-L18-s0 RPAITC-L18-s0 --out experiments/route_follow/results/pai_openloop --name p1

P1 mode (--p1, research/route-follow/p1-openloop.md): the same question on the Tokyo P1 render, i.e. exam-hard PAI turn scenes driven along the
HUMAN LOG by experiments/onpol_pai/lib/onp_log_driver.py with one checkpoint fed and logged (drive.jsonl: `ego`, `rmem`, `poses_model`, `log_now`,
`log_poses` per decision; dump/*.npz: the 8 slot frames `cur` + `ego` once per second, k = 0, 10, ..). Any checkpoint is replayed OFFLINE on
those inputs: pai_core.plan on the dumped frames, the dumped ego row and the logged route memory (`rmem`, the exact features pai_driver fed:
rmem.Tracker on the run's own route messages, so nothing is rebuilt), right-hand traffic. Only dumped decisions can be replayed (1 Hz);
--logged <tag> names the checkpoint the render was driven with: every replayed tag is compared with the logged `poses_model` (largest position
difference over the 8 poses per decision; the logged tag's own replay is the equivalence check, lines: all within 0.01 m, and onp_cache's
cross-machine G-id line <= 1 % of decisions above 0.03 m; any other tag is the control) and the logged plans at all decisions (10 Hz) are
added as tag `<tag>@log`.
  turn      on the human path (the log at every decision, anchor o log_now, plus the last decision's log_poses): the first 30 m window with net
            heading change >= 45 deg; entry / exit where the change over [a - 10, a + 40] m reaches 10 / 90 %; sign s (left +). Scenes whose log
            never turns that much in the clip (the route turns after it) have no human reference and are dropped.
  frames    plan = model poses (rear axle, ego frame); human = log_poses re-expressed in the frame of log_now (the ego tracks the log within
            0.5 m, so both start at their own origin); decisions with all 8 log poses inside the log only. d = distance to the entry along
            the log (> 0 before it); window = decisions from 6 s before the log reaches the entry to the exit.
  speed     arc length of the poses at 2 s / 4 s (v4 = arc4 / 4) at the decision nearest d = 40, 20, 0 m and mid-turn (within 4 m), plan vs
            human; slows = v4 at 0 m below max(v4 at 40, 20 m) - 1 m/s
  lat2      s x y at 2 s (toward the turn), plan vs human (need); window: median of need - plan over decisions with need >= 0.5 m
  side4     signed offset of the plan's 4 s pose from the human path (origin + 8 poses, end segments extended), + = inside the turn;
            inside > 0.5 m, outside < -0.5 m, else on; window shares and median offset
  commit    the first decision (d <= 60 m, time order) whose s x heading at 4 s >= 20 deg, plan and human on the same decisions; lead =
            d_plan - d_human (m, > 0 = the plan commits earlier), only where neither had committed at the first eligible decision
            (else censored: turns entered right after the clip start); never = no such decision before the exit
  class     rf_pai_follow map class of the scene (results/pai_follow/scenes.csv `topo`, the rsg row else rbsg2; it exists only where one of
            them left the corridor at a turn): straight_ok (straight_exit / straight_road), no_straight (curve_single / no_straight_exit), unknown
CIs: jevdrive.stats bootstrap over turns (one turn per scene); paired differences against the first tag (--ref).
Outputs: <out>/<tag>.json per tag (plans per decision, per-turn metrics and per-decision traces, summaries), <out>/<name>.md (comparison),
with --figs DIR two figures (<name>_speed.png, <name>_lat.png). --reuse takes the plans of an existing <tag>.json instead of replaying (CPU,
needs only the drive.jsonl files: report iterations on the Mac).

  rf_pai_openloop.py --p1 $DATA_DIR/runs/route_follow/p1_openloop/rsg_p1 --logged RSG-L18-s0 \
      --tags RSG-L18-s0 RSGboth-L18-s0 RB0-F-s0 --name p1_exam48 [--figs experiments/route_follow/figs]     (box, one pool GPU job, ~2 min)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R / "experiments/route_follow/scripts"), str(R / "experiments/route_mem/lib"), str(R / "experiments/op_parity/scripts"),
                str(R / "experiments/onpol_pai/lib"), str(R / "lib"), str(R / "experiments/body1/lib"), str(R)]
import rf_pait as RP  # noqa: E402

DS = (40, 20, 0)


def rows_of(a):
    """-> per-row arrays of the selected turn rows: cache index, scene, turn id (global), d, sign, topo, route features."""
    T = np.load(Path(a.cache) / "tab.npz")
    scenes = T["scenes"][T["sid"]]
    excl = set(RP.scenes_of([a.exclude])) if a.exclude else set()
    want = [s for s in RP.scenes_of(a.scenes) if s not in excl]
    out = dict(ci=[], sc=[], tid=[], d=[], sgn=[], topo=[], f=[])
    turns = []
    for sc in want:
        f = Path(a.route) / f"{sc}.npz"
        if sc not in set(T["scenes"].tolist()) or not f.exists():
            continue
        Z = np.load(f)
        P = Z["poly"][:, :2]
        idx = np.flatnonzero(scenes == sc)
        idx = idx[np.argsort(T["k"][idx])]
        pos = {int(k): j for j, k in enumerate(Z["k"])}
        j = np.array([pos[int(k)] for k in T["k"][idx]])
        arc = RP.arc_on(P, Z["anchor"][j, :2])
        good = T["usable"][idx] & T["imit_ok"][idx].all(1) & T["log_fut_ok"][idx].all(1)
        for (s_in, s_out, deg), cl in zip(Z["turns"], Z["topo"]):
            if not (arc >= s_in).any() or arc[0] > s_out:
                continue
            d = s_in - arc
            w = good & (d <= 45) & (arc <= s_out)
            if not w.any():
                continue
            tid = len(turns)
            turns.append(dict(scene=sc, deg=float(deg), topo=str(cl), train=sc in a._train))
            n = int(w.sum())
            out["ci"].append(idx[w]), out["sc"].append(np.full(n, sc)), out["tid"].append(np.full(n, tid)), out["d"].append(d[w])
            out["sgn"].append(np.full(n, np.sign(deg))), out["topo"].append(np.full(n, str(cl))), out["f"].append(Z["f"][j[w]])
    return T, {k: np.concatenate(v) for k, v in out.items()}, turns


def plans(tag, T, X, dev, bs=256):
    """Rear-axle plan poses (n, 8, 3) of checkpoint `tag` on the rows (route features fed iff the checkpoint has the branch)."""
    import torch
    import onp_loss as OL
    import pp_train as T_
    import rmem as RM
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    m = RM.load_pmodel(tag, dev)
    br = RM.has_branch(tag)
    W = torch.as_tensor(R2.t_weights(T_.T8), device=dev)
    pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(X["ci"]), bs):
            c = X["ci"][i:i + bs]
            front = torch.as_tensor(np.stack([X["_tok"][T["tix"][r]] for r in c]), device=dev)
            ego = np.asarray(T["ego_in"][c], np.float32)
            if br:
                ego = np.concatenate([ego, X["f"][i:i + bs].astype(np.float32)], 1)
            lht = T["lht"][c].astype(bool)
            tc = torch.as_tensor(np.where(lht[:, None], [[0.0, 1.0]], [[1.0, 0.0]]), dtype=torch.float32, device=dev)
            o = m(front, torch.as_tensor(ego, device=dev), tc).float()
            x, y, psi = OL.rear_xy(o[:, pi].view(-1, 33, 15), torch.as_tensor(T["cam_t"][c][:, :2], dtype=torch.float32, device=dev), W)
            out.append(torch.stack([x, y, psi], -1).cpu().numpy())
    return np.concatenate(out)


def score(F, H, X, turns):
    """Plan poses F, human demo H (n, 8, 3) -> per turn metrics and summary."""
    v = lambda G: np.hypot(G[:, 1, 0], G[:, 1, 1])               # noqa: E731  chord from the origin to the 1.0 s pose over 1 s
    vp, vh = v(F), v(H)
    s = X["sgn"]
    per = []
    for t, tu in enumerate(turns):
        m = X["tid"] == t
        r = dict(tu)
        for d in DS:
            q = np.flatnonzero(m & (np.abs(X["d"] - d) <= 4))
            if len(q):
                i = q[np.argmin(np.abs(X["d"][q] - d))]
                r[f"v_plan_{d}"], r[f"v_human_{d}"] = float(vp[i]), float(vh[i])
        a = m & (X["d"] >= -10) & (X["d"] <= 20)
        need2, need4 = s * H[:, 3, 1], s * H[:, 7, 1]
        q = a & (np.abs(need2) >= 0.5)
        r["lat2"] = float(np.median(s[q] * F[q, 3, 1] / need2[q])) if q.any() else None
        q = a & (np.abs(need4) >= 1.0)
        r["side4"] = float(np.mean(s[q] * F[q, 7, 1] > 0)) if q.any() else None
        per.append(r)

    def summ(rs):
        o = dict(turns=len(rs))
        for d in DS:
            p = [(x[f"v_plan_{d}"], x[f"v_human_{d}"]) for x in rs if f"v_plan_{d}" in x]
            if p:
                o[f"v_plan_{d}"], o[f"v_human_{d}"] = float(np.median([u for u, _ in p])), float(np.median([w for _, w in p]))
                o[f"v_excess_{d}"] = float(np.median([u - w for u, w in p]))
        for k in ("lat2", "side4"):
            z = [x[k] for x in rs if x[k] is not None]
            o[k] = float(np.median(z)) if z else None
        return o
    return per, dict(all=summ(per), held_out=summ([x for x in per if not x["train"]]), in_sample=summ([x for x in per if x["train"]]),
                     straight_ahead=summ([x for x in per if x["topo"] in ("straight_exit", "straight_road")]))


def main(a):
    if a.p1:
        return p1_main(a)
    import torch
    dev = torch.device(a.device)
    a._train = set(RP.scenes_of(a.train_scenes)) if a.train_scenes else set()
    T, X, turns = rows_of(a)
    X["_tok"] = np.load(Path(a.cache) / "tok.npy", mmap_mode="r")
    H = T["imit"][X["ci"]].astype(np.float64)
    res = dict(cache=a.cache, rows=int(len(X["ci"])), turns=len(turns), held_out_turns=sum(not t["train"] for t in turns),
               excluded=a.exclude, tags={})
    for tag in a.tags:
        F = plans(tag, T, X, dev).astype(np.float64)
        per, sm = score(F, H, X, turns)
        res["tags"][tag] = dict(summary=sm, per_turn=per)
        if tag == a.served_tag:                                       # the rows' own driver: the read's plan vs the served one (poses_model)
            d = np.hypot(*(F[..., :2] - T["poses_model"][X["ci"]][..., :2]).transpose(2, 0, 1)).max(1)
            res["served_check"] = dict(tag=tag, median_m=float(np.median(d)), p95_m=float(np.quantile(d, 0.95)), max_m=float(d.max()))
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    hd = "| tag | part | turns | v plan / human @40 | @20 | @0 | v excess @0 | lat2 | side4 |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|\n"
    f = lambda x, k: "-" if x.get(k) is None else f"{x[k]:.2f}"  # noqa: E731
    md = "".join(f"| {t} | {p} | {s['turns']} | {f(s, 'v_plan_40')} / {f(s, 'v_human_40')} | {f(s, 'v_plan_20')} / {f(s, 'v_human_20')} | "
                 f"{f(s, 'v_plan_0')} / {f(s, 'v_human_0')} | {f(s, 'v_excess_0')} | {f(s, 'lat2')} | {f(s, 'side4')} |\n"
                 for t, v in res["tags"].items() for p, s in v["summary"].items())
    (o / f"{a.name}.md").write_text(f"# Offline PAI turn read `{a.name}` ({res['rows']} rows, {res['turns']} turns, {res['held_out_turns']} held out; "
                                    f"excluded {a.exclude or 'none'})\n\n" + hd + md)
    print(hd + md)


# ---------------------------------------------------------------- P1 mode
P1_DS, TOL_D, PRE_S, COMMIT_DEG, COMMIT_D = (40, 20, 0), 4.0, 6.0, 20.0, 60.0
STRAIGHT_OK, NO_STRAIGHT = ("straight_exit", "straight_road"), ("curve_single", "no_straight_exit")
TRACE = ("k", "d", "a2p", "a4p", "a2h", "a4h", "y2p", "y2h", "off4", "psi4p", "psi4h")   # per-decision trace columns (turn sign applied)


def compose(a, b):
    """Pose a (x, y, yaw) o poses b (..., 3) given in a's frame -> poses in a's parent frame."""
    b = np.asarray(b, np.float64)
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.stack([a[0] + c * b[..., 0] - s * b[..., 1], a[1] + s * b[..., 0] + c * b[..., 1], a[2] + b[..., 2]], -1)


def relative(a, b):
    """Poses b (..., 3) in the frame of pose a (both in one frame)."""
    b = np.asarray(b, np.float64)
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[..., 0] - a[0], b[..., 1] - a[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, b[..., 2] - a[2]], -1)


def arcs(P):
    """Cumulative arc length of the poses P (8, 3) from the origin."""
    return np.cumsum(np.hypot(*np.diff(np.vstack([[0.0, 0.0], P[:, :2]]), axis=0).T))


def lat_offset(p, path):
    """Signed (left +) offset of point p from the polyline path (n, 2); the first / last segment extended beyond its end."""
    best = None
    for i in range(len(path) - 1):
        a, u = path[i], path[i + 1] - path[i]
        L = np.hypot(*u)
        if L < 1e-6:
            continue
        u = u / L
        t = float((p - a) @ u)
        tc = t if (i == 0 and t < 0) or (i == len(path) - 2 and t > L) else min(max(t, 0.0), L)
        q = a + tc * u
        dist = np.hypot(*(p - q))
        if best is None or dist < best[0]:
            best = (dist, float(np.sign(u[0] * (p - a)[1] - u[1] * (p - a)[0]) * dist))
    return best[1] if best else np.nan


def p1_load(dirs):
    """drive.jsonl of onp_log_driver runs -> {scene: {k: record}} (inference records with log fields) and the dump files."""
    recs, dumps, seen = {}, [], {}
    for d in dirs:
        for f in sorted(Path(d).rglob("drive.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r.get("kind") != "drive" or not r.get("infer") or "log_poses" not in r:
                    continue
                assert seen.setdefault(r["scene"], r["session"]) == r["session"], f"{r['scene']}: more than one session"
                recs.setdefault(r["scene"], {})[r["k"]] = r
            dumps += sorted((f.parent / "dump").glob("*.npz"))
    return recs, dumps


def p1_replay(tag, recs, dumps, dev):
    """Model poses (8, 3) of checkpoint `tag` at every dumped decision: {(scene, k): poses}."""
    sys.path[:0] = [str(R / "experiments/alpasim/lib")]
    import pai_core as PC
    import sh30_core as C
    core = C.Core(tag, dev)
    out = {}
    for f in dumps:
        with np.load(f) as z:
            sc, k = str(z["scene"]), int(f.stem.split("_k")[1])
            r = recs.get(sc, {}).get(k)
            if r is None or int(z["now"]) != r["now"]:
                continue
            e, cur, valid, cam_t = z["ego"].astype(np.float32), z["cur"], z["valid"], z["cam_t"]
        rf = np.asarray(r["rmem"], np.float32) if core.rm and "rmem" in r else None
        o = PC.plan(core, cur, valid, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.eye(4, dtype=np.float32)[r["cmd"]], cam_t, False,
                    route=rf, ego_fn=lambda _, e=e: e)
        out[(sc, k)] = np.asarray(o["poses"], np.float64)
    return out


def p1_turn(rs):
    """Records {k: rec} of one scene -> the human turn and per-decision arrays, or None (no turn in the clip)."""
    ks = np.array(sorted(rs))
    G = np.array([compose(np.array(rs[k]["anchor"]), np.array(rs[k]["log_now"])) for k in ks])
    last = rs[ks[-1]]
    ext = compose(G[-1], relative(np.array(last["log_now"]), np.array(last["log_poses"])[:last["log_valid"]]))
    path = np.vstack([G, ext])
    path[:, 2] = np.unwrap(path[:, 2])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))]
    grid = np.arange(0.0, s[-1], 1.0)
    psi = np.interp(grid, s, path[:, 2])
    if len(grid) < 31:
        return None
    dw = psi[30:] - psi[:-30]
    hit = np.flatnonzero(np.abs(dw) >= np.radians(45))
    if not len(hit):
        return None
    a = int(hit[0])
    lo, hi = max(a - 10, 0), min(a + 40, len(grid) - 1)
    ch = psi[lo:hi + 1] - psi[lo]
    tot = ch[np.argmax(np.abs(ch))]
    sg = float(np.sign(tot))
    entry = float(grid[lo + np.flatnonzero(sg * ch >= 0.1 * abs(tot))[0]])
    exit_ = float(grid[lo + np.flatnonzero(sg * ch >= 0.9 * abs(tot))[0]])
    sd = s[:len(ks)]
    if entry > sd[-1]:
        return None
    t = ks / 10.0
    t_in, t_out = t[np.argmax(sd >= entry)], (t[np.argmax(sd >= exit_)] if exit_ <= sd[-1] else t[-1])
    H, valid = {}, {}
    for k in ks:
        r = rs[k]
        valid[int(k)] = r["log_valid"] == 8
        H[int(k)] = relative(np.array(r["log_now"]), np.array(r["log_poses"]))
    return dict(sign=sg, deg=float(np.degrees(tot)), entry=entry, exit=exit_, t_entry=float(t_in), t_exit=float(t_out),
                d={int(k): float(entry - x) for k, x in zip(ks, sd)}, t={int(k): float(x) for k, x in zip(ks, t)}, H=H, valid=valid)


def p1_metrics(tu, plans):
    """One turn, the tag's plans {k: poses} -> per-turn metrics and the per-decision trace."""
    sg = tu["sign"]
    ks = sorted(k for k in plans if tu["valid"].get(k))
    rows = []
    for k in ks:
        P, H = plans[k], tu["H"][k]
        ap, ah = arcs(P), arcs(H)
        rows.append([k, tu["d"][k], ap[3], ap[7], ah[3], ah[7], sg * P[3, 1], sg * H[3, 1],
                     sg * lat_offset(P[7, :2], np.vstack([[0.0, 0.0], H[:, :2]])), sg * np.degrees(P[7, 2]), sg * np.degrees(H[7, 2])])
    X = np.array(rows, np.float64).reshape(-1, len(TRACE))
    c = {n: X[:, i] for i, n in enumerate(TRACE)}
    m = {}
    mid = -(tu["exit"] - tu["entry"]) / 2
    for name, D in [*[(str(x), x) for x in P1_DS], ("mid", mid)]:
        if not len(X) or np.abs(c["d"] - D).min() > TOL_D:
            continue
        i = int(np.argmin(np.abs(c["d"] - D)))
        m |= {f"v4p_{name}": c["a4p"][i] / 4, f"v4h_{name}": c["a4h"][i] / 4, f"v2p_{name}": c["a2p"][i] / 2, f"v2h_{name}": c["a2h"][i] / 2,
              f"y2p_{name}": c["y2p"][i], f"y2h_{name}": c["y2h"][i], f"off4_{name}": c["off4"][i]}
        m[f"v4x_{name}"] = m[f"v4p_{name}"] - m[f"v4h_{name}"]
        m[f"y2x_{name}"] = m[f"y2p_{name}"] - m[f"y2h_{name}"]
    for who in "ph":
        if f"v4{who}_0" in m and (f"v4{who}_40" in m or f"v4{who}_20" in m):
            top = max(m.get(f"v4{who}_40", -1), m.get(f"v4{who}_20", -1))
            m[f"drop_{who}"] = top - m[f"v4{who}_0"]
            m[f"slow_{who}"] = float(m[f"drop_{who}"] > 1.0)
    t = np.array([tu["t"][k] for k in ks])
    w = (t >= tu["t_entry"] - PRE_S) & (t <= tu["t_exit"]) if len(ks) else np.zeros(0, bool)
    if w.any():
        q = w & (c["a4h"] >= 2)
        m["v4ratio_win"] = float(np.median(c["a4p"][q] / c["a4h"][q])) if q.any() else np.nan
        q = w & (c["y2h"] >= 0.5)
        if q.any():
            m["y2def_win"] = float(np.median(c["y2h"][q] - c["y2p"][q]))
        o = c["off4"][w]
        m |= dict(off4_win=float(np.median(o)), inside_win=float(np.mean(o > 0.5)), on_win=float(np.mean(np.abs(o) <= 0.5)),
                  outside_win=float(np.mean(o < -0.5)), n_win=int(w.sum()))
    pre = (c["d"] <= COMMIT_D) & (t <= tu["t_exit"]) if len(ks) else np.zeros(0, bool)
    cp = [np.flatnonzero(pre & (c[f"psi4{who}"] >= COMMIT_DEG)) for who in "ph"]
    m["commit_p"] = float(c["d"][cp[0][0]]) if len(cp[0]) else np.nan
    m["commit_h"] = float(c["d"][cp[1][0]]) if len(cp[1]) else np.nan
    m["never_p"] = float(not len(cp[0]))
    first = np.flatnonzero(pre)[0] if pre.any() else -1                # already committed at the first eligible decision: censored
    m["censored"] = float(any(len(x) and x[0] == first for x in cp))
    if len(cp[0]) and len(cp[1]) and not m["censored"]:
        m["lead"] = m["commit_p"] - m["commit_h"]
    return m, X


def p1_classes(f):
    import csv
    out = {}
    for r in csv.DictReader(open(f)):
        if r["label"] in ("rsg", "rbsg2") and r["topo"] and r["scene"] not in out:
            out[r["scene"]] = "straight_ok" if r["topo"] in STRAIGHT_OK else "no_straight" if r["topo"] in NO_STRAIGHT else "unknown"
    return out


P1_KEYS = [("v4p_40", "v4 plan @40"), ("v4h_40", "v4 human @40"), ("v4p_20", "v4 plan @20"), ("v4h_20", "v4 human @20"), ("v4p_0", "v4 plan @0"),
           ("v4h_0", "v4 human @0"), ("v4x_20", "v4 plan - human @20"), ("v4x_0", "v4 plan - human @0"), ("v4x_mid", "v4 plan - human mid-turn"), ("drop_p", "plan drop 40/20 -> 0"),
           ("drop_h", "human drop"), ("slow_p", "plan slows (share)"), ("slow_h", "human slows (share)"), ("v4ratio_win", "arc4 plan / human (window)"),
           ("y2p_20", "lat2 plan @20"), ("y2h_20", "lat2 human @20"), ("y2p_0", "lat2 plan @0"), ("y2h_0", "lat2 human @0"),
           ("y2def_win", "lat2 need - plan (window)"), ("off4_win", "4 s offset, + inside (window)"), ("inside_win", "4 s inside share"),
           ("on_win", "4 s on share"), ("outside_win", "4 s outside share"), ("commit_p", "commit d plan"), ("commit_h", "commit d human"),
           ("lead", "commit lead plan - human (m)"), ("censored", "commit censored (share)"), ("never_p", "plan never commits (share)")]
P1_PAIR = ("v4x_20", "v4x_0", "v4x_mid", "drop_p", "y2def_win", "off4_win", "inside_win", "outside_win", "lead", "never_p")


def p1_groups(per):
    g = dict(all=per, left=[x for x in per if x["sign"] > 0], right=[x for x in per if x["sign"] < 0])
    for c in ("straight_ok", "no_straight", "unknown"):
        g[c] = [x for x in per if x["cls"] == c]
    g["slowing human"] = [x for x in per if x.get("slow_h") == 1.0]
    return g


def p1_summary(per):
    from jevdrive import stats
    out = {}
    for gname, rs in p1_groups(per).items():
        o = dict(turns=len(rs))
        for k, _ in P1_KEYS:
            r = stats.bootstrap([x.get(k, np.nan) for x in rs])
            o[k] = dict(n=r["n"], mean=r["mean"], lo=r["lo"], hi=r["hi"])
        out[gname] = o
    return out


def p1_main(a):
    from jevdrive import stats
    recs, dumps = p1_load(a.p1)
    turns = {sc: tu for sc in sorted(recs) if (tu := p1_turn(recs[sc]))}
    cls = p1_classes(a.classes)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    tags = list(a.tags) + ([f"{a.logged}@log"] if a.logged else [])
    allp, res = {}, {}
    for tag in tags:
        f = o / f"{tag}.json"
        check = None
        if tag.endswith("@log"):
            plans = {(sc, k): np.array(r["poses_model"]) for sc, rs in recs.items() for k, r in rs.items()}
        elif a.reuse and f.exists():
            J = json.loads(f.read_text())
            plans, check = {(sc, int(k)): np.array(v) for sc, d in J["plans"].items() for k, v in d.items()}, J.get("check")
        else:
            import torch
            plans = p1_replay(tag, recs, dumps, torch.device(a.device))
            if a.logged:                                     # every replayed tag against the logged plans (a wrong tag = the control)
                dev = np.array([np.hypot(*(np.array(recs[sc][k]["poses_model"]) - p)[:, :2].T).max() for (sc, k), p in plans.items()])
                check = dict(vs=a.logged, decisions=len(dev), max_m=float(dev.max()), p99_m=float(np.quantile(dev, 0.99)),
                             median_m=float(np.median(dev)), over_1cm=float(np.mean(dev > 0.01)), over_3cm=float(np.mean(dev > 0.03)))
                if tag == a.logged:
                    check["line_1cm"] = "PASS" if dev.max() <= 0.01 else "FAIL"
                    check["line_gid"] = "PASS" if check["over_3cm"] <= 0.01 else "FAIL"     # onp_cache G-id: <= 1 % of decisions > 0.03 m
                print(f"equivalence {tag} vs logged {a.logged}: {check}")
        per, traces = [], {}
        for sc, tu in turns.items():
            pl = {k: p for (s2, k), p in plans.items() if s2 == sc}
            m, X = p1_metrics(tu, pl)
            per.append(dict(scene=sc, sign=tu["sign"], deg=round(tu["deg"], 1), entry=round(tu["entry"], 1), exit=round(tu["exit"], 1),
                            cls=cls.get(sc, "unknown"), decisions=len(X), **m))
            traces[sc] = np.round(X, 3).tolist()
        allp[tag], res[tag] = per, dict(summary=p1_summary(per), check=check)
        if not tag.endswith("@log"):
            J = dict(tag=tag, source=a.p1, check=check, turns=len(per), summary=res[tag]["summary"], per_turn=per, trace_cols=TRACE, traces=traces,
                     plans={sc: {str(k): np.round(p, 4).tolist() for (s2, k), p in plans.items() if s2 == sc} for sc in recs})
            f.write_text(json.dumps(J, indent=None, allow_nan=True))
    ref = a.ref or a.tags[0]
    F = lambda r: "-" if not np.isfinite(r["mean"]) else f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"  # noqa: E731
    md = [f"# PAI P1 open-loop read `{a.name}` (log-following render, {len(recs)} scenes, {len(turns)} with a human turn >= 45 deg / 30 m)\n",
          "Rules: rf_pai_openloop.py docstring (P1 mode). Mean [95 % bootstrap CI over turns] (n turns with the quantity). v4 = 4 s arc / 4 "
          "(m/s); lat2 = 2 s lateral toward the turn (m); stations = distance to the human's turn entry (m).\n"]
    for tag in tags:
        if res[tag]["check"]:
            md.append(f"Equivalence ({tag} replayed vs logged poses_model): {res[tag]['check']}\n")
    groups = list(res[tags[0]]["summary"])
    for gname in groups:
        md.append(f"\n## {gname} ({res[tags[0]]['summary'][gname]['turns']} turns)\n\n| quantity | " + " | ".join(tags) + " |\n|:--|" + "--:|" * len(tags))
        for k, lab in P1_KEYS:
            md.append(f"| {lab} | " + " | ".join(f"{F(res[t]['summary'][gname][k])} ({res[t]['summary'][gname][k]['n']})" for t in tags) + " |")
    pair = []
    for tag in tags:
        if tag == ref:
            continue
        for gname, rs in p1_groups(allp[tag]).items():
            rr = {x["scene"]: x for x in p1_groups(allp[ref])[gname]}
            for k in P1_PAIR:
                b = [rr[x["scene"]].get(k, np.nan) if x["scene"] in rr else np.nan for x in rs]
                r = stats.paired([x.get(k, np.nan) for x in rs], b)
                pair.append(dict(tag=tag, group=gname, key=k, n=r["n"], mean=r["mean"], lo=r["lo"], hi=r["hi"]))
    md.append(f"\n## Paired against {ref} (tag - {ref}, same turns)\n\n| tag | group | " + " | ".join(P1_PAIR) + " |\n|:--|:--|" + "--:|" * len(P1_PAIR))
    for tag in [t for t in tags if t != ref]:
        for gname in groups:
            cells = [next(p for p in pair if p["tag"] == tag and p["group"] == gname and p["key"] == k) for k in P1_PAIR]
            md.append(f"| {tag} | {gname} | " + " | ".join(f"{F(p)} ({p['n']})" for p in cells) + " |")
    md.append("\n## Per turn (" + ", ".join(tags) + ": v4 @0 plan, lat2 deficit, 4 s offset, commit lead)\n\n| scene | side / deg | class | v4 human @0 | "
              + " | ".join(tags) + " |\n|:--|:--|:--|--:|" + "--|" * len(tags))
    g = lambda x, k: "-" if not np.isfinite(x.get(k, np.nan)) else f"{x[k]:.1f}"  # noqa: E731
    for i, x in enumerate(allp[tags[0]]):
        md.append(f"| {x['scene'][7:15]} | {'L' if x['sign'] > 0 else 'R'} {x['deg']:+.0f} | {x['cls']} | {g(x, 'v4h_0')} | " + " | ".join(
            f"{g(allp[t][i], 'v4p_0')} / {g(allp[t][i], 'y2def_win')} / {g(allp[t][i], 'off4_win')} / {g(allp[t][i], 'lead')}" for t in tags) + " |")
    (o / f"{a.name}.md").write_text("\n".join(md) + "\n")
    (o / f"{a.name}_pairs.json").write_text(json.dumps(pair, indent=1))
    print("\n".join(md[:3]))
    if a.figs:
        p1_figs(a, tags, o, turns, allp)


def p1_figs(a, tags, o, turns, allp):
    """<name>_speed.png: v4 vs station; <name>_lat.png: lat2 vs station; columns = map class; median over turns of per-turn 5 m bin means."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    traces = {t: json.loads((o / f"{t}.json").read_text())["traces"] for t in tags if not t.endswith("@log")}
    edges = np.arange(-30, 65, 5.0)
    mids = (edges[:-1] + edges[1:]) / 2
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
    cls = {x["scene"]: x["cls"] for x in allp[tags[0]]}
    panels = [("straight_ok", "straight exit drivable"), ("no_straight", "no straight exit"), ("unknown", "class unknown")]
    i_d = TRACE.index("d")

    def curve(tr, col, scenes):
        Y = np.full((len(scenes), len(mids)), np.nan)
        for j, sc in enumerate(scenes):
            X = np.array(tr[sc]).reshape(-1, len(TRACE))
            if not len(X):
                continue
            b = np.digitize(X[:, i_d], edges) - 1
            for q in range(len(mids)):
                if (b == q).any():
                    Y[j, q] = np.mean(col(X[b == q]))
        n = np.isfinite(Y).sum(0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(Y, 0)
        return np.where(n >= 3, med, np.nan), n

    for fname, ylab, colp, colh in [("speed", "speed intent: 4 s arc / 4 (m/s)", lambda X: X[:, 3] / 4, lambda X: X[:, 5] / 4),
                                    ("lat", "lateral toward the turn at 2 s (m)", lambda X: X[:, 6], lambda X: X[:, 7])]:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, (c, title) in zip(axs, panels):
            scenes = [sc for sc in turns if cls[sc] == c]
            t0 = [t for t in traces][0]
            yh, n = curve(traces[t0], colh, scenes)
            ax.plot(mids, yh, color="#0b0b0b", lw=2.4, label="human (log)")
            for j, (t, cc) in enumerate(zip(traces, cols)):
                y, _ = curve(traces[t], colp, scenes)
                ax.plot(mids, y, color=cc, lw=2 if j < 2 else 1.3, ls="-" if j < 2 else "--", marker="o", ms=4 if j < 2 else 2.5, label=t)
            ax.axvline(0, color="#999", lw=1, ls="--")
            ax.set_xlim(60, -30)
            ax.set_title(f"{title} ({len(scenes)} turns)", fontsize=10)
            ax.set_xlabel("distance to the human's turn entry (m)  <- inside the turn")
            ax.grid(alpha=0.25)
        axs[0].set_ylabel(ylab)
        axs[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"PAI P1 open loop on the human log ({a.name}): median over turns of 5 m bin means, bins with >= 3 turns", fontsize=10)
        fig.tight_layout()
        Path(a.figs).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(a.figs) / f"{a.name}_{fname}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache"), ap.add_argument("--route"), ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--p1", nargs="+", help="P1 mode: dirs holding onp_log_driver drive.jsonl + dump/ (searched recursively)")
    ap.add_argument("--logged", help="P1: the tag the render was driven with (equivalence check, <tag>@log)")
    ap.add_argument("--ref", help="P1: tag the others are paired against (default the first)")
    ap.add_argument("--reuse", action="store_true", help="P1: take plans from existing <out>/<tag>.json")
    ap.add_argument("--figs", help="P1: figure dir")
    ap.add_argument("--classes", default=str(R / "experiments/route_follow/results/pai_follow/scenes.csv"))
    ap.add_argument("--exclude", default=str(R / "experiments/alpasim/results/pai_exam/hard_exam.txt"))
    ap.add_argument("--train-scenes", nargs="*", default=[]), ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=str(R / "experiments/route_follow/results/pai_openloop")), ap.add_argument("--name", default="read")
    ap.add_argument("--device", default="cuda"), ap.add_argument("--served-tag", default="", help="the checkpoint that drove the rows")
    a = ap.parse_args()
    if not a.p1 and not (a.cache and a.route and a.scenes):
        ap.error("--cache, --route and --scenes are required without --p1")
    main(a)
