"""CORR0 map labels (experiments/corridor, plans/2026-10-10-corr0-prereg.md). CPU only; envs/navsim2 (nuPlan map layers).
The map, the logged future and the driven lane sequence are privileged: analysis only, never a method input.

  $DATA_DIR/envs/navsim2/bin/python experiments/corridor/scripts/corr_geom.py build  [--tokens f.txt]   navtest: driven lane sequence,
        corridor centreline R (ego frame), half widths, lane-change blend length, exit pose, plan end class per SH30 seed
        -> $DATA_DIR/runs/corridor/geom/<log>.pkl (jevdrive.cache, one unit per log) + geom.pkl (merged)
  ... corr_geom.py supply [--nlogs N]      navtrain: paths within 40 m on the lane graph, exit headings, the driven one (number 4)
        -> $DATA_DIR/runs/corridor/supply/rows.parquet
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/corridor/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(D / "datasets/navsim/maps"))
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import lanegraph as LG  # noqa: E402

OUT = D / "runs/corridor"
LOGS = D / "datasets/navsim/navsim_logs"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
PLANS = D / "runs/op_parity/pt_swap/poses.npz"          # decision 207's build: sh0_pp / sh1_pp are the stored SH30 plans
SEEDS = ("sh0", "sh1")
NF, AHEAD, BLEND, WINDOW = 20, 90.0, 8.0, 40.0
_J = {}


def log_frames(split, log):
    fr = pickle.load(open(LOGS / split / f"{log}.pkl", "rb"))
    xy = np.array([f["ego2global_translation"][:2] for f in fr], np.float64)
    yaw = np.array([LG.yaw_of(f["ego2global_rotation"]) for f in fr])
    return fr, xy, yaw


def to_ego(pts, o, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(pts, np.float64) - o
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def to_map(p, o, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([o[0] + c * p[:, 0] - s * p[:, 1], o[1] + s * p[:, 0] + c * p[:, 1]], 1)


class Matcher:
    """Candidates per log frame (cached) + the driven sequence of a token."""

    def __init__(self, g, xy, yaw):
        self.g, self.xy, self.yaw, self.c = g, xy, yaw, {}

    def cand(self, j):
        if j not in self.c:
            self.c[j] = self.g.candidates(self.xy[j], self.yaw[j])
        return self.c[j]

    def driven(self, i):
        """-> (status, [(frame offset, node)], gap frames in 0-4 s). Tries the 10 s window, then the 4 s one. Amendment A: frames
        without a candidate are skipped; a token fails when t0 has none, fewer than 5 of the 9 poses of 0-4 s have one, or the
        frames that have one admit no connected sequence."""
        n = len(self.xy)
        if not self.cand(i):
            return "no_candidate_t0", None, 0
        near = [k for k in range(i, min(n - 1, i + 8) + 1) if self.cand(k)]
        gap = min(n - 1, i + 8) - i + 1 - len(near)
        if len(near) < min(5, n - i):
            return "no_candidate_4s", None, gap
        for m in (NF, 8):
            fs = [k for k in range(i, min(n - 1, i + m) + 1) if self.cand(k)]
            p = self.g.viterbi([self.cand(k) for k in fs])
            if p is not None:
                return "ok", [(k - i, nd) for k, nd in zip(fs, p)], gap
        return "no_connected_sequence", None, gap


def geom_log(log):
    fr, xy, yaw = log_frames("test", log)
    g = LG.get(fr[0]["map_location"])
    M = Matcher(g, xy, yaw)
    pos = {f["token"]: i for i, f in enumerate(fr)}
    out = []
    for tok, plans in _J[log]:
        i = pos[tok]
        r = dict(token=tok, log=log, loc=g.loc, o=np.r_[xy[i], yaw[i]])
        st, fpath, gap = M.driven(i)
        r["status"], r["gap"] = st, gap
        if fpath is None:
            out.append(r)
            continue
        path, used = [nd for _, nd in fpath], fpath[-1][0]
        seq, lc = g.sequence(path)
        run = seq[lc:]
        Rxy, used_nodes, own = g.centreline(run, xy[i], AHEAD)
        hw = g.half_width(Rxy, own)
        Re = to_ego(Rxy, xy[i], yaw[i])
        line = LG.Line(Re)
        d0 = float(line.frenet(np.zeros((1, 2)))[1][0])        # ego left of the centreline: positive
        blend = BLEND
        if lc > 0:                                   # arc length the log needs to reach the post-change lane run
            f = next(k for k, nd in fpath if nd in run)
            blend = max(BLEND, float(np.hypot(*np.diff(xy[i:i + f + 1], axis=0).T).sum()))
        conn = [k for k in seq if g.kind[k] == 1]
        r.update(seq=[g.ids[k] for k in seq], lane_change=lc > 0, R=Re.astype(np.float32), hw=hw.astype(np.float32), d0=d0, blend=blend,
                 frames=used, n_conn=len(conn), start_kind=g.kind[seq[0]])
        if conn:
            c0 = g.line[conn[0]]
            r["exit_h"] = float(LG.wrap(c0.h[-1] - yaw[i]))
            r["exit_xy"] = to_ego(c0.xy[-1:], xy[i], yaw[i])[0]
        # plan end class (number 1): chain = driven sequence + centreline extension + 3 hops below the last driven node
        chain = set(seq) | set(used_nodes) | {k for k, v in g.reach(seq[-1]).items() if not v[1]}
        groups = {g.group[k] for k in chain}
        for sd, p8 in zip(SEEDS, plans):
            P = np.vstack([np.zeros(3), p8]).astype(np.float64)
            Pm, Py = to_map(P[:, :2], xy[i], yaw[i]), P[:, 2] + yaw[i]
            cs = [g.candidates(Pm[k], Py[k]) for k in range(9)]
            has = [k for k in range(9) if cs[k]]
            if not has:
                r[f"{sd}_cls"] = r[f"{sd}_cls2"] = "unmatched"
                continue
            vp = g.viterbi([cs[k] for k in has])
            end = vp[-1] if vp is not None else min(cs[has[-1]], key=lambda c: c[1])[0]
            r[f"{sd}_connected"], r[f"{sd}_end"], r[f"{sd}_endgap"] = vp is not None, g.ids[end], 8 - has[-1]
            if end in chain:
                c2 = "same"
            elif any(c in chain for c, _ in cs[has[-1]]):
                c2 = "same_amb"
            else:
                c2 = "a2" if g.group[end] in groups else "a1"
            r[f"{sd}_cls2"] = c2                                   # class of the last pose that has a candidate (amendment A)
            r[f"{sd}_cls"] = c2 if has[-1] == 8 else "unmatched"    # registered: the 4 s pose itself
        out.append(r)
    return out


def cmd_build(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "geom", config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        tab, Z = np.load(TAB), np.load(PLANS)
        names, logs = tab["names"].astype(str), tab["log"].astype(str)
        assert (Z["tokens"].astype(str) == names).all()
        P = np.stack([Z[f"{s}_pp"] for s in SEEDS], 1)
        want = set(open(a.tokens).read().split()) if a.tokens else set(names)
        for i, t in enumerate(names):
            if t in want:
                _J.setdefault(logs[i], []).append((t, P[i]))
        res = par.pmap(geom_log, sorted(_J), run=run, desc="geom logs")
        res.raise_if_failed()
        rows = [r for v in res.values for r in v]
        dst = OUT / ("geom.pkl" if not a.tokens else f"geom_{Path(a.tokens).stem}.pkl")
        dst.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(rows, open(dst, "wb"))
        dyaw = dict(zip(names, np.degrees(np.abs(tab["fut"][:, -1, 2]))))
        st = {}
        for r in rows:
            b = "all" if dyaw[r["token"]] <= 20 else ">20"
            st.setdefault(b, {}).setdefault(r["status"], 0)
            st[b][r["status"]] += 1
        run.summary.update(tokens=len(rows), status=st, out=str(dst))
        run.info(str(st))


# ---------------------------------------------------------------- number 4: navtrain supply
def supply_log(log):
    fr, xy, yaw = log_frames("trainval", log)
    g = LG.get(fr[0]["map_location"])
    M = Matcher(g, xy, yaw)
    want, n = _J["tokens"], len(fr)
    rows = []
    for i, f in enumerate(fr):
        if f["token"] not in want:
            continue
        r = dict(token=f["token"], log=log, loc=g.loc, v=float(np.hypot(*f["ego_dynamic_state"][:2])),
                 dpsi=float(np.degrees(LG.wrap(yaw[i + 8] - yaw[i]))) if i + 8 < n else np.nan)
        st, fpath, _ = M.driven(i)
        r["status"] = st
        if fpath is None:
            rows.append(r)
            continue
        seq, lc = g.sequence([nd for _, nd in fpath])
        s0 = float(g.line[seq[0]].frenet(xy[i][None])[0][0])
        route = {str(x) for x in (f.get("roadblock_ids") or [])}

        def paths(start, s):
            d = {}
            for p, cum in g.paths_ahead(start, s, WINDOW):
                d.setdefault(p[-1], (p, g.heading_at(p, s, WINDOW)))
            return list(d.values())

        own = paths(seq[0], s0)
        drv = [(p, h) for p, h in own if all(seq[k] == p[k] for k in range(min(len(seq), len(p))))]
        if len(drv) > 1 and route:
            onr = [(p, h) for p, h in drv if all(g.group[k] in route for k in p)]
            drv = onr or drv
        r.update(n_paths=len(own), n_driven=len(drv), lane_change=lc > 0, start_kind=g.kind[seq[0]], node=g.ids[seq[0]])
        br = next((k for k in (drv[0][0] if drv else own[0][0]) if len(g.succ[k]) > 1), None)
        r["branch"] = g.ids[br] if br is not None else ""
        if drv:
            dh = np.array([h for _, h in drv])
            r["alt"] = [float(np.degrees(np.abs(LG.wrap(h - dh)).min())) for p, h in own if all(p is not q for q, _ in drv)]
            # secondary: exits of any lane of the ego's roadblock (lane group)
            if g.kind[seq[0]] == 0:
                grp = [k for k in range(len(g.ids)) if g.kind[k] == 0 and g.group[k] == g.group[seq[0]] and k != seq[0]]
                rb = [x for k in grp for x in paths(k, min(float(g.line[k].frenet(xy[i][None])[0][0]), g.line[k].L))]
                r["alt_rb"] = r["alt"] + [float(np.degrees(np.abs(LG.wrap(h - dh)).min())) for _, h in rb]
            else:
                r["alt_rb"] = r["alt"]
        rows.append(r)
    return rows


def cmd_supply(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "supply", config=vars(a)) as run:
        sp = splits.load("navsim/navtrain")
        run.use_split(sp)
        _J["tokens"] = set(sp.members)
        import yaml
        nav = os.environ.get("NAVSIM_DEVKIT", os.path.expanduser("~/data/third_party/navsim"))
        cfg = yaml.safe_load(open(f"{nav}/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"))
        logs = sorted(l for l in cfg["log_names"] if (LOGS / "trainval" / f"{l}.pkl").exists())
        if a.nlogs:
            logs = logs[:: max(1, len(logs) // a.nlogs)][:a.nlogs]
        res = par.pmap(supply_log, logs, run=run, desc="supply logs")
        res.raise_if_failed()
        df = pd.DataFrame([r for v in res.values for r in v])
        dst = OUT / "supply" / ("rows.parquet" if not a.nlogs else f"rows_{a.nlogs}.parquet")
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst)
        run.summary.update(logs=len(logs), tokens=len(df), status=df.status.value_counts().to_dict(), out=str(dst))
        run.info(str(run.summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--tokens", default="")
    s = sub.add_parser("supply"); s.add_argument("--nlogs", type=int, default=0)
    a = ap.parse_args()
    {"build": cmd_build, "supply": cmd_supply}[a.cmd](a)
