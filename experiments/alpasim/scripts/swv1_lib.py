"""Lane SWV1 shared code (plans/2026-10-10-swv1-prereg.md): rollouts of COL1's extractions as one structure, the planned swept footprint
against recorded object boxes, plan kinematics. Analysis side (numpy + shapely 2), runs on the Mac from tmp/swv1/ (copies of
$DATA_DIR/runs/alpasim/col1/{cases.pkl, lead/ctrl_logs.pkl}, the Tokyo box's col1/x/*/logs.pkl and frame-stripped replays, and
$DATA_DIR/runs/alpasim/swv1/replay/*.pkl). Simulator boxes, logged paths and recorded object motion are labels and oracle inputs only.

Frames: `actors` / `logged` are AABB-centre poses (t_us, x, y, yaw) in the rollout's local frame; plans are rear-axle (rig) poses.
Objects beyond the end of their record continue at their last velocity (amendment 1)."""
import csv
import sys
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import col1_lib as CL  # noqa: E402

TMP = ROOT / "tmp/swv1"
RES = ROOT / "experiments/alpasim/results"
DT = 0.1
TAU = np.arange(0, 41) * DT                               # plan sample times, s
LEAD_BOX = (4.8, 2.0)                                     # nominal vehicle placed at the lead head's output (prereg B2)


def boxes(c, L, W):
    """Centre poses (..., 3) x, y, yaw -> shapely polygons (...)."""
    c = np.asarray(c, float)
    hl, hw = np.broadcast_to(np.asarray(L, float) / 2, c.shape[:-1])[..., None], np.broadcast_to(np.asarray(W, float) / 2, c.shape[:-1])[..., None]
    lx, ly = np.array([1, 1, -1, -1.0]) * hl, np.array([1, -1, -1, 1.0]) * hw
    cs, sn = np.cos(c[..., 2:3]), np.sin(c[..., 2:3])
    return shapely.polygons(np.stack([c[..., 0:1] + lx * cs - ly * sn, c[..., 1:2] + lx * sn + ly * cs], -1))


def compose(anchor, p):
    """Poses p (n, 3) in the frame of anchor (x, y, yaw) -> world."""
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return np.c_[anchor[0] + c * p[:, 0] - s * p[:, 1], anchor[1] + s * p[:, 0] + c * p[:, 1], anchor[2] + p[:, 2]]


def dense(anchor, poses):
    """Anchor (rig pose at t0) + 8 poses at 0.5 s in its frame -> (41, 3) world rig poses at 0.1 s (linear, as the driver serves them)."""
    w = np.r_[np.asarray(anchor, float)[None], compose(np.asarray(anchor, float), np.asarray(poses, float))]
    w[:, 2] = np.unwrap(w[:, 2])
    return np.c_[[np.interp(TAU, np.arange(9) * 0.5, w[:, j]) for j in range(3)]].T


def centre(rig, off):
    return np.c_[rig[:, 0] + off * np.cos(rig[:, 2]), rig[:, 1] + off * np.sin(rig[:, 2]), rig[:, 2]]


class Rollout:
    """One rollout: ego, objects (with constant-velocity continuation), decisions with their served plans."""

    def __init__(self, o, dom, st, scene, grp, replay=None):
        self.o, self.dom, self.set, self.scene, self.grp = o, dom, st, scene, grp
        self.log = "_".join(scene.split("_")[:2]) if dom == "nuplan" else scene
        self.off = CL.center_off(o)
        self.Le, self.We = o["size"]["EGO"][:2]
        self.ego = o["actors"]["EGO"]
        self.T0, self.T1 = self.ego[0, 0], self.ego[-1, 0]
        gt = o["logged"][0]["traj"]
        self.gt_rig = np.c_[gt[:, 0], CL.to_rig(gt, self.off)]
        xy = self.gt_rig[:, 1:3]
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-3]
        xy = xy[keep]
        arc = np.hypot(*np.diff(xy, axis=0).T).sum() if len(xy) > 1 else 0.0
        h = np.arctan2(*(xy[-1] - xy[np.searchsorted(np.cumsum(np.hypot(*np.diff(xy, axis=0).T)), arc - 1.0)])[::-1]) if arc > 1.0 else self.gt_rig[0, 3]
        base = xy if arc > 1.0 else xy[:1]
        self.path = np.r_[base, (base[-1] + 40.0 * np.array([np.cos(h), np.sin(h)]))[None]]   # the logged path, continued 40 m straight (COL1)
        self.ids = [k for k in o["actors"] if k != "EGO"]
        self.t_ev = CL.first_event(o, "collision_at_fault") if grp == "C" else None
        self.dec = [d for d in o["drive"] if "traj" in d and len(d["traj"]) == 41 and d["now"] >= self.T0 + o.get("force_gt_us", 0)]
        self.now = np.array([d["now"] for d in self.dec], float)
        self.plan = np.array([np.c_[d["traj"][:, 1:3], np.unwrap(d["traj"][:, 3])] for d in self.dec]).reshape(-1, 41, 3)      # served, rig, world
        self.rp = replay
        self.struck = self.kind = None
        self._obj = {}

    # ---- kinematics of recorded things
    def ego_c(self, t):
        return CL.interp(self.ego, t)

    def ego_rig(self, t):
        return CL.to_rig(self.ego_c(t), self.off)

    def ego_v(self, t):
        return CL.speed(self.ego, t)

    def obj_pose(self, k, t):
        """Object k at times t (us) -> (n, 3) centre poses and a mask (False before the track exists)."""
        if k not in self._obj:
            tr = self.o["actors"][k]
            j = max(np.searchsorted(tr[:, 0], tr[-1, 0] - 0.5e6, "right") - 1, 0)
            dt = max((tr[-1, 0] - tr[j, 0]) * 1e-6, 1e-3)
            v = (tr[-1, 1:3] - tr[j, 1:3]) / dt if len(tr) > 1 and j < len(tr) - 1 else np.zeros(2)
            self._obj[k] = (tr, v if np.hypot(*v) >= 0.3 else np.zeros(2))
        tr, v = self._obj[k]
        t = np.atleast_1d(t).astype(float)
        p = CL.interp(tr, t)
        ex = np.maximum(t - tr[-1, 0], 0)[:, None] * 1e-6
        p[:, :2] += ex * v
        return p, t >= tr[0, 0] - 1e5

    def obj_v(self, k, t):
        return CL.speed(self.o["actors"][k], t)

    def size(self, k):
        return self.o["size"][k][:2]

    # ---- the swept footprint of a plan
    def sweep(self, k_dec, plan=None, ids=None, frozen=False, horizon=None):
        """Ego box along a plan (41, 3 rig, world; default the served plan of decision k_dec), time-aligned against object boxes.
        -> {id: (clr, first intersecting sample index or -1)} for objects that come within 6 m; frozen: objects held at the decision time;
        horizon: only plan samples up to that sim time (us)."""
        plan = self.plan[k_dec] if plan is None else plan
        t = self.now[k_dec] + TAU * 1e6
        n = 41 if horizon is None else int(np.clip(np.searchsorted(t, horizon, "right"), 1, 41))
        ec = centre(plan[:n], self.off)
        eb = boxes(ec, self.Le, self.We)
        out = {}
        for k in (self.ids if ids is None else ids):
            p, ok = self.obj_pose(k, np.full(n, self.now[k_dec]) if frozen else t[:n])
            L, W = self.size(k)
            near = ok & (np.hypot(*(p[:, :2] - ec[:, :2]).T) < (self.Le + L) / 2 + 6.0)
            if not near.any():
                continue
            d = shapely.distance(eb[near], boxes(p[near], L, W))
            hit = np.flatnonzero(near)[d <= 0]
            out[k] = (float(d.min()), int(hit[0]) if len(hit) else -1)
        return out

    def shortfall(self, k_dec, k, plan=None, step=0.1, lim=3.0):
        """Smallest rigid lateral shift (signed, left positive) of the plan after which its sweep clears object k; nan if none within lim."""
        plan = self.plan[k_dec] if plan is None else plan
        nrm = np.c_[-np.sin(plan[:, 2]), np.cos(plan[:, 2])]
        best = np.nan
        for a in np.arange(step, lim + 1e-6, step):
            for sgn in (1, -1):
                q = plan.copy()
                q[:, :2] += sgn * a * nrm
                r = self.sweep(k_dec, q, [k])
                if k not in r or r[k][1] < 0:
                    if not abs(best) <= a:
                        best = sgn * a
            if np.isfinite(best):
                break
        return best

    def lat(self, rig_xy):
        """Signed lateral offset from the logged path and arc position along it, (n, 2)."""
        return CL.lat_to_path(self.path, rig_xy)


def plan_kin(plan):
    """(41, 3) -> speeds per 0.5 s segment (8,), curvature per segment (8,), max lateral acceleration, 4 s heading change, arc."""
    kn = plan[::5]
    ds = np.hypot(*np.diff(kn[:, :2], axis=0).T)
    v = ds / 0.5
    kap = np.where(ds > 0.2, np.diff(kn[:, 2]) / np.maximum(ds, 0.2), 0.0)
    return v, kap, float(np.max(v**2 * np.abs(kap))), float(kn[-1, 2] - kn[0, 2]), float(ds.sum())


# ---------------------------------------------------------------- loading
def nuplan(sets=("P2H10-F-s0", "P2H10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1"), ctrl=True, replay=True):
    C = CL.load(TMP / "nuplan/cases.pkl")
    tax = {(r["driver"], r["scene"]): r for r in csv.DictReader(open(RES / "collisions/nuplan_cases.csv"))}
    R, rep = [], {}
    for g in list(sets) + (["ctrl"] if ctrl else []):
        f = TMP / f"nuplan/replay/{g}.pkl"
        rep[g] = CL.load(f) if replay and f.exists() else {}
    for (drv, scene), o in sorted(C.items()):
        if drv not in sets:
            continue
        r = Rollout(o, "nuplan", drv, scene, "C", rep[drv].get(scene))
        tx = tax[drv, scene]
        r.struck, r.kind, r.tax = tx["obj"], tx["kind"], tx
        R.append(r)
    if ctrl:
        for scene, o in sorted(CL.load(TMP / "nuplan/ctrl_logs.pkl").items()):
            if not o["summary"]["metrics"].get("collision_any"):
                R.append(Rollout(o, "nuplan", "ctrl", scene, "N", rep["ctrl"].get(scene)))
    return R


def pai(arm="base"):
    dirs = ("a10", "b1", "b2a", "b2b") if arm == "base" else ("v1", "v1_b1", "v1_b2a", "v1_b2b")
    tax = {r["scene"]: r for r in csv.DictReader(open(RES / "collisions/pai_cases.csv"))} if arm == "base" else {}
    R = []
    for d in dirs:
        for scene, o in sorted(CL.load(TMP / f"pai/{d}/logs.pkl").items()):
            f = TMP / f"pai_rep/{d}/{scene}.npz"
            m = o["summary"]["metrics"]
            grp = "C" if o["summary"]["score_metrics"].get("collision_at_fault") else "N" if not m.get("collision_any") else "X"
            r = Rollout(o, "pai", f"pai-{arm}", scene[7:15], grp, dict(np.load(f)) if f.exists() else None)
            r.run = d
            if grp == "C":
                r.struck = struck(r)
                import col1_pai as CP
                r.tax = tax.get(r.scene)
                r.kind = CP.classify(o, r.struck, r.t_ev, r.off, r.gt_rig)[0]         # COL1's struck-object class (also for the re-timed arm)
            R.append(r)
    return R


def struck(r):
    """The object nearest to the ego box around the first at-fault step (COL1's rule)."""
    best = (None, 1e9)
    for dt in (0, -1e5, 1e5):
        t = r.t_ev + dt
        e = boxes(r.ego_c(t), r.Le, r.We)[0]
        for k in r.ids:
            p, ok = r.obj_pose(k, t)
            if ok[0] and r.o["actors"][k][-1, 0] >= t - 1e5:
                d = e.distance(boxes(p, *r.size(k))[0])
                if d < best[1]:
                    best = (k, d)
    return best[0]


def boot_ci(fn, groups, n=2000, seed=0):
    """Cluster bootstrap: fn(index array) -> statistic; groups = cluster label per row. -> (point, lo, hi)."""
    groups = np.asarray(groups)
    u = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in u}
    rng = np.random.default_rng(seed)
    pt = fn(np.arange(len(groups)))
    bs = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(u, len(u))])
        v = fn(s)
        if np.isfinite(v):
            bs.append(v)
    return (float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float(pt), np.nan, np.nan)


def wauc(y, s, w=None):
    """Weighted AUC of score s for label y (ties count half)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    o = np.argsort(s, kind="mergesort")
    s, y, w = s[o], y[o], w[o]
    wn = np.where(~y, w, 0.0)
    cn = np.cumsum(wn)
    _, first, inv = np.unique(s, return_index=True, return_inverse=True)
    below = np.r_[0, cn][first][inv]                       # negative weight strictly below each score
    tie = np.bincount(inv, wn)[inv]
    return float((w[y] * (below[y] + 0.5 * tie[y])).sum() / (w[y].sum() * wn.sum()))
