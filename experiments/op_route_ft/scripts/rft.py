"""op_route_ft: route-choice fine-tune of openpilot Cinque (plans/2026-10-05-route-ft-prereg.md), op-train venv, one GPU per arm.

The port, trainable set and recipe of the layer-3 line (experiments/op_adapt_h, it_dw3: stage 1-3 frozen as a trunk bank, stage 4 + plan pathway
trainable, every output head distilled to the original, dw 3) plus
  * lib/route_adapter.RouteAdapter: the route (noised navigation polyline) -> bias on the hidden tokens of the 9 context frames;
  * the on-policy action pathway (ONNX nodes 665-830, `action` = desired lateral acceleration, the closed-loop lateral source, decision 118)
    trainable and supervised: action[0] -> -0.45 kappa_T * max(1, v0)^2, kappa_T = pure-pursuit curvature (left +) of the target at 1 s; 0.45 and
    the sign are the shipped head's own relation to logged curvature (ACT_ALPHA).

Rows of one batch (48):
  P  positives with the route: real (op_adapt_H nav / wod pools, target = logged future) and CARLA exit pairs (route_carla packed set, target = the
     exit polyline timed by the original's own plan, speed capped at 3 m/s^2 lateral). Sampling weight by turn class (T2): straight 1, 25-60 deg 1.5,
     >= 60 deg 2, >= 60 deg with R_min < 15 m 4. Non-plan / non-action heads distilled.
  N  negatives (T3, ~12%): CARLA N1 (an exit class the approach road lacks, map-certain), real N3 (wrong side) / N4 (mid-block U-turn) from
     lib/route_neg.make; target = the original's plan + every head (teacher = original, as q3NA).
  D  no command: plan consistency + every head to the original (nav / wod / carla H pools, CARLA pair poses).
Arms: rc-bear (bear encoding), rc-poly (poly encoding), rc-ctl (command zeroed on every row, same rows and targets), rc-all (rc-bear + op_adapt_L
start / stop slices with stay contrast (T4) + layer-3 H / O pairs (T5) + nuPlan drivable hinge on nav rows (T6); exploratory).

  bank   --root <carla packed root> --tag <tag>     stage-3 trunks + the original's outputs per CARLA pose -> $R/bank/<tag>/
  train  --arm <arm> [--steps n] [--tag t]           -> $R/runs/<arm>-s<seed>/ (Run dir, ckpt-final.pt in op_adapt_l format, adapter.npz)
  evalol --models O rc-bear-s0 ...                   open-loop readouts -> $R/evalol/<model>.json
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib")]
import argparse, json, time  # noqa: E401,E402
from dataclasses import asdict, dataclass, field, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import route_adapter as RA  # noqa: E402
import route_neg as RN  # noqa: E402
import route_poly as RP  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402
from experiments.op_adapt_h.lib import op_adapt_h as H  # noqa: E402

T16 = L.T16
ACT_NODES = (665, 831)            # cinque.ort.onnx: the on-policy pathway that ends in `action` (mul_48, node 829)
SIG_A = 0.5                       # m/s^2, scale of the action loss
ACT_ALPHA = 0.45                  # action[0] = -ACT_ALPHA * kappa_T * max(1, v)^2: the shipped head's own convention (sign: right +), fitted on the
                                  # op_adapt_H teachers at v0 > 3 m/s against the logged 1 s pure-pursuit curvature (slope nav -0.51, wod -0.39, r -0.92)
ALAT_CAP = 3.0                    # m/s^2, speed cap of the CARLA targets on the exit's curvature
ROUTE_NPZ = {"nav": "processed/op_route_cmd/navtrain/route.npz", "wod": "processed/op_route_cmd/wod/route.npz"}
CARLA_ROOTS = {"ol": "runs/op_route_cmd/carla_pairs_s10000ol/packed", "old": "runs/op_route_cmd/carla_pairs_s10000/packed"}
CARLA_CAM = {"ol": 1.59, "old": 1.519}


def rroot(*p) -> Path:
    d = Path(_os.environ.get("OP_RFT_ROOT") or data_dir() / "runs" / "op_route_ft") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- model
def act_weights(model="cinque") -> list:
    import onnx
    g = onnx.load(str(A.MODELS_DIR / A.FILES[model]), load_external_data=False).graph
    fl = {t.name for t in g.initializer if t.data_type in (1, 10, 11, 16) and len(t.dims) >= 1}
    return sorted({i for n in g.node[ACT_NODES[0]:ACT_NODES[1]] for i in n.input if i in fl})


class RModel(L.LModel):
    """op_adapt_l's port with stage 4 + plan pathway + action pathway trainable and a RouteAdapter on the hidden tokens."""

    def __init__(self, cfg: L.LCfg, enc: str | None, act: bool = True, dtype=torch.float16):
        nn.Module.__init__(self)
        self.cfg = cfg
        tr = sorted(set((A.stage4_weights() if cfg.s4 else []) + (L.pol_weights() if cfg.pol else []) + (act_weights() if act else [])))
        self.net = A.load("cinque", dtype, trainable=tr)
        self.adapter = None
        self.route = RA.RouteAdapter(enc) if enc else None
        self.s4 = cfg.s4

    def policy(self, H, valid, tc, feat=None, action_t=L.AT):
        B = H.shape[0]
        if self.route is not None and feat is not None:
            H = self.route.apply(H, feat)
        H = H * valid[:, :, None, None].to(H.dtype)
        o = self.net.run_batched(A.policy_feeds(self.net, H, action_t, tc), A.POLICY_OUT)
        return {"outputs": o["outputs"].reshape(B, -1)}

    def forward(self, trunk, valid, tc, feat=None):
        return self.policy(self.stage4(trunk), valid, tc.to(self.net.dtype), feat)

    def trainable(self):
        base = [p for p in self.net.params.values() if p.requires_grad]
        return base, (list(self.route.parameters()) if self.route is not None else [])

    def state(self) -> dict:
        return {"net": {k: p.detach().cpu() for k, p in self.net.params.items() if p.requires_grad}, "adapter": None,
                "route": self.route.state_dict() if self.route is not None else None}

    def load_state(self, st):
        for k, v in st["net"].items():
            self.net.params[k].data.copy_(v)
        if self.route is not None and st.get("route") is not None:
            self.route.load_state_dict(st["route"])


def load_rmodel(tag, dev):
    """'O' -> the original (no adapter); else a run tag under $R/runs."""
    if tag == "O":
        return RModel(L.LCfg("O", s4=False, intent="none"), None, act=False).to(dev).eval()
    ck = torch.load(rroot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
    c = ck["rcfg"]
    m = RModel(L.LCfg(tag, intent="none"), c["enc"]).to(dev).eval()
    m.load_state(ck["model"])
    m.zero_cmd = c["zero_cmd"]
    return m


# ---------------------------------------------------------------- targets
def plan_arc(mu, cam):
    """original plan (33, 15) -> arc length (16,) and speed (16,) at T16 (rear axle)."""
    p = L.rear_np(mu[None], cam, np.r_[0.0, T16])[0].astype(np.float64)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return s[1:]


def path_target(poly, pmask, mu, cam):
    """CARLA positive: exit polyline (10 m vertices) -> (16, 3) rear-axle x, y, speed at T16: smoothed path, timed by the original's arc length
    with the speed capped at ALAT_CAP on the path's curvature (within the next 60 m)."""
    v = np.asarray(poly, float)[: int(pmask.sum())]
    P = RP.smooth_path(v, 0.5, 1.5)
    s_tea = plan_arc(mu, cam)
    if P is None:
        return None
    P[0] = 0.0
    P, g = RP.poly_resample(P, 0.5)
    k = np.abs(RP.curvature(P, 0.5))
    kmax = k[: int(60 / 0.5)].max() if len(k) else 0.0
    vt = np.diff(np.r_[0.0, s_tea]) / 0.25
    vt = np.minimum(vt, np.sqrt(ALAT_CAP / max(kmax, 1e-3)))
    s = np.cumsum(vt * 0.25)
    s = np.minimum(s, g[-1])
    xy = np.stack([np.interp(s, g, P[:, c]) for c in range(2)], -1)
    return np.concatenate([xy, vt[:, None]], -1).astype(np.float32)


def act_target(hum, v0):
    """(16, 3) target -> (target action[0], weight): pure-pursuit curvature to the 1 s point times max(1, v0)^2; weight 0 below 1 m/s or < 2 m."""
    x, y = float(hum[3, 0]), float(hum[3, 1])
    d2 = x * x + y * y
    if v0 < 1.0 or d2 < 4.0 or not np.isfinite(d2):
        return 0.0, 0.0
    return -ACT_ALPHA * 2.0 * y / d2 * max(1.0, v0) ** 2, 1.0


def turn_weight(deg, rmin, s):
    a = abs(deg) if np.isfinite(deg) and (not np.isfinite(s) or s <= 80.0) else 0.0
    if a < 25:
        return 1.0
    if a < 60:
        return 1.5
    return 4.0 if (np.isfinite(rmin) and rmin < 15.0) else 2.0


def dense_from_poly(poly, pmask):
    v = np.asarray(poly, float)[: int(np.asarray(pmask).sum())]
    if len(v) < 2:
        return None
    return RP.poly_resample(v, 0.5)[0]


# ---------------------------------------------------------------- data
@dataclass
class RCfg:
    name: str
    enc: str | None = "bear"              # bear | poly
    zero_cmd: bool = False                # rc-ctl: command zeroed on every row
    carla: str = "ol"                     # ol (open-loop-aligned rig, 1.86 m) | old (1.22 m, smoke only)
    seed: int = 0
    steps: int = 4000
    rows: dict = field(default_factory=lambda: {"Pnav": 7, "Pwod": 5, "Pcar": 16, "Ncar": 3, "Nreal": 3, "Dnav": 5, "Dwod": 4, "Dhc": 2, "Dcar": 3})
    lam_i: float = 1.0
    lam_a: float = 1.0
    lam_d: float = 10.0
    lam_c: float = 1.0
    dw: float = 3.0
    lr: float = 3e-5
    lr_new: float = 3e-4
    wd: float = 0.01
    warmup: int = 100
    ckpt_every: int = 500
    workers: int = 12
    t4: bool = False                      # rc-all: op_adapt_L start / stop + stay rows
    t5: bool = False                      # rc-all: layer-3 H / O rows
    t6: float = 0.0                       # rc-all: drivable hinge weight on nav rows
    t6_margin: float = 0.4

    def dump(self):
        return asdict(self)

    def lcfg(self):
        return L.LCfg(name=self.name, seed=self.seed, s4=True, pol=True, intent="none", steps=self.steps, dw=self.dw, lam_d=self.lam_d, lam_c=self.lam_c)


ARMS = {
    "smoke": dict(steps=30, carla="old", ckpt_every=10 ** 9, workers=4),
    "rc-bear": dict(enc="bear"),
    "rc-poly": dict(enc="poly"),
    "rc-ctl": dict(enc="bear", zero_cmd=True),
    "rc-all": dict(enc="bear", t4=True, t5=True, t6=0.3),
    "smoke-all": dict(steps=20, carla="old", ckpt_every=10 ** 9, workers=4, t4=True, t5=True, t6=0.3),
}
ROLE = {"P": 1, "D": 2, "N": 3}


class Carla:
    def __init__(self, which):
        root = data_dir() / CARLA_ROOTS[which]
        with np.load(root / "samples/route_carla/tab.npz", allow_pickle=True) as z:
            self.tab = {k: z[k] for k in z.files}
        with np.load(root / "route.npz", allow_pickle=True) as z:
            self.r = {k: z[k] for k in z.files}
        b = rroot("bank", which)
        self.T = np.load(b / "trunk.npy", mmap_mode="r")
        with np.load(b / "teacher.npz") as z:
            self.tea = {k: z[k] for k in z.files}
        self.cam = CARLA_CAM[which]
        self.n = len(self.tab["id"])


class Real:
    def __init__(self, dom):
        self.S = H.Samples(dom)
        self.B = H.Bank(dom)
        self.normal = np.flatnonzero(self.B.v["kind"] == "normal")
        assert (self.B.v["sample"][self.normal] == np.arange(self.S.n)).all()
        self.route = RP.attach(data_dir() / ROUTE_NPZ[dom], self.S.t["id"]) if dom in ROUTE_NPZ else None


def missing_classes(C: Carla, p):
    have = set(C.r["cmd"][C.rows_of[p]].tolist())
    return [c for c in ("left", "straight", "right") if c not in have]


class Batcher(torch.utils.data.Dataset):
    def __init__(self, cfg: RCfg, split="train", n=10 ** 9):
        self.cfg, self.split, self.n, self.ok = cfg, split, n, False

    def __len__(self):
        return self.n

    def _open(self):
        c = self.cfg
        self.C = Carla(c.carla)
        C = self.C
        C.rows_of = {}
        for j, p in enumerate(C.r["pose_row"]):
            C.rows_of.setdefault(int(p), []).append(j)
        C.rows_of = {k: np.array(v) for k, v in C.rows_of.items()}
        self.real = {d: Real(d) for d in ("nav", "wod")}
        self.hc = Real("carla")
        tr = lambda R: R.S.rows(self.split)  # noqa: E731
        self.pool = {}
        for d, R in self.real.items():
            rows = tr(R)
            rows = rows[R.route["has_route"][rows] & (R.route["pmask"][rows].sum(1) >= 3)]
            w = np.array([turn_weight(R.route["turn_deg"][i], R.route["turn_rmin"][i], R.route["turn_s"][i]) for i in rows])
            self.pool[f"P{d}"] = (rows, w / w.sum())
            self.pool[f"D{d}"] = (tr(R), None)
            nr = rows[(R.route["plen"][rows] >= 60) & (R.S.t["v0"][rows] >= 2.0)]
            self.pool[f"N{d}"] = (nr, None)
        self.pool["Dhc"] = (tr(self.hc), None)
        cr = np.flatnonzero((C.r["split"] == self.split) & (C.r["pmask"].sum(1) >= 3))
        w = np.array([turn_weight(C.r["angle"][j], C.r["turn_rmin"][j], 0.0) for j in cr])
        self.pool["Pcar"] = (cr, w / w.sum())
        cp = np.flatnonzero(C.tab["split"] == self.split)
        self.pool["Dcar"] = (cp, None)
        self.pool["Ncar"] = (np.array([p for p in cp if p in C.rows_of and missing_classes(C, p) and C.tab["profile"][p] != "stopped"]), None)
        if c.t6:
            with np.load(data_dir() / "runs/op_route_ft/drivable/nav.npz") as z:
                self.sdf = {k: z[k] for k in z.files}
            pos = {k: i for i, k in enumerate(self.sdf["id"].tolist())}
            self.sdf_row = np.array([pos.get(str(i), -1) for i in self.real["nav"].S.t["id"]])
        if c.t4:
            self.LD = L.Data(("wod", "nus"))
            self.LM = L.Mixer(L.LCfg("t4", slices=("start", "stop"), contrast=True, contrast_mix=(1.0, 0.0, 0.0), n_imit=10, n_contrast=6, n_other=2,
                                     n_nus=2), self.LD, self.split)
        if c.t5:
            self.HB = H.Batcher(H.HCfg("t5", seed=c.seed, roles={"H": 8, "O": 6}, dw=3.0), self.split)
        self.ok = True

    # one row -> dict
    def _real(self, d, i, role, rng):
        R = self.real[d] if d in self.real else self.hc
        t = R.S.t
        j = R.normal[i]
        cam = float(t["cam"][i][0])
        hum = L.human_targets(t["fut20"][i][None])[0]
        out = dict(trunk=R.B.T[j], valid=R.B.v["slot_valid"][j], tc=t["tc"][i], cam=np.float32(cam), tgt=R.S.tea["out"][i], tmu=R.S.tea["mu"][i],
                   hum=hum, role=ROLE[role[0]], fb=np.zeros(RA.ENC_DIM["bear"], np.float32), fp=np.zeros(RA.ENC_DIM["poly"], np.float32),
                   at=np.float32(0), aw=np.float32(0), sdf=np.int64(-1))
        if role[0] == "P":
            rt = R.route
            out["fb"] = RA.features("bear", rt["poly"][i], rt["pmask"][i], np.random.default_rng(rng.integers(1 << 62)))
            out["fp"] = RA.features("poly", rt["poly"][i], rt["pmask"][i], np.random.default_rng(rng.integers(1 << 62)))
            a, w = act_target(hum, float(t["v0"][i]))
            out["at"], out["aw"] = np.float32(a), np.float32(w)
        if role[0] == "N":
            rt = R.route
            P = dense_from_poly(rt["poly"][i], rt["pmask"][i])
            neg = None
            for kind in (("N3_wrong", "N4_uturn") if rng.random() < 0.5 else ("N4_uturn", "N3_wrong")):
                if kind == "N4_uturn" and (int(rt["n_turn"][i]) > 0 or bool(rt["in_turn"][i])):
                    continue
                neg = RN.make(kind, P, rng, lht=bool(t["tc"][i][1] > 0.5)) if P is not None else None
                if neg is not None:
                    break
            if neg is None:
                out["role"] = ROLE["D"]
            else:
                self._neg_feat(out, neg["poly"], neg["pmask"], rng)
        if d == "nav" and self.cfg.t6:
            k = self.sdf_row[i]
            out["sdf"] = np.int64(k if (k >= 0 and self.sdf["logged_inside"][k]) else -1)
        return out

    def _neg_feat(self, out, poly, pmask, rng):
        out["fb"] = RA.features("bear", poly, pmask, np.random.default_rng(rng.integers(1 << 62)))
        out["fp"] = RA.features("poly", poly, pmask, np.random.default_rng(rng.integers(1 << 62)))

    def _carla(self, key, x, rng):
        C = self.C
        if key == "Pcar":
            j = x
            p = int(C.r["pose_row"][j])
        else:
            p = x
        out = dict(trunk=C.T[p], valid=C.tab["slot_valid"][p], tc=C.tab["tc"][p], cam=np.float32(C.cam), tgt=C.tea["out"][p], tmu=C.tea["mu"][p],
                   hum=np.full((16, 3), np.nan, np.float32), role=ROLE[key[0]], fb=np.zeros(RA.ENC_DIM["bear"], np.float32),
                   fp=np.zeros(RA.ENC_DIM["poly"], np.float32), at=np.float32(0), aw=np.float32(0), sdf=np.int64(-1))
        if key == "Pcar":
            hum = path_target(C.r["poly"][j], C.r["pmask"][j], C.tea["mu"][p], C.cam)
            if hum is None:
                out["role"] = ROLE["D"]
                return out
            out["hum"] = hum
            self._neg_feat(out, C.r["poly"][j], C.r["pmask"][j], rng)
            a, w = act_target(hum, float(C.tab["v0"][p]))
            out["at"], out["aw"] = np.float32(a), np.float32(w)
        elif key == "Ncar":
            miss = missing_classes(C, p)
            js = C.rows_of[p]
            base = js[np.argmin(np.abs(C.r["angle"][js]))]                   # the straightest existing exit
            P = dense_from_poly(C.r["poly"][base], C.r["pmask"][base])
            m = miss[rng.integers(len(miss))]
            neg = RN.make("N1_exit", P, rng, s_branch=float(C.tab["d"][p]) + 2.0, missing=m) if P is not None else None
            if neg is None:
                out["role"] = ROLE["D"]
            else:
                self._neg_feat(out, neg["poly"], neg["pmask"], rng)
        return out

    def __getitem__(self, k):
        if not self.ok:
            self._open()
        c = self.cfg
        rng = np.random.default_rng([c.seed, k])
        rows = []
        for key, cnt in c.rows.items():
            if not cnt:
                continue
            if key == "Nreal":
                for _ in range(cnt):
                    d = "nav" if rng.random() < 0.5 else "wod"
                    p, _ = self.pool[f"N{d}"]
                    rows.append(self._real(d, int(p[rng.integers(len(p))]), "N", rng))
                continue
            p, w = self.pool[key]
            pick = rng.choice(p, cnt, p=w) if w is not None else p[rng.integers(len(p), size=cnt)]
            for x in pick:
                if key.endswith("car"):
                    rows.append(self._carla(key, int(x), rng))
                else:
                    d = {"Pnav": "nav", "Pwod": "wod", "Dnav": "nav", "Dwod": "wod", "Dhc": "hc"}[key]
                    rows.append(self._real(d, int(x), key[0], rng))
        out = {k: torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows])) for k in rows[0]}
        if c.zero_cmd:
            out["fb"].zero_(), out["fp"].zero_()
        if c.t6:
            ks = out["sdf"].numpy()
            g = np.zeros((len(ks),) + self.sdf["sdf"].shape[1:], np.float16)
            g[ks >= 0] = self.sdf["sdf"][ks[ks >= 0]]
            out["sdfg"] = torch.from_numpy(g)
        if c.t4:
            b = L.assemble(self.LD, self.LM.draw(rng))
            out["L"] = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
        if c.t5:
            b = self.HB[int(k)]
            out["H"] = {kk: v for kk, v in b.items() if kk != "kind"}
        return out


# ---------------------------------------------------------------- losses
class RLoss:
    def __init__(self, net, cfg: RCfg, tstd, dev):
        self.cfg = cfg
        self.base = L.Losses(net, cfg.lcfg(), tstd, dev)
        di = A.distill_index(net.slices)
        s = net.slices["action"]
        self.act_col = torch.as_tensor(np.isin(di, np.arange(s.start, s.start + 2)), device=dev)
        self.a0 = s.start
        self.dev = dev
        if cfg.t6:
            with np.load(data_dir() / "runs/op_route_ft/drivable/nav.npz") as z:
                self.g0 = (float(z["x0"]), float(z["y0"]), float(z["res"]))

    def sdf_at(self, g, x, y):
        """g (B, Hx, Wy) sdf grid, x / y (B, 16) metres -> sdf values (B, 16) (bilinear, outside the grid = +10)."""
        x0, y0, res = self.g0
        Hx, Wy = g.shape[1:]
        u = ((y - y0) / res - 0.5) / (Wy - 1) * 2 - 1   # cell centres at x0 + (i + 0.5) res; grid_sample grid = (W index, H index)
        v = ((x - x0) / res - 0.5) / (Hx - 1) * 2 - 1
        s = torch.nn.functional.grid_sample(g[:, None].float(), torch.stack([u, v], -1)[:, None], align_corners=True, padding_mode="border")[:, 0, 0]
        inside = (u.abs() <= 1) & (v.abs() <= 1)
        return torch.where(inside, s, torch.full_like(s, 10.0))

    def __call__(self, o, b):
        c, base = self.cfg, self.base
        out = o["outputs"].float()
        plan = base.plan(out)
        role = b["role"]
        Ls, tot = {}, 0.0
        P, D, N = role == 1, role == 2, role == 3
        if P.any():
            Ls["imit"] = base.imit_dist(plan[P], None, b["cam"][P], b["hum"][P]).mean()
            tot = tot + c.lam_i * Ls["imit"]
            aw = b["aw"][P]
            if aw.sum() > 0:
                e = L.huber((out[P][:, self.a0] - b["at"][P]) / SIG_A)
                Ls["act"] = (aw * e).sum() / aw.sum()
                tot = tot + c.lam_a * Ls["act"]
        cn = D | N
        if cn.any():
            Ls["cons"] = base.imit_dist(plan[cn], b["tmu"][cn].float(), b["cam"][cn]).mean()
            tot = tot + c.dw * c.lam_c * Ls["cons"]
            if N.any():
                Ls["neg"] = base.imit_dist(plan[N], b["tmu"][N].float(), b["cam"][N]).mean().detach()
        e = ((out[:, base.di] - b["tgt"].float()) / base.tstd).pow(2)
        free = base.plan_cols | self.act_col
        keep = cn.float()
        num = e[:, ~free].sum(1) + keep * e[:, free].sum(1)
        Ls["distill"] = (num / e.shape[1]).mean()
        tot = tot + c.dw * c.lam_d * Ls["distill"]
        if c.t6 and "sdfg" in b:
            m = b["sdf"] >= 0
            if m.any():
                x, y, _ = base.grid.rear(plan[m], b["cam"][m])
                s = self.sdf_at(b["sdfg"][m], x, y)
                Ls["offroad"] = torch.relu(-s - c.t6_margin).mean()
                tot = tot + c.t6 * Ls["offroad"]
        return tot, Ls


# ---------------------------------------------------------------- bank (CARLA poses)
def cmd_bank(a):
    dev = torch.device("cuda")
    root = data_dir() / CARLA_ROOTS[a.which]
    imgs = np.load(root / "samples/route_carla/imgs.npy", mmap_mode="r")
    with np.load(root / "samples/route_carla/tab.npz", allow_pickle=True) as z:
        sv, tc = z["slot_valid"], z["tc"]
    n = len(imgs) if not a.limit else a.limit
    out = rroot("bank", a.which)
    if (out / "teacher.npz").exists():
        print("exists", out)
        return
    m = L.load_model(None, dev)
    didx = A.distill_index(m.net.slices)
    pi = A.plan_index(m.net.slices)
    tmp = out / "trunk.tmp.npy"
    T = np.lib.format.open_memmap(tmp, "w+", np.float16, (n, 9, 1024, 8, 16))
    to, mu = np.zeros((n, len(didx)), np.float16), np.zeros((n, 33, 15), np.float32)
    from concurrent.futures import ThreadPoolExecutor
    bs = a.batch
    load = lambda i: torch.from_numpy(np.ascontiguousarray(imgs[i:min(i + bs, n)])).pin_memory()  # noqa: E731
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(load, i) for i in range(0, min(n, 4 * bs), bs)]
        for k, i in enumerate(range(0, n, bs)):
            x = futs[k].result().to(dev, non_blocking=True)
            if i + 4 * bs < n:
                futs.append(ex.submit(load, i + 4 * bs))
            v = torch.from_numpy(sv[i:i + len(x)]).to(dev)
            with torch.no_grad():
                tr = H.trunks(m.net, x) * v[:, :, None, None, None]
                o = m(tr, v, torch.from_numpy(tc[i:i + len(x)]).to(dev).half())["outputs"].float()
            T[i:i + len(x)] = tr.cpu().numpy()
            to[i:i + len(x)] = o[:, didx].cpu().numpy()
            mu[i:i + len(x)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
            if k % 20 == 0:
                print(f"bank {a.which}: {i + len(x)}/{n} {(i + len(x)) / (time.time() - t0):.1f} poses/s", flush=True)
    T.flush()
    del T
    tmp.replace(out / "trunk.npy")
    np.savez(out / "teacher.npz", out=to, mu=mu)
    print(f"bank {a.which}: {n} poses in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- train
def cfg_of(arm, seed=0, steps=0, carla=None) -> RCfg:
    c = RCfg(name=arm, seed=seed, **ARMS[arm])
    if steps:
        c = replace(c, steps=steps)
    if carla:
        c = replace(c, carla=carla)
    return c


def cmd_train(a):
    from jevdrive.run import Run
    from jevdrive.data import splits
    cfg = cfg_of(a.arm, a.seed, a.steps, a.carla)
    tag = a.tag or f"{a.arm}-s{a.seed}"
    d = rroot("runs", tag)
    with Run("op_route_ft", tag, resume=d, seed=cfg.seed, config=cfg.dump()) as run:
        for s in ("navsim/op-adapt-h-nav-train", "b2d/op-adapt-h-carla-train", "wod/r2-train", "b2d/route-carla-train"):
            try:
                run.use_split(splits.load(s))
            except Exception as e:  # noqa: BLE001
                run.info(f"split {s} not registered: {e}")
        train(cfg, run, d, a)


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        out[k] = to_dev(v, dev) if isinstance(v, dict) else v.to(dev, non_blocking=True)
    return out


def train(cfg: RCfg, run, d: Path, a):
    dev = torch.device("cuda")
    model = RModel(cfg.lcfg(), cfg.enc).to(dev)
    base, new = model.trainable()
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    tstd = np.load(L.r2t() / "teacher" / "tstd.npy")
    lossf = RLoss(model.net, cfg, tstd, dev)
    lossL = L.Losses(model.net, replace(cfg.lcfg(), dw=3.0), tstd, dev) if cfg.t4 else None
    lossH = H.Losses(model.net, H.HCfg("t5", dw=3.0), tstd, dev) if cfg.t5 else None
    step, ck = 0, d / "ckpt.pt"
    if ck.exists() and not a.fresh:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state(st["model"]), opt.load_state_dict(st["opt"]), scaler.load_state_dict(st["scaler"])
        step = st["step"]
        run.info(f"resumed at step {step}")
    run.info(f"{cfg.name}: {cfg.steps} steps, rows {cfg.rows}, enc {cfg.enc}, zero_cmd {cfg.zero_cmd}, t4 {cfg.t4} t5 {cfg.t5} t6 {cfg.t6}; "
             f"trainable {sum(p.numel() for p in base) / 1e6:.1f} M + adapter {sum(p.numel() for p in new) / 1e6:.2f} M")
    dl = torch.utils.data.DataLoader(Batcher(cfg), batch_size=None, sampler=range(step, 10 ** 9), num_workers=cfg.workers, prefetch_factor=4,
                                     pin_memory=True, persistent_workers=False)
    it = iter(dl)
    model.train()
    hist, wait, t_start, s0 = [], 0.0, time.time(), step
    torch.cuda.reset_peak_memory_stats()
    bar = run.tqdm(total=cfg.steps, initial=step, desc=cfg.name)
    fkey = "fp" if cfg.enc == "poly" else "fb"
    while step < cfg.steps:
        tw = time.time()
        b = next(it)
        wait += time.time() - tw
        b = to_dev(b, dev)
        o = model(b["trunk"], b["valid"], b["tc"], b[fkey])
        total, Ls = lossf(o, b)
        if "L" in b:
            bl = b["L"]
            ol = model(bl["trunk"], bl["valid"], bl["tc"], None)
            tl, LL = lossL(ol, bl)
            total = total + tl
            Ls.update({f"t4_{k}": v for k, v in LL.items()})
        if "H" in b:
            bh = b["H"]
            oh = model(bh["trunk"], bh["valid"], bh["tc"], None)
            th, LH = lossH(oh, bh)
            total = total + th
            Ls.update({f"t5_{k}": v for k, v in LH.items()})
        frac = step / cfg.steps
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(base + new, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        bar.update()
        hist.append({k: float(v) for k, v in Ls.items()} | {"total": float(total)})
        if step % 25 == 0 or step == cfg.steps:
            import pandas as pd
            m = pd.DataFrame(hist[-25:]).mean()
            el = time.time() - t_start
            run.scalars({f"loss/{k}": v for k, v in m.items()}, step)
            run.scalars({"throughput/steps_per_s": (step - s0) / el, "throughput/data_wait_frac": wait / el,
                         "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step)
            if step % 100 == 0 or step == cfg.steps:
                run.info(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                         f"; {(step - s0) / el:.2f} it/s, wait {wait / el:.2f}, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
        if step % cfg.ckpt_every == 0 and step < cfg.steps:
            torch.save({"model": model.state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, d / "ckpt.tmp.pt")
            (d / "ckpt.tmp.pt").replace(ck)
    bar.close()
    del it, dl
    torch.save({"model": model.state(), "cfg": cfg.lcfg().dump(), "rcfg": cfg.dump()}, d / "ckpt-final.pt")
    if model.route is not None:
        model.route.to_npz(d / "adapter.npz")
    ck.unlink(missing_ok=True)
    run.summary.update(steps=step, train_s=time.time() - t_start, data_wait_frac=wait / (time.time() - t_start),
                       peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bank")
    p.add_argument("--which", default="ol", choices=list(CARLA_ROOTS))
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--carla", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--fresh", action="store_true")
    p = sp.add_parser("evalol")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--carla", default="ol")
    p.add_argument("--cap", type=int, default=600)
    a = ap.parse_args()
    if a.cmd == "evalol":
        import rft_eval
        rft_eval.main(a)
    else:
        {"bank": cmd_bank, "train": cmd_train}[a.cmd](a)
