"""factor_wm x op_parity: on-policy training of the input-parity Cinque (op_parity P2, decision 144) in the exact-ego engine (decisions 141 /
143). Plan: plans/2026-10-06-p2-onpolicy-prereg.md (op-train env, one GPU per job, CPU warp workers).

Policy  op_parity PModel P2: frozen Cinque vision -> `view_39` hidden tokens of the 8 newest 0.2 s image pairs (slot 0 zero, as in P2's
        training), parity adapter bias from [cmd, vx, vy, ax, ay, 4 poses at -1.5 / -1.0 / -0.5 / 0 s], trainable plan pathway.
Engine  factor_wm plane warp of the time-synchronous logged frame (fw_common, fw_g0.frame_job), 8 s, validity / failure events of fw_common.Ego.
        Kind `plan` (this experiment): the ego executes the PLAN, lateral by pure pursuit on it (look-ahead max(3, min(20, v x 1 s)) m), longitudinal
        tracks its 0.5 s speed (a = (v_plan(0.5) - v) / 0.5, clipped to [-3.5, 2] m/s2); decision 119 (longitudinal from the plan) and g1-diag H2
        (S3's engine read action[1], which HUGSIM never reads). P2's recipe distils every non-plan head (action included) to shipped, so the
        plan is the only thing training moves; an action-driven engine would roll out shipped's behaviour, not P2's.
Ego inputs in the engine: speed / acceleration of the simulated ego (vy = ay = 0, as the HUGSIM client), poses from the logged history (frame 0's
        5 s future, dg_common.poses_from_fut20) followed by the ego's own trace, command = WOD routing intent of the time-synchronous logged frame
        (1 straight, 2 left, 3 right -> NAVSIM one-hot [L, S, R]; 0 unknown -> zeros).
Labels  fw_train's target path (logged 5 s future, lateral recovery t_rec 4 s, longitudinal catch-up over 3 s) -> 8 rear-axle poses at
        0.5 .. 4 s (x, y, yaw), the target format of op_parity's loss. Visited states are labelled only inside fw_train's validity domain.

  prep     SETS            logged inputs per clip set ($FW/p2op/logged/<set>/tab.npz); train also the logged token store tok.npy (n, 49, 32, 512)
  collect  --policy T --tag xr1-s0 --seed R --shard i/n         DAgger rollouts on the train clips (fw_roll.collect_specs arms), tokens + labels
  eval     --policy T --tag ev-PX-s0 [--sets g0b,g1s]           held-out engine readout (fw_roll.eval_specs arms, kind plan)
  train    --arm X|C --init P2-F-s0 --rolls xr1-s0,.. --steps N --tag PX-s0     continue P2: 64 WOD rows (X: 40 visited + 24 logged; C: 64
           logged) + 64 navtrain rows (P2's own W-protocol rows, 25 % anchor) per batch; checkpoint in op_parity format, linked into
           $DATA_DIR/runs/op_parity/runs/<tag> so pp_eval / pp_hugsim read it like any op_parity tag.
Policy tags T: an op_parity run tag (P2-F-s0, PX-r1-s0, ..).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
sys.path.insert(0, str(C.REPO / "experiments/op_parity/scripts"))
import parity_adapter as PA  # noqa: E402

DG = C.DG
KEYS = np.array([7.5, 5.0, 2.5, 0.0])               # lattice steps back of the 4 history poses (-1.5, -1.0, -0.5, 0 s)
T8 = 0.5 * np.arange(1, 9)
NAV = {1: 1, 2: 0, 3: 2}                            # WOD intent -> NAVSIM one-hot index [left, straight, right]
A_MIN, A_MAX, K_MAX = -3.5, 2.0, 0.25
NS = 8                                              # policy slots fed (P2's training: the 8 newest, slot 0 zero)
NAVTRAIN = tuple(f"navtrain_full.s{i}of12" for i in range(12))
NAV_SPLIT = "navsim/op-parity-full"


def P(*p) -> Path:
    return C.root("p2op", *p)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rel_to(Q, base):
    """poses Q (n, 3) -> in the frame of `base` (3,)."""
    return np.c_[(Q[:, :2] - base[:2]) @ DG.rot(base[2]), wrap(Q[:, 2] - base[2])]


def cmd_of(intent) -> np.ndarray:
    c = np.zeros(4, np.float32)
    if int(intent) in NAV:
        c[NAV[int(intent)]] = 1.0
    return c


def ego_feat(L, v, a, cmd) -> np.ndarray:
    """L (i + 1, 3) lattice poses up to now (common frame) -> parity_adapter.ego_features of the newest state."""
    i = len(L) - 1
    tq = np.maximum(i - KEYS, 0.0)
    k = np.arange(len(L))
    th = np.unwrap(L[:, 2])
    Q = np.c_[np.interp(tq, k, L[:, 0]), np.interp(tq, k, L[:, 1]), np.interp(tq, k, th)]
    return PA.ego_features(rel_to(Q, Q[-1]), np.tile([v, 0.0], (4, 1)), np.tile([a, 0.0], (4, 1)), cmd)


def poses_at(q, tq, s_min=0.6):
    """dg_common.poses_from_fut20 at arbitrary times tq (from 0): (len(tq), 3) x, y, yaw of the path q (20, 2) on the 0.25 s grid."""
    from scipy.interpolate import CubicSpline
    t = np.r_[0.0, DG.T_FUT]
    Pq = np.r_[np.zeros((1, 2)), np.asarray(q, float)]
    xy = CubicSpline(t, Pq, axis=0)(tq)
    seg = np.linalg.norm(np.diff(Pq, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    ok = np.r_[True, seg > 1e-4]
    s_u, P_u = s[ok], Pq[ok]
    psi = np.zeros(len(tq))
    if s_u[-1] >= s_min:
        for j, sj in enumerate(np.interp(tq, t, s)):
            a, b = np.clip([sj - s_min / 2, sj + s_min / 2], 0.0, s_u[-1])
            if b - a < 0.5 * s_min:
                a, b = (max(0.0, b - s_min), b) if b >= s_u[-1] else (a, min(s_u[-1], a + s_min))
            d = np.array([np.interp(b, s_u, P_u[:, c]) - np.interp(a, s_u, P_u[:, c]) for c in range(2)])
            psi[j] = np.arctan2(d[1], d[0])
    return np.c_[xy, psi]


def target8(fut20, off, v) -> np.ndarray:
    """fw_train's target path for a state at offset off with speed v -> (8, 3) poses at 0.5 .. 4 s (heading 0 at t = 0)."""
    import fw_train as FT
    from experiments.op_adapt_h.lib import op_adapt_h as H
    dx, dy, dp = (float(x) for x in off)
    f = np.asarray(fut20, float)
    q = H.recover_target(f, dy, dp, v, t_rec=FT.T_REC).astype(float) if (abs(dy) > 1e-6 or abs(dp) > 1e-9) else f.copy()
    if abs(dx) > 1e-6:
        u = np.clip(FT.TF / FT.T_CATCH, 0, 1)
        q[:, 0] += -dx * (3 * u ** 2 - 2 * u ** 3)
    p = poses_at(q, np.r_[0.0, T8])
    p[:, 2] -= p[0, 2]
    return p[1:].astype(np.float32)


# ---------------------------------------------------------------- prep: logged inputs per clip set
def hist_poses(fut20_row0) -> np.ndarray:
    """(10, 3) logged poses of frames 0..9 (-1.8 .. 0 s) in the t0 frame."""
    Q = DG.poses_from_fut20(fut20_row0, k=C.T0)
    return rel_to(Q, Q[-1])


def cmd_prep(a):
    from experiments.op_adapt_l.lib import op_adapt_l as L
    for name in a.sets:
        d = P("logged", name)
        S = C.Clips(name)
        js = json.load(open(C.root("clips") / f"{name}.json"))
        intent = np.load(L.lroot("prep") / f"{js['src']}.npz", allow_pickle=True)["intent"]
        rows = np.array([c["rows"] for c in js["clips"]])
        nf = rows.shape[1]
        v = S.t["v"].astype(float)
        alog = np.gradient(v, C.DT, axis=1)
        hist = np.stack([hist_poses(S.t["fut20"][c, 0]) for c in range(S.n)])
        cmd = np.stack([[cmd_of(intent[r]) for r in rr] for rr in rows])
        tab = dict(hist=hist.astype(np.float32), cmd=cmd, alog=alog.astype(np.float32))
        if name == "train":
            kl = S.kl
            ego = np.zeros((S.n, kl + 1, PA.EGO_DIM), np.float32)
            fut8 = np.zeros((S.n, kl + 1, 8, 3), np.float32)
            for c in range(S.n):
                Lc = np.r_[hist[c], S.t["pose"][c][1:]]
                for j in range(kl + 1):
                    i = C.T0 + j
                    ego[c, j] = ego_feat(Lc[: i + 1], v[c, i], alog[c, i], cmd[c, i])
                    fut8[c, j] = target8(S.t["fut20"][c, i], (0, 0, 0), v[c, i])
            tab |= dict(ego=ego, fut8=fut8)
            if not (d / "tok.npy").exists():
                import torch
                enc = Encoder(torch.device("cuda"))
                tmp = d / "tok.tmp.npy"
                out = np.lib.format.open_memmap(tmp, "w+", np.float16, (S.n, nf - 1, 32, 512))
                for c0 in range(0, S.n, 8):
                    f = np.asarray(S.imgs[c0: c0 + 8])
                    n = len(f)
                    t = enc(f[:, :-1].reshape(-1, *f.shape[2:]), f[:, 1:].reshape(-1, *f.shape[2:]))
                    out[c0: c0 + n] = t.reshape(n, nf - 1, 32, 512).cpu().numpy()
                out.flush()
                del out
                tmp.replace(d / "tok.npy")
        np.savez(d / "tab.npz", **tab)
        print(f"prep {name}: {S.n} clips, ego |pose x| max {np.abs(hist[..., 0]).max():.1f} m, cmd L/S/R/0 "
              f"{cmd[:, C.T0].sum(0).astype(int).tolist()} / {int((cmd[:, C.T0].sum(1) == 0).sum())}", flush=True)


# ---------------------------------------------------------------- policy, encoder, engine
class Encoder:
    """Frozen Cinque vision: (n, 2, 6, 128, 256) uint8 pairs -> (n, 32, 512) fp16 `view_39` tokens on the GPU (as op_parity pp_prep)."""

    def __init__(self, dev, net=None):
        import torch
        from jevdrive import op_adapt as A
        self.torch, self.A, self.dev = torch, A, dev
        self.net = net if net is not None else A.load("cinque", torch.float16).to(dev).eval()

    def __call__(self, prev, cur, bs=128):
        torch, A = self.torch, self.A
        out = []
        with torch.no_grad():
            for i in range(0, len(cur), bs):
                p, c = (torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(self.dev) for x in (prev, cur))
                out.append(self.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), 32, 512))
        return torch.cat(out)


def control(plan, v, cam_x):
    """plan (33, 15) camera frame -> (curvature, openpilot sign right +; acceleration). Pure pursuit + plan-speed tracking (module doc)."""
    from jevdrive import op_adapt as A
    psi = -plan[:, 11]
    X = plan[:, 0] + cam_x - cam_x * np.cos(psi)
    Y = -plan[:, 1] - cam_x * np.sin(psi)
    a = float(np.clip((np.interp(0.5, A.T_IDXS, plan[:, 3]) - v) / 0.5, A_MIN, A_MAX))
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
    if s[-1] < 1.0:
        return 0.0, a
    ld = min(float(np.clip(v * 1.0, 3.0, 20.0)), s[-1])
    px, py = np.interp(ld, s, X), np.interp(ld, s, Y)
    kl = 2.0 * py / max(px * px + py * py, 1e-6)
    return float(np.clip(-kl, -K_MAX, K_MAX)), a


class PlanEgo(C.Ego):
    """fw_common.Ego with kind `plan`: the curvature / acceleration given are executed directly (no openpilot lateral / longitudinal stack)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.acc = 0.0

    def advance(self, j, kappa, acc):
        if self.event is not None:
            return super().advance(j, kappa, acc)
        P_ = self.P
        v_new = max(0.0, self.v + float(acc) * C.DT)
        self.acc = (v_new - self.v) / C.DT
        ds = 0.5 * (self.v + v_new) * C.DT
        dth = -float(kappa) * ds + (self.exo[j] - self.exo[j - 1])
        mid = self.th + 0.5 * dth
        self.p = self.p + ds * np.array([np.cos(mid), np.sin(mid)])
        self.th += dth
        self.v = v_new
        dxy = DG.rot(-P_[j, 2]) @ (self.p - P_[j, :2])
        off = (float(dxy[0]), float(dxy[1]), float(wrap(self.th - P_[j, 2])))
        self.trace.append(off)
        self.vs.append(self.v)
        self._check(j, off)
        return off


class Runner:
    def __init__(self, policy: str, dev: str, workers: int):
        import torch
        import pp_train as T
        from jevdrive import op_adapt as A
        self.torch, self.dev = torch, torch.device(dev)
        self.m = T.load_pmodel(policy, self.dev).eval()
        sl = self.m.net.slices
        self.pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        self.enc = Encoder(self.dev, self.m.net)                    # the policy's own (frozen, shipped) vision weights
        self.pool = ProcessPoolExecutor(workers)
        self.A = A

    def plans(self, H, ego, tc):
        torch = self.torch
        with torch.no_grad():
            o = self.m(H, torch.from_numpy(np.asarray(ego, np.float32)).to(self.dev), torch.from_numpy(np.asarray(tc, np.float32)).to(self.dev))
        return o.float().cpu().numpy()[:, self.pi].reshape(-1, 33, 15)

    def run(self, S, tab, specs, batch=64, tok_out=None, kl=None):
        """specs -> list of per-rollout dicts; tok_out: (n_specs, kl, 32, 512) memmap for the step tokens (collection)."""
        import fw_g0 as G
        torch = self.torch
        kl = kl or S.kl
        res = [None] * len(specs)
        order = sorted(range(len(specs)), key=lambda i: specs[i]["c"])
        for b0 in range(0, len(order), batch):
            ids = order[b0: b0 + batch]
            sp = [specs[i] for i in ids]
            cs = [s["c"] for s in sp]
            uc = sorted(set(cs))
            f = np.asarray(S.imgs[uc, 1: C.NH])                                     # frames 1..9 -> history pairs 1..8
            hh = self.enc(f[:, :-1].reshape(-1, *f.shape[2:]), f[:, 1:].reshape(-1, *f.shape[2:])).reshape(len(uc), NS, 32, 512)
            H = hh[[uc.index(c) for c in cs]].clone()
            tc = S.t["tc"][cs].astype(np.float32)
            camx = S.t["cam"][cs][:, 0].astype(float)
            egos = [PlanEgo(S.t["pose"][c][: kl + 1], S.t["v"][c], "closed", s.get("exo"), 0.0, 0.0, str(S.t["cat"][c])) for c, s in zip(cs, sp)]
            Ls = [list(tab["hist"][c]) for c in cs]
            ego = [ego_feat(np.array(Ls[b]), e.v, float(tab["alog"][c, C.T0]), tab["cmd"][c, C.T0]) for b, (c, e) in enumerate(zip(cs, egos))]
            rec_e = [ego]
            plan = self.plans(H, ego, tc)
            rec = [[control(plan[b], e.v, camx[b]) + (float(np.interp(0.5, self.A.T_IDXS, plan[b, :, 3])),) for b, e in enumerate(egos)]]
            prev = np.asarray(S.imgs[cs, C.T0])
            for j in range(1, kl + 1):
                for b, e in enumerate(egos):
                    e.advance(j, rec[-1][b][0], rec[-1][b][1])
                jobs = []
                for c, s, e in zip(cs, sp, egos):
                    pose = C.compose(S.t["pose"][c][j], np.array([e.clamp_off()]))[0] if e.event is not None else np.r_[e.p, e.th]
                    jobs.append((S.name, c, j, "plane", "log", pose))
                cur = np.stack(list(self.pool.map(G.frame_job, jobs, chunksize=1)))
                tok = self.enc(prev, cur)
                if tok_out is not None:
                    tok_out[ids, j - 1] = tok.cpu().numpy()
                H = torch.cat([H[:, 1:], tok[:, None]], 1)
                prev = cur
                ego = []
                for b, (c, e) in enumerate(zip(cs, egos)):
                    Ls[b].append(np.r_[e.p, e.th])
                    ego.append(ego_feat(np.array(Ls[b]), e.v, e.acc, tab["cmd"][c, C.T0 + j]))
                rec_e.append(ego)
                plan = self.plans(H, ego, tc)
                rec.append([control(plan[b], e.v, camx[b]) + (float(np.interp(0.5, self.A.T_IDXS, plan[b, :, 3])),) for b, e in enumerate(egos)])
            for b, i in enumerate(ids):
                e = egos[b]
                r = np.array([x[b] for x in rec])
                res[i] = dict(kappa=r[:, 0], acc_cmd=r[:, 1], v_plan=r[:, 2], off=np.array(e.trace), vs=np.array(e.vs),
                              ego=np.array([x[b] for x in rec_e], np.float32), event=e.event or "",
                              t_event=np.nan if e.t_event is None else e.t_event)
            print(f"  {min(b0 + batch, len(order))}/{len(order)} rollouts", flush=True)
        return res


def labels(S, res, specs):
    """Validity mask (fw_train.states_of) and 8-pose targets of every visited state."""
    import fw_train as FT
    kl = len(res[0]["off"]) - 1
    ok = np.zeros((len(res), kl + 1), bool)
    fut8 = np.zeros((len(res), kl + 1, 8, 3), np.float32)
    for k, (r, s) in enumerate(zip(res, specs)):
        off, te = r["off"], r["t_event"]
        m = (np.abs(off[:, 1]) <= FT.CAP_Y) & (np.abs(off[:, 2]) <= FT.CAP_PSI) & (np.abs(off[:, 0]) <= C.DX_CAP)
        m = np.logical_and.accumulate(m)
        jmax = kl if not np.isfinite(te) else int(round(te / C.DT)) - 1
        for j in range(1, min(jmax, kl) + 1):
            if m[j]:
                ok[k, j] = True
                fut8[k, j] = target8(S.t["fut20"][s["c"], C.T0 + j], off[j], r["vs"][j])
    return ok, fut8


def save(path, specs, res, S, extra=None):
    out = {k: np.stack([r[k] for r in res]) for k in res[0]}
    out |= {k: np.array([s.get(k) for s in specs]) for k in ("c", "arm")}
    out |= dict(clip_id=S.t["id"][out["c"]], cat=S.t["cat"][out["c"]]) | (extra or {})
    np.savez(path.with_suffix(".tmp.npz"), **out)
    path.with_suffix(".tmp.npz").replace(path)


def cmd_roll(a):
    import fw_roll as FR
    workers = a.workers or max(1, len(os.sched_getaffinity(0)) - 2)
    R = Runner(a.policy, "cuda", workers)
    t0 = time.time()
    i, n = map(int, a.shard.split("/"))
    sets = ["train"] if a.cmd == "collect" else a.sets.split(",")
    for name in sets:
        S = C.Clips(name)
        tab = dict(np.load(P("logged", name) / "tab.npz"))
        sp = FR.collect_specs(S, a.seed, "plan") if a.cmd == "collect" else FR.eval_specs(S)
        if a.limit:
            sp = [s for s in sp if s["c"] < a.limit]
        cs = set(sorted({s["c"] for s in sp})[i::n])
        sp = [s for s in sp if s["c"] in cs]
        out = P("roll", a.tag) / f"{name}-{a.cmd}{'-smoke' if a.limit else ''}-{i}of{n}.npz"
        if out.exists():
            print("exists", out)
            continue
        print(f"{a.tag} {name}: {len(sp)} rollouts, {len(cs)} clips, {workers} warp workers, policy {a.policy}", flush=True)
        tok = None
        if a.cmd == "collect":
            tp = out.with_suffix(".tok.tmp.npy")
            tok = np.lib.format.open_memmap(tp, "w+", np.float16, (len(sp), S.kl, 32, 512))
        res = R.run(S, tab, sp, a.batch, tok_out=tok)
        extra = None
        if tok is not None:
            tok.flush()
            del tok
            tp.replace(out.with_suffix(".tok.npy"))
            ok, fut8 = labels(S, res, sp)
            extra = dict(ok=ok, fut8=fut8)
            print(f"{a.tag}: {int(ok.sum())} labelled states", flush=True)
        save(out, sp, res, S, extra)
        print(f"{a.tag} {name}: done at {time.time() - t0:.0f} s -> {out}", flush=True)
    import dg_roll as DR
    DR.bye()


# ---------------------------------------------------------------- training
class WodRows:
    """Token windows, ego inputs and targets of WOD states: logged states (clip c, state j) and visited states of rollout files."""

    def __init__(self, rolls):
        d = P("logged", "train")
        self.tok = np.load(d / "tok.npy", mmap_mode="r")                 # (n, 49, 32, 512)
        z = np.load(d / "tab.npz")
        self.ego, self.fut8 = z["ego"], z["fut8"]
        S = C.Clips("train")
        self.camx, self.tc, self.n, self.kl = S.t["cam"][:, 0].astype(np.float32), S.t["tc"].astype(np.float32), S.n, S.kl
        self.R = []                                                       # (tok memmap, c, ego, fut8, [(rollout, j)])
        for r in rolls:
            for f in sorted(P("roll", r).glob("train-collect-[0-9]*of*.npz")):
                z = np.load(f, allow_pickle=True)
                k = np.argwhere(z["ok"])
                self.R.append((np.load(f.with_suffix(".tok.npy"), mmap_mode="r"), z["c"], z["ego"], z["fut8"], k))
        self.cum = np.cumsum([0] + [len(x[4]) for x in self.R])

    def window(self, c, j, roll=None):
        """(8, 32, 512) pairs j + 1 .. j + 8: logged history pairs (<= 8), then the rollout's step tokens."""
        p = np.arange(j + 1, j + 1 + NS)
        if roll is None:
            return self.tok[c, p]
        tok, r = roll
        out = np.empty((NS, 32, 512), np.float16)
        h = p <= C.T0 - 1
        out[h] = self.tok[c, p[h]]
        out[~h] = tok[r, p[~h] - C.T0]
        return out

    def draw(self, rng, n_vis, n_log):
        toks, ego, fut, camx, tc = [], [], [], [], []
        for _ in range(n_vis):
            g = int(rng.integers(self.cum[-1]))
            f = int(np.searchsorted(self.cum, g, side="right") - 1)
            tok, cc, eg, f8, k = self.R[f]
            r, j = k[g - self.cum[f]]
            c = int(cc[r])
            toks.append(self.window(c, j, (tok, r)))
            ego.append(eg[r, j]), fut.append(f8[r, j]), camx.append(self.camx[c]), tc.append(self.tc[c])
        for _ in range(n_log):
            c, j = int(rng.integers(self.n)), int(rng.integers(self.kl + 1))
            toks.append(self.window(c, j))
            ego.append(self.ego[c, j]), fut.append(self.fut8[c, j]), camx.append(self.camx[c]), tc.append(self.tc[c])
        return [np.stack(x) for x in (toks, ego, fut, camx, tc)]


def cmd_train(a):
    import torch
    import pp_train as T
    from dataclasses import asdict
    from jevdrive.data import splits
    from jevdrive.run import Run
    from jevdrive.common import data_dir
    d = P("runs", a.tag)
    link = data_dir() / "runs" / "op_parity" / "runs" / a.tag
    if (d / "ckpt-final.pt").exists():
        print("exists", d)
        if not link.exists():
            link.symlink_to(d)
        return
    dev = torch.device("cuda")
    cfg = T.Cfg(arm="P2", seed=a.seed, steps=a.steps, batch=128, data=NAVTRAIN, split=NAV_SPLIT, frames="warp", host=True, warmup=100,
                eval_every=a.steps)
    rolls = [r for r in a.rolls.split(",") if r and r != "none"]
    assert (a.arm == "X") == bool(rolls), "X needs --rolls, C has none"
    n_vis, n_log, n_nav = (40, 24, 64) if a.arm == "X" else (0, 64, 64)
    rng = np.random.default_rng([a.seed, 11])
    torch.manual_seed(a.seed)
    names = np.concatenate([np.load(data_dir() / "runs/op_parity/cache" / x / "tab.npz")["names"] for x in NAVTRAIN])
    tr_rows, dv_rows, sp = T.split_rows({"names": names}, NAV_SPLIT)
    if a.limit:
        tr_rows = tr_rows[: a.limit]
    with Run("factor_wm", f"p2op-{a.tag}", seed=a.seed, config=asdict(cfg) | vars(a)) as run:
        run.use_split(sp[0]), run.use_split(sp[1]), run.use_split(splits.load("wod/r2-train"))
        S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
        W = WodRows(rolls)
        model = T.load_pmodel(a.init, dev).train()
        teacher = T.load_pmodel("P0", dev)
        base, new = model.groups()
        tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
        LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev)
        opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}],
                                weight_decay=cfg.wd)
        scaler = torch.amp.GradScaler()
        run.info(f"{a.tag}: arm {a.arm} from {a.init}; WOD visited states {W.cum[-1]} ({len(W.R)} files), logged {W.n} x {W.kl + 1}; "
                 f"navtrain train {len(tr_rows)} dev {len(dv_rows)}; rows/batch vis {n_vis} log {n_log} nav {n_nav}")
        run.summary["n_visited"] = int(W.cum[-1])

        def draw():
            r = rng.choice(tr_rows, n_nav)
            an = rng.random(n_nav) < cfg.d_frac
            return r, an, W.draw(rng, n_vis, n_log), S.front[r]
        pre = ThreadPoolExecutor(2)
        nxt = pre.submit(draw)
        t0, hist = time.time(), []
        nw = n_vis + n_log
        for step in range(a.steps):
            r_np, an_np, (wt, we, wf, wc, wtc), front_nav = nxt.result()
            if step + 1 < a.steps:
                nxt = pre.submit(draw)
            r = torch.as_tensor(r_np, device=dev)
            an = torch.as_tensor(an_np, device=dev)
            g = lambda x, dt=torch.float32: torch.from_numpy(x).to(dev, dt)  # noqa: E731
            wtok = g(wt, torch.float16)
            with torch.no_grad():
                o0 = teacher(wtok, g(we), g(wtc)).float()
            front = torch.cat([front_nav.to(torch.float16), wtok])
            ego = torch.cat([S.ego[r] * (~an)[:, None].float(), g(we)])
            tc = torch.cat([S.tc[r], g(wtc)])
            cam_x = torch.cat([S.cam_x[r], g(wc)])
            fut = torch.cat([S.fut[r], g(wf)])
            has = torch.cat([S.has_fut[r], torch.ones(nw, dtype=torch.bool, device=dev)])
            anchor = torch.cat([an, torch.zeros(nw, dtype=torch.bool, device=dev)])
            t_out = torch.cat([S.t_out[r], o0[:, LS.di]])
            t_plan = torch.cat([S.t_plan[r], o0[:, LS.pi].view(-1, 33, 15)])
            frac = step / a.steps
            for grp in opt.param_groups:
                grp["lr"] = grp["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
            out = model(front, ego, tc).float()
            plan = out[:, LS.pi].view(-1, 33, 15)
            imit = ~anchor & has
            Ls = {"imit": LS.dist(plan[imit], cam_x[imit], fut[imit][..., 0], fut[imit][..., 1], fut[imit][..., 2]).mean()}
            with torch.no_grad():
                Ls["imit_wod"] = LS.dist(plan[-nw:], cam_x[-nw:], fut[-nw:, :, 0], fut[-nw:, :, 1], fut[-nw:, :, 2]).mean()
            tx, ty, tp = T.rear(t_plan[anchor], cam_x[anchor], LS.W)
            Ls["cons"] = LS.dist(plan[anchor], cam_x[anchor], tx, ty, tp).mean() if anchor.any() else out.new_zeros(())
            e = ((out[:, LS.di] - t_out) / tstd).pow(2)
            num = e[:, ~LS.plan_cols].sum(1) + (~imit).float() * e[:, LS.plan_cols].sum(1)
            Ls["distill"] = (num / e.shape[1]).mean()
            total = cfg.lam_i * Ls["imit"] + cfg.lam_c * Ls["cons"] + cfg.lam_d * Ls["distill"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({k: float(v) for k, v in Ls.items()})
            if (step + 1) % 25 == 0 or step + 1 == a.steps:
                m = {k: float(np.mean([h[k] for h in hist])) for k in hist[-1]}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{k}": v for k, v in m.items()} | {"throughput/steps_per_s": (step + 1) / el}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == a.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) + f"; {(step + 1) / el:.2f} it/s")
                    run.status(f"step {step + 1}/{a.steps}")
        ev = T.dev_eval(model.eval(), S, dv_rows, LS.W)
        run.info("dev (navtrain): " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
        run.summary.update({f"dev_{k}": v for k, v in ev.items()} | dict(steps=a.steps, train_s=time.time() - t0))
        if not a.limit:
            torch.save({"model": model.state(), "cfg": asdict(cfg) | {"fw_p2": vars(a)}}, d / "ckpt-final.pt")
            if link.is_symlink():
                link.unlink()
            link.symlink_to(d)
            (d / "dev.json").write_text(json.dumps(ev, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("sets", nargs="+")
    for name in ("collect", "eval"):
        p = sp.add_parser(name)
        p.add_argument("--policy", required=True)
        p.add_argument("--tag", required=True)
        p.add_argument("--sets", default="g0b,g1s")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--shard", default="0/1")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--batch", type=int, default=64)
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True, choices=["X", "C"])
    p.add_argument("--init", required=True)
    p.add_argument("--rolls", default="none")
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke: navtrain rows limited, no checkpoint")
    a = ap.parse_args()
    {"prep": cmd_prep, "collect": cmd_roll, "eval": cmd_roll, "train": cmd_train}[a.cmd](a)
