#!/usr/bin/env python3
"""PAI-STOP-PROBE (plans/2026-10-18-pai-stop-probe-prereg.md): can a probe on the policy's own features read "the human stops ahead" on
PAI, early enough to stop? Box; envs/op-train; one GPU (pool job). Results: results/pai_stop_probe.md.

  rf_stopprobe.py feats   per source: newest-slot frozen tokens (N, 32, 512) fp16, and per checkpoint (RSG-L18-s0, JTN-L18-s0) the forward on the
                          cached 8 slots + ego_in (route memory not fed): hidden_state (512), lead means + lead_prob, plan means -> $O/feats/
  rf_stopprobe.py probe   labels (y5 / y3, cue strata), 5-fold scene cross-fitting per feature set, AUC with scene-cluster bootstrap, the 18-scene
                          first-fire table -> $O/probe/{probe.json, probe.md, scores.npz}

Sources (row caches of onp_cache, tab.npz + tok.npy): log = pait/cache_log (80 train160 scenes, ego follows the human log), p1 = pait/cache/p1rsg
(RSG-L18-s0 own rollouts, 48 test281 scenes), c18 = $O/cache/c18 (RSG-L18-s0 own rollouts under the d330 ACC-GAP serve, the 18 class-C scenes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/op_parity/scripts"), str(R_ / "experiments/route_mem/lib"),
                str(R_ / "experiments/op_adapt_r2/lib"), str(R_ / "experiments/onpol_pai/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = D / "runs/route_follow/stop_probe"
SRC = {"log": D / "runs/route_follow/pait/cache_log", "p1": D / "runs/route_follow/pait/cache/p1rsg", "c18": O / "cache/c18"}
TAGS = ("RSG-L18-s0", "JTN-L18-s0")
S0, V_STOP, V_MIN, E_MAX, LAT_AG = 20, 0.5, 2.0, 3.0, 1.8
SPLIT = "physical_ai_av/stop-probe-scenes"
FOLDS, B_BOOT, FPR, K_FIRE, A_COMF = 5, 2000, 0.10, 3, 3.0


# ---------------------------------------------------------------- features (GPU)
def cmd_feats(a):
    import torch
    import rmem
    from jevdrive.run import Run
    dev = torch.device("cuda")
    (O / "feats").mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "stop-probe-feats", config=vars(a)) as run:
        for s, p in SRC.items():
            z = np.load(p / "tab.npz")
            tok = np.load(p / "tok.npy", mmap_mode="r")
            tix, n = z["tix"], len(z["tix"])
            np.save(O / f"feats/{s}_toknew.npy", np.ascontiguousarray(tok[tix[:, 7]]))
            ego = torch.from_numpy(z["ego_in"].astype(np.float32)).to(dev)
            tc = torch.from_numpy(np.where(z["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)).to(dev)
            for tag in TAGS:
                m = rmem.load_pmodel(tag, dev)
                sl = m.net.slices
                cols = dict(hid=np.arange(sl["hidden_state"].start, sl["hidden_state"].stop),
                            lead=np.arange(sl["lead"].start, sl["lead"].start + (sl["lead"].stop - sl["lead"].start) // 2),
                            lp=np.arange(sl["lead_prob"].start, sl["lead_prob"].stop), plan=np.arange(sl["plan"].start, sl["plan"].start + 495))
                out = {k: np.empty((n, len(c)), np.float32) for k, c in cols.items()}
                with torch.no_grad():
                    for i in range(0, n, 256):
                        H = torch.from_numpy(np.ascontiguousarray(tok[tix[i:i + 256]])).to(dev)
                        o = m(H, ego[i:i + 256], tc[i:i + 256]).float().cpu().numpy()
                        for k, c in cols.items():
                            out[k][i:i + 256] = o[:, c]
                np.savez(O / f"feats/{s}_{tag}.npz", **out)
                run.info(f"{s} {tag}: {n} rows, " + ", ".join(f"{k} {v.shape[1]}" for k, v in out.items()))
                del m
                torch.cuda.empty_cache()
        run.summary["sources"] = list(SRC)


# ---------------------------------------------------------------- labels
def labels(z) -> dict:
    """Per row: a_stop (m along the human path to the first logged speed < V_STOP ahead; inf = none within 100 m), inclusion mask, strata."""
    pv = np.where(z["path_ok"], np.nan_to_num(z["path_v"], nan=1e3), 1e3)[:, S0:]
    hit = pv < V_STOP
    a_stop = np.where(hit.any(1), hit.argmax(1), np.inf).astype(np.float64)
    v0 = z["v0"].astype(np.float64)
    inc = (z["usable"] & ~z["after_event"] & ~z["after_end"] & (v0 >= V_MIN) & (np.abs(np.nan_to_num(z["e0"], nan=99)) <= E_MAX)
           & z["path_ok"][:, S0])
    n = len(v0)
    agent = np.zeros(n, bool)
    P, Pok = z["path"].astype(np.float32), z["path_ok"]
    for i in range(0, n, 256):                                  # box centres projected on the human path (nearest 1 m grid point)
        b, ok = z["box"][i:i + 256, :, :, :2].astype(np.float32), z["box_ok"][i:i + 256]
        d = np.linalg.norm(b[:, :, :, None, :] - P[i:i + 256, None, None], axis=-1)
        d = np.where(Pok[i:i + 256, None, None], d, np.inf)
        j, lat = d.argmin(-1), d.min(-1)
        arc = j - S0
        A = a_stop[i:i + 256, None, None]
        agent[i:i + 256] = (ok & (lat < LAT_AG) & (arc >= A - 2) & (arc <= A + 20)).any((1, 2))
    ss = np.where(z["has_stop"][:, None], z["stop_s"], np.nan)
    line = (np.abs(ss - a_stop[:, None] + 5) <= 10).any(1)      # line in [a_stop - 15, a_stop + 5]
    cue = np.where(agent, "agent", np.where(line, "line", "other"))
    return dict(a_stop=a_stop, v0=v0, inc=inc, cue=cue, y5=a_stop <= 5 * v0, y3=a_stop <= 3 * v0)


def fold_of(scene: str) -> int:
    return int(hashlib.sha1(scene.encode()).hexdigest(), 16) % FOLDS


# ---------------------------------------------------------------- AUC with a scene-cluster bootstrap (weighted Mann-Whitney)
def _auc_w(s_sorted_pos, w, pos):
    """w (B, n) weights in score order; pos (n,) bool -> (B,) AUC (ties broken by the sort; scores are continuous)."""
    wn = np.where(pos, 0.0, w)
    below = np.cumsum(wn, 1) - wn
    num = (np.where(pos, w, 0.0) * below).sum(1)
    return num / np.maximum(np.where(pos, w, 0).sum(1) * wn.sum(1), 1e-12)


def boot_auc(scores: dict, y, grp, seed=0):
    """scores {name: (n,)} on the same rows; -> {name: (auc, lo, hi)}, and the bootstrap draws for paired differences."""
    u, g = np.unique(grp, return_inverse=True)
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B_BOOT)]).astype(np.float64)
    res, draws = {}, {}
    for k, s in scores.items():
        o = np.argsort(s, kind="stable")
        pos, gg = y[o], g[o]
        full = _auc_w(None, np.ones((1, len(o))), pos)[0]
        bd = np.concatenate([_auc_w(None, C[b:b + 100][:, gg], pos) for b in range(0, B_BOOT, 100)])
        draws[k] = bd
        res[k] = (float(full), float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975)))
    return res, draws


# ---------------------------------------------------------------- probes
def fit_lin(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    m.fit((Xtr - mu) / sd, ytr)
    return m.decision_function((Xtr - mu) / sd), m.decision_function((Xte - mu) / sd)


def fit_tok(Ttr, Ktr, ytr, Tte, Kte, seed=0):
    import torch
    import torch.nn as nn
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    mu, sd = Ktr.mean(0), Ktr.std(0) + 1e-6

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(512, elementwise_affine=False)
            self.red = nn.Linear(512, 16)
            self.mlp = nn.Sequential(nn.Linear(32 * 16 + Ktr.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, t, k):
            return self.mlp(torch.cat([self.red(self.ln(t.float())).flatten(1), k], 1))[:, 0]

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor(float((1 - ytr).sum() / max(ytr.sum(), 1)), device=dev)
    Kt, Ke = (torch.from_numpy(((x - mu) / sd).astype(np.float32)).to(dev) for x in (Ktr, Kte))
    yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    g = np.random.default_rng(seed)
    for _ in range(3):
        net.train()
        for b in np.array_split(g.permutation(len(ytr)), max(1, len(ytr) // 256)):
            loss = nn.functional.binary_cross_entropy_with_logits(net(torch.from_numpy(Ttr[b]).to(dev), Kt[b]), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def pred(T, K):
        with torch.no_grad():
            return np.concatenate([net(torch.from_numpy(T[i:i + 1024]).to(dev), K[i:i + 1024]).cpu().numpy() for i in range(0, len(T), 1024)])
    return pred(Ttr, Kt), pred(Tte, Ke)


def cmd_probe(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = O / ("probe_smoke" if a.smoke else "probe")
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "stop-probe" + ("-smoke" if a.smoke else ""), config=vars(a)) as run:
        sp = splits.load(SPLIT)
        run.use_split(sp)
        L, F, T = [], {}, []
        for s, p in SRC.items():
            z = dict(np.load(p / "tab.npz"))
            lb = labels(z)
            lb.update(src=np.full(len(lb["v0"]), s), scene=z["scenes"][z["sid"]], t0=z["t0_us"])
            r = np.flatnonzero(lb["inc"])
            if a.smoke:
                r = r[::10]
            L.append({k: v[r] for k, v in lb.items()})
            T.append(np.load(O / f"feats/{s}_toknew.npy", mmap_mode="r")[r])
            kin = z["ego_in"][r].astype(np.float32)
            f = {"KIN": kin}
            for tag, nm in zip(TAGS, ("RSG", "JTN")):
                q = np.load(O / f"feats/{s}_{tag}.npz")
                if nm == "RSG":
                    f["LEAD"] = np.c_[kin, q["lead"][r], q["lp"][r]]
                    f["PLAN"] = np.c_[kin, q["plan"][r]]
                f[f"HID-{nm}"] = np.c_[kin, q["hid"][r]]
            f["TOK-lin"] = np.c_[kin, T[-1].astype(np.float32).mean(1)]
            for k, v in f.items():
                F.setdefault(k, []).append(v)
        L = {k: np.concatenate([x[k] for x in L]) for k in L[0]}
        F = {k: np.concatenate(v) for k, v in F.items()}
        TOK = np.concatenate(T)
        assert set(L["scene"]) <= set(sp.members), "a scene outside the registered split"
        fold = np.array([fold_of(x) for x in L["scene"]])
        n = len(fold)
        run.info(f"rows {n}: " + ", ".join(f"{s} {int((L['src'] == s).sum())}" for s in SRC) + f"; y5 {int(L['y5'].sum())}, y3 {int(L['y3'].sum())}")
        S, TAU = {}, {}
        for yk in ("y5", "y3"):
            y = L[yk].astype(int)
            for fk in list(F) + ["TOK"]:
                sc, tau = np.zeros(n), np.zeros(FOLDS)
                for k in range(FOLDS):
                    tr, te = fold != k, fold == k
                    if fk == "TOK":
                        s_tr, s_te = fit_tok(TOK[tr], F["KIN"][tr], y[tr], TOK[te], F["KIN"][te])
                    else:
                        s_tr, s_te = fit_lin(F[fk][tr], y[tr], F[fk][te])
                    sc[te] = s_te
                    tau[k] = np.quantile(s_tr[y[tr] == 0], 1 - FPR)
                S[(yk, fk)], TAU[(yk, fk)] = sc, tau
                run.info(f"{yk} {fk} done")
        np.savez(out / "scores.npz", **{f"{yk}|{fk}": v for (yk, fk), v in S.items()}, **{k: v for k, v in L.items() if k != "cue"},
                 cue=L["cue"], fold=fold)
        # AUC table: label x source x stratum x feature set
        own = np.isin(L["src"], ["p1", "c18"])
        srcs = {"own (p1 + c18)": own, "log (human states)": L["src"] == "log", "all": np.ones(n, bool)}
        strata = {"all": None, "agent": ["agent"], "no lead (line + other)": ["line", "other"], "line": ["line"], "other": ["other"]}
        fks = list(F) + ["TOK"]
        tab = []
        for yk in ("y5", "y3"):
            y = L[yk]
            for sn, sm in srcs.items():
                for tn, tv in strata.items():
                    keep = sm & (~y | (np.ones(n, bool) if tv is None else np.isin(L["cue"], tv)))
                    npos, nneg = int((y & keep).sum()), int((~y & keep).sum())
                    if npos < 5:
                        tab.append(dict(label=yk, source=sn, stratum=tn, n_pos=npos, n_neg=nneg))
                        continue
                    res, dr = boot_auc({fk: S[(yk, fk)][keep] for fk in fks}, y[keep], L["scene"][keep])
                    row = dict(label=yk, source=sn, stratum=tn, n_pos=npos, n_neg=nneg,
                               n_pos_scenes=int(len(np.unique(L["scene"][keep & y]))), auc={k: v for k, v in res.items()})
                    for b in ("KIN", "LEAD"):
                        d = dr["TOK"] - dr[b]
                        row[f"TOK-{b}"] = (res["TOK"][0] - res[b][0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
                    tab.append(row)
                    run.info(f"{yk} | {sn} | {tn}: TOK {res['TOK'][0]:.3f} KIN {res['KIN'][0]:.3f} LEAD {res['LEAD'][0]:.3f}")
        # the 18 named scenes: first fire before the human's stop point
        named = []
        c = np.flatnonzero(L["src"] == "c18")
        for sc_ in sorted(set(L["scene"][c])):
            r = c[L["scene"][c] == sc_]
            r = r[np.argsort(L["t0"][r])]
            r = r[np.isfinite(L["a_stop"][r]) & (L["a_stop"][r] > 0)]
            e = dict(scene=sc_, n=int(len(r)), cue=(lambda u: u[0][np.argmax(u[1])] if len(u[0]) else "")(np.unique(L["cue"][r][L["y5"][r]], return_counts=True)))
            for fk in ("TOK", "LEAD", "KIN", "HID-RSG"):
                s, k_ = S[("y5", fk)][r], fold[r][0] if len(r) else 0
                on = s >= TAU[("y5", fk)][k_]
                run_ = np.convolve(on.astype(int), np.ones(K_FIRE, int), "valid") == K_FIRE if len(on) >= K_FIRE else np.zeros(0, bool)
                w5 = L["y5"][r]
                if run_.any():
                    i = int(np.argmax(run_))
                    d, v = float(L["a_stop"][r][i]), float(L["v0"][r][i])
                    e[fk] = dict(d=d, v=v, need=v * v / (2 * A_COMF), ok=bool(d >= v * v / (2 * A_COMF)), hit5=float(on[w5].mean()) if w5.any() else None)
                else:
                    e[fk] = dict(d=None, v=None, need=None, ok=False, hit5=float(on[w5].mean()) if w5.any() else None)
            named.append(e)
        json.dump(dict(rows=n, table=tab, named=named, tau={f"{a}|{b}": v.tolist() for (a, b), v in TAU.items()}), open(out / "probe.json", "w"), indent=1)
        run.summary.update(rows=n, out=str(out / "probe.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("feats", "probe"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dict(feats=cmd_feats, probe=cmd_probe)[a.cmd](a)
