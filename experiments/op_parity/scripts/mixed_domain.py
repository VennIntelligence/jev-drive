"""op_parity mixed-domain lane (plans/2026-10-08-mixed-domain-prereg.md, results/mixed_domain.md). The principle under test: standstill / launch is
decided by the scene, not by a per-domain constant of the adapter bias. Two forms: P2HG = the navtrain recipe (P2H10) with the standstill rule trained
in (pp_train.py --stop-gate 0.5; served with the same rule: navtest `:sg`, HUGSIM parity opt stop_gate, WOD gated bias), the main read; MX = one P2
adapter trained on navtrain + WOD r2-train (pp_train.py --wod-mass / --wod-slots). Read on navtest, HUGSIM 64 and WOD val against the stored drivers
(P2H10, WP2, the wod-launch lane's WLG) and shipped. A tag with `:sg` is served through the stop gate (WOD prediction dirs carry the bare tag).

  teacher8  (op-train env, one GPU) shipped Cinque (port fp16) on the 8 newest slots of every wod_* cache row (the oldest slot zero and invalid, as
            NAVSIM rows) -> cache/<data>/teacher8.npz {out, plan, di, pi}; checks that the same path on 9 slots reproduces teacher.npz
  bias      (op-train env, CPU) adapter bias of each tag on the WOD val frames (rater + extra, pp_wod.wod_ego) and on the navtest tokens: the
            decision 162 decomposition (constant = mean over frames, residual = what moves with the ego input), all frames and standstill frames,
            per domain, and the distance between the two domains' constants -> runs/op_parity/mixed/bias_stats.json
            --harness: also the serving variants of the tag on WOD (zero / biasmean / biasresid) -> runs/op_parity/mixed/bias-<tag>_<var>.npz
  gate      (jevdrive env, CPU) MX pilot gate: MX-pilot against both single-domain pilots on both boards -> results/mixed_domain/<name>_gate.json;
            exit 3 = clear negative (below both single-domain pilots on both boards, every CI upper bound < 0)
  gate-ng   (jevdrive env, CPU) P2HG pilot gate on navtest against the ungated pilot of the same recipe (RH0-F-s0): pass = EPDMS diff >= -0.50 and
            the v0 < 0.5 stratum diff >= -5.0 (the serving-side gate of decision 167 cost -1.50 / -19.4) -> <name>_gate.json; exit 3 = stop
  report    (jevdrive env, CPU) --name pilot | full: WOD strata + ADE, navtest strata + sub-scores, HUGSIM, bias tables -> results/mixed_domain/

Conventions: WOD = wod_launch_report.Ctx (479 rater frames, cluster-mean RFS, paired bootstrap over sequences, B 4 000; strata of decisions 163 / 164);
navtest = stop_gate_xboard (12 146 tokens, paired cluster bootstrap over logs, strata v0 < 0.5 / launch / moving / turn > 20 deg); HUGSIM = 64 scenarios,
spec_plan_smooth, paired over scenarios. An arm group is the per-unit mean of its seeds.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, glob, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = _R / "experiments/op_parity/results/mixed_domain"
D = data_dir() / "runs"
MIX = D / "op_parity/mixed"
V_STOP = 0.5
PRESET = "spec_plan_smooth"


def groups(specs):
    """['MX=MX-F-s0+MX-F-s1', 'shipped'] -> {'MX': ['MX-F-s0', 'MX-F-s1'], 'shipped': ['shipped']}"""
    out = {}
    for s in specs:
        k, _, v = s.partition("=")
        out[k] = (v or k).split("+")
    return out


# ---------------------------------------------------------------- teacher8
def cmd_teacher8(a):
    import torch
    import pp_train as T
    dev = torch.device("cuda")
    m = T.PModel("P0", pol=False).to(dev).eval()
    st = {}
    for d in a.data:
        cd = D / "op_parity/cache" / d
        z = np.load(cd / "teacher.npz")
        di, pi = z["di"], z["pi"]
        mm = T.IndexedMM(cd)
        n = len(mm) if not a.limit else min(a.limit, len(mm))
        out, plan = np.zeros((n, len(di)), np.float32), np.zeros((n, 33, 15), np.float32)
        d9, tc = [], torch.tensor([[1.0, 0.0]], device=dev)
        with torch.no_grad():
            for i in range(0, n, 128):
                f = torch.from_numpy(np.ascontiguousarray(mm[np.arange(i, min(i + 128, n))])).to(dev)
                ego = torch.zeros(len(f), 20, device=dev)
                o = m(f[:, 1:], ego, tc.expand(len(f), 2), inputs_on=False).float()
                out[i:i + len(f)], plan[i:i + len(f)] = o[:, di].cpu().numpy(), o[:, pi].view(-1, 33, 15).cpu().numpy()
                if i % (128 * 32) == 0:                                       # the same path on all 9 slots must reproduce the stored teacher
                    o9 = m(f, ego, tc.expand(len(f), 2), inputs_on=False).float()[:, pi].view(-1, 33, 15).cpu().numpy()
                    d9.append(np.linalg.norm(o9[..., :2] - z["plan"][i:i + len(f), :, :2], axis=-1).mean(1))
        e = np.linalg.norm(plan[..., :2] - z["plan"][:n, :, :2], axis=-1)
        st[d] = dict(rows=int(n), path_check_9slot_plan_xy_mean_m=float(np.concatenate(d9).mean()), slot8_vs_slot9_plan_xy_mean_m=float(e.mean()),
                     slot8_vs_slot9_plan_xy_p99_m=float(np.percentile(e.mean(1), 99)), slot8_vs_slot9_at_4s_m=float(e[:, 20].mean()))
        print(d, json.dumps(st[d]), flush=True)
        assert st[d]["path_check_9slot_plan_xy_mean_m"] < 0.02, "the PModel path does not reproduce the stored 9-slot teacher"
        if not a.limit:
            np.savez(cd / "teacher8.npz", out=out, plan=plan, di=di, pi=pi)
    MIX.mkdir(parents=True, exist_ok=True)
    if not a.limit:
        (MIX / "teacher8.json").write_text(json.dumps(st, indent=1))


# ---------------------------------------------------------------- bias decomposition
def _decomp(b, stop):
    """b (n, 32, 512) fp32 biases, stop (n,) bool -> the decision 162 numbers: rms of the bias, of its mean over frames (the constant) and of the
    residual around it; the same on the standstill frames; the standstill mean against the all-frame mean."""
    rms = lambda x: float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))  # noqa: E731
    mu = b.mean(0)
    r = dict(n=int(len(b)), rms=rms(b), const_rms=rms(mu), resid_rms=rms(b - mu), n_stop=int(stop.sum()))
    r["const_share_of_ms"] = r["const_rms"] ** 2 / max(r["rms"] ** 2, 1e-12)
    if stop.any():
        ms = b[stop].mean(0)
        r |= dict(stop_rms=rms(b[stop]), stop_const_rms=rms(ms), stop_resid_rms=rms(b[stop] - ms), stop_const_minus_all_const_rms=rms(ms - mu))
        r["_mu_stop"] = ms
    r["_mu"] = mu
    return r


def _cos(x, y):
    x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))


def cmd_bias(a):
    import torch
    import pp_hugsim as H
    import pp_train as T
    from jevdrive import wod_zeroshot as Z
    from pp_wod import wod_ego
    S = Z.load_sets()
    names = np.concatenate([S[k]["name"].astype(str) for k in ("rater", "extra")])
    ego_w, _ = wod_ego(np.concatenate([S[k]["past"] for k in ("rater", "extra")]), np.concatenate([S[k]["intent"] for k in ("rater", "extra")]))
    ego_n = np.load(D / "op_parity/cache/lb_navtest/tab.npz")["ego"].astype(np.float32)
    e0 = np.zeros((1, 20), np.float32)
    e0[0, 0], e0[0, 2] = 1.0, 1.0                                               # present, command straight, everything else 0: a parked car
    MIX.mkdir(parents=True, exist_ok=True)
    sf = MIX / "bias_stats.json"
    stats = json.loads(sf.read_text()) if sf.exists() else {}
    rms = lambda x: float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))  # noqa: E731
    for tag in a.tags:
        gate = float(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["cfg"].get("stop_gate", 0.0))
        m = H.pmodel(tag, torch.device("cpu"))
        assert m.adapter is not None and not m.adapter.use_side, tag

        def run(e):
            e = np.ascontiguousarray(e, np.float32)
            with torch.no_grad():
                b = np.concatenate([m.adapter(torch.from_numpy(e[i:i + 512]), None, None).numpy() for i in range(0, len(e), 512)])
            if gate > 0:
                b[e[:, 4] * 10.0 < gate] = 0
            return b
        bw, bn = run(ego_w), run(ego_n)
        dw, dn = _decomp(bw, ego_w[:, 4] * 10.0 < V_STOP), _decomp(bn, ego_n[:, 4] * 10.0 < V_STOP)
        b0 = run(e0)[0]
        x = dict(stop_gate=gate, parked_input_rms=rms(b0),
                 const_wod_vs_nav_rms_diff=rms(dw["_mu"] - dn["_mu"]), const_wod_vs_nav_cos=_cos(dw["_mu"], dn["_mu"]),
                 stop_const_wod_vs_nav_rms_diff=rms(dw["_mu_stop"] - dn["_mu_stop"]), stop_const_wod_vs_nav_cos=_cos(dw["_mu_stop"], dn["_mu_stop"]),
                 parked_vs_wod_stop_const_rms_diff=rms(b0 - dw["_mu_stop"]), parked_vs_nav_stop_const_rms_diff=rms(b0 - dn["_mu_stop"]))
        stats[tag] = dict(wod={k: v for k, v in dw.items() if not k.startswith("_")}, navtest={k: v for k, v in dn.items() if not k.startswith("_")}, cross=x)
        print(tag, json.dumps(stats[tag]), flush=True)
        if a.harness:
            for var, b in (("zero", np.zeros_like(bw)), ("biasmean", np.broadcast_to(dw["_mu"], bw.shape)), ("biasresid", bw - dw["_mu"])):
                np.savez(MIX / f"bias-{tag}_{var}.npz", names=names, bias=np.ascontiguousarray(b).astype(np.float16), ego=ego_w)
    sf.write_text(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- WOD
class Wod:
    def __init__(self):
        from wod_launch_report import Ctx
        self.C = C = Ctx()
        self.st = {"all": C.st["all"], "standstill (v0 < 0.5)": C.st["stopped"], "SS (standstill, log stays < 1 m)": C.st["SS (stopped, log stays)"],
                   "SL (standstill, log moves)": C.st["SL (stopped, log moves)"], "launch (v0 < 2, log 5 s > 5 m)": C.st["launch (v<2, log>5m)"],
                   "moving (v0 >= 0.5)": C.st["moving (v>=0.5)"], "turn intent": np.asarray(C.intent) >= 2, "night": C.st["night"], "day": C.st["day"]}
        self.P, self.sco = {}, {}

    def load(self, g):
        C = self.C
        for k, tags in g.items():
            self.P[k] = [C.fut if t == "log" else C.preds(t.partition(":")[0], True) for t in tags]
            self.sco[k] = [C.rfs(p) for p in self.P[k]]

    def rfs(self, k):
        return np.mean(self.sco[k], 0)

    def d(self, k, i):
        return np.mean([np.linalg.norm(p[: self.C.n, i], axis=-1) for p in self.P[k]], 0)

    def strata(self, arms, refs):
        C, rows = self.C, []
        for nm, m in self.st.items():
            for k in arms:
                row = {"stratum": nm, "n": int(m.sum()), "arm": k, "RFS": C.cm(self.rfs(k), m), "d5 median (m)": float(np.median(self.d(k, 19)[m])),
                       "d5 mean (m)": float(self.d(k, 19)[m].mean()), "d3 mean (m)": float(self.d(k, 11)[m].mean())}
                for y in refs:
                    if y != k:
                        c = C.ci(self.rfs(k) - self.rfs(y), m)
                        row[f"- {y}"], row[f"- {y} lo"], row[f"- {y} hi"] = c
                if len(self.sco[k]) == 2:
                    row["seeds"] = "/".join(f"{C.cm(s, m):.3f}" for s in self.sco[k])
                rows.append(row)
        return rows

    def ade(self, arms, refs):
        C, rows = self.C, []
        err = {k: np.mean([np.linalg.norm(p - C.fut, axis=-1) for p in self.P[k]], 0) for k in arms if k != "log"}
        for nm, sl in (("ADE@3s", slice(0, 12)), ("ADE@5s", slice(0, 20))):
            e = {k: x[:, sl].mean(1) for k, x in err.items()}
            for k in e:
                row = {"metric": nm, "arm": k, "value (m)": float(e[k].mean())}
                for y in refs:
                    if y != k and y in e:
                        dd = e[k] - e[y]
                        b = (C.Ka @ dd) / C.Ka.sum(1)
                        row[f"- {y}"], row[f"- {y} lo"], row[f"- {y} hi"] = float(dd.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
                rows.append(row)
        return rows


# ---------------------------------------------------------------- navtest
def nav_units(tag):
    """Per-token devkit scores of a navtest run: the bench run of `tag` (spec, e.g. MX-F-s0 or MX-F-s0:dn), else the stored legacy CSV of the tag."""
    import pandas as pd
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import SUBS, read_devkit_csv
    f = D / "bench/navtest" / resolve(tag).key("navtest") / "units.csv"
    if f.exists():
        return pd.read_csv(f).set_index("token")[["score", *SUBS]]
    name, _, opt = tag.partition(":")
    assert not opt, f"no bench run for {tag}"
    fs = sorted(glob.glob(str(D / f"navsim/eval/v2_navtest_opi_lb_navtest_warp-cinque_PP{name}__base/*/*.csv")))
    assert fs, f"no navtest result for {tag}"
    t, _ = read_devkit_csv(fs[-1])
    t = t.rename(columns={v: k for k, v in SUBS.items()}) if "NC" not in t else t
    return t[["score", *SUBS]]


class Nav:
    def __init__(self):
        import pandas as pd
        self.toks = t = pd.read_parquet(D / "op_probe/joint/navtest_tokens.parquet").set_index("token")
        self.g = t.log.to_numpy()
        self.st = {"all": np.ones(len(t), bool), "v0 < 0.5 m/s": (t.v0 < V_STOP).to_numpy(), "launch (v0 < 2, logged 4 s > 5 m)": ((t.v0 < 2) & (t.path_len > 5)).to_numpy(),
                   "stay (v0 < 0.5, logged 4 s <= 5 m)": ((t.v0 < V_STOP) & (t.path_len <= 5)).to_numpy(),
                   "moving (v0 >= 0.5)": (t.v0 >= V_STOP).to_numpy(), "turn > 20 deg": (t.dyaw.abs() > 20).to_numpy()}
        self.U = {}

    def load(self, g):
        for k, tags in g.items():
            try:
                self.U[k] = [nav_units("P0" if t == "shipped" else t).reindex(self.toks.index) for t in tags]
            except AssertionError as e:                                          # a driver never run on navtest (WOD-only arms): left out of the tables
                print(f"navtest: {k} skipped ({e})", flush=True)

    def col(self, k, c="score"):
        return 100 * np.mean([u[c].to_numpy(float) for u in self.U[k]], 0)

    def strata(self, arms, refs):
        from jevdrive.stats import paired
        rows = []
        for nm, m in self.st.items():
            for k in arms:
                row = {"stratum": nm, "n": int(m.sum()), "arm": k, "EPDMS": float(np.nanmean(self.col(k)[m]))}
                for y in refs:
                    if y != k:
                        r = paired(self.col(k)[m], self.col(y)[m], self.g[m])
                        row[f"- {y}"], row[f"- {y} lo"], row[f"- {y} hi"] = r["mean"], r["lo"], r["hi"]
                if len(self.U[k]) == 2:
                    row["seeds"] = "/".join(f"{100 * np.nanmean(u.score.to_numpy(float)[m]):.2f}" for u in self.U[k])
                rows.append(row)
        return rows

    def subs(self, arms, refs):
        from jevdrive.bench.navsim import SUBS
        from jevdrive.stats import paired
        rows = []
        for c in ["score", *SUBS]:
            for k in arms:
                row = {"metric": "EPDMS" if c == "score" else c, "arm": k, "value": float(np.nanmean(self.col(k, c)))}
                for y in refs:
                    if y != k:
                        r = paired(self.col(k, c), self.col(y, c), self.g)
                        row[f"- {y}"], row[f"- {y} lo"], row[f"- {y} hi"] = r["mean"], r["lo"], r["hi"]
                rows.append(row)
        return rows


# ---------------------------------------------------------------- HUGSIM
def hs_units(tag):
    """units.csv frames of a model's spec_plan_smooth runs: the default-identity bench run, else the stored rr1 / rr2 repeats (mean taken by the caller).
    `<tag>:sg`: the run of the tag whose parity opts carry stop_gate = 0.5."""
    import pandas as pd
    tag, _, opt = tag.partition(":")
    if opt == "sg":
        ds = [f.parent for f in sorted((D / "bench/hugsim").glob(f"{tag}_{PRESET}-c*/config.json"))
              if json.loads(f.read_text())["opts"].get("parity", {}).get("stop_gate") == V_STOP]
        return [pd.read_csv(d / "units.csv").set_index("scenario") for d in ds if (d / "units.csv").exists() and (d / "DONE").exists()][:1]
    ds = [D / "bench/hugsim" / f"{tag}_{PRESET}"] + [D / "bench/hugsim" / f"{tag}_{PRESET}-rr{i}" for i in (1, 2)]
    us = [pd.read_csv(d / "units.csv").set_index("scenario") for d in ds if (d / "units.csv").exists() and (d / "DONE").exists()]
    return us[:1] if (ds[0] / "units.csv").exists() and (ds[0] / "DONE").exists() else us


def hugsim_tables(g, refs):
    import pandas as pd
    from jevdrive.stats import paired
    runs = {k: [u for t in tags for u in hs_units(t)] for k, tags in g.items()}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return [], []
    idx = sorted(set.intersection(*[set(u.index) for v in runs.values() for u in v]))
    hd = {k: np.mean([u.reindex(idx).hdscore.to_numpy(float) for u in v], 0) for k, v in runs.items()}
    b = lambda s: s.astype(str).str.lower().isin(["true", "1"])  # noqa: E731
    rows = []
    for k, v in runs.items():
        us = [u.reindex(idx) for u in v]
        row = {"arm": k, "runs": len(us), "scenarios": len(idx), "HD": float(hd[k].mean())}
        for y in refs:
            if y != k and y in hd:
                r = paired(hd[k], hd[y])
                row[f"- {y}"], row[f"- {y} lo"], row[f"- {y} hi"] = r["mean"], r["lo"], r["hi"]
        row |= {"launch stall": float(np.mean([b(u.launch_stall).sum() for u in us])), "stuck": float(np.mean([b(u.stuck).sum() for u in us])),
                "spin": float(np.mean([b(u.spin).sum() for u in us])), "fg collision": float(np.mean([(u.cls == "fg_coll").sum() for u in us])),
                "bg collision": float(np.mean([(u.cls == "bg_coll").sum() for u in us])), "off route": float(np.mean([(u.cls == "off_route").sum() for u in us])),
                "complete": float(np.mean([(u.cls == "complete").sum() for u in us])),
                "per run HD": "/".join(f"{u.hdscore.mean():.3f}" for u in us), "per run stall,stuck": " ".join(f"{int(b(u.launch_stall).sum())},{int(b(u.stuck).sum())}" for u in us)}
        rows.append(row)
    per = pd.DataFrame({"scenario": idx, **{f"hd_{k}": hd[k] for k in hd}})
    return rows, per


# ---------------------------------------------------------------- gate / report
def cmd_gate(a):
    from jevdrive.stats import paired
    W, N = Wod(), Nav()
    g = {"MX": [a.arm], "WODP": [a.wod_ref], "NAVP": [a.nav_ref], "shipped": ["shipped"]}
    W.load(g)
    N.load(g)
    out = {"arm": a.arm, "wod_ref": a.wod_ref, "nav_ref": a.nav_ref}
    for y in ("WODP", "NAVP", "shipped"):
        out[f"wod: MX - {y}"] = list(W.C.ci(W.rfs("MX") - W.rfs(y)))
        r = paired(N.col("MX"), N.col(y), N.g)
        out[f"navtest: MX - {y}"] = [r["mean"], r["lo"], r["hi"]]
    out["RFS"] = {k: W.C.cm(W.rfs(k)) for k in g}
    out["EPDMS"] = {k: float(np.nanmean(N.col(k))) for k in g}
    neg = [out[f"{b}: MX - {y}"][2] < 0 for b in ("wod", "navtest") for y in ("WODP", "NAVP")]
    out["clear_negative"] = bool(all(neg))
    out["pass"] = not out["clear_negative"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    _sys.exit(0 if out["pass"] else 3)


def cmd_gate_ng(a):
    N = Nav()
    N.load({"NG": [a.arm], "REF": [a.ref]})
    rows = {r["stratum"]: r for r in N.strata(["NG"], ["REF"])}
    out = {"arm": a.arm, "ref": a.ref, "EPDMS": {k: float(np.nanmean(N.col(k))) for k in ("NG", "REF")},
           "strata": {k: [r["- REF"], r["- REF lo"], r["- REF hi"]] for k, r in rows.items()},
           "sub-scores": {r["metric"]: [r["- REF"], r["- REF lo"], r["- REF hi"]] for r in N.subs(["NG"], ["REF"])}}
    out["pass"] = bool(rows["all"]["- REF"] >= -0.50 and rows["v0 < 0.5 m/s"]["- REF"] >= -5.0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    _sys.exit(0 if out["pass"] else 3)


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    g = groups(a.arms)
    arms, refs = list(g), a.refs
    with Run("op_parity", f"mixed-domain-report-{a.name}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val")), run.use_split(splits.load("navsim/navtest"))
        OUT.mkdir(parents=True, exist_ok=True)
        if "wod" in a.boards:
            W = Wod()
            W.load(g | {"log": ["log"]})
            stats.write_table(W.strata(arms + ["log"], refs), OUT / f"{a.name}_wod_strata")
            stats.write_table(W.ade(arms, refs), OUT / f"{a.name}_wod_ade")
            pd.DataFrame({"name": W.C.names[: W.C.n], "v0": W.C.v0, "logd5": W.C.logd, **{f"rfs_{k}": W.rfs(k) for k in arms + ["log"]},
                          **{f"d5_{k}": W.d(k, 19) for k in arms + ["log"]}}).to_csv(OUT / f"{a.name}_wod_frames.csv", index=False, float_format="%.4f")
            if a.wod_vars:                                                       # serving variants of the adapter bias (decision 162 method), per group
                gv = groups(a.wod_vars)
                W.load(gv)
                stats.write_table(W.strata(list(gv), [a.wod_vars_ref] + refs[:1]), OUT / f"{a.name}_wod_biasvars")
        if "navtest" in a.boards:
            N = Nav()
            gn = g | groups(a.nav_extra)
            N.load(gn)
            na, nr = [k for k in gn if k in N.U], [k for k in refs if k in N.U]
            stats.write_table(N.strata(na, nr), OUT / f"{a.name}_navtest_strata", floatfmt=".2f")
            stats.write_table(N.subs(na, nr), OUT / f"{a.name}_navtest_subscores", floatfmt=".2f")
        if "hugsim" in a.boards:
            rows, per = hugsim_tables({k: v for k, v in g.items() if k != "shipped"} | groups(a.hugsim_extra), refs)
            if rows:
                stats.write_table(rows, OUT / f"{a.name}_hugsim")
                per.to_csv(OUT / f"{a.name}_hugsim_scenarios.csv", index=False, float_format="%.4f")
        sf = MIX / "bias_stats.json"
        if sf.exists():
            S = json.loads(sf.read_text())
            rows = []
            for tag, s in S.items():
                for dom in ("wod", "navtest"):
                    rows.append({"tag": tag, "domain": dom} | s[dom])
            stats.write_table(rows, OUT / "bias_decomposition")
            stats.write_table([{"tag": t} | s["cross"] for t, s in S.items()], OUT / "bias_cross_domain")
        run.info("tables under %s", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("teacher8")
    p.add_argument("--data", nargs="+", default=["wod_pilot", "wod_r2"])
    p.add_argument("--limit", type=int, default=0, help="first n rows, nothing written (smoke)")
    p = sp.add_parser("bias")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--harness", action="store_true", help="also write the WOD serving variants zero / biasmean / biasresid")
    p = sp.add_parser("gate")
    p.add_argument("--name", default="pilot")
    p.add_argument("--arm", required=True)
    p.add_argument("--wod-ref", default="WP2-pilot-s0")
    p.add_argument("--nav-ref", default="P2-W-s0")
    p = sp.add_parser("gate-ng")
    p.add_argument("--name", default="pilot_ng")
    p.add_argument("--arm", required=True)
    p.add_argument("--ref", default="RH0-F-s0")
    p = sp.add_parser("report")
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="GROUP=tag[+tag] (the per-unit seed mean); `shipped` = stored shipped Cinque (navtest: P0 under W)")
    p.add_argument("--refs", nargs="+", required=True, help="groups every arm is differenced against")
    p.add_argument("--boards", nargs="+", default=["wod", "navtest", "hugsim"])
    p.add_argument("--wod-vars", nargs="*", default=[], help="WOD serving-variant groups GROUP=predtag[+predtag]")
    p.add_argument("--wod-vars-ref", default="MX")
    p.add_argument("--nav-extra", nargs="*", default=[], help="extra navtest groups (e.g. MXdn=MX-F-s0:dn+MX-F-s1:dn)")
    p.add_argument("--hugsim-extra", nargs="*", default=[])
    a = ap.parse_args()
    {"teacher8": cmd_teacher8, "bias": cmd_bias, "gate": cmd_gate, "gate-ng": cmd_gate_ng, "report": cmd_report}[a.cmd](a)
