"""Image-command fine-tune (Q2): port plans on a trunk bank and the paired adapted-vs-original report (op-train venv).
Plan: ../plans/2026-10-04-img-cmd-ft-prereg.md.

  plans   --models O <run tag> ... --pool eval|train   policy forward of each model on every bank variant -> ft/plans/<model>/<pool>.npz
          (plan mu (n, 33, 15)) and, for the eval pool, raw/nav/port-<model>.npz in img_run's layout (plan in the t0 rear-axle frame)
  report  --model <tag> [--ref O] --pool eval|dev     img_report's per-sample metrics for both models, paired cluster bootstrap by log
          of model - ref per family (Delta, uptake, fixed-set correct, toward per class, dx; straight-frame lateral error),
          the guards (no-overlay plan drift, no-overlay move toward the taken branch) and, for O, port vs TensorRT on `none`
          -> ft/report/<model>_vs_<ref>_<pool>.{csv,md}
"""
import argparse, json, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_common_cause" / "scripts"), str(Path(__file__).resolve().parent)]
import img_overlay as O  # noqa: E402
import img_report as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_img_cmd"
FT = ROOT / "ft"
TAU = 4.0


def bank(pool):
    d = FT / "bank" / pool
    with np.load(d / "var.npz") as z:
        v = {k: z[k] for k in z.files}
    return np.load(d / "trunk.npy", mmap_mode="r"), v


def ckpt(model):
    return None if model == "O" else FT / "runs" / model / "ckpt-final.pt"


def port_plans(model, T, v, dev, bs=96):
    import torch
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from jevdrive import op_adapt as A
    pi = A.plan_index(model.net.slices)
    n = len(T)
    out = np.zeros((n, 33, 15), np.float32)
    sv = torch.from_numpy(np.asarray(v["slot_valid"])).to(dev)
    with torch.no_grad():
        for i in range(0, n, bs):
            j = slice(i, min(i + bs, n))
            tr = torch.from_numpy(np.asarray(T[j])).to(dev)
            o = model(tr, sv[None].expand(len(tr), -1), torch.from_numpy(v["tc"][j]).to(dev).to(model.net.dtype))["outputs"].float()
            out[j] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
    return out


def cmd_plans(a):
    import torch
    import img_run
    from experiments.op_adapt_l.lib import op_adapt_l as L
    dev = torch.device("cuda")
    T, v = bank(a.pool)
    for name in a.models:
        p = FT / "plans" / name / f"{a.pool}.npz"
        if p.exists() and not a.force:
            print(name, a.pool, "exists")
            continue
        m = L.load_model(ckpt(name), dev)
        mu = port_plans(m, T, v, dev)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, mu=mu, token=v["token"], fam=v["fam"], cmd=v["cmd"])
        if a.pool.startswith("eval"):
            xy, yaw = zip(*[img_run.to_rear(q[:, 0:3], q[:, 11], c) for q, c in zip(mu, v["cam"])])
            np.savez(ROOT / "raw" / "nav" / f"port-{name}{a.pool[4:]}.npz", token=v["token"], fam=v["fam"], cmd=v["cmd"],
                     plan_pos=np.array(xy, np.float32), plan_yaw=np.array(yaw, np.float32), plan_v=mu[:, :, 3])
        print(f"{name} {a.pool}: {len(mu)} plans", flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- report
def per_sample(G, runs_by_tok):
    rows, srows = [], []
    for tok, runs in runs_by_tok.items():
        s = G[tok]
        base = dict(token=tok, log=s["log"], v=s["v"], vbin=R.vbin(s["v"]), taken=s["taken"])
        if s["kind"] == "junction":
            rows += [dict(base, **r) for r in R.sample_rows(s, runs) if r["tau"] == TAU]
        else:
            srows += [dict(base, **r) for r in R.straight_rows(s, runs)]
    return pd.DataFrame(rows), pd.DataFrame(srows)


def load_runs(model, pool):
    """{token: {(fam, cmd): (plan_pos (33, 2), plan_yaw (33,))}} in the t0 rear-axle frame."""
    import img_run
    if pool.startswith("eval"):
        z = np.load(ROOT / "raw" / "nav" / f"port-{model}{pool[4:]}.npz")
        P, Y = z["plan_pos"], z["plan_yaw"]
        tok, fam, cmd = z["token"], z["fam"], z["cmd"]
    else:
        z = np.load(FT / "plans" / model / "train.npz")
        _, v = bank("train")
        m = np.isin(v["token"], list(DEV_TOK))
        P, Y = zip(*[img_run.to_rear(q[:, 0:3], q[:, 11], c) for q, c in zip(z["mu"][m], v["cam"][m])])
        tok, fam, cmd = v["token"][m], v["fam"][m], v["cmd"][m]
    out = {}
    for i in range(len(tok)):
        out.setdefault(str(tok[i]), {})[(str(fam[i]), str(cmd[i]))] = (np.asarray(P[i]), np.asarray(Y[i]))
    return out


DEV_TOK = set()


def paired_ratio(da, db, num, den, B=2000):
    """ratio-of-sums (a) - ratio-of-sums (b) with one cluster resample for both (a, b aligned on token)."""
    g = pd.concat([da.groupby("log")[[num, den]].sum().add_suffix("_a"), db.groupby("log")[[num, den]].sum().add_suffix("_b")], axis=1).fillna(0)
    A_, Ad, B_, Bd = (g[c].to_numpy() for c in (f"{num}_a", f"{den}_a", f"{num}_b", f"{den}_b"))
    k = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    ra = A_[k].sum(1) / np.maximum(Ad[k].sum(1), 1e-9)
    rb = B_[k].sum(1) / np.maximum(Bd[k].sum(1), 1e-9)
    pa, pb = A_.sum() / max(Ad.sum(), 1e-9), B_.sum() / max(Bd.sum(), 1e-9)
    lo, hi = np.percentile(ra - rb, [2.5, 97.5])
    alo, ahi = np.percentile(ra, [2.5, 97.5])
    return pa, alo, ahi, pb, pa - pb, lo, hi


def paired_mean(da, db, col):
    from jevdrive import stats
    m = da[["token", "log", col]].merge(db[["token", col]], on="token", suffixes=("_a", "_b")).dropna()
    if not len(m):
        return dict(n=0)
    r = stats.paired(m[f"{col}_a"].to_numpy(), m[f"{col}_b"].to_numpy(), groups=m["log"].to_numpy())
    a = stats.bootstrap(m[f"{col}_a"].to_numpy(), groups=m["log"].to_numpy())
    return dict(n=len(m), a=r["mean_a"], a_lo=a["lo"], a_hi=a["hi"], b=r["mean_b"], d=r["mean"], d_lo=r["lo"], d_hi=r["hi"])


def toward_taken(G, runs):
    """per junction token: on the `none` plan at 4 s, mean over the other classes of d_other - d_taken (m; > 0 = toward the taken branch)."""
    out = {}
    for tok, rr in runs.items():
        s = G[tok]
        if s["kind"] != "junction" or ("none", "") not in rr:
            continue
        cls = sorted({b["cls"] for b in s["branches"]})
        if s["taken"] not in cls:
            continue
        Q = {c: O.full_centre(s, O.cmd_path(s, c)) for c in cls}
        x = R.interp(rr[("none", "")][0], TAU)
        out[tok] = np.mean([R.near(Q[o], x)[0] - R.near(Q[s["taken"]], x)[0] for o in cls if o != s["taken"]])
    return out


def cmd_report(a):
    from jevdrive import stats
    pool = a.pool
    if pool == "dev":
        G = {s["token"]: s for s in pickle.load(open(ROOT / "geom" / "ft.pkl", "rb"))}
        DEV_TOK.update(t for t, s in G.items() if s["split"] == "dev")
    else:
        G = {s["token"]: s for s in pickle.load(open(ROOT / "geom" / "nav.pkl", "rb"))}
    ra, rb = load_runs(a.model, pool), load_runs(a.ref, pool)
    Ja, Sa = per_sample(G, ra)
    Jb, Sb = per_sample(G, rb)
    out = FT / "report"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model}_vs_{a.ref}_{pool}"
    Ja.assign(model=a.model).to_csv(out / f"{stem}_per_sample_a.csv", index=False)
    Jb.assign(model=a.ref).to_csv(out / f"{stem}_per_sample_b.csv", index=False)
    E = []
    for fam in ["none"] + [f for f in O.FAMILIES if f not in O.ROUTE_FREE]:
        da, db = Ja[Ja.fam == fam], Jb[Jb.fam == fam]
        if not len(da):
            continue
        r = dict(fam=fam, n=da.token.nunique(), logs=da.log.nunique())
        for col in ["delta", "dx"] + [f"tw_{c}" for c in ("left", "straight", "right")]:
            if col in da and da[col].notna().any():
                r.update({f"{col}_{k}": v for k, v in paired_mean(da, db, col).items() if k != "n"})
        for nm, num, den in (("uptake", "up_num", "up_den"), ("fcorrect", "n_fcorrect", "n_fdecisive")):
            pa, alo, ahi, pb, d, lo, hi = paired_ratio(da, db, num, den)
            r.update({f"{nm}_a": pa, f"{nm}_a_lo": alo, f"{nm}_a_hi": ahi, f"{nm}_b": pb, f"{nm}_d": d, f"{nm}_d_lo": lo, f"{nm}_d_hi": hi,
                      f"{nm}_n": int(da[den].sum())})
        E.append(r)
    for fam in ("band", "lines", "arrow_road", "sign"):
        if len(Sa) and (Sa.fam == fam).any():
            r = dict(fam=f"straight:{fam}", n=int((Sa.fam == fam).sum()), logs=Sa[Sa.fam == fam].log.nunique())
            r.update({f"dlat_err_{k}": v for k, v in paired_mean(Sa[Sa.fam == fam], Sb[Sb.fam == fam], "dlat_err").items() if k != "n"})
            E.append(r)
    E = pd.DataFrame(E)
    # guards on the unperturbed input
    g = {}
    za, zb = np.load(FT / "plans" / a.model / f"{'train' if pool == 'dev' else pool}.npz"), np.load(FT / "plans" / a.ref / f"{'train' if pool == 'dev' else pool}.npz")
    from jevdrive import op_adapt as A
    mn = (za["fam"] == "none") & np.isin(za["token"], list(G))
    if pool == "dev":
        mn &= np.isin(za["token"], list(DEV_TOK))
    kind = np.array([G[str(t)]["kind"] for t in za["token"][mn]])
    dr = A.plan_drift(za["mu"][mn], zb["mu"][mn])
    for k in ("junction", "straight"):
        g[f"none_drift_median_{k}"] = float(np.median(dr[kind == k]))
    g["none_drift_median_all"] = float(np.median(dr))
    ta, tb = toward_taken(G, ra), toward_taken(G, rb)
    tk = sorted(set(ta) & set(tb))
    r = stats.paired(np.array([ta[t] for t in tk]), np.array([tb[t] for t in tk]), groups=np.array([G[t]["log"] for t in tk]))
    g["none_toward_taken"] = {k: r[k] for k in ("n", "mean", "lo", "hi", "mean_a", "mean_b")}
    lat = lambda rr, s: abs(R.interp(rr[("none", "")][0], 3.0)[1] - np.asarray(s["future"])[5, 1])  # noqa: E731
    st = [t for t in ra if G[t]["kind"] == "straight" and t in rb and len(G[t]["future"]) >= 6]
    if st:
        r = stats.paired(np.array([lat(ra[t], G[t]) for t in st]), np.array([lat(rb[t], G[t]) for t in st]), groups=np.array([G[t]["log"] for t in st]))
        g["straight_none_lat_err_3s"] = {k: r[k] for k in ("n", "mean", "lo", "hi", "mean_a", "mean_b")}
    if a.ref == "O" and pool.startswith("eval") and a.trt:
        trt = {}
        for f in sorted((ROOT / "raw" / "nav").glob("nav-all-*of2.npz")):
            z = np.load(f)
            for i in np.flatnonzero(z["fam"] == "none"):
                trt[str(z["token"][i])] = z["plan_pos"][i]
        d4 = [np.linalg.norm(R.interp(rb[t][("none", "")][0], TAU) - R.interp(trt[t], TAU)) for t in rb if t in trt]
        g["port_vs_trt_none_4s_m"] = {"n": len(d4), "median": float(np.median(d4)), "p90": float(np.percentile(d4, 90))}
    (out / f"{stem}_guards.json").write_text(json.dumps(g, indent=1))
    E.to_csv(out / f"{stem}.csv", index=False)
    f = lambda x: "" if not np.isfinite(x) else f"{x:+.2f}"  # noqa: E731
    q = lambda r, k: f"{f(r.get(k, np.nan))} [{f(r.get(k + '_lo', np.nan))}, {f(r.get(k + '_hi', np.nan))}]"  # noqa: E731
    L = [f"# image-command fine-tune: {a.model} vs {a.ref} ({pool}, tau = {TAU:g} s)", "",
         f"Junction samples {Ja.token.nunique()} ({Ja.log.nunique()} logs), straight {Sa.token.nunique() if len(Sa) else 0}. "
         "a = model, b = ref; d = a - b, 95% paired cluster bootstrap by log (2000 resamples for ratios, jevdrive.stats otherwise). "
         "Metrics as img_report.py (Delta m, uptake = fraction of a full switch, fcorrect = fixed-set correct, tw_c = move toward class c "
         "vs the model's own `none`, dx = plan x change vs `none`).", "",
         "| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |",
         "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for _, r in E[~E.fam.str.startswith("straight")].iterrows():
        r = r.to_dict()
        L.append(f"| {r['fam']} | {r['n']} | {f(r.get('delta_a', np.nan))} | {q(r, 'delta_d')} | "
                 f"{r['uptake_a']:.2f} [{r['uptake_a_lo']:.2f}, {r['uptake_a_hi']:.2f}] (b {r['uptake_b']:.2f}) | {q(r, 'uptake_d')} | "
                 f"{r['fcorrect_a']:.2f} [{r['fcorrect_a_lo']:.2f}, {r['fcorrect_a_hi']:.2f}] (b {r['fcorrect_b']:.2f}, n {r['fcorrect_n']}) | {q(r, 'fcorrect_d')} | "
                 f"{f(r.get('tw_left_a', np.nan))} / {f(r.get('tw_straight_a', np.nan))} / {f(r.get('tw_right_a', np.nan))} | "
                 f"{f(r.get('dx_a', np.nan))} | {q(r, 'dx_d')} |")
    S_ = E[E.fam.str.startswith("straight")]
    if len(S_):
        L += ["", "Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.", "",
              "| family | n | d lat err a | d lat err b | a - b |", "|:--|--:|:--|:--|:--|"]
        for _, r in S_.iterrows():
            r = r.to_dict()
            L.append(f"| {r['fam']} | {r['n']} | {q(r, 'dlat_err_a')} | {f(r['dlat_err_b'])} | {q(r, 'dlat_err_d')} |")
    L += ["", "Guards: " + json.dumps(g)]
    (out / f"{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plans")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--pool", default="eval")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", default="O")
    p.add_argument("--pool", default="eval")
    p.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    {"plans": cmd_plans, "report": cmd_report}[a.cmd](a)
