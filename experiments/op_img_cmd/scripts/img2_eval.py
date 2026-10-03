"""Image-command fine-tune Q3: port plans on every bank and the paired model-vs-original reports (op-train venv).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md. Metrics and tables as img_ft_eval.py (Q2); here per evaluation set:

  set        bank    rows                                     cluster
  navdev     train   Q2 dev logs (136 junction / 30 straight)  log
  naveval    eval    the Q1 / Q2 eval set (385 / 293)          log
  carladev   carla   CARLA dev routes                          route
  carlatest  carla   CARLA test routes (every family)          route
  plus the no-overlay drift on the dist bank's dev rows (L3's nav / WOD / CARLA p6 dev splits).

  plans   --models O q3A-s0 ... [--banks train eval carla dist]  -> ft/plans/<model>/<bank>.npz (mu), shared with Q2's layout
  report  --model q3A-s0 --sets navdev carladev                  -> ft/report2/<model>_<set>.{md,csv}, _guards.json
  select  --models q3A-s0 q3B-s0 q3C-s0                          -> the dev rule of the plan, ft/report2/select.json
"""
import argparse, json, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_common_cause" / "scripts"),
                str(REPO / "experiments" / "op_adapt_h" / "scripts"), str(Path(__file__).resolve().parent)]
import img_ft_eval as FE  # noqa: E402
import img_overlay as O  # noqa: E402
import img_report as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_img_cmd"
FT = ROOT / "ft"
OUT = FT / "report2"
TAU = 4.0
DRIFT_LINE, UPTAKE_DEV_LINE = 0.10, 0.45


def bank(name):
    import img2_train as Q
    return Q.bank(name)


def cmd_plans(a):
    import torch
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    for b in a.banks:
        T, v = bank(b)
        for name in a.models:
            p = FT / "plans" / name / f"{b}.npz"
            if p.exists() and not a.force:
                continue
            m = L.load_model(FE.ckpt(name), dev)
            pi = A.plan_index(m.net.slices)
            n = len(T)
            mu = np.zeros((n, 33, 15), np.float32)
            with torch.no_grad():
                for i in range(0, n, 96):
                    j = slice(i, min(i + 96, n))
                    o = m(torch.from_numpy(np.asarray(T[j])).to(dev), torch.from_numpy(v["slot_valid"][j]).to(dev),
                          torch.from_numpy(np.asarray(v["tc"][j], np.float32)).to(dev).half())["outputs"].float()
                    mu[j] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, mu=mu, token=v["token"], fam=v["fam"], cmd=v["cmd"])
            print(f"{name} {b}: {n} plans", flush=True)
            del m
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- sets
def set_def(name):
    """(bank, G {token: sample}, token filter)"""
    import img2_bank as QB
    if name in ("navdev", "naveval"):
        if name == "navdev":
            G = {s["token"]: s for s in pickle.load(open(ROOT / "geom" / "ft.pkl", "rb")) if s["split"] == "dev"}
            return "train", G
        return "eval", {s["token"]: s for s in pickle.load(open(ROOT / "geom" / "nav.pkl", "rb"))}
    sp = QB.carla_split()
    want = name[5:]
    return "carla", {s["token"]: s for s in QB.carla_samples() if sp[s["token"]] == want}


def runs(model, b, G):
    import img_run
    z = np.load(FT / "plans" / model / f"{b}.npz")
    _, v = bank(b)
    m = np.isin(v["token"], list(G))
    out = {}
    for i in np.flatnonzero(m):
        P, Y = img_run.to_rear(z["mu"][i][:, 0:3], z["mu"][i][:, 11], np.asarray(v["cam"][i], float).reshape(-1))
        out.setdefault(str(v["token"][i]), {})[(str(v["fam"][i]), str(v["cmd"][i]))] = (P, Y)
    mn = m & (v["fam"] == "none")
    return out, z["mu"][mn], v["token"][mn]


def report(model, ref, name):
    from jevdrive import op_adapt as A
    from jevdrive import stats
    b, G = set_def(name)
    ra, mua, tka = runs(model, b, G)
    rb, mub, tkb = runs(ref, b, G)
    assert (tka == tkb).all()
    Ja, Sa = FE.per_sample(G, ra)
    Jb, Sb = FE.per_sample(G, rb)
    E = []
    for fam in ["none"] + [f for f in O.FAMILIES if f not in O.ROUTE_FREE]:
        da, db = Ja[Ja.fam == fam], Jb[Jb.fam == fam]
        if not len(da):
            continue
        r = dict(fam=fam, n=da.token.nunique(), logs=da.log.nunique())
        for col in ["delta", "dx"] + [f"tw_{c}" for c in ("left", "straight", "right")]:
            if col in da and da[col].notna().any():
                r.update({f"{col}_{k}": x for k, x in FE.paired_mean(da, db, col).items() if k != "n"})
        for nm, num, den in (("uptake", "up_num", "up_den"), ("fcorrect", "n_fcorrect", "n_fdecisive")):
            pa, alo, ahi, pb, d, lo, hi = FE.paired_ratio(da, db, num, den)
            r.update({f"{nm}_a": pa, f"{nm}_a_lo": alo, f"{nm}_a_hi": ahi, f"{nm}_b": pb, f"{nm}_d": d, f"{nm}_d_lo": lo, f"{nm}_d_hi": hi,
                      f"{nm}_n": int(da[den].sum())})
        E.append(r)
    for fam in ("band", "lines", "arrow_road", "sign"):
        if len(Sa) and (Sa.fam == fam).any():
            r = dict(fam=f"straight:{fam}", n=int((Sa.fam == fam).sum()), logs=Sa[Sa.fam == fam].log.nunique())
            r.update({f"dlat_err_{k}": x for k, x in FE.paired_mean(Sa[Sa.fam == fam], Sb[Sb.fam == fam], "dlat_err").items() if k != "n"})
            E.append(r)
    E = pd.DataFrame(E)
    g = {}
    kind = np.array([G[str(t)]["kind"] for t in tka])
    dr = A.plan_drift(mua, mub)
    for k in ("junction", "straight"):
        if (kind == k).any():
            g[f"none_drift_median_{k}"] = float(np.median(dr[kind == k]))
    # band_all (no route information) on junction frames: plan change vs the model's own none, 0-5 s mean L2 (negatives)
    ba = [(t, rr[("band_all", "")][0], rr[("none", "")][0]) for t, rr in ra.items() if ("band_all", "") in rr]
    if ba:
        g["band_all_move_median"] = float(np.median([np.linalg.norm(p[A.T5] - q[A.T5], axis=1).mean() for _, p, q in ba]))
        bb = [(rb[t][("band_all", "")][0], rb[t][("none", "")][0]) for t, _, _ in ba]
        g["band_all_move_median_ref"] = float(np.median([np.linalg.norm(p[A.T5] - q[A.T5], axis=1).mean() for p, q in bb]))
    ta, tb = FE.toward_taken(G, ra), FE.toward_taken(G, rb)
    tk = sorted(set(ta) & set(tb))
    if tk:
        r = stats.paired(np.array([ta[t] for t in tk]), np.array([tb[t] for t in tk]), groups=np.array([G[t]["log"] for t in tk]))
        g["none_toward_taken"] = {k: r[k] for k in ("n", "mean", "lo", "hi", "mean_a", "mean_b")}
    lat = lambda rr, s: abs(R.interp(rr[("none", "")][0], 3.0)[1] - np.asarray(s["future"])[5, 1])  # noqa: E731
    st = [t for t in ra if G[t]["kind"] == "straight" and t in rb and len(G[t]["future"]) >= 6]
    if st:
        r = stats.paired(np.array([lat(ra[t], G[t]) for t in st]), np.array([lat(rb[t], G[t]) for t in st]), groups=np.array([G[t]["log"] for t in st]))
        g["straight_none_lat_err_3s"] = {k: r[k] for k in ("n", "mean", "lo", "hi", "mean_a", "mean_b")}
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_vs_{ref}_{name}"
    (OUT / f"{stem}_guards.json").write_text(json.dumps(g, indent=1))
    E.to_csv(OUT / f"{stem}.csv", index=False)
    f = lambda x: "" if not np.isfinite(x) else f"{x:+.2f}"  # noqa: E731
    q = lambda r, k: f"{f(r.get(k, np.nan))} [{f(r.get(k + '_lo', np.nan))}, {f(r.get(k + '_hi', np.nan))}]"  # noqa: E731
    Lines = [f"# image-command fine-tune Q3: {model} vs {ref} ({name}, tau = {TAU:g} s)", "",
             f"Junction samples {Ja.token.nunique()} ({Ja.log.nunique()} clusters), straight {Sa.token.nunique() if len(Sa) else 0}. "
             "a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.", "",
             "| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |",
             "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for _, r in E[~E.fam.str.startswith("straight")].iterrows():
        r = r.to_dict()
        Lines.append(f"| {r['fam']} | {r['n']} | {f(r.get('delta_a', np.nan))} | {q(r, 'delta_d')} | "
                     f"{r['uptake_a']:.2f} [{r['uptake_a_lo']:.2f}, {r['uptake_a_hi']:.2f}] (b {r['uptake_b']:.2f}) | {q(r, 'uptake_d')} | "
                     f"{r['fcorrect_a']:.2f} [{r['fcorrect_a_lo']:.2f}, {r['fcorrect_a_hi']:.2f}] (b {r['fcorrect_b']:.2f}, n {r['fcorrect_n']}) | "
                     f"{q(r, 'fcorrect_d')} | {f(r.get('tw_left_a', np.nan))} / {f(r.get('tw_straight_a', np.nan))} / {f(r.get('tw_right_a', np.nan))} | "
                     f"{f(r.get('dx_a', np.nan))} | {q(r, 'dx_d')} |")
    S_ = E[E.fam.str.startswith("straight")] if len(E) else E
    if len(S_):
        Lines += ["", "Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.", "",
                  "| family | n | d lat err a | d lat err b | a - b |", "|:--|--:|:--|:--|:--|"]
        for _, r in S_.iterrows():
            r = r.to_dict()
            Lines.append(f"| {r['fam']} | {r['n']} | {q(r, 'dlat_err_a')} | {f(r['dlat_err_b'])} | {q(r, 'dlat_err_d')} |")
    Lines += ["", "Guards: " + json.dumps(g)]
    (OUT / f"{stem}.md").write_text("\n".join(Lines) + "\n")
    print("\n".join(Lines), flush=True)
    return E, g


def dist_drift(model, ref="O"):
    from jevdrive import op_adapt as A
    _, v = bank("dist")
    za, zb = np.load(FT / "plans" / model / "dist.npz"), np.load(FT / "plans" / ref / "dist.npz")
    dr = A.plan_drift(za["mu"], zb["mu"])
    return {f"{d}_{s}": float(np.median(dr[(v["dom"] == d) & (v["split"] == s)])) for d in ("nav", "wod", "carla") for s in ("train", "dev")}


def cmd_report(a):
    for s in a.sets:
        report(a.model, a.ref, s)
    if (FT / "plans" / a.model / "dist.npz").exists():
        d = dist_drift(a.model, a.ref)
        (OUT / f"{a.model}_vs_{a.ref}_dist_drift.json").write_text(json.dumps(d, indent=1))
        print("dist drift", d)


def cmd_select(a):
    rows = []
    for m in a.models:
        r = {"model": m}
        for s in ("navdev", "carladev"):
            E = pd.read_csv(OUT / f"{m}_vs_O_{s}.csv").set_index("fam")
            g = json.loads((OUT / f"{m}_vs_O_{s}_guards.json").read_text())
            for fam in ("band", "barrier"):
                r[f"{s}_{fam}_uptake"] = float(E.loc[fam, "uptake_a"])
            for k, x in g.items():
                if k.startswith("none_drift_median"):
                    r[f"{s}_{k[18:]}_drift"] = x
        r |= {f"dist_{k}_drift": x for k, x in json.loads((OUT / f"{m}_vs_O_dist_drift.json").read_text()).items() if k.endswith("dev")}
        dk = ["navdev_junction_drift", "navdev_straight_drift", "carladev_junction_drift"]
        r["drift_ok"] = all(r[k] <= DRIFT_LINE for k in dk)
        r["band_dev_min"] = min(r["navdev_band_uptake"], r["carladev_band_uptake"])
        r["uptake_ok"] = r["band_dev_min"] >= UPTAKE_DEV_LINE
        rows.append(r)
    D = pd.DataFrame(rows)
    ok = D[D.drift_ok & D.uptake_ok]
    sel = None if not len(ok) else str(ok.sort_values("band_dev_min").iloc[-1].model)
    out = {"selected": sel, "table": D.to_dict("records")}
    (OUT / "select.json").write_text(json.dumps(out, indent=1))
    print(D.T.to_string())
    print("selected:", sel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plans")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--banks", nargs="+", default=["train", "eval", "carla", "dist"])
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", default="O")
    p.add_argument("--sets", nargs="+", default=["navdev", "carladev"])
    p = sp.add_parser("select")
    p.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    {"plans": cmd_plans, "report": cmd_report, "select": cmd_select}[a.cmd](a)
