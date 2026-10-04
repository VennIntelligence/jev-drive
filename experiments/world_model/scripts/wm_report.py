"""Stage 4 (CPU, numpy / scipy / matplotlib): readouts and figures of the WM-vs-reprojection check.

Inputs: runs/wm_vs_reproj/{op,w}/*.npz (wm_op.py, wm_w.py). Outputs: <out>/tables.json, <out>/tables.md, <out>/wm-vs-reproj-{fidelity,gain}.png
Row r of an op branch is the frame r - 7 steps from the anchor (r = 7 is the anchor, r = 7 + j is step j, 0.2 s each); the world model's prediction
row k is step k + 1.

Probe: ridge from the standardised openpilot `temporal` (512) to (phi1, curvature), fitted on the G and R rows of the other 9 segments
(leave-one-segment-out); it is the only way to read a plan out of the world model's latent, and it is applied identically to G, R and W.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

H, K = 8, 10
STEPS = (1, 3, 5, 10)
YAW = {"yaw+1": 1.0, "yaw+2": 2.0, "yaw-1": -1.0, "yaw-2": -2.0}


def load(run):
    A = []
    for f in sorted(glob.glob(str(run / "op" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        an = json.loads(str(z["anchor"]))
        w = run / "w" / Path(f).name
        if not w.exists():
            continue
        wz = np.load(w, allow_pickle=True)
        d = {"an": an, "v": z["v"], "br": {}, "W": {}, "pose": {k.split("/")[1]: z[k] for k in z.files if k.startswith("pose/")},
             "img": {k.split("/")[1]: z[k] for k in z.files if k.startswith("img/")}}
        for k in z.files:
            if "/" in k and not k.startswith(("pose/", "img/")):
                b, q = k.split("/")
                d["br"].setdefault(b, {})[q] = z[k]
        for k in wz.files:
            if k != "anchor":
                arm, name, what = k.split("/")
                d["W"].setdefault(arm, {}).setdefault(name, {})[what] = wz[k]
        A.append(d)
    return A


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    Wt = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ Wt + ym


def targets(b):
    return np.stack([b["phi1"], b["curv"] * 1e3], -1)           # deg, 1e-3 / m


def fit_probes(A):
    """LOSO probes per segment: {seg: f(Z) -> (n, 2)} and the held-out R^2 on G rows and on R rows."""
    segs = sorted({a["an"]["seg"] for a in A})
    rows = {s: ([], []) for s in segs}
    rr = {s: ([], []) for s in segs}
    for a in A:
        s = a["an"]["seg"]
        g = a["br"]["G"]
        rows[s][0].append(g["z"]), rows[s][1].append(targets(g))
        for name, b in a["br"].items():
            if name != "G":
                rr[s][0].append(b["z"][H - 1:]), rr[s][1].append(targets(b)[H - 1:])
    out, r2 = {}, {"G": [], "R": []}
    for s in segs:
        trX = np.concatenate([x for t in segs if t != s for x in rows[t][0] + rr[t][0]])
        trY = np.concatenate([y for t in segs if t != s for y in rows[t][1] + rr[t][1]])
        # lambda by an inner split on the training segments' G rows is overkill for 10 segments; one fixed value, checked in the table
        f = ridge_fit(trX, trY, LAM)
        out[s] = f
        for key, src in (("G", rows[s]), ("R", rr[s])):
            if src[0]:
                X, Y = np.concatenate(src[0]), np.concatenate(src[1])
                r2["G" if key == "G" else "R"].append((Y, f(X)))
    res = {}
    for key, lst in r2.items():
        Y = np.concatenate([y for y, _ in lst])
        P = np.concatenate([p for _, p in lst])
        res[key] = {n: float(1 - ((Y[:, i] - P[:, i]) ** 2).sum() / ((Y[:, i] - Y[:, i].mean()) ** 2).sum()) for i, n in enumerate(("phi1", "curv"))}
    return out, res


def med_iqr(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)), len(x)] if len(x) else [np.nan] * 3 + [0]


def boot_med(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    r = np.random.default_rng(seed).choice(x, (n, len(x)))
    m = np.median(r, 1)
    return [float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    global LAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/root/autodl-tmp/ujs/runs/wm_vs_reproj")
    ap.add_argument("--out", default="/root/autodl-tmp/ujs/runs/wm_vs_reproj/report")
    ap.add_argument("--lam", type=float, default=300.0)
    a = ap.parse_args()
    LAM = a.lam
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = load(Path(a.run))
    print(len(A), "anchors", flush=True)
    allz = np.concatenate([x["br"]["G"]["z"] for x in A])
    zsd = allz.std(0) + 1e-3 * np.median(allz.std(0))
    probes, probe_r2 = fit_probes(A)

    def dist(x, y):
        return float((((x - y) / zsd) ** 2).mean())

    T = {"n_anchors": len(A), "cats": {c: sum(x["an"]["cat"] == c for x in A) for c in sorted({x["an"]["cat"] for x in A})}, "probe_r2_loso": probe_r2, "lam": LAM}
    # ---------- fidelity on the nominal (natural) motion: the ground truth G exists
    fid = {s: {k: [] for k in ("zR", "zWB", "zWA", "zP", "p_R", "p_WB", "p_WA", "p_P", "p_floor", "c_R", "c_WB", "c_P", "c_floor", "ph_R", "ph_P", "cu_R", "cu_P", "zH", "zI", "ph_H", "ph_I", "cu_H", "cu_I", "iR", "iH", "iI")} for s in STEPS}
    cat_rows = []
    for x in A:
        pr = probes[x["an"]["seg"]]
        G, R = x["br"]["G"], x["br"]["nom"]
        pG = pr(G["z"])
        z0 = G["z"][H - 1]
        row = {"cat": x["an"]["cat"], "seg": x["an"]["seg"][:8], "i": x["an"]["i"], "v": round(float(x["v"][H - 1]), 1)}
        for s in STEPS:
            r = H - 1 + s
            wB, wA = x["W"]["B"]["nom"]["op"].mean(0)[s - 1], x["W"]["A"]["nom"]["op"].mean(0)[s - 1]
            f = fid[s]
            f["zR"].append(dist(R["z"][r], G["z"][r]))
            f["zWB"].append(dist(wB, G["z"][r]))
            f["zWA"].append(dist(wA, G["z"][r]))
            f["zP"].append(dist(z0, G["z"][r]))
            tG = targets(G)[r]
            f["p_R"].append(abs(pr(R["z"][r:r + 1])[0, 0] - tG[0]))
            f["p_WB"].append(abs(pr(wB[None])[0, 0] - tG[0]))
            f["p_WA"].append(abs(pr(wA[None])[0, 0] - tG[0]))
            f["p_P"].append(abs(pr(z0[None])[0, 0] - tG[0]))
            f["p_floor"].append(abs(pG[r, 0] - tG[0]))
            f["c_R"].append(abs(pr(R["z"][r:r + 1])[0, 1] - tG[1]))
            f["c_WB"].append(abs(pr(wB[None])[0, 1] - tG[1]))
            f["c_P"].append(abs(pr(z0[None])[0, 1] - tG[1]))
            f["c_floor"].append(abs(pG[r, 1] - tG[1]))
            f["ph_R"].append(abs(R["phi1"][r] - G["phi1"][r]))            # true head, R vs G
            f["ph_P"].append(abs(G["phi1"][H - 1] - G["phi1"][r]))        # persistence of the head
            f["cu_R"].append(abs(R["curv"][r] - G["curv"][r]) * 1e3)
            f["cu_P"].append(abs(G["curv"][H - 1] - G["curv"][r]) * 1e3)
            for tag, br in (("H", "hold"), ("I", "idw")):
                f["z" + tag].append(dist(x["br"][br]["z"][r], G["z"][r]))
                f["ph_" + tag].append(abs(x["br"][br]["phi1"][r] - G["phi1"][r]))
                f["cu_" + tag].append(abs(x["br"][br]["curv"][r] - G["curv"][r]) * 1e3)
            f["iR"].append(float(x["img"]["nom"][s - 1])), f["iH"].append(float(x["img"]["hold"][s - 1])), f["iI"].append(float(x["img"]["idw"][s - 1]))
            if s == 5:
                row |= {"zR": f["zR"][-1], "zWB": f["zWB"][-1], "zP": f["zP"][-1], "ph_R": f["ph_R"][-1], "ph_P": f["ph_P"][-1],
                        "phiG": float(G["phi1"][r]), "yaw_deg": float(np.degrees(x["pose"]["nom"][s, 2])), "lat_m": float(x["pose"]["nom"][s, 1])}
        cat_rows.append(row)
    T["fidelity"] = {str(s): {k: med_iqr(v) for k, v in fid[s].items()} for s in STEPS}
    T["fidelity_by_anchor_step5"] = cat_rows
    # ---------- response to the perturbations (no G exists): loop gain
    gain = {fam: {s: {k: [] for k in ("R_true", "R_probe", "WB_probe", "WA_probe")} for s in range(1, K + 1)} for fam in ("yaw", "lat")}
    dz = {fam: {s: {"cos": [], "ratio": [], "cosA": [], "ratioA": []} for s in STEPS} for fam in ("yaw", "lat")}
    ey = {"err": [], "pred": [], "true": []}
    for x in A:
        pr = probes[x["an"]["seg"]]

        def read(name, s, kind):
            if kind == "R_true":
                return np.array([x["br"][name]["phi1"][H - 1 + s], x["br"][name]["curv"][H - 1 + s] * 1e3])
            if kind == "R_probe":
                return pr(x["br"][name]["z"][H - 1 + s][None])[0]
            arm = "B" if kind == "WB_probe" else "A"
            return pr(x["W"][arm][name]["op"].mean(0)[s - 1][None])[0]
        fams = {"yaw": (("yaw+2", "yaw-2", 4.0), ("yaw+1", "yaw-1", 2.0)), "lat": (("lat+0.5", "lat-0.5", 1.0),)}
        for fam, pairs in fams.items():
            for (p, m, den) in pairs:
                if p not in x["br"] or m not in x["br"]:
                    continue
                if fam == "yaw" and den == 2.0:
                    continue                                            # +-1 deg pair kept in the json only through the +-2 pair (same slope, noisier)
                for s in range(1, K + 1):
                    for kind in ("R_true", "R_probe", "WB_probe", "WA_probe"):
                        d = (read(p, s, kind) - read(m, s, kind)) / den
                        gain[fam][s][kind].append(d)
                for s in STEPS:
                    dR = x["br"][p]["z"][H - 1 + s] - x["br"][m]["z"][H - 1 + s]
                    for arm, ck, rk in (("B", "cos", "ratio"), ("A", "cosA", "ratioA")):
                        dW = x["W"][arm][p]["op"].mean(0)[s - 1] - x["W"][arm][m]["op"].mean(0)[s - 1]
                        dz[fam][s][ck].append(float(dW @ dR / (np.linalg.norm(dW) * np.linalg.norm(dR) + 1e-9)))
                        dz[fam][s][rk].append(float(np.linalg.norm(dW / zsd) / (np.linalg.norm(dR / zsd) + 1e-9)))
        # B's predicted e_y change against the kinematic one (lateral offset of the perturbed path from the nominal one, nominal heading normal)
        P0 = x["pose"]["nom"]
        for name, Q in x["pose"].items():
            if name == "nom":
                continue
            n = np.stack([-np.sin(P0[:, 2]), np.cos(P0[:, 2])], 1)
            off = ((Q[:, :2] - P0[:, :2]) * n).sum(1)
            de = x["W"]["B"][name]["s"].mean(0)[:, 0] - x["W"]["B"]["nom"]["s"].mean(0)[:, 0]     # (K,)
            ey["err"].append(float(np.abs(de[-1] - off[K]))), ey["pred"].append(float(de[-1])), ey["true"].append(float(off[K]))
    T["gain"] = {fam: {str(s): {k: ([boot_med([g[i] for g in v], seed=i) for i in range(2)] if v else None) for k, v in d.items()} for s, d in dd.items()} for fam, dd in gain.items()}
    T["gain_n"] = {fam: len(gain[fam][1]["R_true"]) for fam in gain}
    T["dz"] = {fam: {str(s): {k: med_iqr(v) for k, v in d.items()} for s, d in dd.items()} for fam, dd in dz.items()}
    T["ey_check"] = {"median_abs_err_m": float(np.median(ey["err"])), "median_abs_true_m": float(np.median(np.abs(ey["true"]))),
                     "corr": float(np.corrcoef(ey["pred"], ey["true"])[0, 1]), "n": len(ey["err"])}
    json.dump(T, open(out / "tables.json", "w"), indent=1, default=float)
    figs(T, out)
    print(json.dumps({k: T[k] for k in ("n_anchors", "cats", "probe_r2_loso", "gain_n", "ey_check")}, indent=1))


def figs(T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    c = {"R": "#d95f02", "W": "#1b9e77", "A": "#7570b3", "P": "#999999", "G": "#000000"}
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.3))
    st = [str(s) for s in STEPS]
    t = [0.2 * s for s in STEPS]
    for k, lab, col in (("zR", "R (reprojection)", c["R"]), ("zWB", "W (WL-2 B)", c["W"]), ("zWA", "W (WL-2 A)", c["A"]), ("zP", "persistence (z at t0)", c["P"]), ("zH", "R control: hold the anchor frame", "#66a61e"), ("zI", "R control: warp with no motion", "#a6761d")):
        m = np.array([T["fidelity"][s][k] for s in st])
        ax[0].errorbar(t, m[:, 0], [m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], label=lab, color=col, marker="o", ms=3, capsize=2)
    ax[0].set(xlabel="horizon (s)", ylabel="standardised MSE of `temporal` vs G", title="latent distance to the real frame")
    ax[0].legend(frameon=False, fontsize=7)
    for k, lab, col in (("p_R", "R, via probe", c["R"]), ("p_WB", "W (B), via probe", c["W"]), ("p_WA", "W (A), via probe", c["A"]), ("p_P", "persistence, via probe", c["P"]),
                        ("p_floor", "probe floor (G)", c["G"]), ("ph_R", "R, true head", "#e6ab02"), ("ph_H", "hold, true head", "#66a61e")):
        m = np.array([T["fidelity"][s][k] for s in st])
        ax[1].errorbar(t, m[:, 0], [m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], label=lab, color=col, marker="o", ms=3, capsize=2, ls="--" if k in ("p_floor", "ph_R", "ph_H") else "-")
    ax[1].set(xlabel="horizon (s)", ylabel="|phi1 - phi1(G)| (deg)", title="plan direction at 1 s")
    ax[1].legend(frameon=False, fontsize=6)
    for k, lab, col in (("c_R", "R, via probe", c["R"]), ("c_WB", "W (B), via probe", c["W"]), ("c_P", "persistence, via probe", c["P"]), ("c_floor", "probe floor (G)", c["G"]),
                        ("cu_R", "R, true head", "#e6ab02"), ("cu_H", "hold, true head", "#66a61e")):
        m = np.array([T["fidelity"][s][k] for s in st])
        ax[2].errorbar(t, m[:, 0], [m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], label=lab, color=col, marker="o", ms=3, capsize=2, ls="--" if k in ("c_floor", "cu_R", "cu_H") else "-")
    ax[2].set(xlabel="horizon (s)", ylabel="|action curvature - G| (1e-3 / m)", title="action-head curvature")
    ax[2].legend(frameon=False, fontsize=6)
    for k, lab, col in (("iR", "R", c["R"]), ("iH", "hold", "#66a61e"), ("iI", "warp, no motion", "#a6761d")):
        m = np.array([T["fidelity"][s][k] for s in st])
        ax[3].errorbar(t, m[:, 0], [m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], label=lab, color=col, marker="o", ms=3, capsize=2)
    ax[3].set(xlabel="horizon (s)", ylabel="luma L1 to the real frame (0-255)", title="image error, road view, below row 60")
    ax[3].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Nominal real motion, {T['n_anchors']} anchors: medians and IQR", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "wm-vs-reproj-fidelity.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    s_all = np.arange(1, K + 1)
    for j, (fam, idx, ttl, yl) in enumerate((("yaw", 0, "yaw: plan direction", "d phi1 / d yaw (deg / deg)"), ("yaw", 1, "yaw: action curvature", "d kappa / d yaw (1e-3 / m / deg)"),
                                              ("lat", 0, "lateral: plan direction", "d phi1 / d offset (deg / m)"))):
        for kind, lab, col in (("R_true", "R, true head", c["R"]), ("R_probe", "R, via probe", "#e6ab02"), ("WB_probe", "W (B), via probe", c["W"]), ("WA_probe", "W (A), via probe", c["A"])):
            m = np.array([T["gain"][fam][str(s)][kind][idx] if T["gain"][fam][str(s)][kind] else [np.nan] * 3 for s in s_all])
            ax[j].plot(0.2 * s_all, m[:, 0], color=col, label=lab, marker="o", ms=3)
            ax[j].fill_between(0.2 * s_all, m[:, 1], m[:, 2], color=col, alpha=0.15, lw=0)
        ax[j].axhline(0, color="#bbb", lw=0.8)
        ax[j].set(xlabel="horizon (s)", ylabel=yl, title=f"{ttl} (n = {T['gain_n'][fam]} anchors)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Response of openpilot to a perturbed ego pose: median over anchors, 95% bootstrap CI", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "wm-vs-reproj-gain.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
