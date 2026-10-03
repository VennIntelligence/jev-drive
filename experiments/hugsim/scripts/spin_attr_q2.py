"""Q2: loop gain of spin vs non-spin launches.  python spin_attr_q2.py <gain_dir> <traces.json> <jobs_gain.json> <out_dir>
(a) CPU-ONNX replay results of spin_attr_cpu_gain.py: step 1 / 2 local gain, perturbation phi at steps 1-2, large-signal gain, loop growth.
(b) log-only inference on all 64 native Cinque runs: phi1 growth per step and the phi ~ lambda + s H regression at v < 3 m/s."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

gd, tr, jb, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
jobs = {j["key"].replace("|", "_"): j for j in json.load(open(jb))}
W = 2.0
C = 0.19      # decision 100 controller transfer (deg per step per deg of 1 s plan direction)


def zwin(cs):
    r = np.roots(np.r_[1.0, -(1 + cs), np.zeros(5), cs])
    r = r[np.abs(r - 1) > 1e-6]
    return float(np.max(np.abs(r)))


def boot(x, f=np.median, n=3000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    rng = np.random.default_rng(0)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(f(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


rows = []
for f in sorted(gd.glob("*.json")):
    r = json.load(open(f))
    j = jobs[f.stem]
    th = np.array(r["theta"])
    v = np.array(r["v"])
    for s in range(1, len(th)):
        L = r["local"]
        pn, pp, pm = L[f"{s}|+0"]["phi1"], L[f"{s}|{W:+g}"]["phi1"], L[f"{s}|{-W:+g}"]["phi1"]
        H = -np.degrees(th[s] - th[max(0, s - 6)])
        pd_ = r["derot"][str(s)]["phi1"]
        rows.append(dict(group=j["group"], scenario=j["scenario"], step=s, v=v[s], H=H, phi=pn, phi_derot=pd_,
                         s_local=(pp - pm) / 2 / (1.2 * W), s_large=(pn - pd_) / H if 1 <= abs(H) <= 15 else np.nan))
D = pd.DataFrame(rows)
D.to_csv(out / "q2_replay_steps.csv", index=False)
print("replay logs:", D.groupby("group").scenario.nunique().to_dict())
for g, d in D.groupby("group"):
    s1 = d[d.step == 1].s_local
    s2 = d[d.step == 2].s_local
    s39 = d[(d.step >= 3) & (d.step <= 9)].s_local
    ph = d[d.step.isin([1, 2])].phi.abs()
    ls = d.s_large
    print(f"{g:5s} n={d.scenario.nunique():2d}  s_local step1 {np.median(s1):5.2f}  step2 {np.median(s2):5.2f}  steps3-9 {np.median(s39):5.2f}  "
          f"|phi| steps1-2 {np.median(ph):5.2f}  s_large {np.nanmedian(ls) if ls.notna().any() else np.nan:5.2f} (n={int(ls.notna().sum())})  "
          f"growth z(c*s_step1)={zwin(C * np.median(s1)):.2f}")
# per-log rows for the report
P = D[D.step.isin([1, 2])].groupby(["group", "scenario"]).agg(s1=("s_local", "first"), phi12=("phi", lambda x: np.abs(x).mean())).reset_index()
P.to_csv(out / "q2_replay_logs.csv", index=False)
sp = P[P.group == "spin"]
non = P[P.group != "spin"]
for c in ("s1", "phi12"):
    d = np.median(sp[c]) - np.median(non[c])
    print(c, "spin median", np.median(sp[c]).round(2), "non-spin median", np.median(non[c]).round(2), "diff", round(d, 2))
    from scipy.stats import mannwhitneyu
    print("  MWU p", mannwhitneyu(sp[c], non[c]).pvalue.round(3))

# (b) logs only
T = [t for t in json.load(open(tr)) if t["tag"] == "cinque-fixed"]
ep = pd.read_csv(Path(__file__).resolve().parents[1] / "results/spin/spin_episodes.csv")
rows = []
for t in T:
    e = ep[(ep.scenario == t["scenario"]) & (ep.agent == "cinque") & (ep.controller == "fixed")].iloc[0]
    S = t["steps"]
    th = np.unwrap([s["th"] for s in S])
    v = np.array([s["v"] for s in S])
    ph = np.array([-np.degrees(np.arctan2(s["plan"][1][0], max(s["plan"][1][1], 1e-3))) if s.get("plan") else np.nan for s in S])  # + left
    H = np.array([-np.degrees(th[k] - th[max(0, k - 6)]) for k in range(len(S))])
    end = int(e.start) if e.spin else len(S)
    ks = [k for k in range(1, min(end, 12)) if v[k] < 3]
    # empirical per-step growth of |phi| over steps 2..8 where |phi| > 0.5 deg
    g = [abs(ph[k + 1]) / abs(ph[k]) for k in range(2, min(end, 9)) if abs(ph[k]) > 0.5 and np.isfinite(ph[k + 1])]
    # time below 3 m/s from launch (steps until v >= 3 first time, capped at run length)
    ge3 = np.where(v >= 3)[0]
    rows.append(dict(scenario=t["scenario"], spin=int(e.spin), phi12=np.nanmean(np.abs(ph[1:3])), growth=np.median(g) if g else np.nan,
                     t_below3=int(ge3[0]) if len(ge3) else len(S), c_pred=np.nan,
                     dtheta=np.mean(np.abs(np.diff(th[:min(end, 8)]))) if end > 2 else np.nan,
                     H_mean=np.mean(np.abs(H[ks])) if ks else np.nan))
B = pd.DataFrame(rows)
B.to_csv(out / "q2_log_inference.csv", index=False)
from scipy.stats import mannwhitneyu
for c in ("phi12", "growth", "t_below3", "H_mean"):
    a, b = B[B.spin == 1][c].dropna(), B[B.spin == 0][c].dropna()
    print(f"log-only {c}: spin median {a.median():.2f} (n={len(a)}) non-spin {b.median():.2f} (n={len(b)}) MWU p={mannwhitneyu(a, b).pvalue:.3f}")
