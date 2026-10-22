#!/usr/bin/env python3
"""LOG-SYNC (plans/2026-10-22-log-sync-prereg.md): how far the ego is from the log car at the same sim time in stored PAI reads.
CPU, numpy only; box, repo .venv. Results: results/log_sync.md.

Input per read: the rf_tax extract `$DATA_DIR/runs/pai_fix/tax/<tag>-281ag.pkl` (per rollout `ctrl` = the controller CSV that
spin_events.py reads, 10 Hz, rollout local frame, rear axle; `log` = the scene's recorded rig trajectory, same frame and clock) and the
pai_fix facts CSV (score, fclass, t_ev_s).

Per controller row (time t):
  s_log(t)   arc length of the log path at t; the ego is projected on the log path (nearest point in a window around the last
             projection) -> s_e, signed lateral d (left +)
  ds         s_e - s_log(t) (m; behind < 0, ahead > 0)
  tau        t_log(s_e) - t (s); t_log(s) = first time the log reaches arc s; NaN past the log's end arc (flag `past_end`)
  g          |p_e(t) - p_log(t)| (camera distance), v_log(t), v_e(t)
  pre        t <= t_ev (scored failure time from facts; else the whole rollout);  offlog = |d| > 8 m

  log_sync.py --read LABEL=PKL=FACTS ... --out DIR      (writes DIR/steps_<label>.npz, DIR/scenes.csv, DIR/summary.json)
"""
import argparse, csv, json, os, pickle

import numpy as np

OFFLOG, WIN_B, WIN_F = 8.0, 30.0, 80.0


def arc(P):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def project(P, S, q, s0):
    """Nearest point of the log polyline P (arc S) to q, segments with S in [s0 - WIN_B, s0 + WIN_F] -> (s, signed d)."""
    i = np.where((S[1:] >= s0 - WIN_B) & (S[:-1] <= s0 + WIN_F))[0]
    if not len(i):
        i = np.arange(len(P) - 1)
    a, b = P[i], P[i + 1]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    u = np.clip(((q - a) * ab).sum(1) / L2, 0, 1)
    c = a + u[:, None] * ab
    dist = np.hypot(*(q - c).T)
    k = int(np.argmin(dist))
    side = np.sign(ab[k, 0] * (q[1] - a[k, 1]) - ab[k, 1] * (q[0] - a[k, 0])) or 1.0
    return float(S[i[k]] + u[k] * np.sqrt(L2[k])), float(side * dist[k])


def rollout(o, t_ev):
    C, Lg = o["ctrl"], o["log"]
    T, P = np.asarray(Lg["T"], float), np.asarray(Lg["P"], float)[:, :2]
    S = arc(P)
    t = np.asarray(C["t"], float)
    Q = np.c_[C["x"], C["y"]]
    sl = np.interp(t, T, S)
    se, d = np.zeros(len(t)), np.zeros(len(t))
    s0 = sl[0]
    for j, q in enumerate(Q):
        se[j], d[j] = project(P, S, q, s0)
        s0 = se[j]
    k = np.searchsorted(S, se, side="left").clip(1, len(S) - 1)        # first log index with S >= s_e
    f = np.clip((se - S[k - 1]) / np.maximum(S[k] - S[k - 1], 1e-9), 0, 1)
    tl = T[k - 1] + f * (T[k] - T[k - 1])
    past = se > S[-1] - 0.05
    tau = np.where(past | (se < 0), np.nan, (tl - t) * 1e-6)
    vlog_T = np.r_[np.diff(S) / np.maximum(np.diff(T) * 1e-6, 1e-3), 0.0]
    vlog = np.interp(t, T, np.convolve(vlog_T, np.ones(5) / 5, "same"))
    pl = np.c_[np.interp(t, T, P[:, 0]), np.interp(t, T, P[:, 1])]
    tr = (t - t[0]) * 1e-6 + 0.1                                          # the same relative clock as spin_events / facts t_ev_s
    pre = tr <= (t_ev + 0.05 if t_ev is not None else np.inf)
    return dict(t=tr, ds=se - sl, tau=tau, d=d, g=np.hypot(*(Q - pl).T), vlog=vlog, ve=np.hypot(C["vx"], C["vy"]), past=past,
                pre=pre, offlog=np.abs(d) > OFFLOG)


def scene_row(label, o, fa, r):
    ok = r["pre"] & ~r["offlog"]
    a = np.abs(r["ds"])
    row = dict(read=label, scene=o["scene"][7:15], score=round(float(fa.get("score") or o["score"] or 0), 3), fclass=fa.get("fclass", ""),
               primary=fa.get("primary", ""), stratum=fa.get("stratum", ""), t_ev=fa.get("t_ev_s", ""), n_pre=int(r["pre"].sum()),
               offlog_pre=int((r["pre"] & r["offlog"]).sum()))
    if ok.any():
        j = np.where(ok)[0][np.argmax(a[ok])]
        mv = ok & (r["vlog"] >= 3)
        jm = np.where(mv)[0][np.argmax(r["g"][mv])] if mv.any() else None
        row.update(max_abs_ds=round(float(a[j]), 1), ds_at_max=round(float(r["ds"][j]), 1), t_max=round(float(r["t"][j]), 1),
                   tau_at_max=None if np.isnan(r["tau"][j]) else round(float(r["tau"][j]), 2), g_at_max=round(float(r["g"][j]), 1),
                   vlog_at_max=round(float(r["vlog"][j]), 1), ve_at_max=round(float(r["ve"][j]), 1), past_end=bool(r["past"][j]),
                   min_ds=round(float(r["ds"][ok].min()), 1), max_ds=round(float(r["ds"][ok].max()), 1),
                   share10=round(float((a[ok] > 10).mean()), 3),
                   mov_g=None if jm is None else round(float(r["g"][jm]), 1), mov_t=None if jm is None else round(float(r["t"][jm]), 1),
                   mov_ds=None if jm is None else round(float(r["ds"][jm]), 1),
                   mov_tau=None if jm is None or np.isnan(r["tau"][jm]) else round(float(r["tau"][jm]), 2),
                   mov_vlog=None if jm is None else round(float(r["vlog"][jm]), 1))
    return row


def speed_after(r, H=30):
    """Moving-log pre-failure steps with t + 3 s also pre-failure: (ds, ve / vlog, change of ve - vlog over the next 3 s)."""
    n = len(r["t"])
    i = np.arange(n - H)
    m = r["pre"][i + H] & ~r["offlog"][i] & (r["vlog"][i] >= 3) & (r["t"][i] >= 1.7)
    i = i[m]
    rel = (r["ve"][i + H] - r["vlog"][i + H]) - (r["ve"][i] - r["vlog"][i])
    return np.c_[r["ds"][i], r["ve"][i] / r["vlog"][i], rel, r["ve"][i + H] - r["ve"][i]]


def report(out, loss_map):
    """Tables of results/log_sync.md from DIR (printed as markdown) + DIR/ranked.csv."""
    import pandas as pd
    D = pd.read_csv(f"{out}/scenes.csv")
    D["lost"] = D.score < 0.98
    LM = pd.read_csv(loss_map).assign(s8=lambda x: x.scene.str[7:15]) if loss_map else None
    if LM is not None:
        for s in ("s0", "s1"):
            m = dict(zip(LM.s8, LM[f"R6DAG-{s}_cls"]))
            k = D.read == f"R6DAG-{s}"
            D.loc[k, "cls"] = D.loc[k, "scene"].map(m)
    D["cls"] = D.get("cls", pd.Series(index=D.index, dtype=object)).fillna(D.fclass)
    pr = lambda *a: print(*a)  # noqa: E731
    pr("## steps (pre-failure, on the log path)\n")
    pr("| read | steps | off log | p50 / p90 / p99 abs ds m | > 5 m | > 10 m | > 20 m | behind > 10 / > 20 | ahead > 10 / > 20 | ahead past log end | scenes max > 5 / 10 / 20 m |")
    pr("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for lab in D.read.unique():
        z = np.load(f"{out}/steps_{lab}.npz")["steps"]
        P = z[(z[:, 8] > 0) & (z[:, 9] == 0)]
        ad, ds = np.abs(P[:, 2]), P[:, 2]
        S = D[D.read == lab]
        pr(f"| {lab} | {len(P)} | {((z[:, 8] > 0) & (z[:, 9] > 0)).mean() / (z[:, 8] > 0).mean():.1%} | "
           f"{np.median(ad):.1f} / {np.percentile(ad, 90):.1f} / {np.percentile(ad, 99):.1f} | {(ad > 5).mean():.1%} | {(ad > 10).mean():.1%} | "
           f"{(ad > 20).mean():.1%} | {(ds < -10).mean():.1%} / {(ds < -20).mean():.1%} | {(ds > 10).mean():.1%} / {(ds > 20).mean():.1%} | "
           f"{(P[:, 10] > 0).mean():.1%} | " + " / ".join(f"{(S.max_abs_ds > x).mean():.0%}" for x in (5, 10, 20)) + " |")
    pr("\n## time equivalent tau (s) at |ds| > 10 m, pre-failure (behind: tau < 0)\n")
    for lab in D.read.unique():
        z = np.load(f"{out}/steps_{lab}.npz")["steps"]
        P = z[(z[:, 8] > 0) & (z[:, 9] == 0) & (np.abs(z[:, 2]) > 10)]
        b, f = P[P[:, 2] < 0, 3], P[(P[:, 2] > 0) & np.isfinite(P[:, 3]), 3]
        pr(f"- {lab}: behind tau median {np.median(b):.1f} s (p90 {np.percentile(b, 10):.1f}); ahead tau median {np.median(f):.1f} s "
           f"(p90 {np.percentile(f, 90):.1f}); ahead past the log end {np.isnan(P[P[:, 2] > 0, 3]).mean():.0%}")
    pr("\n## scenes by stratum: share with pre-failure max |ds| > 10 m (behind-side min ds < -10 / ahead-side max ds > 10)\n")
    pr("| stratum | " + " | ".join(D.read.unique()) + " |\n|:--|" + "--:|" * D.read.nunique())
    for s, g in D.groupby("stratum"):
        pr(f"| {s} (n {len(g) // D.read.nunique()}) | " + " | ".join(
            f"{(x.max_abs_ds > 10).mean():.0%} ({(x.min_ds < -10).mean():.0%} / {(x.max_ds > 10).mean():.0%})" for _, x in g.groupby("read", sort=False)) + " |")
    pr("\n## outcome: share of scenes with pre-failure max |ds| > 10 m, behind < -10 / ahead > 10; median max |ds|\n")
    pr("| class | " + " | ".join(D.read.unique()) + " |\n|:--|" + "--:|" * D.read.nunique())
    groups = [("ok (score >= 0.98)", lambda x: ~x.lost), ("lost (< 0.98)", lambda x: x.lost)] + [
        (c, (lambda c: lambda x: x.lost & (x.cls == c))(c)) for c in sorted(D[D.lost].cls.dropna().unique())]
    for name, f in groups:
        cells = []
        for lab in D.read.unique():
            x = D[(D.read == lab)]
            x = x[f(x)]
            cells.append(f"{len(x)}: {(x.max_abs_ds > 10).mean():.0%} ({(x.min_ds < -10).mean():.0%} / {(x.max_ds > 10).mean():.0%}), "
                         f"{x.max_abs_ds.median():.1f}" if len(x) else "-")
        pr(f"| {name} | " + " | ".join(cells) + " |")
    pr("\n## after a moving-log step (v_log >= 3, t >= 1.7 s, t + 3 s pre-failure): change of v_e - v_log over 3 s (m/s), mean [n]; "
       "in brackets the subset with v_e / v_log in [0.8, 1.2]\n")
    bins = [-1e9, -20, -10, -5, 5, 10, 20, 1e9]
    names = ["< -20", "-20..-10", "-10..-5", "-5..5", "5..10", "10..20", "> 20"]
    pr("| ds bin m | " + " | ".join(D.read.unique()) + " |\n|:--|" + "--:|" * D.read.nunique())
    sp = {lab: np.load(f"{out}/steps_{lab}.npz")["speed"] for lab in D.read.unique()}
    for i, nm in enumerate(names):
        cells = []
        for lab, s in sp.items():
            m = (s[:, 0] >= bins[i]) & (s[:, 0] < bins[i + 1])
            mm = m & (s[:, 1] >= 0.8) & (s[:, 1] <= 1.2)
            cells.append(f"{s[m, 2].mean():+.2f} [{m.sum()}] ({s[mm, 2].mean():+.2f} [{mm.sum()}])")
        pr(f"| {nm} | " + " | ".join(cells) + " |")
    R = D.dropna(subset=["mov_g"]).sort_values("mov_g", ascending=False)
    top = R.drop_duplicates("scene").head(12)
    oth = {(r.read, r.scene): r for r in D.itertuples()}
    rows = []
    for r in top.itertuples():
        rows.append(dict(scene=r.scene, read=r.read, t=r.mov_t, g=r.mov_g, ds=r.mov_ds, tau=r.mov_tau, vlog=r.mov_vlog, score=r.score, cls=r.cls,
                         stratum=r.stratum, **{f"score_{k}": oth[(k, r.scene)].score for k in D.read.unique() if (k, r.scene) in oth},
                         **{f"g_{k}": oth[(k, r.scene)].mov_g for k in D.read.unique() if (k, r.scene) in oth}))
    pd.DataFrame(rows).to_csv(f"{out}/ranked.csv", index=False)
    pr("\n## ranked (moving log, pre-failure, largest camera distance g)\n")
    pr(pd.DataFrame(rows).to_markdown(index=False))
    # sanity (handoff) and zeros: offset in the 3 s before the scored failure, and the largest moving-log offset before it
    pr("\n## handoff sanity and zeros\n")
    Z = []
    for lab in D.read.unique():
        z = np.load(f"{out}/steps_{lab}.npz")["steps"]
        h = z[z[:, 1] <= 1.5]
        S = D[D.read == lab].reset_index(drop=True)
        pr(f"- {lab}: |ds| at t <= 1.5 s p99 {np.percentile(np.abs(h[:, 2]), 99):.2f} m")
        for i, r in S.iterrows():
            if not (r.score == 0 and pd.notna(r.t_ev)):
                continue
            x = z[(z[:, 0] == i) & (z[:, 8] > 0) & (z[:, 9] == 0)]
            w = x[x[:, 1] >= float(r.t_ev) - 3]
            mv = x[x[:, 6] >= 3]
            j = np.argmax(mv[:, 5]) if len(mv) else None
            Z.append(dict(read=lab, scene=r.scene, cls=r.cls, t_ev=float(r.t_ev), ds_fail=x[-1, 2] if len(x) else np.nan,
                          max3=np.abs(w[:, 2]).max() if len(w) else np.nan, g=mv[j, 5] if j is not None else np.nan,
                          ds=mv[j, 2] if j is not None else np.nan, t=mv[j, 1] if j is not None else np.nan,
                          tau=mv[j, 3] if j is not None else np.nan, vlog=mv[j, 6] if j is not None else np.nan))
    Z = pd.DataFrame(Z)
    pr("\nZeros with a failure time: share with max |ds| > 10 m in the 3 s before it; by class (n: share, median ds at the failure)\n")
    pr("| class | " + " | ".join(D.read.unique()) + " |\n|:--|" + "--:|" * D.read.nunique())
    for c in sorted(Z.cls.unique()):
        pr(f"| {c} | " + " | ".join((lambda x: f"{len(x)}: {(x.max3 > 10).mean():.0%}, {x.ds_fail.median():+.1f}" if len(x) else "-")(Z[(Z.read == lab) & (Z.cls == c)])
                                    for lab in D.read.unique()) + " |")
    Z.round(2).to_csv(f"{out}/zeros.csv", index=False)
    pr("\nTime-matched control: each zero's [t_ev - 3, t_ev] window against the ok scenes over the same window (share |ds| > 10 m, "
       "behind / ahead)\n")
    for lab in D.read.unique():
        z = np.load(f"{out}/steps_{lab}.npz")["steps"]
        z = z[z[:, 9] == 0]
        S = D[D.read == lab].reset_index(drop=True)
        ok = np.isin(z[:, 0], S.index[S.score >= 0.98])
        zs, cs = [], []
        for r in Z[Z.read == lab].itertuples():
            w = (z[:, 1] >= r.t_ev - 3) & (z[:, 1] <= r.t_ev)
            i = S.index[S.scene == r.scene][0]
            x = z[w & (z[:, 0] == i), 2]
            zs.append([np.abs(x).max() > 10, x.min() < -10, x.max() > 10] if len(x) else [False] * 3)
            g = pd.DataFrame(dict(i=z[w & ok, 0], ds=z[w & ok, 2])).groupby("i").ds
            cs.append([(g.apply(lambda v: np.abs(v).max()) > 10).mean(), (g.min() < -10).mean(), (g.max() > 10).mean()])
        zs, cs = np.mean(zs, 0), np.mean(cs, 0)
        pr(f"- {lab}: zeros {zs[0]:.0%} ({zs[1]:.0%} / {zs[2]:.0%}); ok scenes {cs[0]:.0%} ({cs[1]:.0%} / {cs[2]:.0%})")
    pr("\nZeros ranked by moving-log camera distance before the failure\n")
    pr(Z.sort_values("g", ascending=False).drop_duplicates("scene").head(12).round(2).to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--loss-map", default="")
    a = ap.parse_args()
    if a.report:
        return report(a.out, a.loss_map)
    os.makedirs(a.out, exist_ok=True)
    rows, summ = [], {}
    for spec in a.read:
        label, pkl, facts = spec.split("=")
        F = {r["scene"]: r for r in csv.DictReader(open(facts))}
        R = pickle.load(open(pkl, "rb"))["rolls"]
        st, sp = [], []
        for o in R:
            if "ctrl" not in o or o["scene"] not in F:
                continue
            fa = F[o["scene"]]
            r = rollout(o, float(fa["t_ev_s"]) if fa.get("t_ev_s") else None)
            rows.append(scene_row(label, o, fa, r))
            st.append(np.c_[np.full(len(r["t"]), len(st)), r["t"], r["ds"], r["tau"], r["d"], r["g"], r["vlog"], r["ve"], r["pre"],
                            r["offlog"], r["past"]])
            sp.append(speed_after(r))
        st, sp = np.concatenate(st), np.concatenate(sp)
        np.savez_compressed(f"{a.out}/steps_{label}.npz", steps=st, speed=sp,
                            cols="i,t,ds,tau,d,g,vlog,ve,pre,offlog,past", scol="ds,ratio,rel3,dve3")
        P = st[(st[:, 8] > 0) & (st[:, 9] == 0)]
        ad = np.abs(P[:, 2])
        summ[label] = dict(rollouts=len(set(st[:, 0])), steps_pre=len(P), offlog_pre=int(((st[:, 8] > 0) & (st[:, 9] > 0)).sum()),
                           **{f"gt{x}": float((ad > x).mean()) for x in (5, 10, 20)},
                           **{f"behind{x}": float((P[:, 2] < -x).mean()) for x in (5, 10, 20)},
                           **{f"ahead{x}": float((P[:, 2] > x).mean()) for x in (5, 10, 20)},
                           p50=float(np.median(ad)), p90=float(np.percentile(ad, 90)), p99=float(np.percentile(ad, 99)))
        print(label, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ[label].items()}), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{a.out}/scenes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
