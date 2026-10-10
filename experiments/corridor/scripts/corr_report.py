"""CORR0 report (called by corr.py report): tables, read lines and figures of the four numbers -> $DATA_DIR/runs/corridor/report/.
Map, logged future and driven lane sequence are privileged; every arm is an analysis swap. Bootstrap unit = log (decision 207's CB)."""
import json

import numpy as np

from corr import OUT, PT, RES, SEEDS, TAB, SUB8

CLS = ["a1 other exit", "a2 other lane", "b along-track", "c cross-track", "unmatched end", "short (< 2 m)"]
ARM_NAME = {"pp": "PP stored plan", "ce": "CE corridor path + plan error", "cp": "CP corridor path (no error)", "ke": "KE clamped log + plan error",
            "kp": "KP clamped log (no error)", "lp": "LP log path, plan timing (d207)", "ll": "LL log (d207)"}


def main(a):
    import pandas as pd
    import fd_navsim as FD
    import pt_swap as PS
    from jevdrive.bench.navsim import SUBS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "report", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        RES.mkdir(parents=True, exist_ok=True)
        F = np.load(OUT / "feat.npz")
        tab = np.load(TAB)
        names, logs, fut = tab["names"].astype(str), tab["log"].astype(str), tab["fut"].astype(np.float64)
        n = len(names)
        dpsi = np.degrees(fut[:, -1, 2])
        sgn = np.sign(dpsi)
        ok = F["ok"]
        BK = {">20": np.abs(dpsi) > 20, ">45": np.abs(dpsi) > 45, "20-45": (np.abs(dpsi) > 20) & (np.abs(dpsi) <= 45)}
        cb = PS.CB(logs)
        pc = PS.pc
        L, summ = [], {}
        P = L.append
        md = PS.md

        def load(path, keys):
            df = pd.read_csv(path)
            df = df[df.key.isin(keys)]
            return {k: {s: g.set_index("token").reindex(names)[SUBS[s]].to_numpy(float) for s in SUB8}
                    | {"score": 100 * g.set_index("token").reindex(names).score.to_numpy(float)} for k, g in df.groupby("key")}
        S = load(a.score, [f"{m}_{k}" for m in SEEDS for k in ("pp", "cp", "ce", "kp", "ke")])
        S0 = load(PT / "score_all.csv", [f"{m}_{k}" for m in SEEDS for k in ("pp", "lp")] + ["log_ll"])
        for m in SEEDS:
            S[f"{m}_lp"], S[f"{m}_ll"] = S0[f"{m}_lp"], S0["log_ll"]
        rp = pd.read_parquet(PT / "replay.parquet").set_index(["key", "token"])
        Z = np.load(PT / "poses.npz")
        kind = {}
        for m in SEEDS:
            dac = S0[f"{m}_pp"]["DAC"] < 1
            side = rp.loc[f"{m}_pp"].reindex(names).lqr_side.to_numpy(float)
            inside = dac & (side == sgn)
            with np.errstate(invalid="ignore"):
                under = FD.plan_kin(Z[f"{m}_pp"].astype(np.float64), fut)["gain"] < 0.9
            kind[m] = dict(dac=dac, inside=inside, cannot=dac & ~inside & under, lost=S0["log_ll"]["score"] - S0[f"{m}_pp"]["score"])

        # ---- 0. matching
        P("## 0. Lane-sequence matching (privileged: map + logged future)\n")
        st = F["status"]
        rows = []
        for b, mk in (("all navtest", np.ones(n, bool)), *BK.items()):
            rows.append({"tokens": b, "n": int(mk.sum()), "matched": int((mk & ok).sum()), "failed %": 100 * float((mk & ~ok).sum() / mk.sum()),
                         "no candidate t0": int((mk & (st == "no_candidate_t0")).sum()), "< 5 of 9 poses": int((mk & (st == "no_candidate_4s")).sum()),
                         "no connected sequence": int((mk & (st == "no_connected_sequence")).sum()),
                         "lane change %": 100 * float(F["lane_change"][mk & ok].mean()), "gap frames >= 1 %": 100 * float((F["gap"][mk & ok] > 0).mean()),
                         "log leaves interior %": 100 * float((F["kp_shift"][mk & ok] > 0.05).mean()), "median max |d_log| m": float(np.nanmedian(F["log_dmax"][mk & ok])),
                         "mean |d0| m": float(np.nanmean(np.abs(F["d0"][mk & ok])))})
        P(md(pd.DataFrame(rows)) + "\n")
        P("`gap frames`: token has a 0-4 s log pose with no lane candidate (heading > 60 deg off every containing lane, or > 3 m from any). "
          "`log leaves interior`: the logged path needs a shift > 5 cm to keep the ego body 0.2 m inside its own lane (the KP clamp). "
          "`max |d_log|`: largest lateral distance of the logged path from the corridor centreline; d0 = the ego's offset at t0.\n")
        summ["match"] = rows
        pd.DataFrame(rows).to_csv(RES / "match.csv", index=False)

        # ---- 1. error decomposition
        P("## 1. Error decomposition on the lane graph (SH30, token-seeds; shares in % with log-cluster 95% CIs)\n")
        cls = {}
        for m in SEEDS:
            c = np.full(n, "", dtype=object)
            g, r0, rA, rC = F[f"{m}_cls"].astype(str), F[f"{m}_r0"], F[f"{m}_rA"], F[f"{m}_rC"]
            same = np.isin(g, ["same", "same_amb"])
            c[g == "a1"], c[g == "a2"], c[g == "unmatched"] = CLS[0], CLS[1], CLS[4]
            c[same & np.isnan(r0)] = CLS[5]
            c[same & ~np.isnan(r0) & (rA < rC)] = CLS[2]
            c[same & ~np.isnan(r0) & ~(rA < rC)] = CLS[3]
            cls[m] = c
        sets = {"tokens": lambda m: np.ones(n), "DAC failures": lambda m: kind[m]["dac"].astype(float), "inside-cut": lambda m: kind[m]["inside"].astype(float),
                "cannot-make-turn": lambda m: kind[m]["cannot"].astype(float), "lost EPDMS (LL - PP)": lambda m: kind[m]["lost"]}
        tabs1 = []
        for b in (">20", ">45"):
            mk = BK[b] & ok
            rows = []
            for c in CLS + ["same exit, small error (r0 < 0.3 m)", "same exit, neither model explains half", "same_amb (overlap rule)", "lane-change tokens"]:
                r = {"class": c}
                for sn, f in sets.items():
                    num = np.zeros(n); den = np.zeros(n)
                    for m in SEEDS:
                        w = f(m)
                        same = np.isin(cls[m], CLS[2:4])
                        if c in CLS:
                            ind = cls[m] == c
                        elif c.startswith("same exit, small"):
                            ind = same & (F[f"{m}_r0"] < 0.3)
                        elif c.startswith("same exit, neither"):
                            ind = same & (np.minimum(F[f"{m}_rA"], F[f"{m}_rC"]) > 0.5 * F[f"{m}_r0"])
                        elif c.startswith("same_amb"):
                            ind = F[f"{m}_cls"].astype(str) == "same_amb"
                        else:
                            ind = F["lane_change"].astype(bool)
                        num += w * ind; den += w
                    t = cb.ratio(num, den, mk)
                    r[sn] = pc(t, scale=100)
                    summ[f"n1/{b}/{c}/{sn}"] = t
                rows.append(r)
            cnt = {sn: float(sum(np.where(mk, f(m), 0).sum() for m in SEEDS) / len(SEEDS)) for sn, f in sets.items()}
            P(f"**{b} deg** ({int(mk.sum())} matched tokens x 2 seeds; per-seed means: DAC failures {cnt['DAC failures']:.1f}, inside-cut {cnt['inside-cut']:.1f}, "
              f"cannot-make-turn {cnt['cannot-make-turn']:.1f}, lost EPDMS sum {cnt['lost EPDMS (LL - PP)']:.0f})\n")
            P(md(pd.DataFrame(rows)) + "\n")
            pd.DataFrame(rows).to_csv(RES / f"n1_{b.replace('>', 'gt')}.csv", index=False)
            tabs1.append((b, rows))
        # fit sizes on failures
        rows = []
        for b in (">20", ">45"):
            for kn in ("inside", "cannot", "pass"):
                v = {k: [] for k in ("r0", "rA", "rC", "delta", "c", "expl")}
                for m in SEEDS:
                    sel = BK[b] & ok & np.isin(cls[m], CLS[2:4]) & (kind[m][kn] if kn != "pass" else ~kind[m]["dac"])
                    v["r0"].append(F[f"{m}_r0"][sel]); v["rA"].append(F[f"{m}_rA"][sel]); v["rC"].append(F[f"{m}_rC"][sel])
                    v["delta"].append(F[f"{m}_delta"][sel]); v["c"].append(sgn[sel] * F[f"{m}_c"][sel])
                    v["expl"].append(1 - np.minimum(F[f"{m}_rA"][sel], F[f"{m}_rC"][sel]) / np.maximum(F[f"{m}_r0"][sel], 1e-6))
                v = {k: np.concatenate(x) for k, x in v.items()}
                rows.append({"bucket": b, "PP kind": kn, "token-seeds": len(v["r0"]), "r0 RMS m": float(np.mean(v["r0"])), "r after shift": float(np.mean(v["rA"])),
                             "r after offset": float(np.mean(v["rC"])), "median delta m (late +)": float(np.median(v["delta"])), "median c m (inside +)": float(np.median(v["c"])),
                             "mean explained share": float(np.mean(v["expl"])), "along-track wins %": 100 * float(np.mean(v["rA"] < v["rC"]))})
        P("Fit sizes on same-exit token-seeds (r0 = RMS distance plan curve to log curve at equal arc length; `explained` = 1 - min(rA, rC) / r0):\n")
        P(md(pd.DataFrame(rows)) + "\n")
        pd.DataFrame(rows).to_csv(RES / "n1_fits.csv", index=False)
        summ["n1_fits"] = rows
        c2 = pd.Series(np.concatenate([F[f"{m}_cls2"].astype(str)[BK[">20"] & ok & (cls[m] == CLS[4])] for m in SEEDS])).value_counts().to_dict()
        P(f"Fallback for the unmatched plan ends (> 20 deg, class of the last plan pose that has a lane candidate): {c2}\n")
        summ["n1_unmatched_fallback"] = c2

        # ---- 2. re-target swap
        P("## 2. Re-target swap (no-EC EPDMS x 100, non-reactive; seed means; privileged arms)\n")

        def fam(arm, f):
            return np.mean([f(S[f"{m}_{arm}"]) for m in SEEDS], 0)
        rows, rem = [], []
        for b in (">20", "20-45", ">45"):
            mk = BK[b] & ok
            base = fam("pp", lambda s: s["score"])
            for arm in ("pp", "ce", "cp", "ke", "kp", "lp", "ll"):
                x = fam(arm, lambda s: s["score"])
                r = {"bucket": b, "arm": ARM_NAME[arm], "n": int(mk.sum()), "EPDMS": float(np.nanmean(x[mk]))}
                t = cb.mean(x - base, mk)
                r["vs PP"] = "" if arm == "pp" else pc(t, "{:+.2f}")
                summ[f"n2/{b}/{arm}"] = dict(epdms=r["EPDMS"], gain=t)
                for s in ("DAC", "NC", "TTC", "EP", "LK", "DDC"):
                    r[s] = 100 * float(np.nanmean(fam(arm, lambda q: q[s])[mk]))
                rows.append(r)
            for sn in ("dac", "inside", "cannot"):
                for arm in ("ce", "cp", "ke", "kp", "lp"):
                    fp = sum(kind[m][sn].astype(float) for m in SEEDS)
                    fixed = sum((kind[m][sn] & (S[f"{m}_{arm}"]["DAC"] >= 1)).astype(float) for m in SEEDS)
                    fa = sum((S[f"{m}_{arm}"]["DAC"] < 1).astype(float) for m in SEEDS)
                    g = cb.ratio(fixed, fp, mk)
                    r = {"bucket": b, "PP failures": {"dac": "DAC", "inside": "inside-cut", "cannot": "cannot-make-turn"}[sn], "arm": ARM_NAME[arm],
                         "PP failures (seed mean)": float(fp[mk].sum() / 2), "gross removed %": pc(g, scale=100)}
                    if sn == "dac":
                        nt = cb.ratio(fp - fa, fp, mk)
                        r["net removed %"] = pc(nt, scale=100)
                        r["arm DAC failures (seed mean)"] = float(fa[mk].sum() / 2)
                        summ[f"n2rem/{b}/{arm}"] = dict(gross=g, net=nt)
                    else:
                        summ[f"n2rem/{b}/{arm}/{sn}"] = dict(gross=g)
                    rem.append(r)
        P(md(pd.DataFrame(rows)) + "\n")
        P("DAC failures of the stored plan removed by each arm (gross = share of PP's failures that pass; net counts the arm's new failures):\n")
        P(md(pd.DataFrame(rem).fillna("")) + "\n")
        pd.DataFrame(rows).to_csv(RES / "n2_arms.csv", index=False)
        pd.DataFrame(rem).to_csv(RES / "n2_removed.csv", index=False)
        mk = BK[">20"] & ok & ~F["lane_change"].astype(bool)
        t = cb.mean(fam("ce", lambda s: s["score"]) - fam("pp", lambda s: s["score"]), mk)
        t2 = cb.mean(fam("cp", lambda s: s["score"]) - fam("pp", lambda s: s["score"]), mk)
        P(f"Without lane-change tokens (> 20 deg, {int(mk.sum())} tokens): CE vs PP {pc(t, '{:+.2f}')}, CP vs PP {pc(t2, '{:+.2f}')}.\n")
        summ["n2_no_lc"] = dict(ce=t, cp=t2)

        # ---- 3. map heading
        P("## 3. Is map + exit choice enough for the 4 s heading (error against the logged 4 s heading, deg)\n")
        deg = np.degrees
        ex_ok = ok & np.isfinite(F["exit_h"])
        src = {"SH30 plan (own 4 s heading)": lambda: (np.mean([F[f"{m}_h_plan"] ** 2 for m in SEEDS], 0), np.concatenate([F[f"{m}_h_plan"] for m in SEEDS])),
               "map centreline at the plan's 4 s arc length": lambda: (np.mean([F[f"{m}_hR_plan"] ** 2 for m in SEEDS], 0), np.concatenate([F[f"{m}_hR_plan"] for m in SEEDS])),
               "map centreline at the log's 4 s arc length": lambda: (F["hR_log"] ** 2, F["hR_log"]),
               "exit pose (end of the first connector ahead)": lambda: (np.where(ex_ok, wrapd(F["exit_h"] - fut[:, -1, 2]) ** 2, np.nan), wrapd(F["exit_h"] - fut[:, -1, 2]))}
        rows = []
        for b, mk0 in (("all navtest", np.ones(n, bool)), ("< 20", ~BK[">20"]), *[(k, BK[k]) for k in (">20", ">45")]):
            for sn, f in src.items():
                sq, raw = f()
                mk = mk0 & ok & np.isfinite(sq)
                t = cb.ratio(sq, np.ones(n), mk, f=lambda r: np.degrees(np.sqrt(r)))
                e = deg(raw[np.tile(mk, len(raw) // n)])
                rows.append({"tokens": b, "source": sn, "n": int(mk.sum()), "RMS deg": pc(t, "{:.2f}"), "robust sigma": float(1.4826 * np.median(np.abs(e - np.median(e)))),
                             "mean abs": float(np.abs(e).mean()), "> 10 deg %": 100 * float((np.abs(e) > 10).mean())})
                summ[f"n3/{b}/{sn}"] = t
        P(md(pd.DataFrame(rows)) + "\n")
        pd.DataFrame(rows).to_csv(RES / "n3_heading.csv", index=False)

        # ---- 4. supply
        f4 = OUT / "supply/rows.parquet"
        if f4.exists():
            P("## 4. Counterfactual supply on navtrain (lane-graph paths within 40 m of arc ahead of the ego's own lane)\n")
            d = pd.read_parquet(f4)
            d["turn"] = d.dpsi.abs() > 20
            okd = d.status == "ok"
            d["na"] = d.alt.map(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            for thr in (20, 45):
                d[f"n{thr}"] = d.alt.map(lambda x: int(sum(v >= thr for v in x)) if isinstance(x, (list, np.ndarray)) else 0)
                d[f"rb{thr}"] = d.alt_rb.map(lambda x: int(sum(v >= thr for v in x)) if isinstance(x, (list, np.ndarray)) else 0)
            rows = []
            for b, g in (("navtrain all", d), ("turn > 20 deg", d[d.turn])):
                go = g[g.status == "ok"]
                kn = go[go.n_driven > 0]
                def blk(x):
                    return f"{len(x)} / {x.log.nunique()} logs"
                rows.append({"tokens": b, "n": len(g), "matched": len(go), "driven path known": len(kn), ">= 2 paths": blk(go[go.n_paths >= 2]),
                             "1 alt": int((kn.na == 1).sum()), "2 alts": int((kn.na == 2).sum()), "3+ alts": int((kn.na >= 3).sum()),
                             "alt >= 20 deg": blk(kn[kn.n20 > 0]), "rows >= 20": int(kn.n20.sum()), "alt >= 45 deg": blk(kn[kn.n45 > 0]), "rows >= 45": int(kn.n45.sum()),
                             "roadblock-level >= 20": blk(kn[kn.rb20 > 0]), "roadblock-level >= 45": blk(kn[kn.rb45 > 0]),
                             "distinct branch nodes (>= 20)": int(kn[kn.n20 > 0].groupby(["loc", "branch"]).ngroups)})
            P(md(pd.DataFrame(rows)) + "\n")
            pd.DataFrame(rows).to_csv(RES / "n4_supply.csv", index=False)
            kn = d[okd & (d.n_driven > 0)]
            per = kn[kn.n20 > 0].groupby("log").size()
            q = per.quantile([0.1, 0.5, 0.9]).round(0).tolist() if len(per) else []
            st4 = d.status.value_counts().to_dict()
            P(f"Match status: {st4}; lane-change tokens (driven path not one of the own-lane paths): {int((okd & (d.n_driven == 0)).sum())}. "
              f"Tokens per log with a >= 20 deg alternative: 10 / 50 / 90th percentile {q}, top log {int(per.max()) if len(per) else 0}; "
              f"start on a lane {int((kn[kn.n20 > 0].start_kind == 0).sum())}, inside a connector {int((kn[kn.n20 > 0].start_kind == 1).sum())}.\n")
            summ["n4"] = dict(rows=rows, status=st4, per_log_q=q)

        (RES / "tables.md").write_text("\n".join(L) + "\n")
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in summ.items()}, open(RES / "summary.json", "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
        figs(tabs1, summ, RES)
        run.summary.update(out=str(RES))


def wrapd(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def figs(tabs1, summ, RES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PSY
    PSY.apply()
    col = [PSY.PALETTE[k] for k in ("vermillion", "orange", "blue", "sky_blue", "purple")] + [PSY.BASELINE]
    # N1: class shares
    fig, axs = plt.subplots(1, 2, figsize=(PSY.DOUBLE_COLUMN_IN, 2.3), sharey=True)
    sets = ["tokens", "DAC failures", "inside-cut", "cannot-make-turn", "lost EPDMS (LL - PP)"]
    for ax, b in zip(axs, (">20", ">45")):
        bot = np.zeros(len(sets))
        for c, cl in zip(CLS, col):
            v = np.array([100 * summ[f"n1/{b}/{c}/{s}"][0] for s in sets])
            ax.bar(range(len(sets)), v, bottom=bot, color=cl, width=0.7, label=c)
            bot += v
        ax.set_xticks(range(len(sets)), ["tokens", "DAC\nfailures", "inside-\ncut", "cannot-\nmake-turn", "lost\nEPDMS"])
        PSY.bars(ax); PSY.panel(ax, f"|logged 4 s heading| {b[0]} {b[1:]} deg")
    axs[0].set_ylabel("share (%)")
    axs[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.subplots_adjust(0.07, 0.2, 0.83, 0.9, wspace=0.06)
    PSY.save(fig, RES / "n1_classes")
    # N2: arms
    fig, ax = plt.subplots(figsize=(PSY.SINGLE_COLUMN_IN, 2.3))
    arms = ["ce", "cp", "ke", "kp", "lp"]
    cl = [PSY.PALETTE[k] for k in ("vermillion", "green", "orange", "purple", "blue")]
    for j, b in enumerate((">20", ">45")):
        for k, (arm, c) in enumerate(zip(arms, cl)):
            t = summ[f"n2/{b}/{arm}"]["gain"]
            ax.bar(j + (k - 2) * 0.16, t[0], 0.15, color=c, label=arm.upper() if j == 0 else None)
            ax.errorbar(j + (k - 2) * 0.16, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], color="#222222", lw=0.6, capsize=1.5)
    ax.axhline(1.0, color="#999999", lw=0.5, ls="--")
    PSY.zero_line(ax); PSY.bars(ax)
    ax.set_xticks([0, 1], ["> 20 deg", "> 45 deg"]); ax.set_ylabel("no-EC EPDMS gain over the stored plan")
    ax.legend(ncol=5, fontsize=6.5, loc="upper left", columnspacing=0.8, handlelength=1.0)
    fig.subplots_adjust(0.17, 0.12, 0.98, 0.96)
    PSY.save(fig, RES / "n2_arms")
    # N3: heading RMS
    fig, ax = plt.subplots(figsize=(PSY.SINGLE_COLUMN_IN, 2.3))
    srcs = ["SH30 plan (own 4 s heading)", "map centreline at the plan's 4 s arc length", "map centreline at the log's 4 s arc length", "exit pose (end of the first connector ahead)"]
    cl = [PSY.BASELINE, PSY.PALETTE["blue"], PSY.PALETTE["sky_blue"], PSY.PALETTE["orange"]]
    for j, b in enumerate(("all navtest", ">20", ">45")):
        for k, (s, c) in enumerate(zip(srcs, cl)):
            t = summ[f"n3/{b}/{s}"]
            ax.bar(j + (k - 1.5) * 0.2, t[0], 0.19, color=c, label=["plan", "map @ plan arc", "map @ log arc", "exit pose"][k] if j == 0 else None)
            ax.errorbar(j + (k - 1.5) * 0.2, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], color="#222222", lw=0.6, capsize=1.5)
    for y in (3.7, 7.4):
        ax.axhline(y, color="#999999", lw=0.5, ls="--")
    PSY.bars(ax)
    ax.set_xticks([0, 1, 2], ["all", "> 20 deg", "> 45 deg"]); ax.set_ylabel("4 s heading error, RMS (deg)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.subplots_adjust(0.15, 0.12, 0.98, 0.96)
    PSY.save(fig, RES / "n3_heading")
