"""FLOW1 (experiments/flowhead/plans/2026-10-10-flow-head-prereg.md): checks and reads of the flow-matching / regression trajectory heads.

  ident    --a TAG --b TAG      every tensor of two pp_train checkpoints equal (exit 1 if not): the default path of the trainer is unchanged
  smoke    FILE ...             prediction files of the smoke heads written by the bench `poses` stage: tokens of navtest, finite (8, 3) poses
  trained  TAG ...              clean DONE, finite losses, head dev ADE <= 1.2 m, thead.pt present
  div      (envs/op-train, GPU) plans of the flow heads for the 16 fixed noise rows on every navtest token -> $OUT/div/samples.npz, and
                                the first 8 rows on the > 20 deg tokens as a score-poses input -> $OUT/div/poses.npz (keys s<seed>n<k>)
  geom     (.venv)              equal-arc curve offset C(4 s) of every member (decision 207's definition, pt_swap.Curve) -> $OUT/geom.npz
  report   (envs/op-train)      tables and the verdict against the registered lines -> $OUT/report/{tables.md, summary.json, *.csv}

Reads: 2-seed mean per token, cluster bootstrap by log (jevdrive.stats.paired, B 10 000). The turn failure classes are decision 153's, read
through turn_oracle.Data on the replay `flow1` (turn_oracle.py replay --name flow1 --models <the six specs>).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib")]
import argparse, glob, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

PR = data_dir() / "runs" / "op_parity"
OUT = data_dir() / "runs" / "flowhead" / "flow1"
ADE_MAX = 1.2
ARMS = {"FM": ["FMH-F-s0", "FMH-F-s1"], "RG": ["RGH-F-s0", "RGH-F-s1"], "SH30": ["SH30-F-s0", "SH30-F-s1"]}
PAIRS = (("FM", "RG"), ("RG", "SH30"), ("FM", "SH30"))
TURNS = ("<5", "5-20", "20-45", ">45")
SUBS = ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")
NK, NSCORE = 16, 8


def cmd_ident(a):
    import torch
    A, B = (torch.load(PR / "runs" / t / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"] for t in (a.a, a.b))
    diff = {}
    for grp in ("net", "parity"):
        assert A[grp].keys() == B[grp].keys(), f"{grp}: different tensors"
        for k in A[grp]:
            d = float((A[grp][k].float() - B[grp][k].float()).abs().max())
            if d > 0:
                diff[f"{grp}.{k}"] = d
    res = dict(a=a.a, b=a.b, tensors=sum(len(A[g]) for g in ("net", "parity")), differing=len(diff), max_abs=max(diff.values(), default=0.0))
    print(json.dumps(res))
    if a.out:
        _pl.Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(1 if diff else 0)


def cmd_smoke(a):
    names = np.load(PR / "cache/lb_navtest/tab.npz")["names"].astype(str)
    for f in a.files:
        z = np.load(f)
        assert z["tokens"].astype(str).tolist() == names.tolist() and z["poses"].shape == (len(names), 8, 3) and np.isfinite(z["poses"]).all(), f
        print(f"{f}: {z['poses'].shape} finite, tokens = navtest")


def cmd_trained(a):
    bad = []
    for t in a.tags:
        d = sorted(glob.glob(str(PR / f"train-{t}" / "*")))[-1]
        done = json.loads((_pl.Path(d) / "DONE").read_text())
        ok = (PR / "runs" / t / "thead.pt").exists() and done.get("steps") == a.steps and np.isfinite(done["dev_th_ade"]) and done["dev_th_ade"] <= ADE_MAX
        print(json.dumps(dict(tag=t, ok=bool(ok), dev_th_ade=done.get("dev_th_ade"), steps=done.get("steps"), train_s=done.get("train_s"))))
        bad += [] if ok else [t]
    raise SystemExit(1 if bad else 0)


# ---------------------------------------------------------------- reads
def navtest():
    """names, log, fut (n, 8, 3), turn bin of every navtest token (cache order)."""
    from jevdrive.bench import tables as BT
    tab = np.load(PR / "cache/lb_navtest/tab.npz")
    names = tab["names"].astype(str)
    return names, tab["log"].astype(str), tab["fut"].astype(np.float64), BT.navtest_strata().reindex(names).turn.to_numpy(str)


def cmd_div(a):
    import torch
    _sys.path[:0] = [str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
    import pp_train as T
    import traj_head as TH
    from jevdrive.bench import navsim as NV
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("flowhead", "flow1-div", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        names, _, _, turn = navtest()
        dev = torch.device("cuda")
        S = T.Store(["lb_navtest"], dev, need_side=False, frames="warp")
        assert S.tab["names"].astype(str).tolist() == names.tolist()
        out, t20 = {}, np.isin(turn, ["20-45", ">45"])
        sc = {"tokens": names[t20]}
        for tag in ARMS["FM"]:
            m = NV.resolve(tag, check=True)
            P = TH.serve(NV._load_ckpt(T, m.ckpt, dev), TH.load(_pl.Path(m.ckpt).with_name("thead.pt"), dev), S, ks=range(NK))
            served = np.load(NV.pred_file(m, "navtest"))
            assert served["tokens"].astype(str).tolist() == names.tolist()
            d0 = float(np.abs(P[0] - served["poses"]).max())                    # noise row 0 = the served plan
            run.info(f"{tag}: {P.shape}, max |row 0 - served| {d0:.2e}")
            assert d0 < 1e-3, f"{tag}: noise row 0 is not the served plan ({d0})"
            s = tag[-2:]
            out[s] = P
            sc |= {f"{s}n{k}": (served["poses"] if k == 0 else P[k])[t20] for k in range(NSCORE)}
        (OUT / "div").mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "div" / "samples.npz", tokens=names, **out)
        np.savez(OUT / "div" / "poses.npz", **sc)
        run.summary.update(tokens=len(names), scored_tokens=int(t20.sum()), keys=len(sc) - 1)


def cmd_geom(a):
    _sys.path[:0] = [str(_R / "experiments/op_parity/scripts")]
    import pt_swap as PT
    from jevdrive.run import Run
    with Run("flowhead", "flow1-geom", config=vars(a)) as run:
        names, _, fut, _ = navtest()
        sgn = np.sign(np.unwrap(np.concatenate([np.zeros((len(names), 1)), fut[:, :, 2]], 1), axis=1)[:, -1])
        LC = [PT.Curve(f) for f in run.tqdm(fut, desc="log curves")]
        G = {"names": names}
        for spec in sum(ARMS.values(), []):
            P = PT.member_poses(spec, names).astype(np.float64)
            C = np.zeros(len(names))
            for i in run.tqdm(range(len(names)), desc=spec):
                pc, lc = PT.Curve(P[i]), LC[i]
                ss = np.minimum(pc.sv[-1:], lc.sv[-1:])
                p, q = pc.at(ss)[0], lc.at(ss)[0]
                C[i] = sgn[i] * ((p[0] - q[0]) * -np.sin(q[2]) + (p[1] - q[1]) * np.cos(q[2]))
            G[spec] = C
        np.savez(OUT / "geom.npz", **G)


def md(df, digits=2):
    import pandas as pd
    return pd.DataFrame(df).to_markdown(index=False, floatfmt=f".{digits}f")


def cmd_report(a):
    import pandas as pd
    _sys.path[:0] = [str(_R / "research"), str(_R / "experiments/op_parity/scripts"), str(_R / "experiments/op_probe/scripts")]
    import turn_oracle as TO
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    names, log, fut, turn = navtest()
    RES = OUT / "report"
    RES.mkdir(parents=True, exist_ok=True)
    doc, summ = [], {}
    P_ = lambda s: doc.append(s)  # noqa: E731
    B = {"all": np.ones(len(names), bool)} | {t: turn == t for t in TURNS} | {">20": np.isin(turn, ["20-45", ">45"])}

    def contrast(val, m, groups, scale=1.0, per_seed=None):
        """val: arm -> per-unit array (seed mean); rows of arm means and the three paired differences on the mask m."""
        r = {k: scale * float(np.nanmean(v[m])) for k, v in val.items()}
        for x, y in PAIRS:
            q = stats.paired(val[x][m], val[y][m], groups=groups[m])
            r[f"{x} - {y}"] = f"{scale * q['mean']:+.3f} [{scale * q['lo']:+.3f}, {scale * q['hi']:+.3f}]"
            summ_row[f"{x}-{y}"] = [scale * q["mean"], scale * q["lo"], scale * q["hi"]]
        if per_seed is not None:
            r["FM - RG s0 / s1"] = " / ".join(f"{scale * float(np.nanmean(per_seed['FM'][s][m] - per_seed['RG'][s][m])):+.3f}" for s in (0, 1))
        return r

    # ---- navtest units, turn failure classes, curve offsets
    D = TO.Data(["flow1"])
    assert D.tok.astype(str).tolist() == names.tolist()
    U = {k: [BT.load("navtest", s)[0].reindex(names) for s in v] for k, v in ARMS.items()}
    Fr = {k: [D.one(s)[0] for s in v] for k, v in ARMS.items()}
    G = np.load(OUT / "geom.npz")
    seedmean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    P_("# FLOW1 tables\n\nGenerated by `experiments/flowhead/scripts/flow1.py report`. navtest 12 146 tokens, 2 seeds per arm averaged per token, "
       "differences paired, cluster bootstrap by log (B 10 000). FM = flow-matching head, RG = the same head as a regression, SH30 = the stored recipe.\n")
    P_("Seeds: " + "; ".join(f"{s} {100 * u.score.mean():.2f}" for k in ARMS for s, u in zip(ARMS[k], U[k])) + "\n")
    rows = []
    for c in ("score",) + SUBS:
        summ_row = summ.setdefault(f"navtest|all|{c}", {})
        v = {k: seedmean([u[c].to_numpy(float) for u in U[k]]) for k in ARMS}
        ps = {k: [u[c].to_numpy(float) for u in U[k]] for k in ARMS}
        rows.append(dict(metric="EPDMS" if c == "score" else c) | contrast(v, B["all"], log, 100.0, ps))
    P_("## 1. navtest, whole board (x 100)\n\n" + md(rows) + "\n")
    rows = []
    for b, m in B.items():
        summ_row = summ.setdefault(f"navtest|{b}|score", {})
        v = {k: seedmean([u.score.to_numpy(float) for u in U[k]]) for k in ARMS}
        ps = {k: [u.score.to_numpy(float) for u in U[k]] for k in ARMS}
        rows.append(dict(bucket=b, n=int(m.sum())) | contrast(v, m, log, 100.0, ps))
    P_("## 2. navtest EPDMS by logged 4 s heading change (deg)\n\n" + md(rows) + "\n")
    rows = []
    for b in (">45", ">20", "<5"):
        for c in ("DAC fail %", "inside-cut %", "cannot-make-turn %", "other DAC fail %", "NC+TTC fail %", "EP"):
            summ_row = summ.setdefault(f"navtest|{b}|{c}", {})
            v = {k: seedmean([f[c].to_numpy(float) for f in Fr[k]]) for k in ARMS}
            ps = {k: [f[c].to_numpy(float) for f in Fr[k]] for k in ARMS}
            rows.append(dict(bucket=b, metric=c) | contrast(v, B[b], log, 1.0, ps))
    P_("## 3. Turn failure classes (% of tokens; EP x 100)\n\nInside-cut = DAC failure whose first LQR-state corner outside is on the turn side; "
       "cannot-make-turn = DAC failure, not inside, 4 s heading gain < 0.9 (decision 153).\n\n" + md(rows) + "\n\nReplay check: `" + json.dumps(D.check) + "`\n")
    rows = []
    for b in (">45", ">20", "20-45", "<5", "all"):
        for nm, f in (("mean |C(4 s)| (m)", np.abs), ("mean C(4 s) (m, inside +)", lambda x: x), ("C(4 s) > +0.3 m (%)", lambda x: 100.0 * (x > 0.3)),
                      ("C(4 s) < -0.3 m (%)", lambda x: 100.0 * (x < -0.3))):
            summ_row = summ.setdefault(f"geom|{b}|{nm}", {})
            v = {k: seedmean([f(G[s]) for s in ARMS[k]]) for k in ARMS}
            ps = {k: [f(G[s]) for s in ARMS[k]] for k in ARMS}
            rows.append(dict(bucket=b, metric=nm) | contrast(v, B[b], log, 1.0, ps))
    P_("## 4. Curve offset at equal arc length, 4 s (decision 207's C; reference: SH30 0.71 m on > 20 deg, 0.88 m on > 45 deg)\n\n" + md(rows, 3) + "\n")

    # ---- navhard
    H = {k: [BT.load("navhard", s + "@gimm")[0] for s in v] for k, v in ARMS.items()}
    gi = H["RG"][0].index
    rows = []
    for c in ("combined", "stage1", "stage2"):
        summ_row = summ.setdefault(f"navhard|{c}", {})
        v = {k: seedmean([h.reindex(gi)[c].to_numpy(float) for h in H[k]]) for k in ARMS}
        ps = {k: [h.reindex(gi)[c].to_numpy(float) for h in H[k]] for k in ARMS}
        rows.append(dict(metric=c) | contrast(v, np.ones(len(gi), bool), H["RG"][0].log.to_numpy(str), 100.0 if v["RG"].max() <= 1.0 else 1.0, ps))
    P_(f"## 5. navhard two-stage (G frames, {len(gi)} groups, clustered by the log of the stage-1 token)\n\n" + md(rows) + "\n")

    # ---- sample diversity (flow arm): 16 fixed noise rows per token
    Z = np.load(OUT / "div" / "samples.npz")
    nl = np.stack([-np.sin(fut[:, 7, 2]), np.cos(fut[:, 7, 2])], 1)
    tl = np.stack([np.cos(fut[:, 7, 2]), np.sin(fut[:, 7, 2])], 1)
    st = {}
    for s in ("s0", "s1"):
        Ps = Z[s].astype(np.float64)                                            # (16, n, 8, 3)
        d = Ps[:, :, 7, :2] - fut[None, :, 7, :2]
        lat, lon, hd = (d * nl[None]).sum(-1), (d * tl[None]).sum(-1), np.degrees(Ps[:, :, 7, 2])
        srt = np.sort(lat, 0)
        rng = srt[-1] - srt[0]
        e_one = np.hypot(*(Ps[0, :, :, :2] - fut[:, :, :2]).transpose(2, 0, 1)).mean(1)
        e_mean = np.hypot(*(Ps.mean(0)[:, :, :2] - fut[:, :, :2]).transpose(2, 0, 1)).mean(1)
        st[s] = dict(sd_lat=lat.std(0, ddof=1), sd_lon=lon.std(0, ddof=1), sd_hd=hd.std(0, ddof=1), rng_lat=rng, rng_hd=hd.max(0) - hd.min(0),
                     gap=np.diff(srt, axis=0).max(0) / np.maximum(rng, 1e-6), err_lat=np.abs(lat[0]), err_lon=np.abs(lon[0]),
                     in_rng=((srt[0] <= 0) & (srt[-1] >= 0)).astype(float), ade_one=e_one, ade_mean=e_mean)
    q = {k: (st["s0"][k] + st["s1"][k]) / 2 for k in st["s0"]}
    zero = np.zeros(len(names))
    ci = lambda x, m, f="{:.3f}": (lambda r: f"{f} [{f}, {f}]".format(r["mean"], r["lo"], r["hi"]))(stats.paired(x[m], zero[m], groups=log[m]))  # noqa: E731
    rows = []
    for b in TURNS + (">20",):
        m = B[b]
        from scipy.stats import spearmanr
        rms = float(np.sqrt(np.mean(np.concatenate([st[s]["err_lat"][m] for s in st]) ** 2)))
        rows.append({"bucket": b, "n": int(m.sum()), "sd lateral 4 s (m)": ci(q["sd_lat"], m), "sd along 4 s (m)": ci(q["sd_lon"], m), "sd heading 4 s (deg)": ci(q["sd_hd"], m, "{:.2f}"),
                     "range lateral (m)": ci(q["rng_lat"], m), "range > 1 m (%)": ci(100.0 * (q["rng_lat"] > 1.0), m, "{:.1f}"),
                     "heading range > 10 deg (%)": ci(100.0 * (q["rng_hd"] > 10.0), m, "{:.1f}"), "max gap / range": ci(q["gap"], m),
                     "RMS lateral error of the served plan (m)": rms, "mean sd / RMS error": float(q["sd_lat"][m].mean() / rms),
                     "Spearman(sd, |error|)": float(spearmanr(q["sd_lat"][m], q["err_lat"][m]).statistic),
                     "log inside the sample range (%)": ci(100.0 * q["in_rng"], m, "{:.1f}"),
                     "ADE served (m)": float(q["ade_one"][m].mean()), "ADE of the 16-sample mean (m)": float(q["ade_mean"][m].mean())})
        summ[f"div|{b}"] = {k: v for k, v in rows[-1].items() if isinstance(v, float)} | dict(sd_lat=float(q["sd_lat"][m].mean()), rng_gt1=float(100 * (q["rng_lat"][m] > 1).mean()),
                                                                                             sd_hd=float(q["sd_hd"][m].mean()), gap=float(q["gap"][m].mean()))
    t = pd.DataFrame(rows).set_index("bucket").T.reset_index().rename(columns={"index": "read"})
    P_("## 6. Sample diversity of the flow head (16 fixed noise rows per token, both seeds averaged per token)\n\nLateral = along the normal of the logged path at 4 s. "
       "Max gap / range: the largest gap between neighbouring samples over their range (16 uniform draws give about 0.2, two tight clusters about 1).\n\n"
       + t.to_markdown(index=False, floatfmt=".3f") + "\n")

    # ---- best-of-N oracle on > 20 deg tokens (score-poses, non-reactive, no EC): analysis only, the pick is privileged
    sc = pd.read_csv(OUT / "div" / "score.csv")
    cols = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
            "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "LK": "lane_keeping", "HC": "history_comfort"}
    tk = names[B[">20"]]
    idn, O = {}, {}
    for si, s in enumerate(("s0", "s1")):
        S_ = np.stack([sc[sc.key == f"{s}n{k}"].set_index("token").score.reindex(tk).to_numpy(float) for k in range(NSCORE)])
        Dc = np.stack([sc[sc.key == f"{s}n{k}"].set_index("token")[cols["DAC"]].reindex(tk).to_numpy(float) for k in range(NSCORE)])
        assert np.isfinite(S_).all()
        z0, u = sc[sc.key == f"{s}n0"].set_index("token").reindex(tk), U["FM"][si].reindex(tk)
        idn[s] = {k: float(np.abs(z0[c].to_numpy(float) - u[k].to_numpy(float)).max()) for k, c in cols.items()}
        bi = S_.argmax(0)
        O[s] = dict(one=S_[0], mean=S_.mean(0), best=S_.max(0), worst=S_.min(0), dac_one=100.0 * (Dc[0] < 1), dac_mean=100.0 * (Dc < 1).mean(0),
                    dac_best=100.0 * (Dc[bi, np.arange(len(tk))] < 1), dac_any=100.0 * (Dc < 1).all(0), flip=100.0 * ((Dc < 1).any(0) & ~(Dc < 1).all(0)))
    o = {k: (O["s0"][k] + O["s1"][k]) / 2 for k in O["s0"]}
    lg, tt = log[B[">20"]], turn[B[">20"]]
    rows = []
    for b, m in ((">20", np.ones(len(tk), bool)), (">45", tt == ">45")):
        pr = lambda x, y, sc_=1.0: (lambda r: f"{sc_ * r['mean']:+.2f} [{sc_ * r['lo']:+.2f}, {sc_ * r['hi']:+.2f}]")(stats.paired(x[m], y[m], groups=lg[m]))  # noqa: E731
        rows.append({"bucket": b, "n": int(m.sum()), "served (row 0)": 100 * o["one"][m].mean(), "mean of 8 rows": 100 * o["mean"][m].mean(), "best of 8": 100 * o["best"][m].mean(),
                     "worst of 8": 100 * o["worst"][m].mean(), "best - served": pr(o["best"], o["one"], 100.0), "mean - served": pr(o["mean"], o["one"], 100.0),
                     "DAC fail % served": o["dac_one"][m].mean(), "DAC fail % mean of 8": o["dac_mean"][m].mean(), "DAC fail % of the best row": o["dac_best"][m].mean(),
                     "DAC fails on all 8 rows %": o["dac_any"][m].mean(), "DAC outcome differs between rows %": o["flip"][m].mean()})
        summ[f"oracle|{b}"] = {k: v for k, v in rows[-1].items() if not isinstance(v, str)} | {"best-served": rows[-1]["best - served"], "mean-served": rows[-1]["mean - served"]}
    t = pd.DataFrame(rows).set_index("bucket").T.reset_index().rename(columns={"index": "read"})
    P_("## 7. Best-of-8 oracle on > 20 deg tokens (no-EC EPDMS x 100, non-reactive; the pick uses the score: analysis only)\n\n" + t.to_markdown(index=False, floatfmt=".2f")
       + f"\n\nIdentity of row 0 against the bench units (max abs difference of the 8 sub-scores, per seed): `{json.dumps(idn)}`\n")
    summ["oracle_identity"] = idn

    # ---- registered lines
    rg = float(np.mean([100 * u.score.mean() for u in U["RG"]]))
    d45, ep, s5 = summ["navtest|>45|DAC fail %"]["FM-RG"], summ["navtest|all|score"]["FM-RG"], summ["navtest|<5|score"]["FM-RG"]
    c45 = summ["geom|>45|mean |C(4 s)| (m)"]["FM-RG"]
    V = dict(S=dict(rg_epdms=rg, band=[89.25, 89.85], ok=bool(89.25 <= rg <= 89.85)),
             primary=dict(diff=d45, verdict="effective" if d45[0] <= -1.0 and d45[2] < 0 else "worse" if d45[0] >= 1.0 and d45[1] > 0 else "no effect"),
             guard=dict(epdms=ep, s5=s5, ok=bool(ep[1] >= -0.3 and s5[0] >= -0.2)),
             mechanism=dict(c45=c45, ok=bool(c45[0] <= -0.10 and c45[2] < 0)))
    V["continue"] = bool(V["primary"]["verdict"] == "effective" and V["guard"]["ok"] and ep[0] >= 0.3)
    summ["verdict"] = V
    P_("## 8. Registered lines\n\n```json\n" + json.dumps(V, indent=1) + "\n```\n")
    (RES / "tables.md").write_text("\n".join(doc))
    (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("\n".join(doc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    p = sp.add_parser("smoke")
    p.add_argument("files", nargs="+")
    p = sp.add_parser("trained")
    p.add_argument("tags", nargs="+")
    p.add_argument("--steps", type=int, default=10000)
    for c in ("div", "geom", "report"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"ident": cmd_ident, "smoke": cmd_smoke, "trained": cmd_trained, "div": cmd_div, "geom": cmd_geom, "report": cmd_report}[a.cmd](a)
