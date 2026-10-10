#!/usr/bin/env python3
"""HEAD1b pilot reader (experiments/corridor, amendment "补记 2026-10-10（HEAD1b）" of plans/2026-10-10-head1-prereg.md), steps B, B2 and C.
EXPLORATORY: a second attempt after the missed registered gate G3 (user-authorised); every table this writes is labelled so.

Arbitrary pilot arms against a named baseline pair on navtest (12 146 tokens; per-token seed means; differences with the log-cluster paired
bootstrap of geo_oracle.paired, B 4 000; RMS reads with head1_read.CB, B 10 000, token x seed pooled). Scores come from jevdrive.bench only.

  --base  NAME=tag_s0,tag_s1        the baseline pair (bench specs)
  --arms  NAME=tag_s0,tag_s1 ...    the arms; an arm's memory-masked read `<tag>:noside` is picked up for every seed that is scored
  --shuffled ARM=XARM ...           arm - its shuffled-content control (channel-read evidence)
  --lines memory | direct           memory (steps B, C; decision 204's convention): gain >= +0.5 with CI low > 0 = positive; < +0.3 = negative
                                    only if the channel was read, else "ceiling not measured"; between = partial. Channel read = every seed's
                                    dev ADE with another log's field >= 1.05 x its own, or arm - shuffled CI low > 0.
                                    direct (step B2: nothing is fed back): < +0.3 = negative; plus the guard lines (straight bucket CI not
                                    wholly below 0; |4 s arc-length ratio to the baseline - 1| <= 0.01)
  --replays N ...                   turn_oracle replays holding the DAC failures of every unmasked spec (cut-inside / cannot-make-the-turn)
  --widening                        step C: decision 244's measures at > 45 deg (W2 = tokens whose own plan is more than 2 m outside the logged
                                    path; signed lateral at 4 s, + = outside, against the baseline)
  --prof DIR[:DIR] --prof-data D .. --prof-split S   the fed heading profile's realised quality on the pilot's train rows against navtest
  --r2 [--pred NAME=gain ...]       the R2 conversion (decision 243: gain ~ 0.93 x R2 of the head against the baseline's plan error) recomputed
                                    for the fed navtest profile, and each predicted gain against the first arm's measured CI
Reads: EPDMS (all, < 5 / 5-20 / 20-45 / > 45 deg), turn failures at > 45 deg, 4 s arc-length ratio, dev ADE on / masked / mismatched, the
plan's own heading error at its 4 s arc length before (baseline) / after (arm) [stage-1 definition of head1_read.py: the plan's 4 s heading
minus the logged 4 s heading], arm - shuffled, masked - on, the verdict.  -> <out>/<name>_tables.md, <name>_reads.json

  $DATA_DIR/envs/op-train/bin/python experiments/corridor/scripts/head1_pilot_report.py --name b --base H0=GH0-F-s0,GH0-F-s1 \\
      --arms HP=H1P-F-s0,H1P-F-s1 HPX=H1PX-F-s0,H1PX-F-s1 --shuffled HP=HPX --replays geo_s0 h1b --r2 --pred stage1=0.48 --out <dir>
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "research"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/op_probe/scripts"),
                str(REPO / "experiments/body1/lib"), str(Path(__file__).parent)]
import geo_oracle as G  # noqa: E402

EXPL = "EXPLORATORY (HEAD1b: second attempt after the missed registered gate G3, user-authorised; not a registered read)"
BK = ("< 5 deg", "5-20 deg", "20-45 deg", "> 45 deg")
SUBS = ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")
TURN = ("inside-cut %", "cannot-make-turn %")
QS_GAIN = 0.93                                   # decision 204: the shape-only oracle's gain; decision 243: predicted gain = 0.93 x R2
GRID = np.r_[np.arange(0.0, 40.001, 2.5), 45.0, 50.0, 60.0, 70.0, 80.0]
H1 = G.D / "runs/corridor/head1"


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def pair(s):
    k, v = s.split("=", 1)
    return k, v.split(",")


def done(tag):
    fs = sorted(glob.glob(str(G.D / "runs" / "op_parity" / f"train-{tag.split('@')[0]}" / "*" / "DONE")))
    return json.loads(open(fs[-1]).read()) if fs else {}


def dev_read(tag):
    d = done(tag)
    on, mis, off = (d.get(f"e2e_dev_ade_{k}") for k in ("on", "mismatched", "masked"))
    if on is None:
        return dict(on=d.get("dev_ade"), masked=None, mismatched=None, read=None)
    return dict(on=on, masked=off, mismatched=mis, read=bool(mis >= 1.05 * on))


def at(prof, s):
    s = np.clip(s, 0, GRID[-1])
    j = np.clip(np.searchsorted(GRID, s, side="right") - 1, 0, len(GRID) - 2)
    r, f = np.arange(len(s)), (s - GRID[j]) / (GRID[j + 1] - GRID[j])
    return prof[r, j] * (1 - f) + prof[r, j + 1] * f


def arc(P):
    """(n, K, 3) poses from the origin -> arc length of the polyline through them."""
    d = np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[..., :2]], 1), axis=1)
    return np.hypot(d[..., 0], d[..., 1]).sum(1)


def poly_at(prof, s):
    """The qp polyline (path_req.poly: midpoint headings over 2.5 m steps to 40 m, then straight) at arc length s -> (n, 2)."""
    h = prof[:, :17]
    hm = 0.5 * (h[:, 1:] + h[:, :-1])
    V = np.concatenate([np.zeros((len(h), 1, 2)), 2.5 * np.stack([np.cos(hm), np.sin(hm)], -1).cumsum(1)], 1)
    V = np.concatenate([V, V[:, -1:] + 24.0 * np.stack([np.cos(h[:, -1]), np.sin(h[:, -1])], -1)[:, None]], 1)
    A = np.r_[np.arange(17) * 2.5, 64.0]
    s = np.clip(s, 0, 64.0)
    j = np.clip(np.searchsorted(A, s, side="right") - 1, 0, 16)
    f = ((s - A[j]) / (A[j + 1] - A[j]))[:, None]
    r = np.arange(len(s))
    return V[r, j] * (1 - f) + V[r, j + 1] * f


class Reader:
    def __init__(s, a):
        import pandas as pd
        import turn_oracle as TO
        from jevdrive.bench import tables as BT
        s.pd, s.BT, s.TO = pd, BT, TO
        tab = np.load(G.CR / G.TEST / "tab.npz")
        s.tok, s.log, s.fut = tab["names"], tab["log"], tab["fut"].astype(np.float64)
        dpsi = np.abs(np.degrees(np.arctan2(np.sin(s.fut[:, 7, 2]), np.cos(s.fut[:, 7, 2]))))
        s.sets = {"all": np.ones(len(s.tok), bool), "< 5 deg": dpsi < 5, "5-20 deg": (dpsi >= 5) & (dpsi <= 20), "20-45 deg": (dpsi > 20) & (dpsi <= 45),
                  "> 45 deg": dpsi > 45, "> 20 deg": dpsi > 20}
        assert (int(s.sets["> 20 deg"].sum()), int(s.sets["> 45 deg"].sum())) == (3154, 1517)
        s.Dt = TO.Data(a.replays) if a.replays else None
        assert s.Dt is None or (s.Dt.tok == s.tok).all()
        s.h4 = s.fut[:, 7, 2]
        s.arc_log = arc(s.fut)
        s._one = {}

    def scored(s, spec):
        try:
            return s.BT.load("navtest", spec)[0] is not None
        except Exception:
            return False

    def one(s, spec):
        """Per-token frame of one bench spec (scores x 100) and its plan (n, 8, 3), navtest tab order."""
        if spec in s._one:
            return s._one[spec]
        pd = s.pd
        u = s.BT.load("navtest", spec)[0]
        assert u is not None, f"no navtest result for {spec}"
        u = u.reindex(s.tok)
        assert not u.score.isna().any(), f"missing navtest tokens in {spec}"
        f = pd.DataFrame({"EPDMS": u.score.to_numpy(float)}, index=s.tok)
        for c in SUBS:
            if c in u:
                f[c] = u[c].to_numpy(float)
        f["DAC fail %"] = (u.DAC < 1).to_numpy(float)
        f["NC+TTC fail %"] = ((u.NC < 1) | (u.TTC < 1)).to_numpy(float)
        f = f * 100
        z = np.load(s.TO.pf(spec))
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        P = z["poses"][[pos[t] for t in s.tok]].astype(np.float64)
        if s.Dt is not None and not spec.endswith(":noside"):
            t = s.Dt.one(spec)[0]
            ck = s.Dt.check[spec]
            assert ck["replayed"] == ck["dac_fail"], f"{spec}: {ck['dac_fail'] - ck['replayed']} DAC failures are in none of the replays"
            for c in TURN:
                f[c] = t[c].to_numpy()
        f["arc4"] = arc(P)
        f["herr"] = np.degrees(wrap(np.unwrap(np.concatenate([np.zeros((len(P), 1)), P[..., 2]], 1), axis=1)[:, -1] - s.h4))
        s._one[spec] = (f, P)
        return s._one[spec]

    def mean(s, specs):
        fs = [s.one(x)[0] for x in specs]
        return sum(fs) / len(fs)

    def paired(s, A, B, col, sn="all"):
        r = G.paired(A, B, col, s.sets[sn], s.log)
        return [float(r["diff"]), float(r["lo"]), float(r["hi"])]


def main(a):
    import head1_read as HR
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bname, btags = pair(a.base)
    arms = dict(pair(x) for x in a.arms)
    shuf = dict(x.split("=") for x in a.shuffled)
    with Run("corridor", f"head1b/report-{a.name}", config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("navsim/navtest"))
        R = Reader(a)
        F = {bname: R.mean(btags)} | {k: R.mean(v) for k, v in arms.items()}
        J = dict(note=EXPL, name=a.name, lines=a.lines, baseline={bname: btags}, arms=arms, n_tokens=len(R.tok), strata={k: int(v.sum()) for k, v in R.sets.items()},
                 baseline_EPDMS=float(F[bname]["EPDMS"].mean()), baseline_per_seed=[float(R.one(t)[0]["EPDMS"].mean()) for t in btags],
                 baseline_dev=[dev_read(t) for t in btags], reads={})
        cb = HR.CB(np.tile(R.log, len(btags)))
        tile = lambda m: np.tile(m, len(btags))  # noqa: E731
        hb = np.concatenate([R.one(t)[0]["herr"].to_numpy() for t in btags])
        gt45 = R.sets["> 45 deg"]
        if a.widening:
            import route as RT
            off0 = np.zeros((len(R.tok), 2))
            okf = np.isfinite(R.fut).all((1, 2))
            turn = RT.turn_deg(R.fut, off0)

            def wide(spec):
                d, lat = RT.path_dist_np(R.one(spec)[1], R.fut, off0)
                so = -np.sign(turn)[:, None] * lat
                return so[:, -1], (so.max(1) > 2).astype(float)
            WB = [wide(t) for t in btags]
        for arm, tags in arms.items():
            assert len(tags) == len(btags), "arms and the baseline are paired by seed"
            A, B = F[arm], F[bname]
            r = dict(tags=tags, EPDMS=float(A["EPDMS"].mean()), per_seed_gain=[float(R.one(t)[0]["EPDMS"].mean() - R.one(b)[0]["EPDMS"].mean()) for t, b in zip(tags, btags)],
                     gain={sn: R.paired(A, B, "EPDMS", sn) for sn in ("all", *BK, "> 20 deg")},
                     subs={c: R.paired(A, B, c) for c in (*SUBS, "DAC fail %", "NC+TTC fail %") if c in A},
                     dac_fail={sn: R.paired(A, B, "DAC fail %", sn) for sn in ("> 20 deg", "> 45 deg")})
            if R.Dt is not None:
                r["turn_failures_gt45"] = {c: dict(arm=float(A[c][gt45].mean()), base=float(B[c][gt45].mean()), diff=R.paired(A, B, c, "> 45 deg")) for c in TURN}
            # 4 s arc length: ratio of means to the baseline (interval: paired difference / baseline mean) and each one's ratio to the log
            d = R.paired(A, B, "arc4")
            mb = float(B["arc4"].mean())
            r["arc4"] = dict(ratio_to_base=float(A["arc4"].mean()) / mb, lo=1 + d[1] / mb, hi=1 + d[2] / mb, arm_over_log=float(A["arc4"].mean() / R.arc_log.mean()),
                             base_over_log=float(mb / R.arc_log.mean()), by_stratum={sn: float(A["arc4"][m].mean() / B["arc4"][m].mean()) for sn, m in R.sets.items()})
            # the plan's own heading error at its 4 s arc length (= its 4 s heading - the logged 4 s heading), token x seed pooled
            ha = np.concatenate([R.one(t)[0]["herr"].to_numpy() for t in tags])
            r["plan_heading_err_rms_deg"] = {sn: dict(before=cb(HR.rms_a, hb, mask=tile(R.sets[sn])), after=cb(HR.rms_a, ha, mask=tile(R.sets[sn])),
                                                      diff=cb(HR.rms_diff, ha, hb, mask=tile(R.sets[sn]))) for sn in ("all", "> 20 deg", "> 45 deg")}
            r["dev"] = [dev_read(t) for t in tags]
            off = [t for t in tags if R.scored(f"{t}:noside")]
            if off:
                Fo, Fn = R.mean([f"{t}:noside" for t in off]), R.mean(off)
                r["masked"] = dict(seeds=off, EPDMS=float(Fo["EPDMS"].mean()), minus_on={sn: R.paired(Fo, Fn, "EPDMS", sn) for sn in ("all", *BK)},
                                   minus_base={sn: R.paired(Fo, R.mean([b for t, b in zip(tags, btags) if t in off]), "EPDMS", sn) for sn in ("all", *BK)})
            if arm in shuf:
                X = F[shuf[arm]]
                r["minus_shuffled"] = dict(control=shuf[arm], **{sn: R.paired(A, X, "EPDMS", sn) for sn in ("all", *BK)})
            if a.widening:
                WA = [wide(t) for t in tags]
                m = gt45 & okf
                o4 = G.paired(R.pd.DataFrame({"x": sum(w[0] for w in WA) / len(WA)}), R.pd.DataFrame({"x": sum(w[0] for w in WB) / len(WB)}), "x", m, R.log)
                r["widening_gt45"] = dict(n=int(m.sum()), W2=[int(w[1][m].sum()) for w in WA], W2_base=[int(w[1][m].sum()) for w in WB],
                                          lateral_4s_outward_m=[float(o4["diff"]), float(o4["lo"]), float(o4["hi"])],
                                          lateral_4s_mean_arm=float(o4["arm"]), lateral_4s_mean_base=float(o4["ref"]))
            # ---- verdict
            g = r["gain"]["all"]
            dr = [x["read"] for x in r["dev"]]
            vx = r.get("minus_shuffled", {}).get("all")
            has_mem = all(x is not None for x in dr)
            r["channel_read"] = (bool(all(dr) or (vx is not None and vx[1] > 0)) if has_mem else None)
            r["channel_evidence"] = dict(dev_mismatched_over_on=[None if x["mismatched"] is None else x["mismatched"] / x["on"] for x in r["dev"]],
                                         minus_shuffled=vx, masked_minus_on=r.get("masked", {}).get("minus_on", {}).get("all"))
            if g[0] >= 0.5 and g[1] > 0:
                v = "positive"
            elif g[0] < 0.3:
                v = "negative" if (a.lines == "direct" or r["channel_read"]) else "ceiling not measured (the channel was not read)"
            else:
                v = "partial"
            r["verdict"] = v
            if a.lines == "direct":
                st = r["gain"]["< 5 deg"]
                r["guards"] = dict(straight_bucket_ci_not_wholly_below_0=bool(st[2] >= 0), arc_ratio_within_0p01=bool(abs(r["arc4"]["ratio_to_base"] - 1) <= 0.01))
            J["reads"][arm] = r
        # ---- the fed profile: realised input quality, the pilot's train rows against navtest
        if a.prof:
            dirs = [Path(x) for x in a.prof.split(":") if x]
            ld = lambda d: np.load(next(p / f"{d}.npy" for p in dirs if (p / f"{d}.npy").exists())).astype(np.float64)  # noqa: E731

            def quality(P, lab, fut, logs, m):
                ok = m & np.isfinite(fut).all((1, 2))
                P, fut, logs = P[ok], fut[ok], logs[ok]
                L, s4 = lab["L"][ok].astype(np.float64), arc(fut)
                e4 = np.degrees(wrap(at(P, s4) - fut[:, 7, 2]))
                eg = np.degrees(wrap(P[:, 1:17] - L[:, 1:17]))
                ep = np.linalg.norm(poly_at(P, s4) - fut[:, 7, :2], axis=1)
                dy = np.abs(np.degrees(fut[:, 7, 2]))
                c = HR.CB(logs)
                res = dict(rows=int(ok.sum()), logs=int(len(set(logs.tolist()))))
                for b, mk in (("all", np.ones(len(P), bool)), ("> 20 deg", dy > 20), ("> 45 deg", dy > 45)):
                    res[b] = dict(n=int(mk.sum()), heading_err_at_logged_4s_arc_rms_deg=c(HR.rms_a, e4, mask=mk), polyline_position_err_at_logged_4s_arc_rms_m=c(HR.rms_a, ep, mask=mk),
                                  heading_err_on_grid_2p5_to_40m_rms_deg=float(np.sqrt(np.nanmean(eg[mk] ** 2))))
                return res
            lt = np.load(H1 / "labels/navtest.npz")
            assert (lt["names"] == R.tok).all()
            Q = dict(dirs=[str(d) for d in dirs], navtest=quality(ld(G.TEST), lt, R.fut, R.log, np.ones(len(R.tok), bool)))
            ln = np.load(H1 / "labels/navtrain.npz")
            allt = [f"navtrain_full.s{k}of12" for k in range(12)]
            cnt = np.cumsum([0] + [len(np.load(G.CR / d / "tab.npz")["names"]) for d in allt])
            sp = splits.load(a.prof_split)
            run.use_split(sp)
            Ps, labs, futs, logs, ms = [], {"L": []}, [], [], []
            for d in a.prof_data:
                t, k = np.load(G.CR / d / "tab.npz"), allt.index(d)
                assert (ln["names"][cnt[k]:cnt[k + 1]] == t["names"]).all()
                Ps.append(ld(d)), labs["L"].append(ln["L"][cnt[k]:cnt[k + 1]]), futs.append(t["fut"].astype(np.float64)), logs.append(t["log"])
                ms.append(sp.mask(t["log"] if sp.unit in ("log", "sequence") else t["names"]))
            Q["pilot_train_rows"] = quality(np.concatenate(Ps), {"L": np.concatenate(labs["L"])}, np.concatenate(futs), np.concatenate(logs), np.concatenate(ms)) | dict(split=sp.id, data=a.prof_data)
            Q["train_over_navtest_heading_rms"] = {b: Q["pilot_train_rows"][b]["heading_err_at_logged_4s_arc_rms_deg"]["v"] / Q["navtest"][b]["heading_err_at_logged_4s_arc_rms_deg"]["v"]
                                                   for b in ("all", "> 20 deg", "> 45 deg")}
            J["fed_profile"] = Q
            # ---- the R2 conversion of decision 243 for the fed navtest profile against the baseline's plans
            if a.r2:
                FE = np.load(a.feats or H1 / "report/feats.npz")
                assert all(f"T|{t}" in FE for t in btags), f"no plan features of {btags} in the feats file"
                P = ld(G.TEST)
                ep = np.concatenate([np.degrees(wrap(FE[f"T|{t}"][:, 1].astype(np.float64) - R.h4)) for t in btags])
                eh = np.concatenate([np.degrees(wrap(at(P, FE[f"T|{t}"][:, 0].astype(np.float64)) - R.h4)) for t in btags])
                X = {}
                for sn in ("all", "> 20 deg", "> 45 deg"):
                    m = tile(R.sets[sn])
                    X[sn] = dict(head_rms_deg=cb(HR.rms_a, eh, mask=m), policy_rms_deg=cb(HR.rms_a, ep, mask=m), diff=cb(HR.rms_diff, eh, ep, mask=m), corr=cb(HR.corr, ep, eh, mask=m),
                                 r2=cb(HR.r2, ep, eh, mask=m))
                first = next(iter(arms))
                g = J["reads"][first]["gain"]["all"]
                preds = {k: float(v) for k, v in (x.rsplit("=", 1) for x in a.pred)} | {"0.93 x R2 of the fed profile": QS_GAIN * X["all"]["r2"]["v"]}
                J["r2_conversion"] = dict(against=btags, arm=first, measured_gain=g, by_stratum=X,
                                          predictions={k: dict(predicted=v, inside_measured_ci=bool(g[1] <= v <= g[2]), verdict="holds" if g[1] <= v <= g[2] else "does not hold",
                                                               measured_over_predicted=g[0] / v if v else None) for k, v in preds.items()})
        # ---- tables
        f3 = lambda v, f="{:+.2f}": "-" if v is None else f"{f.format(v[0])} [{f.format(v[1])}, {f.format(v[2])}]"  # noqa: E731
        fc = lambda c, f="{:.2f}": f"{f.format(c['v'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"  # noqa: E731
        fl = lambda v: " / ".join("-" if x is None else f"{x:.3f}" for x in v)  # noqa: E731
        L = [f"**{EXPL}.**\n", f"`{a.name}`: navtest {len(R.tok)} tokens, per-token seed means x 100; differences to {bname} = {', '.join(btags)} ({J['baseline_EPDMS']:.2f}; per seed "
             + " / ".join(f"{x:.2f}" for x in J["baseline_per_seed"]) + f") with 95% CI (log-cluster paired bootstrap, B {G.NB}). Lines: {a.lines}.\n",
             "| arm | tags | EPDMS - base | per seed | - shuffled | memory masked - on (seeds) | dev ADE on / masked / mismatched, per seed (m) | channel read | verdict |",
             "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        for arm, r in J["reads"].items():
            mk = r.get("masked")
            L.append(f"| {arm} | {', '.join(r['tags'])} | {f3(r['gain']['all'])} | " + " / ".join(f"{x:+.2f}" for x in r["per_seed_gain"]) + f" | {f3(r.get('minus_shuffled', {}).get('all'))} | "
                     + (f"{f3(mk['minus_on']['all'])} ({len(mk['seeds'])})" if mk else "-") + " | " + " ; ".join(fl([d["on"], d["masked"], d["mismatched"]]) for d in r["dev"])
                     + f" | {'-' if r['channel_read'] is None else 'yes' if r['channel_read'] else 'NO'} | {r['verdict']} |")
        L.append(f"\nBaseline dev ADE (m): " + " ; ".join(fl([d["on"]]) for d in J["baseline_dev"]) + "\n")
        L += ["**EPDMS by logged heading change over 4 s (difference to the baseline).**\n", "| arm / contrast | < 5 deg (straight) | 5-20 deg | 20-45 deg | > 45 deg | > 20 deg |", "|:--|:--|:--|:--|:--|:--|"]
        for arm, r in J["reads"].items():
            L.append(f"| {arm} - {bname} | " + " | ".join(f3(r["gain"][sn]) for sn in (*BK, "> 20 deg")) + " |")
            if "minus_shuffled" in r:
                L.append(f"| {arm} - {r['minus_shuffled']['control']} | " + " | ".join(f3(r["minus_shuffled"][sn]) for sn in BK) + " | - |")
            if "masked" in r:
                L.append(f"| {arm} masked - {arm} | " + " | ".join(f3(r["masked"]["minus_on"][sn]) for sn in BK) + " | - |")
                L.append(f"| {arm} masked - {bname} | " + " | ".join(f3(r["masked"]["minus_base"][sn]) for sn in BK) + " | - |")
        L += ["", "**Turn failures at > 45 deg (1 517 tokens; %, difference in pp), DAC failures, sub-scores (difference to the baseline).**\n",
              "| arm | cut inside: arm / base, diff | cannot make the turn: arm / base, diff | DAC fail %, > 45 | DAC fail %, > 20 | NC | DAC | EP | TTC | NC + TTC fail % |", "|:--|" + ":--|" * 9]
        for arm, r in J["reads"].items():
            tf = r.get("turn_failures_gt45")
            cells = [f"{tf[c]['arm']:.2f} / {tf[c]['base']:.2f}, {f3(tf[c]['diff'])}" if tf else "-" for c in TURN]
            L.append(f"| {arm} | " + " | ".join(cells + [f3(r["dac_fail"]["> 45 deg"]), f3(r["dac_fail"]["> 20 deg"])] + [f3(r["subs"].get(c)) for c in ("NC", "DAC", "EP", "TTC", "NC+TTC fail %")]) + " |")
        L += ["", "**4 s arc length of the plan and the plan's own 4 s heading error (RMS, deg; token x seed pooled; before = baseline, after = arm).**\n",
              "| arm | arc ratio to base | arm / log, base / log | straight / > 45 deg ratio | heading error all: before, after, diff | > 20 deg | > 45 deg |", "|:--|" + ":--|" * 6]
        for arm, r in J["reads"].items():
            q, h = r["arc4"], r["plan_heading_err_rms_deg"]
            L.append(f"| {arm} | {q['ratio_to_base']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}] | {q['arm_over_log']:.4f}, {q['base_over_log']:.4f} | {q['by_stratum']['< 5 deg']:.4f} / {q['by_stratum']['> 45 deg']:.4f} | "
                     + " | ".join(f"{fc(h[sn]['before'])}, {fc(h[sn]['after'])}, {fc(h[sn]['diff'], '{:+.2f}')}" for sn in ("all", "> 20 deg", "> 45 deg")) + " |")
        if a.lines == "direct":
            L += ["", "**Guard lines (step B2).**\n", "| arm | straight bucket CI not wholly below 0 | abs(arc ratio - 1) <= 0.01 |", "|:--|:--|:--|"]
            L += [f"| {arm} | {r['guards']['straight_bucket_ci_not_wholly_below_0']} | {r['guards']['arc_ratio_within_0p01']} |" for arm, r in J["reads"].items()]
        if a.widening:
            L += ["", "**Decision 244's widening measures, navtest tokens over 45 deg, own plan.**\n", "| arm | W2 per seed (plan > 2 m outside the logged path) | baseline W2 | signed lateral at 4 s, arm - base (m, + = outside) | arm / base mean (m) |",
                  "|:--|:--|:--|:--|:--|"]
            for arm, r in J["reads"].items():
                w = r["widening_gt45"]
                L.append(f"| {arm} | {' / '.join(map(str, w['W2']))} | {' / '.join(map(str, w['W2_base']))} | {f3(w['lateral_4s_outward_m'], '{:+.3f}')} | {w['lateral_4s_mean_arm']:+.3f} / {w['lateral_4s_mean_base']:+.3f} |")
        if "fed_profile" in J:
            Q = J["fed_profile"]
            L += ["", f"**The fed heading profile's realised quality (error to the logged path; pilot train rows = {Q['pilot_train_rows']['split']} against navtest).** Profiles from {', '.join(Q['dirs'])}.\n",
                  "| rows | bucket | n | heading error at the logged 4 s arc, RMS deg | polyline position error at the logged 4 s arc, RMS m | heading error over the grid 2.5-40 m, RMS deg |", "|:--|:--|--:|:--|:--|--:|"]
            for k in ("pilot_train_rows", "navtest"):
                for b in ("all", "> 20 deg", "> 45 deg"):
                    q = Q[k][b]
                    L.append(f"| {k} ({Q[k]['rows']} rows, {Q[k]['logs']} logs) | {b} | {q['n']} | {fc(q['heading_err_at_logged_4s_arc_rms_deg'])} | {fc(q['polyline_position_err_at_logged_4s_arc_rms_m'])} | {q['heading_err_on_grid_2p5_to_40m_rms_deg']:.2f} |")
            L.append("\nTrain rows / navtest, heading RMS: " + ", ".join(f"{b} {v:.2f}" for b, v in Q["train_over_navtest_heading_rms"].items())
                     + " (decision 204's QH head: ADE 0.55 m on its pilot train rows against 0.76 m on navtest, ratio 0.72: an in-sample head).")
        if "r2_conversion" in J:
            X = J["r2_conversion"]
            L += ["", f"**R2 conversion (decision 243) for the fed navtest profile against {', '.join(X['against'])}, read at the plan's own 4 s arc length; measured gain of {X['arm']}: {f3(X['measured_gain'])}.**\n",
                  "| stratum | head RMS deg | plan RMS deg | diff | corr | R2 |", "|:--|:--|:--|:--|:--|:--|"]
            L += [f"| {sn} | {fc(q['head_rms_deg'])} | {fc(q['policy_rms_deg'])} | {fc(q['diff'], '{:+.2f}')} | {fc(q['corr'])} | {fc(q['r2'], '{:.3f}')} |" for sn, q in X["by_stratum"].items()]
            L += ["", "| prediction | predicted gain | inside the measured 95% CI | measured / predicted |", "|:--|--:|:--|--:|"]
            L += [f"| {k} | {q['predicted']:+.2f} | {q['verdict']} | {q['measured_over_predicted']:.2f} |" for k, q in X["predictions"].items()]
        (out / f"{a.name}_tables.md").write_text("\n".join(L) + "\n")
        (out / f"{a.name}_reads.json").write_text(json.dumps(J, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)) + "\n")
        run.info("\n".join(L))
        run.summary.update(out=str(out), verdicts={k: v["verdict"] for k, v in J["reads"].items()})


def cmd_fig(a):
    """Gain by heading bucket for every arm, its shuffled control and its masked read (run on the Mac from the committed reads.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PS
    PS.apply()
    J = json.loads(Path(a.src).read_text())
    col = PS.PALETTE
    rows = []
    for arm, r in J["reads"].items():
        rows.append((arm, r["gain"]))
        if "masked" in r:
            rows.append((f"{arm}, memory masked", r["masked"]["minus_base"]))
    cs = [col["blue"], col["orange"], PS.BASELINE, col["vermillion"], col["green"], col["purple"]]
    fig, ax = plt.subplots(figsize=(PS.DOUBLE_COLUMN_IN, 2.6))
    ks = ("all", *BK)
    w = 0.8 / len(rows)
    for i, (nm, g) in enumerate(rows):
        y, lo, hi = (np.array([g[k][j] for k in ks]) for j in range(3))
        x = np.arange(len(ks)) + (i - (len(rows) - 1) / 2) * w
        ax.bar(x, y, w * 0.92, color=cs[i % len(cs)], label=nm)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col["black"], ls="none", capsize=1.5, lw=0.6)
    for line in (0.5, 0.3):
        ax.plot([-0.45, 0.45], [line, line], color=col["orange"], lw=0.7, ls="--")
    ax.set_xticks(range(len(ks)), ["all tokens", "< 5 deg", "5-20 deg", "20-45 deg", "> 45 deg"])
    ax.set_ylabel(f"navtest EPDMS minus {next(iter(J['baseline']))}")
    ax.legend(fontsize=6.5, ncol=2)
    PS.bars(ax), PS.zero_line(ax)
    fig.tight_layout(pad=0.4)
    print(PS.save(fig, Path(a.out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--base")
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--shuffled", nargs="*", default=[])
    ap.add_argument("--lines", default="memory", choices=["memory", "direct"])
    ap.add_argument("--replays", nargs="*", default=[])
    ap.add_argument("--widening", action="store_true")
    ap.add_argument("--prof", default="")
    ap.add_argument("--prof-data", nargs="+", default=list(G.PILOT))
    ap.add_argument("--prof-split", default="navsim/op-parity-s234-train")
    ap.add_argument("--r2", action="store_true")
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--feats", default="")
    ap.add_argument("--out", default=str(G.D / "runs/corridor/head1b/report"))
    ap.add_argument("--fig", default="", help="reads.json -> figure (with --out the figure stem); no data access")
    a = ap.parse_args()
    if a.fig:
        a.src = a.fig
        cmd_fig(a)
    else:
        main(a)
