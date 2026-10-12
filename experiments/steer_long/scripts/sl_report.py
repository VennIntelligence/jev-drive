#!/usr/bin/env python3
"""STEER-LONG report: the registered reads of research/next-round/steer-long.md section 3 from sl_read.py's outputs, plus the human-log
and navhard collateral of the cap (section 3.7 (ii), (iii)). Mac, CPU.

  sl_report.py [--res experiments/steer_long/results] [--root tmp/steer_long]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def pct(x):
    return f"{100 * x:.1f} %"


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def boot_slope(df, xcol, ycol, ccol, n=1000):
    rng = np.random.default_rng(0)
    G = [g[[xcol, ycol]].to_numpy() for _, g in df.groupby(ccol)]

    def slope(A):
        return float(np.polyfit(A[:, 0], A[:, 1], 1)[0])
    b = [slope(np.vstack([G[i] for i in rng.integers(0, len(G), len(G))])) for _ in range(n)]
    return slope(np.vstack(G)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def human_2hz(root):
    """nuPlan per_decision (zeros_mech collect): logged and ego lateral acceleration at 2 Hz from 0.5 s finite differences."""
    D = pd.read_parquet(root / "per_decision.parquet").sort_values(["label", "scene", "k"])
    out = []
    for (lab, sc), g in D.groupby(["label", "scene"]):
        if len(g) < 3:
            continue
        sl = g.s_log_t.to_numpy()
        yl = np.unwrap(g.yaw_log.to_numpy())
        ye = np.unwrap(g.yaw.to_numpy())
        vl = np.diff(sl) / 0.5
        wl = np.diff(yl) / 0.5
        ve = np.hypot(np.diff(g.x), np.diff(g.y)) / 0.5
        we = np.diff(ye) / 0.5
        out.append(pd.DataFrame(dict(label=lab, scene=sc, v_log=vl, a_log=np.abs(vl * wl), v_e=ve, a_e=np.abs(ve * we))))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(REPO / "experiments/steer_long/results"))
    ap.add_argument("--root", default=str(REPO / "tmp/steer_long"))
    a = ap.parse_args()
    res, root = Path(a.res), Path(a.root)
    GT = json.load(open(res / "gates_thresholds.json"))
    E = pd.read_csv(res / "rollouts.csv")
    ER = pd.read_csv(res / "events.csv")
    CAP = pd.read_csv(res / "cap_grid.csv")
    L = ["# STEER-LONG read (pre-registration: research/next-round/steer-long.md section 3, commit 8f48d0c0)", ""]
    J = {}

    # ---- gates
    L += ["## Gates", "", "| track / label | summary rollouts | with controller + driver | G-frame | G-rec |", "|:--|--:|--:|--:|--:|"]
    for k, g in GT["gates"].items():
        L.append(f"| {k} | {g['summary']} | {g['with_ctrl']} | {g['frame_ok']} / {g['frame_pairs']} ({pct(g['frame_ok'] / max(g['frame_pairs'], 1))}) | "
                 f"{g['rec_ok']} / {g['rec_pairs']} ({pct(g['rec_ok'] / max(g['rec_pairs'], 1))}) |")
    zr = pd.read_parquet(root / "per_rollout.parquet")
    zr = zr[zr.has_rollout & (zr.type != "") & zr.label.isin(set(E.label))][["label", "scene", "type"]]
    mine = E[(E.track == "nuplan") & (E.kind.fillna("") != "")][["label", "scene", "kind"]]
    mm = zr.merge(mine, on=["label", "scene"], how="outer")
    mis = mm[mm.type.fillna("-") != mm.kind.fillna("-").replace({"other": "other"})]
    L += ["", f"G-ev: zeros_mech zero rollouts {len(zr)}, this read {len(mine)}, mismatched rows {len(mis)}"]
    if len(mis):
        L += ["", md(mis.head(20))]
    J["G_ev_mismatch"] = int(len(mis))

    # ---- thresholds
    L += ["", "## Thresholds (p99 of clean-rollout rows, v >= 1 m/s)", "", "| track / bucket | n rows | A_path | A_cmd | A_exec | A_req | ddelta (rad/s) |",
          "|:--|--:|--:|--:|--:|--:|--:|"]
    for k, t in GT["thresholds"].items():
        L.append(f"| {k} | {t['n']} | {t['A_path']:.2f} | {t['A_cmd']:.2f} | {t['A_exec']:.2f} | {t['A_req']:.2f} | {t['ddelta']:.3f} |")

    # ---- events: verdicts by track x bucket
    L += ["", "## Zero events: who moved first (section 3.5)", ""]
    rows = []
    for (tr, b), g in ER.groupby(["track", "bucket"]):
        n = len(g)
        rows.append(dict(track=tr, bucket=b, n=n, plan_first=int((g.verdict == "plan_first").sum()), steer_first_amp=int((g.verdict == "steer_first_amp").sum()),
                         steer_first_ref=int((g.verdict == "steer_first_ref").sum()), quiet=int((g.verdict == "quiet").sum()),
                         spin_like=int(g.spin.sum()), loop_O=int(g.loop_O.sum()), loop_R=int(g.loop_R.sum()), loop_R2deg=int(g.loop_R2deg.sum()),
                         amp_median=float(np.nanmedian(g.amp)) if g.amp.notna().any() else np.nan))
    V = pd.DataFrame(rows)
    L += [md(V, 2), ""]
    lines = []
    for r in V.to_dict("records"):
        if r["n"] < 10:
            lines.append(f"- {r['track']} / {r['bucket']}: n = {r['n']} < 10, described only")
            continue
        pf, amp = r["plan_first"] / r["n"], r["steer_first_amp"] / r["n"]
        hl1 = r["bucket"] == "high" and pf >= 0.6
        lines.append(f"- {r['track']} / {r['bucket']}: plan first {pct(pf)}{' -> HL1 holds' if hl1 else ''}; steer first amplified {pct(amp)}"
                     f"{' -> HL2 holds' if amp >= 0.4 else ''}")
        sp = ER[(ER.track == r["track"]) & (ER.bucket == r["bucket"]) & ER.spin]
        if r["bucket"] == "low" and len(sp):
            fr = (sp.loop_O | sp.loop_R).mean()
            lines.append(f"  low-speed spin-like events {len(sp)}: O or R {pct(fr)}{' -> HL3 visible' if fr >= 0.3 and len(sp) >= 10 else ''}")
    L += lines
    by_kind = ER.groupby(["track", "kind", "verdict"]).size().unstack(fill_value=0)
    L += ["", "By zero type:", "", md(by_kind.reset_index())]
    # signal sizes in the window, events vs thresholds
    L += ["", "Event window maxima (median [p90]) by track x bucket:", "", "| track / bucket | A_path | A_cmd | A_exec | ddelta | v at event |", "|:--|--:|--:|--:|--:|--:|"]
    for (tr, b), g in ER.groupby(["track", "bucket"]):
        f = lambda c: f"{np.nanmedian(g[c]):.2f} [{np.nanpercentile(g[c], 90):.2f}]"   # noqa: E731
        L.append(f"| {tr} / {b} | {f('A_path_max')} | {f('A_cmd_max')} | {f('A_exec_max')} | {f('ddelta_max')} | {f('v_ev')} |")
    J["verdicts"] = V.to_dict("records")
    # exploratory (amendment 1): the same onset rule with the bounded steering state, A_delta = v^2 |tan(delta)| / L, in place of A_cmd
    if (root / "rows.parquet").exists():
        Rx = pd.read_parquet(root / "rows.parquet", columns=["track", "session", "t", "v", "delta", "A_path", "A_req"])
        Rx["A_d"] = Rx.v ** 2 * np.abs(np.tan(Rx.delta)) / 2.85
        clean = set(E.session[E.clean])
        Rc = Rx[Rx.session.isin(clean) & (Rx.v >= 1)].copy()
        Rc["b"] = np.where(Rc.v < 5, "low", np.where(Rc.v < 15, "mid", "high"))
        th = {k: (np.nanpercentile(g.A_path, 99), np.nanpercentile(g.A_d, 99)) for k, g in Rc.groupby(["track", "b"])}
        out = []
        for e in ER.itertuples():
            r = Rx[(Rx.session == e.session) & (Rx.t >= e.t_ev - 3e6) & (Rx.t <= e.t_ev)].reset_index(drop=True)
            b = np.where(r.v < 5, "low", np.where(r.v < 15, "mid", "high"))
            tp = np.array([th[(e.track, q)][0] for q in b]); td = np.array([th[(e.track, q)][1] for q in b])
            ip, idd = np.flatnonzero(r.A_path > tp), np.flatnonzero(r.A_d > td)
            op, od = (r.t[ip[0]] if len(ip) else None), (r.t[idd[0]] if len(idd) else None)
            if op is not None and (od is None or op <= od + 2e5):
                v = "plan_first"
            elif od is not None:
                v = "steer_first_amp" if r.A_d[idd[0]] > 1.5 * (r.A_req[idd[0]] if np.isfinite(r.A_req[idd[0]]) else np.inf) else "steer_first_ref"
            else:
                v = "quiet"
            out.append(dict(track=e.track, bucket=e.bucket, verdict_d=v))
        X = pd.DataFrame(out).groupby(["track", "bucket", "verdict_d"]).size().unstack(fill_value=0).reset_index()
        L += ["", "Exploratory (amendment 1): the same rule with A_delta (steering state) in place of A_cmd:", "", md(X)]

    # ---- high-speed large-steer episodes (all rollouts) and the human comparison
    rows_path = root / "rows.parquet"
    if rows_path.exists():
        R = pd.read_parquet(rows_path, columns=["track", "label", "session", "t", "v", "A_exec"])
        tev = E.set_index("session").t_ev
        R["t_ev"] = R.session.map(tev)
        R = R[R.t_ev.isna() | (R.t <= R.t_ev)]
        hs = []
        for (tr, sess), g in R.groupby(["track", "session"]):
            m = ((g.v >= 15) & (g.A_exec >= 3)).to_numpy()
            km = float((g.v[g.v >= 15] * 0.1).sum() / 1000)
            ep, tt = 0, []
            i = 0
            while i < len(m):
                if m[i]:
                    k = i
                    while k < len(m) and m[k]:
                        k += 1
                    if k - i >= 3:
                        ep += 1
                        tt.append(int(g.t.iloc[i]))
                    i = k
                else:
                    i += 1
            te = tev.get(sess)
            hs.append(dict(track=tr, session=sess, km_high=km, episodes=ep,
                           ep_to_zero=sum(1 for x in tt if te is not None and not pd.isna(te) and 0 <= te - x <= 3_000_000)))
        HS = pd.DataFrame(hs).groupby("track").sum(numeric_only=True)
        HS["per_100km"] = HS.episodes / HS.km_high * 100
        L += ["", "## High-speed large-steer episodes (v >= 15 m/s, |vx wz| >= 3 m/s^2 for >= 0.3 s, before any event)", "", md(HS.reset_index(), 2)]
        J["high_speed_episodes"] = HS.reset_index().to_dict("records")
    H = human_2hz(root)
    Hb = H[H.label == "YR10m10-F-s0"]
    L += ["", "nuPlan 2 Hz, human log vs ego (YR10m10-F-s0 visits; ego of each label): share of 0.5 s steps with a_lat >= 3 m/s^2", "",
          "| v bucket | human n | human share | ego (all six labels) n | ego share |", "|:--|--:|--:|--:|--:|"]
    for lo, hi, nm in ((0, 5, "low"), (5, 15, "mid"), (15, 99, "high")):
        h = Hb[(Hb.v_log >= lo) & (Hb.v_log < hi) & (Hb.v_log >= 1)]
        e = H[(H.v_e >= lo) & (H.v_e < hi) & (H.v_e >= 1)]
        L.append(f"| {nm} | {len(h)} | {pct((h.a_log >= 3).mean())} | {len(e)} | {pct((e.a_e >= 3).mean())} |")

    # ---- cap grid with human / navhard collateral and the selection rule
    nh = pd.read_csv(root / "navhard_ref.csv")
    nh = nh[nh.pdm_h1.notna() & nh.pdm_h2.notna() & nh.v0.notna()]
    a_nh = np.abs(np.radians(nh.pdm_h2 - nh.pdm_h1)) * nh.v0
    hm = Hb[Hb.v_log >= 1]
    CAP["coll_human"] = [float((hm.a_log >= am).mean()) for am in CAP.a_max]
    CAP["coll_navhard"] = [float((a_nh >= am).mean()) for am in CAP.a_max]
    CAP["eligible"] = (CAP.coll_clean <= 0.02) & (CAP.coll_human <= 0.01) & (CAP.coll_navhard <= 0.01)
    L += ["", "## Cap grid (section 3.7)", "", f"navhard tokens with a PDM reference: {len(nh)}; human 2 Hz steps (v >= 1): {len(hm)}", "",
          md(CAP[["track", "a_max", "j_max", "coll_clean", "coll_clean_low", "coll_clean_mid", "coll_clean_high", "coll_human", "coll_navhard", "eligible",
                  "events", "cover", "high_plan_first", "cover_high_plan_first"]], 3)]
    for tr in ("nuplan", "pai"):
        c = CAP[(CAP.track == tr) & CAP.eligible].copy()
        if not len(c):
            L.append(f"- {tr}: no eligible cap")
            continue
        c["jkey"] = c.j_max.fillna(99)
        best = c.sort_values(["cover_high_plan_first", "a_max", "jkey"], ascending=[False, False, False]).iloc[0]
        frac = best.cover_high_plan_first / max(best.high_plan_first, 1)
        L.append(f"- {tr}: chosen a_max {best.a_max}, j_max {best.j_max}: covers {int(best.cover_high_plan_first)} / {int(best.high_plan_first)} "
                 f"high-speed plan-first events ({pct(frac)}){' -> below the 25 % line: exploratory arm only' if frac < 0.25 else ''}; all events "
                 f"{int(best.cover)} / {int(best.events)}")
    J["cap"] = CAP.to_dict("records")

    L += ["", "Slowness, stops, onsets, rear impacts, corridor positions: sl_long.md (amendment 1 replaced the first-pass log alignment)."]
    (res / "sl_report.md").write_text("\n".join(L) + "\n")
    json.dump(J, open(res / "sl_report.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
