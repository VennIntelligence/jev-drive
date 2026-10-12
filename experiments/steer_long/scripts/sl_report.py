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
    ST = pd.read_csv(res / "stops.csv")
    CAP = pd.read_csv(res / "cap_grid.csv")
    SL = pd.read_parquet(res / "speed_rows.parquet")
    L = ["# STEER-LONG read (pre-registration: research/next-round/steer-long.md section 3, commit 8f48d0c0)", ""]
    J = {}

    # ---- gates
    L += ["## Gates", "", "| track / label | summary rollouts | with controller + driver | G-frame | G-rec |", "|:--|--:|--:|--:|--:|"]
    for k, g in GT["gates"].items():
        L.append(f"| {k} | {g['summary']} | {g['with_ctrl']} | {g['frame_ok']} / {g['frame_pairs']} ({pct(g['frame_ok'] / max(g['frame_pairs'], 1))}) | "
                 f"{g['rec_ok']} / {g['rec_pairs']} ({pct(g['rec_ok'] / max(g['rec_pairs'], 1))}) |")
    zr = pd.read_parquet(root / "per_rollout.parquet")
    zr = zr[zr.has_rollout & (zr.type != "")][["label", "scene", "type"]]
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

    # ---- slowness (section 3.6)
    S = SL[(SL.v_log >= 2) & ~SL.past].copy()
    S["ratio"] = S.v_e / S.v_log
    S["lr"] = np.log(np.clip(S.ratio, 0.05, None))
    S["d100"] = S.d_e / 100
    L += ["", "## Slowness (section 3.6)", "", "Median v_ego / v_log (position-aligned), by distance travelled:", "",
          "| track / label | 0-10 | 10-25 | 25-50 | 50-100 | > 100 m | slope of log ratio per 100 m [95 % CI] |", "|:--|--:|--:|--:|--:|--:|:--|"]
    sl_out = []
    for (tr, lab), g in S.groupby(["track", "label"]):
        b = pd.cut(g.d_e, [-1, 10, 25, 50, 100, 1e9])
        med = g.groupby(b, observed=False).ratio.median().to_numpy()
        s, lo, hi = boot_slope(g, "d100", "lr", "log")
        sl_out.append(dict(track=tr, label=lab, slope=s, lo=lo, hi=hi))
        L.append(f"| {tr} / {lab} | " + " | ".join("-" if np.isnan(x) else f"{x:.2f}" for x in med) + f" | {s:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                 + (" farther = slower" if hi < 0 else "") + " |")
    L += ["", "By time since the last stop (or start):", "", "| track / label | 0-2 | 2-4 | 4-6 | 6-9 | 9-20 s |", "|:--|--:|--:|--:|--:|--:|"]
    for (tr, lab), g in S.groupby(["track", "label"]):
        b = pd.cut(g.since_stop, [-1, 2, 4, 6, 9, 21])
        med = g.groupby(b, observed=False).ratio.median().to_numpy()
        L.append(f"| {tr} / {lab} | " + " | ".join("-" if np.isnan(x) else f"{x:.2f}" for x in med) + " |")
    Sd = S[S.v_plan.notna()]
    L += ["", "Decomposition of the deficit (means over rows, m/s): v_log - v_ego = (v_log - v_plan) + (v_plan - v_ego), v_plan = model plan's first 1 s",
          "", "| track / label | n | v_log | v_plan | v_ego | total | model part | execution part | model share |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (tr, lab), g in Sd.groupby(["track", "label"]):
        tot = (g.v_log - g.v_e).mean()
        mp = (g.v_log - g.v_plan).mean()
        ep = (g.v_plan - g.v_e).mean()
        L.append(f"| {tr} / {lab} | {len(g)} | {g.v_log.mean():.2f} | {g.v_plan.mean():.2f} | {g.v_e.mean():.2f} | {tot:+.2f} | {mp:+.2f} | {ep:+.2f} | "
                 f"{mp / tot if tot else np.nan:.2f} |")
    J["slope"] = sl_out

    # ---- stops
    L += ["", "## Stops (v < 1 m/s for >= 1 s, before any flag)", ""]
    ST["unexplained"] = ST.unexplained.astype(bool)
    rows = []
    for (tr, lab), g in ST.groupby(["track", "label"]):
        n_roll = int(((E.track == tr) & (E.label == lab)).sum())
        u = g[g.unexplained]
        rows.append(dict(track=tr, label=lab, rollouts=n_roll, stops=len(g), unexplained=len(u), at_start=int(u.at_start.sum()),
                         P=int((u.cls == "P").sum()), L=int((u.cls == "L").sum()), V=int((u.cls == "V").sum()), M=int((u.cls == "M").sum()),
                         other=int((u.cls == "other").sum())))
    STb = pd.DataFrame(rows)
    L += [md(STb, 0), ""]
    for tr, g in STb.groupby("track"):
        u = g[["P", "L", "V", "M", "other"]].sum()
        n = u.sum()
        if n:
            top = u.idxmax()
            L.append(f"- {tr}: unexplained stops {n} ({n / g.rollouts.sum() * 100:.1f} per 100 rollouts), largest class {top} {pct(u[top] / n)}"
                     + (" -> >= 50 %" if u[top] / n >= 0.5 else ""))
    J["stops"] = STb.to_dict("records")
    (res / "sl_report.md").write_text("\n".join(L) + "\n")
    json.dump(J, open(res / "sl_report.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
