#!/usr/bin/env python3
"""STEER-LONG longitudinal reads (research/next-round/steer-long.md, sections 3.6 with amendment 1, 3.13 amendment 3): speed ratio to the
log, stops, slowdown onsets and their triggers, rear impacts, the longitudinal position of corridor zeros. Mac, CPU, on sl_pull.py copies
plus zeros_mech's per_decision table (nuPlan log stations).

  sl_long.py [--root tmp/steer_long] [--out experiments/steer_long/results]

Log speed, position-aligned: nuPlan from per_decision (s_e = the ego's station on the logged path, s_log_t = the log's station at the same
time; the log reached station s at t'(s) = inverse of s_log_t, v_log = ds_log/dt there). PAI keeps no logged path for scored rollouts:
proxy s_log(t) = s_e(t) + sqrt(max(d_loc^2 - d_lat^2, 0)) with s_e = dist_traveled_m (ego assumed behind); checked on nuPlan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sl_read as S  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
S.SER = S.SER + ("dist_to_gt_location", "lateral_dist_to_gt_trajectory", "dist_to_gt_trajectory", "collision_rear", "collision_front",
                 "collision_lateral", "progress")


def pct(x):
    return f"{100 * x:.1f} %"


def mono(t, s):
    s = np.maximum.accumulate(np.asarray(s, float))
    return np.asarray(t, float), s


class LogSpeed:
    """Station of the log against time -> speed at a station (position-aligned)."""

    def __init__(self, t, s):
        t, s = mono(t, s)
        self.t, self.s = t, s
        self.v = np.gradient(s, t / 1e6) if len(t) > 2 else np.zeros_like(s)
        su, iu = np.unique(s, return_index=True)
        self.su, self.tu = su, t[iu]

    def t_at(self, s):
        return np.interp(s, self.su, self.tu)

    def v_at_station(self, s):
        return np.interp(self.t_at(s), self.t, self.v), np.asarray(s) > self.s[-1] - 1.0


def proxy_log(Mr):
    te, se = S.series(Mr, "dist_traveled_m")
    tl, dl = S.series(Mr, "dist_to_gt_location")
    tt, dt_ = S.series(Mr, "lateral_dist_to_gt_trajectory")
    if len(te) < 3 or len(tl) < 3:
        return None, None
    lat = np.interp(te, tt, dt_) if len(tt) else np.zeros_like(te)
    along = np.sqrt(np.maximum(np.interp(te, tl, np.nan_to_num(dl, posinf=0)) ** 2 - lat ** 2, 0))
    return (te, se), (te, se + along)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "tmp/steer_long"))
    ap.add_argument("--out", default=str(REPO / "experiments/steer_long/results"))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    PD = pd.read_parquet(root / "per_decision.parquet")
    PD = {k: g.sort_values("k") for k, g in PD.groupby(["label", "scene"])}
    SPD, STP, ONS, REAR, COR, PX = [], [], [], [], [], []
    for track, labs in S.LABELS.items():
        for lab in labs:
            D, C, M, Sm = S.load(track, lab, root)
            for rid, summ in Sm.items():
                if rid not in C or rid not in D:
                    continue
                scene = summ["clipgt_id"]
                pl, c, Mr = S.Plans(D[rid]), C[rid], M.get(rid, {})
                ct, vx = c.timestamp_us.to_numpy(np.int64), c.vx.to_numpy()
                m_ = summ.get("metrics") or {}
                score = float(summ.get("score") or 0)
                tf = S.first_flag_time(Mr)
                # ego station and log speed
                if track == "nuplan":
                    g = PD.get((lab, scene))
                    if g is None or len(g) < 3:
                        continue
                    tk = g.t_us.to_numpy(float)
                    LS = LogSpeed(tk, g.s_log_t.to_numpy())
                    s_e = lambda t, tk=tk, se=g.s_e.to_numpy(): np.interp(t, tk, se)          # noqa: E731
                    pe, pl_ = proxy_log(Mr)
                    if pe is not None:                                                        # proxy check against the truth
                        ok = (pl_[0] >= tk[0]) & (pl_[0] <= tk[-1])
                        if ok.any():
                            tru = np.interp(pl_[0][ok], tk, g.s_log_t.to_numpy()) - np.interp(pl_[0][ok], tk, g.s_e.to_numpy())
                            prx = pl_[1][ok] - pe[1][ok]
                            PX.append(pd.DataFrame(dict(label=lab, scene=scene, gap_true=tru, gap_proxy=prx)))
                else:
                    pe, pl_ = proxy_log(Mr)
                    if pe is None:
                        continue
                    LS = LogSpeed(*pl_)
                    s_e = lambda t, pe=pe: np.interp(t, pe[0], pe[1])                        # noqa: E731
                Pt_model = np.array([S.arc(P, 4) - S.arc(P, 2) for P in pl.model])            # model plan 1-2 s arc
                Pt_served = np.array([S.arc(P, 4) - S.arc(P, 2) for P in pl.poses])
                Pt_m01 = np.array([S.arc(P, 2) for P in pl.model])
                a2m = np.array([S.arc(P, 4) for P in pl.model])
                a2s = np.array([S.arc(P, 4) for P in pl.poses])
                src = [((f or {}).get("lead") or {}).get("src") for f in pl.fix]
                cut = np.array([float(((f or {}).get("lead") or {}).get("cut") or 0) for f in pl.fix])
                ld = np.array([((f or {}).get("lead") or {}).get("d") for f in pl.fix], dtype=object)
                ot, ov = S.series(Mr, "min_distance_to_obstacle_m")
                # ---- speed rows at 0.5 s steps (section 3.6 with amendment 1)
                tg = ct[::5]
                if tf is not None:
                    tg = tg[tg < tf]
                if len(tg):
                    se = s_e(tg)
                    vl, past = LS.v_at_station(se)
                    ve = np.interp(tg, ct, vx)
                    stopped = vx < 0.5
                    last = np.array([ct[: np.searchsorted(ct, x, side="right")][stopped[: np.searchsorted(ct, x, side="right")]].max(initial=ct[0]) for x in tg])
                    jp = np.maximum(np.searchsorted(pl.now, tg, side="right") - 1, 0)
                    SPD.append(pd.DataFrame(dict(track=track, label=lab, scene=scene, log=S.log_of(track, scene), t=tg, v_e=ve, v_log=vl, past=past,
                                                 d_e=se - se[0], since=(tg - last) / 1e6, v_plan=Pt_m01[jp])))
                # ---- stops (section 3.6, amendment 1 log speed)
                for s0, L in S.runs(vx < 1.0):
                    if L < 10 or (tf is not None and ct[s0] >= tf):
                        continue
                    ts = ct[s0]
                    vl0, past0 = LS.v_at_station(s_e(np.array([ts])))
                    obs = float(np.interp(ts, ot, np.nan_to_num(ov, posinf=1e3))) if len(ot) else np.nan
                    jj = max(np.searchsorted(pl.now, ts, side="right") - 1, 0)
                    cls = ("P" if a2m[jj] < 1 else "L" if track == "pai" and src[jj] == "mpc" and cut[jj] >= 0.5 else
                           "V" if track == "pai" and a2s[jj] < 1 else "M" if a2s[jj] >= 1 else "other")
                    STP.append(dict(track=track, label=lab, scene=scene, t=int(ts), dur=L / 10, v_log=float(vl0[0]), past=bool(past0[0]), obs=obs,
                                    at_start=bool(ts - ct[0] < 500_000), cls=cls,
                                    unexplained=bool(vl0[0] >= 2 and not past0[0] and obs >= 10)))
                # ---- slowdown onsets (amendment 3)
                i, n = 0, len(ct)
                while i < n:
                    if tf is not None and ct[i] >= tf:
                        break
                    if vx[i] >= 5:
                        w = (ct > ct[i]) & (ct <= ct[i] + 3_000_000)
                        if w.any():
                            jmin = np.flatnonzero(w)[np.argmin(vx[w])]
                            if vx[jmin] <= 0.7 * vx[i] and vx[i] - vx[jmin] >= 2 and (tf is None or ct[jmin] < tf):
                                k0 = i + int(np.argmax(vx[i:jmin + 1]))                        # the start of the drop
                                t_s, t_m = ct[k0], ct[jmin]
                                (vls, _), (vlm, pm) = LS.v_at_station(s_e(np.array([t_s]))), LS.v_at_station(s_e(np.array([t_m])))
                                log_ratio = float(vlm[0] / vls[0]) if vls[0] > 0.5 else np.nan
                                win = np.flatnonzero((pl.now >= t_s - 500_000) & (pl.now <= t_s + 1_500_000))
                                v0 = vx[k0]
                                Pf = bool((Pt_model[win] <= 0.8 * v0).any()) if len(win) else False
                                Lf = bool(any(src[j] == "mpc" and cut[j] >= 0.5 for j in win)) if track == "pai" else False
                                phantom = False
                                if Lf:
                                    for j in win:
                                        if src[j] == "mpc" and cut[j] >= 0.5 and ld[j] is not None:
                                            obs = float(np.interp(pl.now[j], ot, np.nan_to_num(ov, posinf=1e3))) if len(ot) else np.nan
                                            phantom |= bool(obs > float(ld[j]) + 10)
                                Vf = bool(track == "pai" and len(win) and ((Pt_served[win] < 0.9 * Pt_model[win]) & (cut[win] < 0.5)).any())
                                cls = "P" if Pf else "L" if Lf else "V" if Vf else "M"
                                rec = np.flatnonzero((ct > t_m) & (vx >= 0.9 * v0))
                                ONS.append(dict(track=track, label=lab, scene=scene, session=rid, t_s=int(t_s), t_min=int(t_m), v0=float(v0),
                                                vmin=float(vx[jmin]), log_ratio=log_ratio, log_past=bool(pm[0]), P=Pf, L=Lf, phantom=phantom, V=Vf, cls=cls,
                                                recover_s=float((ct[rec[0]] - t_m) / 1e6) if len(rec) else np.nan, zero=score == 0,
                                                t_flag=tf, t0_roll=int(ct[0])))
                                i = jmin + 10
                                continue
                    i += 1
                # ---- rear impacts and corridor zeros (amendment 3)
                tr_, vr_ = S.series(Mr, "collision_rear")
                if len(tr_) and (vr_ > 0).any():
                    t_r = int(tr_[vr_ > 0][0])
                    other = [S.series(Mr, n) for n in ("collision_front", "collision_lateral", "offroad", "left_corridor_laterally")]
                    first_other = min([int(t[v > 0][0]) for t, v in other if len(t) and (v > 0).any()], default=None)
                    vl, past = LS.v_at_station(s_e(np.array([t_r])))
                    REAR.append(dict(track=track, label=lab, scene=scene, t=t_r, first=first_other is None or t_r <= first_other,
                                     v_e=float(np.interp(t_r, ct, vx)), v_log=float(vl[0]), score=score,
                                     at_fault=float(m_.get("collision_at_fault") or 0) > 0))
                kind, t_ev = S.event_of(summ, Mr) if score == 0 else ("", None)
                if kind == "corridor":
                    tl, dl = S.series(Mr, "dist_to_gt_location")
                    tt, dt_ = S.series(Mr, "lateral_dist_to_gt_trajectory")
                    dloc = float(np.interp(t_ev, tl, dl))
                    lat = float(np.interp(t_ev, tt, dt_))
                    along = float(np.sqrt(max(dloc ** 2 - lat ** 2, 0)))
                    se_ev = float(s_e(np.array([t_ev]))[0])
                    COR.append(dict(track=track, label=lab, scene=scene, t=t_ev, d_loc=dloc, lat=lat, along=along,
                                    past_end=bool(se_ev > LS.s[-1] - 1.0) if track == "nuplan" else None,
                                    behind=bool(track == "nuplan" and se_ev < float(np.interp(t_ev, LS.t, LS.s)) - 4.0)))
            print(track, lab, flush=True)
    SPD, STP, ONS, REAR, COR = (pd.concat(SPD, ignore_index=True), pd.DataFrame(STP), pd.DataFrame(ONS), pd.DataFrame(REAR), pd.DataFrame(COR))
    PX = pd.concat(PX, ignore_index=True) if PX else pd.DataFrame()
    out.mkdir(parents=True, exist_ok=True)
    SPD.to_parquet(out / "speed_rows2.parquet", index=False)
    for nm, df in (("stops2", STP), ("onsets", ONS), ("rear", REAR), ("corridor_pos", COR)):
        df.to_csv(out / f"{nm}.csv", index=False)
    L = ["# STEER-LONG longitudinal read (sections 3.6 + amendment 1, 3.13 amendment 3)", ""]
    if len(PX):
        e = (PX.gap_proxy - PX.gap_true).abs()
        L += [f"PAI proxy check on nuPlan (along-track gap, proxy vs true): median |err| {e.median():.2f} m, p90 {e.quantile(.9):.2f} m, "
              f"sign right (true gap > 1 m) {pct((PX.gap_proxy[PX.gap_true > 1] > 0.5).mean())}, n {len(PX)}", ""]
    # speed ratio
    Sd = SPD[(SPD.v_log >= 2) & ~SPD.past].copy()
    Sd["ratio"] = Sd.v_e / Sd.v_log
    Sd["lr"] = np.log(np.clip(Sd.ratio, 0.05, None))
    Sd["d100"] = Sd.d_e / 100
    L += ["## Speed ratio v_ego / v_log (position-aligned, log moving >= 2 m/s, before any flag)", "",
          "| track / label | n | median | 0-10 | 10-25 | 25-50 | 50-100 | > 100 m | slope log ratio / 100 m [95 % CI] |", "|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for (tr, lab), g in Sd.groupby(["track", "label"]):
        med = g.groupby(pd.cut(g.d_e, [-1, 10, 25, 50, 100, 1e9]), observed=False).ratio.median().to_numpy()
        s, lo, hi = boot_slope(g, "d100", "lr", "log")
        L.append(f"| {tr} / {lab} | {len(g)} | {g.ratio.median():.2f} | " + " | ".join("-" if np.isnan(x) else f"{x:.2f}" for x in med)
                 + f" | {s:+.3f} [{lo:+.3f}, {hi:+.3f}]{' farther = slower' if hi < 0 else ''} |")
    L += ["", "By time since the last stop (or start): median ratio", "", "| track / label | 0-2 | 2-4 | 4-6 | 6-9 | 9-20 s |", "|:--|--:|--:|--:|--:|--:|"]
    for (tr, lab), g in Sd.groupby(["track", "label"]):
        med = g.groupby(pd.cut(g.since, [-1, 2, 4, 6, 9, 21]), observed=False).ratio.median().to_numpy()
        L.append(f"| {tr} / {lab} | " + " | ".join("-" if np.isnan(x) else f"{x:.2f}" for x in med) + " |")
    L += ["", "Decomposition (means, m/s; v_plan = model plan's first 1 s)", "", "| track / label | v_log | v_plan | v_ego | total | model part | execution part | model share |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for (tr, lab), g in Sd.groupby(["track", "label"]):
        tot, mp, ep = (g.v_log - g.v_e).mean(), (g.v_log - g.v_plan).mean(), (g.v_plan - g.v_e).mean()
        L.append(f"| {tr} / {lab} | {g.v_log.mean():.2f} | {g.v_plan.mean():.2f} | {g.v_e.mean():.2f} | {tot:+.2f} | {mp:+.2f} | {ep:+.2f} | {mp / tot:.2f} |")
    # stops
    L += ["", "## Stops (v < 1 m/s >= 1 s before any flag); unexplained = log moving >= 2 m/s here, nothing within 10 m, not past the log end", "",
          "| track / label | stops | at start | unexplained | P | L | V | M | other |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (tr, lab), g in STP.groupby(["track", "label"]):
        u = g[g.unexplained]
        L.append(f"| {tr} / {lab} | {len(g)} | {int(g.at_start.sum())} | {len(u)} | " + " | ".join(str(int((u.cls == c).sum())) for c in ("P", "L", "V", "M", "other")) + " |")
    # onsets
    L += ["", "## Slowdown onsets (v >= 5, drop to <= 0.7 v and >= 2 m/s within 3 s, before any flag)", "",
          "| track / label | onsets | per 100 rollouts | log did not slow (>= 0.85) | P | L (phantom) | V | M | P flag | L flag | median recover s |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    nroll = SPD.groupby(["track", "label"]).scene.nunique()
    for (tr, lab), g in ONS.groupby(["track", "label"]):
        u = g[g.log_ratio >= 0.85]
        L.append(f"| {tr} / {lab} | {len(g)} | {100 * len(g) / nroll[(tr, lab)]:.1f} | {len(u)} | {int((u.cls == 'P').sum())} | "
                 f"{int((u.cls == 'L').sum())} ({int((u.cls == 'L') .mul(u.phantom).sum())}) | {int((u.cls == 'V').sum())} | {int((u.cls == 'M').sum())} | "
                 f"{int(u.P.sum())} | {int(u.L.sum())} | {np.nanmedian(u.recover_s) if len(u) else np.nan:.1f} |")
    for tr, g in ONS[ONS.log_ratio >= 0.85].groupby("track"):
        c = g.cls.value_counts()
        L.append(f"- {tr}: unexplained onsets {len(g)}, largest class {c.idxmax()} {pct(c.max() / len(g))}" + (" -> >= 50 %" if c.max() / len(g) >= 0.5 else ""))
    L.append("")
    L.append(f"Onsets followed by the rollout's zero within 5 s: " + ", ".join(
        f"{tr} {int(((g.zero) & (g.t_flag.notna()) & (g.t_flag - g.t_s <= 5e6)).sum())} / {len(g)}" for tr, g in ONS.groupby("track")))
    # rear
    L += ["", "## Rear impacts (collision_rear; never at-fault by the scorer)", ""]
    if len(REAR):
        for tr, g in REAR.groupby("track"):
            f = g[g.first]
            L.append(f"- {tr}: rollouts with a rear impact {len(g)} (first event {len(f)}), ego speed median {f.v_e.median():.1f} m/s, "
                     f"v_ego / v_log median {(f.v_e / f.v_log.where(f.v_log > 0.5)).median():.2f}, ego < 1 m/s {int((f.v_e < 1).sum())}, mean score {f.score.mean():.3f}, "
                     f"at-fault flag also set {int(f.at_fault.sum())}")
    # corridor
    L += ["", "## Corridor zeros: where the ego was along the log at the event", ""]
    for tr, g in COR.groupby("track"):
        L.append(f"- {tr}: {len(g)} corridor zeros; lateral at event median {g.lat.median():.2f} m; time-aligned along-track gap median {g.along.median():.1f} m, "
                 f"> 4 m {int((g.along > 4).sum())}; " + (f"behind the log pose by > 4 m {int(g.behind.sum())}, past the log end {int(g.past_end.sum())}" if tr == "nuplan" else ""))
    (out / "sl_long.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def boot_slope(df, xcol, ycol, ccol, n=1000):
    rng = np.random.default_rng(0)
    G = [g[[xcol, ycol]].to_numpy() for _, g in df.groupby(ccol)]

    def slope(A):
        return float(np.polyfit(A[:, 0], A[:, 1], 1)[0])
    b = [slope(np.vstack([G[i] for i in rng.integers(0, len(G), len(G))])) for _ in range(n)]
    return slope(np.vstack(G)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


if __name__ == "__main__":
    main()
