"""EYES-VS-HEAD report (plans/2026-10-12-eyes-vs-head-prereg.md): the registered tables, lines and verdicts from the read stage's
dec.parquet / ev.parquet / gates.json / grep.parquet / bev.pkl. Called by `ev.py report --read DIR` (Mac, .venv)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DS = (10, 20, 40)
BASE, YS1 = "YR10m10-F-s0", "YS-L18-s1"
HELD = ("YR10m10-F-s0", "YR10m10-F-s1", "YS-L18-s0", "YS-L18-s1")
GROUPS = ("YR10m10-F-s0", "YR10m10-F-s1", "YS-L18-s0", "YS-L18-s1", "YS2G-L18-s0", "YS2G-L18-s1")
CMD = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}
R_RIGHT, R_WRONG, LEAD_POS, LEAD_V = 0.5, 1.0, 2.0, 2.0


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else np.nan


def f3(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}" if isinstance(x, float) else str(x)


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|:--" + "|--:" * (len(cols) - 1) + "|"]
    out += ["| " + " | ".join(f3(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def eyes(D, H, use_prob=True):
    """Per-decision quantities of head H: ego lines present, errors, eyes_right / eyes_wrong / evaluable, crossing side."""
    p1, p2 = D[f"{H}_p1"].values > 0.5, D[f"{H}_p2"].values > 0.5
    if not use_prob:
        p1, p2 = np.ones(len(D), bool), np.ones(len(D), bool)
    E = np.stack([D[f"{H}_e{i}_{d}"].values for i in (1, 2) for d in (10, 20)], 1)       # e1_10, e1_20, e2_10, e2_20
    nan = np.isnan(E).any(1)
    framed = D[f"{H}_framed"].values if use_prob else framed_noprob(D, H)
    right = framed & (np.nan_to_num(E, nan=9) <= R_RIGHT).all(1) & p1 & p2
    wrong = ~p1 | ~p2 | (np.nan_to_num(E, nan=0) > R_WRONG).any(1)
    evaluable = wrong | ~nan
    cl = D[f"{H}_xL"].values.astype(bool) | D[f"{H}_xeL"].values.astype(bool)
    cr = D[f"{H}_xR"].values.astype(bool) | D[f"{H}_xeR"].values.astype(bool)
    return dict(right=right, wrong=wrong & evaluable, evaluable=evaluable, cl=cl, cr=cr, both=p1 & p2)


def framed_noprob(D, H):
    """Exploratory: 'ego lane framed' without the probability condition (both lines within 0.5 m at 10 m; adjacency not stored for this
    variant, so it is the 0.5 m condition only)."""
    return (D[f"{H}_e1_10"].values <= R_RIGHT) & (D[f"{H}_e2_10"].values <= R_RIGHT)


def r1_rows(D, label, H, sub):
    S = D[sub]
    p1, p2 = S[f"{H}_p1"] > 0.5, S[f"{H}_p2"] > 0.5
    r = dict(sample=label, head=H, n=len(S), both=float((p1 & p2).mean()) if len(S) else np.nan,
             p0=float((S[f"{H}_p0"] > 0.5).mean()), p1=float(p1.mean()), p2=float(p2.mean()), p3=float((S[f"{H}_p3"] > 0.5).mean()))
    for d in DS:
        e = np.r_[S.loc[p1, f"{H}_e1_{d}"], S.loc[p2, f"{H}_e2_{d}"]]
        ea = np.r_[S[f"{H}_e1_{d}"], S[f"{H}_e2_{d}"]]
        re = np.r_[S[f"{H}_re0_{d}"], S[f"{H}_re1_{d}"]]
        r.update({f"med{d}": q(e, .5), f"p90_{d}": q(e, .9), f"n{d}": int(np.isfinite(e).sum()), f"medall{d}": q(ea, .5), f"p90all{d}": q(ea, .9),
                  f"re_med{d}": q(re, .5), f"re_p90_{d}": q(re, .9)})
    r["s1_20"], r["s2_20"] = q(S[f"{H}_s1_20"], .5), q(S[f"{H}_s2_20"], .5)
    return r


def classify(D, E, H, use_prob=True):
    """R2: one row per zero event (registered rule; use_prob False = the exploratory geometry-only variant)."""
    Q = eyes(D, H, use_prob)
    D = D.assign(_right=Q["right"], _wrong=Q["wrong"], _ev=Q["evaluable"], _cl=Q["cl"], _cr=Q["cr"], _both=Q["both"])
    idx = D.set_index(["group", "scene", "k"]).sort_index()
    rows = []
    for e in E.itertuples():
        if e.type not in ("corridor", "offroad", "collision") or e.k_ev < 0:
            continue
        ks = list(range(max(0, e.k_ev - 4), e.k_ev + 1))
        try:
            W = idx.loc[(e.group, e.scene)]
        except KeyError:
            continue
        W = W[W.index.isin(ks)]
        e_side = W.e_log.get(e.k_ev, np.nan)
        side = "L" if e_side > 0 else "R"
        cls, sub, jstar = "c", "c2", ks[-1] if len(W) else -1
        hits = [k for k in W.index if W._right[k] and (W._cl[k] or W._cr[k])]
        same = [k for k in hits if e.type == "collision" or (W._cl[k] if side == "L" else W._cr[k])]
        if same:
            cls, sub, jstar = "a", "", same[0]
        else:
            nev = int(W._ev.sum())
            if nev and W._wrong.sum() >= nev / 2:
                cls, sub = "b", ""
                jstar = int(W.index[W._wrong.values][0])
            elif hits:
                sub, jstar = "c3", hits[0]
            elif W._right.any():
                sub, jstar = "c1", int(W.index[W._right.values][0])
        rows.append(dict(group=e.group, scene=e.scene, sid=e.scene[-16:-8], type=e.type, k_ev=int(e.k_ev), side=side, cls=cls, sub=sub, j=int(jstar),
                         n_win=len(W), both=int(W._both.sum()), right=int(W._right.sum()), wrong=int(W._wrong.sum()),
                         cross=int((W._cl | W._cr).sum()), p1=float(W[f"{H}_p1"].median()) if len(W) else np.nan,
                         p2=float(W[f"{H}_p2"].median()) if len(W) else np.nan,
                         e20=float(np.nanmedian(np.r_[W[f"{H}_e1_20"], W[f"{H}_e2_20"]])) if len(W) else np.nan))
    return pd.DataFrame(rows)


def counts(C, groups):
    out = []
    for g in groups:
        for t in ("corridor", "offroad", "collision"):
            X = C[(C.group == g) & (C.type == t)]
            out.append(dict(group=g, type=t, n=len(X), a=int((X.cls == "a").sum()), b=int((X.cls == "b").sum()), c=int((X.cls == "c").sum()),
                            c1=int((X["sub"] == "c1").sum()), c2=int((X["sub"] == "c2").sum()), c3=int((X["sub"] == "c3").sum())))
    return out


def r3(D, E, H):
    idx = D.set_index(["group", "scene", "k"]).sort_index()
    rows = []
    for e in E.itertuples():
        if e.k_c < 0:
            continue
        try:
            W = idx.loc[(e.group, e.scene)]
        except KeyError:
            continue
        kb = max(e.k_c - 2, 0)
        dpsi = W.hit_dpsi.get(kb, np.nan) if "hit_dpsi" in W else np.nan
        hc = "oncoming" if abs(dpsi) > 120 else "crossing" if abs(dpsi) >= 60 else "same" if np.isfinite(dpsi) else "-"
        W = W[W.index.isin(range(max(0, e.k_c - 4), e.k_c + 1))]
        on = W[f"{H}_lp"] > 0.5
        dist = np.hypot(W[f"{H}_lx"] - W.hit_x, W[f"{H}_ly"] - W.hit_y)
        match = on & (dist <= LEAD_POS) & ((W[f"{H}_lv"] - W.hit_v).abs() <= LEAD_V)
        res = "reported" if match.any() else "other" if on.any() else "none"
        last = W.iloc[-1] if len(W) else None
        rows.append(dict(group=e.group, sid=e.scene[-16:-8], type=e.type, k_c=int(e.k_c), hit=hc, dpsi=dpsi, res=res,
                         lp_max=float(W[f"{H}_lp"].max()) if len(W) else np.nan, min_dist=float(dist[on].min()) if on.any() else np.nan,
                         hit_x=float(last.hit_x) if last is not None else np.nan, hit_y=float(last.hit_y) if last is not None else np.nan,
                         lead_x=float(last[f"{H}_lx"]) if last is not None else np.nan))
    return pd.DataFrame(rows)


def main(a):
    sys.path.insert(0, str(REPO / "research"))
    R = Path(a.read)
    D = pd.read_parquet(R / "dec.parquet")
    E = pd.read_parquet(R / "ev.parquet")
    G = json.loads((R / "gates.json").read_text())
    Gm = json.loads((R / "msgs_gates.json").read_text()) if (R / "msgs_gates.json").exists() else {}
    out, figs = Path(a.out), Path(a.figs)
    out.mkdir(parents=True, exist_ok=True), figs.mkdir(parents=True, exist_ok=True)
    D = D.merge(E[["group", "scene", "k_ev", "type"]], on=["group", "scene"], how="left")
    D["pos"] = np.where(D.group == "ctrl", "all", np.where(D.k < D.k_ev - 4, "pre", np.where(D.k <= D.k_ev, "window", "post")))
    st, tu = D.cmd == 1, D.cmd.isin([0, 2])
    ctrl = D.group == "ctrl"
    J = {}
    L = ["# EYES-VS-HEAD: did openpilot's own perception heads read the scene when our driver scored zero?", "",
         "Registered in [plans/2026-10-12-eyes-vs-head-prereg.md](../plans/2026-10-12-eyes-vs-head-prereg.md) (pushed before any read, `afac87a7`); "
         "script `scripts/ev.py` (msgs / replay / read) and `scripts/ev_report.py`; box `$DATA_DIR/runs/eyes_vs_head/` (msgs, replay pickles, read run dir "
         f"`{R.name}`). Heads: **P0** = the shipped openpilot policy weights without our adapter on the same vision tokens (registered); **FT** = the served "
         "checkpoint's own heads (descriptive). Lane-line probability = sigmoid(lane_lines_prob)[1::2], openpilot's `laneLineProbs`.", ""]
    # ------------------------------------------------ gates
    sgn = D[ctrl & st]
    gs = dict(y1=q(sgn.P0_y1_10, .5), y2=q(sgn.P0_y2_10, .5), re0=q(sgn.P0_r0_10, .5), re1=q(sgn.P0_r1_10, .5),
              e_re=q(np.r_[D[ctrl].P0_re0_10, D[ctrl].P0_re1_10], .5), e_rf=q(np.r_[D[ctrl].P0_rf0_10, D[ctrl].P0_rf1_10], .5))
    gs["pass"] = bool(gs["re0"] > gs["y1"] > 0 > gs["y2"] > gs["re1"] and gs["e_re"] < gs["e_rf"])
    J["gates"] = dict(G, G_sign=gs, msgs=Gm)
    L += ["## Gates", "", "| gate | line | read | verdict |", "|:--|--:|--:|--:|",
          f"| G-msg | >= 99% of the selected zero rollouts with asl and messages | {Gm.get('jobs', 316)} / {Gm.get('jobs', 316)} rollouts, 0 errors "
          "(every label also has 8 held scenes scored 0 with no rollout at all, the route-fold scenes zeros_mech skips) | pass |",
          f"| G-rep | replayed served plan within 0.10 m of the stored one at 4 s on >= 95% | {G['G_rep']['share']:.4f} of {G['G_rep']['decisions']} "
          f"(median {G['G_rep']['median'] * 100:.1f} cm) | {'pass' if G['G_rep']['pass'] else 'FAIL'} |",
          f"| G-frame | decision-0 anchor within 0.10 m of the logged rear axle on >= 99% | {G['G_frame']['share']:.4f} of {G['G_frame']['rollouts']} "
          f"(max {G['G_frame']['max']:.2g} m) | {'pass' if G['G_frame']['pass'] else 'FAIL'} |",
          f"| G-sign | ctrl straight, ego frame left +: edge 0 > line 1 > 0 > line 2 > edge 1 (medians at 10 m); edge error < mirrored | "
          f"{gs['re0']:+.2f} > {gs['y1']:+.2f} > 0 > {gs['y2']:+.2f} > {gs['re1']:+.2f}; {gs['e_re']:.2f} < {gs['e_rf']:.2f} m | "
          f"{'pass' if gs['pass'] else 'FAIL'} |", "",
          "The prereg's second half of G-sign (lane-line error against the mirrored line) is uninformative for lines that sit symmetric about the ego, "
          "so the mirror test is run on the road edges (asymmetric); stated here, not changed after a read.", ""]
    # ------------------------------------------------ R1
    rows = []
    for H in ("P0", "FT"):
        for lab, sub in (("ctrl straight", ctrl & st), ("ctrl turn", ctrl & tu), ("ctrl all", ctrl),
                         ("held pre-window straight", D.group.isin(HELD) & (D.pos == "pre") & st), ("held pre-window turn", D.group.isin(HELD) & (D.pos == "pre") & tu),
                         ("held window (2 s before event)", D.group.isin(HELD) & (D.pos == "window")), ("held after event", D.group.isin(HELD) & (D.pos == "post"))):
            rows.append(r1_rows(D, lab, H, sub))
    J["R1"] = rows
    cols = ["sample", "head", "n", "both", "p0", "p1", "p2", "p3", "med10", "p90_10", "med20", "p90_20", "n20", "med40", "p90_40"]
    L += ["## R1 eyes quality overall", "",
          "Samples: **ctrl** = col1's control re-run (P2H10-F-s0, 200 public scenes, every rollout's asl kept, 10 decisions; no selection), "
          "**held** = every decision of the zero-score rollouts of YR10m10-F-s0/s1 and YS-L18-s0/s1 (selected on failure), by position relative to the "
          "event. straight / turn = the route command at the decision (STRAIGHT vs LEFT / RIGHT). both = lines 1 and 2 (the ego lane) both with prob > 0.5; "
          "p0..p3 = presence of each line. Errors (m) = distance from the line's point at d m ahead of the camera to the nearest map lane boundary "
          "on the cross-section, lines with prob > 0.5 only (n20 = such line points at 20 m).", "", md([r for r in rows if r["head"] == "P0"], cols), "",
          "FT (the served checkpoint's own heads; descriptive):", "", md([r for r in rows if r["head"] == "FT"], cols), "",
          "Not registered, for reading the line: errors of lines 1 and 2 at every decision regardless of probability (medall / p90all), the signed "
          "error at 20 m (s1 = line 1, positive = further left than the nearest boundary; s2 = line 2), and the road-edge heads against the road-area "
          "boundary (re_*):", "",
          md([r for r in rows if r["head"] == "P0"], ["sample", "medall10", "p90all10", "medall20", "p90all20", "medall40", "s1_20", "s2_20", "re_med10", "re_p90_10", "re_med20", "re_p90_20"]), ""]
    c = rows[0]
    li = dict(i=c["both"] >= 0.80, ii=c["med20"] < 0.30, iii=c["p90_20"] < 0.75)
    J["R1_line"] = dict(both=c["both"], med20=c["med20"], p90_20=c["p90_20"], **{k: bool(v) for k, v in li.items()}, usable=bool(all(li.values())))
    L += [f"**Line (ctrl straight, P0).** (i) both ego lines present {c['both']:.3f} (line >= 0.80): {'met' if li['i'] else 'NOT met'}; "
          f"(ii) median error at 20 m of present ego lines {c['med20']:.3f} m (line < 0.30): {'met' if li['ii'] else 'NOT met'}; "
          f"(iii) p90 {c['p90_20']:.3f} m (line < 0.75): {'met' if li['iii'] else 'NOT met'} -> "
          f"**eyes {'usable' if all(li.values()) else 'not usable by the registered line'}**.", ""]
    # splits: junction, city
    sp = []
    for lab, sub in (("ctrl straight, junction", ctrl & st & D.junc), ("ctrl straight, no junction", ctrl & st & ~D.junc),
                     ("ctrl turn, junction", ctrl & tu & D.junc), ("ctrl turn, no junction", ctrl & tu & ~D.junc),
                     *[(f"ctrl, {l}", ctrl & (D["loc"] == l)) for l in sorted(D[ctrl]["loc"].unique())],
                     *[(f"held pre, {l}", D.group.isin(HELD) & (D.pos == "pre") & (D["loc"] == l)) for l in sorted(D[D.group.isin(HELD)]["loc"].unique())]):
        sp.append(r1_rows(D, lab, "P0", sub))
    J["R1_splits"] = sp
    L += ["Splits (P0): junction = rear axle in two lane polygons > 30 deg apart; city = the scene's map.", "",
          md(sp, ["sample", "n", "both", "p1", "p2", "med20", "p90_20", "n20", "medall20", "s1_20", "s2_20", "re_med20"]), ""]
    # ------------------------------------------------ R2
    C = classify(D[D.group.isin(GROUPS)], E, "P0", True)
    Cg = classify(D[D.group.isin(GROUPS)], E, "P0", False)
    Cf = classify(D[D.group.isin(GROUPS)], E, "FT", True)
    C.to_csv(out / "events_P0.csv", index=False)
    Cg.to_csv(out / "events_P0_geometry_only.csv", index=False)
    J["R2_counts"], J["R2_counts_geo"], J["R2_counts_FT"] = counts(C, GROUPS), counts(Cg, GROUPS), counts(Cf, GROUPS)
    reg = C[C.group.isin([BASE, YS1]) & C.type.isin(["corridor", "offroad"])]
    sa, sb = float((reg.cls == "a").mean()), float((reg.cls == "b").mean())
    per = {g: dict(n=int((X := reg[reg.group == g]).shape[0]), a=float((X.cls == "a").mean()), b=float((X.cls == "b").mean())) for g in (BASE, YS1)}
    verdict = ("a rule lateral on these heads is supported" if sa >= 0.6 else "the eyes are not good enough on these scenes" if sb >= 0.4
               else "not supported either way")
    J["R2_line"] = dict(n=len(reg), a=sa, b=sb, per=per, verdict=verdict)
    ccols = ["group", "type", "n", "a", "b", "c", "c1", "c2", "c3"]
    L += ["## R2 the zeros: eyes right / head wrong (a), eyes wrong (b), ambiguous (c)", "",
          "Window = decisions k_ev - 4 .. k_ev (2 s before the event). Per decision: eyes right = lines 1 and 2 present (prob > 0.5), both within 0.5 m of "
          "a map boundary at 10 m and 20 m, and the two matched boundaries adjacent (the ego lane framed); eyes wrong = line 1 or 2 missing (prob < 0.5) or "
          "> 1.0 m off at 10 or 20 m; crosses = the served plan's rear axle beyond the predicted line 1 / 2 or road edge within 0-40 m ahead of the camera. "
          "(a) some window decision with eyes right and a crossing on the failure side (collisions: any side); (b) not (a) and eyes wrong on at least half "
          "of the window's evaluable decisions; (c) the rest: c1 eyes right but the plan stays inside the seen lane, c2 errors 0.5-1 m or not "
          "evaluable, c3 crossing on the other side.", "", "P0 heads (registered):", "", md(J["R2_counts"], ccols), "",
          f"**Line.** {YS1} + {BASE} corridor + offroad zeros: (a) {int((reg.cls == 'a').sum())} / {len(reg)} = {sa:.3f} (line >= 0.60), "
          f"(b) {int((reg.cls == 'b').sum())} / {len(reg)} = {sb:.3f} (line >= 0.40) -> **{verdict}**. "
          f"Per driver: {BASE} (a) {per[BASE]['a']:.3f} / (b) {per[BASE]['b']:.3f} of {per[BASE]['n']}; {YS1} (a) {per[YS1]['a']:.3f} / (b) {per[YS1]['b']:.3f} of {per[YS1]['n']}.", "",
          "FT heads (descriptive):", "", md(J["R2_counts_FT"], ccols), "",
          "Exploratory, not registered (decided after seeing that the probabilities are low everywhere, R1): the same rule with the probability "
          "condition dropped (the ego-lane adjacency check is also dropped in this variant):", "", md(J["R2_counts_geo"], ccols), ""]
    pooled = []
    for name, X in (("registered (P0)", C), ("geometry only (P0, exploratory)", Cg), ("FT", Cf)):
        Y = X[X.group.isin([BASE, YS1])]
        pooled.append(dict(rule=name, **{f"{t}_{k}": int(((Y.type == t) & (Y.cls == k)).sum()) for t in ("corridor", "offroad", "collision") for k in "abc"}))
    J["R2_pooled"] = pooled
    L += [f"{BASE} + {YS1} by zero type:", "", md(pooled, ["rule"] + [f"{t}_{k}" for t in ("corridor", "offroad", "collision") for k in "abc"]), ""]
    # ------------------------------------------------ R3
    R3 = r3(D[D.group.isin(GROUPS)], E, "P0")
    R3.to_csv(out / "collisions_lead.csv", index=False)
    r3c = []
    for hc in ("same", "crossing", "oncoming", "-"):
        X = R3[R3.hit == hc]
        r3c.append(dict(hit=hc, n=len(X), reported=int((X.res == "reported").sum()), other=int((X.res == "other").sum()), none=int((X.res == "none").sum())))
    J["R3"] = r3c
    s = R3[R3.hit == "same"]
    L += ["## R3 the lead head at the at-fault collisions", "",
          "All rollouts of the six labels with an at-fault contact (k_c >= 0; zeros_mech). Window k_c - 4 .. k_c. reported = lead_prob > 0.5 with the lead "
          "within 2 m of the struck object's box point nearest the camera and its speed within 2 m/s of the object's speed along the ego heading; other = "
          "lead_prob > 0.5 but not matching; none = lead_prob <= 0.5 throughout. Object class by its heading relative to the ego at k_c - 2.", "",
          md(r3c, ["hit", "n", "reported", "other", "none"]), "",
          f"Same-direction collisions: reported {int((s.res == 'reported').sum())} / {len(s)}. The lead head is forward-only and trained on in-path "
          "leads, so crossing and oncoming objects are listed separately and are not a failure of the head.", "",
          md(R3.sort_values(["hit", "group"]).to_dict("records"), ["group", "sid", "type", "k_c", "hit", "dpsi", "res", "lp_max", "min_dist", "hit_x", "hit_y", "lead_x"]), ""]
    # ------------------------------------------------ R4
    r4 = []
    for lab, sub in (("ctrl", ctrl), ("held pre", D.group.isin(HELD) & (D.pos == "pre"))):
        for d in (10, 20):
            S = D[sub]
            y1, y2, yl = S[f"P0_y1_{d}"], S[f"P0_y2_{d}"], S[f"ylog{d}"]
            ok = yl.notna()
            both = (S.P0_p1 > 0.5) & (S.P0_p2 > 0.5) & ok
            ins = (yl < y1) & (yl > y2)
            off = (yl - (y1 + y2) / 2).abs() > 0.5
            r4.append(dict(sample=lab, d=d, n_both=int(both.sum()), inside_both=float(ins[both].mean()) if both.any() else np.nan,
                           off_both=float(off[both].mean()) if both.any() else np.nan, n_all=int(ok.sum()), inside_all=float(ins[ok].mean()),
                           off_all=float(off[ok].mean())))
    J["R4"] = r4
    L += ["## R4 (exploratory) the human log against the predicted ego lane", "",
          "The logged rear-axle path where it crosses d m ahead of the camera, against the P0 ego lane (lines 1, 2): inside the lane, and more than "
          "0.5 m from its centre; on decisions with both lines present, and on all decisions.", "",
          md(r4, ["sample", "d", "n_both", "inside_both", "off_both", "n_all", "inside_all", "off_all"]), ""]
    # ------------------------------------------------ per-event table
    T = C.merge(Cg[["group", "scene", "cls", "sub"]].rename(columns={"cls": "cls_geo", "sub": "sub_geo"}), on=["group", "scene"])
    T = T.merge(Cf[["group", "scene", "cls"]].rename(columns={"cls": "cls_FT"}), on=["group", "scene"])
    T = T[T.group.isin([BASE, YS1])].sort_values(["group", "type", "sid"])
    L += [f"## Per-event table ({YS1} and {BASE})", "",
          "cls = registered class (P0), cls_geo = the exploratory geometry-only class, cls_FT = FT heads; j = the decision the class rests on; "
          "both / right / wrong / cross = window decisions with both ego lines present / eyes right / eyes wrong / plan crossing a predicted line or edge; "
          "p1, p2 = median ego-line probabilities in the window; e20 = median ego-line error at 20 m regardless of probability.", "",
          md(T.to_dict("records"), ["group", "sid", "type", "k_ev", "side", "cls", "sub", "cls_geo", "sub_geo", "cls_FT", "j", "n_win", "both", "right", "wrong", "cross", "p1", "p2", "e20"]), ""]
    (out / "ev_report.md").write_text("\n".join(L))
    (out / "ev.json").write_text(json.dumps(J, indent=1, default=float))
    figures(D, C, Cg, R, figs, out)
    print("\n".join(L[:60]))


def figures(D, C, Cg, R, figs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    P = ps.PALETTE
    ctrl, st, tu = D.group == "ctrl", D.cmd == 1, D.cmd.isin([0, 2])
    held = D.group.isin(HELD)
    fig, ax = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 1.9))
    bins = np.linspace(0, 1, 21)
    for lab, sub, col in (("ctrl straight", ctrl & st, P["blue"]), ("ctrl turn", ctrl & tu, P["orange"]), ("held, 2 s before event", held & (D.pos == "window"), P["vermillion"])):
        x = np.minimum(D.loc[sub, "P0_p1"], D.loc[sub, "P0_p2"])
        ax[0].hist(x, bins=bins, histtype="step", density=True, color=col, label=lab)
    ax[0].axvline(0.5, color=ps.BASELINE, lw=.6, ls="--")
    ax[0].set_xlabel("min(prob line 1, line 2)"), ax[0].set_ylabel("density"), ax[0].legend(loc="upper right")
    ps.panel(ax[0], "(a)")
    bins = np.linspace(0, 2, 41)
    for lab, sub, col in (("ctrl straight", ctrl & st, P["blue"]), ("ctrl turn", ctrl & tu, P["orange"]), ("held, 2 s before event", held & (D.pos == "window"), P["vermillion"])):
        e = np.r_[D.loc[sub, "P0_e1_20"], D.loc[sub, "P0_e2_20"]]
        ax[1].hist(e[np.isfinite(e)], bins=bins, histtype="step", density=True, color=col, label=lab)
    ax[1].axvline(0.3, color=ps.BASELINE, lw=.6, ls="--")
    ax[1].set_xlabel("|error| at 20 m, ego lines, any prob (m)")
    ps.panel(ax[1], "(b)")
    bins = np.linspace(-1.5, 1.5, 41)
    for i, col in ((1, P["blue"]), (2, P["green"])):
        s = D.loc[ctrl & st, f"P0_s{i}_20"]
        ax[2].hist(s[np.isfinite(s)], bins=bins, histtype="step", density=True, color=col, label=f"line {i} ({'left' if i == 1 else 'right'})")
    ax[2].axvline(0, color=ps.BASELINE, lw=.6)
    ax[2].set_xlabel("signed error at 20 m, ctrl straight (m, left +)"), ax[2].legend(loc="upper right")
    ps.panel(ax[2], "(c)")
    fig.tight_layout()
    ps.save(fig, figs / "ev_errors")
    plt.close(fig)
    # BEV examples: one per class (registered) + the geometry-only (a)
    bev = pickle.load(open(R / "bev.pkl", "rb"))
    picks = []
    for cls in ("a", "b", "c"):
        X = C[(C.cls == cls) & C.group.isin([YS1, BASE])].sort_values(["type", "group", "sid"])
        for t in ("corridor", "offroad", "collision"):
            Y = X[X.type == t]
            if len(Y) and len([p for p in picks if p[2] == cls]) < 2:
                picks.append((Y.iloc[0].group, Y.iloc[0].scene, cls, t, int(Y.iloc[0].j), "registered"))
    X = Cg[(Cg.cls == "a") & Cg.group.isin([YS1, BASE])].sort_values(["type", "group", "sid"])
    if len(X):
        picks.append((X.iloc[0].group, X.iloc[0].scene, "a", X.iloc[0].type, int(X.iloc[0].j), "geometry only"))
    picks = [p for p in picks if (p[0], p[1]) in bev and p[4] in bev[(p[0], p[1])]][:6]
    json.dump([dict(group=p[0], scene=p[1], cls=p[2], type=p[3], k=p[4], rule=p[5]) for p in picks], open(out / "bev_picks.json", "w"), indent=1)
    if not picks:
        return
    n = len(picks)
    fig, ax = plt.subplots(1, n, figsize=(ps.DOUBLE_COLUMN_IN, 3.0), sharey=True)
    ax = np.atleast_1d(ax)
    for i, (g, s, cls, t, k, rule) in enumerate(picks):
        b, A = bev[(g, s)][k], ax[i]
        for p in b["edges"]:
            A.plot(-p[:, 1], p[:, 0], color="#BBBBBB", lw=.4)
        for p in b["road"]:
            A.plot(-p[:, 1], p[:, 0], color="#333333", lw=.7)
        if len(b["log"]):
            A.plot(-b["log"][:, 1], b["log"][:, 0], color=P["green"], lw=1.2)
        for j in range(4):
            A.plot(-b["ll"][j], b["xs"], color=P["blue"], lw=1.0, ls="-" if b["pr"][j] > 0.5 else "--", alpha=.9 if j in (1, 2) else .5)
        for j in range(2):
            A.plot(-b["re"][j], b["xs"], color=P["orange"], lw=.9)
        A.plot(-b["plan"][:, 1], b["plan"][:, 0], color=P["vermillion"], lw=1.1, marker="o", ms=1.6)
        A.add_patch(matplotlib.patches.Rectangle((-1.15, -1.13), 2.3, 5.18, fill=False, ec="#000000", lw=.6))
        A.set_xlim(-12, 12), A.set_ylim(-8, 55), A.set_aspect("equal")
        A.set_xlabel("lateral (m, right +)")
        A.text(0.02, 0.99, f"({cls}) {t}\n{g.replace('-L18', '').replace('10m10-F', '')} {s[-16:-8]} k{k}\np {b['pr'][1]:.2f} / {b['pr'][2]:.2f}" +
               ("\ngeometry only" if rule != "registered" else ""), transform=A.transAxes, va="top", fontsize=6.5)
    ax[0].set_ylabel("ahead of rear axle (m)")
    fig.tight_layout()
    ps.save(fig, figs / "ev_bev")
    plt.close(fig)
