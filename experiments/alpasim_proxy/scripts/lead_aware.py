#!/usr/bin/env python3
"""LEAD-AWARE HOLD (offline, results/arena_hold/lead_aware.md): does JEV_HOLD=stop3 pay more if it latches only when no lead is present
and serves the model natively behind a lead? CPU only, stored reads only. Box, repo root, AlpaSim's venv ($P), a few minutes of CPU:

  $P lead_aware.py --arena $DATA_DIR/runs/alpasim_proxy/arena --jt $DATA_DIR/runs/defect_fix/jt/arena --out DIR

Inputs: the held windows of the hold reports (arena_resid.held: 2 228, route-fail dropped) with the set's ground-truth lead flag (wait_set.py
window_stats: a logged vehicle within 1.8 m of the human's heading line, rear ahead of the front bumper, gap <= 30 m, at the window start);
per-window rows of RSG-L18-s0 native / stop3 / lead (hold/report2), NR1F-L18-s0 native (ref_rows), JTN-L18-s0 off / stop3 (collected here
from the defect_fix jt reads); the per-call parquets of arena_resid.py extract (resid/calls_*.parquet: latch state, held plan arc); the
lead outputs of RSG's checkpoint per call (hold/reads/lead: JEV_HOLD=lead logs lead_p (radard-filtered probability), lead_d (gap from the
front), lead_v (speed) at every call, the handoff included).

Crediting. The latch is decided at the handoff call (0.5 s) from the measured speed, which is still the force-GT human's speed, so it is the
same window set for every model. A session that is not latched is the plain driver, i.e. the native read of that window. So "latch only
without a lead, else native" is credited exactly with the native read's score on the switched windows (up to closed-loop re-read noise,
measured on the free windows here). Release rules (latch, release later) are credited with the native score on release (an upper bound:
the late release is ignored). NR1F has no stop3 read: its latched windows are credited with RSG + stop3's score.

Rules (each under the oracle lead flag and the serve flag = RSG's lead head at the handoff call, p > 0.5 and gap <= 30 m):
  A   static latch with a lead -> native;  A' any latch (static or stop) with a lead -> native
  B   A, only when the native plan at the handoff call wants to go (2 s arc >= 0.5 / 1 / 2 m)
  C   A, latched until the held plan's 2 s arc >= 2 m (UB);  C' ... at a call where RSG's lead head also has the lead moving (UB, + grid)
  D   (RSG) the real JEV_HOLD=lead read (full hold, lead-head release) on lead-present static latches
  E   with a lead, a shorter static leash (2.0-2.5 m; diag.md's estimator), alone and with C'
Writes OUT/lead_aware.md (tables), OUT/windows.csv.gz (per window), OUT/rules.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
import arena as A  # noqa: E402
import arena_resid as RS  # noqa: E402

STRATA = ("static", "launch", "braking", "other", "turn", "cruise")


def rows_of(d):
    """Per-window score / zero type of one read directory (c*/run), arena.py collect's fields."""
    import pandas as pd
    o = []
    for f in glob.glob(f"{d}/c*/run/aggregate/results-summary.json"):
        for r in json.load(open(f))["rollouts"]:
            o.append(dict(id=r["clipgt_id"], score=float(r.get("score") or 0.0), type=A.failure_type(r),
                          dist=float((r.get("metrics") or {}).get("dist_traveled_m") or 0.0)))
    return pd.DataFrame(o).drop_duplicates("id").set_index("id")


def zc(t):
    return "AF-front" if t.startswith("AF-front") else "AF-side" if t.startswith("AF-") else "offroad" if "offroad" in t else \
        "corridor" if t == "corridor" else "zero-other"


def load(a):
    import pandas as pd
    AR = a.arena
    H, R, R0 = RS.held(AR)
    W = H[["log", "stratum", "lead", "lead_gap", "v0"]].copy()
    W["lead"] = W.lead.astype(str).str.lower().eq("true")
    W = W.join(pd.read_csv(f"{AR}/corridor/hand.csv.gz").set_index("id")[["gt", "arc_after_hand"]].rename(columns={"gt": "gt_h"}))
    W["still"] = W.index.map(pd.read_csv(f"{AR}/corridor/gtend_arena.csv.gz").set_index("id").seg_last_mm) < 1
    src = {"rsg_n": R0[R0.tag == "RSG-L18-s0"], "nr1f_n": R0[R0.tag == "NR1F-L18-s0"], "rsg_s3": R[R.tag == "stop3"], "rsg_lead": R[R.tag == "lead"]}
    for k, g in src.items():
        g = g.drop_duplicates("id").set_index("id")
        W[k], W[k + "_t"], W[k + "_d"] = g.score, g.type, g["dist"]
    for k, d in (("jtn_n", "JTN-L18-s0@off"), ("jtn_s3", "JTN-L18-s0@stop3")):
        g = rows_of(f"{a.jt}/{d}")
        W[k], W[k + "_t"], W[k + "_d"] = g.score, g.type, g["dist"]
    miss = W.filter(regex="_n$|_s3$|_lead$").isna().sum()
    assert miss.sum() == 0, f"missing rows {miss[miss > 0].to_dict()}"

    def latch(tag):
        C = pd.read_parquet(f"{AR}/resid/calls_{tag}.parquet")
        f = C[C.state.isin(["held", "free"])].sort_values("t0").drop_duplicates("id").set_index("id").reindex(W.index)
        return pd.Series(np.where(f.state == "held", np.where(f.stop_a.notna(), "stop", "static"), "free"), index=W.index), C
    W["latch"], Cr = latch("stop3")
    W["latch_jtn"], Cj = latch("JTN_stop3")
    # held plan arc (2 s) per call from 1 s, for the release rules
    held_calls = {}
    for k, C in (("rsg", Cr), ("jtn", Cj)):
        h = C[(C.state == "held") & (C.t_s >= 1)]
        W[f"{k}_harc"] = h.groupby("id").arc_model.max()
        held_calls[k] = h[["id", "t_s", "arc_model"]]
    # the handoff call's plan (k = 1, 0.5 s) of each native read: the ego is still on the human's pose there
    for k, tag in (("rsg", "RSG"), ("nr1f", "NR1F"), ("jtn", "JTN_off")):
        C = pd.read_parquet(f"{AR}/resid/calls_{tag}.parquet", columns=["id", "k", "arc2"])
        W[f"{k}_hand_arc"] = C[C.k == 1].drop_duplicates("id").set_index("id").arc2
    # RSG's lead outputs (JEV_HOLD=lead read): at the handoff call and, for release, per call from 1 s
    L = pd.DataFrame([json.loads(x) for f in glob.glob(f"{AR}/hold/reads/lead/c*/run/driver-logs/hold.jsonl") for x in open(f)])
    L = L[L.scene.isin(W.index)].drop(columns=["poses"], errors="ignore")
    hd = L[np.isclose(L.t_s, 0.5)].drop_duplicates("scene").set_index("scene")
    W["sl_p"], W["sl_d"], W["sl_v"] = hd.lead_p, hd.lead_d, hd.lead_v
    W["serve"] = (W.sl_p > 0.5) & (W.sl_d <= 30)
    W["serve20"] = (W.sl_p > 0.5) & (W.sl_d <= 20)
    # release table: each model's held calls from 1 s (its own held plan) joined with RSG's lead outputs at the same call time
    late = L[L.t_s >= 1][["scene", "t_s", "lead_p", "lead_v"]].rename(columns={"scene": "id"})
    late["t_s"] = late.t_s.round(2)
    for k, h in held_calls.items():
        h = h.assign(t_s=h.t_s.round(2)).merge(late, on=["id", "t_s"], how="left")
        W.attrs[k] = h
        W[f"{k}_rel"] = h[(h.arc_model >= 2) & (h.lead_p > 0.5) & (h.lead_v > 1.0)].groupby("id").size().reindex(W.index).fillna(0) > 0
    return W


MODELS = {  # name: (native score col, own stop3 col or None, latch col, held plan arc col or None)
    "RSG-L18-s0": ("rsg_n", "rsg_s3", "latch", "rsg"),
    "JTN-L18-s0": ("jtn_n", "jtn_s3", "latch_jtn", "jtn"),
    "NR1F-L18-s0": ("nr1f_n", None, "latch", None),
}


def base(W, m):
    """The model with stop3: its own stop3 read, or (NR1F) its native read on free windows and RSG + stop3's on latched ones."""
    n, s3, lc, _ = MODELS[m]
    if s3:
        return W[s3].values, W[s3 + "_t"].values
    lat = (W[lc] != "free").values
    return np.where(lat, W.rsg_s3, W[n]), np.where(lat, W.rsg_s3_t, W[n + "_t"])


def rules(W, m):
    """{rule name: boolean mask of windows served natively (instead of the model's stop3)}; flags: oracle (ground-truth lead) / serve
    (RSG's lead head at the handoff call: filtered probability > 0.5 and gap <= 30 m)."""
    n, s3, lc, harc = MODELS[m]
    hand_arc = W[{"RSG-L18-s0": "rsg", "JTN-L18-s0": "jtn", "NR1F-L18-s0": "nr1f"}[m] + "_hand_arc"]
    out = {}
    for fl, F in (("oracle", W.lead), ("serve", W.serve)):
        st, any_ = (W[lc] == "static") & F, (W[lc] != "free") & F
        out[f"A static latch, lead -> native [{fl}]"] = st
        out[f"A' any latch (static + stop), lead -> native [{fl}]"] = any_
        for x in (0.5, 1.0, 2.0):
            out[f"B static latch, lead and handoff plan 2 s arc >= {x:g} m -> native [{fl}]"] = st & (hand_arc >= x)
        if harc:
            out[f"C static latch, lead: release when the held plan's 2 s arc >= 2 m (UB) [{fl}]"] = st & (W[harc + "_harc"] >= 2)
            out[f"C' ... at a call where also RSG's lead head has the lead present and moving > 1 m/s (UB) [{fl}]"] = st & W[harc + "_rel"]
    out["A static latch, lead -> native [serve, gap <= 20 m]"] = (W[lc] == "static") & W.serve20
    return out


def summary_row(W, m, name, x, xt, b, bt, r, rt, grp):
    from jevdrive import stats
    d, inc = stats.paired(x, r, groups=grp), stats.paired(x, b, groups=grp)
    z, zr = x == 0, r == 0
    af = np.array([zc(t) == "AF-front" for t in xt]) & z
    return dict(model=m, rule=name, sum_vs=float((x - r).sum()), d=d, inc_sum=float((x - b).sum()), inc=inc, new=int((z & ~zr).sum()),
                fixed=int((~z & zr).sum()), zeros=int(z.sum()), af=int(af.sum()), corr=int(sum(zc(t) == "corridor" for t in xt[z])))


def main(a):
    import pandas as pd
    from jevdrive import stats
    W = load(a)
    N, grp = len(W), W.log.values
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r, rt = W.rsg_s3.values, W.rsg_s3_t.values
    o = ["# LEAD-AWARE HOLD tables (lead_aware.py)", "", f"Held windows {N}; reference RSG-L18-s0 + stop3 {r.mean():.4f}. "
         "Paired differences: mean per held window [95 % CI, bootstrap clustered by log, jevdrive.stats].", ""]
    # checks
    lm = (W.latch != W.latch_jtn).sum()
    fr = W.latch == "free"
    o += ["## Checks", "",
          f"- Latch class (RSG stop3 calls): {W.latch.value_counts().to_dict()}; JTN stop3's own latch differs on {lm} windows.",
          f"- Re-read noise (free windows: the stop3 read is the plain driver there): RSG native vs stop3 differ on "
          f"{int((fr & (W.rsg_n != W.rsg_s3)).sum())} of {int(fr.sum())} free windows, summed {(W.rsg_n - W.rsg_s3)[fr].sum():+.2f}; JTN off vs stop3 "
          f"{int(((W.latch_jtn == 'free') & (W.jtn_n != W.jtn_s3)).sum())}, {(W.jtn_n - W.jtn_s3)[W.latch_jtn == 'free'].sum():+.2f}."]
    S = W[W.latch == "static"]
    o += [f"- Serve-time flag against the ground truth on the {len(S)} static-latched windows: oracle lead {int(S.lead.sum())}, serve {int(S.serve.sum())}, "
          f"both {int((S.lead & S.serve).sum())}, oracle only {int((S.lead & ~S.serve).sum())}, serve only {int((~S.lead & S.serve).sum())}; "
          f"stop-latched: oracle {int(W[W.latch == 'stop'].lead.sum())} / {int((W.latch == 'stop').sum())}, serve {int(W[W.latch == 'stop'].serve.sum())}. "
          f"RSG lead head at the handoff on oracle-lead static windows: gap median {S[S.lead].sl_d.median():.1f} m (logged {S[S.lead].lead_gap.median():.1f} m), "
          f"p median {S[S.lead].sl_p.median():.2f}.",
          f"- The {int((~S.lead & S.serve).sum())} serve-only static latches (head: a lead within 30 m, log: none in the lane within 30 m at the "
          f"window start): logged lead gap {S[~S.lead & S.serve].lead_gap.notna().sum()} with any gap, still end {int(S[~S.lead & S.serve].still.sum())}, "
          f"head gap median {S[~S.lead & S.serve].sl_d.median():.1f} m; RSG native - stop3 there {(S.rsg_n - S.rsg_s3)[~S.lead & S.serve].sum():+.1f}.", ""]
    # where the switched windows sit: static latched, by class and lead
    S = W[W.latch == "static"].copy()
    S["cls"] = np.select([S.stratum.eq("launch") & (S.gt_h >= 5), S.stratum.eq("launch"), S.still], ["launch, human >= 5 m", "launch, human < 5 m",
                         "still end"], "moving end (static)")
    o += ["## Static-latched windows by lead (oracle): native vs stop3 means (zeros)", "",
          "| lead | class | windows | RSG native | RSG stop3 | JTN off | JTN stop3 | NR1F native | NR1F - RSG stop3 (sum) | JTN off - JTN stop3 (sum) | RSG n - s3 (sum) |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for (ld, c), g in S.groupby(["lead", "cls"]):
        f = lambda k: f"{g[k].mean():.3f} ({int((g[k] == 0).sum())})"  # noqa: E731
        o.append(f"| {ld} | {c} | {len(g)} | {f('rsg_n')} | {f('rsg_s3')} | {f('jtn_n')} | {f('jtn_s3')} | {f('nr1f_n')} | "
                 f"{(g.nr1f_n - g.rsg_s3).sum():+.1f} | {(g.jtn_n - g.jtn_s3).sum():+.1f} | {(g.rsg_n - g.rsg_s3).sum():+.1f} |")
    # lead-present still-end: the zeros native serving brings
    o += ["", "Lead-present (oracle) still-end static-latched windows served natively, zero types (stop3 zeros in brackets):", ""]
    g = S[S.lead & S.cls.eq("still end")]
    for m, (n, s3, _, _) in MODELS.items():
        zt = pd.Series([zc(t) for t in g[n + "_t"][g[n] == 0]]).value_counts().to_dict()
        zs = pd.Series([zc(t) for t in (g[s3 + "_t"] if s3 else g.rsg_s3_t)[(g[s3] if s3 else g.rsg_s3) == 0]]).value_counts().to_dict()
        o.append(f"- {m}: {len(g)} windows, native zeros {int((g[n] == 0).sum())} {zt} (stop3{' (RSG credit)' if not s3 else ''}: "
                 f"{int(((g[s3] if s3 else g.rsg_s3) == 0).sum())} {zs})")
    g = S[S.lead & S.cls.eq("still end") & S.serve]
    o.append(f"- with the serve flag ({len(g)} of them): native zeros RSG {int((g.rsg_n == 0).sum())}, JTN {int((g.jtn_n == 0).sum())}, NR1F {int((g.nr1f_n == 0).sum())}")
    # main table
    o += ["", "## Rules vs RSG-L18-s0 + stop3", "",
          "| model | rule | switched | sum vs RSG+stop3 | per window [CI] | rule increment vs the model's stop3: sum | per window [CI] | new / fixed zeros vs RSG+stop3 | zeros | AF-front | corridor |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    allrows, per = [], []
    for m, (n, s3, lc, _) in MODELS.items():
        b, bt = base(W, m)
        rr = [("model + stop3" + (" (latched: RSG + stop3 credit)" if not s3 else ""), np.zeros(N, bool))] + list(rules(W, m).items())
        for name, mask in rr:
            mask = np.asarray(mask, bool)
            x, xt = np.where(mask, W[n], b), np.where(mask, W[n + "_t"], bt)
            s = summary_row(W, m, name, x, xt, b, bt, r, rt, grp)
            s["switched"] = int(mask.sum())
            allrows.append(s)
            o.append(f"| {m} | {name} | {s['switched']} | {s['sum_vs']:+.1f} | {stats.fmt(s['d'], '+.4f')} | {s['inc_sum']:+.1f} | {stats.fmt(s['inc'], '+.4f')} | "
                     f"{s['new']} / {s['fixed']} | {s['zeros']} | {s['af']} | {s['corr']} |")
            if name in ("A static latch, lead -> native [oracle]", "A static latch, lead -> native [serve]") or (name.startswith("model") and m != "RSG-L18-s0"):
                for st in STRATA:
                    k = (W.stratum == st).values
                    ds = stats.paired(x[k], r[k], groups=grp[k]) if k.sum() >= 20 else None
                    per.append(f"| {m} | {name} | {st} | {int(k.sum())} | {(x - r)[k].sum():+.1f} | {(x - r)[k].sum() / N:+.4f} | "
                               f"{'-' if ds is None else stats.fmt(ds, '+.3f')} | {(x - b)[k].sum():+.1f} | {int(((x == 0) & (r > 0))[k].sum())} / "
                               f"{int(((x > 0) & (r == 0))[k].sum())} |")
    # RSG: the real JEV_HOLD=lead read on the lead-present static latches (full hold, released by the lead head)
    for fl, F in (("oracle", W.lead), ("serve", W.serve)):
        mask = ((W.latch == "static") & F).values
        x, xt = np.where(mask, W.rsg_lead, r), np.where(mask, W.rsg_lead_t, rt)
        s = summary_row(W, "RSG-L18-s0", f"D static latch, lead -> JEV_HOLD=lead's read (full hold, lead-head release; a real read) [{fl}]", x, xt, r, rt, r, rt, grp)
        o.append(f"| RSG-L18-s0 | {s['rule']} | {int(mask.sum())} | {s['sum_vs']:+.1f} | {stats.fmt(s['d'], '+.4f')} | {s['inc_sum']:+.1f} | "
                 f"{stats.fmt(s['inc'], '+.4f')} | {s['new']} / {s['fixed']} | {s['zeros']} | {s['af']} | {s['corr']} |")
    # NR1F literal: every no-lead window RSG + stop3, NR1F native only where a lead is
    for fl, F in (("oracle", W.lead), ("serve", W.serve)):
        x = np.where(F, W.nr1f_n, r)
        xt = np.where(F, W.nr1f_n_t, rt)
        s = summary_row(W, "NR1F-L18-s0", f"NR1F native on every lead window, RSG + stop3 elsewhere [{fl}]", x, xt, r, rt, r, rt, grp)
        o.append(f"| NR1F-L18-s0 | {s['rule']} | {int(F.sum())} | {s['sum_vs']:+.1f} | {stats.fmt(s['d'], '+.4f')} | - | - | {s['new']} / {s['fixed']} | "
                 f"{s['zeros']} | {s['af']} | {s['corr']} |")
    # E: with a lead, a shorter static leash (diag.md's estimator), alone and with C' (arc >= 2, lead > 1 m/s)
    o += ["", "Shorter static leash with a lead (E; offline, diag.md (b) / residual.md's estimator: a lead-present static-latched window whose "
          "stop3 read is an AF-front zero with the ego past the new leash is credited as held (score 1 when the human travels < 5 m, else the "
          "progress estimate); other non-zero, non-rear-hit windows get the progress change min((pre + min(reach, L)) / (0.8 gt), 1) minus the same at 3 m, "
          "reach = the native read's travel past the handoff; E + C' = released windows credited with the native read, the rest shortened):", "",
          "| model | flag | leash with a lead (m) | AF-front avoided | launch progress (sum) | increment vs the model's stop3 (sum) | per window | E + C' (sum) | per window |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m, (n, s3, lc, k) in MODELS.items():
        if not k:
            continue
        pre = W.gt_h - W.arc_after_hand
        reach = (W[n + "_d"] - pre).clip(lower=0)
        est = lambda L: np.where(W.gt_h < 5, 1.0, np.minimum((pre + np.minimum(reach, L)) / (0.8 * W.gt_h), 1.0))  # noqa: E731
        for fl, F in (("oracle", W.lead), ("serve", W.serve)):
            st = (W[lc] == "static") & F
            for L in (2.0, 2.3, 2.5):
                af = st & W[s3 + "_t"].str.startswith("AF-front") & (W[s3 + "_d"] > L)
                ok = st & (W[s3] > 0) & ~W[s3 + "_t"].eq("rear-hit")
                x = np.where(af, est(L), np.where(ok, np.clip(W[s3] + est(L) - est(3.0), 0, 1), W[s3]))
                y = np.where(st & W[k + "_rel"], W[n], x)
                lp = (x - W[s3])[ok & ~af].sum()
                o.append(f"| {m} | {fl} | {L:g} | {int(af.sum())} | {lp:+.1f} | {(x - W[s3]).sum():+.1f} | {(x - W[s3]).sum() / N:+.4f} | "
                         f"{(y - W[s3]).sum():+.1f} | {(y - W[s3]).sum() / N:+.4f} |")
    # release grid (C'): in-sample best of arc x lead speed, upper bound (released windows credited with the native score)
    o += ["", "Release grid (C' rule; release at the first held call from 1 s where the held plan's 2 s arc >= arc and RSG's lead head has "
          "the lead present (p > 0.5) at speed > v; released windows credited with the native read: an in-sample upper bound). Top 3 per model and flag "
          "by the increment over the model's stop3:", "", "| model | flag | arc (m) | v (m/s) | released | increment (sum) | per window | new / fixed zeros vs the model's stop3 |",
          "|---|---|--:|--:|--:|--:|--:|--:|"]
    for m, (n, s3, lc, k) in MODELS.items():
        if not k:
            continue
        h = W.attrs[k]
        for fl, F in (("oracle", W.lead), ("serve", W.serve)):
            g = []
            for arc in (0.0, 1.0, 2.0, 3.0):
                for v in (0.5, 1.0, 1.5, 2.0, 3.0):
                    ids = h[(h.arc_model >= arc) & (h.lead_p > 0.5) & (h.lead_v > v)].id.unique()
                    mk = ((W[lc] == "static") & F & W.index.isin(ids)).values
                    x = np.where(mk, W[n], W[s3])
                    g.append(((x - W[s3]).sum(), arc, v, int(mk.sum()), int(((x == 0) & (W[s3] > 0)).sum()), int(((x > 0) & (W[s3] == 0)).sum())))
            for d, arc, v, k_, nz, fz in sorted(g, reverse=True)[:3]:
                o.append(f"| {m} | {fl} | {arc:g} | {v:g} | {k_} | {d:+.1f} | {d / N:+.4f} | {nz} / {fz} |")
    o += ["", f"Reference RSG-L18-s0 + stop3: zeros {int((r == 0).sum())}, AF-front {int(sum(zc(t) == 'AF-front' for t in rt[r == 0]))}, "
          f"corridor {int(sum(zc(t) == 'corridor' for t in rt[r == 0]))}.", "",
          "## By stratum (model + stop3 and rule A; sum / contribution = summed difference / all held windows; increment = vs the model's stop3)", "",
          "| model | rule | stratum | windows | sum vs RSG+stop3 | contribution | per window [CI] | increment (sum) | new / fixed zeros |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|", *per]
    (out / "lead_aware.md").write_text("\n".join(o) + "\n")
    W.to_csv(out / "windows.csv.gz")
    json.dump([{k: v for k, v in s.items()} for s in allrows], open(out / "rules.json", "w"), indent=1, default=float)
    print("\n".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena", required=True), ap.add_argument("--jt", required=True), ap.add_argument("--out", required=True)
    main(ap.parse_args())
