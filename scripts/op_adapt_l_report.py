"""op-adapt L report: every model's readout -> the registered lines (prereg section 6) and the result tables.

  tables   $L/readout/*/metrics.csv + runs/*/dev.json + readout/*/navtest.json -> research/results/op-adapt-L/
             metrics_all.csv   every metric row of every model
             arms.csv          one row per arm (seeds aggregated) with the headline numbers and the registered lines L1-L7
             lines_by_model.csv  the lines per model / seed
             compare.csv       paired arm-vs-arm capture and false-trigger differences (same rows, cluster bootstrap)
  figs     research/figs/op-adapt-L-*.png

  python scripts/op_adapt_l_report.py tables
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt_l as L  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "research" / "results" / "op-adapt-L"
FIG = REPO / "research" / "figs"
SL = ("start", "stop", "turn_onset")
CAP = {"start": "cap_start", "stop": "cap_stop", "turn_onset": "cap_turn_onset"}
FALSE = (("stay", "false_start"), ("control", "false_stop"), ("control", "false_turn"), ("straight_int", "false_turn"))


def arm_of(model: str) -> str:
    return model.rsplit("-s", 1)[0]


def load_metrics() -> pd.DataFrame:
    """Every readout/<model>/metrics.csv with the model name taken from the directory (main-s0 is a symlink to the selected
    selection run; `alias_of` names its target so that a figure can show it once)."""
    parts = []
    for p in sorted((L.lroot() / "readout").glob("*/metrics.csv")):
        d = p.parent
        if d.name.startswith("pilot"):
            continue
        x = pd.read_csv(p)
        x["model"] = d.name
        x["alias_of"] = d.resolve().name if d.is_symlink() else ""
        parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def get(df, model, st, sl, metric, xs=1.0):
    r = df[(df.model == model) & (df.set == st) & (df.slice == sl) & (df.metric == metric) & (df.xscale == xs)]
    return r.iloc[0] if len(r) else None


def lines_for(df: pd.DataFrame, model: str) -> dict:
    """The registered lines L1-L7 of one model (prereg section 6); values and pass flags."""
    o = {}
    for s in SL:
        r = get(df, model, "wodval", s, CAP[s])
        if r is not None:
            o[f"cap_{s}"], o[f"cap_{s}_lo"], o[f"cap_{s}_hi"], o[f"cap_{s}_orig"] = r.delta, r.lo, r.hi, r.orig
            o[f"L1_{s}"] = bool(r.lo > 0)
    for sl, m in FALSE:
        r = get(df, model, "wodval", sl, m)
        if r is not None:
            o[f"{m}@{sl}"], o[f"{m}@{sl}_hi"] = r.delta, r.hi
            o[f"L2_{m}@{sl}"] = bool(r.hi <= 0.02)
    dm, dp = get(df, model, "wodval", "other", "drift_median"), get(df, model, "wodval", "other", "drift_p95")
    if dm is not None:
        o["drift_median"], o["drift_p95"] = dm.adapt, dp.adapt
    for st in ("wodval", "nusval"):
        for m in ("slow", "fast"):
            r = get(df, model, st, "other", m)
            if r is not None:
                o[f"{m}_pp@{st}"] = 100 * r.delta
        d = get(df, model, st, "other", "drift_median")
        if d is not None:
            o[f"drift_median@{st}"], o[f"drift_p95@{st}"] = d.adapt, get(df, model, st, "other", "drift_p95").adapt
    if dm is not None:
        o["L3"] = bool(o["drift_median"] <= 0.10 and o["drift_p95"] <= 0.50 and all(
            o.get(f"{m}_pp@{st}", 0) <= 2.0 for m in ("slow", "fast") for st in ("wodval", "nusval")) and
            all(o.get(f"drift_median@{st}", 0) <= 0.10 and o.get(f"drift_p95@{st}", 0) <= 0.50 for st in ("wodval", "nusval")))
    for xs in (1.0, 1.06):
        r = get(df, model, "rater", "all", "rfs", xs)
        if r is not None:
            o[f"rfs_delta_x{xs}"], o[f"rfs_lo_x{xs}"], o[f"rfs_hi_x{xs}"], o[f"rfs_orig_x{xs}"] = r.delta, r.lo, r.hi, r.orig
    if "rfs_lo_x1.0" in o:
        o["L4_noharm"] = bool(o["rfs_lo_x1.0"] >= -0.10 and o["rfs_lo_x1.06"] >= -0.10)
        o["L4_notbelow"] = bool(o["rfs_delta_x1.0"] >= 0 and o["rfs_delta_x1.06"] >= 0)
    r = get(df, model, "wodval", "all", "ade4")
    if r is not None:
        o["ade_rel"], o["ade_rel_hi"] = r.delta / r.orig, r.hi / r.orig
        o["L5"] = bool(o["ade_rel_hi"] <= 0.02)
    for s in SL:
        r = get(df, model, "nusval", s, CAP[s])
        if r is not None:
            o[f"nus_cap_{s}"], o[f"nus_cap_{s}_lo"] = r.delta, r.lo
            o[f"L6_{s}"] = bool(r.lo > 0)
    p = L.lroot("readout", model, "navtest.json")
    if p.exists():
        j = json.loads(p.read_text())
        if j.get("delta") is not None:
            o["navtest_delta"] = j["delta"]
            o["L7"] = bool(j["delta"] >= -1.0)
    return o


def cmd_tables(a):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(OUT / "metrics_all.csv", index=False)
    models = sorted(df.model.unique())
    by = {m: lines_for(df, m) for m in models}
    pd.DataFrame.from_dict(by, orient="index").rename_axis("model").reset_index().to_csv(OUT / "lines_by_model.csv", index=False)
    rows = []
    for arm in sorted({arm_of(m) for m in models}):
        ms = [m for m in models if arm_of(m) == arm]
        keys = sorted({k for m in ms for k in by[m]})
        r = {"arm": arm, "seeds": len(ms), "models": ",".join(ms)}
        for k in keys:
            v = [by[m][k] for m in ms if k in by[m]]
            if v and isinstance(v[0], (bool, np.bool_)):
                r[k] = all(v)                                             # a registered line passes when every seed passes
            elif v:
                r[k] = float(np.mean(v))
                if k.endswith("_lo"):
                    r[k] = float(np.min(v))
                if k.endswith("_hi") or k in ("drift_p95",):
                    r[k] = float(np.max(v))
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "arms.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string())


# ---------------------------------------------------------------- arm-vs-arm paired comparison
PAIRS = [("main", "noint"), ("main", "only_start"), ("main", "only_stop"), ("main", "only_turn"), ("main", "nocontrast"), ("main", "stayheavy"),
         ("main", "tr_ad"), ("main", "dw03"), ("main", "dw1"), ("main", "dw3"), ("main", "dw10"), ("main", "long"),
         ("sel_s4ia", "sel_polia"), ("sel_s4ia", "sel_s4polia"), ("sel_s4ia", "sel_polid"), ("sel_polia", "sel_polid"),
         ("sel_s4ia", "sel_s4ia_dw3"), ("sel_polia", "sel_polia_dw3"), ("sel_s4polia", "sel_s4polia_dw3"), ("sel_polid", "sel_polid_dw3")]
KEYS = (("start", "cap_start"), ("stop", "cap_stop"), ("turn_onset", "cap_turn_onset"), ("stay", "false_start"), ("control", "false_stop"),
        ("control", "false_turn"), ("straight_int", "false_turn"))


def arm_models(arm: str) -> list:
    return [p.name for p in sorted((L.lroot("readout")).glob(f"{arm}-s*")) if (p / "eval" / "wodval.npz").exists()]


def cmd_compare(a):
    D = L.Data(("wod", "wodval", "nus"))
    tab = D.tab["wodval"]
    rows_ = None
    cache = {}

    def arm_rows(arm):
        nonlocal rows_
        if arm in cache:
            return cache[arm]
        ms = arm_models(arm)
        if not ms:
            return None
        acc = None
        for m in ms:
            z = np.load(L.lroot("readout", m, "eval", "wodval.npz"))
            rows_ = z["rows"]
            mt = L.row_metrics(z["plan"], tab, rows_, D.cam["wodval"])
            acc = {k: mt[k].astype(float) for k in mt} if acc is None else {k: acc[k] + mt[k] for k in mt}
        cache[arm] = ({k: v / len(ms) for k, v in acc.items()}, len(ms))
        return cache[arm]
    out = []
    for A_, B_ in PAIRS:
        ra, rb = arm_rows(A_), arm_rows(B_)
        if ra is None or rb is None:
            continue
        fl = {s: np.asarray(tab[f"s_{s}"])[rows_] for s in ("start", "stop", "turn_onset", "stay", "control", "straight_int")}
        groups = np.asarray(tab["seq"])[rows_]
        for sl, m in KEYS:
            d = L.paired_delta(ra[0][m][fl[sl]], rb[0][m][fl[sl]], groups[fl[sl]])
            out.append({"a": A_, "b": B_, "seeds_a": ra[1], "seeds_b": rb[1], "slice": sl, "metric": m, **{k: v for k, v in d.items() if k in ("n", "adapt", "orig", "delta", "lo", "hi")}})
    df = pd.DataFrame(out).rename(columns={"adapt": "a_mean", "orig": "b_mean"})
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "compare.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tables", "compare"])
    a = ap.parse_args()
    {"tables": cmd_tables, "compare": cmd_compare}[a.cmd](a)
