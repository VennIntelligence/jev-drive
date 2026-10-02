"""Common-cause factorial: per-sample metrics, effects with cluster-bootstrap CIs, cross-domain verdicts.
Plan: ../plans/2026-10-03-common-cause-prereg.md. Main venv, CPU.

Reads $DATA_DIR/runs/op_common_cause/{samples/<d>.json, raw/<d>/<d>-<i>of<n>.npz}; writes per-sample metrics
(<run>/metrics_<d>.parquet) and the tables into experiments/op_common_cause/results/ (effects.{csv,md},
verdicts.{csv,md}, checks.{csv,md}).

  .venv/bin/python experiments/op_common_cause/scripts/cc_report.py [--domains nav wod carla]

The low - mid contrast is a difference between two speed bins of one domain; logs / sequences / routes can hold frames
of both bins, so the clusters are resampled jointly (same convention as jevdrive.stats: default_rng(0), B = 10 000,
percentile CI of the ratio-of-sums mean), which jevdrive.stats does not offer for a two-group contrast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run  # noqa: E402
from jevdrive.wod_zeroshot import openpilot_to_wod  # noqa: E402

ROOT = data_dir() / "runs" / "op_common_cause"
RES = REPO / "experiments" / "op_common_cause" / "results"
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
K3, K2 = 11, 7               # 0.25 s grid index of 3 s and 2 s
MOVING = ("low", "mid", "high")


def load(dom, check=False):
    S = {s["id"]: s for s in json.loads((ROOT / "samples" / f"{dom}.json").read_text())}
    parts = sorted((ROOT / "raw" / dom).glob(f"{dom}-*of*.npz"))
    parts = [p for p in parts if ("check" in p.name) == check]
    full = {p.name.split(".")[0] for p in parts if not p.name.endswith(".part.npz")}
    parts = [p for p in parts if not (p.name.endswith(".part.npz") and p.name.split(".")[0] in full)]   # partial only if unfinished
    rows = []
    for p in parts:
        with np.load(p) as f:
            z = {k: f[k] for k in f.files}          # NpzFile re-reads an array on every key access
        V = z["variants"].tolist()
        for i, sid in enumerate(z["ids"].tolist()):
            s = S[sid]
            cam = np.asarray(s["cam"], float)
            r = dict(id=sid, cluster=s["cluster"], v=s["v"], bin=s["bin"], cmd=s["cmd"])
            fut = np.asarray(s["fut"], float)
            r["ystar3"], r["xstar3"] = fut[K3, 1], fut[K3, 0]
            h0 = z["hidden"][i, V.index("normal|off")].astype(np.float32)
            for j, vn in enumerate(V):
                pp = z["plan_pos"][i, j]
                if not np.isfinite(pp).all():
                    continue
                w = openpilot_to_wod(pp, z["plan_yaw"][i, j], T_IDXS, cam[:2])
                psi = -np.degrees(np.interp(3.0, T_IDXS, z["plan_yaw"][i, j]))
                r[f"y3:{vn}"], r[f"x3:{vn}"], r[f"y2:{vn}"], r[f"psi3:{vn}"] = w[K3, 1], w[K3, 0], w[K2, 1], psi
                h = z["hidden"][i, j].astype(np.float32)
                r[f"hcos:{vn}"] = 1 - float(h @ h0 / (np.linalg.norm(h) * np.linalg.norm(h0) + 1e-9))
                r[f"indiff:{vn}"] = float(z["in_diff"][i, j])
                r[f"pulses:{vn}"] = int(z["pulses"][i, j])
            rows.append(r)
    d = pd.DataFrame(rows)
    nan = pd.Series(np.nan, index=d.index)
    col = lambda c: d[c] if c in d else nan  # noqa: E731
    lat = lambda vn: (col(f"y3:{vn}") - d.ystar3).abs()  # noqa: E731
    sg = -np.sign(d.cmd)          # cmd -1 = left; y is left-positive, so "toward the command" = -cmd * dy
    m = pd.DataFrame(dict(id=d.id, cluster=d.cluster, v=d.v, bin=d.bin, cmd=d.cmd, turn=d.cmd != 0))
    m["E0_lat3"] = lat("normal|off")
    m["E0_bias_y3"] = col("y3:normal|off") - d.ystar3            # signed (left +); read on straight-command frames
    m["E0_bias_y3_repeat"] = col("y3:repeat|off") - d.ystar3
    m["E1_G_deg"] = (col("psi3:rotL|off") - col("psi3:rotR|off")) / 2
    m["E1_Gy_m"] = (col("y3:rotL|off") - col("y3:rotR|off")) / 2
    m["E2_dlat_repeat"] = lat("repeat|off") - lat("normal|off")
    m["E2_dx_repeat"] = col("x3:repeat|off") - col("x3:normal|off")
    m["E3_dlat_single"] = lat("single|off") - lat("normal|off")
    m["E3_dx_single"] = col("x3:single|off") - col("x3:normal|off")
    m["E4_dlat_short"] = lat("normal|off") - lat("long|off")
    m["E4_dx_short"] = col("x3:normal|off") - col("x3:long|off")
    for dn in ("pulse", "sustained", "lc"):
        m[f"E5_dlat_{dn}"] = lat(f"normal|{dn}") - lat("normal|off")
    m["E5_S_steer_m"] = sg * (col("y3:normal|pulse") - col("y3:normal|wrong"))
    m["E5_toward_cmd_pulse_m"] = sg * (col("y3:normal|pulse") - col("y3:normal|off"))
    m["E5_toward_cmd_sust_m"] = sg * (col("y3:normal|sustained") - col("y3:normal|off"))
    m["E6_dG_sust_minus_off"] = (col("psi3:rotL|sustained") - col("psi3:rotR|sustained")) / 2 - m.E1_G_deg
    m["E6_dlat_sust_under_repeat"] = lat("repeat|sustained") - lat("repeat|off")
    for h in ("long", "repeat", "single", "rotL", "rotR"):
        m[f"SC_dy3:{h}"] = (col(f"y3:{h}|off") - col("y3:normal|off")).abs()
        m[f"SC_dpsi3:{h}"] = (col(f"psi3:{h}|off") - col("psi3:normal|off")).abs()
    chk = {c: d[c] for c in d if c.split(":")[0] in ("hcos", "indiff", "pulses")}
    return m, pd.DataFrame(chk).assign(bin=d.bin, cmd=d.cmd)


def contrast(x, g, a_mask, b_mask, B=stats.N_BOOT, seed=stats.SEED, alpha=stats.ALPHA):
    """mean(x | a) - mean(x | b) with clusters resampled jointly (ratio of sums per group)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g, a_mask, b_mask = x[ok], np.asarray(g)[ok], np.asarray(a_mask)[ok], np.asarray(b_mask)[ok]
    if a_mask.sum() < 5 or b_mask.sum() < 5:
        return dict(n=int(a_mask.sum() + b_mask.sum()), mean=np.nan, lo=np.nan, hi=np.nan)
    u, gi = np.unique(g, return_inverse=True)
    sa = np.bincount(gi, x * a_mask, len(u)); ca = np.bincount(gi, a_mask.astype(float), len(u))
    sb = np.bincount(gi, x * b_mask, len(u)); cb = np.bincount(gi, b_mask.astype(float), len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    bs = sa[idx].sum(1) / np.maximum(ca[idx].sum(1), 1e-9) - sb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1e-9)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(n=int(a_mask.sum() + b_mask.sum()), mean=float(x[a_mask].mean() - x[b_mask].mean()), lo=float(lo), hi=float(hi))


EFFECTS = [c for c in ("E0_lat3", "E0_bias_y3", "E0_bias_y3_repeat", "E1_G_deg", "E1_Gy_m", "E2_dlat_repeat", "E2_dx_repeat", "E3_dlat_single", "E3_dx_single",
                       "E4_dlat_short", "E4_dx_short", "E5_dlat_pulse", "E5_dlat_sustained", "E5_dlat_lc", "E5_S_steer_m",
                       "E5_toward_cmd_pulse_m", "E5_toward_cmd_sust_m", "E6_dG_sust_minus_off", "E6_dlat_sust_under_repeat")]
TURN_ONLY = {c for c in EFFECTS if c.startswith(("E5", "E6"))}
STRAIGHT_ONLY = {"E0_bias_y3", "E0_bias_y3_repeat"}


def effects(dom, m):
    rows = []
    for e in EFFECTS:
        base = m[m.turn] if e in TURN_ONLY else m[~m.turn] if e in STRAIGHT_ONLY else m
        groups = [("moving", base.bin.isin(MOVING))] + [(b, base.bin == b) for b in ("stop", "low", "mid", "high")]
        if e in TURN_ONLY:
            groups += [("moving-left", base.bin.isin(MOVING) & (base.cmd < 0)), ("moving-right", base.bin.isin(MOVING) & (base.cmd > 0))]
        for grp, mask in groups:
            x = base[e][mask]
            r = stats.bootstrap(x, groups=base.cluster[mask]) if np.isfinite(x).sum() >= 5 else dict(n=int(np.isfinite(x).sum()), mean=np.nan, lo=np.nan, hi=np.nan)
            rows.append(dict(domain=dom, effect=e, group=grp, **{k: r[k] for k in ("n", "mean", "lo", "hi")}))
        r = contrast(base[e], base.cluster, (base.bin == "low").to_numpy(), (base.bin == "mid").to_numpy())
        rows.append(dict(domain=dom, effect=e, group="low-mid", **r))
    return rows


def verdict(a, b):
    sa = a["lo"] > 0 or a["hi"] < 0
    sb = b["lo"] > 0 or b["hi"] < 0
    if not (np.isfinite(a["mean"]) and np.isfinite(b["mean"])):
        return "n/a"
    if sa and sb:
        if np.sign(a["mean"]) != np.sign(b["mean"]):
            return "opposite"
        ratio = a["mean"] / b["mean"]
        return "consistent" if 0.5 <= ratio <= 2 else "same sign, different size"
    if sa or sb:
        return "one domain only"
    return "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["nav", "wod", "wodt", "carla"])
    ap.add_argument("--interim", action="store_true", help="tables to the run dir (partial shards), not results/")
    ap.add_argument("--check", action="store_true", help="the smoke-test shards only; tables to the run dir, not results/")
    a = ap.parse_args()
    with Run("op_common_cause", "report", config=vars(a)) as run:
        eff, chks = [], []
        for dom in a.domains:
            if not (ROOT / "raw" / dom).exists() or not any(p for p in (ROOT / "raw" / dom).glob("*.npz") if ("check" in p.name) == a.check):
                continue
            m, ck = load(dom, a.check)
            m.to_parquet(run.path(f"metrics_{dom}.parquet"))
            run.info(f"{dom}: {len(m)} samples")
            eff += effects(dom, m)
            for c in ck.columns:
                if c in ("bin", "cmd"):
                    continue
                x = ck[c].astype(float)
                x = x[x.notna() & (x >= 0)]
                if len(x):
                    chks.append(dict(domain=dom, check=c, n=len(x), mean=float(x.mean()), min=float(x.min()), max=float(x.max())))
        global RES
        if a.check or a.interim:
            RES = run.path("tables")
        E = pd.DataFrame(eff)
        stats.write_table(E, RES / "effects")
        stats.write_table(pd.DataFrame(chks), RES / "checks")
        V = []
        key = E.set_index(["domain", "effect", "group"])
        for e in EFFECTS:
            for grp in ("moving", "stop", "low", "mid", "high", "low-mid", "moving-left", "moving-right"):
                for p, q in (("nav", "carla"), ("wod", "carla"), ("wodt", "carla"), ("nav", "wod")):
                    if (p, e, grp) in key.index and (q, e, grp) in key.index:
                        ra, rb = key.loc[(p, e, grp)], key.loc[(q, e, grp)]
                        V.append(dict(effect=e, group=grp, pair=f"{p}-{q}", a=stats.fmt(ra), b=stats.fmt(rb), verdict=verdict(ra, rb)))
        stats.write_table(pd.DataFrame(V), RES / "verdicts")
        run.summary["n_effect_rows"] = len(E)


if __name__ == "__main__":
    main()
