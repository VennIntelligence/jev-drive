"""Night queue 3, Q6 (a): the backbone x head x exam main table under one protocol (todos/2026-09-26-night-queue-3.md,
Q6 and the [D] 16:50 entry, items 1-15; research/ablation-matrix-inventory.md section 9).

Every cell comes from a stored result (the small tables in research/results/ or the box run dirs) and is either taken
("take": the stored number is already in the unified protocol) or recomputed ("recompute": re-judged from stored
per-frame predictions). Conventions: P5 / I3 flips in %, per frame, the exam's own null; 3-seed cells report the seed
mean with the seed-0 CI and the seed range; RFS absolute = cluster mean, RFS paired = frame mean; WOD = W-train;
NAVSIM = navtest PDMS (v1.1) with EPDMS beside. Outputs (research/results/nq3/q6/): main_table.csv (long),
main_table.md (wide), diff_vs_old.csv (the inventory's value, the new value, a reason code), sources.csv.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
R = REPO / "research" / "results"
OUT = R / "nq3" / "q6"
PED = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian", "ParkingCrossingPedestrian")
CUTIN = ("HighwayCutIn", "StaticCutIn", "ParkingCutIn")
BB = {"qwen L18_last": "Qwen3-VL-4B", "vjepa2 mean": "V-JEPA 2 ViT-L", "siglip2 patch_mean": "SigLIP2 so400m",
      "dinov2 patch_mean": "DINOv2-B", "opsmall temporal": "openpilot small"}
MOD = {"cinque": "openpilot Cinque", "lebowski": "openpilot Lebowski"}

# the inventory's numbers (research/ablation-matrix-inventory.md sections 1-5): (row, column) -> old value
OLD = {
    ("ego-only | ridge ego", "P5 BA ped"): 7.6, ("ego-only | ridge ego", "P5 BA cut-in"): 3.8, ("ego-only | ridge ego", "P5 BA null"): 0.5,
    ("Qwen3-VL-4B | ridge_late", "P5 BA ped"): 0.0, ("Qwen3-VL-4B | ridge_late", "P5 BA null"): 5.1,
    ("openpilot Cinque | ridge_late (= prior)", "P5 BA ped"): 0.2, ("openpilot Cinque | ridge_late (= prior)", "P5 BA cut-in"): 76.9,
    ("openpilot Cinque | ridge_late (= prior)", "P5 BA null"): 5.0,
    ("openpilot Lebowski | ridge_late (= prior)", "P5 BA ped"): 2.7, ("openpilot Lebowski | ridge_late (= prior)", "P5 BA cut-in"): 70.6,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "P5 BA ped"): 43.3, ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "P5 BA cut-in"): 80.0,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "P5 BA null"): 5.1,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)", "P5 BA ped"): 41.6, ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)", "P5 BA cut-in"): 77.2,
    ("Qwen3-VL-4B | pair-Δ single (prior Cinque)", "P5 BA ped"): 42.1, ("Qwen3-VL-4B | pair-Δ single (prior Cinque)", "P5 BA cut-in"): 71.4,
    ("openpilot Cinque | M-C pair op (single)", "P5 BA ped"): 6.9, ("openpilot Cinque | M-C pair op (single)", "P5 BA cut-in"): 85.1,
    ("openpilot Cinque | M-C hard (control)", "P5 BA ped"): 0.0, ("openpilot Cinque | M-C hard (control)", "P5 BA cut-in"): 77.5,
    ("openpilot Cinque | M-C uniform (control)", "P5 BA ped"): 0.0, ("openpilot Cinque | M-C uniform (control)", "P5 BA cut-in"): 78.0,
    ("openpilot Cinque | E5 student A", "P5 BA ped"): 52.7, ("openpilot Cinque | E5 student B", "P5 BA ped"): 51.5,
    ("TFv6 | waypoint 2 s", "P5 BA ped"): 29.3, ("TFv6 | waypoint 2 s", "P5 BA cut-in"): 32.5, ("TFv6 | waypoint 2 s", "P5 BA null"): 5.5,
    ("TFv6 | target speed", "P5 BA ped"): 0.0,
    ("ego-only | ridge ego", "I3"): 0.0, ("Qwen3-VL-4B | ridge_late", "I3"): 2.8,
    ("openpilot Cinque | ridge_late (= prior)", "I3"): 70.0, ("openpilot Lebowski | ridge_late (= prior)", "I3"): 70.5,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "I3"): 58.7, ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)", "I3"): 67.4,
    ("Qwen3-VL-4B | pair-Δ single (prior Cinque)", "I3"): 59.6, ("openpilot Cinque | M-C pair op (single)", "I3"): 68.7,
    ("openpilot Cinque | M-C hard (control)", "I3"): 64.6, ("openpilot Cinque | M-C uniform (control)", "I3"): 69.8,
    ("ego-only | ridge ego", "WOD RFS"): 7.065, ("ego-only | cls ego", "WOD RFS"): 7.262,
    ("Qwen3-VL-4B | ridge_late", "WOD RFS"): 6.942, ("Qwen3-VL-4B | ridge_late", "WOD pre-onset ΔADE"): -0.005,
    ("V-JEPA 2 ViT-L | ridge_late", "WOD pre-onset ΔADE"): -0.030,
    ("openpilot Cinque | ridge_late (= prior)", "WOD RFS"): 7.449, ("openpilot Cinque | ridge_late (= prior)", "WOD pre-onset ΔADE"): -0.294,
    ("openpilot Lebowski | ridge_late (= prior)", "WOD RFS"): 7.522, ("openpilot Lebowski | ridge_late (= prior)", "WOD pre-onset ΔADE"): -0.318,
    ("openpilot Cinque | cls_late", "WOD RFS"): 7.637, ("openpilot Lebowski | cls_late", "WOD RFS"): 7.734,
    ("openpilot Cinque | native plan", "WOD RFS"): 8.005, ("openpilot Lebowski | native plan", "WOD RFS"): 7.886,
    ("openpilot small | native plan", "WOD RFS"): 7.640,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "WOD RFS paired Δ"): -1.02, ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)", "WOD RFS paired Δ"): -1.52,
    ("ego-only | ridge ego", "NAVSIM PDMS"): 62.7, ("ego-only | cls ego", "NAVSIM PDMS"): 68.4,
    ("openpilot Cinque | ridge_late (= prior)", "NAVSIM PDMS"): 73.5, ("openpilot Cinque | cls_late", "NAVSIM PDMS"): 77.9,
    ("openpilot Lebowski | ridge_late (= prior)", "NAVSIM PDMS"): 72.4, ("openpilot Lebowski | cls_late", "NAVSIM PDMS"): 77.3,
    ("openpilot Cinque | Hydra", "NAVSIM PDMS"): 84.2, ("openpilot Cinque | Hydra", "NAVSIM EPDMS"): 82.6,
    ("openpilot Cinque | native plan", "NAVSIM PDMS"): 52.1, ("openpilot Lebowski | native plan", "NAVSIM PDMS"): 50.9,
    ("openpilot small | native plan", "NAVSIM PDMS"): 47.4, ("Alpamayo 1.5 | native plan (nav)", "NAVSIM PDMS"): 44.3,
    ("ego-only | ridge ego", "NAVSIM EPDMS"): 64.2, ("ego-only | cls ego", "NAVSIM EPDMS"): 67.8,
    ("openpilot Cinque | ridge_late (= prior)", "NAVSIM EPDMS"): 73.9, ("openpilot Cinque | cls_late", "NAVSIM EPDMS"): 77.4,
    ("openpilot Lebowski | ridge_late (= prior)", "NAVSIM EPDMS"): 72.9, ("openpilot Lebowski | cls_late", "NAVSIM EPDMS"): 76.7,
    ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "NAVSIM PDMS paired Δ"): -8.2, ("Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)", "NAVSIM PDMS paired Δ"): -11.0,
}


class Table:
    def __init__(self):
        self.cells, self.missing = [], []

    def add(self, row, col, value, lo=np.nan, hi=np.nan, seeds=None, source="", status="take", note="", hint=""):
        seeds = [] if seeds is None else [float(s) for s in seeds]
        self.cells.append({"row": row, "column": col, "value": float(value) if value is not None else np.nan,
                           "lo": float(lo), "hi": float(hi), "n_seeds": len(seeds) or 1,
                           "seed_min": min(seeds) if seeds else np.nan, "seed_max": max(seeds) if seeds else np.nan,
                           "source": str(source), "status": status, "note": note, "hint": hint})

    def text(self, row, col, text, source="", note=""):
        self.cells.append({"row": row, "column": col, "value": np.nan, "lo": np.nan, "hi": np.nan, "n_seeds": 0,
                           "seed_min": np.nan, "seed_max": np.nan, "source": source, "status": "text", "note": text + (f"; {note}" if note else ""),
                           "hint": ""})

    def guard(self, name, fn):
        try:
            fn()
        except Exception as e:                      # a missing source leaves its cells empty, it must not stop the table
            log.warning("source %s failed: %r", name, e)
            self.missing.append({"source": name, "error": repr(e)[:300]})


def _csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(R / rel)


def _pct(x):
    return 100 * float(x)


# ---------------------------------------------------------------- P5 v1 BA / PDM

def p5_n6(T: Table):
    """N6's 3-seed criteria: backbone ridge_late, pair-Δ single / dual (Qwen dual = M-C), op priors."""
    c = _csv("night2/N6/criteria_seeds.csv")
    src = "research/results/night2/N6/criteria_seeds.csv"

    def put(row, sub):
        s0 = sub[sub.seed == 0].iloc[0]
        T.add(row, "P5 BA ped", _pct(sub.ped_flip.mean()), _pct(s0.ped_lo), _pct(s0.ped_hi), 100 * sub.ped_flip, src,
              hint="seed-mean")
        T.add(row, "P5 BA cut-in", _pct(sub.cutin_flip.mean()), seeds=100 * sub.cutin_flip, source=src, hint="seed-mean")
        T.add(row, "P5 BA null", _pct(sub.null_ff_oos.mean()), seeds=100 * sub.null_ff_oos, source=src, hint="seed-mean")
    for key, name in BB.items():
        put(f"{name} | ridge_late", c[(c.prior == "cinque") & (c.arm == f"ridge_late {key}")])
        for m in ("cinque", "lebowski"):
            short = "Cinque" if m == "cinque" else "Lebowski"
            s = c[(c.prior == m) & (c.arm == f"pair-Δ {key} [{m}]")]
            if len(s):
                put(f"{name} | pair-Δ single (prior {short})", s)
            s = c[(c.prior == m) & (c.arm == f"pair-Δ dual {key} [{m}]")]
            if len(s):
                put(f"{name} | pair-Δ dual{' = M-C' if key.startswith('qwen') else ''} (⊕ {short})", s)
    for m in ("cinque", "lebowski"):
        put(f"{MOD[m]} | ridge_late (= prior)", c[(c.prior == m) & (c.arm == f"prior [{m}]")])


def p5_mc_controls(T: Table):
    """M-C single op stream, hard-example and uniform controls: 3-seed ped flips (mc_seeds), seed-0 CI and cut-in (E4)."""
    s = _csv("elicitation/seeds/mc_seeds.csv")
    e4 = _csv("elicitation/e4/e4_all.csv")
    for m in ("cinque", "lebowski"):
        for arm, label in (("M-C pair op", "M-C pair op (single)"), ("M-C hard", "M-C hard (control)"),
                           ("M-C uniform", "M-C uniform (control)")):
            row = f"{MOD[m]} | {label}"
            g = s[(s.set == "ba") & (s.model == m) & (s.arm == arm)].set_index("metric")
            e = e4[(e4.set == "ba") & (e4.examinee == f"{arm} [{m}]") & (e4.window == "per frame")].set_index("scope")
            seeds = [g.loc["ped_flip", k] for k in ("s0", "s1", "s2")]
            T.add(row, "P5 BA ped", 100 * np.mean(seeds), _pct(e.loc["pedestrian", "lo"]), _pct(e.loc["pedestrian", "hi"]),
                  100 * np.array(seeds), "elicitation/seeds/mc_seeds.csv + e4/e4_all.csv (CI seed 0)", hint="seed-mean")
            T.add(row, "P5 BA cut-in", _pct(e.loc["cut-in", "flip"]), _pct(e.loc["cut-in", "lo"]), _pct(e.loc["cut-in", "hi"]),
                  source="elicitation/e4/e4_all.csv", note="seed 0 only")
            nulls = [g.loc["null_ff_oos", k] for k in ("s0", "s1", "s2")]
            T.add(row, "P5 BA null", 100 * np.mean(nulls), seeds=100 * np.array(nulls), source="elicitation/seeds/mc_seeds.csv",
                  hint="seed-mean")


def p5_e4(T: Table):
    """ego-only, TFv6 channels (per frame, seed-free) and the PDM set per pair."""
    e4 = _csv("elicitation/e4/e4_all.csv")
    src = "elicitation/e4/e4_all.csv"
    for ex, row, note in (("ridge ego", "ego-only | ridge ego", "τ = 0 label artefact, not a reaction"),
                          ("TFv6 waypoint speed 2 s", "TFv6 | waypoint 2 s", ""), ("TFv6 target speed", "TFv6 | target speed", "")):
        e = e4[(e4.set == "ba") & (e4.examinee == ex) & (e4.window == "per frame")].set_index("scope")
        T.add(row, "P5 BA ped", _pct(e.loc["pedestrian", "flip"]), _pct(e.loc["pedestrian", "lo"]), _pct(e.loc["pedestrian", "hi"]),
              source=src, note=note)
        T.add(row, "P5 BA cut-in", _pct(e.loc["cut-in", "flip"]), _pct(e.loc["cut-in", "lo"]), _pct(e.loc["cut-in", "hi"]), source=src, note=note)
        T.add(row, "P5 BA null", _pct(e.loc["pooled", "null_ff_oos"]), source=src)
    # PDM-Lite set: per pair only (E4's decision)
    names = {"ridge ego": "ego-only | ridge ego", "TFv6 waypoint speed 2 s": "TFv6 | waypoint 2 s",
             "prior [cinque]": "openpilot Cinque | ridge_late (= prior)", "prior [lebowski]": "openpilot Lebowski | ridge_late (= prior)",
             "M-C pair [cinque]": "Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "M-C pair [lebowski]": "Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)",
             "M-C pair qwen [cinque]": "Qwen3-VL-4B | pair-Δ single (prior Cinque)", "ridge_late L18_last": "Qwen3-VL-4B | ridge_late",
             "M-C pair op [cinque]": "openpilot Cinque | M-C pair op (single)", "M-C hard [cinque]": "openpilot Cinque | M-C hard (control)",
             "M-C uniform [cinque]": "openpilot Cinque | M-C uniform (control)"}
    for ex, row in names.items():
        e = e4[(e4.set == "pdm") & (e4.examinee == ex) & (e4.window == "(b) per pair") & (e4.scope == "pooled")]
        if len(e):
            T.add(row, "P5 PDM per pair", _pct(e.flip.iloc[0]), _pct(e.lo.iloc[0]), _pct(e.hi.iloc[0]), source=src,
                  note="PDM-Lite set, per pair (E4 decision), seed 0")


def p5_a3(T: Table):
    """A3 ([L, 3 s] area minus the null floor, pedestrian) beside the per-frame flips, where it was computed."""
    a = _csv("elicitation/e4c/areas.csv")
    names = {"ridge ego": "ego-only | ridge ego", "TFv6 waypoint speed 2 s": "TFv6 | waypoint 2 s",
             "prior [cinque]": "openpilot Cinque | ridge_late (= prior)", "prior [lebowski]": "openpilot Lebowski | ridge_late (= prior)",
             "M-C pair [cinque]": "Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Cinque)", "M-C pair [lebowski]": "Qwen3-VL-4B | pair-Δ dual = M-C (⊕ Lebowski)",
             "ridge_late L18_last": "Qwen3-VL-4B | ridge_late"}
    for ex, row in names.items():
        e = a[(a.set == "ba") & (a.examinee == ex) & (a.scope == "pedestrian")]
        if len(e):
            T.add(row, "P5 BA A3 ped", float(e.A3_minus_null.iloc[0]), float(e.A3_diff_lo.iloc[0]), float(e.A3_diff_hi.iloc[0]),
                  source="elicitation/e4c/areas.csv", note="area minus null floor, seed 0")
    s = _csv("night2/N3/e4c_students_areas.csv")
    for m in ("cinque", "lebowski"):
        for arm in ("A", "B"):
            e = s[(s.set == "ba") & (s.examinee.str.startswith(f"E5 {arm} s")) & (s.examinee.str.endswith(f"[{m}]")) & (s.scope == "pedestrian")]
            if len(e):
                s0 = e[e.examinee == f"E5 {arm} s0 [{m}]"].iloc[0]
                T.add(f"{MOD[m]} | E5 student {arm}", "P5 BA A3 ped", float(e.A3_minus_null.mean()), s0.A3_diff_lo, s0.A3_diff_hi,
                      e.A3_minus_null, "night2/N3/e4c_students_areas.csv", hint="seed-mean")


def p5_n3(T: Table):
    """P5-trained cls_late (3 seeds), NAVSIM-trained heads zero-shot (ridge_late deterministic, cls_late 3 seeds), Hydra."""
    c = _csv("night2/N3/p5_criteria.csv")
    src = "research/results/night2/N3/p5_criteria.csv"
    for m in ("cinque", "lebowski"):
        for pat, row in ((f"cls_late P5 s{{}} [{m}]", f"{MOD[m]} | cls_late (P5-trained)"),
                         (f"cls_late NAV s{{}} [{m}]", f"{MOD[m]} | cls_late (NAVSIM-trained, zero-shot)")):
            sub = c[c.arm.isin([pat.format(s) for s in (0, 1, 2)])]
            s0 = sub[sub.arm == pat.format(0)].iloc[0]
            T.add(row, "P5 BA ped", _pct(sub.ped_flip.mean()), _pct(s0.ped_lo), _pct(s0.ped_hi), 100 * sub.ped_flip, src, hint="seed-mean")
            T.add(row, "P5 BA cut-in", _pct(sub.cutin_flip.mean()), seeds=100 * sub.cutin_flip, source=src, hint="seed-mean")
            T.add(row, "P5 BA null", _pct(sub.null_ff_oos.mean()), seeds=100 * sub.null_ff_oos, source=src)
        r = c[c.arm == f"ridge_late NAV [{m}]"].iloc[0]
        row = f"{MOD[m]} | ridge_late (NAVSIM-trained, zero-shot)"
        T.add(row, "P5 BA ped", _pct(r.ped_flip), _pct(r.ped_lo), _pct(r.ped_hi), source=src, note="deterministic")
        T.add(row, "P5 BA cut-in", _pct(r.cutin_flip), source=src, note="deterministic")
        T.add(row, "P5 BA null", _pct(r.null_ff_oos), source=src)
        for col in ("P5 BA ped", "I3"):
            T.text(f"{MOD[m]} | Hydra", col, "不可比（N3 兼容检查 top-10 重叠 < 30%；Q4b 重检）", "night2/N3/compat.csv")


def p5_students(T: Table):
    c = _csv("elicitation/e5/criteria.csv")
    src = "elicitation/e5/criteria.csv"
    for m in ("cinque", "lebowski"):
        for arm in ("A", "B"):
            sub = c[c.arm.isin([f"E5 {arm} s{s} [{m}]" for s in (0, 1, 2)])]
            s0 = sub[sub.arm == f"E5 {arm} s0 [{m}]"].iloc[0]
            row = f"{MOD[m]} | E5 student {arm}"
            T.add(row, "P5 BA ped", _pct(sub.ped_flip.mean()), _pct(s0.ped_lo), _pct(s0.ped_hi), 100 * sub.ped_flip, src, hint="seed-mean")
            T.add(row, "P5 BA cut-in", _pct(sub.cutin_flip.mean()), seeds=100 * sub.cutin_flip, source=src, hint="seed-mean")
            T.add(row, "P5 BA null", _pct(sub.null_ff_oos.mean()), seeds=100 * sub.null_ff_oos, source=src)


def _families(df, ex):
    d = df[df.examinee == ex]
    out = {}
    for name, fam in (("ped", PED), ("cut-in", CUTIN)):
        s = d[d.scope.isin(fam)]
        out[name] = 100 * float((s.flip_rate * s.n_reactive).sum() / max(s.n_reactive.sum(), 1)) if len(s) else np.nan
    p = d[d.scope == "pooled"].iloc[0]
    return out, p


def p5_top10(T: Table):
    for rel, ex, row in (("top10-exams/t1_p5_flip_rates.csv", "SparseDriveV2", "SparseDriveV2 | native (zero-shot)"),
                         ("top10-exams/t1_p5_flip_rates.csv", "ZTRS", "ZTRS | native (zero-shot)"),
                         ("top10-exams/p5_t3_flip_rates.csv", "BridgeDrive waypoint speed 2 s", "BridgeDrive | waypoint 2 s"),
                         ("top10-exams/p5_t3_flip_rates.csv", "BridgeDrive target speed", "BridgeDrive | target speed"),
                         ("top10-exams/p5_t3_flip_rates.csv", "BLUE waypoint speed 2 s", "BLUE | waypoint 2 s"),
                         ("top10-exams/p5_t3_flip_rates.csv", "SimLingo waypoint speed 2 s", "SimLingo | waypoint 2 s")):
        df = _csv(rel)
        if ex not in set(df.examinee):
            continue
        fam, p = _families(df, ex)
        T.add(row, "P5 BA ped", fam["ped"], source=rel, note="n-weighted over pedestrian families, no CI", status="recompute")
        T.add(row, "P5 BA cut-in", fam["cut-in"], source=rel, note="n-weighted over cut-in families, no CI", status="recompute")
        T.add(row, "P5 BA pooled", _pct(p.flip_rate), _pct(p.flip_lo), _pct(p.flip_hi), source=rel)
        T.add(row, "P5 BA null", _pct(p.false_flip_null_oos), source=rel)
    t2 = _csv("top10-exams/t2_p5_scores.csv")
    for ex in ("DrivoR", "WA-JEPA"):
        e = t2[(t2.examinee == ex) & (t2.window == "per frame")].set_index("scope")
        row = f"{ex} | native (zero-shot)"
        T.add(row, "P5 BA ped", _pct(e.loc["pedestrian", "flip"]), _pct(e.loc["pedestrian", "lo"]), _pct(e.loc["pedestrian", "hi"]),
              source="top10-exams/t2_p5_scores.csv", note="black rear camera (adapter compromise)")
        T.add(row, "P5 BA cut-in", _pct(e.loc["cut-in", "flip"]), source="top10-exams/t2_p5_scores.csv")
        T.add(row, "P5 BA pooled", _pct(e.loc["pooled", "flip"]), _pct(e.loc["pooled", "lo"]), _pct(e.loc["pooled", "hi"]),
              source="top10-exams/t2_p5_scores.csv")
        T.add(row, "P5 BA null", _pct(e.loc["pooled", "null_ff_oos"]), source="top10-exams/t2_p5_scores.csv")


# ---------------------------------------------------------------- I3

def i3(T: Table):
    f = _csv("elicitation/i3/flip_rates.csv")
    f = f[f.scope == "pooled"].set_index("examinee")
    names = {"ridge ego": "ego-only | ridge ego", "ridge_late L18_last": "Qwen3-VL-4B | ridge_late"}
    for m in ("cinque", "lebowski"):
        short = "Cinque" if m == "cinque" else "Lebowski"
        names.update({f"ridge_late op-{m} temporal": f"{MOD[m]} | ridge_late (= prior)",
                      f"M-C pair [{m}]": f"Qwen3-VL-4B | pair-Δ dual = M-C (⊕ {short})",
                      f"M-C pair qwen [{m}]": f"Qwen3-VL-4B | pair-Δ single (prior {short})",
                      f"M-C pair op [{m}]": f"{MOD[m]} | M-C pair op (single)", f"M-C hard [{m}]": f"{MOD[m]} | M-C hard (control)",
                      f"M-C uniform [{m}]": f"{MOD[m]} | M-C uniform (control)"})
    for ex, row in names.items():
        if ex in f.index:
            r = f.loc[ex]
            T.add(row, "I3", _pct(r.flip_rate), _pct(r.flip_lo), _pct(r.flip_hi), source="elicitation/i3/flip_rates.csv",
                  note="single run (5-fold mean head)" + ("; 0 by construction (both sides share the ego)" if ex == "ridge ego" else ""))
            T.add(row, "I3 null", _pct(r.false_flip_null_oos), source="elicitation/i3/flip_rates.csv")
    n3 = _csv("night2/N3/i3_flip_rates.csv")
    n3 = n3[n3.scope == "pooled"]
    for m in ("cinque", "lebowski"):
        for pat, row in ((f"cls_late P5 s{{}} [{m}]", f"{MOD[m]} | cls_late (P5-trained)"),
                         (f"cls_late NAV s{{}} [{m}]", f"{MOD[m]} | cls_late (NAVSIM-trained, zero-shot)")):
            sub = n3[n3.examinee.isin([pat.format(s) for s in (0, 1, 2)])]
            s0 = sub[sub.examinee == pat.format(0)].iloc[0]
            T.add(row, "I3", _pct(sub.flip_rate.mean()), _pct(s0.flip_lo), _pct(s0.flip_hi), 100 * sub.flip_rate,
                  "night2/N3/i3_flip_rates.csv", hint="seed-mean")
        r = n3[n3.examinee == f"ridge_late NAV [{m}]"]
        if len(r):
            T.add(f"{MOD[m]} | ridge_late (NAVSIM-trained, zero-shot)", "I3", _pct(r.flip_rate.iloc[0]), _pct(r.flip_lo.iloc[0]),
                  _pct(r.flip_hi.iloc[0]), source="night2/N3/i3_flip_rates.csv")
    ref = _csv("real-data-transfer/g2/i3_reference.csv")
    for m in ("cinque",):
        for arm in ("A", "B"):
            sub = ref[ref.examinee.isin([f"CARLA student {arm} s{s} [{m}]" for s in (0, 1, 2)])]
            if not len(sub):
                continue
            v = sub["I3 flip"].map(lambda x: float(re.match(r"\s*([-\d.]+)", str(x)).group(1)))
            T.add(f"{MOD[m]} | E5 student {arm}", "I3", float(v.mean()), seeds=v, source="real-data-transfer/g2/i3_reference.csv",
                  hint="seed-mean")
    for rel, sub in (("top10-exams/t1_i3_flip_rates.csv", None), ("top10-exams/t2_i3_flip_rates.csv", "all")):
        df = _csv(rel)
        df = df[df.scope == "pooled"]
        if sub is not None and "subset" in df:
            df = df[df.subset == sub]
        for r in df.itertuples():
            T.add(f"{r.examinee} | native (zero-shot)", "I3", _pct(r.flip_rate), _pct(r.flip_lo), _pct(r.flip_hi), source=rel)
            T.add(f"{r.examinee} | native (zero-shot)", "I3 null", _pct(r.false_flip_null_oos), source=rel)


# ---------------------------------------------------------------- WOD (W-train)

def _dec19(df, arm):
    r = df[(df.arm == arm) & (df.judge == "ADE vs log") & (df.scope == "s_ego deciles 1-9") & (df.subset == "pre_onset")]
    return r.iloc[0] if len(r) else None


def wod(T: Table):
    ht = _csv("driving-backbones/heads-train/heads_rfs_arms_p3drive_heads.csv").set_index("arm")
    src = "driving-backbones/heads-train/heads_rfs_arms_p3drive_heads.csv"
    pre = _csv("driving-backbones/heads-train/pre_onset_dec19_p3drive_heads.csv")
    seeds = {}
    for s in (0, 1, 2):
        fs = sorted(glob.glob(str(data_dir() / f"runs/drive_backbones/heads_train-seed{s}/*/heads_rfs_arms_p3drive_heads.csv")))
        if fs:
            seeds[s] = pd.read_csv(fs[-1]).set_index("arm")
    rows = {"ridge ego": "ego-only | ridge ego", "cls ego K1024": "ego-only | cls ego"}
    for m in ("cinque", "lebowski"):
        rows.update({f"A ridge_late op-{m} temporal": f"{MOD[m]} | ridge_late (= prior)", f"cls_late op-{m} temporal": f"{MOD[m]} | cls_late",
                     f"native op-{m} (no fit)": f"{MOD[m]} | native plan"})
    rows["native op-small (no fit)"] = "openpilot small | native plan"
    for arm, row in rows.items():
        sv = [seeds[s].loc[arm, "rfs_cluster_mean"] for s in seeds if arm in seeds[s].index]
        is_cls = arm.startswith("cls")
        T.add(row, "WOD RFS", float(np.mean(sv)) if is_cls and len(sv) == 3 else ht.loc[arm, "rfs_cluster_mean"],
              seeds=sv if is_cls and len(sv) == 3 else None, source=src + (" + box heads_train-seed{0,1,2}" if is_cls else ""),
              note="cluster mean, W-train" + ("" if is_cls else "; deterministic or no fit"), hint="seed-mean" if is_cls else "")
        r = _dec19(pre, arm)
        if r is not None and arm != "ridge ego":
            T.add(row, "WOD pre-onset ΔADE", r.delta, r.lo, r.hi, source="driving-backbones/heads-train/pre_onset_dec19_p3drive_heads.csv",
                  note="s_ego deciles 1-9 vs ridge ego (decision 22 footnote), n = %d" % r.n)
    tr = _csv("driving-backbones/p3drive_train_arms.csv").set_index("arm")
    tp = _csv("driving-backbones/pre_onset_dec19_p3drive_train.csv")
    a = "A ridge_late pooled (qwen4b L18)"
    T.add("Qwen3-VL-4B | ridge_late", "WOD RFS", tr.loc[a, "rfs_cluster"], tr.loc[a, "rfs_lo"], tr.loc[a, "rfs_hi"],
          source="driving-backbones/p3drive_train_arms.csv", note="W-train arm A = single-frame L18_mean (item 13)")
    r = _dec19(tp, a)
    if r is not None:
        T.add("Qwen3-VL-4B | ridge_late", "WOD pre-onset ΔADE", r.delta, r.lo, r.hi, source="driving-backbones/pre_onset_dec19_p3drive_train.csv",
              note=f"n = {r.n}")
    vt = _csv("p3-backbone-ladder/p3train_arms.csv").set_index("arm")
    vr = _csv("p3-backbone-ladder/rejudge_p3train.csv")
    v = "d vjepa2 (train fit)"
    T.add("V-JEPA 2 ViT-L | ridge_late", "WOD RFS", vt.loc[v, "rfs_cluster"], vt.loc[v, "rfs_lo"], vt.loc[v, "rfs_hi"],
          source="p3-backbone-ladder/p3train_arms.csv", note="W-train, front camera only, n_rater 478 (its ridge ego: %.3f)" % vt.loc["ridge ego", "rfs_cluster"])
    r = _dec19(vr, v)
    if r is not None:
        T.add("V-JEPA 2 ViT-L | ridge_late", "WOD pre-onset ΔADE", r.delta, r.lo, r.hi, source="p3-backbone-ladder/rejudge_p3train.csv",
              note=f"n = {r.n}")
    for bb in ("SigLIP2 so400m", "DINOv2-B", "openpilot small"):
        T.text(f"{bb} | ridge_late", "WOD RFS", "无 W-train（只在 W-half / W-xfit 或从未抽过）")
    rt = _csv("top10-exams/wod_rater.csv").set_index("row")
    for r_, row in (("DrivoR", "DrivoR | native (zero-shot)"), ("WA-JEPA", "WA-JEPA | native (zero-shot)"),
                    ("Alpamayo 1.5 nav (E[1 sample])", "Alpamayo 1.5 | native plan (nav)")):
        if r_ in rt.index:
            T.add(row, "WOD RFS", rt.loc[r_, "rfs"], rt.loc[r_, "rfs_lo"], rt.loc[r_, "rfs_hi"], source="top10-exams/wod_rater.csv",
                  note="cluster mean, zero-shot, 5 s by constant-velocity extrapolation" if r_ != "Alpamayo 1.5 nav (E[1 sample])" else "cluster mean")
    t1 = _csv("top10-exams/t1_wod_rfs.csv").set_index("row")
    for r_ in ("SparseDriveV2", "ZTRS"):
        if r_ in t1.index:
            T.add(f"{r_} | native (zero-shot)", "WOD RFS", t1.loc[r_, "rfs"], t1.loc[r_, "rfs_lo"], t1.loc[r_, "rfs_hi"],
                  source="top10-exams/t1_wod_rfs.csv", note="cluster mean")


def wod_transfer(T: Table):
    """CARLA-trained Delta on real WOD priors: paired RFS frame mean (registered judge) with a recomputed cluster mean."""
    e1 = _csv("elicitation/e1/wod_deltas.csv")
    for m in ("cinque", "lebowski"):
        short = "Cinque" if m == "cinque" else "Lebowski"
        r = e1[(e1.model == m) & (e1.stats == "carla") & (e1.scope == "all") & (e1.judge == "RFS (rater)")].iloc[0]
        T.add(f"Qwen3-VL-4B | pair-Δ dual = M-C (⊕ {short})", "WOD RFS paired Δ", r.delta, r.lo, r.hi,
              source="elicitation/e1/wod_deltas.csv", note="E1, seed 0, frame mean vs WOD-train ridge_late prior")
    g0 = _csv("real-data-transfer/g0/verdict.csv")
    for m in ("cinque", "lebowski"):
        for arm in ("A", "B"):
            sub = g0[(g0.model == m) & (g0.arm == arm)]
            s0 = sub[sub.seed == 0].iloc[0]
            T.add(f"{MOD[m]} | E5 student {arm}", "WOD RFS paired Δ", sub.wod_rfs_all.mean(), s0.wod_rfs_all_lo, s0.wod_rfs_all_hi,
                  sub.wod_rfs_all, "real-data-transfer/g0/verdict.csv", note="G0 zero-shot, frame mean", hint="seed-mean")
            T.add(f"{MOD[m]} | E5 student {arm}", "NAVSIM PDMS paired Δ", sub.nav_pdms_all.mean(), s0.nav_pdms_all_lo, s0.nav_pdms_all_hi,
                  sub.nav_pdms_all, "real-data-transfer/g0/verdict.csv", note="G0 zero-shot vs NAVSIM ridge_late", hint="seed-mean")


def wod_transfer_cluster(T: Table):
    """Recompute: the E1 Delta's paired RFS in cluster mean (the absolute protocol), from the stored per-frame Delta."""
    from . import elicit_e1 as E1
    from .nq3_q6 import _rfs_cluster
    fs = sorted(glob.glob(str(data_dir() / "runs/elicitation/e1-wod/*/wod_delta_cinque.npz")))
    assert fs, "E1 WOD deltas not on disk"
    run = Path(fs[-1]).parent
    d = E1.wod_frames()
    for m in ("cinque", "lebowski"):
        z = np.load(run / f"wod_delta_{m}.npz")
        assert (z["frame_name"] == d["frame_name"]).all()
        c = _rfs_cluster(d, d[f"prior {m}"], d[f"prior {m}"] + z["delta"]).iloc[0]
        short = "Cinque" if m == "cinque" else "Lebowski"
        T.add(f"Qwen3-VL-4B | pair-Δ dual = M-C (⊕ {short})", "WOD RFS paired Δ (cluster)", c.delta, c.lo, c.hi,
              source=str(run), status="recompute", note=f"cluster mean; prior {c.prior_abs:.3f} -> {c.arm_abs:.3f}")


# ---------------------------------------------------------------- NAVSIM

def navsim(T: Table):
    s = _csv("elicitation/seeds/navsim_seeds.csv").set_index(["arm", "metric"])
    src = "elicitation/seeds/navsim_seeds.csv"
    names = {"ridge_ego": "ego-only | ridge ego", "cls_ego_K1024": "ego-only | cls ego"}
    for m in ("cinque", "lebowski"):
        names.update({f"ridge_late_{m}_temporal": f"{MOD[m]} | ridge_late (= prior)", f"cls_late_{m}_temporal": f"{MOD[m]} | cls_late"})
    for arm, row in names.items():
        for met in ("PDMS", "EPDMS"):
            if (arm, met) not in s.index:
                continue
            r = s.loc[(arm, met)]
            sv = [r.s0, r.s1, r.s2]
            T.add(row, f"NAVSIM {met}", float(np.mean(sv)), r.s0 - r.ci_halfwidth_s0 if met != "navhard EPDMS" else np.nan,
                  r.s0 + r.ci_halfwidth_s0, sv, src, note="3 seeds; CI = seed 0 token bootstrap" + ("; 2 Hz protocol" if "op" in row else ""),
                  hint="seed-mean")
    n3 = _csv("night2/N3/navsim_scores.csv")
    for m in ("cinque", "lebowski"):
        for met in ("PDMS", "EPDMS"):
            sub = n3[(n3.metric == met) & n3.row.isin([f"Hydra s{k} [{m}]" for k in (0, 1, 2)])]
            s0 = sub[sub.row == f"Hydra s0 [{m}]"].iloc[0]
            T.add(f"{MOD[m]} | Hydra", f"NAVSIM {met}", sub.score.mean(), s0.lo, s0.hi, sub.score, "night2/N3/navsim_scores.csv",
                  note="3 seeds (N3); openpilot features at 2 Hz", hint="seed-mean")
    z = _csv("navsim-zeroshot/results_navtest.csv")
    for (model, variant), row in ((("openpilot Cinque v3", "none"), "openpilot Cinque | native plan"),
                                  (("openpilot Lebowski", "none"), "openpilot Lebowski | native plan"),
                                  (("openpilot small", "none"), "openpilot small | native plan"),
                                  (("Alpamayo 1.5", "nav"), "Alpamayo 1.5 | native plan (nav)")):
        for met in ("PDMS", "EPDMS"):
            r = z[(z.metric == met) & (z.model == model) & (z.variant == variant)]
            if len(r):
                T.add(row, f"NAVSIM {met}", r.score.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], source="navsim-zeroshot/results_navtest.csv",
                      note="2 Hz sample-and-hold protocol reading (item 14)" if "openpilot" in model else "zero-shot")
    for m in ("cinque", "lebowski"):
        short = "Cinque" if m == "cinque" else "Lebowski"
        e1 = _csv("elicitation/e1/navsim_paired.csv")
        r = e1[(e1.model == m) & (e1.prior == "ridge_late") & (e1.metric == "PDMS") & (e1.group == "all")]
        if len(r):
            T.add(f"Qwen3-VL-4B | pair-Δ dual = M-C (⊕ {short})", "NAVSIM PDMS paired Δ", r.delta.iloc[0], r.lo.iloc[0], r.hi.iloc[0],
                  source="elicitation/e1/navsim_paired.csv", note="E1, seed 0, vs NAVSIM ridge_late")
    for rel, key, met in (("top10-exams/navsim_drivor.csv", "DrivoR", "PDMS"), ("top10-exams/navsim_wajepa.csv", "WA-JEPA", "EPDMS")):
        r = _csv(rel).iloc[0]
        T.add(f"{key} | native (zero-shot)", f"NAVSIM {met}", r.score, source=rel, note=f"reproduction (paper {r.paper}); navtrain-trained, not zero-shot on NAVSIM")


def q6_rows(T: Table):
    """This queue's own rows once they exist: V-JEPA 2 single-frame (P5) and the V-JEPA 2 M-C on real data."""
    p = OUT / "vjepa_single_frame.csv"
    if p.exists():
        d = pd.read_csv(p)
        sub = d[(d.prior == "cinque") & (d.arm_1f == "pair-Δ vjepa2_1f mean [cinque]")]
        s0 = sub[sub.seed == 0].iloc[0]
        T.add("V-JEPA 2 ViT-L (1 frame x 4) | pair-Δ single (prior Cinque)", "P5 BA ped", _pct(sub.ped_1f.mean()), _pct(s0.lo_1f),
              _pct(s0.hi_1f), 100 * sub.ped_1f, str(p), status="recompute", note="Q6 (b)")
        T.add("V-JEPA 2 ViT-L (1 frame x 4) | pair-Δ single (prior Cinque)", "P5 BA null", _pct(sub.null_1f.mean()), seeds=100 * sub.null_1f,
              source=str(p), status="recompute")
    p = OUT / "vjepa_real_verdict.csv"
    if p.exists():
        v = pd.read_csv(p)
        for m in ("cinque", "lebowski"):
            short = "Cinque" if m == "cinque" else "Lebowski"
            sub = v[v.model == m]
            s0 = sub[sub.seed == 0].iloc[0]
            row = f"V-JEPA 2 ViT-L | pair-Δ dual (⊕ {short})"
            T.add(row, "WOD RFS paired Δ", sub.wod_rfs_all.mean(), s0.wod_lo, s0.wod_hi, sub.wod_rfs_all, str(p), status="recompute", note="Q6 (c)")
            T.add(row, "WOD RFS paired Δ (cluster)", sub.wod_rfs_cluster_delta.mean(), s0.wod_rfs_cluster_lo, s0.wod_rfs_cluster_hi,
                  sub.wod_rfs_cluster_delta, str(p), status="recompute", note="Q6 (c)")
            T.add(row, "NAVSIM PDMS paired Δ", sub.nav_pdms_all.mean(), s0.nav_lo, s0.nav_hi, sub.nav_pdms_all, str(p), status="recompute",
                  note="Q6 (c)")


# ---------------------------------------------------------------- output

COLS = ["P5 BA ped", "P5 BA cut-in", "P5 BA pooled", "P5 BA null", "P5 BA A3 ped", "P5 PDM per pair", "I3", "I3 null",
        "WOD RFS", "WOD pre-onset ΔADE", "WOD RFS paired Δ", "WOD RFS paired Δ (cluster)", "NAVSIM PDMS", "NAVSIM EPDMS",
        "NAVSIM PDMS paired Δ", "P6", "closed loop"]


def _fmt(c) -> str:
    if c.status == "text":
        return c.note
    v = f"{c.value:.3f}" if abs(c.value) < 20 and "RFS" in c.column or "ΔADE" in c.column else f"{c.value:.1f}"
    if not np.isnan(c.lo):
        v += f" [{c.lo:.2f}, {c.hi:.2f}]" if "RFS" in c.column or "ΔADE" in c.column else f" [{c.lo:.1f}, {c.hi:.1f}]"
    if c.n_seeds > 1:
        v += f" ({c.n_seeds}s)"
    return v


def run(rl=None) -> dict:
    T = Table()
    for name, fn in (("p5_n6", p5_n6), ("p5_mc_controls", p5_mc_controls), ("p5_e4", p5_e4), ("p5_a3", p5_a3), ("p5_n3", p5_n3),
                     ("p5_students", p5_students), ("p5_top10", p5_top10), ("i3", i3), ("wod", wod), ("wod_transfer", wod_transfer),
                     ("wod_transfer_cluster", wod_transfer_cluster), ("navsim", navsim), ("q6_rows", q6_rows)):
        T.guard(name, lambda fn=fn: fn(T))
    df = pd.DataFrame(T.cells)
    rows = df.row.unique()
    for r in rows:
        for col, txt in (("P6", "待 Q1 / Q2"), ("closed loop", "待 CL")):
            T.text(r, col, txt)
    df = pd.DataFrame(T.cells)
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns="hint").to_csv(OUT / "main_table.csv", index=False, float_format="%.4f")
    wide = df.assign(cell=[_fmt(c) for c in df.itertuples()]).pivot_table(index="row", columns="column", values="cell", aggfunc="first")
    wide = wide[[c for c in COLS if c in wide.columns]]
    (OUT / "main_table.md").write_text(wide.fillna("—").to_markdown())
    diffs = []
    for (row, col), old in OLD.items():
        c = df[(df.row == row) & (df.column == col) & (df.status != "text")]
        new = float(c.value.iloc[0]) if len(c) else np.nan
        tol = 0.0051 if abs(old) < 20 and ("RFS" in col or "ΔADE" in col) else 0.051
        if np.isnan(new):
            reason = "missing-now"
        elif abs(new - old) <= tol:
            reason = "same"
        else:
            reason = c.hint.iloc[0] or "other"
        diffs.append({"row": row, "column": col, "old": old, "new": new, "diff": new - old, "reason_code": reason,
                      "n_seeds": int(c.n_seeds.iloc[0]) if len(c) else 0, "source": c.source.iloc[0] if len(c) else ""})
    old_keys = set(OLD)
    for c in df[df.status != "text"].itertuples():
        if (c.row, c.column) not in old_keys:
            diffs.append({"row": c.row, "column": c.column, "old": np.nan, "new": c.value, "diff": np.nan,
                          "reason_code": "new-cell", "n_seeds": c.n_seeds, "source": c.source})
    pd.DataFrame(diffs).to_csv(OUT / "diff_vs_old.csv", index=False, float_format="%.4f")
    pd.DataFrame(T.missing or [{"source": "", "error": ""}]).to_csv(OUT / "sources_missing.csv", index=False)
    info = {"cells": len(df), "rows": len(rows), "missing_sources": len(T.missing),
            "changed": int(sum(d["reason_code"] not in ("same", "new-cell") for d in diffs))}
    log.info("main table: %s", info)
    return info
