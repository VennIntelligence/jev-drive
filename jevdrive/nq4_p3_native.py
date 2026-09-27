"""nq4 P3, descriptive only: openpilot's own native plan on the P3 worlds. Never enters the gate or the verdict.

The gate reads `ridge_late` heads on the openpilot `temporal` features (jevdrive.nq4_p3 exam). Here the models' native
plan from the same modeld forward (scripts/p5_openpilot.py --arrays temporal plan --out-sub op_streams_plan) is read
on the same frames with the same flip definition: the 2 s longitudinal speed v2, null = x+ vs real, pair = x+ vs x-,
flip = |delta| >= tau and |delta| > 0. There is no native-plan tau calibrated on the I3 null, so tau is borrowed from
each model's I3 `ridge_late` examinee (labelled as borrowed), and the raw |delta| quantiles are reported with it.

Before any number is read, every recomputed `temporal` array must equal the gate's stored op_streams bit for bit
(same names, same targets); the check table is written next to the readout.

  P5_SET=nq4_p3 python -m jevdrive.nq4_p3_native --out $DATA_DIR/runs/nq4/p3/<tag>/native
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger
from .nq4_p3 import I3_EXAM, root
from .p5_exam import boot_ratio
from .p5_pairs import v2

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")


def identity(streams: list) -> pd.DataFrame:
    """Recomputed `temporal` (op_streams_plan) against the gate's stored streams (op_streams), bitwise."""
    rows = []
    for m in MODELS:
        for s in streams:
            with np.load(root("op_streams_plan", m) / f"{s}.npz") as a, np.load(root("op_streams", m) / f"{s}.npz") as b:
                same = (np.array_equal(a["name"], b["name"]) and np.array_equal(a["hist"], b["hist"])
                        and a["temporal"].dtype == b["temporal"].dtype and a["temporal"].shape == b["temporal"].shape
                        and np.array_equal(a["temporal"].view(np.uint8), b["temporal"].view(np.uint8)))
                rows.append({"model": m, "stream": s, "rows": len(a["name"]), "bit_identical": bool(same),
                             "max_abs_diff": float(np.abs(a["temporal"].astype(np.float64) - b["temporal"]).max())
                             if a["temporal"].shape == b["temporal"].shape else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root() / "op_plan.json").read_text())
    streams = [s["key"] for s in plan["streams"]]
    idt = identity(streams)
    idt.to_csv(out / "native_identity.csv", index=False)
    assert idt.bit_identical.all(), f"temporal differs from the stored gate streams:\n{idt[~idt.bit_identical]}"
    log.info("identity: %d streams x %d models bit-identical", len(streams), len(MODELS))

    t = pd.read_parquet(root() / "index.parquet")[["frame_name", "base_id", "world", "t_rel_f0"]]
    t["sfx"] = t.frame_name.str.rsplit("-", n=1).str[-1]
    for m in MODELS:
        parts = [np.load(root("op_streams_plan", m) / f"{s}.npz") for s in streams]
        names = np.concatenate([p["name"] for p in parts]).astype(str)
        speed = pd.Series(v2(np.concatenate([p["plan"] for p in parts]).astype(np.float64)), index=names)
        t[f"{m}|v2"] = t.frame_name.map(speed)
        assert t[f"{m}|v2"].notna().all(), f"{m}: frames without a native plan"
    w = t.pivot_table(index=["base_id", "sfx"], columns="world", values=[f"{m}|v2" for m in MODELS] + ["t_rel_f0"])
    w.columns = [f"{x}_{y}" for x, y in w.columns]
    w = w.rename(columns={"t_rel_f0_real": "t_rel_f0"}).reset_index()
    taus = pd.read_csv(data_dir() / I3_EXAM / "flip_rates.csv").query("scope == 'pooled'").set_index("examinee").tau_model
    rows = []
    for m in MODELS:
        tau = float(taus[f"ridge_late op-{m} temporal"])
        w[f"{m}|null_d"] = w[f"{m}|v2_plus"] - w[f"{m}|v2_real"]
        w[f"{m}|pair_d"] = w[f"{m}|v2_plus"] - w[f"{m}|v2_minus"]
        for scope, g in [("pooled", w), *w.groupby("base_id")]:
            r = {"model": f"native plan op-{m}", "scope": scope, "tau": tau, "tau_source": f"borrowed: I3 ridge_late op-{m} temporal",
                 "n_frames": len(g), "v2_real_median": float(g[f"{m}|v2_real"].median())}
            for kind in ("null", "pair"):
                d = g[f"{m}|{kind}_d"].abs().to_numpy()
                flip = ((d >= tau) & (d > 0)).astype(float)
                r |= {f"{kind}_abs_median": float(np.median(d)), f"{kind}_abs_p90": float(np.quantile(d, 0.9)),
                      f"{kind}_abs_p95": float(np.quantile(d, 0.95)), f"{kind}_abs_max": float(d.max())}
                if scope == "pooled":
                    fr, lo, hi = boot_ratio(flip, np.ones(len(flip)), g.base_id.to_numpy())
                    r |= {f"{kind}_flip": fr, f"{kind}_flip_lo": lo, f"{kind}_flip_hi": hi}
                else:
                    r[f"{kind}_flip"] = float(flip.mean())
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "native_plan.csv", index=False)
    w.to_parquet(out / "native_frames.parquet", index=False)
    (out / "native_plan.md").write_text("Descriptive only; not part of the P3 gate or verdict. tau borrowed from I3 ridge_late.\n\n"
                                        + res.to_markdown(index=False, floatfmt=".3f") + "\n")
    log.info("native plan (descriptive):\n%s", res[res.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
