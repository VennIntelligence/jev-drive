"""Box step 3 (third_party/alpasim/.venv): per-rollout metric series (metrics.parquet) for the same scenes -> /tmp/accread_ser.json."""
import json, glob
import numpy as np, polars as pl
O = "/root/autodl-tmp/ujs/runs/route_follow/pai_g10"
want = set(json.load(open("/tmp/accread.json")))
out = {}
for lab in ("g10", "acc"):
    for s in want:
        for f in glob.glob(f"{O}/{lab}/runs/*/sim/rollouts/{s}/*/metrics.parquet"):
            m = pl.read_parquet(f).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
            d = {}
            for (n,), x in m.group_by("name"):
                t = np.array(x["timestamps_us"].to_list(), float); v = np.array([np.nan if y is None else float(y) for y in x["values"].to_list()])
                o = np.argsort(t); d[n] = [t[o].tolist(), v[o].tolist()]
            out.setdefault(s, {})[lab] = d
json.dump(out, open("/tmp/accread_ser.json", "w"))
print(len(out), sorted(next(iter(out.values()))["g10"].keys()))
