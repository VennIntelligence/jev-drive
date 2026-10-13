#!/usr/bin/env python3
"""Inside-cut share on held turns from an rf_follow.py output (closed loop): a turn rollout is an inside cut when, over its junction-stretch
decisions (entry - 20 m .. exit + 10 m, k <= k_z), the ego's signed lateral offset toward the turn's inside (left of the route for a left
turn, right for a right turn) reaches > THR m. Paired differences A - B over turns (same scene) with a scene-bootstrap 95 % CI.
  rf_inside.py --follow DIR --pairs A:B ... [--thr 1.0 0.5] --out JSON"""
import argparse, json
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--follow", required=True); ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--thr", nargs="+", type=float, default=[1.0, 0.5]); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = pd.read_parquet(f"{a.follow}/decisions.parquet"); S = pd.read_csv(f"{a.follow}/scenes.csv"); S = S[S.turn.astype(str).isin(["True", "1", "1.0"])]
    D = D[D.junction].merge(S[["label", "scene", "side"]], on=["label", "scene"])
    sgn = np.where(D.side.astype(str).str.lower().str.startswith("l") | D.side.astype(str).isin(["1", "1.0"]), 1.0, -1.0)
    D["inw"] = D.lat * sgn
    mx = D.groupby(["label", "scene"]).inw.max().rename("mx").reset_index()
    out, rng = {}, np.random.default_rng(0)
    for t in a.thr:
        mx["cut"] = (mx.mx > t).astype(float)
        out[f"share@{t}"] = {l: dict(n=int(len(g)), share=float(g.cut.mean())) for l, g in mx.groupby("label")}
        for p in a.pairs:
            A, B = p.split(":"); m = mx[mx.label == A].merge(mx[mx.label == B], on="scene", suffixes=("_a", "_b"))
            d = (m.cut_a - m.cut_b).to_numpy(); n = len(d)
            bs = [d[rng.integers(0, n, n)].mean() for _ in range(10000)] if n else [np.nan]
            out.setdefault(f"diff@{t}", {})[p] = dict(n=n, diff=float(d.mean()) if n else None, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
