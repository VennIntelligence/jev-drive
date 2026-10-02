"""Exploratory (not in the registered plan): can a probe refine the route's own stop-line estimate, d_cmd - delta, on the closed-loop routes?

  python stoppos_refine.py --out DIR [--taps temporal,vision,hidden,native]

Closed-loop (CL) light approaches only, where the stop line is the logged CARLA truth. Target: the residual d_stop - (d_cmd - delta), delta = median
over the training routes of (d_cmd - d_stop). Ridge on a tap, leave-one-route-out over the CL routes, three fixed alphas (no selection), so the
numbers are a floor on what a tuned refinement could do with 14 junctions. The P4 labels cannot be used: there the stop line is the entrance minus
3.0 m by construction, so the residual is 0 by definition.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_data as D  # noqa: E402
import stoppos_heads as H  # noqa: E402
import stoppos_probe as P  # noqa: E402

ALPHAS = (100.0, 1e3, 1e4)


def main(a):
    out = Path(a.out)
    frames = D.load_frames()
    have = D.have_features(frames)
    cl = (frames.src == "cl").to_numpy() & have
    M = cl & frames.m_stop.to_numpy() & np.isfinite(frames.d_cmd.to_numpy())
    ids = sorted(set(frames.id[M]))
    rows = []
    for tap in a.taps.split(","):
        X = D.load_feats(frames, tap)
        pred = {al: np.full(len(frames), np.nan) for al in ALPHAS}
        base = np.full(len(frames), np.nan)
        for rid in ids:
            tr_rows = cl & (frames.id != rid).to_numpy()
            te = np.flatnonzero(M & (frames.id == rid).to_numpy())
            tr = np.flatnonzero(M & (frames.id != rid).to_numpy())
            fr = frames.iloc[tr]
            delta = float((fr.d_cmd - fr.d_stop).groupby(fr.id).median().median())
            res = (frames.d_stop - (frames.d_cmd - delta)).to_numpy()
            Z = P.make_features(tap, X, tr_rows, 0, lambda m: None)
            w = D.attempt_weights(frames, M & (frames.id != rid).to_numpy())
            fits = H.ridge_fit(Z[tr], res[tr], w[tr], alphas=ALPHAS)
            base[te] = (frames.d_cmd - delta).to_numpy()[te]
            for al, f in zip(ALPHAS, fits):
                pred[al][te] = base[te] + H.ridge_predict(Z[te], f)
        d0 = frames[M & (frames.d_stop >= 0).to_numpy()]
        for name, p in [("route command - delta", base)] + [(f"route + {tap} residual (alpha {al:g})", pred[al]) for al in ALPHAS]:
            d = d0.assign(err=p[d0.index] - d0.d_stop)
            rows += D.bin_rows(d, "d_stop", "err", dict(tap=tap, model=name))
        del X
    pd.DataFrame(rows).to_csv(out / "refine_cl.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", default="temporal,vision,hidden,native")
    main(ap.parse_args())
