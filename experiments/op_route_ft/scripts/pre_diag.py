"""rc-*-pre diagnosis (op-train env, one leased card): action-head gain on real dev P rows by pose class, against the old target (logged 1 s pure
pursuit) and the turn-in target, for O / rc-ctl / rc-bear / rc-ctl-pre / rc-bear-pre.

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/op_route_ft/scripts/pre_diag.py [--models ...] -> $R/evalol_ol/pre_diag.json
Classes: approach (act_target_pre defined), in turn (route in_turn or a maneuver within v0 * 0.2 + 2 m), straight (no turn within 150 m).
Gain = least-squares slope of action[0] on the old target through the origin (1 = the target's scale); also mean |action| and sign agreement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import rft as F  # noqa: E402
import route_adapter as RA  # noqa: E402
import rft_eval as E  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["O", "rc-ctl-s0", "rc-bear-s0", "rc-ctl-pre-s0", "rc-bear-pre-s0"])
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    sets = {}
    for d in ("nav", "wod"):
        R = F.Real(d)
        t, rt = R.S.t, R.route
        rows = R.S.rows("dev")
        rows = rows[rt["has_route"][rows] & (rt["pmask"][rows].sum(1) >= 3) & (t["v0"][rows] >= 1.0)]
        old, pre, cls, fb = [], [], [], []
        for i in rows:
            tp, tm = F.target_path(rt["poly"][i], rt["pmask"][i])
            v0 = float(t["v0"][i])
            old.append(F.act_target(L.human_targets(t["fut20"][i][None])[0], v0)[0])
            p = F.act_target_pre(tp, tm, v0)
            pre.append(p[0] if p else np.nan)
            strt = not np.isfinite(rt["turn_deg"][i]) and rt["n_turn"][i] == 0 and not rt["in_turn"][i]
            cls.append("approach" if p else ("in turn" if (rt["in_turn"][i] or (np.isfinite(rt["turn_s"][i]) and rt["turn_s"][i] <= v0 * 0.2 + 2)) else
                                              ("straight" if strt else "other")))
            fb.append(RA.features("bear", rt["poly"][i], rt["pmask"][i]))
        sets[d] = (R, rows, np.array(old), np.array(pre), np.array(cls), np.stack(fb).astype(np.float32))
    for name in a.models:
        m = F.load_rmodel(name, dev)
        res = {}
        for d, (R, rows, old, pre, cls, fb) in sets.items():
            _, ac = E.run_rows(m, lambda r: R.B.T[R.normal[r]], rows, fb, R.B.v["slot_valid"][R.normal[rows]], R.S.t["tc"][rows], dev)
            for c in ("approach", "in turn", "straight"):
                mm = cls == c
                if mm.sum() < 5:
                    continue
                x, y = old[mm], ac[mm]
                r = {"n": int(mm.sum()), "gain_old": float((x * y).sum() / max((x * x).sum(), 1e-9)), "abs_action": float(np.abs(y).mean()),
                     "abs_old": float(np.abs(x).mean()), "sign_old": float((np.sign(y) == np.sign(x)).mean())}
                if c == "approach":
                    p = pre[mm]
                    r.update(gain_pre=float((p * y).sum() / max((p * p).sum(), 1e-9)), abs_pre=float(np.abs(p).mean()), corr_pre=float(np.corrcoef(p, y)[0, 1]))
                res[f"{d}/{c}"] = r
        out[name] = res
        print(name, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}), flush=True)
        del m
        torch.cuda.empty_cache()
    p = F.rroot("evalol_ol") / "pre_diag.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
