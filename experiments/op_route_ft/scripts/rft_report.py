"""Open-loop table of op_route_ft from $R/evalol_<set>/<model>.json -> experiments/op_route_ft/results/openloop.{md,json} (envs/jevdrive or any python).

    python experiments/op_route_ft/scripts/rft_report.py --models O rc-bear-s0 rc-poly-s0 rc-ctl-s0 rc-all-s0 [--set ol]
"""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROWS = [  # key, label, field, line
    ("carla_exit_row", "CARLA dev exits: correct per (pose, exit) row", "mean", ">= 0.8"),
    ("carla_exit_reached", "  among rows whose plan reaches d + 12 m", "mean", ""),
    ("carla_exit_short", "  share of rows whose plan stops short", None, ""),
    ("carla_exit_left", "  left commands", "mean", ""),
    ("carla_exit_straight", "  straight commands", "mean", ""),
    ("carla_exit_right", "  right commands", "mean", ""),
    ("carla_exit_3pose_all", "  3-exit poses with all three right", "mean", ""),
    ("carla_turn_lat", "  signed lateral at d + 10 m, turn rows (m)", "mean", ""),
    ("carla_action_sign_turn", "  action[0] sign = command side, turn rows", "mean", ""),
    ("drift_carla_pose", "no-command drift |dy(4 s)|, CARLA dev poses (median, m)", "median", "<= 0.10"),
    ("drift_nav_junction", "  navtrain dev, turn within 60 m", "median", "<= 0.10"),
    ("drift_nav_straight", "  navtrain dev, straight", "median", "<= 0.10"),
    ("drift_wod_junction", "  WOD dev, turn within 60 m", "median", "<= 0.10"),
    ("drift_wod_straight", "  WOD dev, straight", "median", "<= 0.10"),
    ("neg_carla_neg", "negative offset |dy(4 s)| vs original, CARLA N1 (mean, m)", "mean", "<= 0.3"),
    ("neg_nav_neg", "  navtrain screened N1-N4", "mean", "<= 0.3"),
    ("neg_nav_neg_N1_exit", "    N1 exit", "mean", ""),
    ("neg_nav_neg_N2_side", "    N2 side", "mean", ""),
    ("neg_nav_neg_N3_wrong", "    N3 wrong side", "mean", ""),
    ("neg_nav_neg_N4_uturn", "    N4 u-turn", "mean", ""),
    ("neg_wod_neg", "  WOD online N3 / N4", "mean", "<= 0.3"),
]


def fmt(v):
    if isinstance(v, dict):
        m = v.get("mean")
        if m is None:
            return "-"
        s = f"{m:.3f}"
        if v.get("lo") is not None:
            s += f" [{v['lo']:.2f}, {v['hi']:.2f}]"
        return s + f" (n {v.get('n')})"
    return "-" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--set", default="ol")
    ap.add_argument("--root", default=os.path.join(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"), "runs/op_route_ft"))
    a = ap.parse_args()
    R = {m: json.loads((Path(a.root) / f"evalol_{a.set}" / f"{m}.json").read_text()) for m in a.models}
    md = ["| readout | line | " + " | ".join(a.models) + " |", "|---|---|" + "---:|" * len(a.models)]
    for key, label, fld, line in ROWS:
        cells = []
        for m in a.models:
            v = R[m].get(key)
            if v is None:
                cells.append("-")
            elif fld == "median":
                cells.append(f"{v['median']:.3f} (n {v['n']})" if v.get("median") is not None else "-")
            elif fld is None:
                cells.append(fmt(v))
            else:
                cells.append(fmt(v) if "lo" in v else (f"{v['mean']:.3f} (n {v['n']})" if v.get("mean") is not None else "-"))
        md.append(f"| {label} | {line} | " + " | ".join(cells) + " |")
    for d in ("nav", "wod"):
        cells = []
        for m in a.models:
            u = R[m].get(f"uptake_{d}")
            cells.append("-" if not u else f"{u['err_model']:.2f} (orig {u['err_orig']:.2f}, n {u['n']})")
        md.append(f"| real turn frames, {d}: |y(4 s) - logged| with the route (m) | report | " + " | ".join(cells) + " |")
    out = REPO / "experiments/op_route_ft/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "openloop.md").write_text("\n".join(md) + "\n\nCI: 95% cluster bootstrap (junction / log), 2000 resamples. Lines from plans/2026-10-05-route-ft-prereg.md.\n")
    (out / "openloop.json").write_text(json.dumps(R, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
