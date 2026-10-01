"""Result processing, route-cluster bootstrap, and summary tables for VLM arbitration.

Protocol: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]


def bootstrap_delta(df_routes: pd.DataFrame, arm_a: str, arm_b: str, routes: List[str], n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Route-cluster bootstrap paired difference: mean over seeds per route, then resample routes."""
    # Filter to specified routes
    sub = df_routes[df_routes.route.isin(routes)]
    
    # Compute per-route mean for arm_a and arm_b
    mean_a = sub[sub.arm == arm_a].groupby("route")["DS"].mean()
    mean_b = sub[sub.arm == arm_b].groupby("route")["DS"].mean()
    
    common_routes = sorted(list(set(mean_a.index) & set(mean_b.index)))
    if not common_routes:
        return {"arm_a": arm_a, "arm_b": arm_b, "n_routes": 0, "delta_mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    diffs = np.array([mean_a[r] - mean_b[r] for r in common_routes])
    n = len(diffs)
    
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[boot_indices].mean(axis=1)
    
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    delta_mean = float(diffs.mean())

    return {
        "contrast": f"{arm_a} - {arm_b}",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_routes": n,
        "mean_a": float(mean_a[common_routes].mean()),
        "mean_b": float(mean_b[common_routes].mean()),
        "delta_mean": delta_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_str": f"{delta_mean:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"
    }


def parse_route_results(run_root: Path) -> pd.DataFrame:
    """Read all route attempt results from the run directory."""
    records = []
    # Pattern: arms/<tag>-<arm>-s<seed>/attempts/<route>/<attempt>/result.json
    # or done/<route>.json
    for done_file in run_root.glob("arms/*/done/*.json"):
        try:
            arm_dir = done_file.parents[1]
            arm_name = arm_dir.name # e.g. eval-vred-s0
            parts = arm_name.split("-")
            arm = parts[-2]
            seed = int(parts[-1].replace("s", ""))
            
            done_info = json.loads(done_file.read_text())
            route_id = done_info.get("route_id", done_file.stem)
            attempt_idx = str(done_info.get("attempt", 1))
            
            attempt_dir = arm_dir / "attempts" / route_id / attempt_idx
            res_file = attempt_dir / "result.json"
            
            if res_file.exists():
                res = json.loads(res_file.read_text())
                score = res.get("scores", {})
                infractions = res.get("infractions", {})
                records.append({
                    "arm_dir": str(arm_dir),
                    "arm": arm,
                    "seed": seed,
                    "route": str(route_id),
                    "attempt": str(attempt_dir),
                    "DS": float(score.get("score_composed", 0.0)),
                    "RC": float(score.get("score_route", 0.0)),
                    "red_infractions": len(infractions.get("red_light", [])),
                    "stop_infractions": len(infractions.get("stop_sign", [])),
                    "collision_veh": len(infractions.get("collisions_vehicle", [])),
                    "collision_layout": len(infractions.get("collisions_layout", [])),
                    "collision_ped": len(infractions.get("collisions_pedestrian", [])),
                    "blocked": 1 if infractions.get("blocked") else 0,
                    "timeout": 1 if infractions.get("route_timeout") else 0,
                })
        except Exception as e:
            pass

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate VLM arbitration reports")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing arms/")
    parser.add_argument("--out-dir", type=str, default="experiments/vlm_arb/results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_dir)

    df_routes = parse_route_results(run_root)
    if df_routes.empty:
        print("No route records found in", run_root)
        return

    df_routes.to_csv(out_dir / "routes_raw.csv", index=False)
    print(f"Loaded {len(df_routes)} route evaluations.")

    # Arms summary
    summary = df_routes.groupby("arm").agg(
        runs=("route", "count"),
        DS_mean=("DS", "mean"),
        RC_mean=("RC", "mean"),
        red_lights=("red_infractions", "sum"),
        stop_signs=("stop_infractions", "sum"),
        collisions_veh=("collision_veh", "sum"),
        collisions_layout=("collision_layout", "sum"),
        blocked=("blocked", "sum"),
    ).reset_index()
    summary.to_csv(out_dir / "arm_summary.csv", index=False)
    (out_dir / "arm_summary.md").write_text(summary.to_markdown(index=False))

    # Contrasts relative to drive
    all_routes = sorted(df_routes.route.unique())
    contrasts = [
        ("dslow", "drive"),
        ("jslow", "drive"),
        ("jslow", "dslow"),
        ("vred", "drive"),
        ("vbyp", "drive"),
        ("vall", "drive"),
    ]
    paired_results = []
    for a, b in contrasts:
        if a in df_routes.arm.values and b in df_routes.arm.values:
            paired_results.append(bootstrap_delta(df_routes, a, b, all_routes))

    df_paired = pd.DataFrame(paired_results)
    df_paired.to_csv(out_dir / "paired_effects.csv", index=False)
    (out_dir / "paired_effects.md").write_text(df_paired.to_markdown(index=False))
    print("Report generated successfully.")


if __name__ == "__main__":
    main()
