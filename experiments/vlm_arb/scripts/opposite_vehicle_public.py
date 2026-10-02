"""Per-model success / DS of public Bench2Drive results on the opposite-vehicle scenario types.

Reads leaderboard_audit's public_routes.csv (one row per method x run x route) and b2d_route_types.csv, and writes
vlm_arb/results/opposite_vehicle_public.csv (long) and prints markdown tables. Routes 9196 etc. used in vlm_arb are
outside the 220-route public set, so the comparison is by scenario type.
"""
import csv, collections, sys, pathlib
R = pathlib.Path(__file__).resolve().parents[2]
PUB = R / "leaderboard_audit/results/b2d-family/public_routes.csv"
TYP = R / "leaderboard_audit/results/b2d-family/public/sc_irt_success_matrix/b2d_route_types.csv"
OUT = R / "vlm_arb/results/opposite_vehicle_public.csv"
TYPES = ["OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight", "YieldToEmergencyVehicle", "InvadingTurn", "VanillaNonSignalizedTurn"]

types = {a: b for a, b in csv.reader(open(TYP))}  # headerless: route_id, scenario
rows = list(csv.DictReader(open(PUB)))
runs = collections.defaultdict(dict)  # (method, run) -> route -> (ds, succ)
for r in rows:
    runs[(r["method"], r["run"])][r["route_id"]] = (float(r["ds"]), r["success"] == "True")
by_method = collections.defaultdict(list)
for (m, run) in runs: by_method[m].append(run)

def stat(m, routes):
    ds = []; sc = []; n = 0
    for run in by_method[m]:
        d = runs[(m, run)]
        for rt in routes:
            if rt in d: ds.append(d[rt][0]); sc.append(d[rt][1])
    return (len(sc), sum(sc), sum(ds) / len(ds)) if sc else (0, 0, float("nan"))

out = []
for T in TYPES:
    routes = sorted(k for k, v in types.items() if v == T)
    print(f"\n### {T}  routes {routes}\n")
    print("| model | runs | route-runs | success | success rate | mean DS |\n|---|---|---|---|---|---|")
    tab = []
    for m in by_method:
        n, s, ds = stat(m, routes)
        if n: tab.append((s / n, m, len(by_method[m]), n, s, ds))
        for run in by_method[m]:
            for rt in routes:
                if rt in runs[(m, run)]:
                    out.append([T, m, run, rt, runs[(m, run)][rt][0], runs[(m, run)][rt][1]])
    for sr, m, nr, n, s, ds in sorted(tab, reverse=True):
        print(f"| {m} | {nr} | {n} | {s} | {sr:.0%} | {ds:.1f} |")
    tot = sum(t[3] for t in tab); ok = sum(t[4] for t in tab)
    print(f"| all methods pooled | | {tot} | {ok} | {ok/tot:.0%} | |")
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "method", "run", "route_id", "ds", "success"]); w.writerows(out)
# per route for the first two types, table of method x route
for T in TYPES[:2]:
    routes = sorted(k for k, v in types.items() if v == T)
    print(f"\n### per-route success (s/DS) {T}\n")
    print("| model | " + " | ".join(routes) + " |\n|---|" + "---|" * len(routes))
    for m in by_method:
        cells = []
        for rt in routes:
            vals = [runs[(m, run)][rt] for run in by_method[m] if rt in runs[(m, run)]]
            cells.append(" ".join(f"{'Y' if s else 'n'}{d:.0f}" for d, s in vals) or "-")
        print(f"| {m} | " + " | ".join(cells) + " |")
