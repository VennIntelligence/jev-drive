"""Shared definitions of the vlm_arb lane: route sets, unit registry, readers, route-cluster bootstrap.

Protocol: experiments/vlm_arb/plans/2026-10-02-vlm-arb.md (reduced diagnostic batch, deviation D7).
"""
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DATA / "runs/vlm_arb"
XML = DATA / "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"

DEV = "27043 15102 24944 27870 22535 37969 24497 27297 9196 28147".split()
HELD = ("26828 25783 24416 469 25215 27392 27005 25613 27907 26023 24207 26723 24519 25753 24948 26365 27916 24622 "
        "26370").split()
DEBUG = "334 27787 24721 26872 26537 17749 25169 24955".split()
# Target routes, chosen by scenario type only (first routes in XML order outside DEV / HELD / DEBUG):
RED = ["16390", "15612", "15483"]      # VanillaSignalizedTurnEncounterRedLight (all three that are not banned)
STOP = ["17280", "16529", "16508"]     # VanillaNonSignalizedTurnEncounterStopsign (the only one) + VanillaNonSignalizedTurn x2
OBS = ["19324", "2520", "19832"]       # first of Accident, ConstructionObstacle, ParkedObstacle
TGT = RED + STOP + OBS
ROUTES = DEV + TGT
SHARDS = {"dev": DEV, "tgt": TGT}
SEEDS = (0, 1)
# Route groups of the paired reads, by scenario type
LIGHT_ROUTES = RED + ["27043", "15102", "28147"]     # + the dev routes of a Signalized* type
SIGN_ROUTES = STOP
OBS_ROUTES = OBS + ["24497"]                         # + the dev ConstructionObstacle route
TARGET = {"jslow": ROUTES, "dslow": ROUTES, "pred": LIGHT_ROUTES, "vred": LIGHT_ROUTES + SIGN_ROUTES,
          "pbyp": OBS_ROUTES, "vbyp": OBS_ROUTES, "vall": LIGHT_ROUTES + SIGN_ROUTES + OBS_ROUTES}
# Runs of the first executor, all drive at traffic seed 0 (SEED was never passed, "dslow" had no set speeds):
OLD_DRIVE = ["eval-drive-s0", "eval-drive-s0-rep1", "eval-drive-s0-rep2", "shadow-drive-s0"]
INFRACTIONS = ["collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction",
               "outside_route_lanes", "min_speed_infractions", "yield_emergency_vehicle_infractions",
               "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout"]
N_BOOT, BOOT_SEED = 2000, 0            # registered (plan 4.3); jevdrive.stats defaults to 10000


CONES, VEHICLE = ("ConstructionObstacle",), ("Accident", "ParkedObstacle", "HazardAtSideLane", "VehicleOpensDoor")


def obstacle_kinds():
    """Route id -> "cones" | "vehicle" | "other": the static obstacle its scenario places (type prefix in the route XML)."""
    import xml.etree.ElementTree as ET
    out = {}
    for r in ET.parse(str(XML)).getroot().iter("route"):
        types = [x.get("type", "") for x in r.iter("scenario")]
        out[r.get("id")] = ("cones" if any(t.startswith(CONES) for t in types) else
                            "vehicle" if any(t.startswith(VEHICLE) for t in types) else "other")
    return out


def unit_name(arm, seed, shard):
    return "%s-s%d-%s" % (arm, seed, shard)


def unit_dir(arm, seed, shard):
    return RUN / "arms" / ("v2-" + unit_name(arm, seed, shard))


def is_crash(status):
    """Official statuses carry the failure after 'Failed - ' (decision 82: an exact-match check missed them)."""
    return status == "Failed" or any(m in status for m in (
        "Simulation crashed", "Agent crashed", "Agent couldn't be set up", "Agent's sensors were invalid"))


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def attempt_dir(udir, rid):
    done = Path(udir) / "done" / (rid + ".json")
    if not done.exists():
        return None
    return Path(udir) / "attempts" / rid / str(json.loads(done.read_text()).get("attempt", 1))


def route_row(udir, rid):
    """Official result + mean speed of one finished route, or None."""
    a = attempt_dir(udir, rid)
    if a is None or not (a / "results.json").exists():
        return None
    recs = json.loads((a / "results.json").read_text())["_checkpoint"]["records"]
    if not recs:
        return None
    rec = recs[0]
    row = dict(route=rid, attempt=str(a), status=rec["status"], crash=is_crash(rec["status"]),
               DS=float(rec["scores"]["score_composed"]), RC=float(rec["scores"]["score_route"]),
               game_s=float(rec.get("meta", {}).get("duration_game", np.nan)))
    for k in INFRACTIONS:
        row[k] = len(rec["infractions"].get(k, []))
    row["collisions"] = row["collisions_layout"] + row["collisions_pedestrian"] + row["collisions_vehicle"]
    v, ms = [], []
    if (a / "plans.jsonl").exists():
        with open(a / "plans.jsonl") as f:
            for line in f:
                i = line.find('"v": ')
                if '"warm": false' in line and i >= 0:
                    v.append(float(line[i + 5:line.index(",", i)]))
    row["v_mean"] = float(np.mean(v)) if v else np.nan
    row["finite"] = bool(np.isfinite(v).all()) if v else False
    return row


def drive_dir(rid, seed):
    """The unit that holds the reference drive run of (route, seed): seed 0 reuses the first finished run."""
    if seed == 0 and (RUN / "arms/eval-drive-s0/done" / (rid + ".json")).exists():
        return RUN / "arms/eval-drive-s0"
    return unit_dir("drive", seed, "tgt" if rid in TGT else "dev")


def boot_ratio(num, den, groups, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap of sum(num) / sum(den), resampling groups: (estimate, lo, hi, n groups, sum den)."""
    num, den, groups = np.asarray(num, float), np.asarray(den, float), np.asarray(groups)
    if den.sum() <= 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, groups=0, n=0)
    ids, inv = np.unique(groups, return_inverse=True)
    N, D = np.bincount(inv, num, len(ids)), np.bincount(inv, den, len(ids))
    idx = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = N[idx].sum(1) / D[idx].sum(1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=float(N.sum() / D.sum()), lo=float(lo), hi=float(hi), groups=int((D > 0).sum()), n=int(D.sum()))


def boot_mean(per_group, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI of the mean over independent units (routes)."""
    x = np.asarray(list(per_group), float)
    if len(x) == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, groups=0)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(est=float(x.mean()), lo=float(lo), hi=float(hi), groups=len(x))


def fmt(r, pct=False):
    if r.get("groups", 0) == 0 or not np.isfinite(r["est"]):
        return "n/a"
    k = 100.0 if pct else 1.0
    return ("%.1f%% [%.1f, %.1f]" if pct else "%+.2f [%+.2f, %+.2f]") % (k * r["est"], k * r["lo"], k * r["hi"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")
    tmp.replace(path)


VRED_SHARDS = ("q0", "q1", "q2")


def vred_shards():
    """Routes dealt over the three vred shards by decreasing drive game time (LPT), fixed once in gates/vred_shards.json."""
    p = RUN / "gates/vred_shards.json"
    if not p.exists():
        t = {}
        for rid in ROUTES:
            v = [r["game_s"] for r in (route_row(drive_dir(rid, s), rid) for s in SEEDS) if r]
            t[rid] = sum(v) / len(v) if v else 60.0
        out, load = {k: [] for k in VRED_SHARDS}, {k: 0.0 for k in VRED_SHARDS}
        for rid in sorted(ROUTES, key=lambda r: (-t[r], r)):
            k = min(VRED_SHARDS, key=lambda k: (load[k], k))
            out[k].append(rid)
            load[k] += t[rid]
        write_json(p, out)
    return json.loads(p.read_text())
