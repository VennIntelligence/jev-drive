"""TFv6 rules x interface factorial and Bench2Drive by hazard family
(todos/2026-09-25-tfv6-rules-interface/README.md).

  build   route XML for the runs: the 209 Bench2Drive routes that do not crash the server (original ids, each
          tagged p5_suppress=0 so b2d_hooks logs the scenario's hazard actors), plus, for the 45 hazard routes of
          P5 v0, the x- world (hazard hidden) and the weather-only null world as P5 pair variants (seed 0).
          With B2D_RESEED_AFTER_BUILD=1 the Bench2Drive route itself is the x+ world of its pair.
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger
from .p5_pairs import B2D_XML, CRASHERS, _variant

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "tfv6-rules-interface"

# P5 v0's hazard families (Light is R layer, decision 35 / P5 v1), 5 routes each, none of them a crasher.
PAIR_FAMILIES = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian",
                 "ParkingCrossingPedestrian", "OppositeVehicleRunningRedLight", "HardBreakRoute", "StaticCutIn",
                 "ParkingCutIn", "HighwayCutIn")

# Bench2Drive's own multi-ability taxonomy, verbatim from tools/ability_benchmark.py (Bench2Drive 0.0.4); a
# scenario can count for several abilities.
B2D_ABILITY = {
    "Overtaking": ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                   "HazardAtSideLaneTwoWays", "HazardAtSideLane", "ParkedObstacleTwoWays", "ParkedObstacle",
                   "VehicleOpensDoorTwoWays"],
    "Merging": ["CrossingBicycleFlow", "EnterActorFlow", "HighwayExit", "InterurbanActorFlow", "HighwayCutIn",
                "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2", "MergerIntoSlowTraffic",
                "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                "NonSignalizedJunctionLeftTurnEnterFlow", "ParkingExit", "SequentialLaneChange",
                "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow"],
    "Emergency_Brake": ["BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute",
                        "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight", "ParkingCutIn",
                        "PedestrianCrossing", "ParkingCrossingPedestrian", "StaticCutIn", "VehicleTurningRoute",
                        "VehicleTurningRoutePedestrian", "ControlLoss"],
    "Give_Way": ["InvadingTurn", "YieldToEmergencyVehicle"],
    "Traffic_Signs": ["BlockedIntersection", "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight",
                      "PedestrianCrossing", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "EnterActorFlow",
                      "CrossingBicycleFlow", "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                      "NonSignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionLeftTurn",
                      "SignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow", "T_Junction",
                      "VanillaNonSignalizedTurn", "VanillaSignalizedTurnEncounterGreenLight",
                      "VanillaSignalizedTurnEncounterRedLight", "VanillaNonSignalizedTurnEncounterStopsign"],
}

# Our partition (pre-registered): every one of the 44 scenario types in exactly one hazard family. The first
# five are the "sudden hazard" (E layer) families; the rest need planning, negotiation or routine control.
FAMILY = {
    "vru_emerging": ["DynamicObjectCrossing", "ParkingCrossingPedestrian"],          # occluded VRU steps out
    "vru_crossing": ["PedestrianCrossing", "VehicleTurningRoutePedestrian", "VehicleTurningRoute",
                     "CrossingBicycleFlow"],                                          # VRUs crossing at junctions
    "cut_in": ["StaticCutIn", "ParkingCutIn", "HighwayCutIn"],
    "lead_hard_brake": ["HardBreakRoute"],
    "junction_violator": ["OppositeVehicleRunningRedLight", "OppositeVehicleTakingPriority",
                          "BlockedIntersection"],                                     # someone else breaks priority
    "unprotected_turn": ["NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
                         "NonSignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionLeftTurn",
                         "SignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionRightTurn"],
    "merge_lane_change": ["EnterActorFlow", "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "HighwayExit",
                          "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "SequentialLaneChange",
                          "ParkingExit"],
    "obstacle_bypass": ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                        "HazardAtSideLane", "HazardAtSideLaneTwoWays", "ParkedObstacle", "ParkedObstacleTwoWays",
                        "VehicleOpensDoorTwoWays", "InvadingTurn"],
    "emergency_vehicle": ["YieldToEmergencyVehicle"],
    "routine_control": ["T_Junction", "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign",
                        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
                        "ControlLoss"],
}
SUDDEN = ("vru_emerging", "vru_crossing", "cut_in", "lead_hard_brake", "junction_violator")
SCENARIO_FAMILY = {s: f for f, ss in FAMILY.items() for s in ss}


def route_table(src: Path | None = None):
    """route id -> (town, scenario type, family) for all 220 Bench2Drive routes."""
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    rows = []
    for r in root.findall("route"):
        (s,) = r.iter("scenario")
        t = s.get("type")
        rows.append({"route_id": r.get("id"), "town": r.get("town"), "scenario": t, "family": SCENARIO_FAMILY[t],
                     "sudden": SCENARIO_FAMILY[t] in SUDDEN, "crasher": r.get("id") in CRASHERS,
                     "pair": t in PAIR_FAMILIES})
    return pd.DataFrame(rows)


def build(src: Path | None = None) -> Path:
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    out = ET.Element("routes")
    n_route = n_pair = 0
    for r in root.findall("route"):
        if r.get("id") in CRASHERS:
            continue
        (s,) = r.iter("scenario")
        b = copy.deepcopy(r)
        b.set("p5_suppress", "0")                 # log hazard actors (hidden.json); hides nothing
        out.append(b)
        n_route += 1
        if s.get("type") in PAIR_FAMILIES:
            out.append(_variant(r, "minus", 0))
            out.append(_variant(r, "null", 0))
            n_pair += 1
    xml = data_dir() / "runs" / "tfv6_rules" / "routes.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(out)
    ET.ElementTree(out).write(xml)
    RESULTS.mkdir(parents=True, exist_ok=True)
    route_table(src).to_csv(RESULTS / "routes.csv", index=False)
    log.info("%d Bench2Drive routes + %d pairs (x-, null) -> %s", n_route, n_pair, xml)
    return xml


# ---------------------------------------------------------------- results of one runner directory

TICK = 0.05
ARMS = ("A1", "A0", "B1", "B0")
PAIR_WORLD = {1: "plus", 2: "minus", 3: "null"}


def split_id(rid: str) -> tuple[str, str]:
    """Bench2Drive id -> (id, 'plus'); pair variant base*100 + world*10 + seed -> (base, world)."""
    k = int(rid)
    return (rid, "plus") if k < 100000 else (str(k // 100), PAIR_WORLD[(k // 10) % 10])


def _attempt(out: Path, rid: str) -> Path | None:
    import json
    f = out / "done" / f"{rid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return out / "attempts" / rid / str(d.get("attempt", 1))


def route_record(adir: Path) -> dict:
    """Official score and infractions of one attempt (Bench2Drive's records), plus our rule/timing summary."""
    import json
    rec = {"ds": np.nan, "rc": np.nan, "penalty": np.nan, "status": "missing", "success": False}
    f = adir / "results.json"
    if f.exists():
        recs = json.loads(f.read_text())["_checkpoint"]["records"]
        if recs:
            r = recs[-1]
            inf = {k: len(v) for k, v in r["infractions"].items()}
            rec.update(ds=r["scores"]["score_composed"], rc=r["scores"]["score_route"],
                       penalty=r["scores"]["score_penalty"], status=r["status"],
                       success=r["status"] in ("Completed", "Perfect") and
                       not any(n for k, n in inf.items() if k != "min_speed_infractions"),
                       **{"n_" + k: n for k, n in inf.items()})
    f = adir / "rules_summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        rec.update(ticks=s["ticks"], t_trigger=s["t_trigger"], stop=s["stop"],
                   rule_force=s["rule_ticks"]["force"], rule_stop=s["rule_ticks"]["stop"],
                   agent_ms=s["ms_mean"].get("agent"), forward_ms=s["ms_mean"].get("forward"))
    f = adir / "route_result.json"
    if f.exists():
        rr = json.loads(f.read_text())
        rec.update(wall_s=rr.get("wall_s"), tick_ms=(rr.get("profile") or {}).get("total_ms_mean"))
    return rec


def collect(runs: dict[str, Path]) -> pd.DataFrame:
    """One row per (arm label, route variant) over every runner directory: runs = {label: out_dir}."""
    rows = []
    for label, out in runs.items():
        for f in sorted((Path(out) / "done").glob("*.json")):
            rid = f.stem
            base, world = split_id(rid)
            adir = _attempt(Path(out), rid)
            rows.append({"run": label, "arm": label.split(".")[0], "route_id": rid, "base": base, "world": world,
                         "dir": str(adir), **route_record(adir)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pairs (experiment 1)

def load_frames(adir: Path) -> pd.DataFrame:
    f = Path(adir) / "frames.jsonl"
    d = pd.read_json(f, lines=True)
    d["k"] = (d.t / TICK).round().astype(int)
    return d.drop_duplicates("k").set_index("k")


def _events(adir: Path) -> list:
    import json
    f = Path(adir) / "criterion_events.json"
    return json.loads(f.read_text()).get("events", []) if f.exists() else []


def hazard_collision_k(adir: Path, fr: pd.DataFrame, family: str, k_trig: int | None) -> int | None:
    """Tick of x+'s first collision with a hazard actor (HardBreakRoute: any vehicle after the trigger)."""
    import json
    import re
    hz = json.loads((Path(adir) / "hidden.json").read_text()) if (Path(adir) / "hidden.json").exists() else []
    ids = {int(h["id"]) for h in hz}
    f2k = dict(zip(fr.frame, fr.index))
    hits = []
    for e in _events(adir):
        if not e["type"].startswith("COLLISION"):
            continue
        m = re.search(r"id=(\d+)", e.get("message") or "")
        aid = int(m.group(1)) if m else None
        k = f2k.get(e.get("frame"))
        if k is None and e.get("frame") is not None and len(fr):   # frame numbering mismatch: nearest
            k = int(fr.index[np.argmin(np.abs(fr.frame.to_numpy() - e["frame"]))])
        if (aid in ids) or (family == "HardBreakRoute" and e["type"] == "COLLISION_VEHICLE"
                            and k_trig is not None and k is not None and k >= k_trig):
            hits.append(k)
    hits = [h for h in hits if h is not None]
    return min(hits) if hits else None


def divergence_k(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int]:
    """First tick the egos differ by >= 1 cm or >= 0.1 deg (P5's rule), and the last common tick."""
    m = a[["x", "y", "yaw"]].join(b[["x", "y", "yaw"]], rsuffix="_b", how="inner")
    d = np.hypot(m.x - m.x_b, m.y - m.y_b)
    dy = np.abs((m.yaw - m.yaw_b + 180) % 360 - 180)
    bad = np.flatnonzero((d >= 0.01) | (dy >= 0.1))
    last = int(m.index.max())
    return (int(m.index[bad[0]]) if len(bad) else last + 1), last


def pair_stats(plus: Path, other: Path, family: str, window_s: float = 15.0) -> dict:
    """Closed-loop pair quantities of the pre-registration: t_trig, t_div, x+ hazard collision, the reaction
    statistic S = min over the window of v+(t) - v_other(t), rule attribution of the first control divergence,
    and the two channels' readouts on the identical-ego frames before t_div."""
    a, b = load_frames(plus), load_frames(other)
    trig = a.index[a.trig.astype(bool)]
    k_trig = int(trig[0]) if len(trig) else None
    k_div, k_last = divergence_k(a, b)
    k_col = hazard_collision_k(plus, a, family, k_trig)
    out = {"k_trig": k_trig, "k_div": k_div, "k_col": k_col, "k_last": k_last,
           "t_div_rel": None if k_trig is None else (k_div - k_trig) * TICK,
           "t_col_rel": None if (k_trig is None or k_col is None) else (k_col - k_trig) * TICK}
    if k_trig is None:
        out["valid"] = "no_trigger"
        return out
    # README amendment 00:40: TFv6's closed loop is not bit-reproducible, so t_div is descriptive only
    out["valid"] = "ok"
    out["early_div"] = bool(k_div < k_trig)
    k_end = min(k_trig + int(window_s / TICK), k_last, *( [k_col] if k_col is not None else []))
    m = a[["v"]].join(b[["v"]], rsuffix="_o", how="inner").loc[k_trig:k_end]
    dv = (m.v - m.v_o).to_numpy()
    out["S"] = float(dv.min()) if len(dv) else np.nan
    out["S_t_rel"] = float((m.index[int(np.argmin(dv))] - k_trig) * TICK) if len(dv) else np.nan
    out["v_trig"] = float(a.v.get(k_trig, np.nan))
    # first executed-control difference, and whether the pre-post-processor control differed there too
    j = a[["exec", "pre"]].join(b[["exec", "pre"]], rsuffix="_o", how="inner").loc[k_trig:]
    ex = [x != y for x, y in zip(j.exec, j.exec_o)]
    if any(ex):
        k_c = int(j.index[int(np.argmax(ex))])
        out["k_ctrl"] = k_c
        out["ctrl_by_rule"] = bool(j.pre.loc[k_c] == j.pre_o.loc[k_c])
    rule_cols = a.loc[k_trig:k_trig + int(20 / TICK)]
    out["rule_ticks_plus"] = int(rule_cols["rule"].notna().sum()) if "rule" in rule_cols else 0
    # on-policy open-loop readouts: (nearly) identical ego, only the rendered world differs
    jj = a[["x", "y", "v", "ts", "ts_exp", "wp_v2"]].join(b[["x", "y", "v", "ts", "ts_exp", "wp_v2"]], rsuffix="_o",
                                                          how="inner").loc[k_trig:]
    pre = jj[(np.hypot(jj.x - jj.x_o, jj.y - jj.y_o) < 0.1) & ((jj.v - jj.v_o).abs() < 0.1)]
    for c in ("ts", "ts_exp", "wp_v2"):
        d = (pre[c] - pre[c + "_o"]).to_numpy(dtype=float)
        out["pre_n"] = len(d)
        out[f"pre_min_d{c}"] = float(np.nanmin(d)) if len(d) else np.nan
    return out


def pairs(table: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Pair rows (arm x base route): x+ against x- (the reaction) and x+ against null (the noise)."""
    fam = dict(zip(routes.route_id, routes.scenario))
    rows = []
    for (arm, base), g in table[table.base.isin(routes[routes.pair].route_id)].groupby(["arm", "base"]):
        w = {r.world: r for r in g.itertuples()}
        if "plus" not in w:
            continue
        for other in ("minus", "null"):
            if other not in w:
                continue
            try:
                st = pair_stats(Path(w["plus"].dir), Path(w[other].dir), fam[base])
            except (FileNotFoundError, ValueError) as e:
                st = {"valid": f"error: {e}"}
            rows.append({"arm": arm, "base": base, "scenario": fam[base], "vs": other, **st})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def boot_mean(x, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_paired(a, b, n: int = 10000, seed: int = 0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return boot_mean(a[ok] - b[ok], n, seed)


# ---------------------------------------------------------------- experiment 1 summary

def pair_summary(pr: pd.DataFrame, n_split: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per arm: tau from the arm's null, reaction rate RR on x- pairs, in- and out-of-sample null false rate FR,
    hazard collision rate HC, and the rule attribution of the first control divergence."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, g in pr.groupby("arm"):
        nul = g[(g.vs == "null") & g.S.notna()]
        mi = g[(g.vs == "minus") & (g.valid == "ok") & g.S.notna()]
        tau = max(1.0, float(np.percentile(np.abs(nul.S), 95))) if len(nul) else 1.0
        react = (mi.S <= -tau).astype(float).to_numpy()
        fr_in = float((nul.S <= -tau).mean()) if len(nul) else np.nan
        oos = []                                  # tau on one random half of the null routes, measured on the other
        for _ in range(n_split if len(nul) >= 6 else 0):
            idx = rng.permutation(len(nul))
            h1, h2 = nul.S.to_numpy()[idx[: len(idx) // 2]], nul.S.to_numpy()[idx[len(idx) // 2:]]
            for a_, b_ in ((h1, h2), (h2, h1)):
                oos.append(float((b_ <= -max(1.0, np.percentile(np.abs(a_), 95))).mean()))
        allp = g[(g.vs == "minus")]
        hc = allp.k_col.notna().astype(float).to_numpy()
        rr = boot_mean(react)
        rows.append({"arm": arm, "n_pairs": len(allp), "n_valid": len(mi), "n_null": len(nul), "tau": tau,
                     "RR": rr[0], "RR_lo": rr[1], "RR_hi": rr[2], "FR_in": fr_in,
                     "FR_oos": float(np.mean(oos)) if oos else np.nan,
                     "HC": boot_mean(hc)[0], "HC_lo": boot_mean(hc)[1], "HC_hi": boot_mean(hc)[2],
                     "ctrl_div": int(mi.get("k_ctrl", pd.Series(dtype=float)).notna().sum()),
                     "ctrl_div_by_rule": int(mi.get("ctrl_by_rule", pd.Series(dtype=object)).fillna(False)
                                             .astype(bool).sum()),
                     "pairs_with_rule_ticks": int((mi.rule_ticks_plus > 0).sum()) if "rule_ticks_plus" in mi else 0,
                     "median_t_div_rel": float(mi.t_div_rel.median()) if len(mi) else np.nan})
    return pd.DataFrame(rows)


def pair_contrasts(pr: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Paired arm contrasts on the same routes: reaction (S <= -tau_arm) and hazard collision indicators."""
    tau = dict(zip(summ.arm, summ.tau))
    m = pr[pr.vs == "minus"].copy()
    m["react"] = np.where((m.valid == "ok") & m.S.notna(), (m.S <= -m.arm.map(tau)).astype(float), np.nan)
    m["hc"] = m.k_col.notna().astype(float)
    rows = []
    for a, b, what in (("A0", "A1", "rules off - on, interface A"), ("B0", "B1", "rules off - on, interface B"),
                       ("A1", "B1", "interface A - B, rules on"), ("A0", "B0", "interface A - B, rules off")):
        x, y = m[m.arm == a].set_index("base"), m[m.arm == b].set_index("base")
        j = x[["react", "hc"]].join(y[["react", "hc"]], rsuffix="_b", how="inner")
        for k in ("react", "hc"):
            d = boot_paired(j[k], j[k + "_b"])
            rows.append({"contrast": what, "metric": "RR" if k == "react" else "HC", "n": int(j[[k, k + "_b"]]
                         .notna().all(1).sum()), "diff": d[0], "lo": d[1], "hi": d[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment 6 and the 2 x 2 on routes

def load_public(path: Path, label: str) -> pd.DataFrame:
    """A Bench2Drive merged results json (records with route_id 'RouteScenario_<id>_rep<k>') -> route rows."""
    import json
    import re
    d = json.loads(Path(path).read_text())
    recs = d.get("_checkpoint", d).get("records", d.get("records", []))
    rows = []
    for r in recs:
        m = re.search(r"_(\d+)_rep(\d+)", r["route_id"])
        inf = {k: len(v) for k, v in (r.get("infractions") or {}).items()}
        rows.append({"run": label, "route_id": m.group(1), "rep": int(m.group(2)),
                     "ds": float(r["scores"]["score_composed"]), "rc": float(r["scores"]["score_route"]),
                     "success": r.get("status") in ("Completed", "Perfect") and
                     not any(n for k, n in inf.items() if k != "min_speed_infractions")})
    return pd.DataFrame(rows)


def family_table(routes: pd.DataFrame, runs: pd.DataFrame, only: set | None = None) -> pd.DataFrame:
    """Per run x family (ours and Bench2Drive's abilities): mean DS and SR with route-bootstrap CIs."""
    r = runs.merge(routes, on="route_id")
    if only is not None:
        r = r[r.route_id.isin(only)]
    groups = [("all", lambda x: x)] + [("sudden", lambda x: x[x.sudden])] + \
        [(f, (lambda f: lambda x: x[x.family == f])(f)) for f in FAMILY] + \
        [("ability:" + a, (lambda ss: lambda x: x[x.scenario.isin(ss)])(ss)) for a, ss in B2D_ABILITY.items()]
    rows = []
    for run, g in r.groupby("run"):
        for name, sel in groups:
            h = sel(g)
            ds, sr = boot_mean(h.ds), boot_mean(h.success.astype(float) * 100)
            rows.append({"run": run, "family": name, "n": len(h), "DS": ds[0], "DS_lo": ds[1], "DS_hi": ds[2],
                         "SR": sr[0], "SR_lo": sr[1], "SR_hi": sr[2]})
    return pd.DataFrame(rows)


def run_noise(a: pd.DataFrame, b: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run noise from two identical runs: per-route variance d^2/2, SD of a single run's mean."""
    j = a.set_index("route_id")[["ds", "success"]].join(b.set_index("route_id")[["ds", "success"]], rsuffix="_b",
                                                        how="inner").join(routes.set_index("route_id")[["family",
                                                                                                         "sudden"]])
    j["d"] = j.ds - j.ds_b
    out = []
    for name, h in [("all", j), ("sudden", j[j.sudden])] + [(f, j[j.family == f]) for f in FAMILY]:
        if not len(h):
            continue
        out.append({"family": name, "n": len(h), "identical": float((h.d == 0).mean()),
                    "mean_abs_d": float(h.d.abs().mean()), "max_abs_d": float(h.d.abs().max()),
                    "mean_diff": float(h.d.mean()),
                    "sd_single_run_mean": float(np.sqrt((h.d ** 2 / 2).sum()) / len(h)),
                    "sr_flips": int((h.success != h.success_b).sum())})
    return pd.DataFrame(out)


def factorial(runs: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Main effects and interaction of interface and rules on DS / SR, paired by route (cluster bootstrap)."""
    w = {k: runs[runs.run == v].set_index("route_id") for k, v in labels.items()}
    common = sorted(set.intersection(*(set(x.index) for x in w.values())))
    rows = []
    for metric in ("ds", "success"):
        v = {k: x.loc[common, metric].astype(float).to_numpy() * (100 if metric == "success" else 1)
             for k, x in w.items()}
        eff = {"interface (A - B)": (v["A1"] + v["A0"] - v["B1"] - v["B0"]) / 2,
               "rules (on - off)": (v["A1"] + v["B1"] - v["A0"] - v["B0"]) / 2,
               "interaction": (v["A1"] - v["A0"]) - (v["B1"] - v["B0"]),
               "A1 - B1": v["A1"] - v["B1"], "A1 - A0": v["A1"] - v["A0"], "B1 - B0": v["B1"] - v["B0"]}
        for name, d in eff.items():
            m = boot_mean(d)
            rows.append({"metric": "DS" if metric == "ds" else "SR", "effect": name, "n": len(common),
                         "mean": m[0], "lo": m[1], "hi": m[2]})
    return pd.DataFrame(rows)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    a = p.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
