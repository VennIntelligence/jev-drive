"""P5 v0: counterfactual scenario pairs in CARLA and an open-loop exam on them
(todos/2026-09-24-p5-carla-pairs-v0.md).

  build     write the variant route XML (x+ as is, x- with the hazard actors suppressed (HardBreakRoute: the
            scenario dropped; Light: red swapped for green), a weather-only null) and the case table; each variant is its own numeric route id
            base_id * 100 + world * 10 + tm_seed, world 1 = plus, 2 = minus, 3 = null

The recorder is scripts/p5_pair_agent.py, run through scripts/b2d_run.py in envs/scout-tfv6.
"""
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "p5-carla-pairs"
B2D_XML = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
RED, GREEN = "VanillaSignalizedTurnEncounterRedLight", "VanillaSignalizedTurnEncounterGreenLight"
FAMILIES = ("PedestrianCrossing", "DynamicObjectCrossing", "VehicleTurningRoutePedestrian",
            "ParkingCrossingPedestrian", "OppositeVehicleRunningRedLight", "HardBreakRoute", "StaticCutIn",
            "ParkingCutIn", "HighwayCutIn", "Light")
WORLDS = {"plus": 1, "minus": 2, "null": 3}
SEEDS = (0, 1, 2)
# docs/carla.md: every one of these crashed the server three times in the 220-route round
CRASHERS = {"3048", "11715", "11755", "23687", "23708", "3785", "3800", "23670", "23695", "24041", "24071"}
NIGHT = dict(cloudiness=30.0, fog_density=2.0, precipitation=0.0, precipitation_deposits=0.0,
             sun_altitude_angle=-60.0, sun_azimuth_angle=-1.0, wetness=0.0, wind_intensity=10.0)
NOON = dict(cloudiness=5.0, fog_density=2.0, precipitation=0.0, precipitation_deposits=0.0,
            sun_altitude_angle=60.0, sun_azimuth_angle=-1.0, wetness=0.0, wind_intensity=10.0)


def runs_dir(*parts) -> Path:
    p = data_dir() / "runs" / "p5_pairs" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def variant_id(base: str, world: str, seed: int) -> str:
    return str(int(base) * 100 + WORLDS[world] * 10 + seed)


def parse_id(rid: str) -> tuple[str, str, int]:
    k = int(rid)
    return str(k // 100), {v: w for w, v in WORLDS.items()}[(k // 10) % 10], k % 10


def _family(stype: str) -> str:
    return "Light" if stype in (RED, GREEN) else stype


def _variant(route: ET.Element, world: str, seed: int) -> ET.Element:
    """One variant of a single-scenario route: the XML is the only thing that differs between worlds."""
    r = copy.deepcopy(route)
    r.set("id", variant_id(route.get("id"), world, seed))
    sc = r.find("scenarios")
    (s,) = sc.findall("scenario")
    stype = s.get("type")
    if stype in (RED, GREEN):                    # plus = red light, minus = green light, whatever the original was
        want = RED if world in ("plus", "null") else GREEN
        s.set("type", want)
        s.set("name", s.get("name").replace(stype, want))
    elif world == "minus" and stype == "HardBreakRoute":
        # The factor is the background's hard brake itself, so x- drops the scenario. The evaluator names the
        # route after its first scenario config, so the element stays, with a trigger point 10 km away that
        # RouteScenario._filter_scenarios drops ("too far from the route").
        tp = s.find("trigger_point")
        tp.set("x", str(float(tp.get("x")) + 10000.0))
        s.set("name", s.get("name") + "_removed")
    else:
        # Every other scenario also commands the background (clear the junction, leave space), so deleting it
        # changes the background too; x- keeps it and b2d_hooks.track_hazards hides its hazard actors (and
        # names them in every world, so x+ and x- can be matched).
        r.set("p5_suppress", "1" if world == "minus" else "0")
    if world == "null":
        ws = r.find("weathers")
        day = float(ws.find("weather").get("sun_altitude_angle")) > 0
        for w in ws.findall("weather"):
            for k, v in (NIGHT if day else NOON).items():
                w.set(k, str(v))
    return r


def build(src: Path | None = None) -> pd.DataFrame:
    """Case table (one row per base route x seed) and the variant XML with every world of every case."""
    root = ET.parse(src or data_dir() / B2D_XML).getroot()
    out, rows = ET.Element("routes"), []
    for route in root.findall("route"):
        (s,) = route.iter("scenario")
        fam = _family(s.get("type"))
        rid = route.get("id")
        if fam not in FAMILIES or rid in CRASHERS:
            continue
        for seed in SEEDS:
            for world in WORLDS:
                if world == "null" and seed != 0:
                    continue
                out.append(_variant(route, world, seed))
            rows.append({"base_id": rid, "town": route.get("town"), "family": fam, "orig_type": s.get("type"),
                         "seed": seed, "plus": variant_id(rid, "plus", seed), "minus": variant_id(rid, "minus", seed),
                         "null": variant_id(rid, "null", 0) if seed == 0 else ""})
    cases = pd.DataFrame(rows)
    xml = runs_dir() / "pairs.xml"
    ET.indent(out)
    ET.ElementTree(out).write(xml)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases.to_csv(RESULTS / "cases.csv", index=False)
    log.info("%d cases (%d base routes) -> %s; per family %s", len(cases), cases.base_id.nunique(), xml,
             cases.groupby("family").size().to_dict())
    return cases


# ---------------------------------------------------------------- one recorded world

TICK = 0.05
CAM_TICKS = 4
DIV_M, DIV_DEG = 0.01, 0.1                 # ego divergence: position >= 1 cm or heading >= 0.1 deg
ACTOR_DIFF_M = 0.1                         # a matched actor "differs" between worlds beyond this
PX_ACTOR, PX_LIGHT = 20, 10               # visible pixels in the half-resolution segmentation view


def attempt(gen: Path, rid: str) -> Path | None:
    """The finished attempt of a route variant (the one that wrote p5_summary.json and the most pose lines)."""
    best, n = None, -1
    for a in sorted((gen / "attempts" / rid).glob("*")):
        f = a / "pose.jsonl"
        if not (a / "p5_summary.json").exists() or not f.exists():
            continue
        k = sum(1 for _ in open(f))
        if k > n:
            best, n = a, k
    return best


def load_world(adir: Path) -> dict:
    """Everything a pair needs from one attempt, keyed by the tick index k = round(t / 0.05). Frame numbers are
    the server's and differ between runs; actor ids differ by a constant offset (the hero's id) when the same
    actors were spawned in the same order, which is the case for the worlds of one pair."""
    import json
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    pose["k"] = (pose.t / TICK).round().astype(int)
    f2k = dict(zip(pose.frame, pose.k))
    frames = pd.read_json(adir / "frames.jsonl", lines=True)
    frames["k"] = (frames.t / TICK).round().astype(int)
    tf = pd.read_json(adir / "tfv6.jsonl", lines=True) if (adir / "tfv6.jsonl").stat().st_size else pd.DataFrame()
    if len(tf):
        tf["k"] = tf.frame.map(f2k)
    a = np.load(adir / "actors.npz")
    meta = json.loads((adir / "meta.json").read_text())
    hz = json.loads((adir / "hidden.json").read_text()) if (adir / "hidden.json").exists() else []
    lights = json.loads((adir / "lights.json").read_text())
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    return {"dir": adir, "pose": pose.set_index("k"), "frames": frames.set_index("k"), "tfv6": tf.set_index("k") if len(tf) else tf,
            "act": {"k": np.vectorize(f2k.get)(a["frame"]) if len(a["frame"]) else a["frame"], "id": a["id"],
                    "xyz": a["xyz"], "yaw": a["yaw"], "v": a["v"]},
            "meta": meta, "hero": meta.get("hero_id"), "hazards": [h["id"] for h in hz], "hazard_types": [h["type_id"] for h in hz],
            "light_loc": {l["id"]: tuple(np.round(l["loc"], 1)) for l in lights}, "kinds": kinds}


def ego_divergence(A: dict, B: dict) -> tuple[int, int]:
    """(t_div as a tick index, last common tick): the first tick the two egos differ by >= 1 cm or >= 0.1 deg."""
    m = A["pose"][["x", "y", "yaw"]].join(B["pose"][["x", "y", "yaw"]], rsuffix="_b", how="inner")
    d = np.hypot(m.x - m.x_b, m.y - m.y_b)
    dy = np.abs((m.yaw - m.yaw_b + 180) % 360 - 180)
    bad = np.flatnonzero((d >= DIV_M) | (dy >= DIV_DEG))
    last = int(m.index.max())
    return (int(m.index[bad[0]]) if len(bad) else last + 1), last


def _id_offset(A: dict, B: dict) -> int:
    return int(B["hero"] - A["hero"]) if A["hero"] is not None and B["hero"] is not None else 0


def factor_visibility(A: dict, B: dict, family: str) -> dict:
    """Per camera tick of A (x+): is a factor element visible in the front image, and how many other actors that
    differ between the worlds are visible (the purity diagnostic).

    Factor elements: the hazard actors B hides (named in A's own hidden.json); for HardBreakRoute the background
    vehicles whose position differs between the worlds; for Light the traffic lights whose state differs.
    Visible = at least PX_ACTOR (PX_LIGHT for a light) pixels of it in the instance-segmentation view."""
    off = _id_offset(A, B)
    hz = set(A["hazards"]) or {i - off for i in B["hazards"]}
    ka, kb = A["act"], B["act"]
    posb = {(int(k), int(i)): xyz for k, i, xyz in zip(kb["k"], kb["id"], kb["xyz"])}
    out = {}
    for k, row in A["frames"].iterrows():
        px = row.px if isinstance(row.px, dict) else {}
        sel = ka["k"] == k
        differ = set()
        for i, xyz in zip(ka["id"][sel], ka["xyz"][sel]):
            q = posb.get((k, int(i) + off))
            if q is None or np.hypot(*(xyz[:2] - q[:2])) > ACTOR_DIFF_M:
                differ.add(int(i))
        if family == "HardBreakRoute":
            factor = {i for i in differ if A["kinds"].get(str(i), ["", ""])[1] == "background"}
        else:
            factor = hz
        light_px = 0
        if family == "Light" and k in B["frames"].index:
            lb = {B["light_loc"].get(int(i)): st for i, st in B["frames"].loc[k].lights.items()}
            for key, n in px.items():
                if key.startswith("L"):
                    lid = int(key[1:])
                    sa, sb = row.lights.get(str(lid)), lb.get(A["light_loc"].get(lid))
                    if sa is not None and sb is not None and sa != sb:
                        light_px = max(light_px, n)
        fpx = max([px.get(str(i), 0) for i in factor] + [0])
        out[int(k)] = {"factor_visible": bool(fpx >= PX_ACTOR or light_px >= PX_LIGHT), "factor_px": int(max(fpx, light_px)),
                       "impure_visible": sum(1 for i in differ - factor if px.get(str(i), 0) >= PX_ACTOR),
                       "trig": bool(row.trig[0])}
    return out


# ---------------------------------------------------------------- pairs, observation frames, labels

INTENT_LOOKAHEAD_M = 15.0                   # P4's calibration (research/results/p4-carla-gap/intent_lookahead.csv)
NULL_WINDOW_S = 6.0                         # null frames when the seed-0 pair has no observation frame (fallback)
TFV6_SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0])


def world_rows(adir: Path, rid: str, town: str):
    """P4's WOD-E2E-shaped rows for one recorded world: (rows with tick index k and intent, past, future)."""
    from . import p4_carla as p4
    r = p4.route_rows(adir, rid, town)
    if r is None:
        return None
    t, past, fut, route = r
    t["k"] = (t.t / TICK).round().astype(int)
    t["intent"] = p4.route_intent(route, t.route_progress.to_numpy(), INTENT_LOOKAHEAD_M)
    return t, past, fut


def v2(fut: np.ndarray) -> np.ndarray:
    """Speed 2 s ahead: the 1.75 -> 2.0 s displacement over 0.25 s (future index 6 -> 7)."""
    return np.linalg.norm(fut[..., 7, :] - fut[..., 6, :], axis=-1) / 0.25


def stop3(past: np.ndarray, fut: np.ndarray) -> np.ndarray:
    """Speed below 0.5 m/s somewhere in the next 3 s (current speed included)."""
    pts = np.concatenate([np.zeros_like(fut[:, :1]), fut[:, :12]], 1)
    sp = np.linalg.norm(np.diff(pts, axis=1), axis=-1) / 0.25
    return (np.minimum(sp.min(1), np.linalg.norm(past[:, -1, 2:4], axis=1)) < 0.5)


def tfv6_speeds(W: dict, ks) -> pd.DataFrame:
    """TFv6 at the given ticks: expected target speed, the decoded scalar it drives with, waypoint speed at 2 s."""
    tf = W["tfv6"]
    out = pd.DataFrame(index=pd.Index(ks, name="k"))
    if not len(tf):
        return out.assign(ts=np.nan, ts_scalar=np.nan, wp2=np.nan)
    t = tf.reindex(ks)
    ok = t.pred_target_speed_distribution.notna().to_numpy()
    ts, sc, wp = np.full(len(ks), np.nan), np.full(len(ks), np.nan), np.full(len(ks), np.nan)
    ts[ok] = [float(np.dot(d, TFV6_SPEEDS)) for d in t.pred_target_speed_distribution[ok]]
    sc[ok] = [float(np.ravel(d)[0]) for d in t.pred_target_speed_scalar[ok]]
    w = [np.asarray(x) for x in t.pred_future_waypoints[ok]]
    wp[ok] = [np.linalg.norm(x[7] - x[6]) / 0.25 for x in w]
    return out.assign(ts=ts, ts_scalar=sc, wp2=wp)


def pair_case(gen: Path, case: pd.Series, worlds: dict) -> tuple[dict, list, list]:
    """One case: determinism, first visibility, observation frames with expert labels and TFv6 readouts.

    `worlds` caches (world dict, rows) per variant id. Returns the case row, the pair frames and the null frames."""
    def get(rid):
        if rid not in worlds:
            a = attempt(gen, rid)
            worlds[rid] = None if a is None else (load_world(a), world_rows(a, rid, case.town))
        return worlds[rid]

    row = {k: case[k] for k in ("base_id", "town", "family", "seed", "plus", "minus")}
    P, M = get(case.plus), get(case.minus)
    if P is None or M is None or P[1] is None or M[1] is None:
        return {**row, "reason": "missing_run"}, [], []
    (A, ra), (B, rb) = P, M
    t_div, last = ego_divergence(A, B)
    fv = factor_visibility(A, B, case.family)
    cams = sorted(fv)
    trig = next((k for k in cams if fv[k]["trig"]), None)
    vis = next((k for k in cams if fv[k]["factor_visible"]), None)
    reason = ("never_visible" if vis is None else "ok" if t_div > vis else
              "background_drift" if trig is None or t_div < trig else "expert_reacted_before_visible")
    row.update(t_trig=trig, t_vis=vis, t_div=t_div, t_last=last, reason=reason)
    frames = []
    if reason == "ok":
        (tp, pp, fp), (tm, pm, fm) = ra, rb
        ip, im = pd.Series(np.arange(len(tp)), tp.k), pd.Series(np.arange(len(tm)), tm.k)
        ks = [k for k in cams if vis <= k < t_div and k in ip.index and k in im.index]
        if ks:
            a, b = ip[ks].to_numpy(), im[ks].to_numpy()
            ta, tb = tfv6_speeds(A, ks), tfv6_speeds(B, ks)
            for j, k in enumerate(ks):
                frames.append({**{c: row[c] for c in ("base_id", "family", "seed")}, "k": k,
                               "fn_plus": tp.frame_name.iloc[a[j]], "fn_minus": tm.frame_name.iloc[b[j]],
                               "v0": float(np.linalg.norm(pp[a[j], -1, 2:4])),
                               "v2_plus": float(v2(fp[a[j]])), "v2_minus": float(v2(fm[b[j]])),
                               "stop_plus": bool(stop3(pp[a[j]:a[j] + 1], fp[a[j]:a[j] + 1])[0]),
                               "stop_minus": bool(stop3(pm[b[j]:b[j] + 1], fm[b[j]:b[j] + 1])[0]),
                               "impure_visible": fv[k]["impure_visible"], "factor_px": fv[k]["factor_px"],
                               **{f"tf_{c}_plus": ta.loc[k, c] for c in ta.columns},
                               **{f"tf_{c}_minus": tb.loc[k, c] for c in tb.columns}})
    row["n_obs"] = len(frames)
    nulls = []
    if case.seed == 0 and isinstance(case.null, str) and case.null:
        Nw = get(case.null)
        if Nw is not None and Nw[1] is not None:
            N, rn = Nw
            tdn, _ = ego_divergence(A, N)
            row["t_div_null"] = tdn
            if frames:
                ks = [f["k"] for f in frames]
            else:
                t0 = vis if vis is not None else trig
                ks = [] if t0 is None else [k for k in cams if t0 <= k < t0 + NULL_WINDOW_S / TICK]
                row["null_window"] = "fallback"
            (tp, pp, fp), (tn, pn, fn_) = ra, rn
            ip, iN = pd.Series(np.arange(len(tp)), tp.k), pd.Series(np.arange(len(tn)), tn.k)
            ks = [k for k in ks if k < tdn and k in ip.index and k in iN.index]
            if ks:
                a, b = ip[ks].to_numpy(), iN[ks].to_numpy()
                ta, tn_ = tfv6_speeds(A, ks), tfv6_speeds(N, ks)
                for j, k in enumerate(ks):
                    nulls.append({"base_id": row["base_id"], "family": row["family"], "seed": 0, "k": k,
                                  "fn_plus": tp.frame_name.iloc[a[j]], "fn_null": tn.frame_name.iloc[b[j]],
                                  "v2_plus": float(v2(fp[a[j]])), "v2_null": float(v2(fn_[b[j]])),
                                  **{f"tf_{c}_plus": ta.loc[k, c] for c in ta.columns},
                                  **{f"tf_{c}_null": tn_.loc[k, c] for c in tn_.columns}})
            row["n_null"] = len(nulls)
    return row, frames, nulls


def processed(*parts) -> Path:
    p = data_dir() / "processed" / "carla_p5" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _collect_base(gen: Path, cases: pd.DataFrame):
    worlds, rows, frames, nulls = {}, [], [], []
    for _, c in cases.sort_values("seed").iterrows():
        r, f, n = pair_case(gen, c, worlds)
        rows.append(r)
        frames += f
        nulls += n
    return rows, frames, nulls


def collect(gen: Path, only: set | None = None, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every case of cases.csv that has runs in `gen`: the pair table, pair frames and null frames."""
    from joblib import Parallel, delayed
    cases = pd.read_csv(RESULTS / "cases.csv", dtype={"base_id": str, "plus": str, "minus": str, "null": str})
    if only:
        cases = cases[cases.base_id.isin(only)]
    out = Parallel(workers)(delayed(_collect_base)(gen, g) for _, g in cases.groupby("base_id"))
    pairs = pd.DataFrame([r for o in out for r in o[0]])
    frames = pd.DataFrame([f for o in out for f in o[1]])
    nulls = pd.DataFrame([n for o in out for n in o[2]])
    if len(frames):
        frames["d_expert"] = frames.v2_plus - frames.v2_minus
    if len(nulls):
        nulls["d_expert"] = nulls.v2_plus - nulls.v2_null
    return pairs, frames, nulls


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build"])
    a = p.parse_args()
    if a.cmd == "build":
        build()


if __name__ == "__main__":
    main()
