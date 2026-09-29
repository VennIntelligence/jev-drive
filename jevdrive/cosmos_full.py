"""Cosmos-Transfer G4 full generation (todos/2026-09-28-cosmos-pilot.md, "full run"): about 2 000 CARLA x+ / x- pedestrian
pairs re-rendered photoreal with the frozen v2 arm G4, as training data for the next op-adapt B round (sim + real).

Scenes. Town12 single-scenario clips cut from the Leaderboard 2.0 training routes the way Bench2Drive cuts its own
(scripts/nq3_clips.py, runs/cosmos_full/ped_clips.xml), the four pedestrian families, one clip per scenario instance
(trigger points 5 m apart), minus every instance within 60 m of a Bench2Drive 220 / 0.0.4-val pedestrian route's
trigger (the P5 v1 exam routes). Town13, the small towns and the Bench2Drive pedestrian routes are left to exams;
research/results/cosmos/full/scenes.csv lists the pool and the variants actually used.
Variants. Instance i, variant v: TM seed v % 10; weather = the long route's at the trigger for v = 0, else
PRESETS[(7 i + v) % 14]; base id 7 000 000 + 1 000 (v // 10) + i. Route id = base * 100 + world * 10 + seed, world
1 / 2 = x+ / x- of pass 1, 4 / 5 of pass 2, 7 / 8 of a re-render. x- hides the hazard actors (p5_suppress).
Passes (one CARLA recorder, scripts/cosmos_pair_agent.py on scripts/p5_pair_agent.py, BehaviorAgent, no TFv6 shadow,
no Waymo rig): pass 1 drives both worlds and records poses, actors and the hazard's visibility; `select` picks the
93-tick window; pass 2 drives them again and records the 20 Hz 1280 x 704 camera inside the window only, then stops.
Each CARLA invocation runs pass 2 of the last chunk ahead of pass 1 of the next one, so servers start once per chunk.
controls (CPU): determinism against pass 1, the render QC, then the frozen G4 inputs (edgeE, the tight anchor mask
and blend alpha of jevdrive/cosmos_v2.py) and GT; a flagged pair is re-rendered once, then dropped.
Cosmos: scripts/cosmos_full_worker.py, one per Cosmos slot, takes READY pairs (x- once, x+ anchored, feathered blend).

  build | controls-test | run | summary        (python -m jevdrive.cosmos_full <step>; the lane: scripts/cosmos_full.sh)
Selection, QC and checklist rules are registered in the todo before any full-run data exists.
"""
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RES = REPO / "research" / "results" / "cosmos" / "full"
B2D = "third_party/Bench2Drive/leaderboard/data/"
PED = ("PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "VehicleTurningRoutePedestrian")
TOWN, DEDUP_M, EXAM_M = "Town12", 5.0, 60.0
BASE0 = 7_000_000
WORLD = {1: ("plus", 1), 2: ("minus", 1), 4: ("plus", 2), 5: ("minus", 2), 7: ("plus", 3), 8: ("minus", 3)}
PASS = {1: (1, 2), 2: (4, 5), 3: (7, 8)}
WKEYS = ("cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity", "sun_azimuth_angle",
         "sun_altitude_angle", "fog_density")
_P = {  # carla.WeatherParameters presets (CARLA 0.9.15), in WKEYS order; HardRain* and DustStorm left out
    "ClearNoon": (5, 0, 0, 0, 10, -1, 45, 2), "CloudyNoon": (60, 0, 0, 0, 10, -1, 45, 3),
    "WetNoon": (5, 0, 50, 0, 10, -1, 45, 3), "WetCloudyNoon": (60, 0, 50, 0, 10, -1, 45, 3),
    "SoftRainNoon": (20, 30, 50, 0, 30, -1, 45, 3), "MidRainyNoon": (60, 60, 60, 0, 60, -1, 45, 3),
    "ClearSunset": (5, 0, 0, 0, 10, -1, 15, 2), "CloudySunset": (60, 0, 0, 0, 10, -1, 15, 3),
    "WetSunset": (5, 0, 50, 0, 10, -1, 15, 2), "SoftRainSunset": (20, 30, 50, 0, 30, -1, 15, 2),
    "ClearNight": (5, 0, 0, 0, 10, -1, -90, 60), "WetNight": (5, 0, 50, 60, 10, -1, -90, 60),
    "CloudyNight": (60, 0, 0, 0, 10, -1, -90, 60), "SoftRainNight": (60, 30, 50, 60, 30, -1, -90, 60)}
PRESETS = [(k, dict(zip(WKEYS, map(float, v)))) for k, v in _P.items()]
# selection (pass 1): P5's visibility view (half resolution); the window rule is the pilot's (cosmos_pilot.select)
PX_VIS, PX_LAST, WIN_VIS_TICKS, K0_MIN = 20, 100, 5, 8
# controls (pass 2)
POSE_TOL_M, GT_VIS_MIN = 0.01, 10
QC_DY, QC_FRAC, QC_WHITE = 10.0, 0.10, 200.0
T = 93
CRF = 14


def full(*parts) -> Path:
    p = data_dir() / "runs" / os.environ.get("COSMOS_FULL_DIR", "cosmos_full") / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main_dir() -> Path:
    """Where the scene pool lives (staged pilots run in sub-directories of it)."""
    return data_dir() / "runs" / "cosmos_full"


def rid(base: int, world: int, seed: int) -> str:
    return str(base * 100 + world * 10 + seed)


# ------------------------------------------------------------------ build: scene pool and variants

def _scenarios(path: Path) -> list[dict]:
    out = []
    for r in ET.parse(path).getroot().findall("route"):
        for s in r.iter("scenario"):
            if s.get("type") in PED:
                tp = s.find("trigger_point")
                out.append({"clip": r.get("id"), "town": r.get("town"), "family": s.get("type"), "source": r.get("source", ""),
                            "tx": float(tp.get("x")), "ty": float(tp.get("y")), "tz": float(tp.get("z")),
                            "tyaw": float(tp.get("yaw"))})
    return out


def pool() -> pd.DataFrame:
    """Town12 long-route pedestrian clips, one per instance, away from the Bench2Drive pedestrian routes."""
    c = pd.DataFrame(_scenarios(main_dir() / "ped_clips.xml"))
    c = c[c.town == TOWN].reset_index(drop=True)
    xy, keep = c[["tx", "ty"]].to_numpy(), np.ones(len(c), bool)
    for i in range(1, len(c)):
        keep[i] = not (np.hypot(*(xy[:i] - xy[i]).T) < DEDUP_M)[keep[:i]].any()
    c = c[keep].reset_index(drop=True)
    ex = pd.DataFrame(_scenarios(data_dir() / B2D / "bench2drive220.xml") + _scenarios(data_dir() / B2D / "bench2drive_0.0.4_val.xml"))
    e = ex[ex.town == TOWN][["tx", "ty"]].to_numpy()
    c["exam_dist_m"] = [float(np.hypot(*(e - p).T).min()) for p in c[["tx", "ty"]].to_numpy()]
    c = c[c.exam_dist_m >= EXAM_M].reset_index(drop=True)
    c.insert(0, "inst", np.arange(len(c)))
    return c


def variant(inst: int, v: int, own: dict) -> dict:
    name, w = ("route", own) if v == 0 else PRESETS[(7 * inst + v) % len(PRESETS)]
    base = BASE0 + 1000 * (v // 10) + inst
    return {"pair": f"c{inst:03d}v{v:02d}", "inst": inst, "v": v, "base": base, "seed": v % 10, "weather_name": name,
            "weather": json.dumps(w), **{f"id{k}": rid(base, k, v % 10) for k in WORLD}}


def build():
    p = pool()
    src = {r.get("id"): r for r in ET.parse(main_dir() / "ped_clips.xml").getroot().findall("route")}
    p["weather_route"] = [json.dumps({k: float(src[c].find("weathers")[0].get(k)) for k in WKEYS}) for c in p["clip"]]
    p.to_csv(main_dir() / "pool.csv", index=False)
    RES.mkdir(parents=True, exist_ok=True)
    p.drop(columns="weather_route").to_csv(RES / "scenes.csv", index=False)
    log.info("pool: %d %s instances %s", len(p), TOWN, p.family.value_counts().to_dict())
    return p


def route_elems(rows: pd.DataFrame, worlds: list[int]) -> list:
    """Variant routes: the clip's route with its id, weather and (x-) p5_suppress."""
    pl = pd.read_csv(main_dir() / "pool.csv", dtype={"clip": str}).set_index("inst")
    src = {r.get("id"): r for r in ET.parse(main_dir() / "ped_clips.xml").getroot().findall("route")}
    out = []
    for _, r in rows.iterrows():
        for k in worlds:
            e = ET.fromstring(ET.tostring(src[pl.loc[r.inst, "clip"]]))
            e.set("id", r[f"id{k}"])
            e.set("p5_suppress", "1" if WORLD[k][0] == "minus" else "0")
            for w in e.find("weathers").findall("weather"):
                for key, val in json.loads(r.weather).items():
                    w.set(key, str(val))
            out.append(e)
    return out


def write_xml(elems: list, path: Path):
    root = ET.Element("routes")
    root.extend(elems)
    ET.indent(root)
    tmp = path.with_suffix(".tmp")
    ET.ElementTree(root).write(tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ select (pass 1 -> window)

def select_pair(r) -> dict:
    """Window of one pass-1 pair. Rule (todo, registered before the full run): walker hazards only visible; x+ sees a
    hazard (>= 20 px) before the egos diverge; k1 = min(t_div, last tick with >= 100 px + 5), k0 = k1 - 93 >= 8;
    >= 5 camera ticks with >= 100 px inside the window; no other actor that differs between the worlds is visible."""
    from . import p5_pairs as PP
    gen = full("gen")
    out = {"pair": r.pair}
    a, b = PP.attempt(gen, r.id1), PP.attempt(gen, r.id2)
    if a is None or b is None:
        return {**out, "reason": "missing_run"}
    try:
        A, B = PP.load_world(a), PP.load_world(b)
    except Exception as e:                      # noqa: BLE001  (a truncated attempt)
        return {**out, "reason": "load_error", "error": repr(e)[:200]}
    walk = [h for h, t in zip(A["hazards"], A["hazard_types"]) if t.startswith("walker.")]
    other = [h for h in A["hazards"] if h not in walk]
    t_div, last = PP.ego_divergence(A, B)
    fv = PP.factor_visibility(A, B, r.family)
    px = {k: (row.px if isinstance(row.px, dict) else {}) for k, row in A["frames"].iterrows()}
    wpx = pd.Series({k: max([d.get(str(h), 0) for h in walk] + [0]) for k, d in px.items()}).sort_index()
    opx = pd.Series({k: max([d.get(str(h), 0) for h in other] + [0]) for k, d in px.items()}).sort_index()
    imp = pd.Series({k: v["impure_visible"] for k, v in fv.items()}).sort_index()
    out.update(hazards=len(A["hazards"]), walkers=len(walk), t_div=t_div, t_last=last, stop_plus=_stop(a), stop_minus=_stop(b))
    if not walk:
        return {**out, "reason": "no_walker"}
    vis = wpx.index[wpx >= PX_VIS]
    if not len(vis):
        return {**out, "reason": "never_visible"}
    out["t_vis"] = int(vis[0])
    if t_div <= vis[0]:
        return {**out, "reason": "expert_before_visible"}
    lastvis = wpx.index[wpx >= PX_LAST]
    if not len(lastvis):
        return {**out, "reason": "never_100px"}
    k1 = int(min(t_div, lastvis.max() + 5))
    k0 = k1 - T
    out.update(k_lastvis=int(lastvis.max()), k0=k0, k1=k1)
    if k0 < K0_MIN:
        return {**out, "reason": "window_too_early"}
    inw = (wpx.index >= k0) & (wpx.index < k1)
    out.update(win_vis_ticks=int((wpx[inw] >= PX_LAST).sum()), win_px_max=int(wpx[inw].max()),
               other_px_max=int(opx[(opx.index >= k0) & (opx.index < k1)].max()),
               impure_max=int(imp[(imp.index >= k0) & (imp.index < k1)].max()))
    if out["win_vis_ticks"] < WIN_VIS_TICKS:
        return {**out, "reason": "window_low_vis"}
    if out["other_px_max"] >= PX_VIS:
        return {**out, "reason": "non_walker_hazard_visible"}
    if out["impure_max"] > 0:
        return {**out, "reason": "impure"}
    return {**out, "reason": "ok"}


def _stop(adir: Path) -> str:
    try:
        return json.loads((adir / "p5_summary.json").read_text()).get("stop", "")
    except (OSError, ValueError):
        return ""


# ------------------------------------------------------------------ controls (pass 2 -> Cosmos inputs)

def controls_pair(r, rerender: bool = False, clips: Path | None = None, keep_pngs: bool = False, gen: Path | None = None,
                  ref: Path | None = None) -> dict:
    """Determinism against pass 1, render QC, and the frozen G4 inputs of one pair. Same arithmetic as
    cosmos_pilot._controls_pair (+ gt_boxes) and cosmos_v2._controls2 (edgeE, tight anchor mask and alpha), fused into one
    pass over the frames; edgeE is computed on rgb.mp4 decoded back, as v2 did. `gen` / `ref`: the pass-2 and pass-1 run
    dirs (both runs/cosmos_full/gen by default); `clips`: where the Cosmos inputs go (clips/<pair>)."""
    t_start = time.time()
    g, ref = gen or full("gen"), ref or full("gen")
    worlds = PASS[3 if rerender else 2]
    row = {"pair": r.pair, "gen": "gen3" if rerender else "gen"}
    ids = {"plus": r[f"id{worlds[0]}"], "minus": r[f"id{worlds[1]}"]}
    W_ = {}
    try:
        return _controls(r, row, ids, g, ref, W_, clips, t_start)
    finally:
        if not keep_pngs:                      # the 20 Hz PNGs, 3 x 93 x ~0.7 MB per world; all that is needed is extracted
            for a, *_ in W_.values():
                shutil.rmtree(a / "op", ignore_errors=True)


def _controls(r, row, ids, g, ref, W_, clips, t_start) -> dict:
    import cv2

    from . import cosmos_pilot as CP
    from . import cosmos_v2 as C2
    from . import p5_pairs as PP
    ks = list(range(int(r.k0), int(r.k1)))
    for m, i in ids.items():
        a = PP.attempt(g, i)
        if a is None:
            return {**row, "reason": "missing_run"}
        A, O = PP.load_world(a), PP.load_world(PP.attempt(ref, r[f"id{1 if m == 'plus' else 2}"]))
        j = A["pose"][["x", "y", "yaw"]].join(O["pose"][["x", "y", "yaw"]], rsuffix="_o", how="inner")
        j = j[j.index < int(r.k1)]
        row[f"{m}_pose_max_m"] = float(np.hypot(j.x - j.x_o, j.y - j.y_o).max()) if len(j) else np.nan
        fr = pd.read_json(a / "op" / "frames.jsonl", lines=True).set_index("k") if (a / "op" / "frames.jsonl").stat().st_size else None
        W_[m] = (a, A, fr)
        if fr is None or not all(k in fr.index for k in ks):
            return {**row, "reason": "missing_frames"}
    if not max(row["plus_pose_max_m"], row["minus_pose_max_m"]) <= POSE_TOL_M:
        return {**row, "reason": "nondeterministic"}
    (ap, Ap, _), (_, Am, _) = W_["plus"], W_["minus"]
    ego = Ap["pose"].loc[ks, ["x", "y"]].to_numpy() - Am["pose"].loc[ks, ["x", "y"]].to_numpy()
    row["pair_ego_max_m"] = float(np.hypot(*ego.T).max())
    kinds, act = Ap["kinds"], Ap["act"]
    haz = [i for i, t in zip(Ap["hazards"], Ap["hazard_types"]) if t.startswith("walker.")]
    route = pd.read_json(ap / "route.json")[["x", "y", "z"]].to_numpy()
    cd = clips or full("clips", r.pair)
    vids = {m: {"rgb": [], "edge": [], "tag": []} for m in W_}
    gt = {"mask": [], "region": [], "walk_minus": [], "corridor": [], "px": [], "box": []}
    hb_box, hb_px = np.full((len(ks), len(haz), 4), -1, np.int32), np.zeros((len(ks), len(haz)), np.int32)
    cache, masks = {}, []
    for ti, k in enumerate(ks):
        F = {m: (np.array(fr.loc[k, "cam"]), *CP._load_frame(a, k)) for m, (a, _, fr) in W_.items()}
        M, _, _, tag, inst = F["plus"]
        sel = act["k"] == k
        boxes = []
        for j, h in enumerate(haz):
            mm = act["id"][sel] == h
            if not mm.any():
                continue
            bc = CP.box_corners(act["xyz"][sel][mm][0], act["yaw"][sel][mm][0], kinds[str(h)][2])
            boxes.append(bc)
            mk, _ = CP._hazard_mask(M, tag, inst, [bc])
            ys, xs = np.nonzero(mk)
            hb_px[ti, j] = len(xs)
            if len(xs):
                hb_box[ti, j] = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        mask, boxm = CP._hazard_mask(M, tag, inst, boxes)
        _, _, _, tag_m, inst_m = F["minus"]
        region = cv2.dilate((mask | boxm).astype(np.uint8), np.ones((2 * CP.REGION_DILATE + 1,) * 2, np.uint8)) > 0
        th = np.isin(tag, list(CP.THINGS)) & (tag == tag_m) & ~region
        remap = inst_m.copy()
        if th.any():
            pm = pd.DataFrame({"m": inst_m[th], "p": inst[th]}).value_counts().reset_index()
            best = pm.sort_values("count", ascending=False).drop_duplicates("m")
            lut = dict(zip(best.m, best.p))
            thm = np.isin(tag_m, list(CP.THINGS))
            remap[thm] = pd.Series(inst_m[thm]).map(lambda i: lut.get(i, i + 1_000_000)).to_numpy()
        for m, (tg, ins) in (("plus", (tag, inst)), ("minus", (tag_m, remap))):
            _, rgb, depth = F[m][:3]
            vids[m]["rgb"].append(rgb)
            vids[m]["edge"].append(CP.edge_image(CP.seg_image(tg, ins, cache), depth))
            vids[m]["tag"].append(tg)
        masks.append(mask)
        gt["mask"].append(np.packbits(mask))
        gt["region"].append(np.packbits(region))
        gt["walk_minus"].append(np.packbits(tag_m == CP.PED_TAG))
        gt["px"].append(int(mask.sum()))
        ys, xs = np.nonzero(mask)
        gt["box"].append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [-1, -1, -1, -1])
        gt["corridor"].append(CP._corridor(M, route, Ap["pose"].loc[k, ["x", "y"]].to_numpy(np.float64)))
    px = np.array(gt["px"])
    row.update(frames=len(ks), vis_frames=int((px >= CP.VIS_PX).sum()), px_max=int(px.max()), hazards=len(haz))
    # render QC (todo, registered): luma of the raw frames outside the hazard region, x+ against x-
    region = np.stack([np.unpackbits(x)[: CP.H * CP.W].reshape(CP.H, CP.W).astype(bool) for x in gt["region"]])
    Y = {m: np.stack(vids[m]["rgb"]).astype(np.float32) @ np.float32([0.299, 0.587, 0.114]) for m in W_}
    yo = {m: np.array([y[~rg].mean() for y, rg in zip(Y[m], region)]) for m in W_}
    dy = np.abs(yo["plus"] - yo["minus"])
    whole = np.maximum(Y["plus"].mean((1, 2)), Y["minus"].mean((1, 2)))
    row.update(qc_dy_med=float(np.median(dy)), qc_dy_max=float(dy.max()), qc_frac=float((dy > QC_DY).mean()),
               qc_luma_plus=float(np.median(yo["plus"])), qc_luma_minus=float(np.median(yo["minus"])),
               qc_white=float(np.median(whole)))
    del Y
    if row["qc_frac"] >= QC_FRAC or row["qc_white"] >= QC_WHITE:
        return {**row, "reason": "qc_render"}
    if row["vis_frames"] < GT_VIS_MIN:
        return {**row, "reason": "gt_low_vis"}
    # Cosmos inputs
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    tmp = cd / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for m in W_:
        md = cd / m
        md.mkdir(parents=True, exist_ok=True)
        CP.write_mp4(md / "rgb.mp4", np.stack(vids[m]["rgb"]))
        CP.write_mp4(tmp / f"edge_{m}.mp4", np.stack(vids[m]["edge"]))
        rgb = CP.read_mp4(md / "rgb.mp4")
        edge = CP.read_mp4(tmp / f"edge_{m}.mp4")[..., 0] > 127
        ee = []
        for t in range(len(ks)):
            tg = vids[m]["tag"][t]
            obj = cv2.erode(np.isin(tg, C2.OBJ).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            grey = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY)
            e = edge[t] | ((cv2.Canny(grey, 100, 200) > 0) & obj)
            e = e | (cv2.Canny(grey, 30, 90) > 0) & np.isin(tg, C2.GROUND)
            e = e | ((cv2.Canny(clahe.apply(grey), 30, 90) > 0) & obj)
            ee.append(np.repeat((e * 255).astype(np.uint8)[..., None], 3, -1))
        CP.write_mp4(md / "edgeE.mp4", np.stack(ee))
        _write_crf(cd / f"carla_{m}.mp4", rgb)     # the stored CARLA clip (training data, sim domain)
    shutil.rmtree(tmp)
    k24, k6 = np.ones((49, 49), np.uint8), np.ones((13, 13), np.uint8)
    mt = np.stack([cv2.dilate(m_.astype(np.uint8), k24) > 0 for m_ in masks])
    free_t = np.stack([mt[max(0, t - C2.FREE_DILATE_T):t + C2.FREE_DILATE_T + 1].any(0) for t in range(len(mt))])
    CP.write_mp4(cd / "anchor_mask_tight.mp4", np.repeat(((~free_t) * 255).astype(np.uint8)[..., None], 3, -1))
    at = np.stack([np.maximum(cv2.GaussianBlur((cv2.dilate(m_.astype(np.uint8), k6) > 0).astype(np.float32), (0, 0), 3.0),
                              m_.astype(np.float32)) for m_ in masks])
    np.savez_compressed(cd / "alpha_tight.npz", alpha=at.astype(np.float16))
    L = max(len(c) for c in gt["corridor"])
    corr = np.full((len(ks), L, 2), np.nan, np.float32)
    for i, c in enumerate(gt["corridor"]):
        corr[i, :len(c)] = c
    np.savez_compressed(cd / "gt.npz", mask=np.stack(gt["mask"]), region=np.stack(gt["region"]),
                        walk_minus=np.stack(gt["walk_minus"]), px=px, box=np.array(gt["box"]), corridor=corr,
                        k=np.array(ks), shape=np.array([CP.H, CP.W]), support=np.packbits(at > 0, axis=None))
    np.savez(cd / "gt_boxes.npz", box=hb_box, px=hb_px, hazards=np.array(haz))
    meta = json.loads((ap / "meta.json").read_text())
    spec = {"pair": r.pair, "town": TOWN, "weather": meta["weather"], "prompt": prompt2_full(meta["weather"]),
            "k0": int(r.k0), "k1": int(r.k1), "ids": ids, "gen": row["gen"], "family": r.family, "inst": int(r.get("inst", -1)), "v": int(r.get("v", -1))}
    (cd / "spec.json").write_text(json.dumps(spec))
    row.update(reason="ok", controls_s=round(time.time() - t_start, 1))
    (cd / "READY").write_text(json.dumps(row))
    return row


def controls_test(pair: str = "24211-s0"):
    """Numerical equivalence of controls_pair with the v2 pilot's inputs, on a pilot pair (its re-render and P5 v1 run as
    pass 2 / pass 1; PNGs kept): every Cosmos input and GT array must match the pilot's files exactly."""
    from . import cosmos_pilot as CP
    r = CP.pairs().set_index("pair").loc[pair]
    row = pd.Series({"pair": pair, "k0": r.k0, "k1": r.k1, "family": r.family, "id4": r.plus, "id5": r.minus,
                     "id1": r.plus, "id2": r.minus})
    out = full("equiv", pair)
    res = controls_pair(row, clips=out, keep_pngs=True, gen=CP.root("gen"), ref=data_dir() / CP.GEN_V1)
    log.info("controls: %s", res)
    ref = CP.root("clips", pair)
    rep = {}
    for m in ("plus", "minus"):
        for f in ("rgb.mp4", "edgeE.mp4"):
            a, b = CP.read_mp4(out / m / f), CP.read_mp4(ref / m / f)
            rep[f"{m}/{f}"] = int(np.abs(a.astype(np.int16) - b).max())
    rep["anchor_mask_tight"] = int(np.abs(CP.read_mp4(out / "anchor_mask_tight.mp4").astype(np.int16)
                                          - CP.read_mp4(ref / "anchor_mask_tight.mp4")).max())
    rep["alpha_tight"] = float(np.abs(np.load(out / "alpha_tight.npz")["alpha"].astype(np.float32)
                                      - np.load(ref / "alpha_tight.npy").astype(np.float32)).max())
    z, zr = np.load(out / "gt.npz"), np.load(ref / "gt.npz")
    for k in ("mask", "region", "walk_minus", "px", "box", "k"):
        rep[f"gt/{k}"] = int(np.abs(z[k].astype(np.int64) - zr[k]).max())
    rep["gt/corridor"] = float(np.nanmax(np.abs(z["corridor"] - zr["corridor"])))
    b, br = np.load(out / "gt_boxes.npz"), np.load(ref / "gt_boxes.npz")
    rep["gt_boxes"] = int(max(np.abs(b["box"] - br["box"]).max(), np.abs(b["px"] - br["px"]).max()))
    log.info("max abs difference to the pilot's inputs (0 = identical): %s", rep)
    (out / "equiv.json").write_text(json.dumps({"controls": res, "diff": rep}, default=float))
    return rep


def _write_crf(path: Path, frames: np.ndarray, crf: int = CRF):
    from .cosmos_pilot import ffmpeg_bin
    t, h, w, _ = frames.shape
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "20",
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


PLACE12 = "a road in a mid-sized city with residential and commercial areas"


def prompt2_full(weather: dict) -> str:
    """cosmos_v2.prompt2 with a place phrase for Town12 (the pilot's PLACE table has the small towns only)."""
    from . import cosmos_pilot as CP
    from .cosmos_v2 import prompt2
    CP.PLACE.setdefault(TOWN, PLACE12)
    return prompt2(TOWN, weather)


def load_pair(pair: str, kind: str = "cosmos", root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(x+, x-) frames of a stored pair, (93, 704, 1280, 3) uint8 RGB. kind "cosmos": x+ equals x- outside the blend
    support (as generated; each clip is stored lossy on its own); "carla": the raw CARLA pair."""
    from .cosmos_pilot import H, W, read_mp4
    d = (root or full()) / "pairs" / pair
    plus, minus = read_mp4(d / f"{kind}_plus.mp4"), read_mp4(d / f"{kind}_minus.mp4")
    if kind == "cosmos":
        s = np.unpackbits(np.load(d / "gt.npz")["support"])[: plus.size // 3].reshape(len(plus), H, W, 1).astype(bool)
        plus = np.where(s, plus, minus)
    return plus, minus


# ------------------------------------------------------------------ lane driver

def _row() -> dict:
    """The cosmos-full row of runs/sched/table.tsv: per CARLA GPU its server index block and core slice."""
    import csv
    rows = list(csv.DictReader(open(data_dir() / "runs" / "sched" / "table.tsv"), delimiter="\t"))
    r = next(x for x in rows if x["lane"] == os.environ.get("COSMOS_LANE", "cosmos-full"))
    g = [int(x) for x in r["gpus"].split(",")]
    span = int(r["idx_span"])
    idx = dict((int(k), int(v)) for k, v in (e.split(":") for e in r["idx0"].split(","))) if ":" in r["idx0"] else \
        {x: int(r["idx0"]) + i * span for i, x in enumerate(g)}
    cores = []
    for part in r["cpus"].split(","):
        lo, _, hi = part.partition("-")
        cores += range(int(lo), int(hi or lo) + 1)
    return {"gpus": g, "workers": int(r["workers"]), "idx": idx, "span": span, "cores": cores}


class Lane:
    """One-shot, resumable driver. State lives in files under runs/cosmos_full/: variants.csv (scheduled variants),
    sel.csv (pass-1 windows), ctl.csv (controls outcomes), clips/<pair>/READY, pairs/<pair>/done.json.
    Signals in lane/: STATUS (appended), DONE, ERROR, DRAIN (touch to stop at the next boundary), CONTROLS_DONE."""

    STRUCTURAL = {"no_walker", "never_visible", "never_100px", "non_walker_hazard_visible"}

    def __init__(self, target: int, insts: list | None, keep_npy: bool):
        from .runlog import RunLog
        self.target, self.keep_npy = target, keep_npy
        self.L = full("lane")
        self.rl = RunLog(*full().relative_to(data_dir() / "runs").parts, "lane")
        self.pool = pd.read_csv(main_dir() / "pool.csv", dtype={"clip": str})
        if insts is not None:
            self.pool = self.pool[self.pool.inst.isin(insts)]
        self.row = _row()
        self.cw = {int(k): int(v) for k, v in (e.split(":") for e in os.environ["CARLA_W"].split(","))} if \
            os.environ.get("CARLA_W") else {g: self.row["workers"] for g in self.row["gpus"]}
        self.cslots = [s for s in os.environ.get("COSMOS_SLOTS", ",".join(map(str, self.row["gpus"]))).split(",") if s]
        self.floor = float(os.environ.get("DISK_FLOOR_GB", "150"))
        self.ctl_pool = ProcessPoolExecutor(int(os.environ.get("CONTROLS_PROCS", "16")))
        self.fut, self.inv, self.workers = {}, len(list(full("xml").glob("inv*.xml"))), {}
        self.t0 = time.time()

    # -------------------------------------------------------------- bookkeeping

    def status(self, msg: str):
        line = f"{time.strftime('%F %T')} {msg}"
        with open(self.L / "STATUS", "a") as f:
            f.write(line + "\n")
        self.rl.info(msg)

    def table(self, name: str) -> pd.DataFrame:
        f = full() / name
        return pd.read_csv(f, dtype={f"id{k}": str for k in WORLD}) if f.exists() and f.stat().st_size else pd.DataFrame()

    def append(self, name: str, rows: list):
        if rows:
            f = full() / name
            pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists() or not f.stat().st_size, index=False)

    def disk_gb(self) -> float:
        return shutil.disk_usage(data_dir()).free / 2**30

    def done_pairs(self) -> int:
        return len(list(full("pairs").glob("*/done.json")))

    # -------------------------------------------------------------- CARLA

    def agent_json(self, windows: dict) -> Path:
        c = {"tfv6_model_dir": "", "rig": False, "pass_stop_s": 0.5, "cosmos_stop": True, "cosmos_windows": windows,
             "cosmos_rgb_attrs": json.loads(os.environ.get("COSMOS_RGB_ATTRS", "{}"))}
        f = full() / "agent.json"
        f.write_text(json.dumps(c))
        return f

    def carla(self, xml: Path, ids: list, agent: Path) -> dict:
        """One CARLA invocation: a b2d_run per GPU on the same id list (claims shard it), servers started once."""
        D = data_dir()
        lead = D / "third_party/scout/lead-cvpr2026"
        env = dict(os.environ, B2D_RESEED_AFTER_BUILD="1", LEAD_PROJECT_ROOT=str(lead), HF_HUB_OFFLINE="1",
                   OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="3", SAVE_PATH=str(full("lead_save")),
                   PYTHONPATH=f"{lead}:{os.environ.get('PYTHONPATH', '')}", B2D_DRAIN_FILE=str(self.L / "DRAIN"))
        gs = [g for g in self.row["gpus"] if self.cw.get(g, 0) > 0]
        per = len(self.row["cores"]) // len(gs)
        procs = []
        for j, g in enumerate(gs):
            cores = ",".join(map(str, self.row["cores"][j * per:(j + 1) * per]))
            cmd = ["taskset", "-c", cores, str(D / "envs/carla/bin/python"), "scripts/b2d_run.py", "--routes", str(xml),
                   "--route-ids", ",".join(ids), "--out", str(full("gen")), "--workers", str(self.cw[g]),
                   "--server-index", str(self.row["idx"][g]), "--index-span", str(self.row["span"]), "--gpu-rank", str(g),
                   "--tm-seed-from-id", "--agent", "scripts/cosmos_pair_agent.py", "--agent-config", str(agent),
                   "--python", str(D / "envs/scout-tfv6/bin/python"), "--fast-copy", "--no-spectator",
                   "--max-attempts", "2", "--stagger-s", "20"] + ([] if j == 0 else ["--no-reap"])
            logf = open(full("carla_logs") / f"inv{self.inv:03d}-gpu{g}.log", "w")
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=logf,
                                          stderr=subprocess.STDOUT))
            (full("carla_logs") / f"inv{self.inv:03d}-gpu{g}.pid").write_text(str(procs[-1].pid))
            if j == 0:
                time.sleep(10)                       # the reaping runner first (orphans of a crashed invocation)
        while any(p.poll() is None for p in procs):
            self.poll()
            time.sleep(30)
        done = {i for i in ids if (full("gen") / "done" / f"{i}.json").exists()}
        return {"ids": len(ids), "done": len(done), "rc": [p.returncode for p in procs]}

    # -------------------------------------------------------------- CPU steps

    def select(self, V: pd.DataFrame):
        todo = V[~V.pair.isin(self.table("sel.csv").get("pair", pd.Series(dtype=str)))]
        todo = todo.merge(self.pool[["inst", "family"]], on="inst")
        if not len(todo):
            return
        with ProcessPoolExecutor(min(32, len(todo))) as ex:
            rows = list(ex.map(select_pair, [r for _, r in todo.iterrows()]))
        self.append("sel.csv", rows)
        s = pd.DataFrame(rows).reason.value_counts().to_dict()
        self.status(f"select: {len(rows)} pairs -> {s}")

    def poll(self):
        """Collect finished controls, start new ones; supervise the Cosmos workers; disk floor."""
        if time.time() - getattr(self, "_fed", 0) > 120:
            self._fed = time.time()
            self.feed()
        from concurrent.futures.process import BrokenProcessPool
        broken = False
        for f in [f for f in self.fut if f.done()]:
            pair, gen = self.fut.pop(f)
            try:
                r = f.result()
            except BrokenProcessPool:               # a controls process was killed (OOM): redo these pairs
                broken = True
                continue
            except Exception as e:                  # noqa: BLE001
                r = {"pair": pair, "gen": gen, "reason": "controls_error", "error": repr(e)[:300]}
            self.append("ctl.csv", [r])
        if broken:
            self.status("controls pool broken (a process was killed); new pool, the affected pairs are fed again")
            self.ctl_pool = ProcessPoolExecutor(self.ctl_pool._max_workers)
        for slot, (p, n, cmd, env) in list(self.workers.items()):
            if p.poll() is not None and p.returncode != 0 and not (self.L / "DRAIN").exists():
                if n >= 5:
                    self.fail(f"Cosmos worker {slot} died {n + 1} times (rc {p.returncode})")
                self.status(f"Cosmos worker {slot} exited rc {p.returncode}; restart {n + 1}")
                for c in full("cosmos", "claims").iterdir():     # the pair it was on goes back to the queue
                    if c.read_text() == f"g{slot}" and not (full("pairs") / c.name / "done.json").exists():
                        c.unlink()
                self.workers[slot] = (self.spawn_worker(cmd, env, slot), n + 1, cmd, env)
        if self.disk_gb() < self.floor:
            self.fail(f"disk {self.disk_gb():.0f} GB free < floor {self.floor:.0f}")

    def fail(self, msg: str):
        (self.L / "DRAIN").touch()                  # running b2d_run routes and Cosmos pairs finish, nothing new starts
        self.status("ERROR: " + msg)
        (self.L / "ERROR").write_text(msg + "\n")
        raise SystemExit(1)

    def spawn_worker(self, cmd, env, slot):
        logf = open(full("cosmos_logs") / f"{slot}-{time.strftime('%m%d-%H%M%S')}.log", "w")
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
        (full("cosmos_logs") / f"{slot}.pid").write_text(str(p.pid))
        return p

    def start_workers(self):
        late = set(os.environ.get("COSMOS_LATE", "").split(","))
        for s in self.cslots:
            g = s.split("-")[0]
            cmd = [str(data_dir() / "envs/cosmos-transfer/bin/python"), "scripts/cosmos_full_worker.py", "--root", str(full()),
                   "--slot", f"g{s}", "--target", str(self.target)] + (["--keep-npy"] if self.keep_npy else []) + \
                (["--start-after", str(self.L / "CARLA_DONE")] if s in late else [])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONPATH=str(REPO), COSMOS_TE_CACHE=str(full("te_cache")))
            self.workers[s] = (self.spawn_worker(cmd, env, s), 0, cmd, env)
        self.status(f"Cosmos workers: {self.cslots} (late start: {sorted(late & set(self.cslots))})")

    # -------------------------------------------------------------- planning

    @staticmethod
    def done2(r, w: int) -> bool:
        return all((full("gen") / "done" / f"{r[f'id{k}']}.json").exists() for k in PASS[w])

    def feed(self):
        """Controls for every pair whose pass-2 (or re-render) worlds are both done; called while CARLA runs too."""
        V, S, C = self.table("variants.csv"), self.table("sel.csv"), self.table("ctl.csv")
        if not len(S):
            return
        Vi = V.set_index("pair")
        fin = C.drop_duplicates("pair", keep="last") if len(C) else pd.DataFrame(columns=["pair", "gen", "reason"])
        seen, inflight = set(fin.pair), {p for p, _ in self.fut.values()}
        Si = S.drop_duplicates("pair", keep="last").set_index("pair")
        for p in S[S.reason == "ok"].pair:
            if p not in seen and p not in inflight and self.done2(Vi.loc[p], 2):
                self.submit(p, False, Vi, Si)
        for p in fin[(fin.reason == "qc_render") & (fin.gen == "gen")].pair:
            if p not in inflight and self.done2(Vi.loc[p], 3):
                self.submit(p, True, Vi, Si)

    def plan(self):
        """What the next invocation runs: pass 2 of selected pairs (as many as the target still needs), re-renders of
        QC-flagged pairs, and pass 1 of a new chunk of variants when the pairs in hand will not reach the target."""
        self.feed()
        V, S, C = self.table("variants.csv"), self.table("sel.csv"), self.table("ctl.csv")
        ok_sel = S[S.reason == "ok"] if len(S) else S
        fin = C.drop_duplicates("pair", keep="last") if len(C) else pd.DataFrame(columns=["pair", "gen", "reason"])
        final = fin[(fin.reason != "qc_render") | (fin.gen == "gen3")]
        n_ok = int((final.reason == "ok").sum())
        rate = n_ok / len(final) if len(final) >= 20 else 0.85
        inflight = {p for p, _ in self.fut.values()}
        done2 = self.done2
        Vi = V.set_index("pair") if len(V) else V
        pending2 = [p for p in ok_sel.pair if p not in inflight and p not in set(fin.pair)
                    and not done2(Vi.loc[p], 2)] if len(ok_sel) else []
        expect = n_ok + rate * (len(inflight) + len(pending2))
        n2 = max(0, int(np.ceil((self.target * 1.03 - n_ok - rate * len(inflight)) / max(rate, 0.3))))
        p2 = pending2[:n2]
        p3 = [c.pair for _, c in fin.iterrows() if c.reason == "qc_render" and c.gen == "gen" and not done2(Vi.loc[c.pair], 3)
              and c.pair not in inflight]
        # a new chunk only while pass 1 is not still owed and the expectation is short
        if self.disk_gb() < self.floor + 100:        # Cosmos is behind: no more CARLA output until it catches up
            return [], [], pd.DataFrame(), {"disk_pause": round(self.disk_gb())}
        p1_open = [p for p in (V.pair if len(V) else []) if not (len(S) and p in set(S.pair))]
        new = pd.DataFrame()
        sel_rate = len(ok_sel) / len(S) if len(S) >= 20 else 0.6
        if not p1_open and expect < self.target * 1.02 and not (self.L / "DRAIN").exists():
            new = self.new_chunk(int(np.ceil((self.target * 1.03 - expect) / max(sel_rate * rate, 0.1))), V, S)
        return p2, p3, new, {"ok": n_ok, "rate": round(rate, 3), "sel_rate": round(sel_rate, 3), "inflight": len(inflight),
                             "pending2": len(pending2), "expect": round(expect)}

    def new_chunk(self, want: int, V: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        bad = set()
        if len(S):
            s0 = S.merge(V[["pair", "inst", "v"]], on="pair")
            bad = set(s0[(s0.v == 0) & s0.reason.isin(self.STRUCTURAL)].inst)
            fails = s0.groupby("inst").reason.apply(lambda x: (x != "ok").all() and len(x) >= 3)
            bad |= set(fails[fails].index)
        nxt = V.groupby("inst").v.max().to_dict() if len(V) else {}
        act = [i for i in self.pool.inst if i not in bad and nxt.get(i, -1) + 1 < 30]
        if not act:
            return pd.DataFrame()
        rounds = 1 if not len(V) else max(1, min(3, int(np.ceil(want / len(act)))))
        rows = []
        for rr in range(rounds):
            for i in act:
                v = nxt.get(i, -1) + 1 + rr
                if v < 30:
                    rows.append(variant(i, v, json.loads(self.pool.set_index("inst").loc[i, "weather_route"])))
        return pd.DataFrame(rows)

    def submit(self, pair: str, rerender: bool, V: pd.DataFrame, S: pd.DataFrame):
        r = pd.concat([V.loc[pair], S.loc[pair, ["k0", "k1"]]])
        r["pair"], r["family"] = pair, self.pool.set_index("inst").loc[int(r.inst), "family"]
        self.fut[self.ctl_pool.submit(controls_pair, r, rerender)] = (pair, "gen3" if rerender else "gen")

    # -------------------------------------------------------------- main loop

    def run(self):
        for f in ("DONE", "ERROR", "CONTROLS_DONE", "CARLA_DONE"):
            (self.L / f).unlink(missing_ok=True)
        for c in full("cosmos", "claims").iterdir():     # claims of workers that died with the last driver
            if not (full("pairs") / c.name / "done.json").exists():
                c.unlink()
        shutil.rmtree(full("cosmos", "tmp"), ignore_errors=True)
        free = self.disk_gb()
        self.status(f"start: target {self.target}, pool {len(self.pool)} instances, CARLA {self.cw}, Cosmos {self.cslots}, "
                    f"cores {len(self.row['cores'])}, disk {free:.0f} GB free")
        self.start_workers()
        from tqdm import tqdm
        bar = tqdm(total=self.target, desc="pairs (Cosmos done)", initial=self.done_pairs())
        while True:
            if (self.L / "DRAIN").exists():
                self.status("DRAIN: no new CARLA invocation")
                break
            p2, p3, new, info = self.plan()
            if len(new):
                self.append("variants.csv", new.to_dict("records"))
            V = self.table("variants.csv").set_index("pair")
            S = self.table("sel.csv").drop_duplicates("pair", keep="last").set_index("pair") if len(self.table("sel.csv")) else None
            ids = [V.loc[p, f"id{k}"] for p in p3 for k in PASS[3]] + [V.loc[p, f"id{k}"] for p in p2 for k in PASS[2]]
            ids += [r[f"id{k}"] for _, r in new.iterrows() for k in PASS[1]]
            self.status(f"plan: {info}; invocation {self.inv}: pass 2 {len(p2)}, re-render {len(p3)}, pass 1 {len(new)}")
            if not ids:
                if self.fut:
                    while self.fut:
                        self.poll()
                        time.sleep(20)
                    continue
                if "disk_pause" in info:
                    for _ in range(20):
                        self.poll()
                        time.sleep(30)
                    continue
                break
            win = {}
            if S is not None:
                for p in p2 + p3:
                    for k in PASS[2] + PASS[3]:
                        win[V.loc[p, f"id{k}"]] = [int(S.loc[p, "k0"]), int(S.loc[p, "k1"])]
            xml = full("xml") / f"inv{self.inv:03d}.xml"
            elems = route_elems(V.loc[p3].reset_index(), list(PASS[3])) if p3 else []
            elems += route_elems(V.loc[p2].reset_index(), list(PASS[2])) if p2 else []
            elems += route_elems(new, list(PASS[1])) if len(new) else []
            write_xml(elems, xml)
            t = time.time()
            res = self.carla(xml, ids, self.agent_json(win))
            self.status(f"invocation {self.inv}: {res['done']} / {res['ids']} routes done in {(time.time() - t) / 60:.0f} min, "
                        f"rc {res['rc']}")
            self.rl.event("invocation", inv=self.inv, s=time.time() - t, **{k: v for k, v in res.items() if k != "rc"})
            self.inv += 1
            # a pass-2 / re-render pair whose worlds did not finish (b2d_run already tried twice) is not tried again
            self.append("ctl.csv", [{"pair": p, "gen": g, "reason": "pass2_failed"} for g, ps, w in (("gen", p2, 2), ("gen3", p3, 3))
                                    for p in ps if not self.done2(V.loc[p], w)])
            if res["ids"] and res["done"] < 0.8 * res["ids"]:
                self.fail(f"invocation {self.inv - 1}: only {res['done']} / {res['ids']} routes finished")
            if len(new):
                self.select(new)
            bar.n = self.done_pairs()
            bar.refresh()
        (self.L / "CARLA_DONE").touch()
        while self.fut:
            self.poll()
            time.sleep(20)
        (self.L / "CONTROLS_DONE").touch()
        C = self.table("ctl.csv")
        self.status(f"CARLA and controls done: {C.drop_duplicates('pair', keep='last').reason.value_counts().to_dict() if len(C) else {}}")
        last = 0
        while any(p.poll() is None for p, *_ in self.workers.values()):
            self.poll()
            bar.n = self.done_pairs()
            bar.refresh()
            if time.time() - last > 3600:
                last = time.time()
                self.status(f"progress: {bar.n} / {self.target} pairs, disk {self.disk_gb():.0f} GB free")
            time.sleep(60)
        n = self.done_pairs()
        self.summary()
        if (self.L / "DRAIN").exists():
            self.status(f"drained with {n} pairs")
        elif n >= self.target:
            self.status(f"DONE: {n} pairs")
            (self.L / "DONE").write_text(f"{n} pairs\n")
        else:
            self.fail(f"finished with {n} < {self.target} pairs (see ctl.csv, clips/*/FAILED)")

    def summary(self):
        """Small tables for the repo: every variant with its selection / controls / Cosmos outcome."""
        V, S, C = self.table("variants.csv"), self.table("sel.csv"), self.table("ctl.csv")
        d = V.merge(S.drop_duplicates("pair", keep="last"), on="pair", how="left", suffixes=("", "_sel"))
        if len(C):
            c = C.drop_duplicates("pair", keep="last").add_prefix("ctl_").rename(columns={"ctl_pair": "pair"})
            d = d.merge(c, on="pair", how="left")
        done = [json.loads(f.read_text()) for f in full("pairs").glob("*/done.json")]
        if done:
            d = d.merge(pd.DataFrame(done).add_prefix("cosmos_").rename(columns={"cosmos_pair": "pair"}), on="pair", how="left")
        out = RES if full() == data_dir() / "runs" / "cosmos_full" else RES / full().name
        out.mkdir(parents=True, exist_ok=True)
        d.to_csv(out / "variants.csv", index=False)
        s = {"pairs_done": len(done), "variants": len(V), "selected_ok": int((S.reason == "ok").sum()) if len(S) else 0,
             "select_reasons": S.reason.value_counts().to_dict() if len(S) else {},
             "controls_reasons": C.drop_duplicates("pair", keep="last").reason.value_counts().to_dict() if len(C) else {},
             "qc_flags_first_render": int(((C.reason == "qc_render") & (C.gen == "gen")).sum()) if len(C) else 0,
             "cosmos_s_per_pair_median": float(np.median([x["wall_s"] for x in done])) if done else None,
             "instances_used": int(d[d.pair.isin([x["pair"] for x in done])].inst.nunique()) if done else 0,
             "wall_h": round((time.time() - self.t0) / 3600, 2)}
        (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
        return s


def checklist(stage: str, pairs: list | None = None) -> dict:
    """Staged-launch checklist (todo, registered before the full run) for runs/cosmos_full/<stage>: after
    scripts/cosmos_full_check.sh has run the v2 readouts (per_pair_G4b.csv)."""
    root = main_dir() / stage
    out = RES / stage
    V = pd.read_csv(root / "variants.csv", dtype={f"id{k}": str for k in WORLD})
    S = pd.read_csv(root / "sel.csv")
    C = pd.read_csv(root / "ctl.csv") if (root / "ctl.csv").exists() else pd.DataFrame(columns=["pair", "gen", "reason"])
    done = pd.DataFrame([json.loads(f.read_text()) for f in root.glob("pairs/*/done.json")])
    ids = [i for c in (f"id{k}" for k in WORLD) for i in V[c]]
    runs = {i: json.loads((root / "gen" / "done" / f"{i}.json").read_text()) for i in ids if (root / "gen" / "done" / f"{i}.json").exists()}
    tried = {i for i in ids if (root / "gen" / "attempts" / i).exists()}
    wall = {i: r.get("wall_s", np.nan) for i, r in runs.items()}
    per_pair_carla = [sum(wall.get(r[f"id{k}"], 0.0) for k in WORLD) for _, r in V[V.pair.isin(done.pair if len(done) else [])].iterrows()]
    fin = C.drop_duplicates("pair", keep="last")
    first = C[C.gen == "gen"]
    c = {"stage": stage, "variants": len(V), "routes_tried": len(tried), "routes_done": len(runs),
         "route_completion": len(runs) / max(len(tried), 1),
         "select": S.reason.value_counts().to_dict(), "select_ok_rate": float((S.reason == "ok").mean()),
         "controls_first_render": first.reason.value_counts().to_dict(),
         "qc_flag_rate_first_render": float((first.reason == "qc_render").mean()) if len(first) else np.nan,
         "nondeterministic": int((C.reason == "nondeterministic").sum()),
         "controls_final": fin.reason.value_counts().to_dict(), "pairs_done": len(done),
         "cosmos_s_per_pair": {"median": float((done.s_minus + done.s_plus).median()), "max": float((done.s_minus + done.s_plus).max()),
                               "card_peak_gib": float(done.card_peak_gib.max())} if len(done) else {},
         "carla_server_s_per_pair": {"median": float(np.median(per_pair_carla)), "max": float(np.max(per_pair_carla))} if per_pair_carla else {},
         "route_wall_s": {"median": float(np.nanmedian(list(wall.values()))) if wall else np.nan}}
    f = out / "per_pair_G4b.csv"
    if f.exists():
        pp = pd.read_csv(f)
        if pairs:
            pp = pp[pp.pair.isin(pairs)]
        big = pp.vis_frames >= 10
        pp["c_recall"] = ~big | (pp.R_plus >= 0.8 * pp.R_raw_plus)
        pp["c_halluc"] = pp.H_minus <= pp.H_raw_minus + 0.01
        pp["c_outside"] = (pp.psnr_pair >= 98) & (pp.lpips_pair <= 1e-6) & (pp.d_comp <= 1e-9)
        pp["c_ring"] = pp.ring_mad_pair <= 2 * pp.ring_mad_raw + 2
        pp["c_dv"] = pp.dv_med <= 1.0
        pp["all"] = pp.c_recall & pp.c_halluc & pp.c_outside & pp.c_ring
        cols = ["pair", "vis_frames", "R_raw_plus", "R_plus", "H_raw_minus", "H_minus", "psnr_pair", "lpips_pair", "d_comp",
                "ring_mad_pair", "ring_mad_raw", "dv_med", "lead_agree", "c_recall", "c_halluc", "c_outside", "c_ring", "c_dv", "all"]
        pp[cols].to_csv(out / "checklist_pairs.csv", index=False)
        c.update(pairs_checked=len(pp), pairs_all_pass=int(pp["all"].sum()), dv_le_1_share=float(pp.c_dv.mean()),
                 recall_ratio_min=float((pp.R_plus / pp.R_raw_plus.clip(lower=1e-6))[big].min()) if big.any() else np.nan,
                 ring_mad_pair_med=float(pp.ring_mad_pair.median()), ring_mad_raw_med=float(pp.ring_mad_raw.median()))
    # storage round trip: load_pair gives x+ == x- outside the blend support, for every stored pair
    rt = []
    for p in (done.pair if len(done) else []):
        a_, b_ = load_pair(p, root=root)
        sup = np.unpackbits(np.load(root / "pairs" / p / "gt.npz")["support"])[: a_.size // 3].reshape(a_.shape[:3]).astype(bool)
        rt.append(int(np.abs(a_.astype(np.int16) - b_)[~sup].max()))
    c["storage_outside_support_maxdiff"] = max(rt) if rt else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "checklist.json").write_text(json.dumps(c, indent=1, default=float))
    log.info("%s", json.dumps(c, indent=1, default=float))
    return c


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "run", "summary", "controls-test", "checklist"))
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--insts", default="", help="comma list of pool instances (staged pilots)")
    ap.add_argument("--keep-npy", action="store_true")
    ap.add_argument("--pair", default="")
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    insts = [int(x) for x in a.insts.split(",")] if a.insts else None
    if a.step == "build":
        build()
    elif a.step == "run":
        lane = Lane(a.target, insts, a.keep_npy)
        try:
            lane.run()
        except SystemExit:
            raise
        except BaseException as e:               # noqa: BLE001  (a crash of the driver itself is an ERROR too)
            import traceback
            lane.status(f"ERROR: driver crashed: {e!r}")
            (lane.L / "ERROR").write_text(traceback.format_exc())
            raise
    elif a.step == "summary":
        print(json.dumps(Lane(a.target, insts, a.keep_npy).summary(), indent=1, default=str))
    elif a.step == "controls-test":
        controls_test(a.pair)
    elif a.step == "checklist":
        checklist(a.stage)


if __name__ == "__main__":
    main()
