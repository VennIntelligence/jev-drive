"""P4: can a head trained on Waymo features read CARLA frames? (todos/2026-09-23-p4-carla-feature-gap.md)

  routes    freeze the Bench2Drive route list the generator drives (research/results/p4-carla-gap/routes.csv)
  index     turn the generator's per-route logs into WOD-E2E-shaped rows: 3 cameras x 4-frame clip at 0.2 s,
            16-step past (pos / vel / acc) and 20-step future in the current rear-axle frame, intent, weather
  extract   Qwen3-VL-4B native-video features over those clips with P3(d'')'s exact extractor, after a
            16-row Waymo equivalence check against `qwenvid_p3`
  analyze   Q1 domain AUC (+ matched, centred, PCA-k, controls), Q2 vocabulary coverage, Q3 head transfer,
            anchor distributions and probe transfer; tables into the run dir
  figs      the figures, from the run dir

The generator is scripts/p4_carla_agent.py, run through scripts/b2d_run.py in envs/carla.
Coordinates: CARLA is left-handed (+y right, yaw clockwise). Everything here is converted to the right-handed
Waymo convention first: +x forward, +y left, yaw counter-clockwise, origin at the rear axle.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import waymo
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "p4-carla-gap"
B2D_XML = "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
REAR_AXLE_X = -1.388633220
TICK = 0.05
CAM_TICKS = 4                        # the cameras report every 0.2 s = Waymo's stride 2 at 10 Hz
KEY_EVERY = 2                        # keyframes every 2nd camera frame, 0.4 s apart
STEP_TICKS = 5                       # 0.25 s, the WOD-E2E past / future step
# docs/carla.md: every one of these crashed the server three times in the 220-route round
CRASHERS = {"3048", "11715", "11755", "23687", "23708", "3785", "3800", "23670", "23695", "24041", "24071"}
# BehaviorAgent stops behind a static blockage and never changes lane around it; these types need that.
BLOCKING = ("Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
            "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays",
            "YieldToEmergencyVehicle", "ParkingExit", "InvadingTurn")
TURNING = ("SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurn",
           "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
           "NonSignalizedJunctionLeftTurnEnterFlow", "VanillaSignalizedTurnEncounterGreenLight",
           "VanillaSignalizedTurnEncounterRedLight", "VanillaNonSignalizedTurn",
           "VanillaNonSignalizedTurnEncounterStopsign", "T_Junction", "VehicleTurningRoute",
           "VehicleTurningRoutePedestrian", "BlockedIntersection", "OppositeVehicleRunningRedLight",
           "OppositeVehicleTakingPriority", "EnterActorFlow", "CrossingBicycleFlow", "HighwayExit")
QUOTA = {"Town12": 12, "Town13": 6}   # every other town: up to 3
LARGE = ("Town11", "Town12", "Town13", "Town15")


def out_dir(*parts) -> Path:
    p = data_dir() / "processed" / "carla_p4" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- routes

def select_routes(seed: int = 0) -> pd.DataFrame:
    """The frozen route list: every town, Large Maps weighted up, turning scenarios first within a town."""
    root = ET.parse(data_dir() / B2D_XML).getroot()
    rows = []
    for r in root.findall("route"):
        sc = [s.get("type") for s in r.find("scenarios").findall("scenario")]
        w = r.find("weathers")
        sun = float(w.findall("weather")[0].get("sun_altitude_angle")) if w is not None else np.nan
        pts = np.array([[float(p.get("x")), float(p.get("y"))] for p in r.find("waypoints").findall("position")])
        rows.append({"route_id": r.get("id"), "town": r.get("town"), "scenario": ",".join(sc),
                     "length_m": float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), "sun_altitude": sun})
    t = pd.DataFrame(rows)
    ok = ~t.route_id.isin(CRASHERS) & ~t.scenario.isin(BLOCKING)
    t = t[ok].copy()
    t["turning"] = t.scenario.isin(TURNING)
    rng = np.random.default_rng(seed)
    t["r"] = rng.random(len(t))
    pick = []
    for town, g in t.sort_values(["turning", "r"], ascending=[False, True]).groupby("town", sort=True):
        n = QUOTA.get(town, 3)
        # a turning-first list, but at most 2/3 turning so a town also contributes straight driving
        turn, rest = g[g.turning], g[~g.turning]
        k = min(len(turn), max(n - len(rest), int(np.ceil(2 * n / 3))))
        pick.append(pd.concat([turn.head(k), rest.head(n - k)]).head(n))
    out = pd.concat(pick).drop(columns="r").reset_index(drop=True)
    log.info("routes: %d of %d eligible (%d total); per town %s; turning %d", len(out), int(ok.sum()), len(ok),
             out.town.value_counts().sort_index().to_dict(), int(out.turning.sum()))
    return out


# ---------------------------------------------------------------- index

def _route_attempt(gen: Path, rid: str) -> Path | None:
    """The attempt with the most camera frames (a crashed attempt keeps what it wrote before it died)."""
    best, n = None, 0
    for a in sorted((gen / "attempts" / rid).glob("*")):
        f = a / "frames.jsonl"
        k = sum(1 for _ in open(f)) if f.exists() else 0
        if k > n:
            best, n = a, k
    return best


def _rot(v: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Rotate 2-vectors by -th: world -> the frame whose heading is th."""
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def route_rows(adir: Path, rid: str, town: str):
    """Keyframes of one route as (rows, past (n, 16, 6), future (n, 20, 2))."""
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame").sort_index()
    frames = pd.read_json(adir / "frames.jsonl", lines=True).sort_values("frame").reset_index(drop=True)
    route = pd.read_json(adir / "route.json")
    meta = json.loads((adir / "meta.json").read_text())
    f0, f1 = pose.index.min(), pose.index.max()
    full = pd.RangeIndex(f0, f1 + 1)
    pose = pose.reindex(full)                     # a missing tick shows up as NaN and rejects the windows over it
    # right-handed world, rear axle
    th = -np.radians(pose.yaw.to_numpy())
    c = np.stack([pose.x.to_numpy(), -pose.y.to_numpy()], -1)
    head = np.stack([np.cos(th), np.sin(th)], -1)
    ra = c + REAR_AXLE_X * head
    w = -np.radians(pose.wz.to_numpy())           # yaw rate, rad/s, counter-clockwise
    v = np.stack([pose.vx.to_numpy(), -pose.vy.to_numpy()], -1)
    off = REAR_AXLE_X * head
    v_ra = v + w[:, None] * np.stack([-off[:, 1], off[:, 0]], -1)
    a_ra = np.gradient(v_ra, TICK, axis=0)
    a_ra = pd.DataFrame(a_ra).rolling(5, center=True, min_periods=1).mean().to_numpy()   # 0.25 s, like Waymo's
    pos = {f: i for i, f in enumerate(full)}

    cam = frames.frame.to_numpy()
    rows, past, fut = [], [], []
    for i in range(3, len(cam)):
        clip = cam[i - 3:i + 1]
        if not (np.diff(clip) == CAM_TICKS).all():
            continue
        k = pos.get(int(cam[i]))
        if k is None:
            continue
        pk = k + STEP_TICKS * np.arange(-15, 1)
        fk = k + STEP_TICKS * np.arange(1, 21)
        if pk[0] < 0 or fk[-1] >= len(full):
            continue
        idx = np.r_[pk, fk]
        if np.isnan(ra[idx]).any() or np.isnan(v_ra[pk]).any():
            continue
        p = _rot(ra[pk] - ra[k], th[k])
        vv, aa = _rot(v_ra[pk], th[k]), _rot(a_ra[pk], th[k])
        vv[-1], aa[-1] = vv[-2], aa[-2]           # WOD-E2E repeats the previous sample in the last slot
        past.append(np.concatenate([p, vv, aa], -1))
        fut.append(_rot(ra[fk] - ra[k], th[k]))
        files = [str(adir / frames.files[j][cam_name]) for cam_name in waymo.CAMS for j in range(i - 3, i + 1)]
        rows.append({"frame_name": f"{rid}-{int(cam[i]):07d}", "route_id": rid, "town": town, "frame": int(cam[i]),
                     "cam_index": i, "t": float(pose.t.iloc[k]), "files": files,
                     "route_progress": _progress(route, c[k]), "min_std": float(np.min(frames["std"][i]))})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    wth = meta["weather"]
    t["sun_altitude"] = wth["sun_altitude_angle"]
    t["precipitation"] = wth["precipitation"]
    t["fog"] = wth["fog_density"]
    return t, np.asarray(past, np.float32), np.asarray(fut, np.float32), route


def _progress(route: pd.DataFrame, xy_rh: np.ndarray) -> int:
    """Index of the dense route point nearest to a right-handed world position."""
    return int(np.argmin((route.x.to_numpy() - xy_rh[0]) ** 2 + (-route.y.to_numpy() - xy_rh[1]) ** 2))


def route_intent(route: pd.DataFrame, prog: np.ndarray, lookahead_m: float) -> np.ndarray:
    """WOD-E2E intent from the route plan: the first junction command (LEFT = 1, RIGHT = 2) within
    `lookahead_m` of route arc length ahead of the car, else GO_STRAIGHT."""
    xy = np.stack([route.x.to_numpy(), route.y.to_numpy()], -1)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    opt = route.option.to_numpy()
    out = np.ones(len(prog), np.int64)
    for j, p in enumerate(prog):
        ahead = np.flatnonzero((s >= s[p]) & (s <= s[p] + lookahead_m) & np.isin(opt, (1, 2)))
        if len(ahead):
            out[j] = 2 if opt[ahead[0]] == 1 else 3
    return out


def calibrate_lookahead(carla_rows: list, grid=(10, 15, 20, 30, 40, 60)) -> tuple[float, pd.DataFrame]:
    """The route lookahead whose P(turn intent | turn within 3 s) matches Waymo val's.

    Waymo's intent is a routing signal whose horizon is not documented; matching this one conditional on the
    trajectories alone (no features involved) fixes the CARLA side's horizon before anything is compared."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    m = (df.split == "val").to_numpy() & df.has_future.to_numpy()
    b, _, chord = waymo.future_maneuver(future[m])
    turn = (np.abs(b) > waymo.ONSET_BEARING) & (chord >= waymo.MIN_CHORD)
    it = df.intent.to_numpy()[m]
    target = float(np.isin(it[turn], (2, 3)).mean())
    rows = []
    for L in grid:
        hit, n, share = 0, 0, []
        for t, fut, route in carla_rows:
            ci = route_intent(route, t.route_progress.to_numpy(), L)
            bb, _, cc = waymo.future_maneuver(fut)
            tt = (np.abs(bb) > waymo.ONSET_BEARING) & (cc >= waymo.MIN_CHORD)
            hit += int(np.isin(ci[tt], (2, 3)).sum())
            n += int(tt.sum())
            share.append(np.isin(ci, (2, 3)))
        rows.append({"lookahead_m": L, "p_turn_intent_given_turn": hit / max(n, 1),
                     "turn_intent_share": float(np.concatenate(share).mean()), "waymo_target": target})
    tab = pd.DataFrame(rows)
    best = float(tab.lookahead_m[(tab.p_turn_intent_given_turn - target).abs().idxmin()])
    log.info("intent lookahead: Waymo P(turn intent | turn within 3 s) = %.3f -> %g m\n%s", target, best,
             tab.to_markdown(index=False, floatfmt=".3f"))
    return best, tab


def build_index(gen: Path, seed: int = 0) -> pd.DataFrame:
    routes = pd.read_csv(RESULTS / "routes.csv", dtype={"route_id": str})
    per = []
    for rid, town in zip(routes.route_id, routes.town):
        adir = _route_attempt(gen, rid)
        if adir is None or not (adir / "meta.json").exists():
            log.warning("route %s: no usable attempt", rid)
            continue
        r = route_rows(adir, rid, town)
        if r is None:
            log.warning("route %s: no complete keyframe window (%s)", rid, adir)
            continue
        t, past, fut, route = r
        t["attempt"] = adir.name
        per.append((t, past, fut, route))
        log.info("route %s %-8s %s: %d candidate frames", rid, town, adir.name, len(t))
    L, lk = calibrate_lookahead([(t, f, r) for t, _, f, r in per])
    for t, _, _, route in per:
        t["intent"] = route_intent(route, t.route_progress.to_numpy(), L)
    t = pd.concat([p[0] for p in per], ignore_index=True)
    past = np.concatenate([p[1] for p in per])
    fut = np.concatenate([p[2] for p in per])
    # keyframes: every KEY_EVERY-th camera frame of a route, then stationary frames capped at Waymo's share
    keep = (t.cam_index.to_numpy() % KEY_EVERY) == 0
    v0 = np.linalg.norm(past[:, -1, 2:4], axis=1)
    wdf = waymo.load_index()
    wpast, _ = waymo.load_ego()
    wval = (wdf.split == "val").to_numpy() & wdf.has_future.to_numpy()
    w_still = float((np.linalg.norm(wpast[wval, -1, 2:4], axis=1) < 0.5).mean())
    still = keep & (v0 < 0.5)
    cap = int(w_still / (1 - w_still) * (keep & ~still).sum())
    if still.sum() > cap:
        drop = np.random.default_rng(seed).choice(np.flatnonzero(still), int(still.sum()) - cap, replace=False)
        keep[drop] = False
    t, past, fut = t[keep].reset_index(drop=True), past[keep], fut[keep]
    t["v0"] = np.linalg.norm(past[:, -1, 2:4], axis=1)
    t["large_map"] = t.town.isin(LARGE)
    t["night"] = t.sun_altitude < 0
    d = out_dir()
    t.to_parquet(d / "index.parquet", index=False)
    np.save(d / "past.npy", past)
    np.save(d / "future.npy", fut)
    lk.to_csv(RESULTS / "intent_lookahead.csv", index=False)
    log.info("index: %d keyframes from %d routes (%d stationary kept; Waymo val stationary share %.3f); "
             "intent %s; towns %s", len(t), t.route_id.nunique(), int((t.v0 < 0.5).sum()), w_still,
             t.intent.value_counts().sort_index().to_dict(), t.town.value_counts().sort_index().to_dict())
    return t


def load_carla():
    d = out_dir()
    return pd.read_parquet(d / "index.parquet"), np.load(d / "past.npy"), np.load(d / "future.npy")


# ---------------------------------------------------------------- features

FEATURE_SET = "carla_p4"


class ClipFiles(torch.utils.data.Dataset):
    """One item = 12 JPEG paths, camera-major and oldest first, the layout `waymo.Shards` hands over."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform([Image.open(p).convert("RGB") for p in self.items[i]])


def extract(rl, batch: int = 2, workers: int = 4, n_check: int = 16):
    """P3(d'')'s extractor over the CARLA clips, after proving on Waymo rows that nothing but the reader changed."""
    from . import features as F, waymo_qwenvid as qv
    fx = qv.make_fx(compile=False)
    items, idx = qv.ref_items(n_check)
    chk = data_dir() / "scratch" / "p4_check" / rl.dir.name
    chk.mkdir(parents=True, exist_ok=True)
    F.extract(fx, items, batch, workers, chk, None, "p4/check", dataset=waymo.Shards)
    eq = qv.compare(chk, idx)
    rl.event("equivalence", **{k: v for k, v in eq.items()})
    rl.log.info("equivalence on %d qwenvid_p3 rows: %s", n_check, eq)
    t, _, _ = load_carla()
    dst = out_dir("features", FEATURE_SET)
    st = F.extract(fx, t.files.map(list).tolist(), batch, workers, dst, rl, "p4/carla", dataset=ClipFiles)
    t[["frame_name", "route_id", "town"]].to_parquet(dst / "index.parquet", index=False)
    (dst / "meta.json").write_text(json.dumps({"set": FEATURE_SET, "recipe": "P3(d'') qwenvid", "frames_per_clip": 4,
                                               "clip_stride_s": 0.2, "layer": 18, "batch_size": batch,
                                               "compile": False, "equivalence": eq, **st}, indent=2, default=float))
    rl.log.info("carla features: %d clips, %.1f ms/frame, peak %.1f GB", st["n"], st["ms_per_frame"], st["peak_vram_gb"])


def load_carla_features(taps=("L18_last", "L18_mean")) -> dict:
    d = out_dir("features", FEATURE_SET)
    t, _, _ = load_carla()
    idx = pd.read_parquet(d / "index.parquet")
    assert (idx.frame_name.to_numpy() == t.frame_name.to_numpy()).all(), "feature rows out of step with the index"
    return {k: np.load(d / f"{k}.npy").astype(np.float32) for k in taps}


# ---------------------------------------------------------------- entry point

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("routes", "index", "extract", "analyze", "figs"))
    ap.add_argument("--gen", default=str(data_dir() / "runs" / "p4_carla" / "gen"), help="b2d_run.py --out dir")
    ap.add_argument("--run", default=None, help="figs: the analyze run dir")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vram-gb", type=float, default=14.0)
    a = ap.parse_args()
    if a.step == "routes":
        RESULTS.mkdir(parents=True, exist_ok=True)
        t = select_routes()
        t.to_csv(RESULTS / "routes.csv", index=False)
        print(",".join(t.route_id))
    elif a.step == "index":
        build_index(Path(a.gen))
    elif a.step == "extract":
        from .runlog import RunLog
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
        rl = RunLog("p4_carla", "extract")
        extract(rl, a.batch_size, a.workers)
        rl.close()
    else:
        raise SystemExit(f"step {a.step} is added below")


if __name__ == "__main__":
    main()
