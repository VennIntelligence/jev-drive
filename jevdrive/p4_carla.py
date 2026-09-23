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
import os
import zlib
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
STEP_TICKS = 5                       # 0.25 s, the WOD-E2E past / future step
# docs/carla.md: every one of these crashed the server three times in the 220-route round
CRASHERS = {"3048", "11715", "11755", "23687", "23708", "3785", "3800", "23670", "23695", "24041", "24071"}
TURNING = ("SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurn",
           "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
           "NonSignalizedJunctionLeftTurnEnterFlow", "VanillaSignalizedTurnEncounterGreenLight",
           "VanillaSignalizedTurnEncounterRedLight", "VanillaNonSignalizedTurn",
           "VanillaNonSignalizedTurnEncounterStopsign", "T_Junction", "VehicleTurningRoute",
           "VehicleTurningRoutePedestrian", "BlockedIntersection", "OppositeVehicleRunningRedLight",
           "OppositeVehicleTakingPriority", "EnterActorFlow", "CrossingBicycleFlow")
LANE_CHANGE = ("HighwayExit", "HighwayCutIn", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2",
               "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "SequentialLaneChange")
PER_TYPE = 2                          # routes per Bench2Drive scenario type in wave 1; wave 2 adds a 3rd to the above
LARGE = ("Town11", "Town12", "Town13", "Town15")


def out_dir(*parts) -> Path:
    """processed/carla_p4 (P4_SET overrides it, e.g. for a dry run on the routes finished so far)."""
    p = data_dir() / "processed" / os.environ.get("P4_SET", "carla_p4") / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- routes

def select_routes(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(wave 1, wave 2) route lists. Wave 1: PER_TYPE routes of every Bench2Drive scenario type, towns not yet
    used by that type first, in a seeded order. Wave 2: one more route of every junction-turn and lane-change
    type, run only if wave 1 leaves fewer than 600 pre-onset candidates (the todo fixes this rule)."""
    root = ET.parse(data_dir() / B2D_XML).getroot()
    rows = []
    for r in root.findall("route"):
        sc = [s.get("type") for s in r.find("scenarios").findall("scenario")]
        w = r.find("weathers")
        wa = w.findall("weather")[0].attrib if w is not None else {}
        pts = np.array([[float(p.get("x")), float(p.get("y"))] for p in r.find("waypoints").findall("position")])
        rows.append({"route_id": r.get("id"), "town": r.get("town"), "scenario": ",".join(sc),
                     "length_m": float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
                     "sun_altitude": float(wa.get("sun_altitude_angle", "nan")),
                     "precipitation": float(wa.get("precipitation", "nan")), "fog": float(wa.get("fog_density", "nan"))})
    t = pd.DataFrame(rows)
    t = t[~t.route_id.isin(CRASHERS)].copy()
    t["r"] = np.random.default_rng(seed).random(len(t))
    w1, w2 = [], []
    for typ, g in t.sort_values("r").groupby("scenario", sort=True):
        order, seen = [], set()
        for _ in range(len(g)):                       # round-robin over towns in the seeded order
            nxt = next((i for i in g.index if i not in order and g.town[i] not in seen), None)
            if nxt is None:
                seen, nxt = set(), next(i for i in g.index if i not in order)
            order.append(nxt)
            seen.add(g.town[nxt])
        w1 += order[:PER_TYPE]
        if typ in TURNING + LANE_CHANGE:
            w2 += order[PER_TYPE:PER_TYPE + 1]
    out = [t.loc[w].drop(columns="r").reset_index(drop=True) for w in (w1, w2)]
    for name, o in zip(("wave 1", "wave 2"), out):
        log.info("%s: %d routes, %d types; per town %s", name, len(o), o.scenario.nunique(),
                 o.town.value_counts().sort_index().to_dict())
    return tuple(out)


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


LAYERS = ("pre_onset_lateral", "pre_onset_start", "pre_onset_brake", "in_turn", "stop_queue", "straight", "other")
N_MAX = 3000
PRE_CAP = {"pre_onset_lateral": 0.20, "pre_onset_start": 0.075, "pre_onset_brake": 0.075}   # of N_MAX
QUOTA = {"in_turn": 0.25, "straight": 0.25, "stop_queue": 0.10, "other": 0.05}             # of N; pre-onset 0.35


def layers(past: np.ndarray, fut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(layer, lateral pre-onset subtype) per frame, from the ego history and the recorded future only.

    `pre_onset_lateral`, `in_turn` and `straight` are `waymo.subsets`' pre_onset / turn_yaw / straight_yaw verbatim
    (the Waymo judge's definitions). The longitudinal pre-onsets are added for P4: start-from-stop (standing, then
    >= 3 m within 3 s) and brake onset (>= 3 m/s, not yet decelerating harder than 1 m/s^2, speed at 3 s halved).
    Priority is the order of LAYERS: a frame is in the first layer it qualifies for."""
    n = len(past)
    sub = waymo.subsets(pd.DataFrame({"intent": np.ones(n, np.int64)}), past, fut)
    v0 = np.linalg.norm(past[:, -1, 2:4], axis=1)
    a0 = past[:, -1, 4]
    k3 = int(round(3.0 / waymo.DT)) - 1
    d3 = np.linalg.norm(fut[:, k3, :2], axis=1)
    v3 = np.linalg.norm(fut[:, k3, :2] - fut[:, k3 - 1, :2], axis=1) / waymo.DT
    masks = {"pre_onset_lateral": sub["pre_onset"], "pre_onset_start": (v0 < 0.5) & (d3 >= 3.0),
             "pre_onset_brake": (v0 >= 3.0) & (a0 >= -1.0) & (v3 <= 0.5 * v0), "in_turn": sub["turn_yaw"],
             "stop_queue": (v0 < 0.5) & (d3 < 1.0), "straight": sub["straight_yaw"]}
    out = np.full(n, "other", dtype=object)
    for name in reversed(LAYERS[:-1]):
        out[masks[name]] = name
    seg = fut[:, -1, :2] - fut[:, -2, :2]
    psi5 = np.degrees(np.abs(np.arctan2(seg[:, 1], seg[:, 0])))
    y5 = np.abs(fut[:, -1, 1])
    kind = np.where(psi5 >= 30, "junction_turn", np.where((y5 >= 2.0) & (psi5 < 15), "lane_change", "other_lateral"))
    return out.astype(str), np.where(out == "pre_onset_lateral", kind, "").astype(str)


def select_frames(lay: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    """The pre-registered stratified draw. Pre-onset first, each kind up to its cap (lateral is the scarce one and
    is taken whole when short; brake onsets are plentiful and must not crowd it out). The pre-onset draw is then
    35 % of N, and the other layers fill the remaining 65 % in fixed proportions. A short layer is taken whole
    and reported, never back-filled from another."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(lay), bool)
    info = {"candidates": {k: int((lay == k).sum()) for k in LAYERS}}
    for name, cap in PRE_CAP.items():
        c = np.flatnonzero(lay == name)
        keep[rng.choice(c, min(int(round(cap * N_MAX)), len(c)), replace=False)] = True
    N = int(round(keep.sum() / 0.35))
    info["N"] = N
    for name, q in QUOTA.items():
        c = np.flatnonzero(lay == name)
        want = int(round(q * N))
        keep[rng.choice(c, min(want, len(c)), replace=False)] = True
        info[f"shortfall_{name}"] = max(0, want - len(c))
    info["selected"] = {k: int((keep & (lay == k)).sum()) for k in LAYERS}
    return keep, info


def build_index(gen: Path, seed: int = 0) -> pd.DataFrame:
    routes = pd.concat([pd.read_csv(RESULTS / f, dtype={"route_id": str}) for f in ("routes.csv", "routes_wave2.csv")])
    per = []
    for rid, town in zip(routes.route_id, routes.town):
        adir = _route_attempt(gen, rid)
        if adir is None or not (adir / "meta.json").exists() or not (adir / "p4_summary.json").exists():
            continue                              # not run (wave 2, or still running) or died before writing
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
    lay, kind = layers(past, fut)
    keep, info = select_frames(lay, seed)
    t["layer"], t["lateral_kind"] = lay, kind
    t, past, fut = t[keep].reset_index(drop=True), past[keep], fut[keep]
    (out_dir() / "selection.json").write_text(json.dumps(info, indent=1))
    log.info("selection: %s", info)
    t["v0"] = np.linalg.norm(past[:, -1, 2:4], axis=1)
    t["large_map"] = t.town.isin(LARGE)
    t["night"] = t.sun_altitude < 0
    d = out_dir()
    t.to_parquet(d / "index.parquet", index=False)
    np.save(d / "past.npy", past)
    np.save(d / "future.npy", fut)
    lk.to_csv(out_dir() / "intent_lookahead.csv", index=False)
    log.info("index: %d keyframes from %d routes; layers %s; lateral kinds %s; intent %s; towns %s", len(t),
             t.route_id.nunique(), t.layer.value_counts().to_dict(), t.lateral_kind.value_counts().to_dict(),
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


# ---------------------------------------------------------------- analysis helpers

TAPS = ("L18_last", "L18_mean")
MANEUVERS = ("still", "straight", "left", "right")
K_VOCAB = 1024


def maneuver(past: np.ndarray, fut: np.ndarray) -> np.ndarray:
    """0 still / 1 straight / 2 left / 3 right over the next 3 s, from the same chord bearing `waymo.subsets` uses."""
    v0 = np.linalg.norm(past[:, -1, 2:4], axis=1)
    b, _, chord = waymo.future_maneuver(fut)
    out = np.ones(len(v0), np.int64)
    out[(chord >= waymo.MIN_CHORD) & (b > waymo.ONSET_BEARING)] = 2
    out[(chord >= waymo.MIN_CHORD) & (b < -waymo.ONSET_BEARING)] = 3
    out[(v0 < 0.5) & (chord < 1.0)] = 0
    return out


def strata(v0: np.ndarray, lay: np.ndarray) -> np.ndarray:
    """Coarsened-exact-matching cells: speed in 2 m/s bins (20+ pooled) x the layer (LAYERS)."""
    li = pd.Categorical(lay, categories=LAYERS).codes.astype(np.int64)
    return np.minimum(v0 // 2, 10).astype(np.int64) * len(LAYERS) + li


def matched_sample(sw: np.ndarray, sc: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Positions into each domain such that every stratum holds min(n_W, n_C) rows of each."""
    rng = np.random.default_rng(seed)
    iw, ic = [], []
    for s in np.intersect1d(sw, sc):
        a, b = np.flatnonzero(sw == s), np.flatnonzero(sc == s)
        n = min(len(a), len(b))
        iw.append(rng.choice(a, n, replace=False))
        ic.append(rng.choice(b, n, replace=False))
    return np.concatenate(iw), np.concatenate(ic)


def reweight(values: np.ndarray, sw: np.ndarray, sc: np.ndarray) -> tuple[float, float]:
    """Mean of Waymo `values` re-weighted to CARLA's stratum shares; also the CARLA mass on strata Waymo lacks."""
    per = pd.Series(values).groupby(sw).mean()
    share = pd.Series(sc).value_counts(normalize=True)
    common = share.index.intersection(per.index)
    return float((per[common] * share[common]).sum() / share[common].sum()), float(1 - share[common].sum())


def _std(X: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = X[rows].double().mean(0), X[rows].double().std(0, correction=0)
    return mu.float(), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).float()


def logreg(X: torch.Tensor, y: torch.Tensor, lam: float, iters: int = 100, balanced: bool = True):
    """Multinomial (or binary, as 2 classes) logistic regression by full-batch L-BFGS; returns (W, b)."""
    n, d = X.shape
    k = int(y.max()) + 1
    cnt = torch.bincount(y, minlength=k).float()
    wt = (n / (k * cnt))[y] if balanced else torch.ones(n, device=X.device)
    W = torch.zeros(d, k, device=X.device, requires_grad=True)
    b = torch.zeros(k, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=10, tolerance_change=1e-7, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none") * wt).mean() \
            + 0.5 * lam * W.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Binary ROC AUC, or the macro one-vs-rest AUC when `s` has a column per class."""
    from sklearn.metrics import roc_auc_score
    if s.ndim == 1:
        return float(roc_auc_score(y, s))
    present = np.unique(y)
    return float(np.mean([roc_auc_score(y == c, s[:, c]) for c in present]))


def mlp_scores(Ztr: torch.Tensor, ytr: np.ndarray, Zte: torch.Tensor, epochs: int = 30, seed: int = 0) -> np.ndarray:
    """P(class 1) from a small class-balanced MLP (2560 -> 256 -> 2): the classifier that can use a difference
    in shape (covariance) and not only a shift, which is all a linear one sees once each class is centred."""
    torch.manual_seed(seed)
    d = Ztr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, 2)).to(Ztr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    y = torch.as_tensor(ytr, device=Ztr.device)
    cnt = torch.bincount(y, minlength=2).float()
    wt = len(y) / (2 * cnt)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(len(y), generator=g).split(512):
            b = b.to(Ztr.device)
            loss = torch.nn.functional.cross_entropy(net(Ztr[b]), y[b], weight=wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(Zte).softmax(1)[:, 1].cpu().numpy()


def domain_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, center=None, pca_k=None, lam=1e-3,
               model="linear", folds=5, seed=0, dev="cuda") -> float:
    """Out-of-fold AUC of a classifier for label `y`, folds grouped by sequence / route.

    Standardisation and PCA are estimated on the training folds only; per-class normalisation on all rows of the
    class (see below). `center="mean"` subtracts each class's own mean from its rows, `center="z"` also divides by
    its own per-dimension std. After that a linear classifier has nothing left to use (with equal class means the
    balanced logistic loss is minimised at w = 0, so it scores 0.5 by construction), so those variants are
    meaningful only with `model="mlp"`."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y, np.int64)
    folds = min(folds, *(len(np.unique(groups[y == c])) for c in (0, 1)))
    if folds < 2:
        return float("nan")                    # a class lives in a single group: not estimable
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = _std(Xt, tr)
        Z = (Xt - mu) / sd
        if center:
            # Each class is normalised with its own statistics *within each split*: training rows with the training
            # fold's, test rows with the test fold's (transductive, as per-domain standardisation is in deployment:
            # CARLA statistics come from the CARLA frames at hand, no task labels involved). Both simpler choices
            # are biased when a class has few groups (CARLA: ~90 routes): training-fold statistics leave the test
            # rows offset by the between-route spread (a Waymo-vs-Waymo dry run read 0.83), all-rows statistics
            # push the two folds to opposite sides of the mean (the null read 0.36).
            for rows_ in (tr, te):
                for c in (0, 1):
                    m = rows_[y[rows_] == c]
                    if len(m) > 1:
                        mc, sc = _std(Z, m)
                        Z[m] = (Z[m] - mc) / (sc if center == "z" else 1.0)
        if pca_k:
            ref = tr[y[tr] == 0]
            mref = Z[ref].mean(0)
            _, _, V = torch.linalg.svd(Z[ref] - mref, full_matrices=False)
            Z = (Z - mref) @ V[:pca_k].T
        if model == "mlp":
            oof[te] = mlp_scores(Z[tr], y[tr], Z[te], seed=seed)
        else:
            W, b = logreg(Z[tr], torch.as_tensor(y[tr], device=dev), lam)
            oof[te] = (Z[te] @ W + b).softmax(1)[:, 1].cpu().numpy()
        del Z
    ok = ~np.isnan(oof)
    return auc(y[ok], oof[ok]) if len(np.unique(y[ok])) == 2 else float("nan")


def js(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence in bits between two histograms."""
    p, q = p / p.sum() + eps, q / q.sum() + eps
    m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def boot_mean(v: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    from . import traj
    lo, hi = traj.boot_ci(np.asarray(v, np.float64), groups)
    return float(np.mean(v)), lo, hi


# ---------------------------------------------------------------- analysis

GROUPS = {"all": None, "pre-onset (3 kinds)": ("pre_onset_lateral", "pre_onset_start", "pre_onset_brake"),
          **{k: (k,) for k in LAYERS}}


def group_mask(lay: np.ndarray, g: str) -> np.ndarray:
    return np.ones(len(lay), bool) if GROUPS[g] is None else np.isin(lay, GROUPS[g])


def waymo_side() -> dict:
    """The P3(d'') frames: the frozen subset rows `qwenvid_p3` covers, with their labels and halves."""
    from . import waymo_ladder as lad
    ctx = lad.base_context()
    b = lad.align(ctx, lad.QWENVID_SET, list(TAPS))
    keep = lad.load_subset(ctx) & b["covered"]
    sel = np.flatnonzero(keep)
    rows = ctx["rows"][sel]
    return {"X": {t: np.asarray(b[t][sel], np.float32) for t in TAPS}, "fut": ctx["fut"][sel], "ego": ctx["ego"][sel],
            "past": ctx["past"][rows], "seq": ctx["seq"][sel], "half": ctx["half"][sel], "df": ctx["df"],
            "sub": {k: ctx["sub"][k][sel] for k in ctx["sub"]}, "cluster": ctx["df"].cluster.to_numpy()[rows],
            "fname": ctx["fname"][sel], "layer": layers(ctx["past"][rows], ctx["fut"][sel])[0]}


def carla_side() -> dict:
    t, past, fut = load_carla()
    ego = np.concatenate([waymo.ego_state(past), np.eye(len(waymo.INTENTS), dtype=np.float32)[t.intent.to_numpy()]], 1)
    sub = waymo.subsets(pd.DataFrame({"intent": t.intent.to_numpy()}), past, fut)
    lay = layers(past, fut)[0]
    assert (lay == t.layer.to_numpy()).all(), "layer labels changed since the index was built"
    return {"X": load_carla_features(TAPS), "fut": fut, "ego": ego.astype(np.float32), "past": past,
            "seq": ("carla-" + t.route_id).to_numpy(), "t": t, "sub": sub, "layer": lay}


def q1_domain(W: dict, C: dict, rl) -> pd.DataFrame:
    """Domain-classifier AUCs with the shape of the gap and the within-domain controls beside them."""
    sw, sc = strata(np.linalg.norm(W["past"][:, -1, 2:4], axis=1), W["layer"]), strata(C["t"].v0.to_numpy(), C["layer"])
    mw, mc = matched_sample(sw, sc)
    t = C["t"]
    rows = []

    def add(what, tap, X, y, g, **kw):
        a = domain_auc(X, y, g, **kw)
        r = {"comparison": what, "tap": tap, "n0": int((y == 0).sum()), "n1": int((y == 1).sum()),
             "groups": len(np.unique(g)), "auc": a}
        rows.append(r)
        rl.event("auc", **r)
        rl.log.info("AUC %-44s %-8s n=%d/%d groups=%d: %.4f", what, tap, r["n0"], r["n1"], r["groups"], a)

    for tap in TAPS:
        Xw, Xc = W["X"][tap], C["X"][tap]
        X = np.concatenate([Xw, Xc])
        y = np.r_[np.zeros(len(Xw)), np.ones(len(Xc))].astype(np.int64)
        g = np.r_[W["seq"], C["seq"]]
        Xm = np.concatenate([Xw[mw], Xc[mc]])
        ym = np.r_[np.zeros(len(mw)), np.ones(len(mc))].astype(np.int64)
        gm = np.r_[W["seq"][mw], C["seq"][mc]]
        add("Waymo vs CARLA", tap, X, y, g)
        add("Waymo vs CARLA, matched (v0 x layer)", tap, Xm, ym, gm)
        add("Waymo vs CARLA, MLP", tap, X, y, g, model="mlp")
        add("Waymo vs CARLA, per-domain centred, MLP", tap, X, y, g, center="mean", model="mlp")
        add("Waymo vs CARLA, per-domain z-scored, MLP", tap, X, y, g, center="z", model="mlp")
        add("Waymo vs CARLA, matched + centred, MLP", tap, Xm, ym, gm, center="mean", model="mlp")
        # null: a Waymo pseudo-domain of as many sequences and frames as CARLA has routes and frames, through the
        # same estimators -- what each AUC reads when there is no domain gap, at CARLA's group count
        useq = np.unique(W["seq"])
        for draw in range(3 if tap == TAPS[0] else 1):
            rng = np.random.default_rng(draw)
            pseudo = np.isin(W["seq"], rng.choice(useq, min(len(np.unique(C["seq"])), len(useq) // 3), replace=False))
            pi = np.flatnonzero(pseudo)
            pi = rng.choice(pi, min(len(pi), len(Xc)), replace=False)
            rest = np.flatnonzero(~pseudo)
            Xn = np.concatenate([Xw[rest], Xw[pi]])
            yn = np.r_[np.zeros(len(rest)), np.ones(len(pi))].astype(np.int64)
            gn = np.r_[W["seq"][rest], W["seq"][pi]]
            tag = f" (draw {draw})"
            add("null: Waymo pseudo-domain" + tag, tap, Xn, yn, gn)
            add("null: Waymo pseudo-domain, MLP" + tag, tap, Xn, yn, gn, model="mlp")
            add("null: Waymo pseudo-domain, centred, MLP" + tag, tap, Xn, yn, gn, center="mean", model="mlp")
            add("null: Waymo pseudo-domain, z-scored, MLP" + tag, tap, Xn, yn, gn, center="z", model="mlp")
            if tap == TAPS[0]:
                add("null: Waymo pseudo-domain, top-16 PCs" + tag, tap, Xn, yn, gn, pca_k=16)
        if tap != TAPS[0]:                  # the secondary tap gets the headline rows only
            continue
        for k in (1, 4, 16, 64):
            add(f"Waymo vs CARLA, top-{k} Waymo PCs", tap, X, y, g, pca_k=k)
        # per layer: Waymo frames of a layer against CARLA frames of the same layer
        for gname in GROUPS:
            if gname == "all":
                continue
            a, b = group_mask(W["layer"], gname), group_mask(C["layer"], gname)
            if a.sum() < 30 or b.sum() < 30 or len(np.unique(C["seq"][b])) < 5:
                continue
            Xl = np.concatenate([Xw[a], Xc[b]])
            yl = np.r_[np.zeros(a.sum()), np.ones(b.sum())].astype(np.int64)
            gl = np.r_[W["seq"][a], C["seq"][b]]
            add(f"layer {gname}: Waymo vs CARLA", tap, Xl, yl, gl)
            add(f"layer {gname}: Waymo vs CARLA, centred, MLP", tap, Xl, yl, gl, center="mean", model="mlp")
        # controls: how separable are things that are one domain?
        for model, tag in (("linear", ""), ("mlp", ", MLP")):
            add("control: CARLA Large Map vs small town" + tag, tap, Xc, t.large_map.to_numpy().astype(np.int64),
                C["seq"], model=model)
            if 0 < t.night.sum() < len(t):
                add("control: CARLA night vs day" + tag, tap, Xc, t.night.to_numpy().astype(np.int64), C["seq"],
                    model=model)
        rnd = pd.Series(W["seq"]).map(lambda q: zlib.crc32(str(q).encode()) % 2).to_numpy()
        add("control: Waymo random sequence halves", tap, Xw, rnd.astype(np.int64), W["seq"])
        vc = pd.Series(W["cluster"]).groupby(W["seq"]).first().value_counts()
        for cl in vc.index[:4]:
            add(f"control: Waymo cluster {cl} vs rest", tap, Xw, (W["cluster"] == cl).astype(np.int64), W["seq"])
    tab = pd.DataFrame(rows)
    tab.attrs["matched_n"] = len(mw)
    return tab


def q2_vocab(W: dict, C: dict, rl) -> tuple[pd.DataFrame, dict]:
    """Oracle minADE / minFDE and trust-region uncoverable share of the K=1024 train-split vocabulary."""
    from . import traj, waymo_heads as hd
    fut = hd.train_futures()
    anchors = traj.kmeans(torch.as_tensor(fut.reshape(len(fut), -1), device="cuda"), K_VOCAB, seed=0)
    region = traj.region_for(5.0, waymo.RFS_FREQ)
    per = {}
    for name, D in (("Waymo", W), ("CARLA", C)):
        v0 = np.linalg.norm(D["past"][:, -1, 2:4], axis=1)
        o = traj.oracle_metrics(anchors, D["fut"])
        unc = traj.vocab_coverage(anchors, D["fut"].astype(np.float64), v0, waymo.RFS_FREQ, region)
        per[name] = {"ade": o["oracle_ade"], "fde": o["oracle_fde"], "unc": unc, "v0": v0,
                     "s": strata(v0, D["layer"]), "seq": D["seq"], "layer": D["layer"]}
    rows = []
    cw, cc = per["Waymo"], per["CARLA"]
    for gname in GROUPS:
        a, b = group_mask(cw["layer"], gname), group_mask(cc["layer"], gname)
        if not b.any():
            continue
        for key, what in (("ade", "oracle minADE (m)"), ("fde", "oracle minFDE (m)"), ("unc", "uncoverable share")):
            wm, lost = reweight(cw[key][a], cw["s"][a], cc["s"][b])
            r = {"group": gname, "metric": what, "n_waymo": int(a.sum()), "n_carla": int(b.sum()),
                 "carla_routes": len(np.unique(cc["seq"][b])), "waymo": float(cw[key][a].mean()),
                 "carla": float(cc[key][b].mean()), "carla_ci": boot_mean(cc[key][b], cc["seq"][b])[1:],
                 "waymo_matched": wm, "ratio_matched": float(cc[key][b].mean() / wm) if key != "unc" else np.nan,
                 "excess_pp_matched": float(100 * (cc[key][b].mean() - wm)) if key == "unc" else np.nan,
                 "carla_mass_unmatched": lost}
            rows.append(r)
            rl.event("vocab", **r)
    tab = pd.DataFrame(rows)
    rl.log.info("vocabulary coverage\n%s", tab.to_markdown(index=False, floatfmt=".3f"))
    return tab, {"anchors": anchors, "per": per}


def q3_heads(W: dict, C: dict, anchors: torch.Tensor, rl, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Waymo-fitted heads applied to CARLA frames, per half-val direction."""
    from . import planner, traj, waymo_heads as hd, waymo_l0 as l0, waymo_stage_a as sa
    nW, nC = len(W["fut"]), len(C["fut"])
    fut = np.concatenate([W["fut"], C["fut"]]).astype(np.float32)
    ego = np.concatenate([W["ego"], C["ego"]])
    seq = np.r_[W["seq"], C["seq"]]
    past = np.concatenate([W["past"], C["past"]])
    ctrv = waymo.baselines(past)["ctrv"]
    cr = np.arange(nW, nW + nC)
    F = torch.as_tensor(fut.reshape(len(fut), -1), device="cuda")
    ids, _ = traj.nearest(F, anchors, 1)
    tgt = (ids, np.ones(ids.shape, np.float32))
    A = anchors.reshape(len(anchors), -1, 2).cpu().numpy()
    rows, per_frame, anchor_rows = [], {}, []
    sw = strata(np.linalg.norm(W["past"][:, -1, 2:4], axis=1), W["layer"])
    sc = strata(C["t"].v0.to_numpy(), C["layer"])
    for d in (0, 1):
        isw = np.r_[np.ones(nW, bool), np.zeros(nC, bool)]
        h = np.r_[W["half"], np.full(nC, -1)]
        sp = sa.Halves(None, seq, isw & (h == d), isw & (h == 1 - d), seed)
        ev = sp.val                                        # Waymo eval-half rows (positions into the concat)
        Xe = planner.standardize(torch.as_tensor(ego, device="cuda"), sp.train)
        _, st_e, W_ego = sa.ridge_cv(Xe, F, sp, fut)
        base = planner.linear_apply(W_ego, Xe, np.arange(len(fut)))[0]
        R = F - base
        res = R.reshape(-1, 20, 2).cpu().numpy()
        base_np = base.reshape(-1, 20, 2).cpu().numpy()
        preds = {"CTRV": ctrv, "ridge ego": base_np}
        for tap in TAPS:
            Xr = torch.as_tensor(np.concatenate([W["X"][tap], C["X"][tap]]), device="cuda")
            mu, sd = _std(Xr, sp.train)
            Xi = (Xr - mu) / sd
            _, st, Wv = sa.ridge_cv(Xi, R, sp, res)
            preds[f"ridge_late {tap}"] = (planner.linear_apply(Wv, Xi, np.arange(len(fut)))[0].reshape(-1, 20, 2)
                                          .cpu().numpy() + base_np)
            muc, sdc = _std(Xr, cr)                         # label-free: CARLA's own feature statistics
            Xpd = Xi.clone()
            Xpd[cr] = (Xr[cr] - muc) / sdc
            p = preds[f"ridge_late {tap}"].copy()
            p[cr] = planner.linear_apply(Wv, Xpd, cr)[0].reshape(-1, 20, 2).cpu().numpy() + base_np[cr]
            preds[f"ridge_late {tap} (per-domain std)"] = p
            rl.log.info("dir %d ridge_late %s: lambda %g", d, tap, st["lam"])
            if tap == TAPS[0]:
                # P3e's classification head on the same tap: ego classifier, then the vision head on its logits
                he = planner.Heads(Xe, sp, fut, F, waymo.RFS_FREQ)
                _, ste = he.cls(tgt, anchors, 1, keep=True)
                off = hd._cross_fit(he.scores, Xe, tgt, sp, ste["lam"], len(anchors))
                hv = planner.Heads(Xi, sp, fut, F, waymo.RFS_FREQ)
                pv, stv = hv.cls(tgt, anchors, 1, offset=off, keep=True)
                top = (hv.scores + off).argmax(1).cpu().numpy()
                top_e = off.argmax(1).cpu().numpy()
                agree = float((A[top[ev]] == pv[:, 0]).all((1, 2)).mean())
                rl.log.info("dir %d cls_late %s: lambda %g, recomputed top-1 agrees with Heads.cls on %.4f of rows",
                            d, tap, stv["lam"], agree)
                preds[f"cls ego K{len(A)}"] = A[top_e]
                preds[f"cls_late {tap} K{len(A)}"] = A[top]
                K = len(A)
                for dom, r in (("Waymo eval half", ev), ("CARLA", cr)):
                    anchor_rows.append({"direction": d, "domain": dom, "n": len(r),
                                        "top1_hit": float((top[r] == ids[r, 0]).mean()),
                                        "top1_perplexity": float(2 ** (-(lambda q: (q[q > 0] * np.log2(q[q > 0])).sum())(
                                            np.bincount(top[r], minlength=K) / len(r)))),
                                        "gt_perplexity": float(2 ** (-(lambda q: (q[q > 0] * np.log2(q[q > 0])).sum())(
                                            np.bincount(ids[r, 0], minlength=K) / len(r))))})
                anchor_rows.append({"direction": d, "domain": "JS(Waymo eval, CARLA)",
                                    "js_top1_bits": js(np.bincount(top[ev], minlength=K).astype(float),
                                                       np.bincount(top[cr], minlength=K).astype(float)),
                                    "js_gt_bits": js(np.bincount(ids[ev, 0], minlength=K).astype(float),
                                                     np.bincount(ids[cr, 0], minlength=K).astype(float)),
                                    "js_top1_vs_gt_waymo": js(np.bincount(top[ev], minlength=K).astype(float),
                                                              np.bincount(ids[ev, 0], minlength=K).astype(float)),
                                    "js_top1_vs_gt_carla": js(np.bincount(top[cr], minlength=K).astype(float),
                                                              np.bincount(ids[cr, 0], minlength=K).astype(float))})
                per_frame[f"top1_d{d}"] = top
            del Xr, Xi, Xpd
            torch.cuda.empty_cache()
        err = {k: l0.ade(v, fut) for k, v in preds.items()}
        per_frame[f"ade_d{d}"] = err
        base_err = err["ridge ego"]
        for dom, r in (("Waymo eval half", ev), ("CARLA", cr)):
            off_i = 0 if dom.startswith("Waymo") else nW
            lay = (W if dom.startswith("Waymo") else C)["layer"][r - off_i]
            sub = {g: group_mask(lay, g) for g in GROUPS}
            for sname, m in sub.items():
                if not m.any():
                    continue
                rr = r[m]
                for arm, e in err.items():
                    dv = e[rr] - base_err[rr]
                    mean, lo, hi = boot_mean(dv, seq[rr]) if arm != "ridge ego" else (0.0, 0.0, 0.0)
                    row = {"direction": d, "domain": dom, "subset": sname, "arm": arm, "n": len(rr),
                           "groups": len(np.unique(seq[rr])), "ade": float(e[rr].mean()),
                           "delta_vs_ego": mean, "lo": lo, "hi": hi}
                    if dom == "Waymo eval half":
                        cm = group_mask(C["layer"], sname)
                        if cm.any():
                            row["ade_reweighted_to_carla"], row["carla_mass_unmatched"] = reweight(
                                e[rr], sw[rr], sc[cm])
                    rows.append(row)
    tab = pd.DataFrame(rows)
    wv = tab[tab.domain == "Waymo eval half"].set_index(["direction", "subset", "arm"]).ade_reweighted_to_carla
    cm = tab.domain == "CARLA"                          # the matched CARLA / Waymo ADE ratio per arm and layer
    tab.loc[cm, "ratio_vs_waymo_matched"] = [
        a / wv.get((d, g, arm.replace(" (per-domain std)", "")), np.nan)
        for a, d, g, arm in zip(tab.ade[cm], tab.direction[cm], tab.subset[cm], tab.arm[cm])]
    rl.log.info("head transfer (all frames)\n%s", tab[tab.subset == "all"].to_markdown(index=False, floatfmt=".3f"))
    return tab, {"per_frame": per_frame, "anchors": pd.DataFrame(anchor_rows), "ids": ids, "nW": nW}


def q3b_probes(W: dict, C: dict, rl, seed: int = 0) -> pd.DataFrame:
    """Linear probes fitted on Waymo features (fit half), read on the Waymo eval half and on CARLA."""
    from . import waymo_stage_a as sa
    lams = (1e-4, 1e-3, 1e-2, 1e-1)
    lab = {}
    for name, D in (("W", W), ("C", C)):
        v0 = np.linalg.norm(D["past"][:, -1, 2:4], axis=1)
        mv = np.full(len(v0), -1)
        mv[v0 < 0.5], mv[v0 > 2.0] = 0, 1
        m = maneuver(D["past"], D["fut"])
        tn = np.where(m >= 1, m - 1, -1)                 # straight / left / right among moving frames
        lab[name] = {"moving (v0 > 2 vs < 0.5 m/s)": mv, "3 s manoeuvre (straight / left / right)": tn}
    rows = []
    for tap in TAPS:
        Xw = torch.as_tensor(W["X"][tap], device="cuda")
        Xc = torch.as_tensor(C["X"][tap], device="cuda")
        for d in (0, 1):
            fit, ev = np.flatnonzero(W["half"] == d), np.flatnonzero(W["half"] == 1 - d)
            sp = sa.Halves(None, W["seq"], W["half"] == d, W["half"] == 1 - d, seed)
            mu, sd = _std(Xw, fit)
            Zw, Zc = (Xw - mu) / sd, (Xc - mu) / sd
            muc, sdc = _std(Xc, np.arange(len(Xc)))
            Zc_pd = (Xc - muc) / sdc
            for task in lab["W"]:
                yw, yc = lab["W"][task], lab["C"][task]
                def score(Wb, Z, r):
                    p = (Z[r] @ Wb[0] + Wb[1]).softmax(1).cpu().numpy()
                    return p[:, 1] if p.shape[1] == 2 else p

                f, s_ = sp.fit[yw[sp.fit] >= 0], sp.sel[yw[sp.sel] >= 0]
                sel_auc = [auc(yw[s_], score(logreg(Zw[f], torch.as_tensor(yw[f], device="cuda"), l), Zw, s_))
                           for l in lams]
                best = lams[int(np.argmax(sel_auc))]
                tr = fit[yw[fit] >= 0]
                Wb = logreg(Zw[tr], torch.as_tensor(yw[tr], device="cuda"), best)
                e, c = ev[yw[ev] >= 0], np.flatnonzero(yc >= 0)
                a_w, a_c, a_pd = auc(yw[e], score(Wb, Zw, e)), auc(yc[c], score(Wb, Zc, c)), auc(yc[c], score(Wb, Zc_pd, c))
                r = {"tap": tap, "direction": d, "probe": task, "lam": best, "n_waymo_eval": len(e), "n_carla": len(c),
                     "carla_class_counts": np.bincount(yc[c]).tolist(), "auc_waymo": a_w, "auc_carla": a_c,
                     "auc_carla_per_domain_std": a_pd, "transfer": (a_c - 0.5) / (a_w - 0.5),
                     "transfer_per_domain_std": (a_pd - 0.5) / (a_w - 0.5)}
                rows.append(r)
                rl.event("probe", **r)
        del Xw, Xc
    tab = pd.DataFrame(rows)
    rl.log.info("probe transfer\n%s", tab.drop(columns="carla_class_counts").to_markdown(index=False, floatfmt=".3f"))
    return tab


def composition(W: dict, C: dict, heads: dict) -> pd.DataFrame:
    """The three frame sets side by side: all of Waymo val, the P3 subset the heads read, and CARLA's selection."""
    df = waymo.load_index()
    past, future = waymo.load_ego()
    m = (df.split == "val").to_numpy() & df.has_future.to_numpy()
    sets = {"Waymo val (all)": (past[m], waymo.future_xy(future[m]), df.intent.to_numpy()[m], df.sequence.to_numpy()[m]),
            "Waymo P3 subset": (W["past"], W["fut"], W["ego"][:, -4:].argmax(1), W["seq"]),
            "CARLA": (C["past"], C["fut"], C["t"].intent.to_numpy(), C["seq"])}
    cols = {}
    for name, (p, f, it, sq) in sets.items():
        v0 = np.linalg.norm(p[:, -1, 2:4], axis=1)
        w = np.abs(np.degrees(waymo.past_kinematics(p)["w"]))
        seg = f[:, -1, :2] - f[:, -2, :2]
        psi5 = np.degrees(np.abs(np.arctan2(seg[:, 1], seg[:, 0])))
        lay, kind = layers(p, f)
        c = {"frames": len(p), "sequences / routes": len(np.unique(sq)),
             **{f"v0 p{q} (m/s)": np.percentile(v0, q) for q in (10, 50, 90)},
             **{f"|yaw rate| p{q} (deg/s)": np.percentile(w, q) for q in (50, 90)},
             **{f"|heading change at 5 s| p{q} (deg)": np.percentile(psi5, q) for q in (50, 90)},
             **{f"layer {k}": float((lay == k).mean()) for k in LAYERS},
             **{f"lateral pre-onset: {k}": float((kind == k).sum() / max((lay == "pre_onset_lateral").sum(), 1))
                for k in ("junction_turn", "lane_change", "other_lateral")},
             **{f"intent {waymo.INTENTS[k]}": float((it == k).mean()) for k in (1, 2, 3)}}
        cols[name] = c
    # s_ego-style difficulty: the Waymo-fitted ridge ego's own 5 s ADE, in Waymo eval-half deciles (direction 0)
    e = heads["per_frame"]["ade_d0"]["ridge ego"]
    nW = heads["nW"]
    ev = np.flatnonzero(W["half"] == 1)
    edges = np.quantile(e[ev], np.linspace(0, 1, 11)[1:-1])
    for name, vals in (("Waymo P3 subset", e[ev]), ("CARLA", e[nW:])):
        k = np.searchsorted(edges, vals)
        for dcl in range(10):
            cols[name][f"s_ego decile {dcl + 1}"] = float((k == dcl).mean())
    t = C["t"]
    cols["CARLA"] |= {"night (sun altitude < 0)": float(t.night.mean()), "rain (precipitation > 0)": float(
        (t.precipitation > 0).mean()), "Large Map": float(t.large_map.mean()), "towns": int(t.town.nunique())}
    return pd.DataFrame(cols)


def analyze(rl):
    W, C = waymo_side(), carla_side()
    rl.log.info("Waymo: %d frames / %d sequences; CARLA: %d frames / %d routes", len(W["fut"]),
                len(np.unique(W["seq"])), len(C["fut"]), C["t"].route_id.nunique())
    t1 = q1_domain(W, C, rl)
    t1.to_csv(rl.dir / "q1_domain_auc.csv", index=False)
    t2, voc = q2_vocab(W, C, rl)
    t2.to_csv(rl.dir / "q2_vocab.csv", index=False)
    t3, heads = q3_heads(W, C, voc["anchors"], rl)
    t3.to_csv(rl.dir / "q3_heads.csv", index=False)
    heads["anchors"].to_csv(rl.dir / "q3_anchor_distribution.csv", index=False)
    t3b = q3b_probes(W, C, rl)
    t3b.to_csv(rl.dir / "q3b_probes.csv", index=False)
    comp = composition(W, C, heads)
    comp.to_csv(rl.dir / "composition.csv")
    rl.log.info("composition\n%s", comp.to_markdown(floatfmt=".3f"))
    # what the figures need, small
    Xw, Xc = W["X"]["L18_last"], C["X"]["L18_last"]
    mu, sd = Xw.mean(0), Xw.std(0) + 1e-6
    Zw, Zc = (Xw - mu) / sd, (Xc - mu) / sd
    _, _, Vt = np.linalg.svd(Zw[::4] - Zw[::4].mean(0), full_matrices=False)
    np.savez_compressed(rl.dir / "figdata.npz", pc_w=(Zw - Zw.mean(0)) @ Vt[:2].T, pc_c=(Zc - Zw.mean(0)) @ Vt[:2].T,
                        v0_w=voc["per"]["Waymo"]["v0"], v0_c=voc["per"]["CARLA"]["v0"],
                        ade_w=voc["per"]["Waymo"]["ade"], ade_c=voc["per"]["CARLA"]["ade"],
                        unc_w=voc["per"]["Waymo"]["unc"], unc_c=voc["per"]["CARLA"]["unc"],
                        town=C["t"].town.to_numpy().astype(str), nW=heads["nW"],
                        **{f"{k}__{a}": v for k, e in heads["per_frame"].items() if k.startswith("ade") for a, v in e.items()},
                        **{k: v for k, v in heads["per_frame"].items() if k.startswith("top1")},
                        gt_ids=heads["ids"][:, 0], anchors=voc["anchors"].cpu().numpy().reshape(-1, 20, 2))
    rl.log.info("done: %s", rl.dir)


# ---------------------------------------------------------------- figures

def _style():
    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_style", REPO / "research" / "plot_style.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    ps.apply()
    return ps


COLORS = {"Waymo": "#0072B2", "CARLA": "#D55E00"}


def _shrink_png(stem: Path, limit: int = 480 * 1024):
    """Photos and dense scatters do not compress in PNG; an adaptive 256-colour palette keeps them under the
    repo's ~500 KB rule without changing the pixel size."""
    from PIL import Image
    p = Path(str(stem) + ".png")
    img, dpi = Image.open(p).convert("RGB"), 300
    while p.stat().st_size > limit and dpi > 120:
        q = img.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q.save(p, optimize=True, dpi=(dpi, dpi))
        if p.stat().st_size > limit:              # a photo grid: fewer pixels, same physical size
            dpi = int(dpi * 0.8)
            img = img.resize((round(img.width * 0.8), round(img.height * 0.8)), Image.LANCZOS)
    return p.stat().st_size


def fig_rig(ps, out: Path, seed: int = 3):
    """One Waymo clip's last frame and one CARLA keyframe, the three cameras side by side, at matched scale."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from . import waymo_qwenvid as qv
    items, _ = qv.ref_items(400)
    it = items[np.random.default_rng(seed).integers(len(items))]
    ds = waymo.Shards([it], lambda imgs: imgs)
    w_imgs = ds[0][3::4]                                   # last frame of each camera's clip
    t, _, _ = load_carla()
    moving = t[(t.v0 > 4) & ~t.night].reset_index(drop=True)
    c_row = moving.iloc[np.random.default_rng(seed).integers(len(moving))]
    c_imgs = [Image.open(p).convert("RGB") for p in c_row.files[3::4]]
    order = (1, 0, 2)                                      # front_left, front, front_right on the page
    fig, axes = plt.subplots(2, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2 * ps.DOUBLE_COLUMN_IN / 3 * 1079 / 972 + 0.1),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})
    for r, (name, imgs) in enumerate((("Waymo", w_imgs), ("CARLA", c_imgs))):
        for j, k in enumerate(order):
            ax = axes[r, j]
            ax.imshow(imgs[k].resize((324, 360), Image.BILINEAR))
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            if j == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(waymo.CAMS[k].replace("_", " "))
    fig.subplots_adjust(left=0.04, right=1, top=0.95, bottom=0.01)
    info = ps.save(fig, out / "p4-rig")
    info["png_bytes"] = _shrink_png(out / "p4-rig")
    return info, {"carla_frame": c_row.frame_name}


def fig_gap(ps, run: Path, out: Path):
    import matplotlib.pyplot as plt
    q1 = pd.read_csv(run / "q1_domain_auc.csv")
    fd = np.load(run / "figdata.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    rng = np.random.default_rng(0)
    pw, pc = fd["pc_w"], fd["pc_c"]
    sw = rng.choice(len(pw), min(len(pw), 4000), replace=False)
    a.scatter(pw[sw, 0], pw[sw, 1], s=2, lw=0, alpha=0.35, color=COLORS["Waymo"], label="Waymo (P3 subset)",
              rasterized=True)
    a.scatter(pc[:, 0], pc[:, 1], s=2, lw=0, alpha=0.5, color=COLORS["CARLA"], label="CARLA", rasterized=True)
    a.set_xlabel("PC 1 of Waymo features")
    a.set_ylabel("PC 2")
    a.legend(markerscale=4, loc="best")
    ps.panel(a, "(a)")
    q = q1.copy()
    q["comparison"] = q.comparison.str.replace(r" \(draw \d\)", "", regex=True)
    agg = q.groupby(["comparison", "tap"], sort=False).auc.agg(["mean", "min", "max"]).reset_index()
    keep = ~agg.comparison.str.contains("top-1 |top-64 ", regex=True)
    t = agg[keep & (agg.tap == "L18_last")].reset_index(drop=True)
    m = agg[agg.tap == "L18_mean"].set_index("comparison")["mean"]
    y = np.arange(len(t))[::-1]
    ctrl = t.comparison.str.startswith(("control", "null")).to_numpy()
    col = np.where(ctrl, ps.BASELINE, COLORS["CARLA"])
    for yi, (mn, lo, hi, c) in enumerate(zip(t["mean"], t["min"], t["max"], col)):
        if hi > lo:
            b.errorbar(mn, y[yi], xerr=[[mn - lo], [hi - mn]], fmt="none", ecolor=c, lw=0.7, capsize=1.5)
    b.scatter(t["mean"], y, s=14, color=col, zorder=3, label="L18_last")
    b.scatter(m.reindex(t.comparison).to_numpy(), y, s=14, facecolors="none", edgecolors=col, zorder=3,
              label="L18_mean")
    b.axvline(0.5, color="#999999", lw=0.5)
    b.set_yticks(y)
    b.set_yticklabels([c.replace("Waymo vs CARLA, ", "").replace("Waymo vs CARLA", "raw").replace("control: ", "ctrl: ")
                       .replace(": raw", "").replace("null: Waymo pseudo-domain", "null")
                       for c in t.comparison], fontsize=6)
    b.set_xlim(0.4, 1.01)
    b.set_xlabel("out-of-fold ROC AUC")
    b.legend(loc="lower left", fontsize=7)
    ps.panel(b, "(b)")
    fig.tight_layout(pad=0.3)
    info = ps.save(fig, out / "p4-domain-gap")
    info["png_bytes"] = _shrink_png(out / "p4-domain-gap")
    return info


def fig_transfer(ps, run: Path, out: Path):
    """(a) vocabulary floor by speed, (b) the vision increment of ridge_late per layer, (c) layer composition."""
    import matplotlib.pyplot as plt
    fd = np.load(run / "figdata.npz")
    q3 = pd.read_csv(run / "q3_heads.csv")
    comp = pd.read_csv(run / "composition.csv", index_col=0)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    edges = np.arange(0, 22, 2)
    for name, v0, e in (("Waymo", fd["v0_w"], fd["ade_w"]), ("CARLA", fd["v0_c"], fd["ade_c"])):
        k = np.minimum(np.digitize(v0, edges) - 1, len(edges) - 2)
        mean = np.array([e[k == i].mean() if (k == i).sum() >= 10 else np.nan for i in range(len(edges) - 1)])
        a.plot(edges[:-1] + 1, mean, marker="o", ms=2.5, color=COLORS[name], label=name)
    a.set_xlabel("speed at $t_0$ (m/s)")
    a.set_ylabel("oracle minADE, K=1024 (m)")
    a.legend()
    ps.panel(a, "(a)")
    groups = ["all", "pre-onset (3 kinds)", "in_turn", "straight", "stop_queue"]
    labels = ["all", "pre-onset", "in-turn", "straight", "stop"]
    arm = "ridge_late L18_last"
    d0 = q3[q3.direction == 0]
    series = (("Waymo eval half", arm, COLORS["Waymo"], "Waymo eval half"),
              ("CARLA", arm, COLORS["CARLA"], "CARLA"),
              ("CARLA", arm + " (per-domain std)", ps.PALETTE["orange"], "CARLA, per-domain std"))
    x = np.arange(len(groups))
    for j, (dom, name, col, lab) in enumerate(series):
        r = d0[(d0.domain == dom) & (d0.arm == name)].set_index("subset").reindex(groups)
        b.errorbar(x + (j - 1) * 0.22, r.delta_vs_ego, yerr=[r.delta_vs_ego - r.lo, r.hi - r.delta_vs_ego], fmt="o",
                   ms=3, color=col, capsize=1.5, lw=0.8, label=lab)
    ps.zero_line(b)
    b.set_xticks(x)
    b.set_xticklabels(labels, fontsize=7)
    b.set_ylabel("ADE(ridge_late) $-$ ADE(ridge ego) (m)")
    b.legend(fontsize=6.5)
    ps.panel(b, "(b)")
    rows = [f"layer {k}" for k in LAYERS]
    cols = ["Waymo val (all)", "Waymo P3 subset", "CARLA"]
    bottom = np.zeros(len(cols))
    shades = ["#0072B2", "#56B4E9", "#CC79A7", "#009E73", "#999999", "#E69F00", "#DDDDDD"]
    for rname, col in zip(rows, shades):
        v = comp.loc[rname, cols].astype(float).to_numpy()
        c.bar(np.arange(len(cols)), v, bottom=bottom, color=col, width=0.6, label=rname.replace("layer ", "").replace("_", " "))
        bottom += v
    c.set_xticks(np.arange(len(cols)))
    c.set_xticklabels(["Waymo\nval", "Waymo\nP3 subset", "CARLA"], fontsize=7)
    c.set_ylabel("share of frames")
    ps.bars(c)
    c.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ps.panel(c, "(c)")
    fig.tight_layout(pad=0.3)
    return ps.save(fig, out / "p4-transfer")


def figs(run: Path):
    ps = _style()
    out = data_dir() / "runs" / "p4_carla" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    info = {"rig": fig_rig(ps, out), "gap": fig_gap(ps, run, out), "transfer": fig_transfer(ps, run, out)}
    (out / "figs.json").write_text(json.dumps(info, indent=1, default=str))
    log.info("figures in %s: %s", out, info)


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
        w1, w2 = select_routes()
        w1.to_csv(RESULTS / "routes.csv", index=False)
        w2.to_csv(RESULTS / "routes_wave2.csv", index=False)
        print(",".join(w1.route_id))
    elif a.step == "index":
        build_index(Path(a.gen))
    elif a.step == "extract":
        from .runlog import RunLog
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
        rl = RunLog("p4_carla", "extract")
        extract(rl, a.batch_size, a.workers)
        rl.close()
    elif a.step == "analyze":
        from .runlog import RunLog
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / total))
        rl = RunLog("p4_carla", "analyze")
        analyze(rl)
        rl.close()
    else:
        figs(Path(a.run))


if __name__ == "__main__":
    main()
