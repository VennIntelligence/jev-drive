"""Cosmos-Transfer2.5 pilot: CARLA counterfactual pairs re-rendered photoreal (todos/2026-09-28-cosmos-pilot.md).

  select     10 P5 v1 BehaviorAgent pedestrian pairs (small towns, seed 0) and a 93-tick window each, ending at the
             last tick the two egos share -> research/results/cosmos/pairs.csv, runs/cosmos/agent.json
  controls   per re-rendered world: determinism check against the P5 v1 attempt, then the Cosmos inputs (rgb / edge /
             seg / depth mp4) and the hazard ground truth (mask per frame) -> runs/cosmos/clips/<pair>/<member>/
  specs      Cosmos inference specs (jsonl) for a control variant and a set of pairs; both members of a pair get the
             same prompt, seed and settings
The GPU work lives in scripts/cosmos_gen.sh (CARLA re-render), scripts/cosmos_infer.py (Cosmos, envs/cosmos-transfer),
scripts/cosmos_openpilot.py (openpilot, envs/openpilot) and jevdrive/cosmos_eval.py (checks 1-4, figures).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "cosmos"
P5 = "processed/carla_p5v1_ba"
GEN_V1 = "runs/p5v1/gen-ba"
T = 93                     # Cosmos chunk (the distilled model takes exactly 93 frames); 4.65 s at CARLA's 20 Hz
FPS = 20
PED = ("PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "VehicleTurningRoutePedestrian")
SMALL = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town07", "Town10HD", "Town11")
# the pilot's base routes: small-town pedestrian routes of P5 v1 (27529 dropped: hazard visible for < 93 ticks)
BASES = ("24211", "24224", "24294", "24206", "27515", "27582", "24519", "25863", "24252", "27297")
PILOT1 = "24211"


def root(*parts) -> Path:
    p = data_dir() / "runs" / "cosmos" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pairs() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "pairs.csv", dtype={"base_id": str, "plus": str, "minus": str})


def select():
    p = pd.read_csv(data_dir() / P5 / "pairs.csv", dtype={"base_id": str, "plus": str, "minus": str})
    o = pd.read_parquet(data_dir() / P5 / "obs.parquet")
    p = p[p.base_id.isin(BASES) & (p.seed == 0) & (p.reason == "ok")].copy()
    assert len(p) == len(BASES) and p.family.isin(PED).all() and p.town.isin(SMALL).all(), p
    # The window ends at the earlier of: the last tick the egos share (t_div - 1), and 5 ticks after the hazard was
    # last visible in P5's front view (>= 100 px at half resolution). BehaviorAgent often waits at a light or never
    # reacts, so t_div can come long after the pedestrian has left the image.
    from . import p5_pairs as PP
    last = {}
    for _, r in p.iterrows():
        A = PP.load_world(PP.attempt(data_dir() / GEN_V1, r.plus))
        hz = [str(h) for h in A["hazards"]]
        px = A["frames"].px.map(lambda d: max([d.get(h, 0) for h in hz] + [0]) if isinstance(d, dict) else 0)
        last[r.base_id] = int(px.index[px >= 100].max())
    p["k_lastvis"] = p.base_id.map(last)
    p["k1"] = np.minimum(p.t_div.astype(int), p.k_lastvis + 5)
    p["k0"] = p.k1 - T
    assert (p.k0 >= 8).all(), p[["base_id", "k0"]]
    rows = []
    for _, r in p.iterrows():
        g = o[(o.base_id == r.base_id) & (o.seed == 0) & (o.k >= r.k0) & (o.k < r.k1)]
        rows.append({"obs_frames": len(g), "px_max": int(g.factor_px.max()) if len(g) else 0,
                     "vis_frac": float((g.factor_px >= 100).mean()) if len(g) else 0.0,
                     "impure_max": int(g.impure_visible.max()) if len(g) else 0, "v0_mean": float(g.v0.mean()) if len(g) else np.nan})
    p = pd.concat([p.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    p["pair"] = p.base_id + "-s0"
    cols = ["pair", "base_id", "seed", "town", "family", "plus", "minus", "t_vis", "t_div", "k_lastvis", "k0", "k1", "obs_frames",
            "px_max", "vis_frac", "impure_max", "v0_mean"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    p[cols].to_csv(RESULTS / "pairs.csv", index=False)
    tf = json.loads((data_dir() / "runs/p5_pairs/agent_config.json").read_text())["tfv6_model_dir"]
    win = {r[w]: [int(r.k0), int(r.k1)] for _, r in p.iterrows() for w in ("plus", "minus")}
    (root() / "agent.json").write_text(json.dumps({"tfv6_model_dir": tf, "save_threads": 3, "cosmos_windows": win}, indent=1))
    log.info("\n%s", p[cols].to_markdown(index=False))
    return p


# ------------------------------------------------------------------ controls (CPU, envs/jevdrive)

W, H, FOV = 1280, 704, 64.0                        # scripts/cosmos_pair_agent.CAM
F = W / 2.0 / np.tan(np.radians(FOV) / 2.0)
THINGS = set(range(12, 20))                       # pedestrian, rider, car, truck, bus, train, motorcycle, bicycle
PED_TAG = 12
VIS_PX = 300                                      # GT "visible": hazard mask >= 300 px (todo, check 1)
REGION_DILATE = 24                                # check 2: mask region dilation (px)
CORRIDOR_M, CORRIDOR_HALF = 40.0, 3.0


def ffmpeg_bin() -> str:
    import glob
    import shutil
    return shutil.which("ffmpeg") or sorted(glob.glob(str(data_dir() / "envs/*/lib/python3*/site-packages/"
                                                           "imageio_ffmpeg/binaries/ffmpeg-linux-*")))[0]


def write_mp4(path: Path, frames: np.ndarray, fps: int = FPS, lossless: bool = True):
    """(T, H, W, 3) uint8 RGB -> H.264 mp4 (lossless yuv444p for controls and inputs, crf 18 yuv420p for viewing)."""
    import subprocess
    t, h, w, _ = frames.shape
    enc = ["-c:v", "libx264", "-preset", "veryfast"] + (["-crf", "0", "-pix_fmt", "yuv444p"] if lossless else
                                                         ["-crf", "18", "-pix_fmt", "yuv420p"])
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *enc, str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def read_mp4(path: Path) -> np.ndarray:
    import subprocess
    out = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    probe = np.frombuffer(out, np.uint8)
    return probe.reshape(-1, H, W, 3)


def _color(key: int) -> np.ndarray:
    """Fixed pseudo-random, well saturated colour per (tag, instance) key: the same object gets the same colour in
    both members (SAM2-style instance colouring, cosmos seg augmentor)."""
    import colorsys
    r = np.random.default_rng(key * 2654435761 % (2 ** 32))
    return np.array(colorsys.hsv_to_rgb(r.uniform(), r.uniform(0.5, 1.0), r.uniform(0.5, 1.0))) * 255


def seg_image(tag: np.ndarray, inst: np.ndarray, cache: dict) -> np.ndarray:
    key = np.where(np.isin(tag, list(THINGS)), tag.astype(np.int64) * 65536 + inst, tag.astype(np.int64))
    uk, inv = np.unique(key, return_inverse=True)
    lut = np.stack([cache.setdefault(int(k), _color(int(k))) for k in uk]).astype(np.uint8)
    return lut[inv.reshape(tag.shape)]


def depth_image(d: np.ndarray) -> np.ndarray:
    """Relative inverse depth as grey (near bright, sky dark), with a FIXED scale (1.5 m = white), so both members of
    a pair get identical values wherever the scene is identical (Video Depth Anything's per-clip min-max would not)."""
    v = np.clip(1.5 / np.maximum(d, 1e-3), 0, 1) ** 0.5 * 255
    return np.repeat(v[..., None], 3, -1).astype(np.uint8)


def edge_image(seg: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Variant (b) edges: instance / class boundaries plus depth discontinuities (> 8% in log depth), 1 px."""
    import cv2
    g = seg.astype(np.int32)
    code = g[..., 0] * 65536 + g[..., 1] * 256 + g[..., 2]
    e = np.zeros(code.shape, bool)
    e[:, 1:] |= code[:, 1:] != code[:, :-1]
    e[1:, :] |= code[1:, :] != code[:-1, :]
    ld = np.log(np.maximum(d, 0.1))
    e[:, 1:] |= np.abs(ld[:, 1:] - ld[:, :-1]) > 0.08
    e[1:, :] |= np.abs(ld[1:, :] - ld[:-1, :]) > 0.08
    e = cv2.morphologyEx(e.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    return np.repeat((e * 255)[..., None], 3, -1).astype(np.uint8)


def _load_frame(adir: Path, k: int):
    import cv2
    rgb = cv2.imread(str(adir / "op" / "rgb" / f"{k:05d}.png"))[..., ::-1]
    dp = cv2.imread(str(adir / "op" / "depth" / f"{k:05d}.png")).astype(np.float64)
    depth = (dp[..., 2] + dp[..., 1] * 256 + dp[..., 0] * 65536) / (256 ** 3 - 1) * 1000.0
    ins = cv2.imread(str(adir / "op" / "inst" / f"{k:05d}.png"))
    tag = ins[..., 2]
    inst = ins[..., 1].astype(np.int64) + 256 * ins[..., 0].astype(np.int64)
    return rgb, depth, tag, inst


def project(M_cam: np.ndarray, pts: np.ndarray):
    """World points (n, 3) -> pixel u, v and depth x along the optical axis (CARLA camera: x fwd, y right, z up)."""
    p = (np.linalg.inv(M_cam) @ np.c_[pts, np.ones(len(pts))].T)[:3]
    x = p[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return W / 2.0 + F * p[1] / x, H / 2.0 - F * p[2] / x, x


def box_corners(xyz, yaw, bb) -> np.ndarray:
    """8 world corners of an actor's box: bb = [loc x, y, z, extent x, y, z] in the actor frame."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    loc, ext = np.asarray(bb[:3]), np.asarray(bb[3:])
    sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], np.float64)
    local = loc + sg * ext
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.asarray(xyz, np.float64) + local @ R.T


def _hazard_mask(M, tag, inst, boxes):
    """GT mask of the hazard walkers: inside each projected box (+4 px), pedestrian pixels of the box's dominant
    instance (p5_pair_agent._visibility's rule); also returns the union of the projected boxes."""
    mask, boxm = np.zeros(tag.shape, bool), np.zeros(tag.shape, bool)
    for corners in boxes:
        u, v, x = project(M, corners)
        if (x <= 0.3).all():
            continue
        ok = x > 0.3
        u0, u1 = int(np.floor(u[ok].min())) - 4, int(np.ceil(u[ok].max())) + 4
        v0, v1 = int(np.floor(v[ok].min())) - 4, int(np.ceil(v[ok].max())) + 4
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if u1 <= u0 or v1 <= v0:
            continue
        boxm[v0:v1, u0:u1] = True
        sub_t, sub_i = tag[v0:v1, u0:u1], inst[v0:v1, u0:u1]
        ped = sub_t == PED_TAG
        if ped.any():
            ids, n = np.unique(sub_i[ped], return_counts=True)
            mask[v0:v1, u0:u1] |= ped & (sub_i == ids[n.argmax()])
    return mask, boxm


def _corridor(M, route_xyz, ego_xy) -> np.ndarray:
    """Image polygon (n, 2) of the ground band 0-40 m ahead along the dense route, +- 3 m (clipped at 1 m ahead)."""
    i0 = int(np.argmin(np.hypot(*(route_xyz[:, :2] - ego_xy).T)))
    seg = route_xyz[i0:]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(seg[:, :2], axis=0).T))]
    n = max(2, int(np.searchsorted(s, CORRIDOR_M)) + 1)
    c = seg[:n]
    t = np.gradient(c[:, :2], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    z = np.full(len(c), float(np.median(route_xyz[i0:i0 + 5, 2])) - 0.0)
    left = np.c_[c[:, :2] + CORRIDOR_HALF * nrm, z]
    right = np.c_[c[:, :2] - CORRIDOR_HALF * nrm, z]
    poly = np.r_[left, right[::-1]]
    u, v, x = project(M, poly)
    keep = x > 1.0
    return np.c_[u[keep], v[keep]].astype(np.float32)


def _attempt(gen: Path, rid: str) -> Path:
    from . import p5_pairs as PP
    a = PP.attempt(gen, rid)
    assert a is not None, f"no finished attempt for {rid} in {gen}"
    return a


def controls(which: str = "all", workers: int = 12):
    """Determinism check and every Cosmos input + ground truth for the re-rendered pairs."""
    p = pairs()
    if which == "pilot1":
        p = p[p.base_id == PILOT1]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(workers, 2 * len(p))) as ex:
        rows = list(ex.map(_controls_pair, [r for _, r in p.iterrows()]))
    d = pd.DataFrame(rows)
    out = RESULTS / "determinism.csv"
    if out.exists() and which != "all":
        old = pd.read_csv(out, dtype={"pair": str})
        d = pd.concat([old[~old.pair.isin(d.pair)], d])
    d.sort_values("pair").to_csv(out, index=False)
    log.info("\n%s", d.to_markdown(index=False))
    return d


def _controls_pair(r) -> dict:
    import cv2
    from . import p5_pairs as PP
    gen, v1 = root("gen"), data_dir() / GEN_V1
    ks = list(range(int(r.k0), int(r.k1)))
    row = {"pair": r.pair}
    worlds = {}
    for m in ("plus", "minus"):
        rid = r[m]
        a, o = _attempt(gen, rid), _attempt(v1, rid)
        A, O = PP.load_world(a), PP.load_world(o)
        j = A["pose"][["x", "y", "yaw"]].join(O["pose"][["x", "y", "yaw"]], rsuffix="_o", how="inner")
        j = j[j.index < int(r.k1)]
        row[f"{m}_pose_max_m"] = float(np.hypot(j.x - j.x_o, j.y - j.y_o).max())
        row[f"{m}_ticks_checked"] = len(j)
        fr = pd.read_json(a / "op" / "frames.jsonl", lines=True).set_index("k")
        assert all(k in fr.index for k in ks), f"{rid}: missing op frames {sorted(set(ks) - set(fr.index))[:5]}"
        worlds[m] = (a, A, fr)
    (ap, Ap, fp), (am, Am, fm) = worlds["plus"], worlds["minus"]
    ego = Ap["pose"].loc[ks, ["x", "y"]].to_numpy() - Am["pose"].loc[ks, ["x", "y"]].to_numpy()
    row["pair_ego_max_m"] = float(np.hypot(*ego.T).max())
    kinds = Ap["kinds"]
    haz = [i for i, t in zip(Ap["hazards"], Ap["hazard_types"]) if t.startswith("walker.")]
    act = Ap["act"]
    route = pd.read_json(ap / "route.json")[["x", "y", "z"]].to_numpy()
    cd = root("clips", r.pair)
    vids = {m: {v: [] for v in ("rgb", "seg", "depth", "edge")} for m in worlds}
    gt = {"mask": [], "box": [], "region": [], "walk_minus": [], "corridor": [], "px": []}
    cache, diff_inst, seg_diff, rgb_diff = {}, [], [], []
    for k in ks:
        fr_ = {m: (np.array(fr.loc[k, "cam"]), *_load_frame(a, k)) for m, (a, _, fr) in worlds.items()}
        M, _, _, tag, inst = fr_["plus"]
        sel = act["k"] == k
        boxes = [box_corners(act["xyz"][sel][act["id"][sel] == h][0], act["yaw"][sel][act["id"][sel] == h][0],
                             kinds[str(h)][2]) for h in haz if (act["id"][sel] == h).any()]
        mask, boxm = _hazard_mask(M, tag, inst, boxes)
        _, _, _, tag_m, inst_m = fr_["minus"]
        region = cv2.dilate((mask | boxm).astype(np.uint8), np.ones((2 * REGION_DILATE + 1,) * 2, np.uint8)) > 0
        # Instance ids are per run, not per actor (measured: 100% of x+ / x- thing pixels differ). Give each x-
        # instance the x+ id it overlaps most outside the hazard region, so the same object gets the same seg colour.
        th = np.isin(tag, list(THINGS)) & (tag == tag_m) & ~region
        diff_inst.append(float((inst != inst_m)[th].mean()) if th.any() else 0.0)
        remap = inst_m.copy()
        if th.any():
            pm = pd.DataFrame({"m": inst_m[th], "p": inst[th]}).value_counts().reset_index()
            best = pm.sort_values("count", ascending=False).drop_duplicates("m")
            lut = dict(zip(best.m, best.p))
            thm = np.isin(tag_m, list(THINGS))
            remap[thm] = pd.Series(inst_m[thm]).map(lambda i: lut.get(i, i + 1_000_000)).to_numpy()
        for m, (tg, ins) in (("plus", (tag, inst)), ("minus", (tag_m, remap))):
            _, rgb, depth = fr_[m][:3]
            seg = seg_image(tg, ins, cache)
            vids[m]["rgb"].append(rgb)
            vids[m]["seg"].append(seg)
            vids[m]["depth"].append(depth_image(depth))
            vids[m]["edge"].append(edge_image(seg, depth))
        seg_diff.append(float((vids["plus"]["seg"][-1] != vids["minus"]["seg"][-1]).any(-1)[~region].mean()))
        rgb_diff.append(float(np.abs(vids["plus"]["rgb"][-1].astype(np.int16) - vids["minus"]["rgb"][-1])[~region].mean()))
        gt["mask"].append(np.packbits(mask))
        gt["region"].append(np.packbits(region))
        gt["walk_minus"].append(np.packbits(tag_m == PED_TAG))
        gt["px"].append(int(mask.sum()))
        ys, xs = np.nonzero(mask)
        gt["box"].append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [-1, -1, -1, -1])
        gt["corridor"].append(_corridor(M, route, Ap["pose"].loc[k, ["x", "y"]].to_numpy(np.float64)))
    for m in worlds:
        md = cd / m
        md.mkdir(parents=True, exist_ok=True)
        for v, fs in vids[m].items():
            write_mp4(md / f"{v}.mp4", np.stack(fs))
    L = max(len(c) for c in gt["corridor"])
    corr = np.full((len(ks), L, 2), np.nan, np.float32)
    for i, c in enumerate(gt["corridor"]):
        corr[i, :len(c)] = c
    np.savez_compressed(cd / "gt.npz", mask=np.stack(gt["mask"]), region=np.stack(gt["region"]),
                        walk_minus=np.stack(gt["walk_minus"]), px=np.array(gt["px"]), box=np.array(gt["box"]),
                        corridor=corr, k=np.array(ks), shape=np.array([H, W]))
    px = np.array(gt["px"])
    row.update(frames=len(ks), vis_frames=int((px >= VIS_PX).sum()), px_max=int(px.max()),
               first_vis=int(np.argmax(px >= VIS_PX)) if (px >= VIS_PX).any() else -1,
               inst_id_differs=float(np.mean(diff_inst)), seg_diff_outside=float(np.mean(seg_diff)),
               rgb_absdiff_outside=float(np.mean(rgb_diff)), hazards=len(haz),
               weather=json.dumps(json.loads((ap / "meta.json").read_text())["weather"]))
    return row


# ------------------------------------------------------------------ Cosmos specs

PLACE = {"Town01": "a small town street with two lanes", "Town02": "a small town street with two lanes",
         "Town03": "a city street", "Town04": "a town road", "Town05": "a city street",
         "Town07": "a rural village road", "Town10HD": "a downtown city street", "Town11": "a suburban road"}
VARIANTS = {   # name: (model, num_steps, control key, control file or None = computed by Cosmos from rgb.mp4)
    "edgeA": ("edge/distilled", 4, "edge", None),
    "edgeB": ("edge/distilled", 4, "edge", "edge.mp4"),
    "seg": ("seg", 35, "seg", "seg.mp4"),
}
SEED, SEED_ALT = 2025, 2026


def prompt(town: str, weather: dict) -> str:
    w = weather
    when = ("at night, lit by street lamps" if w["sun_altitude_angle"] < 0 else
            "at dusk with a low sun" if w["sun_altitude_angle"] < 15 else "in daylight")
    sky = ("under a heavy overcast sky" if w["cloudiness"] > 70 else "under a clear sky" if w["cloudiness"] < 20
           else "under a partly cloudy sky")
    wx = []
    if w["precipitation"] > 30:
        wx.append("in steady rain")
    if w["wetness"] > 50 or w["precipitation_deposits"] > 50:
        wx.append("with a wet road surface, puddles and reflections")
    if w["fog_density"] > 30:
        wx.append("in light fog")
    return (f"A realistic dashcam video recorded from behind the windshield of a car driving on {PLACE[town]} "
            f"{when} {sky}{', ' + ', '.join(wx) if wx else ''}. The road, buildings, trees, parked cars and road "
            "markings look like real-world footage from a car camera, with natural lighting, realistic textures and "
            "materials, and slight sensor noise.")


def specs(variant: str, which: str = "all", floors: str = "both") -> Path:
    """One jsonl per variant and selection. Every member: seed SEED; with floors, x- again with SEED (determinism
    floor, name suffix _rep) and with SEED_ALT (seed spread)."""
    model, steps, key, ctrl = VARIANTS[variant]
    p = pairs()
    if which == "pilot1":
        p = p[p.base_id == PILOT1]
    elif which != "all":                      # explicit pair list, e.g. 24252-s0,27582-s0
        p = p[p.pair.isin(which.split(","))]
    det = pd.read_csv(RESULTS / "determinism.csv", dtype={"pair": str}).set_index("pair")
    lines = []
    for _, r in p.iterrows():
        pr = prompt(r.town, json.loads(det.loc[r.pair, "weather"]))
        jobs = [("plus", SEED, ""), ("minus", SEED, "")] + ([("minus", SEED, "_rep")] if floors == "both" else []) \
            + ([("minus", SEED_ALT, "")] if floors in ("both", "alt") else [])
        for m, seed, suf in jobs:
            cd = root("clips", r.pair, m)
            c = {"control_weight": 1.0} | ({"control_path": str(cd / ctrl)} if ctrl else {})
            lines.append({"name": f"{r.pair}_{m}_{variant}_s{seed}{suf}", "prompt": pr, "video_path": str(cd / "rgb.mp4"),
                          "seed": seed, "num_steps": steps, "guidance": 3, key: c})
    f = root("specs") / f"{variant}_{which.replace(',', '+')}_{floors}.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    log.info("%d samples -> %s (model %s)", len(lines), f, model)
    return f


def hazard_boxes(which: str = "all"):
    """Per hazard walker (PedestrianCrossing has three): GT mask pixels and tight box per frame -> gt_boxes.npz.
    Check 1 matches a detection against ONE pedestrian's box; gt.npz only has the union of all hazards."""
    from . import p5_pairs as PP
    p = pairs()
    if which == "pilot1":
        p = p[p.base_id == PILOT1]
    for _, r in p.iterrows():
        a = _attempt(root("gen"), r.plus)
        A = PP.load_world(a)
        fr = pd.read_json(a / "op" / "frames.jsonl", lines=True).set_index("k")
        act, kinds = A["act"], A["kinds"]
        haz = [i for i, t in zip(A["hazards"], A["hazard_types"]) if t.startswith("walker.")]
        ks = list(range(int(r.k0), int(r.k1)))
        box, px = np.full((len(ks), len(haz), 4), -1, np.int32), np.zeros((len(ks), len(haz)), np.int32)
        for i, k in enumerate(ks):
            _, _, tag, inst = _load_frame(a, k)
            M, sel = np.array(fr.loc[k, "cam"]), act["k"] == k
            for j, h in enumerate(haz):
                m = act["id"][sel] == h
                if not m.any():
                    continue
                mask, _ = _hazard_mask(M, tag, inst, [box_corners(act["xyz"][sel][m][0], act["yaw"][sel][m][0], kinds[str(h)][2])])
                ys, xs = np.nonzero(mask)
                px[i, j] = len(xs)
                if len(xs):
                    box[i, j] = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        np.savez(root("clips", r.pair) / "gt_boxes.npz", box=box, px=px, hazards=np.array(haz))
        log.info("%s: %d hazards, visible units %d", r.pair, len(haz), int((px >= VIS_PX).sum()))


def world_ids(which: str = "all") -> str:
    p = pairs()
    if which == "pilot1":
        p = p[p.base_id == PILOT1]
    return ",".join(r[w] for _, r in p.iterrows() for w in ("plus", "minus"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("select", "ids", "controls", "specs", "boxes"))
    ap.add_argument("--which", default="all")
    ap.add_argument("--variant", default="edgeA")
    ap.add_argument("--floors", default="both", choices=("both", "alt", "none"))
    a = ap.parse_args()
    if a.step == "specs":
        print(specs(a.variant, a.which, a.floors))
    elif a.step == "select":
        select()
    elif a.step == "boxes":
        hazard_boxes(a.which)
    elif a.step == "controls":
        controls(a.which)
    elif a.step == "ids":
        print(world_ids(a.which))


if __name__ == "__main__":
    main()
