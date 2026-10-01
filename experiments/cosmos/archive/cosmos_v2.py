"""Cosmos pilot v2 (fc65452:todos/2026-09-28-cosmos-pilot.md, "v2"): tune the generator so that a translated pair differs only
by the pedestrian. envs/jevdrive for everything here; the GPU steps are experiments/cosmos/archive/cosmos_infer.py (Cosmos),
experiments/cosmos/lib/cosmos_eval.py (YOLO, pixels), experiments/cosmos/lib/cosmos_openpilot.py.

  controls2  per member: edgeC.mp4 (CARLA geometry edges + Canny of the CARLA frame inside object classes, so objects
             are not empty outlines), segc.mp4 (CARLA's per-class Cityscapes palette, so every pedestrian has the same
             "person" colour); per pair: anchor_mask.mp4 (white = keep x-, black = free region for x+), alpha.npy
  specs2     Cosmos spec for one arm and a pair list
  anchor     Cosmos x- of an arm as a lossless mp4, the anchor video of the guided x+ regeneration
  blend      guided x+ -> x+ = alpha * regenerated + (1 - alpha) * x-, feathered; x- / seed floors linked
Arms:
  P2   Edge Distilled, the v1 geometry edges (edgeB), prompt v2 (roof camera, no windshield)
  E2   Edge Distilled, edgeC, prompt v2; x+ and x- generated independently
  E3 / G3  as E2 / G2 with edgeD (ground texture edges too, against the dashboard) and, for G3, the tight free region
       and blend support (pedestrian pixels, not its box)
  G2   x- = E2's x-; x+ regenerated with the edgeC x+ control while the latents outside the free region are held at
       E2's x- at every step (guided generation, patched into the distilled sampler); then feathered pixel blend (G2b)
  M2   base 35 steps, multicontrol edgeC 1.0 + depth 0.5 + class seg 1.0, prompt v2 + negative prompt v2
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive.common import get_logger
from experiments.cosmos.lib.cosmos_pilot import PLACE, RESULTS, SEED, SEED_ALT, _attempt, _load_frame, pairs, read_mp4, root, write_mp4

log = get_logger(__name__)
OBJ = (5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
GROUND = (1, 2, 10, 24, 25)               # road, sidewalk, terrain, road line, ground
# CARLA 0.9.15 CityScapesPalette, by semantic tag
PALETTE = {0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 4: (102, 102, 156), 5: (190, 153, 153),
           6: (153, 153, 153), 7: (250, 170, 30), 8: (220, 220, 0), 9: (107, 142, 35), 10: (152, 251, 152),
           11: (70, 130, 180), 12: (220, 20, 60), 13: (255, 0, 0), 14: (0, 0, 142), 15: (0, 0, 70), 16: (0, 60, 100),
           17: (0, 80, 100), 18: (0, 0, 230), 19: (119, 11, 32), 20: (110, 190, 160), 21: (170, 120, 50),
           22: (55, 90, 80), 23: (45, 60, 150), 24: (157, 234, 50), 25: (81, 0, 81), 26: (150, 100, 100),
           27: (230, 150, 140), 28: (180, 165, 180)}
LUT = np.zeros((256, 3), np.uint8)
for k, c in PALETTE.items():
    LUT[k] = c
FREE_DILATE_T = 3          # free region: union over +-3 frames (a latent frame covers 4)
FEATHER_ERODE, FEATHER_SIGMA = 16, 5.0
ARMS = {  # name: (model, steps, controls {key: (file, weight)}, anchored)
    "P2": ("edge/distilled", 4, {"edge": ("edge.mp4", 1.0)}, False),
    "E2": ("edge/distilled", 4, {"edge": ("edgeC.mp4", 1.0)}, False),
    "G2": ("edge/distilled", 4, {"edge": ("edgeC.mp4", 1.0)}, True),
    "E3": ("edge/distilled", 4, {"edge": ("edgeD.mp4", 1.0)}, False),
    "G3": ("edge/distilled", 4, {"edge": ("edgeD.mp4", 1.0)}, True),
    "E4": ("edge/distilled", 4, {"edge": ("edgeE.mp4", 1.0)}, False),
    "G4": ("edge/distilled", 4, {"edge": ("edgeE.mp4", 1.0)}, True),
    "M2": ("seg", 35, {"edge": ("edgeC.mp4", 1.0), "depth": ("depth.mp4", 0.5), "seg": ("segc.mp4", 1.0)}, False),
}
NEG2 = ("The video is recorded from inside a car: dashboard, windshield glass, reflections on the glass, raindrops "
        "on the windshield, wipers, the hood of the car, car interior. White mannequins, blank white silhouettes, "
        "ghostly transparent people. The video captures a game playing, with bad crappy graphics and cartoonish frames. "
        "The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive.")


BASE_OF = {"G2": "E2", "G3": "E3", "G4": "E4"}


def prompt2(town: str, weather: dict) -> str:
    w = weather
    when = ("at night, lit by street lamps" if w["sun_altitude_angle"] < 0 else
            "at dusk with a low sun" if w["sun_altitude_angle"] < 15 else "in daylight")
    sky = ("under a heavy overcast sky" if w["cloudiness"] > 70 else "under a clear sky" if w["cloudiness"] < 20
           else "under a partly cloudy sky")
    wx = []
    if w["precipitation"] > 30:
        wx.append("in steady rain")
    if w["wetness"] > 50 or w["precipitation_deposits"] > 50:
        wx.append("with a wet road surface and puddles")
    if w["fog_density"] > 30:
        wx.append("in light fog")
    return (f"A realistic video from a camera mounted on the roof of a car driving on {PLACE[town]} {when} {sky}"
            f"{', ' + ', '.join(wx) if wx else ''}. The camera is outside the car, so the view shows only the street "
            "ahead, and the asphalt road surface reaches the bottom edge of the frame. People, cars, poles, buildings, "
            "trees and road markings look like real-world footage, with natural colours, lighting and textures.")


def _pair_rows(which: str) -> pd.DataFrame:
    p = pairs()
    return p if which == "all" else p[p.pair.isin(which.split(","))]


def _controls2(pair: str):
    import cv2
    global CLAHE
    CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    r = pairs().set_index("pair").loc[pair]
    ks = range(int(r.k0), int(r.k1))
    for m in ("plus", "minus"):
        a = _attempt(root("gen"), r[m])
        rgb = read_mp4(root("clips", pair, m) / "rgb.mp4")
        edge = read_mp4(root("clips", pair, m) / "edge.mp4")[..., 0] > 127
        ec, sc, ed, ee = [], [], [], []
        for t, k in enumerate(ks):
            _, _, tag, _ = _load_frame(a, k)
            obj = cv2.erode(np.isin(tag, OBJ).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            canny = cv2.Canny(cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY), 100, 200) > 0
            e = edge[t] | (canny & obj)
            ec.append(np.repeat((e * 255).astype(np.uint8)[..., None], 3, -1))
            # edgeD: also the ground's own texture (low Canny thresholds), so the bottom of the frame is not empty
            gnd = np.isin(tag, GROUND)
            e2 = e | (cv2.Canny(cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY), 30, 90) > 0) & gnd
            ed.append(np.repeat((e2 * 255).astype(np.uint8)[..., None], 3, -1))
            # edgeE: edgeD, plus dense interior edges of objects on a contrast-equalised frame (CLAHE, Canny 30 / 90):
            # a dark coat in fog has no Canny 100 / 200 edges and came out as a white mannequin (25863, E3 / G3)
            eq = CLAHE.apply(cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY))
            e3 = e2 | ((cv2.Canny(eq, 30, 90) > 0) & obj)
            ee.append(np.repeat((e3 * 255).astype(np.uint8)[..., None], 3, -1))
            sc.append(LUT[tag])
        only = os.environ.get("COSMOS_CTRL_ONLY", "")       # e.g. "edgeE": leave files another job may be reading
        for name, fr in (("edgeC", ec), ("edgeD", ed), ("edgeE", ee), ("segc", sc)):
            if not only or name in only.split(","):
                write_mp4(root("clips", pair, m) / f"{name}.mp4", np.stack(fr))
    if os.environ.get("COSMOS_CTRL_ONLY"):
        return pair, np.nan, np.nan
    from experiments.cosmos.lib.cosmos_eval import gt
    g = gt(pair)
    region = g["region"]
    free = np.stack([region[max(0, t - FREE_DILATE_T):t + FREE_DILATE_T + 1].any(0) for t in range(len(region))])
    write_mp4(root("clips", pair) / "anchor_mask.mp4", np.repeat(((~free) * 255).astype(np.uint8)[..., None], 3, -1))
    er = np.ones((2 * FEATHER_ERODE + 1,) * 2, np.uint8)
    alpha = np.stack([np.maximum(cv2.GaussianBlur(cv2.erode(r_.astype(np.float32), er), (0, 0), FEATHER_SIGMA),
                                 cv2.dilate(m_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32))
                      for r_, m_ in zip(region, g["mask"])])
    np.save(root("clips", pair) / "alpha.npy", alpha.astype(np.float16))
    # tight variant (G3): free latent region = the pedestrian's own pixels dilated 24 px (not its box), blend support =
    # pixels dilated 6 px with a 3 px feather, so the background inside the box (e.g. a bin behind the person) stays x-
    k24, k6 = np.ones((49, 49), np.uint8), np.ones((13, 13), np.uint8)
    mt = np.stack([cv2.dilate(m_.astype(np.uint8), k24) > 0 for m_ in g["mask"]])
    free_t = np.stack([mt[max(0, t - FREE_DILATE_T):t + FREE_DILATE_T + 1].any(0) for t in range(len(mt))])
    write_mp4(root("clips", pair) / "anchor_mask_tight.mp4",
              np.repeat(((~free_t) * 255).astype(np.uint8)[..., None], 3, -1))
    at = np.stack([np.maximum(cv2.GaussianBlur((cv2.dilate(m_.astype(np.uint8), k6) > 0).astype(np.float32), (0, 0), 3.0),
                              m_.astype(np.float32)) for m_ in g["mask"]])
    np.save(root("clips", pair) / "alpha_tight.npy", at.astype(np.float16))
    return pair, float(free.mean()), float((alpha > 0.01).mean())


def controls2(which: str = "all"):
    with ProcessPoolExecutor(10) as ex:
        for pair, fr, al in ex.map(_controls2, list(_pair_rows(which).pair)):
            log.info("%s: free region %.3f of pixels, blend support %.3f", pair, fr, al)


def specs2(arm: str, which: str, floors: bool = True, members: str = "plus,minus") -> Path:
    model, steps, ctrls, anchored = ARMS[arm]
    det = pd.read_csv(RESULTS / "determinism.csv", dtype={"pair": str}).set_index("pair")
    lines = []
    for _, r in _pair_rows(which).iterrows():
        pr = prompt2(r.town, json.loads(det.loc[r.pair, "weather"]))
        jobs = [("plus", SEED)] if anchored else [(m, SEED) for m in members.split(",")] + \
            ([("minus", SEED_ALT)] if floors else [])
        for m, seed in jobs:
            cd = root("clips", r.pair, m)
            x = {"name": f"{r.pair}_{m}_{arm}_s{seed}", "prompt": pr, "seed": seed, "num_steps": steps, "guidance": 3,
                 "video_path": str(cd / "rgb.mp4")}
            if model != "edge/distilled":
                x["negative_prompt"] = NEG2
            for key, (f, wgt) in ctrls.items():
                x[key] = {"control_path": str(cd / f), "control_weight": wgt}
            if anchored:     # anchor = the x- of E2 (lossless mp4), free region = the pedestrian region
                x["video_path"] = str(root("anchor") / f"{r.pair}_{BASE_OF[arm]}.mp4")
                x["guided_generation_mask"] = str(root("clips", r.pair) / ("anchor_mask_tight.mp4" if arm in ("G3", "G4") else
                                                                          "anchor_mask.mp4"))
                x["guided_generation_step_threshold"] = 99
            lines.append(x)
    f = root("specs") / f"{arm}_{which.replace(',', '+')}{'' if floors else '_nofloor'}.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    log.info("%d samples -> %s (model %s)", len(lines), f, model)
    return f


def anchor(arm: str, which: str):
    for pair in _pair_rows(which).pair:
        write_mp4(root("anchor") / f"{pair}_{arm}.mp4", np.load(root("out", arm) / f"{pair}_minus_{arm}_s{SEED}.npy"))


def blend(which: str, src: str = "G2", base: str | None = None):
    """G2b: feathered blend of the regenerated x+ into E2's x-; x- and the seed floor are E2's."""
    base = base or BASE_OF[src]
    d = root("out", src + "b")
    for pair in _pair_rows(which).pair:
        a = np.load(root("clips", pair) / ("alpha_tight.npy" if src in ("G3", "G4") else "alpha.npy")).astype(np.float32)[..., None]
        gen = np.load(root("out", src) / f"{pair}_plus_{src}_s{SEED}.npy").astype(np.float32)
        neg = np.load(root("out", base) / f"{pair}_minus_{base}_s{SEED}.npy")
        np.save(d / f"{pair}_plus_{src}b_s{SEED}.npy", np.rint(a * gen + (1 - a) * neg).astype(np.uint8))
        for s in (SEED, SEED_ALT):
            dst, sf = d / f"{pair}_minus_{src}b_s{s}.npy", root("out", base) / f"{pair}_minus_{base}_s{s}.npy"
            dst.unlink(missing_ok=True)
            if sf.exists():
                dst.symlink_to(sf)
        log.info("%s blended", pair)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("controls2", "specs2", "anchor", "blend"))
    ap.add_argument("--which", default="all")
    ap.add_argument("--arm", default="E2")
    ap.add_argument("--no-floors", action="store_true")
    ap.add_argument("--members", default="plus,minus")
    a = ap.parse_args()
    if a.step == "controls2":
        controls2(a.which)
    elif a.step == "specs2":
        print(specs2(a.arm, a.which, not a.no_floors, a.members))
    elif a.step == "anchor":
        anchor(a.arm, a.which)
    else:
        blend(a.which, a.arm if a.arm in BASE_OF else "G2")


if __name__ == "__main__":
    main()
