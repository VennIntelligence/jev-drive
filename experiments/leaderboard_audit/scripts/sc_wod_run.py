"""Scale check: Cinque (trt, 20 Hz, zero state, no desire) on WOD frames, saving every head (the layout of the op_lb plan files:
heads minus hidden_state / pad, in slice order) per target, for the base rig and the virtual-height rigs of
experiments/model_smoke/lib/wod_openpilot_rigs.py (same warp, rays through the ground plane).

  e2e  WOD-E2E rater + extra frames (479 + 958), the exam protocol: 10 s of history, each 10 Hz frame fed twice, output at the target
  sf   the 10 sceneflow perception segments (sc_wod_prep.py): one continuous stream per segment, output kept at every frame >= 3 s

envs/openpilot on a leased card: CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/leaderboard_audit/scripts/sc_wod_run.py e2e|sf
Outputs: $DATA_DIR/runs/scale_check/{e2e,sf}_<variant>.npz (resumable per segment for sf)."""
import argparse, io, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments/model_smoke/lib")]
from jevdrive import camgeom as G  # noqa: E402
import wod_openpilot_rigs as R  # noqa: E402

D = Path(os.environ["DATA_DIR"])
OUT = D / "runs/scale_check"
VARS = ["base", "h1.22"]
G0 = -0.05          # ground z in the WOD vehicle frame (sc_wod_ground.py); the rig's h_cam below is the extrinsic z (ground at 0)


def keep_layout(m):
    heads = sorted(((q, s) for q, s in m.slices.items() if q not in ("hidden_state", "pad")), key=lambda x: x[1].start)
    keep = np.concatenate([np.arange(s.start, s.stop) for _, s in heads])
    hs, o = {}, 0
    for q, s in heads:
        hs[q] = o
        o += s.stop - s.start
    return keep, hs


# ---------------------------------------------------------------- sceneflow rendering (worker)
def sf_render(seg):
    from PIL import Image
    d = pickle.load(open(OUT / "wod_sf" / f"{seg}.pkl", "rb"))
    cal = d["cal"]
    h_cam = float(cal[1]["extrinsic"][2, 3])
    sizes = [(cal[c]["width"], cal[c]["height"]) for c in (1, 2, 3)]
    idx = {}
    for v in VARS:
        hv = R.VARIANTS[v][0]
        for k in ("road", "wide"):
            rays = R.height_rays(G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), h_cam, hv)
            src, U, V = G.choose_sources(np, rays, cal)
            src = np.where(src >= 0, src, -1)
            idx[(v, k)] = G.nn_gather_index(src, U, V, sizes).ravel()
    out = {v: np.empty((len(d["frames"]), 2, 6, 128, 256), np.uint8) for v in VARS}
    black = np.array([[0, 128, 128]], np.uint8)
    for j, f in enumerate(d["frames"]):
        planes = []
        for c in (1, 2, 3):
            im = Image.open(io.BytesIO(f["jpg"][c]))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes + [black])
        for v in VARS:
            for m, k in enumerate(("road", "wide")):
                out[v][j, m] = R._pack(cat[idx[(v, k)]].reshape(G.OP_H, G.OP_W, 3))
    return seg, out


def stream(m, frames, keep, first):
    """Continuous 20 Hz rollout from a zero state; heads after the 2nd feed of every frame >= first."""
    m.reset()
    res = []
    for j, s in enumerate(frames):
        for _ in range(2):
            raw = m.step(s, action_t=(0.275, 0.525))
        if j >= first:
            res.append(raw[keep])
    return np.stack(res)


def cmd_sf(a):
    from jevdrive.openpilot.model import OPModel
    segs = sorted(p.stem for p in (OUT / "wod_sf").glob("*.pkl"))
    m = OPModel("cinque", "trt")
    keep, hs = keep_layout(m)
    info = json.dumps(dict(heads_slices=hs, first=a.first))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for seg, fr in ex.map(sf_render, segs):
            for v in VARS:
                f = OUT / f"sf_{v}_{seg}.npz"
                if f.exists():
                    continue
                H = stream(m, fr[v], keep, a.first)
                np.savez(f, heads=H, info=info, first=a.first)
            print(seg, f"{time.time() - t0:.0f}s", flush=True)


def cmd_e2e(a):
    from jevdrive import wod_zeroshot as Z
    from jevdrive.common import data_dir
    from jevdrive.openpilot.model import OPModel
    from wod_zeroshot_openpilot import MODELS
    sets = Z.load_sets()
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    todo = sorted(str(n) for w in ("rater", "extra") for n in sets[w]["name"])[: a.limit or None]
    m = OPModel("cinque", MODELS["cinque"])
    keep, hs = keep_layout(m)
    H = {v: [] for v in VARS}
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=R._init, initargs=(spans, op_calib, str(shard_dir), VARS)) as ex:
        for i, (name, names, fr) in enumerate(ex.map(R.model_frames, todo, chunksize=1)):
            for v in VARS:
                H[v].append(stream(m, fr[v], keep, len(fr[v]) - 1)[0])
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(todo)}] {(time.time() - t0) / (i + 1):.2f} s/target", flush=True)
    info = json.dumps(dict(heads_slices=hs))
    for v in VARS:
        np.savez(OUT / f"e2e_{v}.npz", names=np.array(todo), heads=np.stack(H[v]), info=info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e2e", "sf"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--first", type=int, default=30)
    a = ap.parse_args()
    {"e2e": cmd_e2e, "sf": cmd_sf}[a.cmd](a)
