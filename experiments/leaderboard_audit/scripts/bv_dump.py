#!/usr/bin/env python
"""Board views, step 1 (box, CPU): per board two scenes (straight, turn) of the source camera frame and what openpilot receives
(road / wide 512x256 model frames, RGB), plus geometry stats. Output $DATA_DIR/runs/leaderboard_audit/board_views/{<board>_<scene>_{src,road,wide}.png, stats.json}.
bv_sheet.py (Mac) draws the sheet. Boards: wod, navtest, navhard, hugsim, b2d.

  $DATA_DIR/envs/hugsim/bin/python experiments/leaderboard_audit/scripts/bv_dump.py <board> ...
"""
import io, json, os, pickle, sys, csv, glob
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from jevdrive import camgeom as G  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
OUT = D / "runs" / "leaderboard_audit" / "board_views"
OUT.mkdir(parents=True, exist_ok=True)
STATS = OUT / "stats.json"


def rgb(packed):
    from jevdrive import op_interp as I
    Y, U, V = (z.astype(np.float32) for z in I.unpack(packed))
    U, V = (np.repeat(np.repeat(z, 2, 0), 2, 1) - 128 for z in (U, V))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def save(board, scene, src, road, wide):
    for k, a in (("src", src), ("road", road), ("wide", wide)):
        if a is not None:
            Image.fromarray(np.ascontiguousarray(a)).save(OUT / f"{board}_{scene}_{k}.png")


def add_stats(board, d):
    s = json.loads(STATS.read_text()) if STATS.exists() else {}
    s.setdefault(board, {}).update(d)
    STATS.write_text(json.dumps(s, indent=1, default=float))


def hfov(f, w):
    return float(np.degrees(2 * np.arctan(w / 2 / f)))


def wod():
    import wod_zeroshot_openpilot as W
    from jevdrive import wod_zeroshot as Z
    sets = Z.load_sets()["rater"]
    spans, _ = Z.load_spans()
    cal = json.loads((Z.root() / "op_calib.json").read_text())
    shard = D / "datasets" / "waymo_e2e" / "front3"
    W._init(spans, cal, str(shard))
    names = [str(n) for n in sets["name"]]
    speed = np.linalg.norm(sets["past"][:, -1, 2:4], axis=-1)
    it = np.asarray(sets["intent"])
    print("intents", np.unique(it, return_counts=True))
    fut = sets["future"][:, -1, :2]            # 20 pts; last point lateral offset
    rng = np.random.default_rng(0)
    # straight: low lateral future offset and speed; turn: large lateral offset
    lat = np.abs(fut[:, 1])
    ok = speed > 6
    straight = np.flatnonzero(ok & (lat < 1.5))
    turn = np.flatnonzero((speed > 3) & (lat > 8))
    print(len(straight), len(turn))
    picks = {"straight": int(rng.choice(straight)), "turn": int(rng.choice(turn))}
    for sc, i in picks.items():
        name, hist, frames = W.model_frames(names[i])
        sp = spans[name]
        f = open(shard / sp[0], "rb")
        f.seek(sp[1]); src = np.asarray(Image.open(io.BytesIO(f.read(sp[2]))).convert("RGB"))
        save("wod", sc, src, rgb(frames[-1, 0]), rgb(frames[-1, 1]))
        print("wod", sc, name, "speed", speed[i], "intent", it[i], "lat", lat[i])
    # stats
    seq = names[picks["straight"]].rsplit("-", 1)[0]
    c = {int(k): v for k, v in cal[seq].items()}
    st = {}
    for k in ("road", "wide"):
        src_, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), {x: c[x] for x in Z.OP_SRC})
        st[f"src_frac_{k}"] = {str(x): float((src_ == j).mean()) for j, x in enumerate(Z.OP_SRC)}
    E = np.array(c[1]["extrinsic"]).reshape(4, 4)
    st.update(cam_z=float(E[2, 3]), cam_x=float(E[0, 3]), pitch_deg=float(np.degrees(np.arcsin(E[2, 0]))),
              f=c[1]["intrinsic"][0], w=c[1]["width"], h=c[1]["height"], hfov=hfov(c[1]["intrinsic"][0], c[1]["width"]),
              side_yaw_deg=[float(np.degrees(np.arctan2(np.array(c[x]["extrinsic"]).reshape(4, 4)[1, 0], np.array(c[x]["extrinsic"]).reshape(4, 4)[0, 0]))) for x in (2, 3)])
    # scale proxy: plan speed at t0 / logged speed (Cinque)
    pd = D / "processed" / "wod_zeroshot" / "preds" / "op_cinque"
    r = []
    for n, v in zip(names, speed):
        p = pd / f"{n}.npz"
        if v > 3 and p.exists():
            r.append(float(np.load(p)["plan_vel"][0][0]) / v)
    st["plan_speed_ratio_median"] = float(np.median(r)) if r else None
    st["plan_speed_ratio_n"] = len(r)
    add_stats("wod", st)


def nav(board):
    from jevdrive import navsim_zs as Z
    split = "navtest" if board == "navtest" else "navhard_two_stage"
    idx = Z.load_index(split, slim=True)
    frames = np.load(Z.root("openpilot", split) / "frames.npy", mmap_mode="r")
    cand = [k for k, e in enumerate(idx) if board == "navtest" or e["stage"] == 2]
    sp = np.array([np.linalg.norm(idx[k]["vel"][-1]) for k in cand])
    cm = np.array([int(np.argmax(idx[k]["cmd"][-1])) for k in cand])
    rng = np.random.default_rng(1)
    print(board, len(cand), np.bincount(cm))
    pools = {"straight": [k for k, s, c in zip(cand, sp, cm) if c == 1 and s > 6], "turn": [k for k, s, c in zip(cand, sp, cm) if c in (0, 2) and s > 3]}
    for sc, pool in pools.items():
        k = int(rng.choice(pool))
        e = idx[k]
        src = np.asarray(Image.open(e["cams"][-1]["CAM_F0"]["path"]).convert("RGB"))
        save(board, sc, src, rgb(np.asarray(frames[k, 3, 0])), rgb(np.asarray(frames[k, 3, 1])))
        print(board, sc, e["token"], "cmd", cm[cand.index(k)], "v", sp[cand.index(k)], e["stage"])
    cam = idx[cand[0]]["cams"][-1]["CAM_F0"]
    Kc = np.asarray(cam["K"], float)
    R = Z.cam_to_ego(cam)
    t = np.asarray(cam["t"], float)
    m = Z.OpenpilotMaps(cam)
    add_stats(board, dict(f=float(Kc[0, 0]), w=Z.NUPLAN_WH[0], h=Z.NUPLAN_WH[1], hfov=hfov(Kc[0, 0], Z.NUPLAN_WH[0]), cam_z_above_rear_axle=float(t[2]),
                          cam_x=float(t[0]), pitch_deg=float(np.degrees(np.arcsin(R[2, 2]))), coverage_road_wide=m.coverage, cam_dist=cam["D"]))


def hugsim():
    import cv2
    from jevdrive import hugsim_zs as Z
    rows = [r for r in csv.DictReader(open(REPO / "experiments/leaderboard_audit/results/loss_budget/hugsim_inputs/exam_scored_op.csv"))
            if r["tag"] == "cinque-fixed" and r["dataset"] == "nuscenes" and r["end"] in ("complete", "max_steps")]
    cams_yaml = D / "third_party/HUGSIM/configs/sim/{}_camera.yaml"
    picks, ratios = {}, []
    for r in rows:
        d = Path(r["run_dir"].replace("/root/autodl-tmp/ujs", str(D)))
        try:
            infos = pickle.load(open(d / "infos.pkl", "rb"))
        except Exception:
            continue
        steps = [json.loads(x) for x in open(d / "zs_steps.jsonl")][1:]
        steps = {s["step"]: s for s in steps if "step" in s}
        yaw = np.unwrap([float(np.asarray(i["ego_box"], float)[6]) for i in infos])
        v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
        for k, s in steps.items():
            if v[k] > 3 and s.get("model_pos"):
                ratios.append(float(np.asarray(s["model_pos"])[0][0]) / 0.156 / v[k])
        for k in range(10, len(infos) - 25):
            dy = abs(np.degrees(yaw[k + 20] - yaw[k]))
            if v[k] > 3:
                if dy < 2 and "straight" not in picks:
                    picks["straight"] = (d, k, r)
                if dy > 20 and "turn" not in picks:
                    picks["turn"] = (d, k, r)
        if len(picks) == 2 and len(ratios) > 400:
            break
    for sc, (d, k, r) in picks.items():
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        for _ in range(k + 1):
            ok, f = cap.read()
        h, w = f.shape[0] // 2, f.shape[1] // 3
        f = f[..., ::-1]
        vid = {"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]}
        infos = pickle.load(open(d / "infos.pkl", "rb"))
        cal = Z.calibs(infos[0]["cam_params"], Z.rect_matrix(str(cams_yaml).format("nuscenes")))
        op = Z.OpenpilotFrames(cal)
        cat = np.concatenate([vid[c].reshape(-1, 3) for c in op.cams] + [np.zeros((1, 3), np.uint8)])
        ims = [cat[op.idx[m]].reshape(G.OP_H, G.OP_W, 3) for m in ("road", "wide")]
        save("hugsim", sc, vid["CAM_FRONT"], *ims)
        print("hugsim", sc, d, k, r["scenario"])
    E = np.asarray(cal["CAM_FRONT"]["extrinsic"])
    c = cal["CAM_FRONT"]
    st = dict(cam_z=float(E[2, 3]), cam_x=float(E[0, 3]), pitch_deg=float(np.degrees(np.arcsin(E[2, 0]))), f=c["intrinsic"][0], fy=c["intrinsic"][1], w=c["width"], h=c["height"],
              hfov=hfov(c["intrinsic"][0], c["width"]), src_frac=op.src_frac, coverage=op.coverage,
              side_yaw_deg=[float(np.degrees(np.arctan2(np.asarray(cal[x]["extrinsic"])[1, 0], np.asarray(cal[x]["extrinsic"])[0, 0]))) for x in ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")],
              plan_speed_ratio_median=float(np.median(ratios)) if ratios else None, plan_speed_ratio_n=len(ratios),
              rect=str(Z.rect_matrix(str(cams_yaml).format("nuscenes"))[:3, 3].tolist()))
    add_stats("hugsim", st)


def b2d():
    root = D / "runs" / "lbx_b2d" / "dump"
    best = {}
    for p in sorted(glob.glob(str(root / "*" / "dump.jsonl"))):
        recs = [json.loads(x) for x in open(p)]
        for r in recs:
            pos = np.asarray(r["pos"], float)
            if r["v"] < 5:
                continue
            lat = abs(pos[-1, 1])
            fn = Path(p).parent / f"{r['seq']:07d}.jpg"
            if not fn.exists():
                continue
            if lat < 1.0 and "straight" not in best:
                best["straight"] = (fn, r, p)
            if lat > 8 and "turn" not in best:
                best["turn"] = (fn, r, p)
    for sc, (fn, r, p) in best.items():
        im = np.asarray(Image.open(fn).convert("RGB"))
        save("b2d", sc, None, im[:, :512], im[:, 512:])
        print("b2d", sc, fn, "v", r["v"])
    import zeroshot_rigs as R
    add_stats("b2d", dict(cam_z=R.OP_MOUNT_RIG[2], cam_x=R.OP_MOUNT_RIG[0], pitch_deg=0.0, f_road=R.OP_FOCAL["road"], f_wide=R.OP_FOCAL["wide"], w=R.OP_CAMERA_WH[0], h=R.OP_CAMERA_WH[1],
                          hfov_road=hfov(R.OP_FOCAL["road"], R.OP_CAMERA_WH[0]), hfov_wide=hfov(R.OP_FOCAL["wide"], R.OP_CAMERA_WH[0]), src="dump jpgs (lbx_dump_server)"))


if __name__ == "__main__":
    for b in sys.argv[1:]:
        {"wod": wod, "navtest": lambda: nav("navtest"), "navhard": lambda: nav("navhard"), "hugsim": hugsim, "b2d": b2d}[b]()
