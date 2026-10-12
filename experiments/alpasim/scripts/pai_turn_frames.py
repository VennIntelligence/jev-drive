#!/usr/bin/env python3
"""What openpilot saw on low-scoring PAI turn scenes: one JPEG per rollout, rows = decision times, columns = the openpilot road and
wide model frames with the served 4 s plan (rear axle, on the ground) drawn on them. Runs on CPU in the tfv6 env on the Tokyo box.

Selection: scenes where the base arm scores < --smax and the route turns by more than --deg degrees. The turn is the net heading
change (last 15 m chord against the first) of the route the simulator served (drive.jsonl `route0`, the route point ~40 m ahead, put into the world frame with `anchor`),
taken from the arm that drove furthest on that scene; positive = left. Per scene the base arm and the first --cmp arm that has the
scene are rendered.

Frames: the run keeps no model-frame dumps (PAI_DUMP = 0), so they are re-derived with pai_core.Maps (deterministic f-theta ->
openpilot road / wide sampling) from (a) the exact front-wide JPEGs the driver received when the rollout kept its rollout.asl ("asl"),
else (b) the camera panel of the eval mp4 (900 x 1000 composite, the panel is ~840 px wide and carries the eval's own trajectory
overlay, so these frames are blurrier and show an orange / green line that the model did not see) ("mp4"). The path is in the title.

  pai_turn_frames.py [--root /data/runs/onpol_long/pai281] [--out ~/Desktop/pai_turns] [--base ybase] [--cmp ys2g0 ys1]
                     [--smax 0.3] [--deg 30] [--n 40] [--procs 8] [--scenes <scene8> ...]
"""
import argparse, functools, glob, io, json, struct, subprocess, sys, types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parents[1] / "lib"
sys.path[:0] = [str(LIB), str(LIB.parents[2])]
FAIL = {"collision_at_fault": "crash", "offroad": "offroad", "left_corridor_laterally": "corridor"}
EVENT = {"crash": ["collision_front", "collision_lateral"], "offroad": ["offroad"], "corridor": ["left_corridor_laterally"]}
GRPC = "/data/third_party/alpasim/src/grpc/alpasim_grpc"


def kind(r):
    return next((v for k, v in FAIL.items() if (r.get("score_metrics") or {}).get(k)), "ok" if r["score"] > 0 else "zero")


def index(root: Path) -> dict:
    """{(arm, scene): (run dir, rollout record)}; the first run that has the scene wins."""
    out = {}
    for f in sorted(glob.glob(str(root / "*/runs/*/sim/aggregate/results-summary.json"))):
        f = Path(f)
        for r in json.loads(f.read_text())["rollouts"]:
            if r.get("score") is not None:
                out.setdefault((f.parents[4].name, r["clipgt_id"]), (f.parents[2], r))
    return out


@functools.lru_cache(4)
def _log(run: Path):
    return [json.loads(x) for x in (run / "driver/drive.jsonl").read_text().splitlines()]


def drive(run: Path, scene: str):
    rows = _log(run)
    start = next(json.loads(x) for x in (run / "driver/start.jsonl").read_text().splitlines() if f'"{scene}"' in x)
    return [r for r in rows if r["kind"] == "drive" and r["scene"] == scene and r.get("infer")], start


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def turn_deg(dr) -> float:
    """Net heading change (deg, left +, wrapped to +-180) of the served route point in the world frame: last 15 m chord vs first."""
    p = np.array([np.array(r["anchor"][:2]) + rot(r["anchor"][2]) @ np.array(r["route0"]) for r in dr if r.get("route0")]).reshape(-1, 2)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 2:
        return 0.0
    q = [p[0]]                                            # drop back-steps (the served point jitters when the car stalls or spins)
    for x in p[1:]:
        if np.hypot(*(x - q[-1])) >= 3 and (len(q) < 2 or np.dot(x - q[-1], q[-1] - q[-2]) > 0):
            q.append(x)
    q = np.array(q)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    if s[-1] < 30:
        return 0.0
    c0, c1 = q[np.searchsorted(s, 15)] - q[0], q[-1] - q[np.searchsorted(s, s[-1] - 15) - 1]
    return float(np.degrees(np.angle(complex(*c1) / complex(*c0))))


def asl_frames(path: Path, cam: str, want: list) -> dict:
    """{wanted t0: JPEG bytes} of camera cam from a rollout.asl (length-prefixed LogEntry protobufs)."""
    m = types.ModuleType("alpasim_grpc")
    m.__path__ = [GRPC]
    sys.modules.setdefault("alpasim_grpc", m)
    from alpasim_grpc.v0.logging_pb2 import LogEntry
    got, want = {}, set(want)
    with open(path, "rb") as f:
        while (h := f.read(4)) and len(got) < len(want):
            e = LogEntry.FromString(f.read(struct.unpack(">L", h)[0]))
            if e.WhichOneof("log_entry") == "driver_camera_image":
                c = e.driver_camera_image.camera_image
                if c.logical_id == cam and c.frame_end_us in want:
                    got[c.frame_end_us] = c.image_bytes
    return got


def mp4_frames(mp4: Path, idx: list) -> dict:
    """{frame index: RGB camera panel} from the eval composite mp4 (panel = non-white bbox of the lower half)."""
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"select='{sel}'", "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 1000, 900, 3)
    low = fr[0, 500:].astype(int).sum(-1) < 700
    ys, xs = np.flatnonzero(low.any(1)) + 500, np.flatnonzero(low.any(0))
    y0, y1, x0, x1 = ys[0] + 2, ys[-1] - 1, xs[0] + 2, xs[-1] - 1                  # inside the panel border
    return {i: f[y0:y1, x0:x1] for i, f in zip(idx, fr)}


def rows_k(n: int, ke) -> list:
    ks = list(np.linspace(0, n - 1, 8 if ke is None else 5).round().astype(int))
    if ke is not None:
        ks += [ke - 20, ke - 10, ke - 5, ke]
    out = []
    for k in sorted(k for k in set(ks) if 0 <= k < n):
        if not out or k - out[-1] >= 3 or k == ke:
            out.append(k)
    return out


def event_t(run: Path, r, kd):
    import pandas as pd
    p = glob.glob(str(run / "sim/rollouts" / r["clipgt_id"] / r["rollout_id"] / "metrics.parquet"))
    if kd not in EVENT or not p:
        return None
    df = pd.read_parquet(p[0])
    df = df[df.name.isin(EVENT[kd]) & (df["values"] > 0)]
    return None if df.empty else int(df.timestamps_us.min())


def render(job):
    from PIL import Image, ImageDraw, ImageFont
    from pai_core import Maps, decode
    from jevdrive.openpilot.frames import MEDMODEL_K, SBIGMODEL_K
    from pai_report import rgb
    arm, run, r, turn, out = job
    scene, kd = r["clipgt_id"], kind(r)
    dr, st = drive(run, scene)
    if not dr:
        return f"skip {scene} {arm}: no drive log"
    te = event_t(run, r, kd)
    t0s = np.array([d["t0"] for d in dr])
    ke = int(np.abs(t0s - te).argmin()) if te is not None else None
    ks = rows_k(len(dr), ke)
    rd = run / "sim/rollouts" / scene / r["rollout_id"]
    asl = rd / "rollout.asl"
    if asl.exists():
        src, jp = "asl", asl_frames(asl, st["cam"], [int(dr[k]["t0"]) for k in ks])
        img = {k: decode(jp[int(dr[k]["t0"])]) for k in ks if int(dr[k]["t0"]) in jp}
    else:
        mp4 = sorted(rd.glob("*camera_front_wide*.mp4"))
        if not mp4:
            return f"skip {scene} {arm}: no mp4"
        src, fr = "mp4", mp4_frames(mp4[0], [d["k"] for d in (dr[k] for k in ks)])
        img = {k: np.asarray(Image.fromarray(fr[dr[k]["k"]]).convert("YCbCr")) for k in ks if dr[k]["k"] in fr}
    maps = {}
    S, W, PAD, TOP = 1.5, 512, 6, 30
    w, h = int(W * S), int(256 * S)
    canvas = Image.new("RGB", (2 * w + 3 * PAD, 60 + len(img) * (h + TOP + PAD)), "white")
    dw = ImageDraw.Draw(canvas)
    big, small = ImageFont.load_default(20), ImageFont.load_default(16)
    dw.text((PAD, 6), f"{scene}  arm {arm}  score {r['score']:.2f}  {kd}  route turn {turn:+.0f} deg (left +)  frames: {src}"
            f"  dist {r['metrics'].get('dist_traveled_m', 0):.0f} / GT {r['metrics'].get('gt_dist_traveled_m', 0):.0f} m", fill="black", font=big)
    dw.text((PAD, 32), "left: road (focal 910), right: wide (focal 455); magenta = served 4 s plan, dots every 0.5 s" + (" (orange / green lines are the eval video overlay, not model input)" if src == "mp4" else ""), fill="black", font=small)
    ct = np.array(st["cam_t"], float)
    for i, k in enumerate(sorted(img)):
        y = 60 + i * (h + TOP + PAD)
        d, im = dr[k], img[k]
        if im.shape[:2] not in maps:
            maps[im.shape[:2]] = Maps(st["spec"], st["R"], (im.shape[1], im.shape[0]))
        f = maps[im.shape[:2]](im)
        p = np.r_[[[0.0, 0.0]], np.array(d["poses"])[:, :2]] - ct[:2]
        v0 = (d.get("fix") or {}).get("v0", float("nan"))
        cmd = {0: "L", 1: "S", 2: "R"}.get(d.get("cmd"), "?")
        tag = f"t = {d['k'] / 10:.1f} s  cmd {cmd}  v {v0:.1f} m/s  plan {np.hypot(*np.array(d['poses'][-1][:2])):.0f} m in 4 s" + \
              ("   <- first zero event" if k == ke else "")
        dw.text((PAD, y + 5), tag, fill="red" if k == ke else "black", font=big)
        for j, K in enumerate((MEDMODEL_K, SBIGMODEL_K)):
            x0 = PAD + j * (w + PAD)
            canvas.paste(Image.fromarray(rgb(f[j])).resize((w, h), Image.BILINEAR), (x0, y + TOP))
            ok = p[:, 0] > 1.0
            u = (K[0, 0] * -p[ok, 1] / p[ok, 0] + K[0, 2]) * S + x0
            v = (K[1, 1] * ct[2] / p[ok, 0] + K[1, 2]) * S + y + TOP
            pts = [(a, b) for a, b in zip(u, v) if x0 <= a < x0 + w and y + TOP <= b < y + TOP + h]
            if len(pts) > 1:
                dw.line(pts, fill=(255, 0, 255), width=3)
            for a, b in pts[1:]:
                dw.ellipse((a - 3, b - 3, a + 3, b + 3), fill=(255, 0, 255))
    s8 = scene.replace("clipgt-", "")[:8]
    name = out / f"{s8}_{arm}_{r['score']:.2f}_{kd}_{'L' if turn > 0 else 'R'}{abs(turn):.0f}.jpg"
    canvas.save(name, quality=88)
    return f"{name.name} {src} rows {len(img)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/runs/onpol_long/pai281"), ap.add_argument("--out", default=str(Path.home() / "Desktop/pai_turns"))
    ap.add_argument("--base", default="ybase"), ap.add_argument("--cmp", nargs="+", default=["ys2g0", "ys1"])
    ap.add_argument("--smax", type=float, default=0.3), ap.add_argument("--deg", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=40), ap.add_argument("--procs", type=int, default=8), ap.add_argument("--scenes", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    I = index(Path(a.root))
    low = [s for (arm, s), (_, r) in I.items() if arm == a.base and r["score"] < a.smax]
    if a.scenes:
        low = [s for s in low if s.replace("clipgt-", "")[:8] in a.scenes]
    turns = {}
    for s in low:                                         # route of the arm that drove furthest on the scene
        run, _ = max((v for (arm, sc), v in I.items() if sc == s), key=lambda v: v[1]["metrics"].get("dist_traveled_m", 0))
        turns[s] = turn_deg(drive(run, s)[0])
    pick = sorted((s for s in low if abs(turns[s]) > a.deg), key=lambda s: -abs(turns[s]))[:a.n]
    print(f"{len(low)} {a.base} scenes below {a.smax}, {sum(abs(t) > a.deg for t in turns.values())} turn > {a.deg} deg, rendering {len(pick)}")
    jobs = []
    for s in pick:
        for arm in [a.base] + [next((c for c in a.cmp if (c, s) in I), None)]:
            if arm:
                jobs.append((arm, *I[(arm, s)], turns[s], out))
    with ProcessPoolExecutor(a.procs) as ex:
        for m in ex.map(render, jobs):
            print(m, flush=True)
    (out / "README.txt").write_text(
        f"What openpilot saw on low-scoring PAI turn scenes ({a.root}).\n"
        f"Selection: {a.base} score < {a.smax} and |route heading change| > {a.deg} deg, the {len(pick)} largest turns; the turn is the\n"
        f"net heading change of the simulator's route (driver log route0 in the world frame) on the arm that drove furthest.\n"
        f"Each scene: the {a.base} rollout and the first of {a.cmp} that has the scene.\n"
        "File name: <scene8>_<arm>_<score>_<zero kind: crash / offroad / corridor / zero / ok>_<L|R><turn deg>.jpg\n"
        "Rows: decision times (8 evenly over the rollout; with a zero event 5 evenly plus 2 s, 1 s, 0.5 s before and at the first\n"
        "event, marked red). Columns: openpilot road model frame (focal 910), wide model frame (focal 455), 512x256 shown at 1.5x.\n"
        "Magenta: the served 4 s plan (rear axle on the ground, dots every 0.5 s). Row text: time, route command, speed, plan length.\n"
        "Frames are re-derived (no dumps were kept) with pai_core.Maps from: 'asl' = the exact JPEGs the driver received\n"
        "(rollout.asl), or 'mp4' = the camera panel of the eval video (lower resolution, carries the eval's orange / green overlay\n"
        "which the model did not see). The source is in each image's title.\n"
        "Script: experiments/alpasim/scripts/pai_turn_frames.py\n")


if __name__ == "__main__":
    main()
