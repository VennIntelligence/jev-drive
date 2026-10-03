"""HUGSIM per source dataset (the 64 exam scenes, shipped Cinque run videos): render sharpness and the plan-speed ratio.
Sharpness = variance of the Laplacian of the grey CAM_FRONT render (800x450) and of openpilot's road model frame (512x256, as bv_dump),
5 frames per scene at steps 5..; plan-speed ratio = model plan point 0 / 0.156 / ego speed on steps with v > 3 m/s (as bv_dump).
CPU: $DATA_DIR/envs/hugsim/bin/python experiments/leaderboard_audit/scripts/sc_hugsim_probe.py [out.json]"""
import csv, json, os, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import cv2  # noqa: E402
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
cams_yaml = D / "third_party/HUGSIM/configs/sim/{}_camera.yaml"
rows = [r for r in csv.DictReader(open(REPO / "experiments/leaderboard_audit/results/loss_budget/hugsim_inputs/exam_scored_op.csv")) if r["tag"] == "cinque-fixed"]
lap = lambda g: float(cv2.Laplacian(g, cv2.CV_64F).var())  # noqa: E731
def hf(g, r0=0.25):
    """Share of spectral power (DC excluded) above r0 cycles/px: a content-light softness measure (soft upsampled renders have little)."""
    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(g.shape[0])), np.fft.fftshift(np.fft.fftfreq(g.shape[1])), indexing="ij")
    return float(F[np.hypot(fy, fx) > r0].sum() / F.sum())


out = []
for r in rows:
    d = Path(r["run_dir"].replace("/root/autodl-tmp/ujs", str(D)))
    try:
        infos = pickle.load(open(d / "infos.pkl", "rb"))
        steps = {s["step"]: s for s in (json.loads(x) for x in open(d / "zs_steps.jsonl")) if "step" in s}
        v = np.array([float(np.ravel(i["ego_velo"])[0]) for i in infos])
        ratios = [float(np.asarray(s["model_pos"])[0][0]) / 0.156 / v[k] for k, s in steps.items() if k < len(v) and v[k] > 3 and s.get("model_pos")]
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = sorted({min(n - 1, k) for k in (5, 8, 11, 14, 17)})
        cal = Z.calibs(infos[0]["cam_params"], Z.rect_matrix(str(cams_yaml).format(r["dataset"])))
        op = Z.OpenpilotFrames(cal)
        s_native, s_road, h_native, h_road = [], [], [], []
        for k in range(max(want) + 1):
            ok, f = cap.read()
            if not ok:
                break
            if k in want:
                f = f[..., ::-1]
                h, w = f.shape[0] // 2, f.shape[1] // 3
                vid = {"CAM_FRONT_LEFT": f[:h, :w], "CAM_FRONT": f[:h, w:2 * w], "CAM_FRONT_RIGHT": f[:h, 2 * w:]}
                cat = np.concatenate([vid[c].reshape(-1, 3) for c in op.cams] + [np.zeros((1, 3), np.uint8)])
                road = cat[op.idx["road"]].reshape(G.OP_H, G.OP_W, 3)
                gn = cv2.cvtColor(np.ascontiguousarray(vid["CAM_FRONT"]), cv2.COLOR_RGB2GRAY)
                gr = cv2.cvtColor(np.ascontiguousarray(road), cv2.COLOR_RGB2GRAY)
                s_native.append(lap(gn)), s_road.append(lap(gr))
                h_native.append(hf(gn.astype(np.float64))), h_road.append(hf(gr.astype(np.float64)))
        c = cal["CAM_FRONT"]
        out.append(dict(scenario=r["scenario"], dataset=r["dataset"], speed_ratio=float(np.median(ratios)) if ratios else None, n_speed=len(ratios),
                        lap_native=float(np.mean(s_native)), lap_road=float(np.mean(s_road)), hf_native=float(np.mean(h_native)), hf_road=float(np.mean(h_road)), size=[c["width"], c["height"]], f=float(c["intrinsic"][0]),
                        cam_z=float(np.asarray(c["extrinsic"])[2, 3])))
        print(out[-1], flush=True)
    except Exception as ex:
        print("FAIL", r["scenario"], repr(ex)[:200], flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"))
