"""vis_train prereg read 5: arm C's forgetting read, the straight-road ADE on comma1M's native cameras against the shipped model
(decision 137's guardrail: the fine-tuned arms there had ADE x2.6 on these windows). Reuses experiments/op_wide_ft/scripts/wide_real.py's comma1M
replay (op_fov's ArmWarper frames: road 910 / wide 455 px focal, the native rig; windows of experiments/op_fov/results/events.json, split
comma1m/fov-full@v1, kind straight, 30 windows) and op_fov's `window_metrics` (ADE5 of the plan over the first 5 s against the logged path,
v_ratio, lane_w10), unchanged.

A P2-format checkpoint (VT-C / VT-F0 / VT-F / SH30-F) is served as in HUGSIM: its ONNX (pp_hugsim.py onnx: trained initializers, vision weights
included, `intent_bias` input) plus the bias of its ego adapter. The ego features of a comma frame follow lib/parity_hugsim's conventions on the
real clock: 4 poses at -1.5 / -1.0 / -0.5 / 0 s relative to the current camera pose (x forward, y left, yaw left), forward speed from the
localizer, acceleration = the 5-frame smoothed speed derivative, no lateral components, command straight. The pose is the camera pose, not the rear
axle (straight windows: a difference below the pose resolution). `shipped` = the stock Cinque ONNX, no bias.

  prep   (op-train)   --tags T...  ONNX + per-window ego bias for each tag  -> $D/runs/vis_train/native/{onnx,bias}/
  run    (openpilot)  --tags T...  decode the windows once, replay every missing tag -> native/res/<tag>/<window>.npz (fov_replay layout, cached)
  table  (any, numpy) --tags T...  ADE5 per window x tag, ratio to shipped and paired difference to the control, by-segment cluster bootstrap
                                   -> native/table.csv, native/table.md   (vt_read.py `build` renders it into reads.md)
  submit (box, .venv) [--tags auto|T...]  the three stages as chained pool jobs (CPU / one GPU / CPU); `auto` = shipped, SH30-F-s0,
                                   and every existing VT-C-s0* / VT-F0-s0* checkpoint. Cached per tag: rerun after each new C snapshot.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/op_adapt_l/scripts"),
                str(REPO / "experiments/op_fov/scripts"), str(REPO / "experiments/op_wide_ft/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NAT = D / "runs/vis_train/native"
RUNS = D / "runs/op_parity/runs"
SPLIT = "comma1m/fov-full@v1"
VERSION = "vt_native-1"
KEYS_S = np.array([-1.5, -1.0, -0.5, 0.0])


def windows():
    import fov_replay as FR
    from jevdrive.data import splits
    ev = {FR.wname(e): e for e in json.load(open(REPO / "experiments/op_fov/results/events.json"))}
    return [ev[m] for m in splits.load(SPLIT).members if ev[m]["kind"] == "straight"]


def tag_ckpt(tag):
    return RUNS / tag / "ckpt-final.pt"


def ego_rows(meta, lo, hi):
    """(hi - lo, 20) parity ego features of the window's frames (module doc for the conventions)."""
    from scipy.spatial.transform import Rotation
    import parity_adapter as PA
    from fov_report import calib_future
    t, R = meta["t_loc"], meta["R"]
    z = dict(t=t, pos=meta["pos"], R=R, rpy_calib=meta["rpy_calib"])
    cfd = Rotation.from_euler("xyz", meta["rpy_calib"]).as_matrix().T
    v = np.linalg.norm(meta["vel"], axis=1)
    a = np.gradient(np.convolve(v, np.ones(5) / 5, mode="same"), t)
    out = []
    for i in range(lo, hi):
        p = calib_future(z, i, KEYS_S)                                   # x fwd, y right in frame i's calibrated frame; row 3 = 0
        j = np.clip(np.searchsorted(t, t[i] + KEYS_S), 0, len(t) - 1)
        d = np.stack([(cfd @ R[i].T @ R[k] @ cfd.T)[:, 0] for k in j])   # forward axis of the past frames in frame i's frame
        pose = np.stack([p[:, 0], -p[:, 1], -np.arctan2(d[:, 1], d[:, 0])], -1)
        out.append(PA.ego_features(pose, np.tile([v[i], 0.0], (4, 1)), np.tile([a[i], 0.0], (4, 1)), np.array([0, 1, 0, 0], np.float32)))
    return np.stack(out)


# ---------------------------------------------------------------- prep (op-train)
def cmd_prep(a):
    import torch
    import parity_adapter as PA
    import pp_hugsim as H
    from jevdrive.openpilot.frames import load_segment_meta
    import fov_replay as FR
    wins = windows()
    for tag in a.tags:
        if tag == "shipped":
            continue
        onnx = NAT / "onnx" / f"{tag}.onnx"
        if not onnx.exists():
            onnx.parent.mkdir(parents=True, exist_ok=True)
            H.onnx_cmd(argparse.Namespace(tag=tag, out=str(onnx)))
        ad = PA.ParityAdapter(use_ego=True, use_side=False)
        ad.load_state_dict(torch.load(tag_ckpt(tag), map_location="cpu", weights_only=False)["model"]["parity"])
        ad.eval()
        for w in wins:
            f = NAT / "bias" / tag / f"{FR.wname(w)}.npy"
            if f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            ego = ego_rows(load_segment_meta(FR.ROOT / w["seg"]), w["start"], w["hi"])
            with torch.no_grad():
                b = torch.cat([ad(torch.from_numpy(ego[i:i + 64]), None, None) for i in range(0, len(ego), 64)])
            np.save(f.with_suffix(".tmp.npy"), b.half().numpy())
            os.replace(f.with_suffix(".tmp.npy"), f)
        print("prep done", tag, flush=True)


# ---------------------------------------------------------------- run (openpilot env)
def cmd_run(a):
    import fov_replay as FR
    from jevdrive.openpilot.frames import decode_hevc, load_segment_meta
    from jevdrive.openpilot.model import OPModel, mdn_mu, sigmoid
    for tag in a.tags:
        wins = [w for w in windows() if not (NAT / "res" / tag / f"{FR.wname(w)}.npz").exists()]
        if not wins:
            continue
        m = OPModel("cinque" if tag == "shipped" else str(NAT / "onnx" / f"{tag}.onnx"), a.backend)
        sl = m.slices
        for w in wins:
            meta = load_segment_meta(FR.ROOT / w["seg"])
            lo, hi = w["start"], w["hi"]
            warp = FR.ArmWarper(meta["rpy_calib"], 910.0, 455.0)
            fr = np.zeros((hi - lo, 2, 6, 128, 256), np.uint8)
            for n, (pr, pw) in enumerate(zip(decode_hevc(FR.ROOT / w["seg"] / "fcamera.hevc"), decode_hevc(FR.ROOT / w["seg"] / "ecamera.hevc"))):
                if n >= hi:
                    break
                if n >= lo:
                    warp(pr, pw, fr[n - lo])
            v, yr = FR.motion(meta)
            bias = None if tag == "shipped" else np.load(NAT / "bias" / tag / f"{FR.wname(w)}.npy")
            m.reset()
            m.extra = {}
            raw = []
            for i, x in enumerate(fr):
                if bias is not None:
                    m.extra["intent_bias"] = bias[i][None]
                raw.append(m.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=(0.275, 0.525)))
            raw = np.array(raw)
            res = {f"{tag}_act": raw[:, sl["action"]].astype(np.float32),
                   f"{tag}_plan": np.array([mdn_mu(r[sl["plan"]], (33, 15)) for r in raw], np.float32),
                   f"{tag}_ll": np.array([mdn_mu(r[sl["lane_lines"]], (4, 33, 2)) for r in raw], np.float32),
                   f"{tag}_lp": np.array([sigmoid(r[sl["lane_lines_prob"]])[1::2] for r in raw], np.float32),
                   f"{tag}_miss": np.array([warp.miss["road"], warp.miss["wide"]])}
            f = NAT / "res" / tag / f"{FR.wname(w)}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f.with_suffix(".tmp.npz"), **res, t=meta["t_loc"], v=v, yr=yr, lo=lo, hi=hi, pos=meta["pos"], R=meta["R"],
                     rpy_calib=meta["rpy_calib"], ev=json.dumps(w))
            os.replace(f.with_suffix(".tmp.npz"), f)
            print("replayed", tag, FR.wname(w), flush=True)


# ---------------------------------------------------------------- table
FINAL = {"VT-C": 40, "VT-F0": 60}                                 # thousand steps of the final checkpoints (vt_read.STEPS)


def step_of(tag):
    import re
    m = re.search(r"-k(\d+)$", tag)
    return int(m.group(1)) if m else FINAL.get(tag.rsplit("-s", 1)[0], 0)


def cmd_table(a):
    import pandas as pd
    from fov_report import window_metrics
    from jevdrive import stats
    rows = []
    for tag in a.tags:
        for f in sorted((NAT / "res" / tag).glob("*.npz")):
            rows += window_metrics(f)
    d = pd.DataFrame(rows)
    d["arm"] = d.arm.astype(str)
    assert len(d), "no replayed window"
    out = []
    for m in ("ADE5", "v_ratio", "lane_w10"):
        p = d.pivot(index="win", columns="arm", values=m)
        seg = d.groupby("win").seg.first()[p.index].to_numpy()
        for tag in p.columns:
            ok = p[tag].notna().to_numpy()
            out.append(dict(metric=m, tag=tag, ref="", kind="value", nwin=int(ok.sum()), **stats.bootstrap(p[tag].to_numpy(float)[ok], groups=seg[ok])))
            if "shipped" in p and tag != "shipped":
                ok = (p[tag].notna() & p["shipped"].notna()).to_numpy()
                if ok.sum() > 1:
                    out.append(dict(metric=m, tag=tag, ref="shipped", kind="ratio", nwin=int(ok.sum()),
                                    **stats.bootstrap((p[tag] / p["shipped"]).to_numpy(float)[ok], groups=seg[ok])))
            if tag.startswith("VT-C-"):                              # the paired control: F0 at the nearest registered step (every 10k)
                f0 = [c for c in p.columns if c.startswith("VT-F0-s0")]
                if f0:
                    c = min(f0, key=lambda c: (abs(step_of(c) - step_of(tag)), step_of(c)))
                    ok = (p[tag].notna() & p[c].notna()).to_numpy()
                    if ok.sum() > 1:
                        out.append(dict(metric=m, tag=tag, ref=c, kind="diff", nwin=int(ok.sum()),
                                        **stats.paired(p[tag].to_numpy(float)[ok], p[c].to_numpy(float)[ok], groups=seg[ok])))
    t = pd.DataFrame(out)
    t.to_csv(NAT / "table.csv", index=False)
    ade = t[t.metric == "ADE5"]
    lines = ["| tag | vs | kind | n windows | mean [lo, hi] |", "|:--|:--|:--|--:|--:|"]
    for _, r in ade.iterrows():
        lines.append(f"| {r.tag} | {r.ref or '-'} | {r.kind} | {r.nwin} | {r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}] |")
    (NAT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- submit (box, .venv)
def auto_tags():
    tags = ["shipped", "SH30-F-s0"]
    for pat in ("VT-F0-s0", "VT-C-s0"):
        tags += sorted(p.parent.name for p in RUNS.glob(f"{pat}*/ckpt-final.pt") if p.parent.name == pat or p.parent.name.startswith(pat + "-k"))
    return tags


def cmd_submit(a):
    tags = auto_tags() if a.tags == ["auto"] else a.tags
    me = Path(__file__).resolve()
    ld = NAT / "pool" / subprocess.run(["date", "+%Y%m%d-%H%M%S"], capture_output=True, text=True).stdout.strip()
    py = lambda env: str(D / "envs" / env / "bin/python")  # noqa: E731
    cl = [sys.executable, "-m", "jevdrive.cl", "submit", "--owner", "vis_train", "--priority", "3"]
    T = ["--tags", *tags]

    def sub(name, extra, cmd):
        return subprocess.run([*cl, "--name", name, "--log-dir", str(ld / name), *extra, "--", *cmd], check=True, capture_output=True, text=True, cwd=REPO).stdout.strip()
    j1 = sub("vt-native-prep", ["--vram", "0.5", "--cpu", "4", "--ram", "24"], [py("op-train"), str(me), "prep", *T])
    j2 = sub("vt-native-run", ["--vram", "8", "--cpu", "4", "--ram", "24", "--after", j1], [py("openpilot"), str(me), "run", *T])
    j3 = sub("vt-native-table", ["--vram", "0.5", "--cpu", "2", "--ram", "8", "--after", j2], [py("navsim2"), str(me), "table", *T])
    print(tags, j1, j2, j3, ld)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("prep", "run", "table", "submit"):
        p = sp.add_parser(n)
        p.add_argument("--tags", nargs="+", default=["auto"] if n == "submit" else None, required=n != "submit")
        if n == "run":
            p.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "table": cmd_table, "submit": cmd_submit}[a.cmd](a)
