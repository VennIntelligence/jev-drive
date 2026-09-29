"""Cosmos pilot checks (todos/2026-09-28-cosmos-pilot.md). Each step runs in the env that has its model; numpy + the
ffmpeg binary are the only shared dependencies.

  detect   envs/ultralytics  YOLO26x-seg 640 fp16 person boxes on raw and Cosmos clips -> runs/cosmos/det/<v>/<pair>.npz
  pixels   envs/r3d2         PSNR / LPIPS outside the hazard region, flicker -> runs/cosmos/pix/<v>/<pair>.json
  report   envs/jevdrive     checks 1-4 against the pass lines -> research/results/cosmos/, WebP side-by-sides
"""
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np

DATA = Path(os.environ["DATA_DIR"])
ROOT = Path(os.environ.get("COSMOS_ROOT", DATA / "runs" / "cosmos"))   # COSMOS_ROOT / COSMOS_RESULTS: full-run staged pilots
REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("COSMOS_RESULTS", REPO / "research" / "results" / "cosmos"))
FIGS = REPO / "research" / "figs" / "cosmos"
H, W = 704, 1280
SEED, SEED_ALT = 2025, 2026
VIS_PX, IOU, CONF = 300, 0.3, 0.25
WARM = 40                                   # openpilot comparisons use frames 40-92 (todo)


def ffmpeg() -> str:
    return sorted(glob.glob(str(DATA / "envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*")))[0]


def read_mp4(path: Path) -> np.ndarray:
    out = subprocess.run([ffmpeg(), "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W, 3)


def pairs() -> list[str]:
    import csv
    return [r["pair"] for r in csv.DictReader(open(RESULTS / "pairs.csv"))]


ALT_FROM = {"M2": "E2", "P2": "E2"}     # G*b link their base arm's x- and floor     # arms without their own seed floor borrow E2's (same prompt, same x- geometry)


def streams(pair: str, v: str) -> dict:
    """name -> loader for every clip of a pair that exists."""
    c, o = ROOT / "clips" / pair, ROOT / "out" / v
    s = {"raw_plus": c / "plus" / "rgb.mp4", "raw_minus": c / "minus" / "rgb.mp4",
         "plus": o / f"{pair}_plus_{v}_s{SEED}.npy", "minus": o / f"{pair}_minus_{v}_s{SEED}.npy",
         "rep": o / f"{pair}_minus_{v}_s{SEED}_rep.npy", "alt": o / f"{pair}_minus_{v}_s{SEED_ALT}.npy"}
    if not s["alt"].exists() and v in ALT_FROM:
        b = ALT_FROM[v]
        s["alt"], s["alt_base"] = (ROOT / "out" / b / f"{pair}_minus_{b}_s{SEED_ALT}.npy",
                                   ROOT / "out" / b / f"{pair}_minus_{b}_s{SEED}.npy")
    return {k: p for k, p in s.items() if p.exists()}


def load(p: Path) -> np.ndarray:
    return read_mp4(p) if p.suffix == ".mp4" else np.load(p)


def gt(pair: str) -> dict:
    z = np.load(ROOT / "clips" / pair / "gt.npz")
    un = lambda a: np.unpackbits(a, axis=1)[:, : H * W].reshape(-1, H, W).astype(bool)  # noqa: E731
    return {"mask": un(z["mask"]), "region": un(z["region"]), "walk_minus": un(z["walk_minus"]), "px": z["px"],
            "box": z["box"], "corridor": z["corridor"]}


# ------------------------------------------------------------------ detect (envs/ultralytics)

def detect(v: str, only: str = ""):
    from ultralytics import YOLO
    m = YOLO(str(DATA / "models" / "ultralytics" / "yolo26x-seg.pt"))
    od = ROOT / "det" / v
    od.mkdir(parents=True, exist_ok=True)
    for pair in [p for p in pairs() if not only or p in only.split(",")]:
        res = {}
        for name, path in streams(pair, v).items():
            frames = load(path)
            rows = []
            for i in range(0, len(frames), 16):
                for j, r in enumerate(m.predict([f[..., ::-1] for f in frames[i:i + 16]], imgsz=640, conf=CONF,
                                                half=True, classes=[0], verbose=False)):
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        rows.append([i + j, *b, c])
            res[name] = np.array(rows, np.float32).reshape(-1, 6)
        np.savez(od / f"{pair}.npz", **res)
        print(pair, {k: len(x) for k, x in res.items()}, flush=True)


# ------------------------------------------------------------------ pixels (envs/r3d2: torch + lpips)

def pixels(v: str, only: str = "", gpu: int = 0):
    import lpips
    import torch
    dev = f"cuda:{gpu}"
    net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    od = ROOT / "pix" / v
    od.mkdir(parents=True, exist_ok=True)

    def lp(a, b):
        out = []
        with torch.no_grad():
            for i in range(0, len(a), 8):
                ta = torch.from_numpy(a[i:i + 8]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
                tb = torch.from_numpy(b[i:i + 8]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
                out.append(net(ta, tb).flatten().cpu().numpy())
        return np.concatenate(out)

    def psnr_out(a, b, region):
        d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
        keep = ~region
        mse = np.array([d[t][keep[t]].mean() for t in range(len(a))])
        return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))

    def flicker(a, region):
        g = a.astype(np.float32).mean(-1)
        d2 = np.abs(g[2:] - 2 * g[1:-1] + g[:-2])
        keep = ~region[1:-1]
        return np.array([d2[t][keep[t]].mean() for t in range(len(d2))])

    for pair in [p for p in pairs() if not only or p in only.split(",")]:
        g = gt(pair)
        region = g["region"]
        S = {k: load(p) for k, p in streams(pair, v).items()}
        out = {"pair": pair, "vis": (g["px"] >= VIS_PX).tolist(), "px": g["px"].tolist()}
        # (x, y): compare x to y outside the region; LPIPS on x with the region filled from y
        # edge leak (v2): ring 1-12 px outside the pedestrian's own pixel mask, frames where it is >= VIS_PX
        mt = torch.from_numpy(g["mask"]).float()[:, None]
        ring = (torch.nn.functional.max_pool2d(mt, 25, 1, 12)[:, 0].numpy() > 0) & ~g["mask"]   # 12 px dilation
        rv = g["px"] >= VIS_PX
        alt_ref = "alt_base" if "alt_base" in S else "minus"
        for name, (x, y) in {"pair": ("plus", "minus"), "alt": ("alt", alt_ref), "raw": ("raw_plus", "raw_minus")}.items():
            if x in S and y in S:
                out[f"ring_mad_{name}"] = [float(np.abs(S[x][t].astype(np.int16) - S[y][t])[ring[t]].mean()) if rv[t] else None
                                           for t in range(len(rv))]
        for name, (x, y) in {"pair": ("plus", "minus"), "rep": ("rep", "minus"), "alt": ("alt", alt_ref),
                             "raw": ("raw_plus", "raw_minus"), "tr_minus": ("minus", "raw_minus")}.items():
            if x in S and y in S:
                comp = np.where(region[..., None], S[y], S[x])
                out[f"psnr_{name}"] = psnr_out(S[x], S[y], region).tolist()
                out[f"lpips_{name}"] = lp(comp, S[y]).tolist()
        for k in ("plus", "minus", "raw_plus", "raw_minus", "rep", "alt"):
            if k in S:
                out[f"flicker_{k}"] = flicker(S[k], region).tolist()
        (od / f"{pair}.json").write_text(json.dumps(out))
        print(pair, {k: round(float(np.median(x)), 4) for k, x in out.items() if k.startswith(("psnr", "lpips", "flicker"))},
              flush=True)


def composite(src: str = "edgeB", dst: str = "edgeBc"):
    """Pixel composite, no new generation: x+ = Cosmos x- outside the dilated hazard region, Cosmos x+ inside; the x-
    members are links to the source variant's (todo, "what to change" 1)."""
    s, d = ROOT / "out" / src, ROOT / "out" / dst
    d.mkdir(parents=True, exist_ok=True)
    for p in pairs():
        r = gt(p)["region"][..., None]
        plus, minus = np.load(s / f"{p}_plus_{src}_s{SEED}.npy"), np.load(s / f"{p}_minus_{src}_s{SEED}.npy")
        np.save(d / f"{p}_plus_{dst}_s{SEED}.npy", np.where(r, plus, minus))
        for a, b in ((f"{p}_minus_{src}_s{SEED}.npy", f"{p}_minus_{dst}_s{SEED}.npy"),
                     (f"{p}_minus_{src}_s{SEED_ALT}.npy", f"{p}_minus_{dst}_s{SEED_ALT}.npy")):
            (d / b).unlink(missing_ok=True)
            (d / b).symlink_to(s / a)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect", "pixels", "report", "webp", "composite"))
    ap.add_argument("--variant", default="edgeA")
    ap.add_argument("--only", default="")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.step == "detect":
        detect(a.variant, a.only)
    elif a.step == "pixels":
        pixels(a.variant, a.only, a.gpu)
    elif a.step == "composite":
        composite(a.variant, a.variant + "c")
    elif a.step == "webp":
        from .cosmos_report import webp
        FIGS.mkdir(parents=True, exist_ok=True)
        for pair in a.only.split(","):
            out = FIGS / f"{pair}_{a.variant}.webp"
            print(out, webp(a.variant, pair, out))
    else:
        from .cosmos_report import report
        report(a.variant.split(","), a.only)


if __name__ == "__main__":
    main()
