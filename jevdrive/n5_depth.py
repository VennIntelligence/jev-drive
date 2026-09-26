"""Night queue 2, N5 (todos/2026-09-26-night-queue-2.md): camera-conditioned monocular metric depth in place of the
flat-ground lift, in the recall *measurement* only (decision 45). Everything around the depth is decision 45's code:
the subset S, the stored detections, the contact pixel, the depth sampling rule of `fastperc.depth_sample`, the ray
placement `fastperc._depth_place` and `fastperc.evaluate --contact depth`.

  depth  (envs/depth, GPU)  per S image with a detection: metric depth map at the original resolution from
         unidepth  UniDepth v2 ViT-L/14, `infer(rgb, K)` with the pinhole intrinsics (primary)
         da3       Depth Anything 3 DA3METRIC-LARGE, `inference([img])` at its default processing resolution,
                   metres = focal * output / 300 (authors' README), focal = mean(fx, fy) at the processed size
         then, per stored detection of every detector, the median depth in the 5 x 5 window whose centre is 3 px above
         the contact pixel -> processed/night2/n5/dets/<det>-<model>/{p5,nusc}/part-0000.parquet (depth_sample's schema)
  eval   (envs/jevdrive, CPU)  fastperc.evaluate on each of those dirs; side: the same with a fixed 2 m match gate
  table  the N5 readout table -> research/results/night2/N5/
"""
import json
import os
from pathlib import Path

import numpy as np

DETS = {"yolo26x-640": ("processed/fastperc/dets/yolo26x-640", 0.25),     # tag -> (stored detections, default thr)
        "sam31-orig": ("processed/fusion_diag/sam", 0.5)}
MODELS = ("unidepth", "da3")
MIN_SCORE = 0.05


def data() -> Path:
    return Path(os.environ["DATA_DIR"])


def out_root(*p) -> Path:
    d = data() / "processed/night2/n5" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================ GPU (envs/depth)

class UniDepth:
    def __init__(self):
        import torch
        from unidepth.models import UniDepthV2
        self.m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
        self.torch = torch

    def __call__(self, img, K):
        """img uint8 CHW tensor, K (3, 3) -> z depth (H, W) float32 tensor on the GPU."""
        with self.torch.inference_mode():
            return self.m.infer(img[None].cuda(), self.torch.as_tensor(K, dtype=self.torch.float32)[None].cuda())["depth"][0, 0].float()


class DA3Metric:
    def __init__(self):
        import torch
        from depth_anything_3.api import DepthAnything3
        self.m = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
        self.torch = torch

    def __call__(self, img, K):
        torch = self.torch
        H, W = img.shape[1:]
        with torch.inference_mode():
            p = self.m.inference([img.permute(1, 2, 0).numpy()])
        d = torch.as_tensor(np.asarray(p.depth[0]), dtype=torch.float32, device="cuda")
        h, w = d.shape
        focal = 0.5 * (K[0, 0] * w / W + K[1, 1] * h / H)          # focal length at the processed size
        d = d * focal / 300.0
        return torch.nn.functional.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]


def _intrinsics(ds: str, key: str, cal: dict) -> np.ndarray:
    c = cal[key.split("|")[1]] if ds == "p5" else cal[key]
    fu, fv, cu, cv = (float(x) for x in c["intrinsic"][:4])
    return np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], np.float64)


def _sample(D, cu, cv):
    """fastperc.depth_sample's rule: median of the 5 x 5 window centred 3 px above the contact pixel."""
    import torch
    H, W = D.shape
    u = np.clip(np.nan_to_num(cu).round().astype(int), 2, W - 3)
    v = np.clip(np.nan_to_num(cv).round().astype(int) - 3, 2, H - 3)
    if not len(u):
        return np.zeros(0, np.float32)
    win = torch.stack([D[vv - 2:vv + 3, uu - 2:uu + 3].reshape(-1) for uu, vv in zip(u, v)])
    return win.median(1).values.cpu().numpy()


def depth(model: str, workers: int = 16, limit: int = 0, rl=None) -> dict:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from . import fusion_q4 as Q
    from .sam_detect import _collate, _Images
    L = data() / "processed/fusion_diag/lists"
    S = json.loads((data() / "processed/fastperc/subset.json").read_text())
    torch.set_num_threads(1)     # DA3's CPU pre/post-processing is 16x slower with many threads on the loaded box
    net = {"unidepth": UniDepth, "da3": DA3Metric}[model]()
    info = {}
    for ds, lst in (("p5", "p5.parquet"), ("nusc", "nusc.parquet")):
        cal = Q.p5_calib() if ds == "p5" else json.loads((L / "nusc_calib.json").read_text())
        dets = {}
        for tag, (d, _) in DETS.items():
            x = pd.concat([pd.read_parquet(p) for p in sorted((data() / d / ds).glob("part-*.parquet"))], ignore_index=True)
            dets[tag] = x[(x.score > MIN_SCORE) & x.key.isin(set(S[ds]))].reset_index(drop=True)
        keys = sorted(set().union(*(set(x.key) for x in dets.values())))
        t = pd.read_parquet(L / lst)
        rows = t[t.key.isin(set(keys))].to_dict("records")[: limit or None]
        gi = {tag: x.groupby("key").indices for tag, x in dets.items()}
        dep = {tag: np.full(len(x), np.nan, np.float32) for tag, x in dets.items()}
        dl = DataLoader(_Images(rows), batch_size=1, num_workers=workers, collate_fn=_collate, prefetch_factor=8)
        for idx, ims in tqdm(dl, desc=f"{model} {ds}", mininterval=30):
            k = rows[idx[0]]["key"]
            D = net(ims[0], _intrinsics(ds, k, cal))
            for tag, x in dets.items():
                j = gi[tag].get(k)
                if j is not None:
                    dep[tag][j] = _sample(D, x.cu.to_numpy()[j], x.cv.to_numpy()[j])
        for tag, x in dets.items():
            o = out_root("dets", f"{tag}-{model}", ds)
            x.assign(depth=dep[tag]).to_parquet(o / "part-0000.parquet", index=False)
            info[f"{ds} {tag}"] = {"detections": len(x), "with_depth": int(np.isfinite(dep[tag]).sum())}
        info[f"{ds} images"] = len(rows)
        if rl:
            rl.info(f"{model} {ds}: {json.dumps(info)}")
        torch.cuda.empty_cache()
    return info


# ================================================================ CPU (envs/jevdrive)

def evaluate(tag: str, rl) -> None:
    """fastperc.evaluate --contact depth on one depth dir; side: the flat-ground and depth readings with a fixed 2 m
    gate (fusion_q4.gate patched for the call)."""
    from . import fastperc as FP
    from . import fusion_q4 as Q
    det, model = tag.rsplit("-", 1)
    src, thr = DETS[det]
    real_gate = Q.gate
    for gate_name, g in (("gate", None), ("fixed2m", lambda d: np.full_like(np.asarray(d, float), 2.0))):
        if g is not None:
            Q.gate = g
        try:
            if model == "flat":
                row = FP.evaluate(str(data() / src), f"{det}-flat-{gate_name}", thr, rl.dir / f"{det}-flat-{gate_name}",
                                  "mask", sweep=False)
            else:
                row = FP.evaluate(str(out_root("dets", tag)), f"{tag}-{gate_name}", thr, rl.dir / f"{tag}-{gate_name}",
                                  "depth", sweep=False)
        finally:
            Q.gate = real_gate
        rl.info(f"eval {tag} {gate_name}: {json.dumps(row, default=float)}")


def placement_ratio(tag: str) -> "pd.DataFrame":
    """Median (depth-placed distance / flat-ground distance) per dataset and flat-distance bin, over detections with
    both placements (decision 45 reported this for YOLO26x-depth)."""
    import pandas as pd
    from . import fastperc as FP
    from . import fusion_q4 as Q
    det, model = tag.rsplit("-", 1)
    thr = DETS[det][1]
    L = Q.root("lists")
    ncal = json.loads((L / "nusc_calib.json").read_text())
    S = FP.subset()
    d = pd.concat([pd.read_parquet(p) for s in ("p5", "nusc") for p in sorted(out_root("dets", tag, s).glob("part-*.parquet"))],
                  ignore_index=True)
    d = d[(d.score > thr) & d.prompt.isin(("pedestrian", "vehicle"))]
    rows = []
    for ds in ("p5", "nusc"):
        x = d[d.key.isin(set(S[ds]))].reset_index(drop=True)
        x = Q.lift_dets(x, x.key.str.split("|").str[1].to_numpy() if ds == "p5" else x.key.to_numpy(),
                        Q.p5_calib() if ds == "p5" else ncal)
        flat = x.gdist.to_numpy()
        ok = x.lift_ok.to_numpy() & np.isfinite(x.depth.to_numpy())
        y = FP._depth_place(x)
        r = (y.gdist.to_numpy() / flat)[ok]
        b = Q.dist_bin(flat[ok])
        for (cls, bb), g in pd.DataFrame({"cls": x.prompt.to_numpy()[ok], "bin": b, "r": r}).groupby(["cls", "bin"], observed=True):
            rows.append({"tag": tag, "dataset": ds, "cls": cls, "flat_bin": str(bb), "n": len(g), "ratio_median": float(g.r.median())})
    return pd.DataFrame(rows)


COLS = ["p5_ped_0-10", "p5_ped_10-20", "p5_ped_20-40", "p5_ped_le40", "nusc_ped_0-10", "nusc_ped_10-20", "nusc_ped_20-40",
        "nusc_ped_le40", "p5_haz_ped_all", "p5_haz_ped_le20", "p5_haz_ped_le20_bev_med", "nusc_ped_prec", "p5_veh_le40",
        "nusc_veh_le40", "p5_ped_img_0-10", "p5_ped_img_10-20", "p5_ped_img_20-40", "nusc_ped_img_0-10",
        "nusc_ped_img_10-20", "nusc_ped_img_20-40"]


def verdict(x: float) -> str:
    return "fixed (>= 0.40)" if x >= 0.40 else "not enough (< 0.30)" if x < 0.30 else "in between"


def table(run: Path, out: Path) -> "pd.DataFrame":
    """Every row.json of an eval run -> out/recall.csv (all readings) and out/recall_main.md (the N5 table)."""
    import pandas as pd
    rows = []
    for f in sorted(run.glob("*/row.json")):
        r = json.loads(f.read_text())
        det = next(d for d in DETS if r["name"].startswith(d + "-"))
        place, gate = r["name"][len(det) + 1:].rsplit("-", 1)
        rows.append({"detector": det, "placement": place, "gate": gate, **{c: r.get(c) for c in COLS}})
    t = pd.DataFrame(rows).sort_values(["gate", "detector", "placement"])
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "recall.csv", index=False, float_format="%.4f")
    m = t[t.gate == "gate"].assign(p5_verdict=lambda d: d["p5_ped_20-40"].map(verdict),
                                   nusc_verdict=lambda d: d["nusc_ped_20-40"].map(verdict))
    (out / "recall_main.md").write_text(m.to_markdown(index=False, floatfmt=".3f") + "\n\nfixed 2 m gate (side)\n\n"
                                        + t[t.gate == "fixed2m"].to_markdown(index=False, floatfmt=".3f") + "\n")
    return t


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevdrive.runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("depth", "eval", "ratio", "table"))
    ap.add_argument("--run", default="", help="table: the eval run dir")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--tags", default="", help="eval / ratio: comma list of <det>-<model|flat>")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rl = RunLog("night2", "n5", a.step + (f"-{a.model}" if a.model else ""))
    rl.event("start", args=vars(a), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if a.step == "depth":
        rl.info(f"depth {a.model}: {json.dumps(depth(a.model, a.workers, a.limit, rl))}")
    elif a.step == "table":
        t = table(Path(a.run), Path(__file__).resolve().parents[1] / "research/results/night2/N5")
        rl.info("\n" + t.to_markdown(index=False, floatfmt=".3f"))
    elif a.step == "eval":
        for tag in a.tags.split(","):
            evaluate(tag, rl)
    else:
        import pandas as pd
        t = pd.concat([placement_ratio(tag) for tag in a.tags.split(",")], ignore_index=True)
        t.to_csv(rl.dir / "placement_ratio.csv", index=False)
        rl.info("\n" + t.to_markdown(index=False, floatfmt=".3f"))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
