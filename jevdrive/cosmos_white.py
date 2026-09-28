"""Cosmos pilot v2: where do objects come out pure white? (todos/2026-09-28-cosmos-pilot.md, review point 1). envs/jevdrive.

Unit: one CARLA object instance (semantic tag + instance id from the re-render's instance segmentation) in one frame,
>= 300 px, in one Cosmos output clip. "White" = at least half of its pixels are bright and unsaturated in the Cosmos
frame (max channel >= 215, HSV saturation <= 0.15) while at most 20% are in the CARLA frame (a car that is white in
CARLA is not a failure). Per unit we also keep what the control gave the model inside the object: edge density
(share of edge pixels of the control edge video), the seg colour's brightness, the depth, the frame index.

  python -m jevdrive.cosmos_white [--variants edgeA,edgeB,seg]  -> research/results/cosmos/white_units.csv.gz, white.md
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .cosmos_eval import H, RESULTS, ROOT, SEED, W, load, read_mp4
from .cosmos_pilot import _attempt, _load_frame, pairs as _pairs, root

TAGS = {1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole", 7: "traffic light",
        8: "traffic sign", 9: "vegetation", 10: "terrain", 12: "pedestrian", 13: "rider", 14: "car", 15: "truck",
        16: "bus", 18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic", 28: "guard rail"}
OBJ = {5, 6, 7, 8, 12, 13, 14, 15, 16, 18, 19, 20, 21}          # the object-like classes the review is about
MIN_PX, WHITE_V, WHITE_S, WHITE_FRAC, RAW_MAX = 300, 215, 0.15, 0.5, 0.2


def _white(img: np.ndarray) -> np.ndarray:
    mx, mn = img.max(-1).astype(np.float32), img.min(-1).astype(np.float32)
    return (mx >= WHITE_V) & ((mx - mn) / np.maximum(mx, 1) <= WHITE_S)


def _one(job) -> list:
    pair, member, variant, out_path, ctrl_edge = job
    r = _pairs().set_index("pair").loc[pair]
    a = _attempt(root("gen"), r[member])
    cos, raw = load(Path(out_path)), read_mp4(root("clips", pair, member) / "rgb.mp4")
    edge = read_mp4(Path(ctrl_edge))[..., 0] > 127 if ctrl_edge and Path(ctrl_edge).exists() else None
    seg = read_mp4(root("clips", pair, member) / "seg.mp4")
    wc, wr = _white(cos), _white(raw)

    def hsv(img, m):
        x = img[m].astype(np.float32)
        mx, mn = x.max(1), x.min(1)
        return float(np.median(mx)), float(np.median((mx - mn) / np.maximum(mx, 1))), float(x.mean(1).std())
    rows = []
    for t, k in enumerate(range(int(r.k0), int(r.k1))):
        _, depth, tag, inst = _load_frame(a, k)
        key = tag.astype(np.int64) * 65536 + np.where(np.isin(tag, list(OBJ)), inst, 0)
        sel = np.isin(tag, list(OBJ))
        ks, inv, n = np.unique(key[sel], return_inverse=True, return_counts=True)
        for j in np.flatnonzero(n >= MIN_PX):
            m = np.zeros(tag.shape, bool)
            m[sel] = inv == j
            vc, sc, tc = hsv(cos[t], m)
            vr, sr, tr = hsv(raw[t], m)
            rows.append({"v_cos": vc, "s_cos": sc, "tex_cos": tc, "v_raw": vr, "s_raw": sr, "tex_raw": tr,
                         "variant": variant, "pair": pair, "member": member, "t": t, "cls": TAGS.get(int(ks[j] // 65536), "other"),
                         "px": int(n[j]), "depth": float(np.median(depth[m])), "white": float(wc[t][m].mean()),
                         "white_raw": float(wr[t][m].mean()), "std_cos": float(cos[t][m].std()),
                         "edge_density": float(edge[t][m].mean()) if edge is not None else np.nan,
                         "seg_luma": float(seg[t][m].mean())})
    return rows


def _flags(d: pd.DataFrame) -> pd.DataFrame:
    d["is_white"] = (d.white >= WHITE_FRAC) & (d.white_raw <= RAW_MAX)
    pale = lambda v, s_, t, t_ref: (v >= 140) & (s_ <= 0.25) & (t <= 0.6 * t_ref)  # noqa: E731
    d["is_pale"] = pale(d.v_cos, d.s_cos, d.tex_cos, d.tex_raw) & ~((d.v_raw >= 140) & (d.s_raw <= 0.25) & (d.tex_raw <= 20))
    # absolute white mannequin (added after seeing 25863 in fog, where CARLA's own object is pale too and the relative
    # flag cannot fire): Cosmos object near-white, grey, flat, while CARLA's object is not near-white
    d["is_mannequin"] = (d.v_cos >= 170) & (d.s_cos <= 0.15) & (d.tex_cos <= 25) & ~((d.v_raw >= 170) & (d.s_raw <= 0.15))
    d["is_white_any"] = d.is_pale | d.is_mannequin
    return d


def run(variants=("edgeA", "edgeB", "seg"), tag: str = ""):
    jobs = []
    for v in variants:
        od = ROOT / "out" / v
        for pair in _pairs().pair:
            for m in ("plus", "minus"):
                f = od / f"{pair}_{m}_{v}_s{SEED}.npy"
                if not f.exists():
                    continue
                ctrl = {"edgeA": od / f"{pair}_{m}_{v}_s{SEED}_control_edge.mp4",
                        "edgeB": root("clips", pair, m) / "edge.mp4", "P2": root("clips", pair, m) / "edge.mp4",
                        **{k: root("clips", pair, m) / "edgeC.mp4" for k in ("E2", "G2b", "M2")},
                        **{k: root("clips", pair, m) / "edgeD.mp4" for k in ("E3", "G3b")},
                        **{k: root("clips", pair, m) / "edgeE.mp4" for k in ("E4", "G4b")}}.get(v, "")
                jobs.append((pair, m, v, str(f), str(ctrl)))
    with ProcessPoolExecutor(min(20, len(jobs))) as ex:
        d = pd.DataFrame([x for rows in ex.map(_one, jobs) for x in rows])
    d = _flags(d)
    d.to_csv(RESULTS / f"white_units{tag}.csv.gz", index=False)
    return summarize(d, tag=tag)


def summarize(d: pd.DataFrame, flag: str = "is_white_any", tag: str = "") -> str:
    out = [f"flag: {flag}"]
    d = d.assign(is_white=d[flag])
    g = d.groupby(["variant", "cls"]).agg(units=("is_white", "size"), white=("is_white", "mean")).unstack(0)
    out.append("## White units by variant and class (share of object-frames >= 300 px)\n\n" + g.round(3).to_markdown())
    d["depth_bin"] = pd.cut(d.depth, [0, 10, 20, 40, 1000], labels=["<10 m", "10-20", "20-40", ">40"])
    d["size_bin"] = pd.cut(d.px, [0, 1000, 3000, 10000, 1e9], labels=["<1k px", "1-3k", "3-10k", ">10k"])
    for col in ("depth_bin", "size_bin"):
        out.append(f"## by {col}\n\n" + d.pivot_table(index=col, columns="variant", values="is_white", aggfunc="mean",
                                                       observed=False).round(3).to_markdown())
    d["t_bin"] = pd.cut(d.t, [-1, 15, 45, 77, 93], labels=["0-15", "16-45", "46-77", "78-92"])
    out.append("## by frame index\n\n" + d.pivot_table(index="t_bin", columns="variant", values="is_white", aggfunc="mean",
                                                        observed=False).round(3).to_markdown())
    e = d[d.edge_density.notna()].copy()
    if len(e):
        e["edge_bin"] = pd.qcut(e.edge_density, 4, duplicates="drop")
        out.append("## edge variants: by control edge density inside the object\n\n" + e.pivot_table(
            index="edge_bin", columns="variant", values="is_white", aggfunc="mean", observed=False).round(3).to_markdown())
    s = d[d.variant == "seg"].copy()
    if len(s):
        s["luma_bin"] = pd.cut(s.seg_luma, [0, 80, 140, 200, 256])
        out.append("## seg: by the instance colour's brightness\n\n" + s.groupby("luma_bin", observed=False).is_white.agg(
            ["size", "mean"]).round(3).to_markdown())
    w = d[d.is_white].groupby(["variant", "pair", "member", "cls"]).agg(units=("t", "size"), first_t=("t", "min"),
                                                                          last_t=("t", "max"), px_med=("px", "median"))
    out.append("## where (pair, member, class)\n\n" + w.to_markdown())
    txt = "\n\n".join(out) + "\n"
    (RESULTS / f"white{tag}.md").write_text(txt)
    print(txt)
    return txt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="edgeA,edgeB,seg")
    ap.add_argument("--summarize", action="store_true", help="re-summarize white_units.csv.gz")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.summarize:
        from .cosmos_white import _flags
        summarize(_flags(pd.read_csv(RESULTS / f"white_units{a.tag}.csv.gz")), tag=a.tag)
    else:
        run(tuple(a.variants.split(",")), a.tag)
