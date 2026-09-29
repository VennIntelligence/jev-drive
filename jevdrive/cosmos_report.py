"""Cosmos pilot report (todos/2026-09-28-cosmos-pilot.md): checks 1-4 against the pass lines written before the run,
tables to research/results/cosmos/, WebP side-by-sides (raw CARLA x+ | Cosmos x+ | Cosmos x-) to research/figs/cosmos/.
envs/jevdrive (numpy, pandas, PIL, cv2)."""
import json

import numpy as np
import pandas as pd

from .cosmos_eval import (FIGS, H, IOU, RESULTS, ROOT, VIS_PX, W, WARM, gt, load, pairs, streams)

NEAR_PX = 2000
PASS = {"recall_ratio": 0.9, "recall_abs": 0.70, "recall_pair_ratio": 0.5, "halluc_abs": 0.02, "halluc_over_raw": 0.01,
        "lpips_frac_of_seed": 1 / 3, "psnr_min": 28.0, "feat_frac_of_seed": 1 / 3, "flicker_ratio": 1.5,
        "dv_med": 0.5, "dv_p90": 1.5, "dy_med": 0.3, "lead_agree": 0.85, "gpuh_2000": 250.0}


def _iou(b, g):
    x0, y0 = np.maximum(b[:, 0], g[0]), np.maximum(b[:, 1], g[1])
    x1, y1 = np.minimum(b[:, 2], g[2]), np.minimum(b[:, 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)


def check1(v: str, names: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per pair and stream: recall on GT-visible frames (near / far) and hallucinated corridor persons in x-."""
    import cv2
    rows, frames = [], []
    for pair in names:
        g = gt(pair)
        det = np.load(ROOT / "det" / v / f"{pair}.npz")
        vis = g["px"] >= VIS_PX
        # per hazard pedestrian (gt_boxes.npz): a recall unit is one visible pedestrian in one frame
        hb = np.load(ROOT / "clips" / pair / "gt_boxes.npz")
        for s in det.files:
            d = det[s]
            for t in range(len(g["px"])):
                b = d[d[:, 0] == t, 1:5]
                if s in ("raw_plus", "plus"):
                    for h in range(hb["px"].shape[1]):
                        if hb["px"][t, h] >= VIS_PX:
                            frames.append({"pair": pair, "stream": s, "t": t, "h": h, "px": int(hb["px"][t, h]), "vis": True,
                                           "hit": bool(len(b) and (_iou(b, hb["box"][t, h]) >= IOU).any())})
                    continue
                rec = {"pair": pair, "stream": s, "t": t, "px": int(g["px"][t]), "vis": bool(vis[t])}
                if s in ("raw_minus", "minus", "rep", "alt"):
                    poly = g["corridor"][t]
                    poly = poly[~np.isnan(poly[:, 0])]
                    n = 0
                    for x0, y0, x1, y1 in b:
                        foot = (float((x0 + x1) / 2), float(min(y1, H - 1)))
                        inside = len(poly) >= 3 and cv2.pointPolygonTest(poly.reshape(-1, 1, 2), foot, False) >= 0
                        ix0, iy0, ix1, iy1 = (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))
                        real = g["walk_minus"][t, iy0:iy1, ix0:ix1].any()
                        n += bool(inside and not real)
                    rec["halluc"] = n > 0
                frames.append(rec)
    f = pd.DataFrame(frames)
    rec = f[f.stream.isin(["raw_plus", "plus"]) & f.vis].assign(near=lambda x: x.px >= NEAR_PX)
    hal = f[f.stream.isin(["raw_minus", "minus", "rep", "alt"])]
    for pair, gp in rec.groupby("pair"):
        r = {"pair": pair, "vis_frames": int(gp[gp.stream == "plus"].shape[0])}
        for s in ("raw_plus", "plus"):
            q = gp[gp.stream == s]
            r[f"R_{s}"] = q.hit.mean()
            r[f"R_{s}_near"] = q[q.near].hit.mean() if q.near.any() else np.nan
            r[f"R_{s}_far"] = q[~q.near].hit.mean() if (~q.near).any() else np.nan
        for s in ("raw_minus", "minus"):
            r[f"H_{s}"] = hal[(hal.pair == pair) & (hal.stream == s)].halluc.mean()
        rows.append(r)
    return pd.DataFrame(rows), f


def check2_3(v: str, names: list) -> pd.DataFrame:
    rows = []
    for pair in names:
        pj = json.loads((ROOT / "pix" / v / f"{pair}.json").read_text())
        op = np.load(ROOT / "op" / v / f"{pair}.npz")
        r = {"pair": pair}
        for k in ("pair", "rep", "alt", "raw", "tr_minus"):
            if f"lpips_{k}" in pj:
                r[f"lpips_{k}"] = float(np.median(pj[f"lpips_{k}"]))
                r[f"psnr_{k}"] = float(np.median(np.minimum(pj[f"psnr_{k}"], 99.0)))
        for k in ("pair", "alt", "raw"):
            x = [v_ for v_ in pj.get(f"ring_mad_{k}", []) if v_ is not None]
            r[f"ring_mad_{k}"] = float(np.median(x)) if x else np.nan
        pre = np.array(pj["px"]) < 50
        if pre.any():
            r["lpips_pair_previs"] = float(np.median(np.array(pj["lpips_pair"])[pre]))
        for m in ("plus", "minus"):
            r[f"flicker_ratio_{m}"] = float(np.mean(pj[f"flicker_{m}"]) / max(np.mean(pj[f"flicker_raw_{m}"]), 1e-6))

        def dcos(a, b):
            A, B = op[f"{a}/temporal"][WARM:], op[f"{b}/temporal"][WARM:]
            return float(np.median(1 - (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))
        pv = lambda s: s if s.startswith("raw") else f"{v}_{s}"  # noqa: E731
        alt_ref = "altbase" if f"{v}_altbase/temporal" in op.files else "minus"
        for name, (a, b) in {"comp": ("comp", "minus"), "rep": ("rep", "minus"), "alt": ("alt", alt_ref),
                             "raw_comp": ("raw_comp", "raw_minus"), "full": ("plus", "minus"),
                             "raw_full": ("raw_plus", "raw_minus")}.items():
            if f"{pv(a)}/temporal" in op.files and f"{pv(b)}/temporal" in op.files:
                r[f"d_{name}"] = dcos(pv(a), pv(b))
        t, c = f"{v}_minus", "raw_minus"
        dv = np.abs(op[f"{t}/v2"][WARM:] - op[f"{c}/v2"][WARM:])
        dy = np.abs(op[f"{t}/y2"][WARM:] - op[f"{c}/y2"][WARM:])
        la, lb = op[f"{t}/lead_prob"][WARM:] > 0.5, op[f"{c}/lead_prob"][WARM:] > 0.5
        r.update(dv_med=float(np.median(dv)), dv_p90=float(np.percentile(dv, 90)), dy_med=float(np.median(dy)),
                 lead_agree=float((la == lb).mean()), lead_raw=float(lb.mean()), lead_tr=float(la.mean()),
                 v2_raw=float(np.median(op[f"{c}/v2"][WARM:])), v2_tr=float(np.median(op[f"{t}/v2"][WARM:])))
        rows.append(r)
    return pd.DataFrame(rows)


def per_pair_rule(per: pd.DataFrame) -> pd.DataFrame:
    """v2 per-pair pass rule (registered in the todo before any v2 output): a pair passes when
    1. R_T >= 0.8 R_C (pairs with >= 10 visible units) and H_T <= H_C + 1 pp;
    2. outside-region LPIPS <= 1/3 seed-change LPIPS, PSNR >= 28 dB, openpilot d_comp <= d_raw_comp + 0.001;
    3. ring MAD (1-12 px outside the pedestrian mask) <= 2 x CARLA's + 2 grey levels and <= 1/4 seed-change ring MAD."""
    p = per.copy()
    for c in ("lpips_alt", "ring_mad_alt", "d_alt", "d_rep"):     # runs without seed floors (the full run)
        if c not in p:
            p[c] = np.nan
    small = p.vis_frames < 10
    p["pp1"] = (small | (p.R_plus >= 0.8 * p.R_raw_plus)) & (p.H_minus <= p.H_raw_minus + 0.01)
    p["pp2"] = (p.lpips_pair <= p.lpips_alt / 3) & (p.psnr_pair >= 28) & (p.d_comp <= p.d_raw_comp + 0.001)
    if "ring_mad_pair" in p:
        p["pp3"] = (p.ring_mad_pair <= 2 * p.ring_mad_raw + 2) & (p.ring_mad_pair <= p.ring_mad_alt / 4)
    else:
        p["pp3"] = False
    p["pair_pass"] = p.pp1 & p.pp2 & p.pp3
    return p


def cost(v: str) -> dict:
    ev = []
    for f in sorted((ROOT / "infer").glob("*/*/events.jsonl")):
        for line in open(f):
            e = json.loads(line)
            if e.get("kind") == "sample" and f"_{v}_" in e["name"]:
                ev.append(e)
    if not ev:
        return {}
    s = np.array([e["s"] for e in ev])
    return {"variant": v, "clips": len(s), "s_per_clip_median": float(np.median(s)), "s_per_clip_max": float(s.max()),
            "card_peak_gib": float(max(e["card_peak_gib"] for e in ev)),
            "torch_peak_gib": float(max(e["torch_peak_gib"] for e in ev)),
            "gpuh_2000_pairs": float(4000 * np.median(s) / 3600)}


def webp(v: str, pair: str, out, step: int = 2, width: int = 426, quality: int = 70):
    """raw CARLA x+ | Cosmos x+ | Cosmos x-, every `step`-th frame, played at 20 / step fps."""
    import cv2
    from PIL import Image, ImageDraw
    S = streams(pair, v)
    a, b, c = load(S["raw_plus"]), load(S["plus"]), load(S["minus"])
    h = int(round(H * width / W))
    frames = []
    for t in range(0, len(a), step):
        tiles = [cv2.resize(x[t], (width, h), interpolation=cv2.INTER_AREA) for x in (a, b, c)]
        im = Image.fromarray(np.concatenate([np.pad(x, ((18, 0), (0, 2), (0, 0))) for x in tiles], 1))
        d = ImageDraw.Draw(im)
        for i, lab in enumerate(("CARLA x+ (raw)", "Cosmos x+ (with pedestrian)", "Cosmos x- (without)")):
            d.text((i * (width + 2) + 4, 3), lab, fill=(255, 255, 255))
        frames.append(im)
    for q in (quality, 55, 40, 30):
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * step / 20), loop=0,
                       quality=q, method=6)
        if out.stat().st_size <= 2_000_000:
            break
    return out.stat().st_size


def report(variants: list, only: str = ""):
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    names = [p for p in pairs() if not only or p in only.split(",")]
    summ = []
    for v in variants:
        names_v = [p for p in names if (ROOT / "det" / v / f"{p}.npz").exists() and (ROOT / "pix" / v / f"{p}.json").exists()
                   and (ROOT / "op" / v / f"{p}.npz").exists()]
        c1, fr = check1(v, names_v)
        c23 = check2_3(v, names_v)
        per = c1.merge(c23, on="pair")
        per = per_pair_rule(per)                 # also adds NaN floor columns a run without floors lacks
        per.to_csv(RESULTS / f"per_pair_{v}.csv", index=False)
        rec = fr[fr.stream.isin(["raw_plus", "plus"]) & fr.vis]
        R = {s: rec[rec.stream == s].hit.mean() for s in ("raw_plus", "plus")}
        Rn = {s: rec[(rec.stream == s) & (rec.px >= NEAR_PX)].hit.mean() for s in ("raw_plus", "plus")}
        Rf = {s: rec[(rec.stream == s) & (rec.px < NEAR_PX)].hit.mean() for s in ("raw_plus", "plus")}
        hal = fr[fr.stream.isin(["raw_minus", "minus"])]
        Hh = {s: hal[hal.stream == s].halluc.mean() for s in ("raw_minus", "minus")}
        pix = {k: [] for k in ("lpips_pair", "lpips_rep", "lpips_alt", "lpips_raw", "psnr_pair", "psnr_rep", "psnr_alt", "psnr_raw")}
        for p in names_v:
            pj = json.loads((ROOT / "pix" / v / f"{p}.json").read_text())
            for k in pix:
                if k in pj:
                    pix[k] += list(np.minimum(pj[k], 99.0) if k.startswith("psnr") else pj[k])
        med = {k: float(np.median(x)) if x else np.nan for k, x in pix.items()}
        big = per[per.vis_frames >= 10]
        s = {"variant": v, "pairs": len(names_v), "vis_units": int(len(rec) // 2),
             "R_raw": R["raw_plus"], "R_cosmos": R["plus"], "R_raw_near": Rn["raw_plus"], "R_cosmos_near": Rn["plus"],
             "R_raw_far": Rf["raw_plus"], "R_cosmos_far": Rf["plus"],
             "worst_pair_ratio": float((big.R_plus / big.R_raw_plus.clip(lower=1e-6)).min()) if len(big) else np.nan,
             "H_raw": Hh["raw_minus"], "H_cosmos": Hh["minus"], **{f"{k}_med": x for k, x in med.items()},
             "d_comp": float(per.d_comp.median()) if "d_comp" in per else np.nan,
             "d_rep": float(per.d_rep.median()) if "d_rep" in per else np.nan,
             "d_alt": float(per.d_alt.median()) if "d_alt" in per else np.nan,
             "d_raw_comp": float(per.d_raw_comp.median()), "d_full": float(per.d_full.median()),
             "d_raw_full": float(per.d_raw_full.median()),
             "flicker_ratio_med": float(np.median(np.r_[per.flicker_ratio_plus, per.flicker_ratio_minus])),
             "dv_med": float(per.dv_med.median()), "dv_p90": float(per.dv_p90.median()), "dy_med": float(per.dy_med.median()),
             "lead_agree": float(per.lead_agree.mean()), **{k: x for k, x in cost(v).items() if k != "variant"}}
        P = PASS
        # descriptive, not a pass line: pairs whose own median LPIPS breaks the pooled rule
        s["pairs_lpips_over_third_of_seed"] = int((per.lpips_pair > P["lpips_frac_of_seed"] * per.lpips_alt).sum())
        s["pairs_psnr_below_28"] = int((per.psnr_pair < P["psnr_min"]).sum())
        s["pass1"] = bool(s["R_cosmos"] >= P["recall_ratio"] * s["R_raw"] and s["R_cosmos"] >= P["recall_abs"]
                          and not (s["worst_pair_ratio"] < P["recall_pair_ratio"])
                          and s["H_cosmos"] <= P["halluc_abs"] and s["H_cosmos"] <= s["H_raw"] + P["halluc_over_raw"])
        s["pass2"] = bool(s["lpips_pair_med"] <= P["lpips_frac_of_seed"] * s["lpips_alt_med"] and s["psnr_pair_med"] >= P["psnr_min"]
                          and s["d_comp"] <= max(s["d_rep"] if s["d_rep"] == s["d_rep"] else 0.0, P["feat_frac_of_seed"] * s["d_alt"]))
        s["pass3"] = bool(s["flicker_ratio_med"] <= P["flicker_ratio"] and s["dv_med"] <= P["dv_med"] and s["dv_p90"] <= P["dv_p90"]
                          and s["dy_med"] <= P["dy_med"] and s["lead_agree"] >= P["lead_agree"])
        s["pass4"] = bool(s.get("gpuh_2000_pairs", np.inf) <= P["gpuh_2000"])
        s["pairs_pass"] = f"{int(per.pair_pass.sum())}/{len(per)}"
        s["ring_mad_pair_med"] = float(per.ring_mad_pair.median()) if "ring_mad_pair" in per else np.nan
        s["ring_mad_raw_med"] = float(per.ring_mad_raw.median()) if "ring_mad_raw" in per else np.nan
        s["ring_mad_alt_med"] = float(per.ring_mad_alt.median()) if "ring_mad_alt" in per else np.nan
        summ.append(s)
    out = pd.DataFrame(summ)
    out.to_csv(RESULTS / "summary.csv", index=False)
    (RESULTS / "summary.md").write_text(out.T.to_markdown())
    print(out.T.to_string())
    return out
