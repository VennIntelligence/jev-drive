"""Image-command Q3 after the course change: trunk banks of the sky-arrow variants (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md (addendum "设计变更"). Writer and layout: img2_bank.write_bank.

  pool       samples                                          variants
  skytrain   Q2 train pool (geom/ft.pkl, lb_imgtrain)         junction: none, sky x every exit class, sky_disc, sky_wrong (if a class
  skyeval    Q1 / Q2 eval set (geom/nav.pkl, lb_navtrain)     is missing); straight: none, sky (straight), sky_disc
  skycarla   CARLA junctions (img2_bank.carla_split != drop)  as junction above

  CUDA_VISIBLE_DEVICES=1 taskset -c 100-149 $DATA_DIR/envs/op-train/bin/python experiments/op_img_cmd/scripts/img3_bank.py skytrain skyeval skycarla
"""
import os, sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_adapt_h" / "scripts"), str(Path(__file__).resolve().parent)]
import img2_bank as QB  # noqa: E402
import img_ft_bank as FB  # noqa: E402
import img_overlay as O  # noqa: E402

NAV = {"skytrain": "train", "skyeval": "eval"}


def sky_variants(s, pool=None):
    if s["kind"] == "straight":
        return [("none", ""), ("sky", "straight"), ("sky_disc", "")]
    v = [("none", "")] + [("sky", c) for c in FB.classes(s)] + [("sky_disc", "")]
    return v + ([("sky_wrong", "")] if O.sky_missing(s) else [])


def _nav_init(data):
    FB.variants = sky_variants
    FB._init(data)


def nav_render(job):
    return FB.render(job)


def carla_render(job):
    s, split = job
    with np.load(s["frames"]) as z:
        fr = z["frames"]
    vs = sky_variants(s)
    out = np.zeros((len(vs), 10) + fr.shape[1:], np.uint8)
    for v, (fam, c) in enumerate(vs):
        lay = O.primitives(s, fam, c or None)
        out[v, 1:] = [O.draw(f, lay, np.asarray(p, float), np.asarray(s["cam"], float)) for f, p in zip(fr, s["pose"])]
    return vs, out


def build_nav(name, net, dev, workers):
    import h_prep
    pool = NAV[name]
    data = FB.POOLS[pool][1]
    G = FB.samples(pool)
    src = h_prep.NavSrc(data)

    def meta(job, i):
        s = job[0]
        r = s["row"]
        return {"dom": name, "split": s.get("split", "eval"), "token": s["token"], "log": s["log"], "kind": s["kind"],
                "tc": np.array([0.0, 1.0] if src.mt["lht"][r] else [1.0, 0.0], np.float32), "cam": np.asarray(src.mt["cam"][r], np.float32),
                "v0": float(s["v"])}
    QB.write_bank(name, [(s, pool) for s in G], nav_render, meta, net, dev, workers, sum(len(sky_variants(s)) for s in G),
                  init=_nav_init, initargs=(data,))


def build_carla(net, dev, workers):
    G, sp = QB.carla_samples(), QB.carla_split()
    rows = [(s, sp[s["token"]]) for s in G if sp[s["token"]] != "drop"]

    def meta(job, i):
        s, split = job
        return {"dom": "skycarla", "split": split, "token": s["token"], "log": s["log"], "kind": "junction",
                "tc": np.array([1.0, 0.0], np.float32), "cam": np.asarray(s["cam"], np.float32), "v0": float(s["v"])}
    QB.write_bank("skycarla", rows, carla_render, meta, net, dev, workers, sum(len(sky_variants(s)) for s, _ in rows))


def main():
    import torch
    from jevdrive.data import splits
    from jevdrive.run import Run
    from experiments.op_adapt_l.lib import op_adapt_l as L
    todo = sys.argv[1:] or ["skytrain", "skyeval", "skycarla"]
    workers = int(os.environ.get("IMG_FT_WORKERS", 40))
    with Run("op_img_cmd", "sky-bank", config={"pools": todo, "workers": workers}) as run:
        for nm in ("navsim/img-ft-train", "navsim/img-ft-dev", "navsim/navtrain", "b2d/img-carla-train", "b2d/img-carla-dev", "b2d/img-carla-test"):
            run.use_split(splits.load(nm))
        dev = torch.device("cuda")
        net = L.load_model(None, dev).net
        for p in todo:
            build_carla(net, dev, workers) if p == "skycarla" else build_nav(p, net, dev, workers)
            run.summary[p] = "done"


if __name__ == "__main__":
    main()
