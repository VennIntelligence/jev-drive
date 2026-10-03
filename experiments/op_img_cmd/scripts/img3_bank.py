"""Image-command Q3 after the course change: trunk banks of the sky-arrow variants (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md (addendum "设计变更"). Writer and layout: img2_bank.write_bank.

  pool       samples                                          variants
  skytrain   Q2 train pool (geom/ft.pkl, lb_imgtrain)         junction: none, sky x every exit class, sky_disc, sky_wrong (if a class
  skyeval    Q1 / Q2 eval set (geom/nav.pkl, lb_navtrain)     is missing); straight: none, sky (straight), sky_disc
  skycarla   CARLA junctions (img2_bank.carla_split != drop)  as junction above
  sgtrain / sgeval / sgcarla   the same samples, the sky + green-line arm: none, sg x every exit class, sg_wrong; straight: none,
             sg (straight). The green line is placed from the original model's own t0 lane lines (`lanes`, below), never the map.

  lanes      the original (port) on the `none` rows of skytrain / skyeval / skycarla -> ft/lanes/<bank>.npz (token, lanes (n, 4, 33, 2))

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

NAV = {"skytrain": "train", "skyeval": "eval", "sgtrain": "train", "sgeval": "eval"}
SKYB = {"sgtrain": "skytrain", "sgeval": "skyeval", "sgcarla": "skycarla"}


def sg_variants(s, pool=None):
    if s["kind"] == "straight":
        return [("none", ""), ("sg", "straight")]
    return [("none", "")] + [("sg", c) for c in FB.classes(s)] + ([("sg_wrong", "")] if O.sky_missing(s) else [])


def variants_for(name):
    return sg_variants if name.startswith("sg") else sky_variants


def with_lanes(name, G, cam):
    """sg pools: attach the original's t0 lane lines (ft/lanes/<sky bank>.npz) and the camera to every sample."""
    if not name.startswith("sg"):
        return G
    z = np.load(FB.FT / "lanes" / f"{SKYB[name]}.npz")
    L = dict(zip(z["token"].astype(str), z["lanes"]))
    return [dict(s, lanes=L[s["token"]], lane_cam=np.asarray(c, float)) for s, c in zip(G, cam)]


def cmd_lanes(net_model, dev):
    import torch
    T_ = None
    sl = net_model.net.slices["lane_lines"]
    for b in ("skytrain", "skyeval", "skycarla"):
        p = FB.FT / "lanes" / f"{b}.npz"
        if p.exists():
            continue
        import img2_train as Q2
        T_, v = Q2.bank(b)
        nn = np.flatnonzero(v["fam"] == "none")
        out = np.zeros((len(nn), 4, 33, 2), np.float32)
        with torch.no_grad():
            for i in range(0, len(nn), 64):
                j = nn[i:i + 64]
                o = net_model(torch.from_numpy(np.stack([T_[x] for x in j])).to(dev), torch.from_numpy(v["slot_valid"][j]).to(dev),
                              torch.from_numpy(np.asarray(v["tc"][j], np.float32)).to(dev).half())["outputs"].float()
                raw = o[:, sl].cpu().numpy()
                out[i:i + len(j)] = raw[:, : raw.shape[1] // 2].reshape(-1, 4, 33, 2)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, token=v["token"][nn], lanes=out)
        print(f"lanes {b}: {len(nn)}", flush=True)


def sky_variants(s, pool=None):
    if s["kind"] == "straight":
        return [("none", ""), ("sky", "straight"), ("sky_disc", "")]
    v = [("none", "")] + [("sky", c) for c in FB.classes(s)] + [("sky_disc", "")]
    return v + ([("sky_wrong", "")] if O.sky_missing(s) else [])


def _nav_init(data, name="skytrain"):
    FB.variants = variants_for(name)
    FB._init(data)


def nav_render(job):
    return FB.render(job)


def carla_render(job):
    s, split = job
    with np.load(s["frames"]) as z:
        fr = z["frames"]
    vs = (sg_variants if "lanes" in s else sky_variants)(s)
    out = np.zeros((len(vs), 10) + fr.shape[1:], np.uint8)
    for v, (fam, c) in enumerate(vs):
        lay = O.primitives(s, fam, c or None)
        out[v, 1:] = [O.draw(f, lay, np.asarray(p, float), np.asarray(s["cam"], float)) for f, p in zip(fr, s["pose"])]
    return vs, out


def build_nav(name, net, dev, workers):
    import h_prep
    pool = NAV[name]
    data = FB.POOLS[pool][1]
    src = h_prep.NavSrc(data)
    G = FB.samples(pool)
    G = with_lanes(name, G, [src.mt["cam"][s["row"]] for s in G])
    var = variants_for(name)

    def meta(job, i):
        s = job[0]
        r = s["row"]
        return {"dom": name, "split": s.get("split", "eval"), "token": s["token"], "log": s["log"], "kind": s["kind"],
                "tc": np.array([0.0, 1.0] if src.mt["lht"][r] else [1.0, 0.0], np.float32), "cam": np.asarray(src.mt["cam"][r], np.float32),
                "v0": float(s["v"])}
    QB.write_bank(name, [(s, pool) for s in G], nav_render, meta, net, dev, workers, sum(len(var(s)) for s in G),
                  init=_nav_init, initargs=(data, name))


def build_carla(net, dev, workers, name="skycarla"):
    G, sp = QB.carla_samples(), QB.carla_split()
    G = [s for s in G if sp[s["token"]] != "drop"]
    G = with_lanes(name, G, [s["cam"] for s in G])
    rows = [(s, sp[s["token"]]) for s in G]

    def meta(job, i):
        s, split = job
        return {"dom": name, "split": split, "token": s["token"], "log": s["log"], "kind": "junction",
                "tc": np.array([1.0, 0.0], np.float32), "cam": np.asarray(s["cam"], np.float32), "v0": float(s["v"])}
    QB.write_bank(name, rows, carla_render, meta, net, dev, workers, sum(len(variants_for(name)(s)) for s, _ in rows))


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
        model = L.load_model(None, dev)
        net = model.net
        for p in todo:
            if p == "lanes":
                cmd_lanes(model, dev)
            elif p.endswith("carla"):
                build_carla(net, dev, workers, p)
            else:
                build_nav(p, net, dev, workers)
            run.summary[p] = "done"


if __name__ == "__main__":
    main()
