"""WOD-E2E zero-shot exam, project-venv side (todos/2026-09-24-zeroshot-exam/wod-e2e.md).

  sets                   freeze the pre-registered frame sets -> $DATA_DIR/processed/wod_zeroshot/sets.{npz,json}
  fetch                  8-camera records of frames f-3..f from the raw GCS val shards (needs gcloud + Clash)
  packages               Alpamayo input packages (JPEGs + calibration, no protobuf needed to read) + op_calib.json
  views --set check      render the adapter views (Alpamayo's four cameras, openpilot's two model frames) as PNGs
  score                  RFS / ADE with bootstrap CIs from the model runners' prediction files

The model runners live in scripts/wod_zeroshot_alpamayo.py and scripts/wod_zeroshot_openpilot.py (own venvs).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

SETS = ("rater", "extra", "check")


def targets(sets, which):
    return [str(n) for w in which for n in sets[w]["name"]]


def cmd_sets(a, log):
    log.info(f"sets: {Z.build_sets(a.seed)}")


def cmd_fetch(a, log):
    sets, (spans, ordinal) = Z.load_sets(), Z.load_spans()
    want = {}
    for n in targets(sets, a.set):
        for hn in Z.history_names(n, 3):
            want.setdefault(spans[hn][0], {})[ordinal[hn]] = hn
    log.info(f"{sum(map(len, want.values()))} records from {len(want)} raw shards")
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or "http://127.0.0.1:7890"
    r = Z.fetch_records(want, a.route, proxy, a.workers, log=log.info)
    log.event("fetch", **r)
    log.info(f"fetched {r}")


def cmd_packages(a, log):
    from concurrent.futures import ProcessPoolExecutor
    sets = Z.load_sets()
    names = targets(sets, a.set)
    t0 = time.time()
    with ProcessPoolExecutor(min(16, Z.n_cpus())) as ex:
        list(ex.map(Z.write_package, names, chunksize=8))
    p = Z.write_op_calib(targets(sets, SETS))
    log.info(f"{len(names)} packages in {time.time() - t0:.0f} s; {p}")


def alp_views(pkg, scale=0.3):
    """Four Alpamayo views (numpy path, for looking only) of the newest frame in a package."""
    from PIL import Image
    import io
    z = np.load(pkg)
    cal = Z.read_calib(z, Z.ALP_SRC)
    imgs = [np.asarray(Image.open(io.BytesIO(z[f"jpg_3_{c}"].tobytes())).convert("RGB")) for c in Z.ALP_SRC]
    out, cover = [], []
    for v in G.PAI_ORDER:
        rays = G.ftheta_rays(np, G.PAI_RIG[v], scale)
        src, U, V = G.choose_sources(np, rays, {c: cal[c] for c in Z.ALP_SRC})
        out.append(G.render_np(src, U, V, imgs))
        cover.append(float((src >= 0).mean()))
    return out, cover


def op_views(name, spans, op_calib):
    """openpilot road / wide model frames (RGB, for looking) of one frame from the slim shard."""
    from PIL import Image
    import io
    from jevdrive import waymo as W
    sp = spans[name]
    imgs = []
    with open(W.shard_dir() / sp[0], "rb") as f:
        for k in range(3):
            f.seek(sp[1 + 2 * k])
            imgs.append(np.asarray(Image.open(io.BytesIO(f.read(sp[2 + 2 * k]))).convert("RGB")))
    cal = {int(c): d for c, d in op_calib[name.rsplit("-", 1)[0]].items()}
    out = []
    for k in ("road", "wide"):
        rays = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H)
        src, U, V = G.choose_sources(np, rays, {c: cal[c] for c in Z.OP_SRC})
        out.append(G.render_np(src, U, V, imgs, nearest=True))
    return out


def cmd_views(a, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets, (spans, _) = Z.load_sets(), Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    covers = []
    for n in targets(sets, a.set)[: a.limit]:
        views, cover = alp_views(Z.root("packages") / f"{n}.npz")
        covers.append(cover)
        road, wide = op_views(n, spans, op_calib)
        fig, ax = plt.subplots(2, 3, figsize=(15, 5.5))
        for i, (img, t) in enumerate(zip(views + [road, wide],
                                         [f"{v} cover {c:.3f}" for v, c in zip(G.PAI_ORDER, cover)] + ["op road", "op wide"])):
            ax.flat[i].imshow(img)
            ax.flat[i].set_title(t, fontsize=8)
            ax.flat[i].axis("off")
        fig.suptitle(n, fontsize=8)
        fig.tight_layout()
        fig.savefig(log.dir / f"views_{n}.png", dpi=110)
        plt.close(fig)
        log.info(f"{n}: cover {np.round(cover, 3)}")
    c = np.array(covers)
    log.event("coverage", views=list(G.PAI_ORDER), mean=c.mean(0).tolist(), min=c.min(0).tolist())
    log.info(f"mean coverage {dict(zip(G.PAI_ORDER, np.round(c.mean(0), 4)))}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--seed", type=int, default=0)
    for name in ("fetch", "packages", "views"):
        p = sub.add_parser(name)
        p.add_argument("--set", nargs="+", default=list(SETS), choices=SETS)
        if name == "fetch":
            p.add_argument("--route", default="proxy", choices=("proxy", "direct"))
            p.add_argument("--workers", type=int, default=48)
        if name == "views":
            p.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    log = RunLog("wod_zeroshot", a.cmd)
    log.event("start", args=vars(a))
    globals()[f"cmd_{a.cmd}"](a, log)
    log.event("end")


if __name__ == "__main__":
    main()
