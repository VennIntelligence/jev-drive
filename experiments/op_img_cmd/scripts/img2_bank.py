"""Image-command fine-tune Q3 (drift-free): the extra trunk banks (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md. Same layout as img_ft_bank.py (stage-3 outputs of the port, fp16), but with
per-row slot validity / traffic convention / camera, so banks of different domains mix in one batch.

  pool    rows                                                                           variants
  dist    op_adapt_H/samples/{nav,wod,carla} (L3's pools; nav rows whose log holds an     none
          op_img_cmd eval token are dropped), train + dev splits
  carla   runs/op_img_cmd/carla/carla.pkl (179 B2D junctions, re-recorded), split by     train / dev: none, band and barrier x every
          route (carla_split(): test = closed-loop + decision-95 routes + random to 60,   exit class, band_all; test: img_run.variants
          dev 18, train the rest minus samples sharing a junction node with test)        (every family)
  nba     Q2's train pool (geom/ft.pkl, lb_imgtrain), junction frames                    band_all (negatives)
Output $DATA_DIR/runs/op_img_cmd/ft/bank/<pool>/{trunk.npy (n, 9, 1024, 8, 16) fp16, var.npz}.

  CUDA_VISIBLE_DEVICES=1 taskset -c 100-149 $DATA_DIR/envs/op-train/bin/python experiments/op_img_cmd/scripts/img2_bank.py split dist carla nba
"""
import json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("JEV_REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments" / "op_adapt_h" / "scripts"), str(Path(__file__).resolve().parent)]
import img_ft_bank as FB  # noqa: E402
import img_overlay as O  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "runs" / "op_img_cmd"
FT = ROOT / "ft"
HS = data_dir() / "runs" / "op_adapt_H" / "samples"
SMOKE_ROUTES = ("27297", "27043", "9196", "24944")
D95_ROUTES = ("27043", "15102", "24944", "27870", "22535", "37969", "24497", "27297", "9196", "28147",
              "16390", "15612", "15483", "17280", "16529", "16508", "19324", "2520", "19832")
N_TEST, N_DEV = 60, 18


# ---------------------------------------------------------------- CARLA split
def carla_samples():
    return [s for s in pickle.load(open(ROOT / "carla" / "carla.pkl", "rb")) if O.valid(s)]


def carla_split(write=False):
    """{token: 'train' | 'dev' | 'test' | 'drop'} by route; written once to carla/split.json (+ jevdrive splits)."""
    p = ROOT / "carla" / "split.json"
    if p.exists():
        return json.loads(p.read_text())
    G = carla_samples()
    routes = sorted({s["log"] for s in G})
    rng = np.random.default_rng(20261004)
    test = [r for r in D95_ROUTES if r in routes]
    rest = [r for r in rng.permutation(routes) if r not in test]
    test += rest[: N_TEST - len(test)]
    rest = [r for r in rest if r not in test]
    dev, train = rest[:N_DEV], rest[N_DEV:]
    jt = {(s["map"], s["node"]) for s in G if s["log"] in test}
    out = {}
    for s in G:
        r = s["log"]
        out[s["token"]] = "test" if r in test else "dev" if r in dev else ("drop" if (s["map"], s["node"]) in jt else "train")
    if write:
        from jevdrive.data import splits
        origin = ("op_img_cmd CARLA junction set (179 bench2drive220 routes re-recorded with the P4 rig, img_carla_*.py); test = the "
                  "decision-95 routes present + random routes to 60 (seed 20261004), dev 18 random routes, train = the rest")
        for nm, mem in (("img-carla-train", train), ("img-carla-dev", dev), ("img-carla-test", test)):
            splits.define("b2d", nm, sorted(mem), unit="route", origin=origin, status="frozen", used_by=["op_img_cmd"],
                          notes="op_img_cmd Q3 fine-tune split; train samples sharing a junction node with test are dropped")
        p.write_text(json.dumps(out, indent=0))
        from collections import Counter
        print("carla split:", Counter(out.values()), "routes train / dev / test", len(train), len(dev), len(test))
    return out


def carla_variants(s, split):
    import img_run
    if split == "test":
        return img_run.variants(s)
    cls = FB.classes(s)
    return [("none", "")] + [(f, c) for f in FB.TRAIN_FAMS for c in cls] + [("band_all", "")]


def carla_render(job):
    s, split = job
    with np.load(s["frames"]) as z:
        fr = z["frames"]
    assert len(fr) == 9, s["token"]
    vs = carla_variants(s, split)
    out = np.zeros((len(vs), 10) + fr.shape[1:], np.uint8)
    cam = np.asarray(s["cam"], float)
    for v, (fam, c) in enumerate(vs):
        lay = O.primitives(s, fam, c or None)
        out[v, 1:] = [O.draw(f, lay, np.asarray(p, float), cam) for f, p in zip(fr, s["pose"])]
    return vs, out


# ---------------------------------------------------------------- generic writer
def write_bank(name, rows, render, meta, net, dev, workers, est, init=None, initargs=(), batch=32):
    """rows: list of jobs; render(job) -> (variants [(fam, cmd)], imgs (V, 10, 2, 6, 128, 256)); meta(job, k) -> dict of per-sample
    fields (repeated over its variants). Slot validity per row from the zero images (slot k = pair (k, k+1), both nonzero)."""
    import torch
    from drive_backbones_openpilot import bounded_map
    from experiments.op_adapt_h.lib import op_adapt_h as H
    out = FT / "bank" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "var.npz").exists():
        print(name, "bank exists", flush=True)
        return
    t0 = time.time()
    V = {}
    nvar = 0
    tmp = out / "trunk.tmp.npy"
    T = np.lib.format.open_memmap(tmp, "w+", np.float16, (est, 9, 1024, 8, 16))
    buf = []

    def flush():
        nonlocal buf, nvar
        if not buf:
            return
        imgs = np.concatenate(buf)
        for k in range(0, len(imgs), batch):
            x = torch.from_numpy(imgs[k:k + batch]).to(dev)
            sv = (x[:, :-1].flatten(2).amax(2) > 0) & (x[:, 1:].flatten(2).amax(2) > 0)
            T[nvar + k:nvar + k + len(x)] = (H.trunks(net, x, chunk=64) * sv[:, :, None, None, None]).cpu().numpy()
            V.setdefault("slot_valid", []).append(sv.cpu().numpy())
        nvar += len(imgs)
        buf = []

    ex = ProcessPoolExecutor(workers, initializer=init, initargs=initargs) if workers else None
    it = bounded_map(ex, render, rows, 2 * workers) if ex else map(render, rows)
    for i, (vs, imgs) in enumerate(it):
        m = meta(rows[i], i)
        for fam, c in vs:
            for k, x in m.items():
                V.setdefault(k, []).append(x)
            V.setdefault("fam", []).append(fam)
            V.setdefault("cmd", []).append(c)
            V.setdefault("sample", []).append(i)
        buf.append(imgs)
        if sum(len(b) for b in buf) >= 4 * batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"{name}: {i + 1}/{len(rows)} samples, {nvar / (time.time() - t0):.1f} var/s", flush=True)
    flush()
    if ex:
        ex.shutdown()
    assert nvar == est, (nvar, est)
    T.flush()
    del T
    tmp.replace(out / "trunk.npy")
    V["slot_valid"] = np.concatenate(V["slot_valid"])
    np.savez(out / "var.npz", **{k: np.asarray(v) for k, v in V.items()})
    print(f"{name}: {len(rows)} samples, {nvar} variants in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- pools
_D = {}


def _dist_init():
    _D["imgs"] = {d: np.load(HS / d / "imgs.npy", mmap_mode="r") for d in ("nav", "wod", "carla")}


def dist_render(job):
    d, k = job
    return [("none", "")], np.asarray(_D["imgs"][d][k])[None]


def build_dist(net, dev, workers):
    ev_logs = {s["log"] for s in pickle.load(open(ROOT / "geom" / "nav.pkl", "rb"))}
    rows, tabs = [], {}
    for d in ("nav", "wod", "carla"):
        with np.load(HS / d / "tab.npz") as z:
            tabs[d] = {k: z[k] for k in z.files}
        t = tabs[d]
        keep = np.isin(t["split"], ["train", "dev"])
        if d == "nav":
            keep &= ~np.isin(t["cluster"], list(ev_logs))
        rows += [(d, int(k)) for k in np.flatnonzero(keep)]
        print(d, "rows", int(keep.sum()), "dropped (eval logs)" if d == "nav" else "", int((np.isin(t["split"], ["train", "dev"]) & ~keep).sum()))

    def meta(job, i):
        d, k = job
        t = tabs[d]
        return {"dom": d, "split": str(t["split"][k]), "id": str(t["id"][k]), "cluster": str(t["cluster"][k]), "tc": t["tc"][k],
                "cam": t["cam"][k], "v0": float(t["v0"][k]), "token": f"{d}:{t['id'][k]}"}
    write_bank("dist", rows, dist_render, meta, net, dev, workers, len(rows), init=_dist_init)


def build_carla(net, dev, workers):
    G, sp = carla_samples(), carla_split()
    rows = [(s, sp[s["token"]]) for s in G if sp[s["token"]] != "drop"]

    def meta(job, i):
        s, split = job
        return {"dom": "cjunc", "split": split, "token": s["token"], "log": s["log"], "tc": np.array([1.0, 0.0], np.float32),
                "cam": np.asarray(s["cam"], np.float32), "v0": float(s["v"])}
    write_bank("carla", rows, carla_render, meta, net, dev, workers, sum(len(carla_variants(*r)) for r in rows))


def nba_variants(s, pool):
    return [("band_all", "")]


def nba_render(s):
    return FB.render((s, "train"))


def _nba_init(data):
    FB.variants = nba_variants
    FB._init(data)


def build_nba(net, dev, workers):
    G = [s for s in FB.samples("train") if s["kind"] == "junction"]
    import h_prep
    src = h_prep.NavSrc("lb_imgtrain")

    def meta(s, i):
        r = s["row"]
        return {"dom": "nba", "split": s["split"], "token": s["token"], "log": s["log"],
                "tc": np.array([0.0, 1.0] if src.mt["lht"][r] else [1.0, 0.0], np.float32), "cam": np.asarray(src.mt["cam"][r], np.float32),
                "v0": float(s["v"])}
    write_bank("nba", G, nba_render, meta, net, dev, workers, len(G), init=_nba_init, initargs=("lb_imgtrain",))


def main():
    import torch
    from jevdrive.run import Run
    from experiments.op_adapt_l.lib import op_adapt_l as L
    todo = sys.argv[1:] or ["split", "dist", "carla", "nba"]
    workers = int(os.environ.get("IMG_FT_WORKERS", 40))
    if "split" in todo:
        carla_split(write=True)
    with Run("op_img_cmd", "ft2-bank", config={"pools": todo, "workers": workers}) as run:
        from jevdrive.data import splits
        for nm in ("navsim/op-adapt-h-nav-train", "navsim/op-adapt-h-nav-dev", "b2d/op-adapt-h-carla-train", "b2d/op-adapt-h-carla-dev",
                   "navsim/img-ft-train", "navsim/img-ft-dev", "b2d/img-carla-train", "b2d/img-carla-dev", "b2d/img-carla-test"):
            run.use_split(splits.load(nm))
        dev = torch.device("cuda")
        net = L.load_model(None, dev).net
        for p in ("dist", "carla", "nba"):
            if p in todo:
                globals()[f"build_{p}"](net, dev, workers)
                run.summary[p] = "done"


if __name__ == "__main__":
    main()
