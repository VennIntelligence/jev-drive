"""Re-encode the step-1 probe rows and read sets with an arm's trunk (plan 4.5 read protocol): the t0 pair of every row through the
checkpoint's vision weights -> view_39 tokens, for executor 1's HEAD1 probes on P and the reads on navtest-16 / held-452 / zero-98.

  python je_encode.py --tag JE-L-F0-s0 [--which best] --sets sets.json [--only P navtest held zero98]
  python je_encode.py --tag factory --sets sets.json          the frozen trunk (Z; reproduces the frozen caches: a check)

sets.json: {"<set>": {"px": npy (N, [V,] 2, 2, 6, 128, 256), "view": null, "names": npy or npz with "names" (optional)}}; the J_on /
held-452 / zero-98 entries are executor 2's slot-pair extracts ((slot 6, slot 7) pairs, the driver's pairing), J_nav / navtest the
protocol-W t0 pairs (results/pixels.md). Output: $DATA_DIR/runs/junction_eyes/step1/tokens/<tag>/<set>.npy (N, 32, 512) fp16 in
the pixel file's row order (+ <set>.names.npy when names are given), written tmp + rename, skipped when present (--force redoes).
Only the t0 slot is re-encoded: the J_nav / navtest pixel caches hold the t0 pair only (results/trainer.md, "Read protocol").
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "experiments/junction_eyes/lib")]
import argparse, json, os, time  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import je_data as D  # noqa: E402
import je_model as M  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

STEP1 = data_dir() / "runs" / "junction_eyes" / "step1"


def main(a):
    from jevdrive.run import Run
    dev = torch.device("cuda")
    sets = json.loads(_pl.Path(a.sets).read_text())
    names = a.only or list(sets)
    with Run("junction_eyes", f"step1/encode-{a.tag}", config=vars(a)) as run:
        if a.tag == "factory":
            m = M.JE("none", heads=False).to(dev).eval()
        else:
            m, ck = M.load_ckpt(STEP1 / "ckpt" / a.tag / f"{a.which}.pt", dev, heads=False)
            run.info(f"{a.tag}: {a.which} checkpoint of step {ck['step']}")
        od = STEP1 / "tokens" / a.tag
        od.mkdir(parents=True, exist_ok=True)
        for s in names:
            spec, out = sets[s], od / f"{s}.npy"
            if out.exists() and not a.force:
                run.info(f"{s}: {out} exists, skipped")
                continue
            px = D.PairStore(spec["px"], spec.get("view"))
            n, t0 = len(px), time.time()
            tmp = od / f".{s}.{os.getpid()}.npy"
            o = np.lib.format.open_memmap(tmp, "w+", np.float16, (n, 32, 512))
            pre = D.Prefetch(iter(range(0, n, a.batch)).__next__, lambda i: (i, px.get(np.arange(i, min(i + a.batch, n)))),
                             (n + a.batch - 1) // a.batch, 4, 3)
            for _ in run.tqdm(range((n + a.batch - 1) // a.batch), desc=s):
                i, (p, c) = pre.get()
                with torch.no_grad():
                    o[i:i + len(p)] = m.encode(p.to(dev, non_blocking=True), c.to(dev, non_blocking=True)).half().cpu().numpy()
            o.flush()
            del o
            os.replace(tmp, out)
            if spec.get("names"):
                z = np.load(spec["names"], allow_pickle=False)
                nm = z["names"] if isinstance(z, np.lib.npyio.NpzFile) else z
                assert len(nm) == n, f"{s}: {len(nm)} names for {n} rows"
                np.save(od / f"{s}.names.npy", nm)
            run.info(f"{s}: {n} rows -> {out} ({n / (time.time() - t0):.0f} pairs/s)")
        run.summary.update(out=str(od), sets=names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="a step-1 checkpoint tag, or factory")
    ap.add_argument("--which", default="best", choices=("best", "last"))
    ap.add_argument("--sets", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
