"""WODSCOUT Q4a: which fields of E2EDFrame are populated, on every frame of every WOD-E2E shard (plans/2026-10-10-wodscout-prereg.md).

  python experiments/wod_scout/scripts/ws_fields.py [--workers 32] [--limit-shards N]        (jevdrive env, CPU, box)

Schema-free walk of the protobuf wire format with pread: tags and lengths are read, large payloads (JPEGs) are skipped, so the 0.73 TB of slim shards
is not read. Per record the set of populated field paths is collected to depth 3 (E2EDFrame -> Frame -> Context / CameraImage; the ego state messages);
names come from the compiled descriptors where they resolve, else the field number is kept. The slim shards differ from the bucket only in the dropped
five cameras (docs/waymo-e2e.md), so every other path is as released.

Tables -> experiments/wod_scout/results/q4_label_supply/fields{,_by_split}.{csv,md}, fields_meta.json.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json, os, struct  # noqa: E401,E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

OUT = _R / "experiments/wod_scout/results/q4_label_supply"
DESCEND = {(1,), (1, 1), (1, 4), (5,), (6,), (8,)}                                # Frame, Context, CameraImage, future / past / preference states
READ = {(1, 4, 3), (1, 3)}                                                        # poses: read the payload to see whether it is non-zero


def varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if c < 0x80:
            return x, i
        s += 7


def walk(fd, pos, end, path, out):
    """Collect field paths of the message in [pos, end) into out: path -> [count, bytes, nonzero]."""
    while pos < end:
        b = os.pread(fd, min(20, end - pos), pos)
        tag, i = varint(b, 0)
        num, wt = tag >> 3, tag & 7
        p = path + (num,)
        if wt == 0:
            v, i = varint(b, i)
            size, nz = 0, v != 0
        elif wt == 1:
            size, nz = 8, any(b[i:i + 8])
        elif wt == 5:
            size, nz = 4, any(b[i:i + 4])
        elif wt == 2:
            size, i = varint(b, i)
            nz = size > 0
            if p in DESCEND and size:
                walk(fd, pos + i, pos + i + size, p, out)
            elif p in READ and size:
                raw = os.pread(fd, size, pos + i)                               # Transform { repeated double transform = 1 [packed] }
                t = np.frombuffer(raw[-128:], "<f8") if size >= 130 and raw[0] == 0x0A else np.zeros(16)
                nz = bool(np.abs(t).max() > 0 and not np.allclose(t, np.eye(4).ravel()))    # "non-zero" = neither all zeros nor the identity
        else:
            raise ValueError(f"wire type {wt} at {pos}")
        e = out.setdefault(p, [0, 0, 0])
        e[0] += 1
        e[1] += size
        e[2] += bool(nz)
        pos += i + size
    return out


def scan(path):
    """One shard -> Counter of (path) -> frames with the path, plus byte and non-zero tallies."""
    from jevdrive import waymo as W
    frames, has, nbytes, nz, reps = 0, Counter(), Counter(), Counter(), Counter()
    fd = os.open(path, os.O_RDONLY)
    try:
        pos, size = 0, os.fstat(fd).st_size
        while pos < size:
            (n,) = struct.unpack("<Q", os.pread(fd, 8, pos))
            out = walk(fd, pos + 12, pos + 12 + n, (), {})
            for p, (c, b, z) in out.items():
                has[p] += 1
                nbytes[p] += b
                nz[p] += z > 0
                reps[p] += c
            frames += 1
            pos += 12 + n + 4
    finally:
        os.close(fd)
    return dict(shard=os.path.basename(path), split=W.split_of(os.path.basename(path)), frames=frames, has=dict(has), bytes=dict(nbytes), nz=dict(nz), reps=dict(reps))


def names():
    """Field path -> dotted name from the compiled descriptors (number kept where a level does not resolve)."""
    from jevdrive import waymo as W
    root = W.e2ed_frame().DESCRIPTOR

    def name(p):
        d, parts = root, []
        for k in p:
            f = d.fields_by_number.get(k) if d is not None else None
            parts.append(f.name if f is not None else f"#{k}")
            d = f.message_type if f is not None else None
        return ".".join(parts)
    return name


def main(a):
    import pandas as pd
    from jevdrive import par, stats
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("wod_scout", "fields", seed=0, config=vars(a)) as run:
        for s in ("wod/train", "wod/val", "wod/test"):
            run.use_split(splits.load(s))
        shards = sorted(str(p) for p in W.shard_dir().glob("*.tfrecord-*"))
        if a.limit_shards:
            shards = shards[::max(1, len(shards) // a.limit_shards)][:a.limit_shards]
        res = par.pmap(scan, shards, run=run, workers=a.workers)
        res.raise_if_failed()
        nm = names()
        tot = {s: 0 for s in ("train", "val", "test")}
        agg = {s: {k: Counter() for k in ("has", "bytes", "nz", "reps")} for s in tot}
        for r in res.values:
            tot[r["split"]] += r["frames"]
            for k in agg[r["split"]]:
                agg[r["split"]][k].update(r[k])
        paths = sorted({p for s in agg.values() for p in s["has"]})
        rows = []
        for p in paths:
            row = {"path": ".".join(map(str, p)), "field": nm(p)}
            for s in tot:
                h = agg[s]["has"].get(p, 0)
                row |= {f"{s} frames with field": h, f"{s} share": h / max(tot[s], 1), f"{s} non-zero": agg[s]["nz"].get(p, 0),
                        f"{s} mean count per frame": agg[s]["reps"].get(p, 0) / max(h, 1), f"{s} mean bytes": agg[s]["bytes"].get(p, 0) / max(h, 1)}
            rows.append(row)
        stats.write_table(rows, OUT / "fields")
        idx = W.load_index()
        sup = []
        for s in tot:
            d = idx[idx.split == s]
            sup.append({"split": s, "frames scanned": tot[s], "frames in index": len(d), "sequences": d.sequence.nunique(), "frames with a full logged future": int(d.has_future.sum()),
                        "sequences with a full logged future": int(d[d.has_future].sequence.nunique()), "frames with rated trajectories (score >= 0)": int((d.n_pref > 0).sum()),
                        "sequences with rated trajectories": int(d[d.n_pref > 0].sequence.nunique()), "intent straight / left / right / unknown":
                        " / ".join(str(int((d.intent == k).sum())) for k in (1, 2, 3, 0))})
        stats.write_table(sup, OUT / "fields_by_split")
        run.info("fields:\n%s", pd.DataFrame(rows).to_string())
        run.info("by split:\n%s", pd.DataFrame(sup).T.to_string())
        (OUT / "fields_meta.json").write_text(json.dumps(dict(shards=len(shards), frames=tot, limit_shards=a.limit_shards), indent=1))
        run.summary.update(frames=tot, shards=len(shards))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-shards", type=int, default=0)
    main(ap.parse_args())
