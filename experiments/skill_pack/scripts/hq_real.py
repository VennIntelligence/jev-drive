#!/usr/bin/env python
"""Lane C (history quality): the real 10 Hz CAM_F0 history of NAVSIM tokens, read from the public nuPlan v1.1 sensor
archives, as openpilot context frames (plans/2026-10-04-history-quality-prereg.md).

NAVSIM / OpenScene ship CAM_F0 at 2 Hz only; nuPlan recorded it at 10 Hz. The nuPlan test camera archives
(sensor_blobs/test_set/nuplan-v1.1_test_camera_{0..11}.zip) are uncompressed tars on a public S3 bucket with range
support, so the few frames we need are read in place: the log DB (nuplan-v1.1_test.zip, one deflated sqlite per log)
gives each CAM_F0 image's timestamp, a tar header walk gives each image's byte offset. Run dir $DATA_DIR/runs/op_lb/hq/.

  select   seeded navtest subset from a seeded set of logs (+ every navhard stage-1 token of those logs) -> sel.json
  dbs      per selected log: CAM_F0 (filename, timestamp) from its DB -> dbs/<log>.json (DB deleted after)
  locate   per selected log: the tar and byte range of its CAM_F0 directory -> locate.json
  index    tar header walk of each CAM_F0 range -> index/<log>.json (name -> offset, size)
  fetch    the CAM_F0 images at t0 - 0.2 k (k = 1, 2, 3, 4, 6, 7) and the 4 keys' DB match -> jpg/<log>/<name>.jpg, need.json
  render   (envs/openpilot or jevdrive) real context frames in op_lb's layout -> <data>/real.npy for a run dir <data>
           (lb_hq_navtest / lb_hq_navhard1: subsets of lb_navtest / lb_navhard, meta + tokens written here)

Network: S3 through the box's Clash (http://127.0.0.1:7890), set here; ~0.8 s per request, so everything is threaded.
"""
import argparse, io, json, os, random, sqlite3, struct, sys, time, urllib.request, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

S3 = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
DBZIP = f"{S3}/nuplan-v1.1_test.zip"
TARS = [f"{S3}/sensor_blobs/test_set/nuplan-v1.1_test_camera_{k}.zip" for k in range(12)]
CTX_T = np.array([-1.4, -1.2, -0.8, -0.6, -0.4, -0.2])      # op_lb SYN_T order (ascending)
KEY_T = np.array([-1.5, -1.0, -0.5, 0.0])
R = data_dir() / "runs" / "op_lb" / "hq"


def proxy():
    p = os.environ.get("HQ_PROXY", "http://127.0.0.1:7890")
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({"http": p, "https": p})))


def get(url, a, b, tries=6):
    """bytes [a, b) of url."""
    err = "short read"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = r.read()
            if len(d) == b - a:
                return d
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(1 + 2 * k)
    raise RuntimeError(f"range {a}-{b} of {url}: {err}")


def size(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(1 + 2 * k)


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.size = url, 0, size(url)

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        d = get(self.url, self.pos, self.pos + n)
        b[:n] = d
        self.pos += n
        return n


# ---------------------------------------------------------------- tar headers

def _parse(blk):
    if blk[257:262] != b"ustar":
        return None
    chk = int(blk[148:156].split(b"\0")[0].strip() or b"0", 8)
    if chk != sum(blk[:148]) + 256 + sum(blk[156:]):
        return None
    return blk[:100].split(b"\0")[0].decode(), int(blk[124:136].split(b"\0")[0].strip() or b"0", 8)


def header_after(url, off, win=1 << 18):
    """(offset, name, size) of the first tar header at or after off (searched in 512-aligned blocks)."""
    off -= off % 512
    while True:
        d = get(url, off, off + win)
        for i in range(0, len(d) - 511, 512):
            h = _parse(d[i:i + 512])
            if h:
                return off + i, h[0], h[1]
        off += win


def nxt(off, sz):
    return off + 512 + (sz + 511) // 512 * 512


def where(name):
    """tar member name -> (log, cam) ('' when a directory above that level)."""
    p = name.rstrip("/").split("/")
    return (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")


# ---------------------------------------------------------------- steps

def cmd_select(a):
    from jevdrive import navsim_zs as Z
    nt = Z.load_index("navtest", slim=True)
    logs = sorted({e["log_name"] for e in nt})
    rng = random.Random(a.seed)
    rng.shuffle(logs)
    by = {}
    for e in nt:
        by.setdefault(e["log_name"], []).append(e["token"])
    sel, used = [], []
    for lg in logs:
        t = sorted(by[lg])
        rng.shuffle(t)
        sel += t[:a.per_log]
        used.append(lg)
        if len(sel) >= a.n:
            break
    nh = Z.load_index("navhard_two_stage", slim=True)
    s1 = [e["token"] for e in nh if e.get("stage") == "one" and e["log_name"] in set(used)]
    R.mkdir(parents=True, exist_ok=True)
    (R / "sel.json").write_text(json.dumps({"seed": a.seed, "logs": used, "navtest": sorted(sel), "navhard1": sorted(s1)}))
    print(f"{len(used)} logs, {len(sel)} navtest tokens, {len(s1)} navhard stage-1 tokens")


def _one_db(lg, z):
    out = R / "dbs" / f"{lg}.json"
    if out.exists():
        return lg, "cached"
    info = z.getinfo(f"data/cache/test/{lg}.db")
    # local header -> data offset, then one ranged read of the deflated member (the zip object is shared: no seeks)
    lh = get(DBZIP, info.header_offset, info.header_offset + 30)
    n, m = struct.unpack("<HH", lh[26:30])
    a0 = info.header_offset + 30 + n + m
    raw = get(DBZIP, a0, a0 + info.compress_size)
    db = R / "dbs" / f"{lg}.db"
    db.write_bytes(zlib.decompress(raw, -15) if info.compress_type == 8 else raw)
    c = sqlite3.connect(str(db))
    rows = c.execute("select i.filename_jpg, i.timestamp from image i join camera c on i.camera_token = c.token "
                     "where c.channel = 'CAM_F0' order by i.timestamp").fetchall()
    c.close()
    db.unlink()
    out.write_text(json.dumps([[f.split("/")[-1][:-4], t] for f, t in rows]))
    return lg, len(rows)


def cmd_dbs(a):
    sel = json.loads((R / "sel.json").read_text())
    (R / "dbs").mkdir(exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(HTTPFile(DBZIP), 1 << 22))
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for lg, n in ex.map(lambda lg: _one_db(lg, z), sel["logs"]):
            print(f"{time.time() - t0:5.0f} s  {lg}: {n} CAM_F0 images", flush=True)


def cmd_locate(a):
    """Coarse probes every --step bytes of every tar -> which logs each tar holds and where; then for each wanted log a
    bisection of the first and last byte of its CAM_F0 directory between the probes."""
    sel = json.loads((R / "sel.json").read_text())
    want = set(sel["logs"])
    sizes = dict(zip(TARS, ThreadPoolExecutor(12).map(size, TARS)))
    jobs = [(u, o) for u in TARS for o in range(0, sizes[u], a.step)]
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        probes = list(ex.map(lambda j: (j[0],) + header_after(*j), jobs))
    print(f"{len(probes)} probes in {time.time() - t0:.0f} s", flush=True)
    out = {}
    for u in TARS:
        pu = sorted((o, where(nm), nm, s) for uu, o, nm, s in probes if uu == u)
        for lg in want:
            hit = [p for p in pu if p[1] == (lg, "CAM_F0")]
            near = [k for k, p in enumerate(pu) if p[1][0] == lg]
            if not near:
                continue
            out.setdefault(lg, {"tar": u, "hit": [p[0] for p in hit], "lo": pu[max(0, near[0] - 1)][0],
                                "hi": pu[min(len(pu) - 1, near[-1] + 1)][0], "size": sizes[u]})
    miss = want - set(out)
    print(f"located {len(out)} of {len(want)} logs; missing {sorted(miss)[:5]}", flush=True)

    def bounds(lg):
        o = out[lg]
        u = o["tar"]
        inside = lambda off: where(header_after(u, off)[1]) == (lg, "CAM_F0")  # noqa: E731
        if not o["hit"]:                      # no coarse probe landed in CAM_F0: scan the log's span finer
            span = range(o["lo"], o["hi"], max(a.step // 64, 1 << 21))
            hs = list(ThreadPoolExecutor(32).map(lambda off: (off, header_after(u, off)), span))
            o["hit"] = [off for off, h in hs if where(h[1]) == (lg, "CAM_F0")]
            if not o["hit"]:
                return lg, None
        lo, hi = o["lo"], min(o["hit"])       # first CAM_F0 header lies in (lo, hi]
        while hi - lo > (1 << 20):
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if inside(mid) else (mid, hi)
        start = header_after(u, lo)[0]
        while not inside(start):
            _, nm, s = header_after(u, start)
            start = nxt(start, s)
        lo, hi = max(o["hit"]), o["hi"]       # last CAM_F0 byte lies in [lo, hi)
        while hi - lo > (1 << 20):
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if inside(mid) else (lo, mid)
        return lg, {"tar": u, "start": start, "end": hi + (1 << 20)}

    with ThreadPoolExecutor(len(out) or 1) as ex:
        res = dict(ex.map(bounds, sorted(out)))
    (R / "locate.json").write_text(json.dumps(res, indent=1))
    print(f"bounds for {sum(v is not None for v in res.values())} logs in {time.time() - t0:.0f} s")


def cmd_index(a):
    """Header walk of each log's CAM_F0 range; --threads walkers per log, --logs logs at a time."""
    loc = json.loads((R / "locate.json").read_text())
    (R / "index").mkdir(exist_ok=True)
    t0 = time.time()

    def one_log(item):
        lg, o = item
        out = R / "index" / f"{lg}.json"
        if o is None or out.exists():
            return
        u, s0, s1 = o["tar"], o["start"], o["end"]
        seg = np.linspace(s0, s1, a.threads + 1).astype(np.int64)

        def walk(k):
            off = s0 if k == 0 else header_after(u, int(seg[k]))[0]
            res = {}
            while off < seg[k + 1]:
                h = _parse(get(u, off, off + 512))
                if h is None:
                    break
                nm, sz = h
                if where(nm) != (lg, "CAM_F0"):     # past the end of the directory
                    break
                if nm.endswith(".jpg"):
                    res[nm.split("/")[-1][:-4]] = [off + 512, sz]
                off = nxt(off, sz)
            return res

        idx = {}
        with ThreadPoolExecutor(a.threads) as ex:
            for r in ex.map(walk, range(a.threads)):
                idx |= r
        out.write_text(json.dumps({"tar": u, "files": idx}))
        print(f"{time.time() - t0:5.0f} s  {lg}: {len(idx)} CAM_F0 images indexed", flush=True)

    with ThreadPoolExecutor(a.logs) as ex:
        list(ex.map(one_log, loc.items()))


def cmd_fetch(a):
    from jevdrive import navsim_zs as Z
    sel = json.loads((R / "sel.json").read_text())
    want = {"navtest": set(sel["navtest"]), "navhard_two_stage": set(sel["navhard1"])}
    need, jobs, stats = {}, {}, []
    for split, ws in want.items():
        for e in Z.load_index(split, slim=True):
            if e["token"] not in ws:
                continue
            lg = e["log_name"]
            dbf, ixf = R / "dbs" / f"{lg}.json", R / "index" / f"{lg}.json"
            if not (dbf.exists() and ixf.exists()):
                continue
            db = json.loads(dbf.read_text())
            names = [r[0] for r in db]
            ts = np.array([r[1] for r in db], np.int64)
            keyn = [Path(c["CAM_F0"]["path"]).stem for c in e["cams"]]
            if keyn[-1] not in names:
                stats.append(("t0 key not in DB", e["token"]))
                continue
            T0 = ts[names.index(keyn[-1])]
            keyerr = [float((ts[names.index(n)] - T0) / 1e6 - t) if n in names else None for n, t in zip(keyn, KEY_T)]
            pick = [int(np.argmin(np.abs(ts - (T0 + t * 1e6)))) for t in CTX_T]
            ix = json.loads(ixf.read_text())
            if any(names[p] not in ix["files"] for p in pick):
                stats.append(("frame not in tar index", e["token"]))
                continue
            need[f"{split}/{e['token']}"] = {"log": lg, "files": [names[p] for p in pick],
                                             "dt": [float((ts[p] - T0) / 1e6) for p in pick], "key_dt": keyerr}
            for p in pick:
                jobs[(lg, names[p])] = (ix["tar"], *ix["files"][names[p]])
    (R / "need.json").write_text(json.dumps(need))
    print(f"{len(need)} tokens, {len(jobs)} images to fetch; skipped {len(stats)}: {stats[:5]}", flush=True)

    def one(k):
        lg, nm = k
        f = R / "jpg" / lg / f"{nm}.jpg"
        if f.exists():
            return 0
        u, off, sz = jobs[k]
        d = get(u, off, off + sz)
        assert d[:2] == b"\xff\xd8", f"not a jpeg: {k}"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.with_suffix(".tmp").write_bytes(d)
        f.with_suffix(".tmp").replace(f)
        return sz

    t0, tot = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i, n in enumerate(ex.map(one, list(jobs))):
            tot += n
            if i % 200 == 0:
                print(f"{time.time() - t0:5.0f} s  {i}/{len(jobs)}  {tot / 1e6:.0f} MB", flush=True)
    print(f"done: {tot / 1e6:.0f} MB in {time.time() - t0:.0f} s")


def cmd_check_keys(a):
    """The 4 OpenScene keyframes must be byte-identical to the nuPlan archive's copies (same image source)."""
    need = json.loads((R / "need.json").read_text())
    from jevdrive import navsim_zs as Z
    idx = {e["token"]: e for e in Z.load_index("navtest", slim=True)}
    ok = bad = 0
    for k, v in list(need.items())[:a.n]:
        e = idx.get(k.split("/")[1])
        if e is None:
            continue
        ix = json.loads((R / "index" / f"{v['log']}.json").read_text())
        p = e["cams"][-1]["CAM_F0"]["path"]
        nm = Path(p).stem
        if nm not in ix["files"]:
            continue
        off, sz = ix["files"][nm]
        same = get(ix["tar"], off, off + sz) == Path(p).read_bytes()
        ok += same
        bad += not same
    print(f"t0 key byte-identical: {ok}, different: {bad}")


def cmd_render(a):
    from concurrent.futures import ProcessPoolExecutor
    need = json.loads((R / "need.json").read_text())
    src = {"lb_hq_navtest": ("lb_navtest", "navtest"), "lb_hq_navhard1": ("lb_navhard", "navhard_two_stage")}[a.data]
    base = data_dir() / "runs" / "op_lb" / src[0]
    mt = json.loads((base / "meta.json").read_text())
    rows = [i for i, t in enumerate(mt["names"]) if f"{src[1]}/{t}" in need]
    sub = {k: ([v[i] for i in rows] if isinstance(v, list) and len(v) == len(mt["names"]) else v) for k, v in mt.items()}
    sub["parent"], sub["parent_rows"] = src[0], rows
    d = data_dir() / "runs" / "op_lb" / a.data
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(sub))
    (d / "tokens.txt").write_text("\n".join(sub["names"]) + "\n")
    from jevdrive import navsim_zs as Z
    idx = Z.load_index(src[1], slim=True)
    ents = [idx[k] for k in sub["index"]]
    jobs = [(ents[j], [str(R / "jpg" / need[f"{src[1]}/{t}"]["log"] / f"{n}.jpg") for n in need[f"{src[1]}/{t}"]["files"]])
            for j, t in enumerate(sub["names"])]
    out = np.lib.format.open_memmap(d / "real.npy", "w+", np.uint8, (len(rows), len(CTX_T), 2, 6, 128, 256))
    with ProcessPoolExecutor(a.workers) as ex:
        for j, fr in enumerate(ex.map(_render, jobs, chunksize=4)):
            out[j] = fr
    out.flush()
    for arm in a.copy:                       # the parent's synthesized frames for the same rows (paired arms)
        f = base / f"{arm}.npy"
        if f.exists():
            np.save(d / f"{arm}.npy", np.load(f, mmap_mode="r")[rows])
    print(f"{a.data}: {len(rows)} tokens rendered; copied {[x for x in a.copy if (base / f'{x}.npy').exists()]}")


def cmd_hold(a):
    """hold.npy: every context slot shows the latest keyframe at or before it (the exam's sample-and-hold feed)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from jevdrive import op_interp as I
    import op_lb as L
    d = data_dir() / "runs" / "op_lb" / a.data
    keys = L.Keys(a.data)
    out = np.lib.format.open_memmap(d / "hold.npy", "w+", np.uint8, (len(keys), len(CTX_T), 2, 6, 128, 256))
    for i in range(len(keys)):
        out[i] = I.synth_cpu(keys[i], "hold", L.SYN_T)
    out.flush()
    print(f"{a.data}: hold for {len(keys)} tokens")


_M = {}


def _render(job):
    from jevdrive import navsim_zs as Z
    e, paths = job
    cam = e["cams"][-1]["CAM_F0"]
    key = Z.calib_key({"CAM_F0": cam})
    m = _M.get(key) or _M.setdefault(key, Z.OpenpilotMaps(cam))
    return np.stack([m(m.decode(p)) for p in paths])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "dbs", "locate", "index", "fetch", "check-keys", "render", "hold"])
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per-log", type=int, default=20)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--logs", type=int, default=6)
    ap.add_argument("--step", type=int, default=1 << 29)
    ap.add_argument("--data", default="lb_hq_navtest")
    ap.add_argument("--copy", nargs="*", default=["gimm", "warp"])
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.cmd not in ("select", "render", "hold"):
        proxy()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)
