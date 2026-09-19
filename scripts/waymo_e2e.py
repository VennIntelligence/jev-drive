#!/usr/bin/env python3
"""WOD-E2E v1.0.0 stream-and-slim download, and a quick shard inspector (docs/waymo-e2e.md).

Runs in the throwaway venv $DATA_DIR/envs/waymo (protobuf/upb, google-crc32c, tqdm; protos compiled into
$WAYMO_PROTO_GEN), set up by scripts/download_waymo_e2e.sh. No TensorFlow.

  download [split ...]   splits: small val train test (default: all, in that order)
  inspect <tfrecord>     frame count, cameras, resolution, frame rate, trajectory fields, intent

Download pipeline:
  bucket listing -> download pool (range GETs on many streams, pwrite into a sparse .part file)
  -> raw shards on disk (bounded: downloads pause while raw shards > 2x slim pool)
  -> slim pool (one process per shard: md5 + TFRecord CRC check, drop every camera image except
     FRONT/FRONT_LEFT/FRONT_RIGHT, keep all other fields and the original JPEG bytes)
  -> re-read and verify the slim shard -> delete the raw shard.
Resumable: shards in manifest.csv whose slim file exists with the recorded size are skipped.
Disk guard: no new shard starts if free space on the data disk would drop below --min-free-gb.
"""
import argparse
import base64
import collections
import csv
import hashlib
import http.client
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import google_crc32c
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.environ.get("WAYMO_PROTO_GEN", ""))
from jevdrive.common import n_cpus  # noqa: E402  (stdlib-only module)

BUCKET = "waymo_open_dataset_end_to_end_camera_v_1_0_0"
HOST = "storage.googleapis.com"
SPLITS = ("small", "val", "train", "test")
FRONT3 = frozenset((1, 2, 3))  # CameraName.Name: FRONT, FRONT_LEFT, FRONT_RIGHT
CAM_NAMES = {0: "UNKNOWN", 1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT",
             6: "REAR_LEFT", 7: "REAR", 8: "REAR_RIGHT"}
MANIFEST_COLS = ("name", "split", "raw_bytes", "slim_bytes", "ratio", "frames", "dl_s", "slim_s", "done_at")
GB = 1e9
log = logging.getLogger("waymo_e2e")


def split_of(name: str) -> str:
    if ".tfrecord-" not in name:
        return "small"
    return {"val_": "val", "trai": "train", "test": "test"}[name[:4]]


# ---------------------------------------------------------------- TFRecord framing (C crc32c, no per-byte Python)

def _masked_crc(data: bytes) -> int:
    c = google_crc32c.value(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_records(f, hasher=None):
    """Yield record payloads from a TFRecord file object, checking both CRCs. Optionally hash all bytes read."""
    while head := f.read(12):
        if len(head) < 12:
            raise ValueError("truncated record header")
        n, hcrc = struct.unpack("<QI", head)
        if _masked_crc(head[:8]) != hcrc:
            raise ValueError("bad length crc")
        data, tail = f.read(n), f.read(4)
        if len(data) < n or len(tail) < 4:
            raise ValueError("truncated record")
        if _masked_crc(data) != struct.unpack("<I", tail)[0]:
            raise ValueError("bad data crc")
        if hasher:
            hasher.update(head), hasher.update(data), hasher.update(tail)
        yield data


def write_record(f, data: bytes):
    head = struct.pack("<Q", len(data))
    f.write(head + struct.pack("<I", _masked_crc(head)))
    f.write(data)
    f.write(struct.pack("<I", _masked_crc(data)))


def e2ed_frame():
    from google.protobuf.internal import api_implementation
    assert api_implementation.Type() == "upb", f"slow protobuf backend {api_implementation.Type()}"
    from waymo_open_dataset.protos.end_to_end_driving_data_pb2 import E2EDFrame
    return E2EDFrame


# ---------------------------------------------------------------- slim worker (one process per shard)

def slim_shard(raw: str, out: str, md5_b64: str) -> dict:
    """raw TFRecord -> slim TFRecord with only FRONT3 images. Deletes raw once the slim copy is verified."""
    E2EDFrame, raw, out = e2ed_frame(), Path(raw), Path(out)
    tmp = out.with_name(out.name + ".tmp")
    t0, md5, kept = time.time(), hashlib.md5(), []
    with open(raw, "rb", buffering=16 << 20) as fi, open(tmp, "wb", buffering=16 << 20) as fo:
        for rec in read_records(fi, md5):
            fr = E2EDFrame.FromString(rec)
            imgs = fr.frame.images
            for i in range(len(imgs) - 1, -1, -1):
                if imgs[i].name not in FRONT3:
                    del imgs[i]
            kept.append(tuple(im.name for im in imgs))
            write_record(fo, fr.SerializeToString())
    if base64.b64encode(md5.digest()).decode() != md5_b64:
        tmp.unlink(), raw.unlink()
        return {"ok": False, "why": "md5 mismatch"}
    with open(tmp, "rb", buffering=16 << 20) as f:  # verify: parses, same frames, same kept cameras
        got = [tuple(im.name for im in E2EDFrame.FromString(r).frame.images) for r in read_records(f)]
    if got != kept:
        tmp.unlink()
        return {"ok": False, "why": f"verify failed: {len(got)} vs {len(kept)} frames"}
    raw_bytes, slim_bytes = raw.stat().st_size, tmp.stat().st_size
    tmp.rename(out)
    raw.unlink()
    return {"ok": True, "frames": len(kept), "raw_bytes": raw_bytes, "slim_bytes": slim_bytes,
            "slim_s": round(time.time() - t0, 1), "front_cams": sorted({c for k in kept for c in k})}


# ---------------------------------------------------------------- GCS access (bearer token from gcloud)

class Gcs:
    """Range GETs on persistent per-thread connections. The token is always refreshed through `proxy`
    (Google OAuth is unreachable directly from the box); data goes direct unless route == 'proxy'."""

    def __init__(self, route: str, proxy: str):
        self.route, self.proxy = route, proxy
        self._tok, self._tok_t, self._lock, self._local = None, 0.0, threading.Lock(), threading.local()
        self.nbytes = 0  # downloaded bytes, read by the main thread for progress

    def token(self, force=False) -> str:
        with self._lock:
            age = time.time() - self._tok_t
            if self._tok is None or age > 40 * 60 or (force and age > 60):
                env = dict(os.environ, http_proxy=self.proxy, https_proxy=self.proxy,
                           HTTP_PROXY=self.proxy, HTTPS_PROXY=self.proxy)
                self._tok = subprocess.run(["gcloud", "auth", "print-access-token"], env=env, check=True,
                                           capture_output=True, text=True, timeout=180).stdout.strip()
                self._tok_t = time.time()
            return self._tok

    def _conn(self) -> http.client.HTTPSConnection:
        c = getattr(self._local, "c", None)
        if c is None:
            if self.route == "proxy":
                u = urllib.parse.urlsplit(self.proxy)
                c = http.client.HTTPSConnection(u.hostname, u.port, timeout=60)
                c.set_tunnel(HOST, 443)
            else:
                c = http.client.HTTPSConnection(HOST, 443, timeout=60)
            self._local.c = c
        return c

    def _drop(self):
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
        self._local.c = None

    def _get(self, path: str, headers=None):
        r = None
        try:
            c = self._conn()
            c.request("GET", path, headers={"Authorization": "Bearer " + self.token(), **(headers or {})})
            r = c.getresponse()
            if r.status == 401:
                self.token(force=True)
            if r.status not in (200, 206):
                raise OSError(f"HTTP {r.status}: {r.read(300)!r}")
            return r
        except BaseException:
            self._drop()
            raise

    def _retry(self, fn, what: str, tries=40):
        for k in range(tries):
            try:
                return fn()
            except (OSError, http.client.HTTPException) as e:
                self._drop()
                if k == tries - 1:
                    raise
                if k >= 2:
                    log.warning(f"{what}: {e!r}, retry {k + 1}")
                time.sleep(min(60, 2 ** k))

    def list(self, prefix="") -> list[dict]:
        items, page = [], ""
        while True:
            q = urllib.parse.urlencode({"prefix": prefix, "pageToken": page, "maxResults": 1000,
                                        "fields": "items(name,size,md5Hash),nextPageToken"})
            js = self._retry(lambda: json.loads(self._get(f"/storage/v1/b/{BUCKET}/o?{q}").read()), "list")
            items += [{"name": o["name"], "size": int(o["size"]), "md5": o["md5Hash"]} for o in js.get("items", [])]
            if not (page := js.get("nextPageToken")):
                return items

    def fetch_range(self, name: str, fd: int, start: int, end: int):
        """Write bytes [start, end] of object `name` into fd at the same offsets; resumes within the range."""
        pos, path = start, f"/{BUCKET}/{urllib.parse.quote(name)}"

        def once():
            nonlocal pos
            r = self._get(path, {"Range": f"bytes={pos}-{end}"})
            while pos <= end:
                b = r.read(min(1 << 20, end - pos + 1))
                if not b:
                    raise OSError("connection closed mid-range")
                os.pwrite(fd, b, pos)
                pos += len(b)
                self.nbytes += len(b)  # GIL makes this += safe enough for a progress counter

        self._retry(once, f"{name}[{start}:{end}]")


# ---------------------------------------------------------------- orchestration

class Run:
    """log.txt + events.jsonl in $DATA_DIR/runs/waymo_e2e/<tag>/<time>/ (docs/long-runs.md), stdlib only."""

    def __init__(self, tag: str):
        self.dir = Path(os.environ["DATA_DIR"]) / "runs" / "waymo_e2e" / tag / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

        class TqdmHandler(logging.Handler):
            def emit(self, rec):
                tqdm.write(self.format(rec))

        for h in (TqdmHandler(), logging.FileHandler(self.dir / "log.txt")):
            h.setFormatter(fmt)
            logging.getLogger().addHandler(h)
        logging.getLogger().setLevel(logging.INFO)
        self._ev = open(self.dir / "events.jsonl", "a", buffering=1)

    def event(self, kind: str, **fields):
        self._ev.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}) + "\n")


def load_manifest(path: Path, out: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    return {n: r for n, r in rows.items()
            if (out / n).exists() and (out / n).stat().st_size == int(r["slim_bytes"])}


def download(a):
    root = Path(a.root)
    out, raw_dir = root / "front3", root / "raw"
    out.mkdir(parents=True, exist_ok=True), raw_dir.mkdir(exist_ok=True)
    run = Run("download")
    gcs = Gcs(a.route, a.proxy)
    slim_workers = a.slim_workers or max(1, n_cpus() - 4)
    raw_cap = 2 * slim_workers
    chunk = a.chunk_mb << 20

    objs = gcs.list()
    done = load_manifest(root / "manifest.csv", out)
    order = {s: i for i, s in enumerate(a.splits)}
    todo = sorted((o for o in objs if split_of(o["name"]) in order and o["name"] not in done),
                  key=lambda o: (order[split_of(o["name"])], o["name"]))
    for p in raw_dir.glob("*.part"):
        p.unlink()
    for p in out.glob("*.tmp"):
        p.unlink()
    new_manifest = not (root / "manifest.csv").exists()
    mf = open(root / "manifest.csv", "a", newline="", buffering=1)
    mw = csv.DictWriter(mf, MANIFEST_COLS)
    if new_manifest:
        mw.writeheader()

    total = sum(o["size"] for o in todo)
    log.info(f"route={a.route} streams={a.streams} slim_workers={slim_workers} raw_cap={raw_cap} "
             f"chunk={a.chunk_mb}MB min_free={a.min_free_gb}GB run_dir={run.dir}")
    log.info(f"{len(objs)} objects in bucket, {len(done)} already done, {len(todo)} to do "
             f"({total / GB:.1f} GB raw) for splits {a.splits}")
    run.event("start", route=a.route, streams=a.streams, slim_workers=slim_workers, todo=len(todo),
              todo_gb=round(total / GB, 1), done=len(done))

    def record(o, dl_s, res):
        row = {"name": o["name"], "split": split_of(o["name"]), "raw_bytes": res["raw_bytes"],
               "slim_bytes": res["slim_bytes"], "ratio": f"{res['slim_bytes'] / res['raw_bytes']:.4f}",
               "frames": res.get("frames", ""), "dl_s": dl_s, "slim_s": res.get("slim_s", 0),
               "done_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        mw.writerow(row)
        return row

    # Small files: fetch whole, md5 check, copy as-is into front3/.
    for o in [o for o in todo if split_of(o["name"]) == "small"]:
        t0, dst = time.time(), out / o["name"]
        body = gcs._retry(lambda: gcs._get(f"/{BUCKET}/{urllib.parse.quote(o['name'])}").read(), o["name"])
        if len(body) != o["size"] or base64.b64encode(hashlib.md5(body).digest()).decode() != o["md5"]:
            raise RuntimeError(f"{o['name']}: size/md5 mismatch")
        dst.write_bytes(body)
        record(o, round(time.time() - t0, 1), {"raw_bytes": o["size"], "slim_bytes": o["size"]})
        log.info(f"small file {o['name']}: {o['size'] / 1e6:.2f} MB")
    todo = collections.deque(o for o in todo if split_of(o["name"]) != "small")

    events = queue.Queue()
    dl_pool = ThreadPoolExecutor(a.streams, thread_name_prefix="dl")
    slim_pool = ProcessPoolExecutor(slim_workers, mp_context=mp.get_context("spawn"))
    downloading, slimming, fails = {}, {}, collections.Counter()
    stop_reason = None

    def submit_slim(o, dl_s):
        f = slim_pool.submit(slim_shard, str(raw_dir / o["name"]), str(out / o["name"]), o["md5"])
        slimming[o["name"]] = (o, dl_s)
        f.add_done_callback(lambda f, n=o["name"]: events.put(("slim", n, f)))

    # Complete raw shards left by an interrupted run go straight to slimming.
    for o in list(todo):
        p = raw_dir / o["name"]
        if p.exists() and p.stat().st_size == o["size"]:
            todo.remove(o)
            submit_slim(o, 0)

    def pending_bytes():
        dl = sum(d["left"] for d in downloading.values())
        return dl + sum(0.5 * o["size"] for o, _ in slimming.values())  # slim copy is written before raw is freed

    def admit():
        nonlocal stop_reason
        while todo and stop_reason is None and len(downloading) < a.max_dl_shards \
                and len(downloading) + len(slimming) < raw_cap:
            o = todo[0]
            free = shutil.disk_usage(root).free
            after = free - pending_bytes() - 1.5 * o["size"]
            if after < a.min_free_gb * GB:
                stop_reason = (f"disk guard: free {free / GB:.0f} GB, in-flight needs {pending_bytes() / GB:.0f} GB, "
                               f"next shard {o['size'] / GB:.2f} GB would leave {after / GB:.0f} GB "
                               f"< floor {a.min_free_gb} GB; stopping after in-flight shards")
                log.warning(stop_reason)
                run.event("stop", reason=stop_reason, left=len(todo))
                return
            todo.popleft()
            part = raw_dir / (o["name"] + ".part")
            fd = os.open(part, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, o["size"])
            ranges = [(s, min(s + chunk, o["size"]) - 1) for s in range(0, o["size"], chunk)]
            downloading[o["name"]] = {"o": o, "fd": fd, "n": len(ranges), "left": o["size"], "err": None,
                                      "t0": time.time()}
            for s, e in ranges:
                f = dl_pool.submit(gcs.fetch_range, o["name"], fd, s, e)
                f.add_done_callback(lambda f, n=o["name"], sz=e - s + 1: events.put(("chunk", n, f, sz)))

    bar = tqdm(total=total, unit="B", unit_scale=True, desc="download", dynamic_ncols=True, smoothing=0.05)
    t_start, last_status, n_done, slim_bytes_done, raw_bytes_done = time.time(), time.time(), 0, 0, 0
    while True:
        admit()
        if not downloading and not slimming:
            break
        try:
            ev = events.get(timeout=5)
        except queue.Empty:
            ev = None
        bar.update(gcs.nbytes - bar.n)
        if time.time() - last_status > 600:
            last_status, el = time.time(), time.time() - t_start
            rate = gcs.nbytes / el
            left = sum(o["size"] for o in todo) + pending_bytes()
            log.info(f"status: {gcs.nbytes / GB:.1f} GB in {el / 3600:.2f} h = {rate / 1e6:.1f} MB/s, "
                     f"{n_done} shards done, {len(todo)} queued, {len(downloading)} downloading, "
                     f"{len(slimming)} slimming, free {shutil.disk_usage(root).free / GB:.0f} GB, "
                     f"ETA {left / max(rate, 1) / 3600:.1f} h")
            run.event("status", gb=round(gcs.nbytes / GB, 2), mb_s=round(rate / 1e6, 2), done=n_done,
                      queued=len(todo), downloading=len(downloading), slimming=len(slimming))
        if ev is None:
            continue
        if ev[0] == "chunk":
            _, name, f, sz = ev
            d = downloading[name]
            d["n"] -= 1
            d["left"] -= sz
            if f.exception() is not None:
                d["err"] = d["err"] or f.exception()
            if d["n"]:
                continue
            os.close(d["fd"])
            o, part = d["o"], raw_dir / (name + ".part")
            del downloading[name]
            if d["err"]:
                part.unlink()
                fails[name] += 1
                log.error(f"{name}: download failed ({d['err']!r}), attempt {fails[name]}")
                run.event("shard_failed", name=name, stage="download", error=repr(d["err"]))
                if fails[name] < 3:
                    todo.appendleft(o)
                continue
            dl_s = round(time.time() - d["t0"], 1)
            part.rename(raw_dir / name)
            log.info(f"{name}: downloaded {o['size'] / GB:.2f} GB in {dl_s:.0f} s "
                     f"({o['size'] / dl_s / 1e6:.1f} MB/s)")
            run.event("shard_downloaded", name=name, bytes=o["size"], dl_s=dl_s)
            submit_slim(o, dl_s)
        else:
            _, name, f = ev
            o, dl_s = slimming.pop(name)
            try:
                res = f.result()
            except Exception as e:  # corrupt TFRecord or proto: redownload
                res = {"ok": False, "why": repr(e)}
                for p in (raw_dir / name, out / (name + ".tmp")):
                    p.unlink(missing_ok=True)
            if not res["ok"]:
                fails[name] += 1
                log.error(f"{name}: slim failed ({res['why']}), attempt {fails[name]}")
                run.event("shard_failed", name=name, stage="slim", error=res["why"])
                if fails[name] < 3:
                    todo.appendleft(o)
                continue
            row = record(o, dl_s, res)
            n_done += 1
            raw_bytes_done += res["raw_bytes"]
            slim_bytes_done += res["slim_bytes"]
            log.info(f"{name}: slim {res['frames']} frames, {res['raw_bytes'] / GB:.2f} -> "
                     f"{res['slim_bytes'] / GB:.2f} GB (ratio {row['ratio']}, total {slim_bytes_done / raw_bytes_done:.3f}) "
                     f"in {res['slim_s']:.0f} s ({res['raw_bytes'] / res['slim_s'] / 1e6:.0f} MB/s/worker), "
                     f"cams {res['front_cams']}")
            run.event("shard_slimmed", name=name, frames=res["frames"], raw_bytes=res["raw_bytes"],
                      slim_bytes=res["slim_bytes"], slim_s=res["slim_s"], dl_s=dl_s)
    bar.close()
    dl_pool.shutdown()
    slim_pool.shutdown()
    failed = [n for n, k in fails.items() if k >= 3]
    left = len(todo)
    reason = stop_reason or ("finished" if not failed else f"{len(failed)} shards failed 3 times")
    log.info(f"end: {n_done} shards this run, {left} not started, failed {failed}; {reason}. Rerun to resume.")
    run.event("end", done=n_done, left=left, failed=failed, reason=reason)
    return 1 if failed else 0


def inspect(a):
    """Print the facts that decide our camera subset. Reads all records of one shard."""
    E2EDFrame = e2ed_frame()
    import collections as C
    from waymo_open_dataset import dataset_pb2
    n, cams, res, intents, seqs = 0, C.Counter(), {}, C.Counter(), C.defaultdict(list)
    past, fut, pref, img_bytes = C.Counter(), C.Counter(), C.Counter(), C.Counter()
    first = None
    with open(a.file, "rb", buffering=16 << 20) as f:
        for rec in read_records(f):
            fr = E2EDFrame.FromString(rec)
            first = first or fr
            n += 1
            for im in fr.frame.images:
                cams[im.name] += 1
                img_bytes[im.name] += len(im.image)
            for cal in fr.frame.context.camera_calibrations:
                res[cal.name] = (cal.width, cal.height)
            intents[E2EDFrame.DESCRIPTOR.fields_by_name["intent"].enum_type.values_by_number[fr.intent].name] += 1
            name = fr.frame.context.name
            seqs[name.rsplit("-", 1)[0]].append(fr.frame.timestamp_micros)
            past[(len(fr.past_states.pos_x), len(fr.past_states.vel_x), len(fr.past_states.accel_x))] += 1
            fut[(len(fr.future_states.pos_x), len(fr.future_states.pos_z), len(fr.future_states.vel_x))] += 1
            pref[len(fr.preference_trajectories)] += 1
    cn = dataset_pb2.CameraName.Name.Name
    print(f"file {a.file}: {n} frames, {len(seqs)} sequences (context.name prefix before last '-')")
    print(f"example context.name: {first.frame.context.name}")
    for c in sorted(cams):
        print(f"  camera {cn(c):12s} frames {cams[c]:5d}  {res.get(c, ('?', '?'))[0]}x{res.get(c, ('?', '?'))[1]}"
              f"  mean JPEG {img_bytes[c] / cams[c] / 1e3:.0f} KB")
    dts = [(b - a_) / 1e6 for ts in seqs.values() for a_, b in zip(sorted(ts), sorted(ts)[1:])]
    frames_per_seq = C.Counter(len(v) for v in seqs.values())
    print(f"frames per sequence: {dict(sorted(frames_per_seq.items()))}")
    if dts:
        print(f"timestamp step within a sequence: min {min(dts):.3f} s, median {sorted(dts)[len(dts) // 2]:.3f} s, "
              f"max {max(dts):.3f} s")
    print(f"past_states (len pos, vel, accel): {dict(past)}")
    print(f"future_states (len pos_x, pos_z, vel): {dict(fut)}")
    print(f"preference_trajectories per frame: {dict(pref)}")
    print(f"intent: {dict(intents)}")
    print(f"first frame past_states.pos_x[-3:] {list(first.past_states.pos_x)[-3:]}, "
          f"future_states.pos_x[:3] {list(first.future_states.pos_x)[:3]}")
    print("Frame fields set in first frame:", [fd.name for fd, _ in first.frame.ListFields()],
          "| context:", [fd.name for fd, _ in first.frame.context.ListFields()],
          "| image:", [fd.name for fd, _ in first.frame.images[0].ListFields()] if first.frame.images else [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    d.add_argument("--root", default=str(Path(os.environ.get("DATA_DIR", ".")) / "datasets" / "waymo_e2e"))
    d.add_argument("--route", choices=("direct", "proxy"), default="direct", help="data path; token always via proxy")
    d.add_argument("--proxy", default="http://127.0.0.1:7890")
    d.add_argument("--streams", type=int, default=32, help="concurrent range GETs")
    d.add_argument("--chunk-mb", type=int, default=32)
    d.add_argument("--max-dl-shards", type=int, default=3, help="shards downloading at once")
    d.add_argument("--slim-workers", type=int, default=0, help="0: cgroup cores - 4")
    d.add_argument("--min-free-gb", type=float, default=200)
    i = sub.add_parser("inspect")
    i.add_argument("file")
    a = p.parse_args()
    sys.exit(download(a) if a.cmd == "download" else inspect(a))


if __name__ == "__main__":
    main()
