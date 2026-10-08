"""Parallel, resumable ModelScope downloader: the mirror caps one connection at ~0.6 MB/s but the aggregate scales with
connections, so each file is cut into 16 MB ranges fetched by many threads.

  small_vlm_pdl.py <repo> [<repo> ...] [--streams 48]      -> $DATA_DIR/models/<name>/, state in $DATA_DIR/runs/small_vlm/dl/<name>/
Restart-safe: done ranges are recorded in <file>.chunks (one line per finished range); a finished file is sha256-checked
against the listing, then renamed. DONE / ERROR / STATUS per repo in the state dir.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
CH = 16 << 20
API = "https://modelscope.cn/api/v1/models/%s/repo"


def get(url, rng=None, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % rng} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:                      # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def listing(repo):
    j = json.loads(get("https://modelscope.cn/api/v1/models/%s/repo/files?Revision=master&Recursive=true" % repo))
    return [f for f in j["Data"]["Files"] if f["Type"] == "blob" and not f["Name"].startswith(".DS_Store")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def fetch_chunk(url, tmp, lo, hi, state, lock_f):
    data = get(url, (lo, hi))
    assert len(data) == hi - lo + 1, (lo, hi, len(data))
    fd = os.open(tmp, os.O_WRONLY)
    os.pwrite(fd, data, lo)
    os.close(fd)
    lock_f.write("%d\n" % lo)
    lock_f.flush()
    state["bytes"] += len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--streams", type=int, default=48)
    a = ap.parse_args()
    ex = ThreadPoolExecutor(a.streams)
    t0, state = time.time(), dict(bytes=0)
    for repo in a.repos:
        name = repo.split("/")[1]
        sd = DATA / "runs/small_vlm/dl" / name
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "ERROR").unlink(missing_ok=True)
        out = DATA / "models" / name
        files, futs = listing(repo), []
        total = sum(f["Size"] for f in files)
        (sd / "STATUS").write_text("pdl start %s, %d files, %.2f GB\n" % (time.strftime("%FT%T"), len(files), total / 2 ** 30))
        opened = []
        for f in files:
            dst = out / f["Path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == f["Size"]:
                continue
            tmp = Path(str(dst) + ".part")
            ck = Path(str(dst) + ".chunks")
            if not tmp.exists():
                with open(tmp, "wb") as fh:
                    fh.truncate(f["Size"])
                ck.unlink(missing_ok=True)
            done = {int(x) for x in ck.read_text().split()} if ck.exists() else set()
            lf = open(ck, "a")
            opened.append(lf)
            url = API % repo + "?Revision=master&FilePath=" + urllib.parse.quote(f["Path"])
            for lo in range(0, f["Size"], CH):
                if lo not in done:
                    futs.append(ex.submit(fetch_chunk, url, tmp, lo, min(lo + CH, f["Size"]) - 1, state, lf))
        left = len(futs)
        while any(not x.done() for x in futs):
            time.sleep(10)
            (sd / "STATUS").write_text("pdl %s: %d/%d chunks left, %.1f MB/s avg this run\n" % (
                time.strftime("%T"), sum(not x.done() for x in futs), left, state["bytes"] / 1e6 / (time.time() - t0)))
        for x in futs:
            x.result()
        for lf in opened:
            lf.close()
        for f in files:
            dst = out / f["Path"]
            tmp = Path(str(dst) + ".part")
            if tmp.exists():
                if f.get("Sha256") and sha256(tmp) != f["Sha256"]:
                    (sd / "ERROR").write_text("sha256 mismatch %s\n" % f["Path"])
                    tmp.unlink()
                    Path(str(dst) + ".chunks").unlink(missing_ok=True)
                    sys.exit(1)
                tmp.rename(dst)
                Path(str(dst) + ".chunks").unlink(missing_ok=True)
        (sd / "DONE").write_text("pdl %s %.2f GB, %.1f MB/s avg\n" % (time.strftime("%FT%T"), total / 2 ** 30, state["bytes"] / 1e6 / (time.time() - t0)))
        (sd / "STATUS").write_text("done %s\n" % time.strftime("%T"))


if __name__ == "__main__":
    main()
