"""Range-read members of the public nuPlan v1.1 zip archives without downloading the whole archive.

  python nuplan_zip.py list  [--url U]                      -> member name, compressed and raw size (TSV on stdout)
  python nuplan_zip.py fetch --logs logs.txt --out DIR [--url U] [--streams N]
      extracts <DIR>/<log>.db for every log name in logs.txt (one per line); skips logs already complete.

S3 serves byte ranges, so zipfile reads the central directory and each member through HTTP Range requests.
Members are fetched in parallel (one connection per member); a member is written to <name>.part and renamed when
its CRC checks out (zipfile verifies it on read).
"""
import argparse, io, os, sys, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip"


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with a read-ahead buffer."""

    def __init__(self, url, block=8 << 20):
        self.url, self.pos, self.block, self.buf, self.buf_at = url, 0, block, b"", -1
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get(self, a, b):
        for attempt in range(40):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}"})
                d = urllib.request.urlopen(req, timeout=120).read()
                if len(d) == b - a: return d
                raise IOError(f"short read {len(d)} of {b - a}")
            except Exception as e:  # transient network errors (Clash drops TLS streams): retry with backoff
                if attempt == 39: raise
                print(f"retry {attempt + 1} {a}-{b}: {e}", file=sys.stderr)
                time.sleep(min(60, 2 ** min(attempt, 6)))

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if not (self.buf_at <= self.pos and self.pos + n <= self.buf_at + len(self.buf)):
            end = min(self.size, self.pos + max(n, self.block))
            self.buf, self.buf_at = self._get(self.pos, end), self.pos
        o = self.pos - self.buf_at
        self.pos += n
        return self.buf[o:o + n]

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def members(url):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url, 1 << 20), 1 << 20)).infolist()


def fetch_one(url, info, out):
    dst = os.path.join(out, os.path.basename(info.filename))
    if os.path.exists(dst) and os.path.getsize(dst) == info.file_size: return dst, "have"
    zf = zipfile.ZipFile(HttpFile(url, 32 << 20))
    with zf.open(info.filename) as src, open(dst + ".part", "wb") as f:
        while chunk := src.read(32 << 20): f.write(chunk)  # zipfile checks the CRC at EOF
    os.replace(dst + ".part", dst)
    return dst, "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["list", "fetch"])
    p.add_argument("--url", default=URL)
    p.add_argument("--logs"); p.add_argument("--out"); p.add_argument("--streams", type=int, default=8)
    a = p.parse_args()
    infos = members(a.url)
    if a.cmd == "list":
        for i in infos: print(f"{i.filename}\t{i.compress_size}\t{i.file_size}")
        return
    want = {l.strip() for l in open(a.logs) if l.strip()}
    sel = [i for i in infos if os.path.basename(i.filename).removesuffix(".db") in want]
    missing = want - {os.path.basename(i.filename).removesuffix(".db") for i in sel}
    print(f"{len(sel)} of {len(want)} logs in the archive, {sum(i.compress_size for i in sel) / 1e9:.1f} GB compressed; "
          f"missing {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    sel.sort(key=lambda i: -i.compress_size)  # largest first so the tail is short
    with ThreadPoolExecutor(a.streams) as ex:
        for k, (dst, st) in enumerate(ex.map(lambda i: fetch_one(a.url, i, a.out), sel), 1):
            print(f"[{k}/{len(sel)}] {st} {dst}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
