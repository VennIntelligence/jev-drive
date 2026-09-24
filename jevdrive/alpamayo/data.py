"""PhysicalAI-AV clips for Alpamayo without downloading 2 GB chunk zips.

Each dataset feature is packed per chunk (~100 clips) into one zip on HF. We read a remote chunk's central
directory with HTTP range requests, fetch only the members of the clips we want (parallel ranged streams),
and write them into a *sparse* chunk zip under a private HF-cache-layout dir. The official loader
(`alpamayo1_5.load_physical_aiavdataset` + `physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=...)`) then
reads those clips offline, unchanged. The global HF cache is never touched with sparse files.
"""
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from jevdrive.common import data_dir
from jevdrive.hfdl import auth, hf_url, repo_info

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAMERAS = ("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
           "camera_front_tele_30fov")  # the 4 cameras the Alpamayo 1.5 loader uses by default
FEATURES = ("egomotion",) + CAMERAS


def cache_dir() -> Path:
    return data_dir() / "datasets" / "physical_ai_av" / "hub"


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file via Range requests, with an explicit parallel prefetch."""

    def __init__(self, url: str, streams: int = 8, block: int = 1 << 20):
        r = requests.head(url, headers=auth(), allow_redirects=False, timeout=60)
        # hf-mirror answers with a signed CDN redirect; range-read the CDN URL directly (no auth header)
        self.url = r.headers["location"] if r.status_code in (301, 302, 307) else url
        self.size = int(r.headers.get("x-linked-size") or r.headers["content-length"])
        self.pos, self.streams, self.block, self.segs = 0, streams, block, []  # segs: [(start, bytes)]
        self.nbytes = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get(self, a: int, b: int) -> bytes:  # inclusive range, retried
        for attempt in range(8):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=(10, 60))
                r.raise_for_status()
                assert len(r.content) == b - a + 1, "short read"
                self.nbytes += len(r.content)
                return r.content
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2 * (attempt + 1))

    def prefetch(self, a: int, n: int, chunk: int = 4 << 20):
        starts = range(a, min(a + n, self.size), chunk)
        with ThreadPoolExecutor(self.streams) as ex:
            parts = list(ex.map(lambda s: self._get(s, min(s + chunk, self.size, a + n) - 1), starts))
        self.segs.append((a, b"".join(parts)))

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        for s, data in self.segs:
            if s <= self.pos and self.pos + n <= s + len(data):
                buf[:n] = data[self.pos - s:self.pos - s + n]
                self.pos += n
                return n
        m = min(max(n, self.block), self.size - self.pos)
        self.segs.append((self.pos, self._get(self.pos, self.pos + m - 1)))
        return self.readinto(buf)


def fetch_clips(clip_ids: list[str], features=FEATURES, streams: int = 8, log=print) -> dict:
    """Make `clip_ids` readable offline through PhysicalAIAVDatasetInterface(cache_dir=cache_dir(), revision=rev).
    Returns {"revision", "bytes", "seconds"}. Already-present members are skipped."""
    import physical_ai_av
    rev = repo_info(REPO, dataset=True)["sha"]
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev, cache_dir=cache_dir(),
                                                       confirm_download_threshold_gb=1.0)
    snap = cache_dir() / f"datasets--{REPO.replace('/', '--')}" / "snapshots" / rev
    by_file: dict[str, list[str]] = {}
    for c in clip_ids:
        for f in features:
            path = avdi.features.get_chunk_feature_filename(avdi.get_clip_chunk(c), f)
            by_file.setdefault(path, []).extend(avdi.features.get_clip_files_in_zip(c, f).values())
    t0, total = time.monotonic(), 0

    def one(item):
        path, members = item
        dst = snap / path
        have = set(zipfile.ZipFile(dst).namelist()) if dst.exists() else set()
        todo = [m for m in members if m not in have]
        if not todo:
            return 0
        rf = RemoteFile(hf_url(REPO, path, dataset=True, rev=rev), streams=streams)
        zf = zipfile.ZipFile(rf)
        infos = [zf.getinfo(m) for m in todo if m in zf.NameToInfo]  # blurred_boxes can be absent
        for i in infos:  # local header (30 B + name + extra) sits right before the data
            rf.prefetch(i.header_offset, 30 + len(i.filename) + 1024 + i.compress_size)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "a", zipfile.ZIP_STORED) as out:
            for i in infos:
                out.writestr(i.filename, zf.read(i.filename))
        return rf.nbytes

    with ThreadPoolExecutor(4) as ex:  # 4 files x `streams` ranged streams each
        for n in ex.map(one, by_file.items()):
            total += n
    dt = time.monotonic() - t0
    log(f"{len(clip_ids)} clips, {len(by_file)} chunk files, {total / 1e6:.0f} MB in {dt:.0f} s "
        f"= {total / 1e6 / max(dt, 1e-9):.1f} MB/s")
    return {"revision": rev, "bytes": total, "seconds": dt}


def interface(revision: str):
    """Dataset interface over the sparse cache; only clips fetched by fetch_clips are readable."""
    import physical_ai_av
    return physical_ai_av.PhysicalAIAVDatasetInterface(revision=revision, cache_dir=cache_dir())
