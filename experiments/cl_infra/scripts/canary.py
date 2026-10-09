"""Memory canary for the SIGKILL study (INFRA2, 2026-10-10): one process that grows at a set rate and holds.

It is meant to be the largest process by RSS in the container, so that an outside killer that takes the largest one
takes this and no job. Progress goes to stdout once a second (unbuffered); a SIGKILL shows as the missing "done" line
and the shell's rc 137.

    python canary.py --gb 60 --rate 3 --hold 120                 # anonymous memory
    python canary.py --gb 60 --rate 3 --hold 120 --file PATH     # page cache through a shared file mapping
    python canary.py --gb 60 --rate 3 --hold 120 --read PATH     # page cache through read() (no mapping, no RSS)
"""
import argparse, mmap, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--gb", type=float, required=True)
ap.add_argument("--rate", type=float, default=3.0, help="GiB per second")
ap.add_argument("--hold", type=float, default=120.0)
ap.add_argument("--file", default="", help="touch a shared mapping of this file instead of anonymous memory")
ap.add_argument("--read", default="", help="read() this file instead (page cache only)")
ap.add_argument("--advise", action="store_true", help="with --read: POSIX_FADV_DONTNEED behind the read position")
a = ap.parse_args()

G, STEP = 2**30, 64 * 2**20
size = int(a.gb * G)
t0 = time.time()
say = lambda s: print(f"{time.strftime('%H:%M:%S')} +{time.time() - t0:6.1f}s {s}", flush=True)
say(f"pid {os.getpid()} start: {a.gb} GiB at {a.rate} GiB/s, hold {a.hold}s, kind {'read' if a.read else 'file' if a.file else 'anon'}")
if a.read:
    fd = os.open(a.read, os.O_RDONLY)
    n = min(size, os.fstat(fd).st_size)
    buf = bytearray(STEP)
else:
    if a.file:
        fd = os.open(a.file, os.O_RDONLY)
        n = min(size, os.fstat(fd).st_size)
        m = mmap.mmap(fd, n, mmap.MAP_SHARED, mmap.PROT_READ)
    else:
        n = size
        m = mmap.mmap(-1, n, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        one = b"\x01" * STEP
off, last = 0, 0.0
while off < n:
    k = min(STEP, n - off)
    if a.read:
        os.preadv(fd, [memoryview(buf)[:k]], off)
        if a.advise:
            os.posix_fadvise(fd, off, k, os.POSIX_FADV_DONTNEED)
    elif a.file:
        for p in range(off, off + k, 4096):
            m[p]
    else:
        m[off:off + k] = one[:k]
    off += k
    ahead = off / G / a.rate - (time.time() - t0)
    if ahead > 0:
        time.sleep(ahead)
    if time.time() - last >= 1:
        last = time.time()
        say(f"{off / G:6.1f} GiB")
say(f"{off / G:6.1f} GiB reached, holding")
end = time.time() + a.hold
while time.time() < end:
    time.sleep(5)
    say("holding")
say("done")
