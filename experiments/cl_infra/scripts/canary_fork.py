"""Fork canary for the SIGKILL study (INFRA2, 2026-10-10): committed memory without resident memory.

The parent touches --gb of anonymous memory, then forks --kids children that only sleep. Every child shares the
parent's pages copy-on-write (its RSS shows the full size, the container's anon does not grow), but the kernel adds
the parent's writable mappings to the host's Committed_AS once per child. This is what a job does that loads a model
and then forks decode / dataloader workers.

    python canary_fork.py --gb 30 --kids 12 --hold 90 [--step-s 2]
"""
import argparse, mmap, os, signal, time

ap = argparse.ArgumentParser()
ap.add_argument("--gb", type=float, required=True)
ap.add_argument("--kids", type=int, default=12)
ap.add_argument("--hold", type=float, default=90.0)
ap.add_argument("--step-s", type=float, default=2.0, help="seconds between forks")
a = ap.parse_args()
t0 = time.time()


def commit():
    for line in open("/proc/meminfo"):
        if line.startswith("Committed_AS"):
            return int(line.split()[1]) / 2**20


say = lambda s: print(f"{time.strftime('%H:%M:%S')} +{time.time() - t0:6.1f}s {s} | Committed_AS {commit():.1f} GiB", flush=True)
say(f"pid {os.getpid()} start: {a.gb} GiB, {a.kids} children")
n, STEP = int(a.gb * 2**30), 64 * 2**20
m = mmap.mmap(-1, n, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
one = b"\x01" * STEP
for off in range(0, n, STEP):
    m[off:off + min(STEP, n - off)] = one[:min(STEP, n - off)]
say("parent resident")
kids = []
for i in range(a.kids):
    pid = os.fork()
    if pid == 0:
        time.sleep(a.hold + a.kids * a.step_s + 30)
        os._exit(0)
    kids.append(pid)
    time.sleep(a.step_s)
    dead = [(p, os.waitpid(p, os.WNOHANG)) for p in kids]
    say(f"forked {i + 1}: pid {pid}; reaped {[(p, s) for p, (r, s) in dead if r]}")
end = time.time() + a.hold
alive = set(kids)
while time.time() < end and alive:
    time.sleep(2)
    for p in list(alive):
        r, s = os.waitpid(p, os.WNOHANG)
        if r:
            alive.discard(p)
            say(f"child {p} gone, wait status {s} (9 = SIGKILL)")
    say(f"holding, {len(alive)} children alive")
for p in alive:
    os.kill(p, signal.SIGTERM)
say("done")
