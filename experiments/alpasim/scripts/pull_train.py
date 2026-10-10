#!/usr/bin/env python3
"""Pull the adapter-training data GPU box -> Tokyo box over N long-lived ssh master connections, chunked, resumable, verified.
Runs ON the Tokyo box. The box link is capped per TCP connection (a stream is ~0.05-3 MB/s, the sum grows with the stream count), so
every worker owns one connection (ssh ControlMaster opened by open_masters.sh with a forwarded agent: no key is ever copied here)
and fetches 16 MB byte ranges (dd on the box, nice) written in place at their offset into <dest>.pulling.
  pull_train.py plan            write files.tsv (group, rel path, bytes, compress flag) from the box
  pull_train.py run [N]         pull every group in priority order (tail group only if ENABLE_TAIL exists), verify sha256 against the box
Never truncates or replaces a complete file: an existing dest of the right size is only verified; a mismatch is reported (ERROR), not deleted.
State: $DATA_DIR/runs/op_parity/pull_train/{files.tsv,state/,log.txt,STATUS,DONE,ERROR,events.jsonl}."""
import hashlib, json, os, subprocess, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/data")); W = D / "runs/op_parity/pull_train"; ST = W / "state"; CTL = W / "ctl"
BOXD = os.environ.get("BOX_DATA", "/root/autodl-tmp/ujs"); CH = 16 << 20
env = dict(l.strip().split("=", 1) for l in (D / "runs/alpasim/tokyo_setup/box.env").read_text().split("\n") if "=" in l)
BOX = f"{env['BOX_USER']}@{env['BOX_HOST']}"; PORT = env["BOX_PORT"]
LAB = ["runs/op_probe/labels/navtrain_all.npz", "runs/op_parity/agent_labels/navtrain_all-k32.npz",
       "runs/body1/labels_road/navtrain_s01234567891011.npz", "runs/body1/labels/offsets.npy"]
FAM = ["", "ot1_", "yr1_", "bd4_"]
BASE = ["tab.npz", "tab.npz.key", "MANIFEST.jsonl", "timing.json"]; WARP = ["front.npy", "front.npy.key", "teacher.npz", "teacher.npz.key", "MANIFEST.jsonl", "timing.json"]
CR = "runs/op_parity/cache"
log_lock = threading.Lock()


def log(*a):
    with log_lock, (W / "log.txt").open("a") as f:
        f.write(time.strftime("%F %T ") + " ".join(map(str, a)) + "\n")


def ssh(i, cmd, **kw):  # i = master index (None: any)
    return ["ssh", "-S", str(CTL / f"m{i}"), "-o", "ControlMaster=no", "-o", "BatchMode=yes", "-p", PORT, BOX, cmd]


def box_out(cmd):
    for i in range(64):
        if subprocess.run(["ssh", "-S", str(CTL / f"m{i}"), "-O", "check", BOX], capture_output=True).returncode == 0:
            return subprocess.run(ssh(i, cmd), capture_output=True, text=True, timeout=600, check=True).stdout
    sys.exit("no live master")


def plan():
    shard = [2] + [k for k in range(12) if k != 2]
    groups = [("0-labels", LAB, "")]
    for fi, k in enumerate(shard):  # s2 first: all four families of the shard together, then shard by shard
        pass
    def dirs(k, fams):
        return [(f"{f}navtrain_full.s{k}of12", BASE) for f in fams] + [(f"{f}navtrain_full.s{k}of12@warp", WARP) for f in fams]
    order = [("1-s2-navtrain", dirs(2, FAM[:1])), ("2-s2-others", dirs(2, FAM[1:]))] + [(f"3-s{k}", dirs(k, FAM)) for k in shard[1:]]
    cmds = [f"cd {BOXD}; for f in {' '.join(LAB)}; do stat -c '0-labels\t%n\t%s\t1' $f; done"]
    for g, ds in order:
        for d, names in ds:
            cmds.append(f"cd {BOXD}/{CR}/{d}; for f in {' '.join(names)}; do [ -f $f ] && stat -c '{g}\t{CR}/{d}/%n\t%s\t0' $f; done")
    # optional tail: files the P2 / frames=warp recipe does not read
    tail = [(f"{f}navtrain_full.s{k}of12", "side.npy side.npy.key side_sample0.npy side_sample1.npy side_sample2.npy side_sample3.npy") for f in FAM[:1] for k in range(12)] + \
           [(f"{f}navtrain_full.s{k}of12@warp", "x4.npy x4.npy.key samples.npz") for f in FAM for k in range(12)]
    for d, names in tail:
        cmds.append(f"cd {BOXD}/{CR}/{d} 2>/dev/null && for f in {names}; do [ -f $f ] && stat -c '9-tail\t{CR}/{d}/%n\t%s\t0' $f; done; true")
    out = box_out("; ".join(cmds)); (W / "files.tsv").write_text(out); log("plan:", len(out.splitlines()), "files")


def load():
    rows = [l.split("\t") for l in (W / "files.tsv").read_text().splitlines() if l]
    return [(g, p, int(s), c == "1") for g, p, s, c in rows]


class Job:  # one file
    def __init__(s, g, rel, size, comp):
        s.g, s.rel, s.size, s.comp = g, rel, size, comp
        s.dest = D / rel; s.part = Path(str(s.dest) + ".pulling"); s.id = hashlib.sha1(rel.encode()).hexdigest()[:16]
        s.nch = max(1, -(-size // CH)); s.lock = threading.Lock(); s.fd = None; s.status = "new"
        s.stf = ST / f"{s.id}.chunks"; s.done = set(int(x) for x in s.stf.read_text().split()) if s.stf.exists() else set()

    def open(s):
        with s.lock:
            if s.fd is None:
                s.dest.parent.mkdir(parents=True, exist_ok=True)
                s.fd = os.open(s.part, os.O_RDWR | os.O_CREAT, 0o664)
                if os.fstat(s.fd).st_size != s.size: os.ftruncate(s.fd, s.size)
        return s.fd

    def mark(s, c):
        with s.lock:
            s.done.add(c)
            with s.stf.open("a") as f: f.write(f"{c}\n")


class Pull:
    def __init__(s, jobs, n):
        s.jobs, s.n, s.bytes, s.wire, s.t0, s.err, s.hist = jobs, n, 0, 0, time.time(), [], []
        s.q, s.qlock, s.vpool, s.vfut = None, threading.Lock(), ThreadPoolExecutor(2), []
        s.todo = 0; s.gen = s.chunks(); s.live = 0

    def chunks(s):
        for j in s.jobs:
            if j.g == "9-tail" and not (W / "ENABLE_TAIL").exists(): break
            if j.dest.exists() and j.dest.stat().st_size == j.size:  # complete (maybe the other lane's): verify only
                j.status = "present"; s.vfut.append(s.vpool.submit(s.verify, j)); continue
            if j.dest.exists(): s.fail(f"{j.rel}: dest exists with size {j.dest.stat().st_size} != {j.size}, left alone"); continue
            j.open()
            if len(j.done) >= j.nch: s.vfut.append(s.vpool.submit(s.verify, j)); continue
            j.status = "pulling"
            for c in range(j.nch):
                if c not in j.done: yield j, c
    def next(s):
        with s.qlock: return next(s.gen, None)

    def fail(s, m): s.err.append(m); log("ERROR", m); (W / "ERROR").open("a").write(m + "\n")

    def fetch(s, j, c, i):  # one range via master i; byte-resumable inside the chunk for plain files
        a = c * CH; n = min(CH, j.size - a); got = 0
        fd = j.open()
        for att in range(1000):
            if j.comp:
                cmd = f"nice -n19 dd if={BOXD}/{j.rel} bs=1M iflag=skip_bytes,count_bytes skip={a} count={n} 2>/dev/null | pigz -1 -p2 -c"; z = zlib.decompressobj(31); got = 0
            else:
                cmd = f"nice -n19 dd if={BOXD}/{j.rel} bs=1M iflag=skip_bytes,count_bytes skip={a + got} count={n - got} 2>/dev/null"
            p = subprocess.Popen(ssh(i, cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while True:
                b = p.stdout.read(1 << 20)
                if not b: break
                s.wire += len(b)
                if j.comp: b = z.decompress(b)
                os.pwrite(fd, b, a + got); got += len(b); s.bytes += len(b)
            p.wait()
            if got == n: os.fsync(fd) if c == j.nch - 1 else None; return True
            if j.comp: s.bytes -= got
            time.sleep(min(60, 2 + att))  # dead master or dropped connection: wait for open_masters.sh, retry
        return False

    def worker(s, i):
        while True:
            t = s.next()
            if t is None: return
            j, c = t; s.live += 1
            try:
                if s.fetch(j, c, i): j.mark(c)
                else: s.fail(f"{j.rel} chunk {c}")
            finally: s.live -= 1
            if len(j.done) >= j.nch and j.status == "pulling":
                j.status = "pulled"; s.vfut.append(s.vpool.submit(s.verify, j))

    def verify(s, j):
        try:
            if j.status == "pulled":
                os.fsync(j.fd); os.close(j.fd); src = j.part
            else: src = j.dest
            h = hashlib.sha256()
            with open(src, "rb") as f:
                while b := f.read(8 << 20): h.update(b)
            want = box_out(f"nice -n19 sha256sum {BOXD}/{j.rel}").split()[0]
            if h.hexdigest() != want:
                s.fail(f"{j.rel}: sha256 mismatch (local {h.hexdigest()[:12]} box {want[:12]})")
                if j.status == "pulled": j.stf.unlink(missing_ok=True); j.part.unlink(missing_ok=True)  # my own partial: refetch on the next run
                return
            if j.status == "pulled":
                try: os.link(j.part, j.dest)  # fails if the other lane created it meanwhile: never replace
                except FileExistsError: log(j.rel, "appeared meanwhile, kept theirs")
                j.part.unlink(missing_ok=True)
            j.status = "verified"
            with (W / "events.jsonl").open("a") as f: f.write(json.dumps(dict(t=time.strftime("%F %T"), rel=j.rel, bytes=j.size, sha256=want)) + "\n")
            log("verified", j.rel, j.size)
        except Exception as e: s.fail(f"{j.rel}: verify {e!r}")

    def status(s):
        tot = sum(j.size for j in s.jobs if j.g != "9-tail" or (W / "ENABLE_TAIL").exists())
        s.hist.append((time.time(), s.wire)); s.hist = [h for h in s.hist if h[0] > time.time() - 600]
        rate = (s.hist[-1][1] - s.hist[0][1]) / max(1, s.hist[-1][0] - s.hist[0][0])
        have = sum(j.size for j in s.jobs if j.status == "verified" or j.status == "present") + sum(len(j.done) * CH for j in s.jobs if j.status == "pulling")
        ver = sum(j.status == "verified" for j in s.jobs)
        cur = next((j.g for j in s.jobs if j.status in ("pulling", "new")), "-")
        left = sum((j.size - (len(j.done) * CH if j.status == "pulling" else 0)) / (3 if j.comp else 1) for j in s.jobs   # wire bytes left; label files gzip ~3x
                   if j.status not in ("verified", "present") and (j.g != "9-tail" or (W / "ENABLE_TAIL").exists()))
        eta = left / rate / 3600 if rate > 1e4 else float("nan")
        (W / "STATUS").write_text(f"{time.strftime('%F %T')} group {cur}, ~{have / 1e9:.1f}/{tot / 1e9:.1f} GB, {rate / 1e6:.2f} MB/s wire (10 min), {s.live} streams live, "
                                  f"{ver}/{len(s.jobs)} files verified, ETA {eta:.1f} h, errors {len(s.err)}\n")

    def run(s):
        ths = [threading.Thread(target=s.worker, args=(i,), daemon=True) for i in range(s.n)]
        for t in ths: t.start()
        while any(t.is_alive() for t in ths): s.status(); time.sleep(30)
        for f in s.vfut: f.result()
        s.vpool.shutdown(); s.status()
        bad = [j.rel for j in s.jobs if j.status not in ("verified",) and (j.g != "9-tail" or (W / "ENABLE_TAIL").exists())]
        if s.err or bad: log("finished with problems", len(s.err), len(bad)); (W / "ERROR").open("a").write(f"unverified: {bad[:20]}\n")
        else: (W / "DONE").write_text(time.strftime("%F %T") + "\n"); log("DONE")


if __name__ == "__main__":
    W.mkdir(parents=True, exist_ok=True); ST.mkdir(exist_ok=True)
    if sys.argv[1] == "plan": plan()
    else:
        (W / "ERROR").unlink(missing_ok=True); (W / "DONE").unlink(missing_ok=True)
        Pull([Job(*r) for r in load()], int(sys.argv[2]) if len(sys.argv) > 2 else 32).run()
