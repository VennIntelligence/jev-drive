"""Disk hygiene shared by both boxes: jobs write their caches and temp files to the data disk, the system disk is pruned.

The GPU box's system disk is 30 GB; jobs filled it again and again through /tmp (nvimgcodec extracts ~157 MB of .so
per host process into a fresh temp dir and never deletes it: ~2 200 dirs, 4.5 GB in 3 days of AlpaSim reads;
torch.compile's /tmp/torchinductor_<user> grows 0.5-1.8 GB on compile-heavy days; NVRTC leaves
/tmp/__nvrtc_auto_pch_<pid>_* behind killed processes). Two parts:

  job_env(tag)   env for one job: TMPDIR = <data>/tmp/cl/<tag> (removed by release(tag) when the job ends, any rc),
                 RAY_TMPDIR = /tmp/ray-<tag> (short: Ray's socket paths must stay under 107 bytes; also removed by release),
                 TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR / CUDA_CACHE_PATH = shared caches under <data>/cache/
                 (persistent, so later jobs still start warm; the CUDA JIT cache is seeded once from ~/.nv/ComputeCache).
                 A job's own env wins over these. Users: the GPU pool (every job, jevdrive/cl/pool.py), tmux_run.sh,
                 and any script through `eval "$(python3 -m jevdrive.cl.disk env --tag T)"` ... `release --tag T --guard`.
  prune()        removes known-safe junk only, logs every deletion (JSON lines): job temp dirs under <data>/tmp/cl that no
                 live process holds, /tmp/__nvrtc_auto_pch_<pid>_* of dead pids, and the least recently used leaf dirs
                 of capped compile caches above their cap. Nothing a readable live process holds (cwd, open fd, mapping,
                 TMPDIR env) is touched. Other /tmp entries are listed (`strays`) for a person, never deleted.
                 The pool runs it when the system disk is low or every every_s; elsewhere `release --guard` runs it when low.
  docker         (Tokyo) dangling images that no container uses and build cache older than docker_until_h; dry run unless
                 --run; never removes a tagged image (so never jev-alpasim:*) nor one any container uses.

Paths and thresholds come from the host: data = $DATA_DIR (else ~/data), system disk = the filesystem of the default
temp dir, low = free < low_frac x its size (30 GB box: 6 GB). Overrides per host in <data>/runs/disk.json (keys of
DEFAULTS). CLI: python3 -m jevdrive.cl.disk {env, release, prune, docker} (stdlib only).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

G = 2 ** 30
DEFAULTS = dict(
    low_frac=0.2,                   # prune when the system disk has less than this share free
    every_s=6 * 3600.0,             # and at least this often (pool)
    min_age_s=600.0,                # never remove a job dir / pch dir younger than this
    caps_gb={"{tmp}/torchinductor_{user}": 2.0, "{data}/cache/torchinductor": 40.0, "{data}/cache/triton": 10.0},
    cuda_cache_bytes=4 * G,         # CUDA_CACHE_MAXSIZE (the driver's own cap, 4 GiB maximum)
    stray_days=1.0,                 # strays: other /tmp entries of this user older than this are listed
    docker_until_h=72.0,            # build cache not used for this long is pruned
)
PCH_RE = re.compile(r"^__nvrtc_auto_pch_(\d+)_\d+$")


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR") or Path.home() / "data")


def sys_tmp() -> Path:
    """The system temp dir, ignoring a TMPDIR that already points elsewhere (e.g. inside a job)."""
    return Path("/tmp")


def cfg(data: Path = None) -> dict:
    c = dict(DEFAULTS)
    try:
        c.update(json.loads(((data or data_dir()) / "runs" / "disk.json").read_text()))
    except (OSError, ValueError):
        pass
    return c


def job_root(data: Path = None) -> Path:
    return (data or data_dir()) / "tmp" / "cl"


def cache_dir(name: str, data: Path = None) -> Path:
    return (data or data_dir()) / "cache" / name


def _seed(dst: Path, src: Path) -> None:
    """Create dst once, as a copy of src when src exists (a warm start); atomic against concurrent seeders."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".seed-%d" % os.getpid())
    try:
        if src.is_dir():
            shutil.copytree(src, part, symlinks=True)
        else:
            part.mkdir()
        part.rename(dst)
    except OSError:
        shutil.rmtree(part, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)


def ray_tmp(tag: str) -> Path:
    return sys_tmp() / f"ray-{tag}"


def job_env(tag: str, data: Path = None, make: bool = True) -> dict:
    """Env that keeps one job's temp files and compile caches on the data disk (see module doc)."""
    data = data or data_dir()
    t = job_root(data) / tag
    # Ray puts AF_UNIX sockets under its temp dir (<dir>/ray/session_<ts>_<pid>/sockets/plasma_store, limit 107 bytes);
    # below <data>/tmp/cl/<tag> that is too long (NAVSIM devkit scoring failed), so Ray gets a short dir on /tmp.
    env = dict(TMPDIR=str(t), RAY_TMPDIR=str(ray_tmp(tag)), TORCHINDUCTOR_CACHE_DIR=str(cache_dir("torchinductor", data)),
               TRITON_CACHE_DIR=str(cache_dir("triton", data)), CUDA_CACHE_PATH=str(cache_dir("cuda", data)),
               CUDA_CACHE_MAXSIZE=str(int(cfg(data)["cuda_cache_bytes"])))
    if make:
        t.mkdir(parents=True, exist_ok=True)
        ray_tmp(tag).mkdir(parents=True, exist_ok=True)
        for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            Path(env[k]).mkdir(parents=True, exist_ok=True)
        _seed(Path(env["CUDA_CACHE_PATH"]), Path.home() / ".nv" / "ComputeCache")
    return env


def du(p: Path) -> int:
    if p.is_symlink() or not p.is_dir():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    n = 0
    for r, ds, fs in os.walk(p):
        for f in fs:
            try:
                n += os.lstat(os.path.join(r, f)).st_size
            except OSError:
                pass
    return n


def _rm(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)


def release(tag: str, data: Path = None) -> int:
    """Remove one job's temp dir (only ever below job_root). Returns the bytes it held."""
    r = ray_tmp(tag)
    if "/" not in tag and r.is_dir() and not r.is_symlink():
        shutil.rmtree(r, ignore_errors=True)
    root = job_root(data).resolve()
    t = (root / tag).resolve()
    if t.parent != root or not t.exists():
        return 0
    n = du(t)
    _rm(t)
    return n


def held(proc: Path = Path("/proc")) -> tuple:
    """(paths, tmpdirs): every path a readable live process has as cwd, open fd or file mapping, and every TMPDIR in
    their env. A dir is held when one of them lies inside it."""
    paths, tmps = set(), set()
    for p in proc.glob("[0-9]*"):
        try:
            paths.add(os.readlink(p / "cwd"))
            for fd in os.scandir(p / "fd"):
                try:
                    paths.add(os.readlink(fd.path))
                except OSError:
                    pass
            for line in (p / "maps").read_text().splitlines():
                f = line.split(None, 5)
                if len(f) == 6 and f[5].startswith("/"):
                    paths.add(f[5].replace(" (deleted)", ""))
            for kv in (p / "environ").read_bytes().split(b"\0"):
                if kv.startswith(b"TMPDIR="):
                    tmps.add(kv[7:].decode(errors="replace"))
        except OSError:
            continue
    return paths, tmps


def is_held(d: Path, h: tuple) -> bool:
    s = str(d)
    return any(x == s or x.startswith(s + "/") for x in h[0] | h[1])


def pid_alive(pid: int) -> bool:
    return Path("/proc/%d" % pid).exists()


def _age(p: Path, now: float) -> float:
    try:
        st = p.lstat()
        return now - max(st.st_mtime, st.st_ctime)
    except OSError:
        return 0.0


def leaves(root: Path) -> list:
    """[(last use, bytes, dir)] of the leaf directories (no subdirs) under root; last use = max atime / mtime of the files."""
    out = []
    for r, ds, fs in os.walk(root):
        if ds or r == str(root):
            continue
        t, n = 0.0, 0
        for f in fs:
            try:
                st = os.lstat(os.path.join(r, f))
                t, n = max(t, st.st_atime, st.st_mtime), n + st.st_size
            except OSError:
                pass
        out.append((t, n, Path(r)))
    return out


def prune(data: Path = None, tmp: Path = None, dry: bool = False, held_fn=held, alive_fn=pid_alive, now: float = None,
          c: dict = None) -> dict:
    """One pass of the safe-prune rule (module doc). Returns {removed: [[path, bytes, rule]], freed_gb, free_gb (system
    disk, after), strays: [[path, bytes]] (largest first), strays_n}."""
    data, tmp = Path(data or data_dir()), Path(tmp or sys_tmp())
    c, now = c or cfg(data), now or time.time()
    h = held_fn()
    removed = []

    def rm(p: Path, rule: str, n: int = None):
        n = du(p) if n is None else n
        if not dry:
            _rm(p)
        removed.append([str(p), n, rule])

    root = job_root(data)
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir() and not d.name.startswith(".") and _age(d, now) > c["min_age_s"] and not is_held(d, h):
            rm(d, "job_tmp")
    for d in tmp.glob("__nvrtc_auto_pch_*"):
        m = PCH_RE.match(d.name)
        if m and not alive_fn(int(m.group(1))) and _age(d, now) > c["min_age_s"] and not is_held(d, h):
            rm(d, "nvrtc_pch")
    user = os.environ.get("USER") or Path.home().name
    for pat, cap in (c.get("caps_gb") or {}).items():
        d = Path(pat.format(tmp=tmp, data=data, user=user))
        if not d.is_dir():
            continue
        ls = sorted(leaves(d), key=lambda x: x[0])
        total, goal = sum(x[1] for x in ls), 0.8 * cap * G
        if total <= cap * G:
            continue
        for t, n, p in ls:
            if total <= goal:
                break
            if now - t > c["min_age_s"] and not is_held(p, h):
                rm(p, "cache_cap", n)
                total -= n
    known = {"torchinductor_" + user} | {Path(r[0]).name for r in removed}
    strays = []
    for p in tmp.iterdir() if tmp.is_dir() else []:
        try:
            st = p.lstat()
        except OSError:
            continue
        if (st.st_uid == os.getuid() and p.name not in known and not PCH_RE.match(p.name)
                and not p.name.startswith("tmux-") and now - st.st_mtime > c["stray_days"] * 86400):
            strays.append([str(p), du(p)])
    strays.sort(key=lambda x: -x[1])
    return dict(removed=removed, freed_gb=round(sum(r[1] for r in removed) / G, 2), free_gb=round(free_gb(tmp), 1),
                strays=strays[:20], strays_n=len(strays), strays_gb=round(sum(s[1] for s in strays) / G, 2), dry=dry)


def free_gb(p: Path = None) -> float:
    return shutil.disk_usage(p or sys_tmp()).free / G


def low(tmp: Path = None, c: dict = None) -> bool:
    u = shutil.disk_usage(tmp or sys_tmp())
    return u.free < (c or cfg())["low_frac"] * u.total


def log_path(data: Path = None) -> Path:
    return (data or data_dir()) / "runs" / "disk.jsonl"


def prune_logged(log: Path = None, dry: bool = False, **kw) -> dict:
    """prune() under a lock (one pass at a time per host), with one JSON line appended to `log` when it removed
    anything (every path and its size) or was a dry run."""
    data = Path(kw.get("data") or data_dir())
    root = job_root(data)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".prune.lock", "a") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return dict(skipped=True)
        r = prune(dry=dry, **kw)
    if r["removed"] and not dry:
        log = Path(log or log_path(data))
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps(dict(t=time.strftime("%F %T"), kind="disk_prune", **{k: r[k] for k in (
                "removed", "freed_gb", "free_gb", "strays_n", "strays_gb")})) + "\n")
    return r


# ---------------------------------------------------------------------------------------------------- docker (Tokyo)
def _sh(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _bytes(s: str) -> float:
    m = re.match(r"([\d.]+)\s*([kMGT]?B)", s.strip().rstrip("*"))
    return float(m.group(1)) * {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[m.group(2)] if m else 0.0


def _df(kind: str) -> str:
    for l in _sh(["docker", "system", "df", "--format", "{{json .}}"]).splitlines():
        d = json.loads(l)
        if d.get("Type") == kind:
            return "%s (reclaimable %s)" % (d.get("Size"), d.get("Reclaimable"))
    return ""


def docker_plan(until_h: float = None) -> dict:
    """What `docker --run` would remove: dangling (untagged) images that no container (running or stopped) uses, and
    the build cache not used for until_h. Tagged images are never candidates; unused tagged ones are only listed."""
    until_h = until_h or cfg()["docker_until_h"]
    used = set(_sh(["docker", "ps", "-a", "--format", "{{.Image}}"]).split())          # names (or ids) containers run
    cids = _sh(["docker", "ps", "-aq"]).split()
    used_ids = set(_sh(["docker", "inspect", "-f", "{{.Image}}"] + cids).split()) if cids else set()
    rows = [l.split("\t") for l in _sh(["docker", "images", "--no-trunc", "--format",
                                        "{{.ID}}\t{{.Repository}}:{{.Tag}}\t{{.Size}}"]).splitlines() if l]
    dangling = [r for r in rows if r[1] == "<none>:<none>" and r[0] not in used_ids]
    unused_tagged = [r for r in rows if r[1] != "<none>:<none>" and r[0] not in used_ids and r[1] not in used]
    cache = 0.0
    try:
        for l in _sh(["docker", "buildx", "du", "--filter", "until=%dh" % until_h]).splitlines():
            f = l.split()
            if len(f) >= 3 and f[1] == "true" and not f[2].endswith("*"):     # "*": shared, counted in another row
                cache += _bytes(f[2])
    except (subprocess.CalledProcessError, OSError):
        cache = float("nan")
    return dict(dangling=[[r[0], r[2]] for r in dangling], dangling_gb=round(sum(_bytes(r[2]) for r in dangling) / 1e9, 1),
                build_cache_gb=round(cache / 1e9, 1), build_cache_total=_df("Build Cache"), until_h=until_h,
                unused_tagged=[[r[1], r[2]] for r in unused_tagged], used=sorted(used))


def docker_prune(run: bool = False, until_h: float = None, log: Path = None) -> dict:
    p = docker_plan(until_h)
    if not run:
        return dict(p, dry=True)
    ids = [r[0] for r in p["dangling"]]
    for i in ids:                      # `docker rmi` of an untagged id refuses images a container uses (checked above too)
        subprocess.run(["docker", "rmi", i], capture_output=True)
    out = _sh(["docker", "builder", "prune", "-f", "--filter", "until=%dh" % p["until_h"]]).strip().splitlines()
    r = dict(p, dry=False, removed_images=ids, builder=out[-1] if out else "")
    log = Path(log or log_path())
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(dict(t=time.strftime("%F %T"), kind="docker_prune", images=ids, builder=r["builder"],
                                dangling_gb=p["dangling_gb"], build_cache_gb=p["build_cache_gb"])) + "\n")
    return r


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m jevdrive.cl.disk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("env", help="print `export K=V` lines for one job (eval them in the job's shell)")
    s.add_argument("--tag", required=True)
    s = sub.add_parser("release", help="remove a job's temp dir")
    s.add_argument("--tag", required=True)
    s.add_argument("--guard", action="store_true", help="then prune when the system disk is low")
    s = sub.add_parser("prune", help="one pass of the safe-prune rule")
    s.add_argument("--dry", action="store_true")
    s.add_argument("--log", default=None, help="JSON-lines file for the deletions (default <data>/runs/disk.jsonl)")
    s.add_argument("--if-low", action="store_true", help="only when the system disk is low")
    s = sub.add_parser("docker", help="dangling images + old build cache; a dry run unless --run")
    s.add_argument("--run", action="store_true")
    s.add_argument("--until-h", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "env":
        print("".join("export %s=%s\n" % (k, shlex.quote(v)) for k, v in job_env(a.tag).items()), end="")
        return 0
    if a.cmd == "release":
        n = release(a.tag)
        if a.guard and low():
            prune_logged()
        print("released %s: %.2f GB" % (a.tag, n / G), file=sys.stderr)
        return 0
    if a.cmd == "prune":
        if a.if_low and not low():
            return 0
        print(json.dumps(prune_logged(log=a.log, dry=a.dry), indent=1))
        return 0
    print(json.dumps(docker_prune(a.run, a.until_h), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
