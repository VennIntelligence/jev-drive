#!/usr/bin/env python3
"""Fixed handoff actions, without models or scheduling policy. Run once in tmux.

Remote: python3 scripts/cx_maintenance.py watch
Local export companion: python3 scripts/cx_maintenance.py collect --host autodl
One pass: add --once. Unknown cases are WAIT, never repairs or relaxed gates.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
MAX_FILE = 16 * 1024 * 1024
FLAG_BLOCKED = "> 50 % of routes blocked"
FLAG_MOVES = "ego drove (RC > 5 %) on < 50 % of routes"
FLAG_DS = "DS beats the CL1 ceiling by > 25 points"
DEADLINE = dt.datetime(2026, 9, 27, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=9))).timestamp()
TABLES = ("summary.md", "arms.csv", "paired.csv", "per_route.csv")
D_STEPS = ("q4a-prep", "q4a-qwen", "q4a-fit", "q4a-hydra", "q4a-hold", "q4a-nav", "q4b", "q5", "q6")


def digest(b):
    return hashlib.sha256(b).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".cx-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextlib.contextmanager
def lock(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def append_unique(path, entries):
    with lock(path.with_name(path.name + ".lock")):
        old = path.read_text() if path.exists() else ""
        lines = list(dict.fromkeys(old.splitlines()))
        lines += [x for x in entries if x not in lines]
        new = "\n".join(lines) + "\n"
        if new != old: atomic(path, new.encode())
        return new != old


def flag_action(v):
    """Only complete exact reason sets registered in T1 are actionable."""
    if v.get("verdict") != "FLAG" or v.get("fail"):
        return None
    reasons = set(v.get("flag", []))
    if not reasons: return None
    if v.get("arm") in {"cl2", "cl7"}:
        return "SKIP" if reasons <= {FLAG_BLOCKED, FLAG_MOVES} else "WAIT"
    return "APPROVED" if reasons <= {FLAG_BLOCKED, FLAG_DS} else "WAIT"


def alive(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError: return False


def executing_processes():
    """Read process identities; never signal a process selected by its name."""
    rows = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] != "Z":
                rows.append((int(p.name), int(stat[2]), (p / "cmdline").read_bytes().split(b"\0")))
        except (FileNotFoundError, ProcessLookupError): pass
    return rows


def cache_writers_stopped(data, job, current):
    """Require producer completion and recorded groups gone, plus no D chain."""
    producer = data / f"runs/sched/pull_forward/{job}"
    if not (producer / "DONE").exists() or not (producer / "pgid").exists(): return False
    groups = {int((producer / "pgid").read_text().strip())}
    step_pid = data / f"runs/nq3/d/{current}/pid"
    if not step_pid.exists(): return False
    groups.add(int(step_pid.read_text().strip()))
    for pid, pgid, argv in executing_processes():
        if pid in groups or pgid in groups: return False
        if any(x == b"scripts/nq3_d.sh" or x.endswith(b"/scripts/nq3_d.sh") for x in argv): return False
    return True


def run(cmd, log, cwd=REPO, timeout=3600):
    """Own child group only; no process-name killing or hidden retry."""
    with log.open("ab") as f:
        f.write((json.dumps(cmd) + "\n").encode()); f.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=f, start_new_session=True)
        atomic(log.with_suffix(".pid"), str(p.pid).encode())
        try: return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=120)
            except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
            return 124


class Watch:
    def __init__(self, data):
        self.data = Path(data)
        self.out = self.data / "runs/nq4/cx/maintenance"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.notes = {}

    def note(self, key, value):
        self.notes[key] = value
        if self.state.get(key) != value:
            self.state[key] = value
            event = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "key": key, "value": value}
            with (self.out / "events.jsonl").open("a") as f: f.write(json.dumps(event) + "\n")
            with (self.out / "log.txt").open("a") as f: f.write(json.dumps(event) + "\n")

    def flags(self):
        b = self.data / "runs/nq3/b"
        append_unique(b / "SKIP", [f"{a} {s}" for a in ("cl2", "cl7", "cl5d") for s in range(3)])
        for p in sorted((self.data / "runs/sched/pilot/b/arms").glob("*/s0/verdict.json")):
            v = json.loads(p.read_text())
            if v.get("arm") != p.parent.parent.name or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", v.get("arm", "")):
                self.note("T1:" + p.parent.parent.name, "WAIT: arm identity differs from canonical path"); continue
            action = flag_action(v)
            if action in {"APPROVED", "SKIP"}:
                entries = [v["arm"]] if action == "APPROVED" else [f"{v['arm']} {s}" for s in range(3)]
                append_unique(b / action, entries)
            if action: self.note("T1:" + p.parent.parent.name, {"action": action, "flag": v.get("flag"), "source": str(p)})
            if v.get("verdict") == "PASS":
                self.note("T2:" + v["arm"], "READY_FOR_SCHEDULING: PASS; no deterministic free-resource/FIFO allocation")

    def alpamayo(self):
        if self.state.get("T9_complete") or self.state.get("T9_failed"): return
        if alive(684982) or alive(37228):
            self.note("T9", "WAIT: owner 684982 or helper 37228 still executing"); return
        c = self.data / "runs/nq3/c"
        try:
            with lock(c / "q1_judge_final/lock", nonblocking=True):
                self._consolidate(c)
        except BlockingIOError: self.note("T9", "WAIT: final judge owns its step lock")

    def frame_count(self):
        p = self.data / "processed/carla_p6/nq3_alpamayo.npz"
        if not p.exists(): return 0
        result = subprocess.run([str(REPO / ".venv/bin/python"), "-c",
            "import numpy as np,sys; a=np.load(sys.argv[1]); print(len(a['frame_name']))", str(p)],
            capture_output=True, text=True, timeout=120, check=True)
        return int(result.stdout.strip())

    def _consolidate(self, c):
        if alive(684982) or alive(37228): return
        before = self.frame_count()
        judged_before = (c / "q1_judge_final/DONE").exists()
        rc = run([str(self.data / "third_party/alpamayo1.5/.venv/bin/python"),
                  "scripts/sch_gpu_helpers/alpamayo_units.py", "consolidate", "--owner-pid", "684982"], self.out / "T9.log")
        if rc:
            self.note("T9_failed", {"rc": rc, "action": "WAIT; no retry", "before": before}); return
        after = self.frame_count()
        counts = {"before": before, "after": after, "final_judged_before": judged_before}
        self.note("T9_counts", counts)
        if judged_before:
            steps = (REPO / "scripts/nq3_c_steps.sh").read_text()
            chain = (REPO / "scripts/nq3_c.sh").read_text()
            exact = 'q1_judge()    { OMP_NUM_THREADS=16 taskset -c 164-179 $PY -m jevdrive.nq3_q1 judge; }'
            if exact not in steps or '[[ $n == q1_judge_final ]] && fn=q1_judge' not in chain:
                self.note("T9_failed", "WAIT: registered exact final-judge command changed"); return
            rc = run(["bash", "-lc", "source scripts/nq3_c_steps.sh; q1_judge"], self.out / "T9-final-judge.log")
            if rc: self.note("T9_failed", {"rc": rc, "action": "WAIT: final-judge rerun failed"}); return
        self.note("T9_complete", counts)

    def provenance(self):
        d = self.data / "runs/nq3/d"
        current = (d / "current").read_text().strip() if (d / "current").exists() else ""
        for job in ("q4b", "q6"):
            p = self.data / f"runs/sched/pull_forward/{job}/provenance.txt"
            if not p.exists(): self.note("T11:" + job, "WAIT: provenance missing"); continue
            if current not in D_STEPS: self.note("T11:" + job, "WAIT: unknown D stage"); continue
            if D_STEPS.index(current) >= D_STEPS.index(job):
                self.note("T11:" + job, "WAIT: D already reached stage; no live cache deletion"); continue
            mismatches = []
            for line in p.read_text().splitlines():
                if line.startswith("commit "): continue
                m = re.fullmatch(r"([0-9a-f]{32})  (.+)", line)
                if not m: raise ValueError(f"invalid provenance line: {p}")
                source = (REPO / m[2]).resolve()
                if not source.is_relative_to(REPO.resolve()): raise ValueError("provenance escaped repo")
                if not source.is_file() or hashlib.md5(source.read_bytes()).hexdigest() != m[1]: mismatches.append(m[2])
            if not mismatches:
                self.note("T11:" + job, "OK: all registered md5 match; untouched"); continue
            # A stopped D chain makes before-stage invalidation reviewable and race-free.
            if not (d / "ERROR").exists() or not cache_writers_stopped(self.data, job, current):
                self.note("T11:" + job, {"action": "WAIT: mismatch while D active", "mismatch": mismatches}); continue
            paths = [self.data / "processed/carla_p5v1_ba/nq3_q4b_navsim_protocol" / f"{m}.npz" for m in ("cinque", "lebowski")]
            if job == "q6":
                paths = [self.data / f"runs/nq3/q6/{n}" for n in ("sf_units", "sf_check_1f", "sf_check_4f")]
                paths += [self.data / "processed/navsim_vjepa2/navtest"]
                heads = self.data / "runs/nq3/q6/heads"
                paths += list(heads.glob("vjmc_*_s1.pt")) + list(heads.glob("vjmc_*_s2.pt")) + [heads / "vjmc_lebowski_s0.pt"]
            for path in paths:
                if not path.resolve().is_relative_to(self.data.resolve()): raise ValueError("cache escaped data root")
                if path.is_symlink(): raise ValueError("symlink cache: WAIT")
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink(missing_ok=True)
            self.note("T11:" + job, {"action": "invalidated exact registered caches before stage", "mismatch": mismatches,
                                      "paths": [str(x) for x in paths]})

    def candidates(self):
        """Fixed handoff lists; completion gates never synthesize scientific verdicts."""
        r = self.data / "runs"
        pairs = []
        def add(root, pattern, dest):
            for p in sorted(root.glob(pattern)):
                if p.is_file(): pairs.append((p, str(Path(dest) / p.relative_to(root))))
        if (r / "nq3/a/DONE").exists():
            for n in ("v0rr_worlds.csv", "v0rr_e1.csv", "carla_rig_summary.csv", "carla_rig_summary.md", "carla_rig_per_scenario.csv", "carla_rig_mode_agreement.csv"):
                add(REPO / "research/results/nq3/q1", n, "research/results/nq3/q1")
            add(REPO / "research/results/nq3/q3", "*", "research/results/nq3/q3")
        if (r / "nq3/c/DONE").exists() and self.state.get("T9_complete"):
            for pat in ("summary.*", "per_scenario.csv", "world_modes.csv", "mode_agreement.csv", "alpamayo_cot.csv"):
                add(REPO / "research/results/nq3/q1", pat, "research/results/nq3/q1")
            for name in ("q2", "q2_v1"): add(REPO / f"research/results/nq3/{name}", "*", f"research/results/nq3/{name}")
        if (r / "nq3/d/DONE").exists():
            for name in ("q4", "q5", "q6"): add(REPO / f"research/results/nq3/{name}", "*", f"research/results/nq3/{name}")
        # B can publish per-arm tables at the registered 3-5 h collection interval.
        if time.time() - self.state.get("B_export_time", 0) >= 3 * 3600 or (r / "nq3/b/DONE").exists():
            for n in TABLES: add(r / "nq3/b/results", n, "research/results/nq3/cl")
            if any(x[0].parent == r / "nq3/b/results" for x in pairs): self.state["B_export_time"] = time.time()
        if (r / "nq4/k/READY").exists():
            add(r / "nq4/k", "READY", "research/results/nq4/k")
            add(r / "nq4/k/checks", "*.json", "research/results/nq4/k/checks")
            add(r / "nq4/k/steps/cl", "verdict.json", "research/results/nq4/k")
        if all((r / f"nq4/opl/accept/{a}.json").exists() for a in ("cl2p", "cl7p")):
            add(r / "nq4/opl/accept", "*.json", "research/results/nq3/cl/opl_smoke")
        if (r / "nq4/gk/DONE").exists():
            for name in ("g", "k", "x"): add(r / f"nq4/gk/results/{name}", "*", f"research/results/nq4/{name}")
        if (r / "nq4/w/chain/DONE").exists(): add(REPO / "research/results/nq4/w", "*", "research/results/nq4/w")
        return pairs

    def export(self):
        index_path = self.out / "export.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        for source, dest in self.candidates():
            if source.stat().st_size > MAX_FILE: self.note("T7:" + dest, "WAIT: exceeds small-file limit"); continue
            before = source.stat()
            b = source.read_bytes()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): continue
            atomic(self.out / "exports" / dest, b)
            index[dest] = {"source": str(source), "size": len(b), "mtime_ns": before.st_mtime_ns,
                           "sha256": digest(b), "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic(index_path, json.dumps(index, indent=2).encode())
        self.note("T7", f"{len(index)} fixed small files ready for Mac collector")
        missing = [n for n, marker in (("A", "nq3/a/DONE"), ("B", "nq3/b/DONE"), ("C", "nq3/c/DONE"),
                   ("D", "nq3/d/DONE"), ("G/K/X", "nq4/gk/DONE"), ("K-prep", "nq4/k/READY"), ("P3", "nq4/p3/prep.done"))
                   if not (self.data / "runs" / marker).exists()]
        self.note("remaining_raw", {"unfinished": missing, "no_final_verdict": True})

    def tick(self):
        self.notes = {}
        for name, fn in (("T1", self.flags), ("T9", self.alpamayo), ("T11", self.provenance), ("T7", self.export)):
            try: fn()
            except Exception as e: self.note(name + "_error", f"WAIT: {type(e).__name__}: {e}")
        for name, p in (("D", "runs/nq3/d/ERROR"), ("P3", "runs/nq4/p3/prep.done"), ("G", "runs/nq4/gk/DONE")):
            exists = (self.data / p).exists()
            self.note("coverage:" + name, {"marker": p, "exists": exists, "action": "WAIT/no repair" if name == "D" and exists or name == "P3" and not exists else "record only"})
        self.note("deadline", {"deadline_jst": "2026-09-27 23:59", "elapsed": time.time() >= DEADLINE,
                               "policy": "export existing partial outputs; unfinished jobs stay running"})
        atomic(self.state_path, json.dumps(self.state, indent=2).encode())
        atomic(self.out / "STATUS.md", ("# CX mechanical maintenance\n\n" + "\n".join(f"- {k}: {v}" for k, v in self.notes.items()) + "\n").encode())


def collect(host, once, interval):
    """Copy immutable exports to Mac; leave commits as a reviewable root task."""
    out = REPO / "tmp/2026-09-26-diagnostic-bundle"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "collector-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    with lock(out / "collector.lock", nonblocking=True):
        atomic(out / "collector.pid", str(os.getpid()).encode())
        while True:
            try:
                cmd = ["ssh", "-o", "ControlPath=none", "-o", "ConnectTimeout=20", host,
                       "cd ~/data/jev-drive && python3 scripts/cx_maintenance.py bundle"]
                result = subprocess.run(cmd, capture_output=True, timeout=180, check=True)
                with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
                    metadata = json.load(tar.extractfile("export.json"))
                    for m in tar.getmembers():
                        if m.name in {"export.json", "STATUS.md", "state.json"}:
                            if m.isfile(): atomic(out / "raw/maintenance" / m.name, tar.extractfile(m).read())
                            continue
                        dest = (REPO / m.name).resolve()
                        if not m.isfile() or m.size > MAX_FILE or not dest.is_relative_to((REPO / "research/results").resolve()):
                            raise ValueError("invalid export member")
                        b = tar.extractfile(m).read()
                        if digest(b) != metadata[m.name]["sha256"]: raise ValueError("export digest mismatch")
                        if dest.exists() and digest(dest.read_bytes()) != state.get(m.name, {}).get("sha256"):
                            dirty = subprocess.run(["git", "diff", "--quiet", "--", m.name], cwd=REPO)
                            untracked = subprocess.run(["git", "ls-files", "--error-unmatch", m.name], cwd=REPO,
                                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if dirty.returncode != 0 or untracked.returncode != 0:
                                raise ValueError(f"WAIT: unrelated local change preserved: {m.name}")
                        # Per-file snapshots are reviewable; no bulk features, no git stash/reset.
                        atomic(dest, b)
                        state[m.name] = metadata[m.name]
                atomic(state_path, json.dumps(state, indent=2).encode())
                atomic(out / "collector-STATUS.md", (f"Captured {len(state)} fixed raw files. READY_TO_COMMIT: root review.\n"
                       "Unfinished/failed coverage remains in raw/maintenance/STATUS.md; no final verdict inferred.\n").encode())
            except Exception as e:
                with (out / "collector-events.jsonl").open("a") as f:
                    f.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "action": "WAIT/retry next interval", "error": str(e)}) + "\n")
            if once: return
            time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("watch", "collect", "bundle"))
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", str(Path.home() / "data")))
    ap.add_argument("--host", default="autodl")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if not 300 <= a.interval <= 600: ap.error("interval must be 300-600 seconds")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): os.environ[k] = "1"
    if a.mode == "collect": collect(a.host, a.once, a.interval); return
    w = Watch(a.data)
    if a.mode == "bundle":
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            for n in ("export.json", "STATUS.md", "state.json"):
                p = w.out / n
                if p.exists(): tar.add(p, arcname=n)
            index = json.loads((w.out / "export.json").read_text())
            for dest in index: tar.add(w.out / "exports" / dest, arcname=dest)
        return
    with lock(w.out / "watch.lock", nonblocking=True):
        atomic(w.out / "pid", str(os.getpid()).encode())
        while True:
            w.tick()
            if a.once: return
            time.sleep(a.interval)


if __name__ == "__main__": main()
