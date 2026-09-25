"""b2d_run's orphan reaper against real processes (Linux, /proc): it must kill what the runner
started, including a CARLA binary whose wrapper already exited, and never a process that merely
reuses a recorded pid or carries a matching command line.

    python scripts/test_b2d_reaper.py
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2d_run  # noqa: E402


def alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class Reaper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"
        (self.out / "servers").mkdir(parents=True)
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)   # setsid ones: the whole group
            except OSError:
                p.kill()
            p.wait()
        self.tmp.cleanup()

    def spawn(self, script, log, argv0="bash", session=True):
        with open(log, "wb") as fh:
            p = subprocess.Popen(["bash", "-c", "exec -a %s bash -c '%s'" % (argv0, script)],
                                 stdout=fh, stderr=subprocess.STDOUT,
                                 preexec_fn=os.setsid if session else None)
        self.procs.append(p)
        time.sleep(0.3)
        return p

    def runner(self):
        args = b2d_run.parse_args(["--out", str(self.out), "--workers", "1"])
        r = b2d_run.Runner(args, [("1", "Town01")])
        r.events.close()
        return r

    def test_owned_group_is_reaped_even_after_its_leader_exited(self):
        # A "wrapper" that leaves a long-lived child in its group and exits, like CarlaUE4.sh
        # after a crash; the recorded pid is the wrapper's.
        p = self.spawn("sleep 300 & echo $! > %s/child; exit 0" % self.tmp.name,
                       self.out / "servers" / "carla-7-1.log", argv0="CarlaUE4.sh")
        p.wait()
        child = int(Path(self.tmp.name, "child").read_text())
        (self.out / "servers" / "carla-7.pid").write_text(str(p.pid))
        self.assertTrue(alive(child))
        self.runner()
        time.sleep(0.5)
        self.assertFalse(alive(child))

    def test_reused_pid_of_a_foreign_process_is_left_alone(self):
        # Someone else's process, with a CARLA-looking argv and its own group, sits on the pid
        # ("; true" keeps bash from exec'ing sleep, so the argv stays).
        foreign = self.spawn("sleep 300; true", Path(self.tmp.name) / "foreign.log", argv0="CarlaUE4")
        (self.out / "servers" / "carla-7.pid").write_text(str(foreign.pid))
        self.runner()
        time.sleep(0.5)
        self.assertTrue(alive(foreign.pid))

    def test_non_leader_is_never_group_killed(self):
        # The old reaper did killpg(getpgid(pid)): a reused pid in a shell's group took the whole
        # group (a tmux window's slot_run, say). Here that group would be this test process's.
        foreign = self.spawn("sleep 300; true", self.out / "attempts-log.txt", argv0="b2d_route.py",
                             session=False)
        rid = self.out / "attempts" / "r1" / "1"
        rid.mkdir(parents=True)
        (rid / "route.pid").write_text(str(foreign.pid))
        self.runner()
        time.sleep(0.5)
        self.assertTrue(alive(foreign.pid))
        self.assertTrue(alive(os.getpid()))


if __name__ == "__main__":
    unittest.main()
