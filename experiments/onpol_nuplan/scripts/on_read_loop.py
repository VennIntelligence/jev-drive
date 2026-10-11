#!/usr/bin/env python3
"""ONPOL-NUPLAN read loop: experiments/alpasim/scripts/ot2_loop.py, imported unedited, for the lane's closed-loop reads (on_read.sh calls it).

  1. scene lists are the lane's own: `smoke48_train` (results/split/smoke48_train.txt: the first 48 scenes of a numpy default_rng(0)
     permutation of results/split/train.txt, stored sorted) and `held` (results/split/held.txt, the lane's only read). $ON_SPLIT = train | held
     names the split of the stage: a job on any other list, on a list with a scene outside that split, or (held) on a list that is not
     exactly held.txt is refused before anything is submitted.
  2. before ot2_loop prunes rollout.asl (it keeps the log only for rollouts with a failure reason), every finished run is read with
     c1_extract.py `logs` (unedited, AlpaSim's venv) into <run dir>/read_sim.pkl: per scene the simulator's ego and actor poses, the logged
     trajectories, every served plan, the metric series and the summary row (about 90 KB per scene). The clips of on_clips.py need the
     rollout.asl itself (driver messages for the model frames): ot2_loop's pruning leaves it for the failed rollouts, ON_KEEP_ASL=all keeps
     every rollout's (about 25 MB each: 12 GB per 460-scene job) until on_clips.sh prunes them.

Job specs, state files, retries, checkpoint waits and the stall watchdog are ot2_loop's (its docstring).
  ON_SPLIT=held ON_LISTS=<dir> OT_LANE=onpol OT_OUT=$DATA_DIR/runs/onpol_nuplan/read OT_OWNER=alpasim-onpol python3 on_read_loop.py <stage> <job> ...
  python3 on_read_loop.py --lists            writes / verifies results/split/smoke48_train.txt (numpy)
  python3 on_read_loop.py --check <tag> <manifest.json> <label> <out.json>
      the smoke checklist of one checkpoint (stdlib): all rollouts of the list scored, 0 non-finite served plans, 0 driver inference / input
      errors, one driver session per scene with 10 drive calls each. The scores are not read into the checklist: the smoke is not a read.
"""
import glob
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ALP = REPO / "experiments/alpasim/scripts"
sys.path[:0] = [str(ALP), str(REPO)]
SPLIT = REPO / "experiments/onpol_nuplan/results/split"
LISTS = {"train": {"smoke48_train": SPLIT / "smoke48_train.txt"}, "held": {"held": SPLIT / "held.txt"}}
N_SMOKE, N_DRIVE = 48, 10


def smoke_list():
    """Write results/split/smoke48_train.txt, or verify the file that is there against the rule."""
    import numpy as np
    tr = (SPLIT / "train.txt").read_text().split()
    sm = sorted(tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:N_SMOKE])
    f, txt = LISTS["train"]["smoke48_train"], "\n".join(sm) + "\n"
    assert not f.exists() or f.read_text() == txt, f"{f} differs from the rule"
    f.write_text(txt)
    print(f"{f.name}: {len(sm)} train scenes, {len({'_'.join(s.split('_')[:2]) for s in sm})} logs")


def check(tag, man, label, out):
    dirs = [Path(d) for d in json.loads(Path(man).read_text()).get(label, [])]
    want = set(LISTS["train"]["smoke48_train"].read_text().split())
    got, calls, n_plan, n_bad, ie, ip = set(), Counter(), 0, 0, 0, 0
    for d in dirs:
        got |= {r["clipgt_id"] for r in json.loads((d / "aggregate/results-summary.json").read_text())["rollouts"] if r.get("score") is not None}
        for ln in (d / "driver-logs/drive.jsonl").open():
            r = json.loads(ln)
            if r["kind"] == "drive":
                n_plan += 1
                n_bad += not all(isinstance(v, (int, float)) and math.isfinite(v) for p in r["poses"] for v in p)
            elif r["kind"] == "close":
                calls[r["scene"]] += r.get("drive", 0)
                ie, ip = ie + r.get("inference_error", 0), ip + r.get("input_error", 0)
    odd = {s: n for s, n in sorted(calls.items()) if n != N_DRIVE}
    res = dict(tag=tag, list="smoke48_train", scenes=len(want), rollouts=len(got & want), sessions=len(calls), drive_calls=sum(calls.values()), served_plans=n_plan,
               nonfinite_plans=n_bad, inference_errors=ie, input_errors=ip, sessions_not_10_calls=odd, run_dirs=[str(d) for d in dirs],
               note="the scores of this smoke are not a read and are not in this file")
    bad = ([] if dirs else ["no finished run in the manifest"]) + [m for ok, m in (
        (got >= want, f"rollouts {len(got & want)} / {len(want)}"), (not n_bad, f"{n_bad} non-finite served plans"), (not ie and not ip, f"driver errors: inference {ie}, input {ip}"),
        (set(calls) == want, f"driver sessions for {len(set(calls) & want)} of {len(want)} scenes"), (not odd, f"{len(odd)} sessions without {N_DRIVE} drive calls")) if not ok]
    res |= {"fail": bad, "pass": not bad}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "run_dirs"}, indent=1))
    sys.exit(0 if not bad else 2)


def main():
    if sys.argv[1] == "--lists":
        return smoke_list()
    if sys.argv[1] == "--check":
        return check(*sys.argv[2:6])
    import ot2_loop as L
    from jevdrive.common import n_cpus
    B = L.B
    split, lists, keep_all = os.environ["ON_SPLIT"], Path(os.environ["ON_LISTS"]), os.environ.get("ON_KEEP_ASL", "fail") == "all"
    member = set((SPLIT / f"{split}.txt").read_text().split())
    assert not set((SPLIT / "train.txt").read_text().split()) & set((SPLIT / "held.txt").read_text().split()), "train and held overlap"
    lists.mkdir(parents=True, exist_ok=True)
    for n, src in LISTS[split].items():
        f = lists / f"{n}.txt"
        assert not f.exists() or f.read_text() == src.read_text(), f"{f} exists with other scenes"
        f.write_text(src.read_text())
    py = str(Path(os.environ.get("ALPASIM_SRC") or B.DATA / "third_party/alpasim") / ".venv/bin/python")

    class Job(L.Job):
        def __init__(self, label, name, drv, env, scenes, need):
            n = Path(scenes).stem
            assert n in LISTS[split], f"{name}: list {n} is not a list of stage split `{split}` ({sorted(LISTS[split])})"
            super().__init__(label, name, drv, env, lists / f"{n}.txt", need)
            sc = self.scenes.read_text().split()
            assert len(sc) == len(set(sc)) and set(sc) <= member, f"{name}: scenes outside alpasim_nuplan/onpol-{split}"
            assert split != "held" or set(sc) == member, f"{name}: the held stage runs on exactly results/split/held.txt"

        def finish(self):
            kept = []
            if (self.D / "aggregate/results-summary.json").exists():           # before the pruning in ot2_loop.Job.finish
                f = self.D / "read_sim.pkl"
                r = subprocess.run([py, str(ALP / "c1_extract.py"), "logs", "--run", str(self.D), "--out", str(f), "--jobs", str(max(4, min(32, n_cpus() // 4)))],
                                   cwd=B.REPO, capture_output=True, text=True)
                if r.returncode or not f.exists():
                    B.fail(f"{self.name}: simulator-state extraction failed in {self.D} (rollout logs left as they are): {r.stderr[-400:]}")
                B.log(f"{self.name}: {r.stdout.strip()[-200:]}")
                if keep_all:
                    kept = glob.glob(str(self.D / "rollouts/*/*/rollout.asl"))
                    for a in kept:
                        os.link(a, a + ".keep")
            try:
                return super().finish()
            finally:
                for a in kept:
                    os.unlink(a + ".keep") if os.path.exists(a) else os.rename(a + ".keep", a)

    L.Job = Job
    L.main()


if __name__ == "__main__":
    main()
