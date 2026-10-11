#!/usr/bin/env python3
"""ONPOL-NUPLAN rollout loop: experiments/alpasim/scripts/ot2_loop.py, imported unedited, with three additions for on-policy rows.

  1. scene lists come from the lane's own directory ($ON_LISTS), written here from results/split/train.txt: on_smoke1 (1 scene), on_smoke10
     (every 104th scene of the sorted train list, 10 scenes), on_c0 .. on_c3 (the train list in four interleaved quarters, one pool job each)
  2. no job is built from a list with a scene outside `alpasim_nuplan/onpol-train` (the prereg: no rollout of a held scene)
  3. before ot2_loop prunes rollout.asl (it keeps the log only for failed rollouts), every finished run is read with
     experiments/alpasim/scripts/c1_extract.py `logs` (unedited, AlpaSim's venv) into <run dir>/onpol_sim.pkl: per scene the simulator's ego and
     actor poses at every simulation step, actor sizes, the logged object trajectories, the driver requests and returns, the metric series
     and the summary row. A run whose extraction fails stops the chain with its logs untouched.

Job specs, state files, retries and the stall watchdog are ot2_loop's (its docstring). Called by on_roll.sh:
  ON_LISTS=<dir> OT_LANE=onpol OT_OUT=$DATA_DIR/runs/onpol_nuplan/roll OT_OWNER=alpasim-onpol python3 on_loop.py <stage> <job> [<job> ...]
  python3 on_loop.py --lists <dir>          only writes the lists (stdlib only)
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ALP = REPO / "experiments/alpasim/scripts"
sys.path[:0] = [str(ALP), str(REPO)]
SPLIT = REPO / "experiments/onpol_nuplan/results/split"
N_CHUNK, SMOKE_STEP = 4, 104


def write_lists(d: Path) -> dict:
    """The lane's scene lists (module docstring) -> {name: scenes}. An existing file must already hold the same scenes."""
    tr = (SPLIT / "train.txt").read_text().split()
    assert tr == sorted(set(tr)) and not set(tr) & set((SPLIT / "held.txt").read_text().split())
    smoke = tr[::SMOKE_STEP][:10]
    L = {"on_smoke1": smoke[:1], "on_smoke10": smoke} | {f"on_c{i}": tr[i::N_CHUNK] for i in range(N_CHUNK)}
    d.mkdir(parents=True, exist_ok=True)
    for n, v in L.items():
        f, txt = d / f"{n}.txt", "\n".join(v) + "\n"
        assert not f.exists() or f.read_text() == txt, f"{f} exists with other scenes"
        f.write_text(txt)
    return L


def main():
    if sys.argv[1] == "--lists":
        print({k: len(v) for k, v in write_lists(Path(sys.argv[2])).items()})
        return
    import ot2_loop as L
    from jevdrive.common import n_cpus
    B = L.B
    lists = Path(os.environ["ON_LISTS"])
    write_lists(lists)
    train = set((SPLIT / "train.txt").read_text().split())
    py = str(Path(os.environ.get("ALPASIM_SRC") or B.DATA / "third_party/alpasim") / ".venv/bin/python")

    class Job(L.Job):
        def __init__(self, label, name, drv, env, scenes, need):
            mine = lists / Path(scenes).name
            assert mine.exists(), f"{mine}: on-policy rollouts run on the lane's own lists only"
            super().__init__(label, name, drv, env, mine, need)
            out = set(self.scenes.read_text().split()) - train
            assert not out, f"{self.name}: {len(out)} scenes outside alpasim_nuplan/onpol-train, e.g. {sorted(out)[:2]}"

        def finish(self):
            if (self.D / "aggregate/results-summary.json").exists():           # before the pruning in ot2_loop.Job.finish
                f = self.D / "onpol_sim.pkl"
                r = subprocess.run([py, str(ALP / "c1_extract.py"), "logs", "--run", str(self.D), "--out", str(f), "--jobs", str(max(4, min(32, n_cpus() // 4)))],
                                   cwd=B.REPO, capture_output=True, text=True)
                if r.returncode or not f.exists():
                    B.fail(f"{self.name}: simulator-state extraction failed in {self.D} (rollout logs left as they are): {r.stderr[-400:]}")
                B.log(f"{self.name}: {r.stdout.strip()[-200:]}")
            return super().finish()

    L.Job = Job
    L.main()


if __name__ == "__main__":
    main()
