"""The vlm_thin chain: one resumable script, STATUS / DONE / ERROR in its run dir (jevdrive.run.Run).

  .venv/bin/python experiments/vlm_arb/scripts/vlm_thin_chain.py [--resume RUN_DIR] [--limit K] [--tag T] [--stages a,b]

Stages, in order (a stage with `stages/<name>.done` is skipped on a rerun; GPU stages run one process per card):
  frames    frames.csv (every labelled request with frames, truth columns, split part) + the registered route splits
  selftest  preprocessing / truncation equivalence checks
  baseline  part 1 accuracy of the training-free variants on the 233 sweep frames
  extract   hidden-state features, 4 resolutions, sharded over the cards (jevdrive.cache)
  fit       heads per resolution on the cards (CPU-free), grouped CV
  gen       Phase A (a): full zero-shot generate on the test frames
  bench     batch-1 latency of every variant, one card each, nothing else running
  select    grid + the registered selection rule; then retrain the chosen head
  serve     Phase A (b): the chosen truncated variant on the test frames
  report    tables, figure, Phase A tables -> results/
--limit K keeps K frames per part (end-to-end smoke test); everything is then cached under another frame tag.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_thin_common import DATA, REPO, RES, SPLIT, CUTS, load_frames  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive.cl.box import probe  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402

PY = str(REPO / ".venv/bin/python")
SCR = Path(__file__).resolve().parent
STAGES = ["frames", "selftest", "baseline", "extract", "fit", "gen", "bench", "select", "serve", "report"]


class Chain:
    def __init__(self, run, a):
        self.run, self.a = run, a
        self.d = run.dir
        self.cards = [c.index for c in probe().cards]
        self.n = len(self.cards)
        run.info("cards %s", self.cards)

    def spawn(self, script, args_per_card, name):
        """One subprocess per entry, entry i on card cards[i % n]; waits for all, raises with the log tail on a failure."""
        (self.d / "workers").mkdir(exist_ok=True)
        procs = []
        for k, args in enumerate(args_per_card):
            gpu = self.cards[k % self.n]
            log = open(self.d / "workers" / ("%s-%d.log" % (name, k)), "a")
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            procs.append((k, subprocess.Popen([PY, str(SCR / script), *args], cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT), log))
        bad = []
        while procs:
            time.sleep(5)
            for item in list(procs):
                k, p, log = item
                if p.poll() is not None:
                    procs.remove(item)
                    log.close()
                    if p.returncode:
                        bad.append(k)
        if bad:
            tails = ""
            for k in bad:
                tails += "\n--- %s-%d.log (tail) ---\n%s" % (name, k, "".join(open(self.d / "workers" / ("%s-%d.log" % (name, k))).readlines()[-40:]))
            raise RuntimeError("%s: worker(s) %s failed%s" % (name, bad, tails))

    def sharded(self, script, cmd, extra=()):
        self.spawn(script, [[cmd, "--run", str(self.d), "--shard", "%d/%d" % (i, self.n), *extra] for i in range(self.n)], cmd)

    # -------------------------------------------------------------------------------- stages
    def frames(self):
        df = load_frames()
        for part in SPLIT:                                          # the registered splits are what the code assigned
            s = splits.load("b2d/vlm-thin-" + part)
            self.run.use_split(s)
            assert set(s.members) == set(SPLIT[part]), part
        splits.check_disjoint(*[splits.load("b2d/vlm-thin-" + p) for p in SPLIT])
        assert not (df.part == "none").any(), "frames outside the three splits"
        if self.a.limit:
            keep = []
            for part, g in df.groupby("part"):
                keep += list(g[g["sweep"]].index[: self.a.limit // 2]) + list(g[~g["sweep"]].index[:: max(1, len(g) // self.a.limit)][: self.a.limit])
            df = df.loc[sorted(set(keep))].reset_index(drop=True)
        df.to_csv(self.d / "frames.csv", index=False)
        self.run.info("frames: %d (%s), sweep %d, labelled %d", len(df), df.part.value_counts().to_dict(), df["sweep"].sum(), (df.y >= 0).sum())
        self.run.summary["frames"] = {k: int(v) for k, v in df.part.value_counts().items()}

    def selftest(self):
        self.spawn("vlm_thin_stage.py", [["selftest", "--run", str(self.d)]], "selftest")

    def baseline(self):
        self.sharded("vlm_thin_stage.py", "baseline")

    def extract(self):
        self.sharded("vlm_thin_stage.py", "extract")

    def fit(self):
        self.spawn("vlm_thin_fit.py", [["fit", "--run", str(self.d), "--res", r] for r in RES], "fit")

    def gen(self):
        self.sharded("vlm_thin_stage.py", "gen")

    def bench(self):
        g0 = ["gen_ref", "gen_gpu"] + ["fwd_" + r for r in RES]
        g1 = ["cut_r4573_%d" % N for N in CUTS] + ["vis_r4573"]
        g2 = ["%s_%s" % (p, r) if p == "vis" else "cut_%s_%d" % (r, N) for r in list(RES)[1:] for p, N in [("vis", 0)] + [("cut", N) for N in CUTS]]
        groups = [g0, g1, g2]
        self.spawn("vlm_thin_stage.py", [["bench", "--run", str(self.d), "--group", ",".join(g)] for g in groups[:self.n]], "bench")
        self.spawn("vlm_thin_stage.py", [["bench", "--run", str(self.d), "--group", "fwd_r4573_compile"]], "bench-compile")

    def select(self):
        self.cpu("vlm_thin_report.py", "select")
        self.spawn("vlm_thin_fit.py", [["emit", "--run", str(self.d)]], "emit")

    def serve(self):
        self.sharded("vlm_thin_stage.py", "serve")

    def report(self):
        self.cpu("vlm_thin_report.py", "report")

    def cpu(self, script, cmd):
        r = subprocess.run([PY, str(SCR / script), cmd, "--run", str(self.d)], cwd=str(REPO), capture_output=True, text=True,
                           env=dict(__import__("os").environ, HF_HUB_OFFLINE="1"))
        (self.d / "workers").mkdir(exist_ok=True)
        (self.d / "workers" / ("%s-%s.log" % (script, cmd))).write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError("%s %s failed:\n%s" % (script, cmd, (r.stdout + r.stderr)[-3000:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    root = DATA / "runs"
    with Run("vlm_thin", a.tag, root=root, resume=a.resume, config=vars(a), seed=0) as run:
        ch = Chain(run, a)
        (ch.d / "stages").mkdir(exist_ok=True)
        for st in STAGES:
            if st not in a.stages.split(","):
                continue
            mark = ch.d / "stages" / (st + ".done")
            if mark.exists():
                run.info("stage %s already done", st)
                continue
            run.status("stage %s" % st)
            t = time.time()
            getattr(ch, st)()
            mark.write_text(time.strftime("%F %T") + "\n")
            run.event("stage", name=st, wall_s=round(time.time() - t, 1))
            run.info("stage %s done in %.0f s", st, time.time() - t)


if __name__ == "__main__":
    main()
