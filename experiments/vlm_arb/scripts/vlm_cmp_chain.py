"""The vlm_cmp chain: Qwen3-VL-4B against 8B as the slow-channel model, offline, one resumable script (jevdrive.run.Run:
STATUS / DONE / ERROR in its run dir).

  .venv/bin/python experiments/vlm_arb/scripts/vlm_cmp_chain.py [--resume RUN_DIR] [--limit K] [--tag T] [--cards 0] [--stages a,b]

Stages, in order (a stage with `stages/<name>.done` is skipped on a rerun; GPU workers are one process per model):
  frames    every framed request with its truth, the directive truth and the core set (frames.csv), the registered splits
  selftest  cache path == full forward on a few frames (both models)
  extract   single-frame option log-probs of five questions + answer-position states + pooled image tokens, 4 resolutions
  twoframe  previous + current frame (4 images), directive / light / block, r1153
  fit       linear heads per (model, resolution), grouped CV
  bench     batch-1 latency and peak memory per variant, one process at a time on the first card (nothing else running)
  report    tables -> experiments/vlm_arb/results/vlm_4b_vs_8b.md
--cards: the two models run at the same time in selftest / extract / twoframe (one card each, or sharing the only card); bench runs alone on the first card.
--limit K keeps about K core frames (end-to-end smoke test).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import DATA, REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402
from vlm_cmp_frames import SPLIT, load_all, select  # noqa: E402

PY = str(REPO / ".venv/bin/python")
SCR = Path(__file__).resolve().parent
STAGES = ["frames", "selftest", "extract", "twoframe", "fit", "bench", "report"]
RESN = ["r4573", "r2335", "r1153", "r559"]
KEEP = list(range(2, 37, 2))


class Chain:
    def __init__(self, run, a):
        self.run, self.a, self.d = run, a, run.dir
        self.cards = [int(c) for c in a.cards.split(",")]
        run.info("cards %s", self.cards)

    def popen(self, script, args, card, name):
        (self.d / "workers").mkdir(exist_ok=True)
        log = open(self.d / "workers" / ("%s.log" % name), "a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        return subprocess.Popen([PY, str(SCR / script), *args], cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT), name

    def wait(self, procs):
        bad = []
        while procs:
            time.sleep(5)
            for item in list(procs):
                p, name = item
                if p.poll() is not None:
                    procs.remove(item)
                    if p.returncode:
                        bad.append(name)
        if bad:
            tails = ""
            for n in bad:
                tails += "\n--- %s.log (tail) ---\n%s" % (n, "".join(open(self.d / "workers" / (n + ".log")).readlines()[-40:]))
            raise RuntimeError("worker(s) %s failed%s" % (bad, tails))

    def per_model(self, script, cmd, extra=()):
        """cmd for both models at the same time (card i % n_cards: with one card they share it; the GPU stages here are partly
        launch bound, so the two overlap; latency is measured later, alone)."""
        self.wait([self.popen(script, [cmd, "--run", str(self.d), "--model", m, *extra], self.cards[i % len(self.cards)], "%s-%s" % (cmd, m))
                   for i, m in enumerate(["4b", "8b"])])

    # ------------------------------------------------------------------------------ stages
    def frames(self):
        df = select(load_all())
        for part in SPLIT:
            s = splits.load("b2d/vlm-thin-" + part)
            self.run.use_split(s)
            assert set(s.members) == set(SPLIT[part]), part
        splits.check_disjoint(*[splits.load("b2d/vlm-thin-" + p) for p in SPLIT])
        assert not (df.part == "none").any()
        if self.a.limit:
            df = df.iloc[:: max(1, len(df) // self.a.limit)].reset_index(drop=True)
        df.to_csv(self.d / "frames.csv", index=False)
        self.run.info("core set: %d frames (%s), two-frame instants %d, routes %d", len(df), df.part.value_counts().to_dict(),
                      int(df.tf.sum()), df.route.nunique())
        self.run.summary["frames"] = int(len(df))

    def selftest(self):
        self.per_model("vlm_cmp_stage.py", "selftest")

    def extract(self):
        self.per_model("vlm_cmp_stage.py", "extract")

    def twoframe(self):
        self.per_model("vlm_cmp_stage.py", "twoframe", ["--res", "r1153"])

    def fit(self):
        """Heads on the CPU, one process per (model, resolution), each with a pool of every core; one after the other."""
        for m in ("4b", "8b"):
            for r in RESN:
                self.wait([self.popen("vlm_cmp_fit.py", ["fit", "--run", str(self.d), "--model", m, "--res", r] + (["--dir-too"] if r == "r1153" else []),
                                      self.cards[0], "fit-%s-%s" % (m, r))])

    def bench(self):
        full = ["q1_" + r for r in RESN] + ["dir_" + r for r in RESN] + ["four_r559", "four_r1153", "two_r559", "two_r1153"]
        cuts = ["cut_%s_%d" % (r, n) for r in ("r1153", "r559") for n in KEEP]
        for m in ("4b", "8b"):
            self.wait([self.popen("vlm_cmp_stage.py", ["bench", "--run", str(self.d), "--model", m, "--group", ",".join(full + cuts)],
                                  self.cards[0], "bench-" + m)])

    def report(self):
        r = subprocess.run([PY, str(SCR / "vlm_cmp_report.py"), "--run", str(self.d)], cwd=str(REPO), capture_output=True, text=True)
        (self.d / "workers").mkdir(exist_ok=True)
        (self.d / "workers" / "report.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError("report failed:\n" + (r.stdout + r.stderr)[-3000:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cards", default="0")
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    with Run("vlm_cmp", a.tag, root=DATA / "runs", resume=a.resume, config=vars(a), seed=0) as run:
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
