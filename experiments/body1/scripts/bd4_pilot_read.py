"""BODY1 arm 4.3: per-term / per-family loss table of a bd4_train.py run and the pilot gate's loss half (prereg Amendment 4 item 5, note vii).

Reads the run's events.jsonl (25-step means written by bd4_train.py). Table: every loss scalar averaged over the 100 steps ending at --at.
Gate: the agent hinge on own-plan positives (`loss/agent_posmean`, hinge-only and imitation rows pooled, weighted by the number of steps that
had a positive row) over steps 2 701-3 000 against steps 201-400; pass = a fall of at least 30 %.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd4_pilot_read.py --tag P2H10B-P-s0 [--ref P2H10-P-s0]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

OUT = B.REPO / "experiments/body1/results/loss"


def events(tag):
    import pandas as pd
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs/op_parity" / f"train-{tag}").iterdir())[-1]
    rows = [json.loads(ln) for ln in open(d / "events.jsonl")]
    E = pd.DataFrame([r for r in rows if r.get("kind") == "scalar"])
    return E[E.tag.str.startswith(("loss/", "dev/"))].pivot_table(index="step", columns="tag", values="value")


def window(E, col, lo, hi):
    """Mean of a `_posmean` column over steps (lo, hi], weighted by its `_posn`."""
    w = E.loc[(E.index > lo) & (E.index <= hi), [col, col.replace("_posmean", "_posn")]].dropna()
    return float(np.average(w.iloc[:, 0], weights=w.iloc[:, 1])) if len(w) and w.iloc[:, 1].sum() > 0 else float("nan"), int(w.iloc[:, 1].sum()) if len(w) else 0


def main(a):
    import pandas as pd
    from jevdrive.run import Run
    with Run("body1", f"pilot-read-{a.tag}", config=vars(a)) as run:
        E = events(a.tag)
        T = pd.DataFrame({s: E[(E.index > s - 100) & (E.index <= s)].mean() for s in a.at}).filter(regex="^loss/", axis=0)
        T.index = T.index.str.removeprefix("loss/")
        if a.ref:
            R = events(a.ref)
            for s in a.at:
                T[f"{a.ref} @{s}"] = R[(R.index > s - 100) & (R.index <= s)].mean().filter(regex="^loss/").rename(lambda q: q.removeprefix("loss/"))
        OUT.mkdir(parents=True, exist_ok=True)
        T.to_csv(OUT / f"pilot_terms_{a.tag}.csv", float_format="%.6f")
        gate = {}
        for col in [c for c in E.columns if c.endswith("_posmean")]:
            (e, ne), (l, nl) = window(E, col, 200, 400), window(E, col, a.steps - 300, a.steps)
            gate[col.removeprefix("loss/")] = dict(early=e, early_steps=ne, late=l, late_steps=nl, fall=(e - l) / e if e > 0 else float("nan"))
        g = gate.get("agent_posmean", {})
        res = dict(tag=a.tag, gate_of="Amendment 4 item 5, loss half (Amendment 5 replaces it by hold-log rates: bd4_g3.py)", windows=[[201, 400], [a.steps - 299, a.steps]], terms=gate, gate_term="agent_posmean", fall=g.get("fall"), passed=bool(g.get("fall", 0) >= 0.30),
                   dev={c.removeprefix("dev/"): float(E[c].dropna().iloc[-1]) for c in E.columns if c.startswith("dev/")})
        (OUT / f"pilot_gate_{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
        run.info("\n" + T.to_string(float_format=lambda x: f"{x:.5f}"))
        run.info(json.dumps(res, indent=1))
        run.summary.update(fall=res["fall"], passed=res["passed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--at", type=int, nargs="+", default=[300, 1000, 2000, 3000])
    main(ap.parse_args())
