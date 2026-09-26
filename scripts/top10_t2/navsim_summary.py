"""Summarise a NAVSIM reproduction run (scripts/top10_t2/navsim_repro.sh) into one small CSV row per metric.

The score is the devkit's own average row (v1.1: `average`, v2 one-stage: `average_all_frames`), as shipped;
sub-scores are the same row, x100. Writes research/results/top10-exams/navsim_<model>.csv.

  python scripts/top10_t2/navsim_summary.py drivor  <run dir>
  python scripts/top10_t2/navsim_summary.py wajepa  <run dir>
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAPER = {("drivor", "PDMS v1.1"): 93.7, ("wajepa", "EPDMS v2"): 91.7, ("wajepa", "PDMS v1.1"): 91.8}
DEVKIT = {"PDMS v1.1": "navsim v1.1", "EPDMS v2": "navsim main @0a380a9 (navsim/ tree identical to tag v2.2)"}
CACHE = {"PDMS v1.1": "runs/navsim/metric_cache/v1_navtest", "EPDMS v2": "runs/navsim/metric_cache/v2_navtest"}


def row(model: str, metric: str, csv: Path, run: Path, devkit: str) -> dict:
    d = pd.read_csv(csv)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    avg = d[d.token.astype(str).str.startswith("average")].iloc[-1]
    tok = d[~d.token.astype(str).str.startswith("average")]
    subs = [c for c in d.columns if c not in ("token", "valid", "score")]
    out = {"model": model, "metric": metric, "score": round(100 * float(avg.score), 2), "paper": PAPER.get((model, metric)),
           "n_tokens": len(tok), "n_valid": int(tok.valid.astype(bool).sum()), "devkit": devkit, "metric_cache": CACHE[metric],
           **{c: round(100 * float(avg[c]), 2) for c in subs}, "run_dir": str(run), "csv": csv.name}
    out["gap"] = None if out["paper"] is None else round(out["score"] - out["paper"], 2)
    return out


def main():
    model, run = sys.argv[1], Path(sys.argv[2])
    if model == "drivor":
        rows = [row(model, "PDMS v1.1", sorted(run.glob("*.csv"))[-1], run, "DrivoR repo (navsim 1.1.0 fork, fc6e5aa)")]
    else:
        rows = [row(model, "EPDMS v2", sorted((run / "v2").glob("*.csv"))[-1], run, DEVKIT["EPDMS v2"])]
        v1 = sorted((run / "v1_1").glob("*.csv"))
        if v1:
            rows.append(row(model, "PDMS v1.1", v1[-1], run, DEVKIT["PDMS v1.1"]))
    out = REPO / "research" / "results" / "top10-exams" / f"navsim_{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
