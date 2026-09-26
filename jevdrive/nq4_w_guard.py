"""Serialize full W seed runs and reuse only validated complete outputs."""
import fcntl
import json
import os
from pathlib import Path

OUTPUTS = ("pair_scores.parquet", "action_scores.parquet", "pairs.parquet", "train_curves.json")


def signature_for(cfg, steps, data_paths):
    return {"cfg": cfg, "steps": steps, "data": [
        {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in map(Path, data_paths)]}


def valid_outputs(directory, steps=None):
    directory = Path(directory)
    try:
        if not all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in OUTPUTS):
            return False
        curves = json.loads((directory / "train_curves.json").read_text())
        return (len(curves) == 5 and {row["fold"] for row in curves} == set(range(5))
                and (steps is None or all(row["curve"][-1]["step"] == steps for row in curves)))
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return False


def guarded_seed(root, seed, signature, worker):
    root = Path(root)
    guard = root / "seed-locks" / f"seed{seed}"
    guard.mkdir(parents=True, exist_ok=True)
    record = guard / "complete.json"
    with (guard / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if record.exists():
            saved = json.loads(record.read_text())
            if saved["seed"] != seed:
                raise RuntimeError(f"W seed {seed} completion seed differs")
            if saved["signature"] != signature:
                raise RuntimeError(f"W seed {seed} completion configuration differs")
            if valid_outputs(saved["result"]["dir"], signature["steps"]):
                return {**saved["result"], "reused": True}
            raise RuntimeError(f"W seed {seed} completion outputs are invalid")
        result = worker()
        if not valid_outputs(result["dir"], signature["steps"]):
            raise RuntimeError(f"W seed {seed} has incomplete outputs")
        temporary = guard / f"complete.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({"seed": seed, "signature": signature, "result": result}, indent=2) + "\n")
        temporary.replace(record)
        return result
