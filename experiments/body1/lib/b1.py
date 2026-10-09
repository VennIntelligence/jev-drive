"""BODY1 shared paths and loaders (plans/2026-10-10-body1-prereg.md): state families, the memory-mapped label store, the log split rule.

State families (cache dirs under $DATA_DIR/runs/op_parity/cache/, 12 shards each, rows = tab.npz order):
  log   navtrain_full.s<k>of12            every navtrain token on the log
  ot1   ot1_navtrain_full.s<k>of12        +-0.5 m / +-2 deg plane-reprojected states (ot_rows.py; speed > 3 m/s, logged future)
  yr1   yr1_navtrain_full.s<k>of12        yaw-rate states (ot3_rows.py)
  bd1   bd1_navtrain_full.s<k>of12        +-1.0 m / +-4 deg (bd1_prep.py; only if built)
Global row = position of the token in the concatenated `log` family = row of the agent / SDF label files.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "lib", REPO / "scripts", REPO / "experiments/body1/lib", REPO / "experiments/op_parity/scripts", REPO / "experiments/op_adapt_r2/lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from jevdrive.common import data_dir  # noqa: E402

NSH = 12
FAMS = {"log": "", "ot1": "ot1_", "yr1": "yr1_", "bd1": "bd1_"}
MAN = ("launch", "stop", "turn 20-45", "turn >45", "go-around", "straight", "unknown")
HAZ = ("in-path", "side", "boundary", "none")
CLS = ("other", "1 obstacle ahead", "2 turn", "3 leaving the road")
HOLD, TRAIN = "navsim/body1-hold-logs", "navsim/body1-train-logs"
VAL = "navsim/body1-val-logs"                       # prereg Amendment 5: the validation part of the train logs (is_val)


def root() -> Path:
    return data_dir() / "runs" / "body1"


def cache_root() -> Path:
    return data_dir() / "runs" / "op_parity" / "cache"


def cdir(fam: str, k: int) -> str:
    return f"{FAMS[fam]}navtrain_full.s{k}of{NSH}"


def is_hold(log: str) -> bool:
    """body1-hold-logs: sha256(log) % 10 == 0 (a superset of op-parity-full-dev's % 50 == 0)."""
    return int(hashlib.sha256(log.encode()).hexdigest(), 16) % 10 == 0


def is_val(log: str) -> bool:
    """body1-val-logs: sha256(log) % 10 == 1, a part of body1-train-logs (prereg Amendment 5 item 3)."""
    return int(hashlib.sha256(log.encode()).hexdigest(), 16) % 10 == 1


def tab(fam: str, k: int) -> dict:
    return dict(np.load(cache_root() / cdir(fam, k) / "tab.npz"))


def shard_offsets() -> np.ndarray:
    """(13,) first global row of every `log` shard."""
    f = root() / "labels" / "offsets.npy"
    if f.exists():
        return np.load(f)
    return np.cumsum([0] + [len(np.load(cache_root() / cdir("log", k) / "tab.npz")["names"]) for k in range(NSH)])


def state_index(fam: str, k: int, t: dict) -> tuple:
    """-> (global label row (n,), off (n, 2) = (dy, dpsi) of the state's t0 pose in the logged frame)."""
    n = len(t["names"])
    row = t["src_row"] if "src_row" in t else np.arange(n)
    return shard_offsets()[k] + row, (t["off"].astype(np.float32) if "off" in t else np.zeros((n, 2), np.float32))


def build_labels(run=None) -> None:
    """One-off: the two label npz files -> memory-mappable .npy under runs/body1/labels/ (workers share them through the page cache)."""
    d = root() / "labels"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE").exists():
        return
    a = np.load(data_dir() / "runs/op_parity/agent_labels/navtrain_all-k32.npz")
    s = np.load(data_dir() / "runs/op_probe/labels/navtrain_all.npz")
    toks = np.concatenate([np.load(cache_root() / cdir("log", k) / "tab.npz")["names"] for k in range(NSH)])
    assert (a["tokens"] == toks).all() and (s["tokens"] == toks).all() and a["ok"].all() and s["ok"].all(), "label files are not in cache order"
    assert float(s["x0"]) == -8.0 and float(s["y0"]) == -24.0 and float(s["res"]) == 0.5
    for name, x in (("box", a["box"]), ("valid", a["valid"]), ("cls", a["cls"]), ("sdf", s["sdf"]), ("tokens", toks), ("log", a["log"])):
        np.save(d / f"{name}.npy", x)
        if run:
            run.info(f"labels/{name}.npy {x.shape} {x.dtype}")
    np.save(d / "offsets.npy", shard_offsets())
    (d / "classes.txt").write_text("\n".join(a["classes"].tolist()) + "\n")
    (d / "DONE").write_text("ok\n")


def labels() -> dict:
    d = root() / "labels"
    assert (d / "DONE").exists(), "run bd1_tax.py labels first"
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("box", "valid", "cls", "sdf")} | {k: np.load(d / f"{k}.npy") for k in ("tokens", "log")}
