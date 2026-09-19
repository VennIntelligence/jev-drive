"""Run directory with the three standard outputs of every long job (docs/long-runs.md):

  log.txt        human-readable log (same lines as the terminal; tqdm bars stay on the terminal only)
  events.jsonl   machine-readable stream, one JSON object per line, flushed per line: {"t", "kind", ...}
  tb/            TensorBoard scalars, for curves whose values mean something (loss, metric vs layer, throughput)
"""
import json
import logging
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .common import data_dir, get_logger


class RunLog:
    def __init__(self, *parts: str):
        self.dir = data_dir() / "runs" / Path(*parts) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True)
        self.log = get_logger(parts[0])
        fh = logging.FileHandler(self.dir / "log.txt")
        fh.setFormatter(logging.getLogger().handlers[0].formatter)
        logging.getLogger().addHandler(fh)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self.tb = SummaryWriter(self.dir / "tb", flush_secs=10)

    def event(self, kind: str, **fields):
        self._events.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}, default=float) + "\n")

    def scalar(self, tag: str, value: float, step: int):
        self.tb.add_scalar(tag, value, step)
        self.event("scalar", tag=tag, value=value, step=step)

    def close(self):
        self.tb.close()
        self._events.close()
