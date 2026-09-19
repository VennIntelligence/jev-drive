"""Paths (from env) and logging shared by all steps."""
import logging
import os
from pathlib import Path

CLASSES = ("left", "straight", "right")


def data_dir() -> Path:
    return Path(os.environ["DATA_DIR"])


def dataroot() -> Path:
    return data_dir() / "datasets" / "nuscenes"


def processed_dir(version: str) -> Path:
    d = data_dir() / "processed" / "nuscenes" / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def n_cpus() -> int:
    """Cores this process may use: the container's cgroup CPU quota if set (the box: 16 of the host's 128),
    else the affinity mask. os.cpu_count() reports the host and oversubscribes."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def get_logger(name: str = "jevdrive") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)
