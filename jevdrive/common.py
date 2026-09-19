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


def get_logger(name: str = "jevdrive") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)
