"""Run the repo's src/alpamayo1_5/test_inference.py main() as shipped, with one wrapper-level change.

The clip is streamed from HF exactly as shipped. The only change: the hub goes offline right after the clip is
loaded, so the model load reads nvidia/Cosmos-Reason2-8B's tokenizer/config from the local cache (fetched from
ModelScope) instead of hitting its HF gate, which is not accepted for our account (403 on an optional-file probe).
Run from the alpamayo1.5 repo root in its venv, with HF_ENDPOINT=https://hf-mirror.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpamayo1_5 import test_inference as T
from jevdrive.alpamayo.infer import hub_offline

_load = T.load_physical_aiavdataset


def load_then_offline(*a, **k):
    out = _load(*a, **k)
    hub_offline(True)
    return out


T.load_physical_aiavdataset = load_then_offline
T.main()
