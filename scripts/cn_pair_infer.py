#!/usr/bin/env python
"""ControlNet pair pilot: scripts/cosmos_infer.py (the Cosmos v2 pilot's runner, imported read-only) with its run log
redirected to runs/cn_pair/infer/ instead of runs/cosmos/infer/. Same arguments as cosmos_infer.py."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jevdrive.runlog as RL  # noqa: E402

_Orig = RL.RunLog
RL.RunLog = lambda *parts: _Orig("cn_pair", *parts[1:])
spec = importlib.util.spec_from_file_location("cosmos_infer", Path(__file__).with_name("cosmos_infer.py"))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
ci.main()
