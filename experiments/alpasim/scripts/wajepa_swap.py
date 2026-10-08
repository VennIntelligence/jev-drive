"""Where does the online plan differ from the offline one? Decision 3 of a WA-JEPA AlpaSim run (= the navtest token's t0) re-planned offline
with every combination of the three input groups taken from the simulator (S) or from NAVSIM (N):
  images   4 history frames x [L0, F0, R0, B0]: rendered frames of the run's dump / the real JPEGs of the token
  state    4 history poses, t0 velocity and acceleration: what AlpaSim fed at decision 3 / the NAVSIM index
  command  one-hot: the route rule / NAVSIM driving_command
Reports the mean distance of the 8 poses to the logged future and to the all-NAVSIM plan, per arm. Needs a run with WAJ_DUMP >= the scene count.

  cd $DATA_DIR/third_party/wajepa && PYTHONPATH=$PWD:$DATA_DIR/third_party/navsim $DATA_DIR/envs/wajepa/bin/python \
      <repo>/experiments/alpasim/scripts/wajepa_swap.py --run <run dir> --out <json>
"""
import argparse
import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import wajepa_core as C  # noqa: E402


def main(a):
    D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
    run = Path(a.run)
    tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
    row = {t: k for k, t in enumerate(tab["names"].tolist())}
    drive = {}
    for x in map(json.loads, (run / "driver-logs/drive.jsonl").read_text().splitlines()):
        if x["kind"] == "drive" and x["k"] == 3:
            drive[x["session"]] = x
    core, root, logs, rows = C.Core("cuda"), D / "datasets/navsim/sensor_blobs/test", {}, []
    ade = lambda x, y: float(np.linalg.norm(x[:, :2] - y[:, :2], axis=1).mean())  # noqa: E731
    for f in sorted((run / "driver-logs/dump").glob("s*_k3.npz")):
        z = np.load(f)
        x = next(v for v in drive.values() if v["scene"] == str(z["scene"]))
        tok = x["scene"].rsplit("-", 1)[1]
        if tok not in row:
            continue
        r, lg = row[tok], str(tab["log"][row[tok]])
        if lg not in logs:
            fr = pickle.load(open(D / "datasets/navsim/navsim_logs/test" / f"{lg}.pkl", "rb"))
            logs = {lg: (fr, {q["token"]: i for i, q in enumerate(fr)})}
        fr, at = logs[lg]
        i = at[tok]
        img = {"S": [{c: z[f"{c}_{j}"] for c in C.CAMS} for j in range(4)],
               "N": [{c: C.decode((root / fr[j]["cams"][c]["data_path"]).read_bytes()) for c in C.CAMS} for j in range(i - 3, i + 1)]}
        ego = np.array(x["ego"])
        st = {"S": (np.array(x["hist"]), np.tile(ego[:2], (4, 1)), ego[2:]),
              "N": (tab["pose"][r].astype(np.float64), tab["vel"][r].astype(np.float64), tab["acc"][r][-1])}
        cm = {"S": np.eye(4)[x["cmd"]], "N": tab["cmd"][r][-1]}
        out = {}
        for k in itertools.product("SN", repeat=3):
            p, v, ac = st[k[1]]
            out["".join(k)] = core.plan(img[k[0]], p, v, ac, cm[k[2]])["poses"]
        fut = tab["fut"][r]
        rows.append({"token": tok, **{f"{k}_log": ade(p, fut) for k, p in out.items()}, **{f"{k}_nnn": ade(p, out["NNN"]) for k, p in out.items()},
                     "cmd_S": int(x["cmd"]), "cmd_N": int(np.argmax(tab["cmd"][r][-1]))})
        print(len(rows), tok, {k: round(v, 2) for k, v in rows[-1].items() if k.endswith("_log")}, flush=True)
    S = {k: float(np.mean([q[k] for q in rows])) for k in rows[0] if k.endswith(("_log", "_nnn"))}
    S["n"], S["n_cmd_differs"] = len(rows), sum(q["cmd_S"] != q["cmd_N"] for q in rows)
    S["same_cmd"] = {k: float(np.mean([q[k] for q in rows if q["cmd_S"] == q["cmd_N"]])) for k in rows[0] if k.endswith(("_log", "_nnn"))}
    Path(a.out).write_text(json.dumps({"summary": S, "rows": rows}, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
