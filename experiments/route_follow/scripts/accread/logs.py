"""Box step 2 (third_party/alpasim/.venv): the human log (rig trajectory) and wait lines within 30 m for the same scenes -> /tmp/accread_logs.json."""
import json, glob, sys, os
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/ujs/jev-drive/experiments/route_follow/scripts")
import rf_tax as TX
O = "/root/autodl-tmp/ujs/runs/route_follow/pai_g10"
want = set(json.load(open("/tmp/accread.json")))
from trajdata.maps.vec_map_elements import MapElementType as M
out = {}
for sim in sorted(glob.glob(f"{O}/g10/runs/*/sim")):
    cfg = f"{sim}/generated-user-config-0.yaml"
    for r in json.load(open(f"{sim}/aggregate/results-summary.json"))["rollouts"]:
        s = r["clipgt_id"]
        if s not in want or s in out:
            continue
        ds = TX.loader(cfg).get_data_source(s)
        tr = ds.rig.trajectory
        P = np.asarray(tr.positions, float)[:, :2]
        wl = []
        for w in ds.map.elements[M.WAIT_LINE].values():
            q = np.asarray(w.polyline.points, float)[:, :2]
            if len(q) >= 2 and np.hypot(*(P[:, None] - q[None]).transpose(2, 0, 1)).min() < 30:
                wl.append(dict(kind=str(w.wait_line_type), pts=q.tolist()))
        out[s] = dict(T=np.asarray(tr.timestamps_us, float).tolist(), P=P.tolist(), Y=np.unwrap(np.asarray(tr.yaws, float)).tolist(), wl=wl)
        print(s, len(P), len(wl), flush=True)
json.dump(out, open("/tmp/accread_logs.json", "w"))
