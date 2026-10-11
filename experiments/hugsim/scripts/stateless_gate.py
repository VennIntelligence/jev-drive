#!/usr/bin/env python
"""Gate of the stateless HUGSIM serving mode (agent opt `stateless`): the recorded decisions of the frame A/B run, fed step by step through the
serving code (lib/parity_hugsim.Stateless -> scripts/stateless_server.py over the wire), must reproduce the offline arm `b` of
frame_ab_torch.py. experiments/hugsim/plans/2026-10-11-stateless-cl-prereg.md, results/stateless_cl.md.

Per decision: max |position difference| over the plan points inside 4 s (real clock), against `b` (the gate), against `wd` (the wrong-arm
control: the same lattice code on the dilated clock) and against the logged served plan; tolerance 0.03 m (docs/long-runs.md fp16 rule: share
of rows over tolerance). `b` was computed with lht = False, so the gate pass uses lht = False; left-hand-traffic scenes are replayed a second
time with their own flag (what the closed loop serves) and reported, not gated. Also the per-decision latency of a single stream.
Writes <out>/gate.json, <out>/gate_rows.{csv,md}.

  .venv/bin/python experiments/hugsim/scripts/stateless_gate.py --run-dir <recorded bench run dir> --ab-out <frame_ab out dir> --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/hugsim/scripts")]
import argparse, json, os, subprocess, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import parity_hugsim as PH  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from frame_ab_onnx import load  # noqa: E402
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.openpilot.interface import HUGSIM_HEIGHT  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

TOL, T_MAX = 0.03, 4.0
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])


def replay(sock_path, S, lht):
    sock = wire.connect_retry(sock_path)
    wire.send(sock, {"cmd": "reset"}, {})
    assert wire.recv(sock)[0]["server"]["server"] == "stateless"
    sl = PH.Stateless(sock, float(S["setup"]["rear_offset"]), HUGSIM_HEIGHT.get(S["setup"]["dataset"], 1.5), lht=lht)
    N = len(S["img2"])
    pos, act, ms, nk = np.zeros((N, 33, 3), np.float32), np.zeros((N, 2), np.float32), np.zeros((N, 3)), np.zeros(N, int)
    for n in range(N):
        r, out = sl.step(S["img2"][n], S["state"][n])
        pos[n], act[n], ms[n], nk[n] = out["pos"], out["act"], (r["infer_ms"], r["rtt_ms"], r["ms"]["frames"]), r["nkeys"]
    sock.close()
    return pos, act, ms, nk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ab-out", required=True)
    ap.add_argument("--tag", default="P2H10-F-s0")
    ap.add_argument("--out", required=True)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("hugsim_stateless_cl", "gate", seed=a.seed, config=vars(a), resume=a.resume) as run:
        sp, ready = f"/tmp/slgate-{os.getpid()}.sock", out / "gate-server.ready"
        ready.unlink(missing_ok=True)
        with open(out / "gate-server.log", "w") as lf:
            srv = subprocess.Popen([str(data_dir() / "envs/op-train/bin/python"), "-u", str(_R / "experiments/hugsim/scripts/stateless_server.py"),
                                    "--tag", a.tag, "--socket", sp, "--ready-file", str(ready)], cwd=_R, stdout=lf, stderr=subprocess.STDOUT)
        try:
            while not ready.exists():
                if srv.poll() is not None:
                    raise RuntimeError(f"stateless server died at start: {out / 'gate-server.log'}")
                time.sleep(2)
            near = T_IDXS <= T_MAX + 1e-9
            rows, lat = [], []
            for d in run.tqdm(sorted(Path(a.run_dir).glob("bench/zs/*/zs_frames")), desc="scenarios"):
                S, name = load(d), d.parent.name
                zt = np.load(Path(a.ab_out) / "torch" / f"{name}.npz")
                assert np.array_equal(zt["state"], S["state"]), name
                scene_lht = list(S["setup"]["traffic"]) == [0, 1]
                for mode, lht in [("gate", False)] + ([("scene_lht", True)] if scene_lht else []):
                    pos, act, ms, nk = replay(sp, S, lht)
                    dm = lambda ref: np.abs(pos[:, near, :2] - ref[:, near, :2]).max((1, 2))  # noqa: E731
                    db, dw, dl = dm(zt["pos_b"]), dm(zt["pos_wd"]), dm(S["pos"])
                    rows.append(dict(scenario=name, mode=mode, n=len(db), nkeys_equal=bool(np.array_equal(nk, zt["nkeys_b"])),
                                     over_b=int((db > TOL).sum()), max_b_m=float(db.max()), median_b_m=float(np.median(db)),
                                     act_max_diff=float(np.abs(act - zt["act_b"]).max()),
                                     over_wd=int((dw > TOL).sum()), median_wd_m=float(np.median(dw)), over_log=int((dl > TOL).sum()), median_log_m=float(np.median(dl))))
                    if mode == "gate":
                        lat.append(ms)
                run.info("%s: %s", name, json.dumps([r for r in rows if r["scenario"] == name]))
        finally:
            srv.terminate()                                    # by pid: the server this script started
            srv.wait(30)
        g = [r for r in rows if r["mode"] == "gate"]
        n = sum(r["n"] for r in g)
        lat = np.concatenate(lat)
        q = lambda x: {k: float(np.quantile(x, p)) for k, p in (("median", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))}  # noqa: E731
        G = dict(tol_m=TOL, t_max_s=T_MAX, decisions=n, scenarios=len(g), share_over_b=sum(r["over_b"] for r in g) / n, max_b_m=max(r["max_b_m"] for r in g),
                 act_max_diff=max(r["act_max_diff"] for r in g), nkeys_equal=all(r["nkeys_equal"] for r in g),
                 control_share_over_wd=sum(r["over_wd"] for r in g) / n, share_over_log=sum(r["over_log"] for r in g) / n,
                 scene_lht=[dict(scenario=r["scenario"], n=r["n"], over_b=r["over_b"], max_b_m=r["max_b_m"], median_b_m=r["median_b_m"]) for r in rows if r["mode"] == "scene_lht"],
                 latency_ms=dict(server_infer=q(lat[:, 0]), client_rtt=q(lat[:, 1]), lattice_frames=q(lat[:, 2])))
        G["passed"] = bool(G["share_over_b"] <= 0.01 and G["control_share_over_wd"] >= 0.5 and G["nkeys_equal"])
        (out / "gate.json").write_text(json.dumps(G, indent=1))
        stats.write_table(rows, out / "gate_rows")
        run.summary.update(gate=G["passed"], share_over_b=G["share_over_b"], control_share_over_wd=G["control_share_over_wd"])
        run.info("gate %s: %s", "PASS" if G["passed"] else "FAIL", json.dumps({k: v for k, v in G.items() if k != "scene_lht"}))
        if not G["passed"]:
            raise SystemExit("stateless gate failed")


if __name__ == "__main__":
    main()
