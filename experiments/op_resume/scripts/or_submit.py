#!/usr/bin/env python
"""Submit op_resume's closed-loop units to the GPU pool (plans/2026-10-05-op-resume-prereg.md section 4). Each unit is one pool job;
the pool picks the card. Two arms everywhere: `spec` (interface preset spec as it is) and `rule` (spec + the shared resume rule,
jevdrive/openpilot/resume.py defaults). Resumable: a unit whose outputs are complete is skipped, a unit still in the pool is reused.

  HUGSIM  scripts/or_hugsim.sh (resident Cinque + zs_run --preset spec), out $R/hugsim/<stage>/<arm>
  B2D     experiments/op_closed_loop/archive/op_arb.sh arm spec (lib/op_arb_agent.py, open-loop-aligned camera, zones on), the rule arm
          adds DRIVE_ARGS '"resume_rule": {}'; 3 routes (= CARLA workers) per unit; out $R/b2d/<stage>/<arm>-s<seed>-k<k>

  .venv/bin/python experiments/op_resume/scripts/or_submit.py pilot [--dry-run] [--wait]
  .venv/bin/python experiments/op_resume/scripts/or_submit.py full  [--dry-run] [--wait]
One preflight per batch: the first HUGSIM unit's smoke (one scene, rule arm) gates every unit of the batch.
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.cl import pool as P  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
R = DATA / "runs/op_resume"
ARMS = {"spec": None, "rule": {}}               # rule params: {} = the pre-registered defaults
STAGES = {
    "pilot": dict(hugsim=HERE / "pilot_hugsim.txt", b2d=["24944", "9196", "17280"], seeds=[2]),
    "full": dict(hugsim=REPO / "experiments/hugsim/scripts/derot_all64.txt",
                 b2d="16390 15612 15483 27043 15102 28147 24944 9196 17280 16529 16508 26153 26365".split(), seeds=[2, 3]),
}
SHARD = 3
OP_ARB = "experiments/op_closed_loop/archive/op_arb.sh"


def hugsim_unit(stage, arm):
    out = R / "hugsim" / stage / arm
    scen = STAGES[stage]["hugsim"]
    opts = json.dumps({} if ARMS[arm] is None else {"resume": ARMS[arm]})
    env = dict(GPU="{gpu}", OUT=str(out), SCEN=str(scen), TAG="or-%s" % arm, OPTS=opts, WORKERS="5")
    return dict(name="or-%s-hug-%s" % (stage, arm), cmd="bash %s" % shlex.quote(str(HERE / "or_hugsim.sh")), env=env,
                vram_gb=30.0, carla=0, cpu=5, out=out,
                done=lambda: (out / "results.csv").exists() and _hug_done(out, scen, "or-%s" % arm))


def _hug_done(out, scen, tag):
    import csv
    want = {Path(s).stem for s in Path(scen).read_text().split()}
    return want <= {r["scenario"] for r in csv.DictReader(open(out / "results.csv")) if r["tag"] == tag and r["end"] != "crash"}


def b2d_units(stage, arm):
    ids = STAGES[stage]["b2d"]
    n = -(-len(ids) // SHARD)
    units = []
    for seed in STAGES[stage]["seeds"]:
        for k in range(n):
            part = ids[k::n]
            out = R / "b2d" / stage / ("%s-s%d-k%d" % (arm, seed, k))
            env = dict(GPU="{gpu}", IDX0="{idx}", WORKERS="{carla}", CPUS="{cpus}", SEED=str(seed), OP_ARB_DIR="{job_dir}/op",
                       OP_ARB_ARMS=str(out.parent), SRV_NO_TWIN="1", OPENBLAS_CORETYPE="Haswell", OP_ARB_AGENT="lib/op_arb_agent.py",
                       DESIRE="true")
            if ARMS[arm] is not None:
                env["DRIVE_ARGS"] = '"resume_rule": %s' % json.dumps(ARMS[arm])
            check = " && ".join("test -f %s" % shlex.quote(str(out / "done" / (r + ".json"))) for r in part)
            cmd = "%s && %s" % (shlex.join(["bash", OP_ARB, "arm", "spec", ",".join(part), str(out)]), check)
            units.append(dict(name="or-%s-b2d-%s-s%d-k%d" % (stage, arm, seed, k), cmd=cmd, env=env, vram_gb=7.5 * len(part),
                              carla=len(part), cpu=12, out=out,
                              done=lambda out=out, part=tuple(part): all((out / "done" / (r + ".json")).exists() for r in part)))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    a = ap.parse_args()
    units = [hugsim_unit(a.stage, arm) for arm in ARMS] + [u for arm in ARMS for u in b2d_units(a.stage, arm)]
    log_root = R / "pool" / a.stage
    pf = None
    if not a.no_preflight and not a.dry_run and not all(u["done"]() for u in units):
        smoke = R / "hugsim" / "preflight"
        scen = smoke / "smoke.txt"
        smoke.mkdir(parents=True, exist_ok=True)
        scen.write_text(Path(STAGES["pilot"]["hugsim"]).read_text().split()[0] + "\n")    # one stuck scene (0051-easy is first: short)
        env = dict(GPU="{gpu}", OUT=str(smoke), SCEN=str(scen), TAG="or-smoke", OPTS=json.dumps({"resume": {}}), WORKERS="1")
        pf = P.preflight("bash %s" % shlex.quote(str(HERE / "or_hugsim.sh")), name="or-pf", vram_gb=12.0, cpu=4, env=env,
                         log_dir=str(log_root / "preflight"), cwd=str(REPO), owner="op_resume")
        print("preflight %s" % pf)
    ids = {}
    for u in units:
        ld = log_root / u["name"]
        if u["done"]():
            print("finished  %s" % u["name"])
            continue
        old = (ld / "pool_id").read_text().strip() if (ld / "pool_id").exists() else ""
        if old and P.job_states([old])[old] in ("inbox", "queued", "running"):
            ids[u["name"]] = old
            print("in pool   %s  %s" % (u["name"], old))
            continue
        kw = dict(owner="op_resume", vram_gb=u["vram_gb"], carla=u["carla"], cpu=u["cpu"], tries=2, env=u["env"], log_dir=str(ld),
                  cwd=str(REPO), after=[pf] if pf else [])
        if a.dry_run:
            print("would submit %s\n  %s\n  %s" % (u["name"], json.dumps(kw), u["cmd"]))
            continue
        ld.mkdir(parents=True, exist_ok=True)
        jid = P.submit(u["cmd"], name=u["name"], **kw)
        (ld / "pool_id").write_text(jid + "\n")
        ids[u["name"]] = jid
        print("submitted %s  %s" % (u["name"], jid))
    if a.wait and ids:
        st = P.wait(list(ids.values()), 60.0)
        bad = [n for n, i in ids.items() if st.get(i) != "done"]
        print("not done: %s" % bad if bad else "all done")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
