"""Gap-rule sweep of the perceived bypass (plan 2026-10-04-vmerge3.md, addendum "gap-rule sweep").

  python -m jevdrive.cl run experiments/vlm_arb/scripts/vmerge3gap_chain.py --lane vlm-vm3gap --workers-per-card 6 \
      --arg stage=pre|obs|full [--arg arm=gap25] [--arg ports=8200+card]

Arms = vm3norel (vmerge2 + VM3_BYP=perc, no release check) with the follower headway T of the start rule (d = T + T x v) set by VM3_T:
gap20 (2.0 s), gap25 (2.5 s), gap30 (3.0 s, the existing default). Stages: pre = smoke on the obstacle debug route 25169 (outside the 19);
obs = the 4 obstacle routes in two units of 2 (oa / ob), gap20 and gap25 seeds 0-3, gap30 seeds 2-3 (seeds 0, 1 of gap30 are the vm3norel
runs); full = the 15 other routes x seeds 0, 1 of --arg arm (q0..q2 shards minus the obstacle routes). `report` last.
One Qwen server on the leased card, VLM_PORTS = that port. Units under $DATA_DIR/runs/vlm_arb/arms/v2-<arm>-s<seed>-<oa|ob|q*>.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
import vmerge3_chain as m3  # noqa: E402  (sets SLOT_WORKERS / SLOT_CORES through vmerge_chain)
from vlm_arb_common import OBS_ROUTES, RUN  # noqa: E402

NAME, ROOT = "vlm-vm3gap", "vlm_arb_vm3gap"
T = {"gap20": "2.0", "gap25": "2.5", "gap30": "3.0"}
OBS_UNITS = {"oa": ["19324", "2520"], "ob": ["19832", "24497"]}


def env(arm, ports):
    e = m3.env("vm3norel", ports)
    e.update(VM3_T=T[arm], VLM_SLOT_DIR=str(RUN.parent / ROOT / "qwen_slots"))
    return e


def jobs(args):
    ports = args["ports"]
    smoke = [base.unit("dbg-gap", 0, "25169", ["25169"], env("gap20", ports), "", 0, base="vmerge")]
    if args.get("stage", "pre") == "pre":
        return smoke
    out, names = [], []
    if args["stage"] == "obs":
        plan = [(a, s) for a in ("gap20", "gap25") for s in range(4)] + [("gap30", 2), ("gap30", 3)]
        # lowest-T arm first on seeds 0-1, so the early read exists soonest
        plan.sort(key=lambda x: (x[1] > 1, x[1], T[x[0]]))
        for i, (a, s) in enumerate(plan):
            for k, ids in OBS_UNITS.items():
                out.append(base.unit(a, s, k, ids, env(a, ports), "", i, deps=[smoke[0].name], base="vmerge"))
    else:
        a, sh = args["arm"], vc.shards()
        for i, s in enumerate((0, 1)):
            for k in vc.SHARDS:
                ids = [r for r in sh[k] if r not in OBS_ROUTES]
                out.append(base.unit(a, s, k, ids, env(a, ports), "", i, base="vmerge"))
    names = [x.name for x in out]
    return smoke + out + [base.tool("report", "vmerge3gap_report.py", "report", deps=names, prio=99)]
