"""Worker-profile experiment (todos/2026-10-01-cl-lib.md): PDM-Lite on 40 fixed Bench2Drive routes, one exclusive run per
arm and card, through jevdrive.cl. Also the library re-expression of scripts/carla_threads_routes.sh (2026-09-27).

  python -m jevdrive.cl run scripts/lanes/cl_worker_profile.py --arg stage=pilot
  python -m jevdrive.cl run scripts/lanes/cl_worker_profile.py --arg stage=A
  python -m jevdrive.cl run scripts/lanes/cl_worker_profile.py --arg stage=B --arg profile=reduced --arg t=2
  python -m jevdrive.cl run scripts/lanes/cl_worker_profile.py --arg stage=C --arg t=2
  python -m jevdrive.cl run scripts/lanes/cl_worker_profile.py --arg stage=old     # carla_threads_routes.sh as it ran
"""
from jevdrive.cl import Profile, b2d
from jevdrive.cl.lane import REPO, data_dir

NAME, ROOT = "cl-lib", "cl_lib/profile"
OLD20 = "2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381,1825,2084,2086,2091,2115,2286,24330,17563,26458"
NEW20 = "1773,2201,2534,2844,3080,3248,3464,3666,3785,4468,14909,20920,23910,24240,24784,25383,25951,26408,27494,28099"
ROUTES = (OLD20 + "," + NEW20).split(",")
CARDS = (1, 2)


def prof(kind, t):
    t = None if t in (None, "unset") else int(t)
    return (Profile("stock", pools="stock", client_threads=0, num_threads=t) if kind == "stock"
            else Profile("reduced", pools="reduced", pool_threads=4, client_threads=8, num_threads=t))


def pdm(name, p, w, card, prio, ids=ROUTES, **kw):
    sim = data_dir() / "third_party/simlingo"
    return b2d(name, data_dir() / "runs" / ROOT / "arms" / name, ids, routes=sim / "leaderboard/data/bench2drive220.xml",
               agent=REPO / "scripts/b2d_expert_agent.py", agent_config="expert+" + name,
               python=data_dir() / "envs/simlingo/bin/python", workers=w, max_attempts=2, min_done=0.95, tries=1,
               profile=p, env=dict(BENCH2DRIVE_ROOT=sim / "Bench2Drive", WORK_DIR=sim), exclusive=True, gpus=(card,),
               priority=prio, meta=dict(kind=p.name, t=p.num_threads, w=w, card=card), **kw)


def jobs(args):
    stage = args.get("stage", "A")
    if stage == "pilot":
        return [pdm("A-reduced-t2-w8-a", prof("reduced", 2), 8, 1, 0)]
    if stage == "A":                       # two counterbalanced sequences; the pilot run is reduced/t2 repeat a
        seq = {1: [("reduced", 2), ("stock", 2), ("reduced", 1), ("stock", 1), ("reduced", 4), ("stock", 4)],
               2: [("stock", 2), ("reduced", 2), ("stock", 1), ("reduced", 1), ("stock", 4), ("reduced", 4)]}
        return [pdm("A-%s-t%d-w8-%s" % (k, t, "a" if c == 1 else "b"), prof(k, t), 8, c, i)
                for c in CARDS for i, (k, t) in enumerate(seq[c])]
    if stage == "B":                       # workers per card with the Stage-A choice, plus the other profile at 12
        k, t = args["profile"], args.get("t", "2")
        other = "stock" if k == "reduced" else "reduced"
        runs = [(k, 4, 1), (k, 6, 2), (k, 10, 1), (k, 12, 2), (other, 12, 1)]
        return [pdm("B-%s-t%s-w%d" % (kk, t, w), prof(kk, t), w, c, i) for i, (kk, w, c) in enumerate(runs)]
    card = int(args["card"]) if "card" in args else None     # put stage C / old on one card (e.g. a free card 0)
    if stage == "C":                       # camera stub (front3 1600x900, no model): GPU-bound regime, throughput only
        t = args.get("t", "2")
        return [b2d("C-%s-t%s-w6" % (k, t), data_dir() / "runs" / ROOT / "arms" / ("C-%s-t%s-w6" % (k, t)), ROUTES,
                    workers=6, max_attempts=2, min_done=0.95, tries=1, profile=prof(k, t), exclusive=True,
                    gpus=(card if card is not None else c,), priority=1,
                    meta=dict(kind=k, t=t, w=6, card=c, camera="front3"))
                for k, c in (("reduced", 1), ("stock", 2))]
    if stage == "old":                     # scripts/carla_threads_routes.sh, 2026-09-27 settings: 5 workers, client 8,
        olds = [pdm("old-%s" % k, Profile(k, pools=k, client_threads=8, num_threads=None), 5,
                    card if card is not None else c, 0, ids=OLD20.split(",")) for k, c in (("reduced", 1), ("stock", 2))]
        for j in olds:
            j.exclusive = False            # DS reproduction only: two 5-worker runs may share a card
        return olds
    if stage == "extra":                   # old + C together on one card (stage C runs alone once old is done)
        return jobs(dict(args, stage="old")) + jobs(dict(args, stage="C"))
    raise SystemExit("unknown stage %s" % stage)
