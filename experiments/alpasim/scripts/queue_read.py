"""AlpaSim reads on persistent stacks that pull scenes from a shared queue (research/alpasim-read-throughput.md).

A read is (driver, checkpoint env, scene list, out dir). One queue holds the scenes of one or more reads that share the
wizard overrides; N slot jobs (pool jobs) each start one stack: renderer + controller + the runtime in AlpaSim's shipped
daemon mode (`wizard.run_mode=SERVER`, `alpasim_runtime.simulate --serve`), and one driver per read, started on demand.
A slot claims one scene at a time (atomic rename), sends it to its runtime as a one-scene `simulate` request with the
read's driver as the request's driver pool, and keeps the runtime's slots full, so a slot never idles while scenes are
left and the read has no fixed chunks. The runtime does not wait for the driver at start (driver pool skipped in the
config, given per request), so runtime start-up and driver warm-up overlap. A finished rollout dir is moved into its
read's out dir; when every slot is done, `collect` (CPU-only) runs AlpaSim's own aggregation on each out dir, which then
has the layout of a run.sh run: rollouts/<scene>/<uuid>/, aggregate/results-summary.json, driver-logs/ (one dir per slot
driver, s<k>/), native_summary.json. Nothing in the AlpaSim source is edited; the per-scene results are the same as a
run.sh run's (a rollout does not depend on which other scenes share its stack; check in the note above).

  python queue_read.py plan    <queue dir> --read <out>=<scene list>[=<driver>[=K=V,K=V]] ... -- <wizard overrides>
  python queue_read.py slot    <queue dir> <slot index>        (a pool job, AlpaSim venv, like run_native.py)
  python queue_read.py collect <queue dir>                     (after every slot; CPU only)
Lanes submit through experiments/alpasim/scripts/read.sh, not by hand.

Queue dir: spec.json, todo/<seq> (one line "<read index> <scene>"), claim/s<k>/, done/, slots/s<k>/ (the stack's run dir).
A slot that starts again (pool retry) first puts its own claimed scenes back and drops their partial outputs.
"""
import argparse, asyncio, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

def RN():
    import run_native
    return run_native


CFG_FILES = ("run_metadata.yaml", "eval-config.yaml", "wizard-config.yaml", "wizard-config-loadable.yaml")


def plan(a) -> int:
    q = Path(a.queue).resolve()
    if (q / "spec.json").exists():
        raise SystemExit(f"{q} already planned")
    reads, items, scenes = [], [], []
    for i, r in enumerate(a.read):
        out, lst, drv, env = (r.split("=", 3) + ["sh30", ""])[:4]
        env = dict(kv.split(":", 1) for kv in env.split(",") if kv) if env else {}
        ids = Path(lst).read_text().split()
        reads.append({"out": str(Path(out).resolve()), "list": str(Path(lst).resolve()), "driver": drv, "env": env, "n": len(ids)})
        items += [(i, s) for s in ids]
        scenes += ids
    for i, r in enumerate(reads):
        o = Path(r["out"])
        if (o / "aggregate" / "results-summary.json").exists():
            raise SystemExit(f"{o} already ran")
        shutil.rmtree(o, ignore_errors=True)
        o.mkdir(parents=True)
    for d in ("todo", "claim", "done", "slots"):
        (q / d).mkdir(parents=True, exist_ok=True)
    if a.order == "scene":   # the reads of one scene back to back: they share the renderer's scene cache
        first = {}
        for n, (_, sc) in enumerate(items):
            first.setdefault(sc, n)
        items.sort(key=lambda x: (first[x[1]], x[0]))
    for n, (i, s) in enumerate(items):
        (q / "todo" / f"{n:06d}").write_text(f"{i} {s}\n")
    json.dump({"reads": reads, "scenes": sorted(set(scenes)), "overrides": a.overrides, "t": time.time()},
              open(q / "spec.json", "w"), indent=1)
    print(f"[queue] {q}: {len(reads)} reads, {len(items)} rollouts, {len(set(scenes))} scenes")
    return 0


def submit(a, pool_args: list) -> int:
    """plan, then N slot jobs (cl fanout) and the collect job after them; prints the collect job's id (chain on it)."""
    plan(a)
    n = a.slots or max(1, math.ceil(sum(1 for _ in (Path(a.queue) / "todo").iterdir()) / 150))
    q, sh = str(Path(a.queue).resolve()), str(HERE / "read.sh")
    defaults = {"--vram": "20", "--cpu": "6", "--ram": "30", "--tries": "2", "--timeout-h": "3"}
    flags = [x for k, v in defaults.items() if k not in pool_args for x in (k, v)]
    cmd = [str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl", "fanout", "--arms", ",".join(map(str, range(n))), "--name", a.name,
           "--log-dir", f"{q}/pool", "--env", "CL_HIST_KEY=alpasim-slot", "--no-check", *flags, *pool_args,
           "--collect", f"bash {sh} collect {q}", "--", "bash", sh, "slot", q, "{arm}"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    if out.returncode:
        raise SystemExit(f"fanout rc {out.returncode}: {out.stdout[-500:]}")
    (Path(q) / "jobs.txt").write_text(out.stdout)
    print(out.stdout.split()[-2])   # the collect job (fanout prints "<id> <name>-collect" last)
    return 0


def driver_cmd(rd: dict) -> tuple:
    """(cwd, shell command) of a read's driver: run.sh's own driver block (RUN_SH_DRIVER_ONLY=1), log dir = the read's."""
    env = dict(os.environ, **rd["env"], RUN_SH_DRIVER_ONLY="1")
    out = subprocess.run(["bash", str(HERE / "run.sh"), rd["out"], rd["driver"]], env=env, capture_output=True, text=True, check=True)
    cwd, cmd = out.stdout.rstrip("\n").split("\n", 1)
    return cwd, cmd


class Slot:
    def __init__(self, q: Path, k: int):
        self.q, self.k, self.spec = q, k, json.load(open(q / "spec.json"))
        self.reads = self.spec["reads"]
        self.sd, self.cl = q / "slots" / f"s{k}", q / "claim" / f"s{k}"
        self.drv, self.ready, self.inflight, self.stopped = {}, {}, {}, set()
        self.times = {"t_start": time.time()}
        self.n_done = self.n_failed = 0

    def requeue(self) -> None:
        """A previous try of this slot: claimed scenes go back, their partial outputs are dropped."""
        self.cl.mkdir(parents=True, exist_ok=True)
        for f in self.cl.iterdir():
            i, s = f.read_text().split()
            shutil.rmtree(Path(self.reads[int(i)]["out"]) / "rollouts" / s, ignore_errors=True)
            os.rename(f, self.q / "todo" / f.name)
        shutil.rmtree(self.sd, ignore_errors=True)
        self.sd.mkdir(parents=True)

    def claim(self):
        for name in sorted(os.listdir(self.q / "todo")):
            try:
                os.rename(self.q / "todo" / name, self.cl / name)
            except FileNotFoundError:
                continue
            i, s = (self.cl / name).read_text().split()
            return name, int(i), s
        return None

    def todo_reads(self, n: int = 0) -> list:
        """Read indices of the next n todo items (all when n = 0), in queue order."""
        names = sorted(os.listdir(self.q / "todo"))
        out = []
        for name in names[:n] if n else names:
            try:
                out.append(int((self.q / "todo" / name).read_text().split()[0]))
            except (FileNotFoundError, IndexError, ValueError):
                pass
        return out

    def start_driver(self, i: int) -> None:
        if i in self.drv:
            return
        rd, port = self.reads[i], RN().free_port()
        cwd, cmd = driver_cmd(rd)
        # each slot's driver logs to its own dir under the read's driver-logs (dump files are numbered per driver)
        log = Path(rd["out"]) / "driver-logs" / f"s{self.k}"
        log.mkdir(parents=True, exist_ok=True)
        cmd = cmd.replace(f"ALPASIM_DRIVER_LOG_DIR={rd['out']}/driver-logs", f"ALPASIM_DRIVER_LOG_DIR={log}")
        name = f"driver-r{i}"
        self.procs.start(name, cmd, dict(self.base, **rd["env"], ALPASIM_DRIVER_HOST="127.0.0.1", ALPASIM_DRIVER_PORT=str(port)), cwd)
        self.drv[i] = (name, port)
        self.ready[i] = asyncio.get_running_loop().run_in_executor(None, self.procs.wait_port, name, port, self.timeout)

    def stop_driver(self, i: int) -> None:
        name, _ = self.drv.pop(i)
        self.ready.pop(i)
        self.stopped.add(name)
        p = self.procs.p[name]
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass

    @staticmethod
    def capacity(ov: list) -> int:
        """Rollouts the stack runs at once: min over renderer (x replicas) and controller of their slots (wizard defaults 4 / 1)."""
        val = lambda key, d: int(([o.split("=", 1)[1] for o in ov if o.startswith(key + "=")] or [d])[-1])
        rend = val("runtime.endpoints.renderer.n_concurrent_rollouts", 4) * val("services.renderer.replicas_per_container", 1)
        ctl = val("runtime.endpoints.controller.n_concurrent_rollouts", 4) * val("services.controller.replicas_per_container", 1)
        return min(rend, ctl)

    async def run_item(self, stub, name: str, i: int, scene: str) -> None:
        from alpasim_grpc.v0 import runtime_pb2
        await self.ready[i]
        port = self.drv[i][1]
        req = runtime_pb2.SimulationRequest(rollout_specs=[runtime_pb2.RolloutSpec(scenario_id=scene, nr_rollouts=1)],
                                            available_drivers=[runtime_pb2.SimulationRequest.DriverAddress(ip="127.0.0.1", port=port)],
                                            n_concurrent_per_driver=self.conc)
        ret = await stub.simulate(req)
        out = Path(self.reads[i]["out"])
        for rr in ret.rollout_returns:
            src = self.sd / "rollouts" / scene / rr.rollout_uuid
            if rr.rollout_uuid and src.is_dir():
                dst = out / "rollouts" / scene
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst / rr.rollout_uuid))
            if not rr.success:
                self.n_failed += 1
                with open(out / f"failed.s{self.k}.jsonl", "a") as f:
                    f.write(json.dumps({"scene": scene, "uuid": rr.rollout_uuid, "error": rr.error[:2000]}) + "\n")
        os.rename(self.cl / name, self.q / "done" / name)
        self.n_done += 1
        self.times.setdefault("first_done_s", time.time() - self.times["t_start"])

    async def main(self, timeout: float) -> int:
        import grpc
        import yaml
        from alpasim_grpc.v0 import runtime_pb2_grpc
        self.timeout, ov = timeout, list(self.spec["overrides"])
        conc = [int(o.split("=")[1]) for o in ov if o.startswith("runtime.endpoints.driver.n_concurrent_rollouts=")]
        self.conc = conc[-1] if conc else 4
        self.requeue()
        self.procs, rc = RN().Procs(self.sd), 1
        self.base = {**dict(RN().profiles.PROFILES["alpasim"].env), **os.environ, "UV_NO_SYNC": "1", "PYTHONUNBUFFERED": "1",
                     "ALPASIM_NUPLAN_ROOT": os.path.expandvars("$DATA_DIR/datasets/alpasim_nuplan")}
        src = os.path.expandvars("$DATA_DIR/third_party/alpasim")
        import threading
        threading.Thread(target=self.procs.sample, daemon=True).start()
        tasks = {}
        try:
            first = self.todo_reads(1)
            if not first:
                print("[queue] nothing to do", flush=True)
                return 0
            self.start_driver(first[0])                     # warms up while the stack starts
            ids = ",".join(f'"{x}"' for x in self.spec["scenes"])
            wiz = ["uv", "run", "alpasim_wizard", *ov, f"scenes.scene_ids=[{ids}]", "scenes.limit_to_first_n=0",
                   "wizard.run_method=NONE", "wizard.run_mode=SERVER", "runtime.endpoints.driver.skip=true",
                   f"wizard.log_dir={self.sd}", f"wizard.baseport={random.randrange(20000, 30000)}"]
            with open(self.sd / "native-logs" / "wizard.log", "w") as log:
                log.write(" ".join(wiz[:3 + len(ov)]) + " ...\n")
                log.flush()
                subprocess.run(wiz, cwd=src, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(self.base, ALPASIM_DRIVER_HOST="localhost", ALPASIM_DRIVER_PORT="1"))
            services = yaml.safe_load(open(self.sd / "docker-compose.yaml"))["services"]
            rt = [n for n in services if n.startswith("runtime")]
            for name in [n for n in services if n not in rt] + rt:
                svc = services[name]
                self.procs.start(name, RN().host_cmd(svc, src), dict(self.base, **dict(e.split("=", 1) for e in svc.get("environment", []))), src)
            for name in services:
                for p in services[name].get("ports", []):
                    self.times[f"{name}_ready_s"] = await asyncio.get_running_loop().run_in_executor(
                        None, self.procs.wait_port, name, int(str(p).split(":")[0]), timeout)
            port = int(str(services[rt[0]]["ports"][0]).split(":")[0])
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}", options=[("grpc.max_receive_message_length", 256 << 20)]) as ch:
                stub = runtime_pb2_grpc.RuntimeServiceStub(ch)
                # the runtime listens once its engine has started; get_runtime_info fails on the trajdata provider at 0bb4c4b
                # ('TrajdataSceneProvider' has no 'scene_infos'), so the capacity comes from the overrides
                cap = self.capacity(ov) + 2   # keep the runtime's queue non-empty
                self.times["runtime_ready_s"] = time.time() - self.times["t_start"]
                rt_log, rt_off, empty = self.sd / "native-logs" / f"{rt[0]}.log", 0, False
                while True:
                    while not empty and len(tasks) < cap:
                        c = self.claim()
                        if c is None:
                            empty = True
                            break
                        name, i, s = c
                        self.start_driver(i)
                        tasks[asyncio.create_task(self.run_item(stub, name, i, s))] = i
                        for j in set(self.todo_reads(cap)) - set(self.drv):   # the next read's driver warms up early
                            self.start_driver(j)
                    if not tasks:
                        break
                    done, _ = await asyncio.wait(tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        tasks.pop(t)
                        t.result()
                    if done:
                        left = set(self.todo_reads()) | set(tasks.values())
                        for j in [j for j in self.drv if j not in left]:
                            self.stop_driver(j)
                    dead = [n for n, p in self.procs.p.items() if p.poll() is not None and n not in self.stopped]
                    if dead:
                        raise RuntimeError(f"{dead} died while the slot was running")
                    crash, rt_off = RN().worker_crash(rt_log, rt_off)
                    if crash:
                        raise RuntimeError(f"runtime worker crashed and the runtime does not recover: {crash}")
            rc = 0
        except Exception as e:
            print(f"[queue] ERROR {e!r}", flush=True)
            self.times["error"] = repr(e)
            for t in tasks:
                t.cancel()
        finally:
            self.procs.stop.set()
            self.procs.kill_all()
            self.times.update(total_s=time.time() - self.times["t_start"], rollouts=self.n_done, rollouts_failed=self.n_failed)
            json.dump({"rc": rc, "slot": self.k, "times": self.times, "peak": self.procs.peak, "overrides": self.spec["overrides"],
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, open(self.sd / "native_summary.json", "w"), indent=1)
            print(f"[queue] slot {self.k} rc {rc}: {self.times}", flush=True)
        return rc


def collect(a) -> int:
    """AlpaSim's aggregation (as the one-shot runtime runs it) on each read's out dir, once every scene is done."""
    from alpasim_utils.yaml_utils import typed_parse_config
    from eval.aggregation.failed_rollouts import FailedRollout
    from eval.aggregation.main import run_aggregation_from_runtime
    from eval.schema import EvalConfig
    import yaml
    q = Path(a.queue).resolve()
    spec = json.load(open(q / "spec.json"))
    left = list((q / "todo").iterdir()) + [f for d in (q / "claim").iterdir() for f in d.iterdir()]
    if left:
        raise SystemExit(f"{len(left)} scenes not done (a slot failed?): {left[:3]}")
    slots = sorted(d for d in (q / "slots").iterdir() if (d / "run_metadata.yaml").exists())
    sums = [json.load(open(d / "native_summary.json")) for d in slots if (d / "native_summary.json").exists()]
    rc = 0
    for rd in spec["reads"]:
        out, t0 = Path(rd["out"]), time.time()
        for f in CFG_FILES:
            shutil.copy(slots[0] / f, out / f)
        run_name = yaml.safe_load(open(out / "run_metadata.yaml")).get("run_name")
        fails = [json.loads(l) for f in sorted(out.glob("failed.s*.jsonl")) for l in open(f)]
        failed = [FailedRollout(run_name=run_name, run_uuid=None, clipgt_id=x["scene"], rollout_id=x["uuid"] or f"failed-{n}",
                                error=x["error"]) for n, x in enumerate(fails)]
        ok = run_aggregation_from_runtime(log_dir=str(out), eval_config=typed_parse_config(str(out / "eval-config.yaml"), EvalConfig),
                                          array_job_dir=str(out), failed_rollouts=failed)
        rows = json.load(open(out / "aggregate" / "results-summary.json"))["rollouts"]
        n_fail = sum(bool(r.get("failure_reason")) and not r.get("metrics") for r in rows)
        err = "" if ok and rows and n_fail < len(rows) and len(rows) == rd["n"] else \
            f"aggregation ok={ok}, {len(rows)} rows of {rd['n']}, {n_fail} failed"
        rc = rc or bool(err)
        json.dump({"rc": int(bool(err)), "queue": str(q), "times": {"aggregate_s": time.time() - t0, "rollouts": len(rows),
                   "rollouts_failed": n_fail, **({"error": err} if err else {})}, "slots": [s["times"] for s in sums],
                   "overrides": spec["overrides"], "driver": rd["driver"], "env": rd["env"]}, open(out / "native_summary.json", "w"), indent=1)
        print(f"[queue] {out}: {len(rows)} rollouts, {n_fail} failed {err}", flush=True)
    return int(rc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for verb in ("plan", "submit"):
        p = sub.add_parser(verb)
        p.add_argument("queue")
        p.add_argument("--read", action="append", required=True, metavar="OUT=LIST[=DRIVER[=K:V,K:V]]")
        p.add_argument("--order", choices=("read", "scene"), default="scene", help="queue order: read by read, or scene by scene")
        if verb == "submit":
            p.add_argument("--slots", type=int, default=0, help="slot jobs (default: one per 150 rollouts)")
            p.add_argument("--name", required=True)
            p.add_argument("--pool", default="", help="extra cl fanout flags, one string (e.g. '--priority 14 --after <id> --owner x')")
        p.add_argument("overrides", nargs="+")
    s = sub.add_parser("slot")
    s.add_argument("queue")
    s.add_argument("k", type=int)
    s.add_argument("--ready-timeout", type=float, default=900)
    sub.add_parser("collect").add_argument("queue")
    a = ap.parse_args()
    if a.cmd == "plan":
        return plan(a)
    if a.cmd == "submit":
        return submit(a, a.pool.split())
    if a.cmd == "slot":
        return asyncio.run(Slot(Path(a.queue).resolve(), a.k).main(a.ready_timeout))
    return collect(a)


if __name__ == "__main__":
    sys.exit(main())
