"""Staged, resumable privileged-ceiling queue using the existing op-drive runners.

Every route shard is read immediately, and the whole batch stops at a failed checklist.
All artifacts are confined to DATA_DIR/runs/b2d_privileged_ceiling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET

from tqdm import tqdm

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from jevdrive.runlog import RunLog
from b2d_privileged_geometry import ARMS, PARAMS
from b2d_privileged_checks import numeric, route_checks

ROOT=Path(os.environ["DATA_DIR"])/"runs/b2d_privileged_ceiling"
DEV="27043,15102,24944,27870,22535,37969,24497,27297,9196,28147".split(",")
HELD="26828,25783,24416,469,25215,27392,27005,25613,27907,26023,24207,26723,24519,25753,24948,26365,27916,24622,26370".split(",")
DEBUG="334,27787,24721,26872,26537,17749,25169,24955".split(",")
JTYPES="NonSignalizedJunctionLeftTurn,NonSignalizedJunctionLeftTurnEnterFlow,NonSignalizedJunctionRightTurn,OppositeVehicleTakingPriority,SignalizedJunctionLeftTurn,SignalizedJunctionLeftTurnEnterFlow,T_Junction,MergerIntoSlowTrafficV2,SignalizedJunctionRightTurn,VehicleTurningRoute".split(",")
BTYPES="Accident,AccidentTwoWays,ConstructionObstacle,ConstructionObstacleTwoWays,ParkedObstacle,ParkedObstacleTwoWays,HazardAtSideLane,HazardAtSideLaneTwoWays".split(",")


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def manifest():
    xml=Path(os.environ["DATA_DIR"])/"third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml"
    rows=[dict(id=r.get("id"),town=r.get("town"),type=next(r.iter("scenario")).get("type")) for r in ET.parse(xml).getroot().iter("route")]
    banned=set(DEV+HELD+DEBUG);out={}
    for group,types in (("junction",JTYPES),("obstacle",BTYPES)):
        out[group]=[r for typ in types for r in [r for r in rows if r["type"]==typ and r["id"] not in banned][:3]]
    out["dev"]=[next(r for r in rows if r["id"]==rid) for rid in DEV]
    assert len(out["junction"])==24 and len(out["obstacle"])==24
    assert len(set(r["id"] for rs in out.values() for r in rs))==58
    out["debug"]=[next(r for r in rows if r["id"]==rid) for rid in DEBUG]
    p=ROOT/"manifest.json"
    if p.exists():assert json.loads(p.read_text())==out,"Manifest changed after registration"
    else:write(p,out)
    return out


def cores(spec):
    result=[]
    for part in spec.split(","):
        lo,_,hi=part.partition("-");result.extend(range(int(lo),int(hi or lo)+1))
    return result


def grant():
    d=dict(line.split("=",1) for line in (ROOT/"GO").read_text().splitlines() if line and not line.startswith("#") and "=" in line)
    assert set(os.sched_getaffinity(0))<=set(cores(d["PC_CPUS"]))
    assert d["GPUS"].strip('"').split()==["0","1","2"]
    assert os.environ.get("OPENBLAS_CORETYPE")=="Haswell"
    return d


class Chain:
    def __init__(self,args):
        ROOT.mkdir(parents=True,exist_ok=True)
        self.args=args;self.log=RunLog("b2d_privileged_ceiling","chain-"+args.phase)
        assert self.log.tb is not None,"TensorBoard writer missing"
        self.lock=threading.Lock();self.active={};self.stop=threading.Event();self.error=None
        self.metadata=manifest();self.slots=queue.Queue()
        for g in range(3):
            allcores=list(range(g*25,(g+1)*25))
            for k in range(args.slots):
                part=allcores[k*(25//args.slots):(k+1)*(25//args.slots)] if k<args.slots-1 else allcores[k*(25//args.slots):]
                self.slots.put((g,k,f"{part[0]}-{part[-1]}"))

    def status(self):
        with self.lock:message="; ".join(f"GPU{g}.{k}: {name}" for (g,k),name in sorted(self.active.items()))
        (ROOT/"STATUS").write_text(time.strftime("%F %T")+" "+(message or "between stages")+"\n")

    def attempt(self,d,rid):
        done=json.loads((d/"done"/(rid+".json")).read_text())
        return d/"attempts"/rid/str(done["attempt"])

    def unit(self,tag,arm,seed,ids,workers=None,record=False):
        slot=self.slots.get();g,k,cpus=slot
        name=f"{tag}-{arm}-s{seed}";adir=ROOT/"arms"/name
        count=len(ids);workers=min(count,workers or self.args.workers)
        try:
            grant()
            if self.stop.is_set():raise RuntimeError("Queue stopped after another unit failed")
            if (ROOT/"units"/(name+".json")).exists():return adir
            with self.lock:self.active[(g,k)]=name
            self.status();self.log.info(f"Starting {name} GPU{g}.{k}, cores {cpus}, workers {workers}")
            self.log.event("unit_start",unit=name,gpu=g,slot=k,cpus=cpus,workers=workers,ids=ids,record=record)
            env=dict(os.environ,GPU=str(g),IDX0=str(300+12*g+6*k),CPUS=cpus,WORKERS=str(workers),
                     ARMS=arm,SEEDS=str(seed),LAT_EXEC="curv",RESUME_S="5",KEEP_SRV="1",SRV_NO_TWIN="1",
                     DESIRE="true",DRIVE_ARGS='"resume": "nored"',PC_ENABLE="1",OPL_IDS=",".join(ids),
                     OP_ARB_DIR=str(ROOT/f"card{g}s{k}"),OP_ARB_ARMS=str(ROOT/"arms"),
                     OPENBLAS_CORETYPE="Haswell",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                     B2D_PIDS_WAIT="17000")
            env.pop("SRV_ONNX",None)
            env["OP_ARB_AGENT"]="scripts/op_drive_record_agent.py" if record else "scripts/op_arb_agent.py"
            start=time.perf_counter()
            for attempt in range(2):
                with (ROOT/("unit-"+name+".log")).open("a") as out:
                    rc=subprocess.run(["bash","scripts/op_arb.sh","set","2",tag],cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
                finished=all((adir/"done"/(rid+".json")).exists() for rid in ids)
                if rc==0 and finished:break
                # Only our own sentinel can be removed. Preserve every attempt and all its logs.
                if (adir/"DONE").exists():(adir/"DONE").unlink()
                self.log.info(f"Incomplete unit {name}, wrapper rc={rc}, retry={attempt+1}")
            assert rc==0 and finished,f"Unit failed: {name} rc={rc}; see unit log"
            for rid in ids:
                path=self.attempt(adir,rid)
                basic=route_checks(path,"drive")
                assert basic["checks"]["no_crash"] and basic["checks"]["finite"],basic
            if tag.startswith("eval"):
                from b2d_privileged_report import read_unit
                read_unit(adir,tag,arm,seed,self.log)
            if record:
                python=Path(os.environ["DATA_DIR"])/"envs/openpilot/bin/python"
                for rid in ids:
                    path=self.attempt(adir,rid)
                    with (ROOT/("video-"+name+".log")).open("a") as out:
                        subprocess.run([str(python),"scripts/op_drive_record_video.py","annotate",str(path)],cwd=REPO,stdout=out,stderr=subprocess.STDOUT,check=True)
            elapsed=time.perf_counter()-start
            write(ROOT/"units"/(name+".json"),dict(unit=name,ids=ids,arm=arm,seed=seed,gpu=g,slot=k,cpus=cpus,wall_s=elapsed,workers=workers,record=record))
            self.log.info(f"Completed {name}: {count} routes in {elapsed/60:.2f} min")
            self.log.event("unit_end",unit=name,wall_s=elapsed)
            self.log.scalar("performance/routes_per_hour",count*3600/elapsed,len(list((ROOT/"units").glob("*.json"))))
            return adir
        except BaseException as exc:
            self.stop.set();self.error=exc
            (ROOT/"ERROR").write_text(traceback.format_exc())
            self.log.info(f"ERROR in {name}: {exc}")
            raise
        finally:
            with self.lock:self.active.pop((g,k),None)
            self.status();self.slots.put(slot)

    def monitor(self):
        with (ROOT/"util.csv").open("a",buffering=1) as f:
            if f.tell()==0:f.write("time,gpu,utilization_pct,memory_used_mib,pids\n")
            while not self.stop.wait(5):
                pids=int(Path("/sys/fs/cgroup/pids.current").read_text())
                output=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True)
                for line in output.splitlines():
                    g,util,mem=[int(v.strip()) for v in line.split(",")]
                    f.write(f"{time.time():.3f},{g},{util},{mem},{pids}\n")
                    self.log.scalar(f"gpu{g}/utilization_pct",util,int(time.time()))
                if pids>=17500:
                    self.error=RuntimeError(f"PID hard checklist failed: {pids}")
                    (ROOT/"ERROR").write_text(str(self.error)+"\n");self.stop.set()

    def pilot(self):
        if (ROOT/"DONE-pilot").exists():return
        supplemental=ROOT/"pilot_diagnostics.json"
        if supplemental.exists():
            proof=json.loads(supplemental.read_text()).get("pilot-light-probe",{})
            # Explicit staged-launch deviation: route 334 is in the registered debug set.
            # Keep the original failed route and do not replace any of its measurements.
            if proof.get("passed") and route_checks(Path(proof["attempt"]),"pred")["passed"]:
                write(ROOT/"pilot_supplement.json",dict(original="failed: 27787 lacked a stoppable red opportunity",
                      supplement=proof,protocol_deviation="S1 integration evidence uses registered debug route 334"))
                (ROOT/"DONE-pilot").write_text(time.strftime("%F %T")+" explicit route-334 supplement; original pilot failed\n")
                self.log.event("pilot_supplement",route="334",original_failed=True)
                return
        grant();numeric(self.log,self.log.dir)
        d=self.unit("pilot","pred",0,["27787"],workers=1,record=True)
        check=route_checks(self.attempt(d,"27787"),"pred")
        write(ROOT/"pilot_checks.json",check)
        self.log.info("Pilot checks: "+json.dumps(check))
        assert check["passed"],"Substantive pilot checklist failed; inspect pilot_checks.json"
        (ROOT/"DONE-pilot").write_text(time.strftime("%F %T\n"))

    def diagnose(self):
        """Repeated fixed-debug probes; every result is retained and no gate is bypassed."""
        spec=[("pilot-repeat", "pred", 0, ["27787"]),
              ("pilot-seed1", "pred", 1, ["27787"]),
              ("pilot-light-probe", "pred", 0, ["334"])]
        checks={}
        with ThreadPoolExecutor(max_workers=3) as pool:
            fs={pool.submit(self.unit,*task,workers=1,record=True):task for task in spec}
            for f in tqdm(as_completed(fs),total=len(fs),desc="Fixed pilot diagnostics"):
                tag,arm,seed,ids=fs[f];d=f.result()
                checks[tag]=route_checks(self.attempt(d,ids[0]),"pred")
        write(ROOT/"pilot_diagnostics.json",checks)
        self.log.info("Pilot diagnostics: "+json.dumps(checks))
        # The original pilot failure remains authoritative. Diagnosis never opens the full batch.
        if not all(v["passed"] for v in checks.values()):
            raise RuntimeError("Pilot diagnostics failed; full batch remains stopped")

    def debug(self):
        if (ROOT/"DONE-debug").exists():return
        self.pilot()
        spec=[("27787",a) for a in ("drive","pred")]+[("26872",a) for a in ("drive","pjunc","pall")]+[("25169",a) for a in ("drive","pbyp","pbypgap","pall")]+[("24955",a) for a in ("drive","pbyp","pbypgap")]
        spec.append(("334","pred"))
        results={}
        with ThreadPoolExecutor(max_workers=3*self.args.slots) as pool:
            fs={pool.submit(self.unit,"debug"+rid,arm,0,[rid],record=True):(rid,arm) for rid,arm in spec}
            for f in tqdm(as_completed(fs),total=len(fs),desc="Debug units"):
                rid,arm=fs[f];results[(rid,arm)]=f.result()
        checks={}
        for rid,arm,kind in (("27787","pred","drive"),("334","pred","pred"),("26872","pjunc","pjunc"),("25169","pbyp","pbyp"),("24955","pbyp","pbyp")):
            checks[rid+"-"+arm]=route_checks(self.attempt(results[(rid,arm)],rid),kind)
        write(ROOT/"debug_checks.json",checks)
        assert all(v["passed"] for v in checks.values()),"Substantive debug checklist failed; inspect debug_checks.json"
        write(ROOT/"lock.json",dict(params=PARAMS,arms=ARMS,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                                   manifest_sha256=hashlib.sha256((ROOT/"manifest.json").read_bytes()).hexdigest()))
        (ROOT/"DONE-debug").write_text(time.strftime("%F %T\n"))

    def full(self):
        self.debug()
        tasks=[]
        for arm in ARMS:
            for seed in (0,1):
                for group in ("junction","obstacle","dev"):
                    ids=[r["id"] for r in self.metadata[group]]
                    for i in range(0,len(ids),4):tasks.append((f"eval-{group}-{i//4:02d}",arm,seed,ids[i:i+4]))
        with ThreadPoolExecutor(max_workers=3*self.args.slots) as pool:
            fs=[pool.submit(self.unit,*task) for task in tasks]
            for f in tqdm(as_completed(fs),total=len(fs),desc="Registered route shards"):f.result()
        from b2d_privileged_report import summarize
        summarize(ROOT,self.log)
        (ROOT/"DONE-full").write_text(time.strftime("%F %T\n"))

    def cleanup(self):
        for directory in ROOT.glob("card*s*"):
            p=directory/"srv/op.pid"
            if not p.exists():continue
            pid=int(p.read_text())
            try:
                cmd=Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"scripts/op_arb_server.py" in cmd and str(directory).encode() in cmd:
                    os.kill(pid,signal.SIGTERM)
                    self.log.event("server_stop",pid=pid)
            except (ProcessLookupError,FileNotFoundError):pass

    def run(self):
        monitor=threading.Thread(target=self.monitor,daemon=True);monitor.start()
        self.log.event("start",phase=self.args.phase,resources=grant())
        try:
            if self.args.phase=="pilot":self.pilot()
            elif self.args.phase=="diagnose":self.diagnose()
            elif self.args.phase=="debug":self.debug()
            else:self.full()
            if self.error:raise self.error
            (ROOT/("DONE-"+self.args.phase)).write_text(time.strftime("%F %T\n"))
            self.log.info("Stage completed: "+self.args.phase)
            self.log.event("end",status="complete",phase=self.args.phase)
        except BaseException:
            (ROOT/"ERROR").write_text(traceback.format_exc());raise
        finally:
            self.stop.set();monitor.join(timeout=10);self.cleanup();self.log.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("pilot","diagnose","debug","all"),default="pilot")
    p.add_argument("--slots",type=int,choices=(1,2),default=1)
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args();assert 1<=args.workers<=4
    Chain(args).run()
